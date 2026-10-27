"""Диспетчер: единый цикл Инженер -> Пилот -> Награды -> Оптимизатор.

Цикл одной попытки:
    задача оператора -> расчёт бюджета ΔV и подбор деталей (AutonomousEngineer)
    -> отчёт kia_design_report.md -> чертёж .craft -> старт -> полёт
    (журнал kia_flight_log.md в реальном времени) -> очки -> история -> эволюция.

Оператор может в любой момент сменить задачу и цены наград: диспетчер
читает их перед каждой попыткой.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime

from ..config import CONFIG, DATA_DIR
from ..engineer.autodesign import AutonomousEngineer, DesignPolicy, VehicleDesign
from ..engineer.design_report import write_report
from ..environment.connection import ConnectionFailed, KRPCConnection
from ..environment.control import ShipControl
from ..flight_log import FLIGHT_LOG, FlightLog
from ..learning.genome import Individual
from ..learning.history import HistoryStore, RunRecord
from ..learning.optimizer import GeneticOptimizer
from ..learning.reward_book import REWARD_BOOK
from ..learning.rewards import Failure, Milestone, RewardSystem
from ..logging_setup import get_logger
from ..memory import GLOBAL_MEMORY
from ..mission_spec import MissionSpec
from ..pilot.mission import Mission
from ..run_workspace import WORKSPACE, RunFolder
from .provision import VesselProvisioner

log = get_logger("agent.dispatcher")

OPTIMIZER_STATE = DATA_DIR / "optimizer_state.json"


@dataclass
class RunResult:
    individual: Individual
    score: float
    record: RunRecord
    design: VehicleDesign | None = None
    run_folder: RunFolder | None = None


class Dispatcher:
    """Главный управляющий агент KIA."""

    def __init__(self, spec: MissionSpec | None = None, offline: bool = False,
                 resume: bool = True, seed: int | None = None,
                 flight_log: FlightLog | None = None, book=None):
        self.spec = spec or MissionSpec.load()
        self.offline = offline
        self.book = book or REWARD_BOOK
        self.flight_log = flight_log or FLIGHT_LOG
        self.engineer = AutonomousEngineer()
        self.history = HistoryStore()
        self.optimizer = GeneticOptimizer(seed=seed)
        self.connection: KRPCConnection | None = None
        self.provisioner: VesselProvisioner | None = None
        self.control: ShipControl | None = None
        self.resume = resume
        self.abort_flag = threading.Event()
        self.busy = False
        self.last_design: VehicleDesign | None = None
        self.memory = GLOBAL_MEMORY
        self.workspace = WORKSPACE
        self.last_run_folder: RunFolder | None = None
        self._prepare_population()

    # ------------------------------------------------------------------
    def set_spec(self, spec: MissionSpec) -> None:
        self.spec = spec
        spec.save()
        log.info("Новая задача принята: %s", spec.describe())

    def _prepare_population(self) -> None:
        restored = self.resume and self.optimizer.load_state(OPTIMIZER_STATE)
        if restored and self.optimizer.population:
            log.info("Продолжаем обучение с поколения %d", self.optimizer.generation)
            return
        seeds = [Individual()]
        best = self.history.best_genome()
        if best:
            try:
                seeds.insert(0, Individual.from_dict(best))
                log.info("В затравку добавлен лучший геном из истории")
            except Exception as exc:
                log.debug("Лучший геном не восстановлен: %s", exc)
        self.optimizer.seed_population(seeds)

    # ------------------------------------------------------------------
    def connect(self) -> bool:
        if self.offline:
            log.info("Офлайн-режим: подключение к KSP не требуется")
            return True
        if self.connection is not None and self.connection.is_alive():
            return True
        try:
            self.connection = KRPCConnection().connect()
            self.control = ShipControl(self.connection, flight_log=self.flight_log)
            self.provisioner = VesselProvisioner(self.connection, self.control)
            self.provisioner.detect_save_name()
            return True
        except ConnectionFailed as exc:
            log.error("%s", exc)
            return False

    def connection_alive(self) -> bool:
        return self.connection is not None and self.connection.is_alive()

    def disconnect(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    # ------------------------------------------------------------------
    # Проектирование
    # ------------------------------------------------------------------
    def design_only(self, policy: DesignPolicy | None = None) -> VehicleDesign:
        """Считает проект под текущую задачу и пишет отчёт."""
        design = self.engineer.design(self.spec, policy or DesignPolicy())
        write_report(design)
        self.last_design = design
        return design

    # ------------------------------------------------------------------
    # Одна попытка
    # ------------------------------------------------------------------
    def run_attempt(self, individual: Individual) -> RunResult:
        run_id = self.history.next_run_id()
        started = datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        self.busy = True
        self.abort_flag.clear()
        self._current_individual = individual
        rewards = RewardSystem(book=self.book, flight_log=self.flight_log,
                               spec=self.spec)

        run_folder = self.workspace.create()
        self.last_run_folder = run_folder
        self.flight_log.retarget(run_folder.flight_log, run_folder.flight_json)
        log.info("──── ЗАПУСК #%d (папка kia/%d) | %s ────",
                 run_id, run_folder.number, individual.describe())

        # 1) Инженер: расчёт под задачу
        design = self.engineer.design(self.spec, individual.design)
        self.last_design = design
        write_report(design, run_folder.design_report, run_folder.design_json)

        if design.viable:
            rewards.award(Milestone.VIABLE_DESIGN, 0.0,
                          {"twr": round(design.launch_twr, 2),
                           "dv": round(design.total_delta_v)})
        else:
            rewards.penalise(Failure.DESIGN_REJECTED, 0.0,
                             {"problems": "; ".join(design.problems)})

        self.flight_log.start(
            mission=f"{self.spec.title} — {self.spec.describe()}",
            craft=design.blueprint.name,
            design_summary={
                "total_mass_t": round(design.total_mass, 2),
                "total_delta_v": round(design.total_delta_v),
                "launch_twr": round(design.launch_twr, 2),
                "part_count": design.part_count,
            },
            extra={"run_id": run_id, "generation": individual.generation,
                   "folder": f"kia/{run_folder.number}"})
        self.flight_log.fact(
            "Конструктор завершил расчёт",
            f"{len(design.stages)} ступени, ΔV {design.total_delta_v:.0f} м/с "
            f"при требуемых {design.budget.total:.0f}, TWR {design.launch_twr:.2f}")

        # 2) Полёт или офлайн-оценка
        if not design.viable or self.offline:
            result = self._offline_result(rewards, design)
        else:
            result = self._flight_result(rewards, design)

        record = RunRecord(
            run_id=run_id,
            generation=individual.generation,
            individual=individual.index,
            genome=individual.to_dict(),
            score=result["total_score"],
            milestones=result["milestones"],
            failures=result["failures"],
            terminated=result["terminated"],
            termination_reason=result.get("termination_reason"),
            design_summary=design.summary(),
            verdict={"viable": design.viable, "problems": design.problems,
                     "warnings": design.warnings,
                     "required_delta_v": round(design.budget.total, 1)},
            telemetry_final=result.get("telemetry_final", {}),
            events=result["events"],
            started_at=started,
            duration_s=round(time.time() - t0, 1),
            notes=result.get("final_phase", "offline" if self.offline else ""),
        )
        self.history.append(record)
        self.history.save_best_genome()
        self.optimizer.record_result(individual, record.score)

        # 3) Глобальная память: фиксируем подтверждённые навыки и уроки
        goal = self.goal_key()
        fresh = self.memory.record_success(
            record.milestones, individual, record.score, goal,
            evidence=f"запуск kia/{run_folder.number}, "
                     f"аппарат {design.total_mass:.2f} т, ΔV {design.total_delta_v:.0f}")
        for failure in record.failures:
            self.memory.record_failure_lesson(
                failure, individual, record.termination_reason or "")
        self.memory.remember_best(goal, individual, record.score)

        run_folder.write_meta({
            "score": record.score,
            "goal": goal,
            "outcome": ", ".join(record.failures) or ", ".join(record.milestones[-2:]),
            "milestones": record.milestones,
            "failures": record.failures,
            "mass_t": design.summary().get("total_mass_t"),
            "delta_v": design.summary().get("total_delta_v"),
            "twr": design.summary().get("launch_twr"),
            "new_skills": fresh,
            "genome": individual.to_dict(),
        })
        self.busy = False
        return RunResult(individual=individual, score=record.score,
                         record=record, design=design, run_folder=run_folder)

    # ------------------------------------------------------------------
    def _offline_result(self, rewards: RewardSystem, design: VehicleDesign) -> dict:
        """Аналитическая оценка без полёта (проверка конвейера и отбор проектов)."""
        if design.viable:
            margin = (design.total_delta_v - design.budget.total) / 1000.0
            rewards.shape("dv_margin", min(300.0, max(-300.0, margin * 150.0)))
            twr_error = abs(design.launch_twr - 1.6)
            rewards.shape("twr_quality", max(0.0, 60.0 - twr_error * 80.0))
            rewards.shape("mass_efficiency", max(0.0, 100.0 - design.total_mass * 0.8))
            rewards.metrics.total_mass = design.total_mass
            rewards.metrics.part_count = design.part_count
        summary = rewards.finalize({"final_phase": "offline", "telemetry_final": {}})
        self.flight_log.status("офлайн-оценка проекта (без полёта)")
        self.flight_log.finish(summary)
        return summary

    def _flight_result(self, rewards: RewardSystem, design: VehicleDesign) -> dict:
        """Реальный полёт в KSP."""
        if not self.connect():
            rewards.penalise(Failure.CONNECTION_LOST, 0.0)
            summary = rewards.finalize({"final_phase": "no_connection"})
            self.flight_log.finish(summary)
            return summary

        self.flight_log.status("подготовка судна на стартовом столе")
        provision = self.provisioner.provision(design.blueprint)
        log.info("Подготовка судна: %s (%s)", provision.message, provision.strategy)
        if not provision.ok:
            rewards.penalise(Failure.UNEXPECTED_ERROR, 0.0,
                             {"provision": provision.message})
            summary = rewards.finalize({"final_phase": "no_vessel"})
            self.flight_log.error("Судно не удалось вывести на стол", provision.message)
            self.flight_log.finish(summary)
            return summary
        self.flight_log.fact("Судно на стартовом столе", provision.message)

        mission = Mission(self.connection, individual_flight_genome(self),
                          spec=self.spec, rewards=rewards,
                          flight_log=self.flight_log, abort_flag=self.abort_flag)
        return mission.run()

    # ------------------------------------------------------------------
    # Циклы
    # ------------------------------------------------------------------
    def run_generation(self) -> list[RunResult]:
        results = []
        population = list(self.optimizer.population)
        log.info("════ ПОКОЛЕНИЕ %d | особей: %d ════",
                 self.optimizer.generation, len(population))
        for individual in population:
            if self.abort_flag.is_set():
                log.warning("Поколение прервано оператором")
                break
            try:
                self._current_individual = individual
                results.append(self.run_attempt(individual))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                log.error("Попытка сорвалась: %s", exc, exc_info=True)
                individual.score = -1000.0
                self.optimizer.record_result(individual, -1000.0)
            self.optimizer.save_state(OPTIMIZER_STATE)
        return results

    def train(self, generations: int | None = None) -> dict:
        generations = generations or CONFIG.learning.max_generations
        if not self.connect() and not self.offline:
            log.error("Обучение невозможно без kRPC. Запустите KSP "
                      "или используйте офлайн-режим")
            return {"error": "no_connection"}
        try:
            for _ in range(generations):
                if self.abort_flag.is_set():
                    break
                self.run_generation()
                self.optimizer.evolve()
                self.optimizer.save_state(OPTIMIZER_STATE)
                log.info("\n%s", self.history.report())
        except KeyboardInterrupt:
            log.warning("Обучение остановлено пользователем")
        finally:
            self.history.save()
            self.history.save_best_genome()
        return self.history.compute_stats()

    def run_best(self) -> RunResult | None:
        """Один полёт по лучшему известному геному (без эволюции)."""
        best = self.history.best_genome()
        individual = Individual.from_dict(best) if best else Individual()
        individual.generation = self.optimizer.generation
        individual.index = 999
        self._current_individual = individual
        if not self.connect():
            return None
        return self.run_attempt(individual)

    def abort(self) -> None:
        self.abort_flag.set()
        try:
            if self.control is not None:
                self.control.full_stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def goal_key(self) -> str:
        """Идентификатор глобальной цели — своя эволюционная ветка ракеты."""
        s = self.spec
        parts = [s.target_body, f"{s.park_orbit_altitude/1000:.0f}km"]
        if s.needs_transfer:
            parts.append(f"target{s.target_orbit_altitude/1000:.0f}km")
        if s.needs_landing:
            parts.append("landing")
        if s.needs_return:
            parts.append("return")
        return "/".join(parts)

    def status(self) -> str:
        lines = ["=== KERBAL INTELLIGENCE AGENCY ===",
                 f"Задача: {self.spec.title} — {self.spec.describe()}",
                 self.history.report(),
                 f"Поколение оптимизатора: {self.optimizer.generation}",
                 f"Радиус мутации: {self.optimizer.mutation_scale:.3f}",
                 f"Максимум очков за миссию: {self.book.max_possible():.0f}",
                 f"Связь с KSP: {'есть' if (self.connection and self.connection.is_alive()) else 'нет'}",
                 f"Состояние: {'выполняется полёт' if self.busy else 'ожидание команды'}"]
        if self.optimizer.best:
            lines.append(f"Лучшая особь: {self.optimizer.best.describe()} "
                         f"(счёт {self.optimizer.best.score:.1f})")
        return "\n".join(lines)


def individual_flight_genome(dispatcher: "Dispatcher"):
    """Профиль полёта текущей особи (или значения по умолчанию)."""
    individual = getattr(dispatcher, "_current_individual", None)
    if individual is None:
        from ..learning.genome import FlightGenome
        return FlightGenome()
    return individual.flight
