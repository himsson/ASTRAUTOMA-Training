"""Полностью автономный режим KIA: бесконечный цикл штурма космоса.

    проектирование → сборка → полёт → разбор отказа → правка чертежа → снова

Никакого участия оператора не требуется. Цикл переживает падение KSP и
потерю связи: ждёт возвращения сервера kRPC и продолжает с того места,
где остановился. Каждая итерация пишется в kia_autonomy_report.md.

Логика поиска двухуровневая:
  * быстрый контур — аналитик (FailureAnalyst) ставит диагноз по итогам
    полёта и целенаправленно правит политику конструктора и профиль полёта
    следующей итерации (ламарковское наследование);
  * медленный контур — генетический алгоритм: раз в поколение популяция
    обновляется отбором, скрещиванием и мутацией, чтобы не застрять
    в локальном минимуме.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from ..config import DATA_DIR, ROOT
from ..learning.diagnosis import AUTONOMY_REPORT, Diagnosis, FailureAnalyst
from ..learning.genome import Individual, load_design_lock
from ..logging_setup import get_logger
from ..memory import GLOBAL_MEMORY

log = get_logger("agent.autonomy")

STATE_FILE = DATA_DIR / "autonomy_state.json"


@dataclass
class Iteration:
    number: int
    started_at: str
    score: float = 0.0
    milestones: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    design: dict = field(default_factory=dict)
    diagnoses: list = field(default_factory=list)
    changes: dict = field(default_factory=dict)
    duration_s: float = 0.0
    folder: str = ""
    kept_design: bool = False
    new_skills: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class AutonomousLoop:
    """Бесконечный цикл обучения без участия человека."""

    def __init__(self, dispatcher, analyst: FailureAnalyst | None = None,
                 max_iterations: int | None = None,
                 evolve_every: int | None = None):
        self.dispatcher = dispatcher
        self.analyst = analyst or FailureAnalyst()
        self.max_iterations = max_iterations
        self.evolve_every = evolve_every or max(2, dispatcher.optimizer.cfg.population_size)
        self.stop_flag = threading.Event()
        self.iterations: list[Iteration] = []
        self.best_score: float = float("-inf")
        self.best_individual: Individual | None = None
        self.current: Individual | None = None
        self.started_at = datetime.now()
        self.consecutive_errors = 0
        self.memory = GLOBAL_MEMORY
        self.goal = dispatcher.goal_key()
        self._load_state()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self.stop_flag.set()
        self.dispatcher.abort()

    @property
    def running(self) -> bool:
        return not self.stop_flag.is_set()

    # ------------------------------------------------------------------
    def next_candidate(self) -> Individual:
        """Особь для следующей попытки.

        Приоритет: правка аналитика → незавершённая особь популяции →
        новая от эволюции.

        Подтверждённые навыки из памяти накладываются НЕ ВСЕГДА, и это
        важно. Память тянет кандидата к доказанным значениям на половину
        пути — для новой, ниоткуда не взявшейся особи это правильно: она
        начинает не с чистого листа, а с рабочего режима.

        Но если особь пришла от аналитика, она уже несёт правку, сделанную
        по итогам ТОЛЬКО ЧТО состоявшегося полёта. Накладывать поверх неё
        память значит спорить со свежими данными в пользу старых. На
        замерах это выглядело так:

            правка -0.080 -> применилось -0.040
            правка -5000  -> применилось -2500

        Ровно половина, то есть ряд сходится, не доходя до цели. Сначала
        так гасились попытки уйти от провального профиля, потом — попытки
        улучшить рабочий. Оба раза причина одна: последнее слово оставалось
        за памятью, а не за последним полётом.

        Сначала память для таких особей отключили совсем. Стало хуже:
        правки пошли вразнос, и за две итерации счёт упал 562 -> 370 -> -116,
        а навык «атмосфера пройдена» потерялся. Оказалось, память не только
        мешала — она ещё и удерживала доказанный профиль от развала.

        Поэтому не «или-или», а ослабленный якорь: у особи от аналитика
        память тянет вчетверо слабее обычного. Правка по свежему полёту
        остаётся главной и накапливается от итерации к итерации, но уйти
        далеко от режима, который однажды довёл аппарат до космоса, она
        уже не даёт.
        """
        from_analyst = self.current is not None
        if self.current is not None:
            candidate = self.current
        else:
            pending = [i for i in self.dispatcher.optimizer.population
                       if i.score is None]
            if pending:
                candidate = pending[0]
            else:
                self.dispatcher.optimizer.evolve()
                self.dispatcher.optimizer.save_state(
                    Path(DATA_DIR) / "optimizer_state.json")
                candidate = self.dispatcher.optimizer.population[0]

        # Незыблемая основа из памяти: чем больше подтверждений, тем сильнее.
        # Для особи от аналитика — вчетверо слабее (см. пояснение выше).
        if self.memory.skills:
            strength = 0.85 if self.memory.has_skill("stable_orbit") else 0.5
            if from_analyst:
                strength *= 0.25
            self.memory.apply_to(candidate, strength=strength)
        # ЗАКРЕПЛЁННАЯ ОПЕРАТОРОМ КОНСТРУКЦИЯ ГЛАВНЕЕ ЭВОЛЮЦИИ.
        #
        # `main.py craft` строит по настройкам ПО УМОЛЧАНИЮ, а цикл — по
        # геному эволюции плюс параметры из памяти. Это разные ракеты, и
        # оператор, одобрив собранную в ангаре, видел в полёте другую:
        # вместо 60.7 т на «Бобкэте» с четырьмя «Колотушками» летели
        # 74.7 т на «Мастодонте». Если конструкция закреплена, эволюция
        # продолжает искать профиль ПОЛЁТА, но саму ракету не трогает.
        locked = load_design_lock()
        if locked is not None:
            candidate.design = locked
            candidate.design.clamp()

        warning = self.memory.forbidden_zone(candidate)
        if warning:
            log.info("Память блокирует известную ошибку: %s", warning)
            candidate.design.liftoff_twr = max(candidate.design.liftoff_twr + 0.2,
                                               1.5)
            candidate.design.fin_level = max(candidate.design.fin_level, 0.6)
            candidate.design.clamp()
        return candidate

    # ------------------------------------------------------------------
    def wait_for_game(self, timeout: float = 3600.0) -> bool:
        """Ждёт возвращения kRPC: KSP мог упасть или его перезапускают."""
        deadline = time.time() + timeout
        delay = 10.0
        while time.time() < deadline and self.running:
            if self.dispatcher.connect():
                log.info("Связь с KSP восстановлена")
                return True
            log.warning("KSP недоступен — жду %.0f с и пробую снова", delay)
            slept = 0.0
            while slept < delay and self.running:
                time.sleep(1.0)
                slept += 1.0
            delay = min(120.0, delay * 1.6)
        return False

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Главный цикл. Работает, пока не остановят."""
        log.info("════════ АВТОНОМНЫЙ РЕЖИМ ЗАПУЩЕН ════════")
        log.info("Задача: %s", self.dispatcher.spec.describe())
        log.info("Ветка эволюции: %s", self.goal)
        if self.memory.skills:
            log.info("Из памяти доступно навыков: %d (%s)", len(self.memory.skills),
                     ", ".join(sorted(self.memory.skills)))
        seed = self.memory.branch_seed(self.goal)
        if seed is not None and self.current is None:
            log.info("Ветка «%s» продолжается с лучшего известного генома", self.goal)
            seed.score = None
            self.current = seed
        if not self.dispatcher.offline and not self.wait_for_game():
            log.error("Связь с KSP так и не появилась — цикл не запущен")
            return self.summary()

        # Предел считается для ТЕКУЩЕЙ сессии, а не для всей истории:
        # цикл продолжает нумерацию с прошлого раза, но лимит начинает с нуля.
        start_number = len(self.iterations)
        number = start_number
        while self.running:
            if self.max_iterations and (number - start_number) >= self.max_iterations:
                log.info("Достигнут предел итераций сессии (%d)", self.max_iterations)
                break
            number += 1
            iteration = Iteration(number=number,
                                  started_at=datetime.now().isoformat(timespec="seconds"))
            t0 = time.time()
            individual = self.next_candidate()
            individual.generation = self.dispatcher.optimizer.generation
            individual.index = number

            log.info("──────── ИТЕРАЦИЯ %d ────────", number)
            log.info("Кандидат: %s", individual.describe())

            try:
                result = self.dispatcher.run_attempt(individual)
            except Exception as exc:
                self.consecutive_errors += 1
                log.error("Итерация сорвалась: %s", exc, exc_info=True)
                if not self.dispatcher.offline and not self.dispatcher.connection_alive():
                    if not self.wait_for_game():
                        break
                if self.consecutive_errors >= 5:
                    log.error("Пять сбоев подряд — останавливаю цикл")
                    break
                continue

            self.consecutive_errors = 0
            record = result.record.to_dict()
            record.update({"metrics": (result.record.telemetry_final or {}).get("metrics", {})})

            # --- разбор и правка ---
            diagnoses = self.analyst.analyze(self._record_for_analysis(result),
                                             result.record.design_summary)
            keep_design = FailureAnalyst.should_keep_design(diagnoses)
            next_individual = individual.copy()
            next_individual.score = None
            if keep_design:
                changes = {}
                log.info("Конструкция признана исправной — следующий полёт "
                         "на той же ракете")
            else:
                changes = FailureAnalyst.apply(next_individual, diagnoses)

            iteration.score = result.score
            iteration.milestones = result.record.milestones
            iteration.failures = result.record.failures
            iteration.design = {
                "mass_t": result.record.design_summary.get("total_mass_t"),
                "delta_v": result.record.design_summary.get("total_delta_v"),
                "twr": result.record.design_summary.get("launch_twr"),
                "parts": result.record.design_summary.get("part_count"),
                "stages": len(result.record.design_summary.get("stages", [])),
            }
            iteration.diagnoses = [d.to_dict() for d in diagnoses]
            iteration.changes = {k: list(v) for k, v in changes.items()}
            iteration.duration_s = round(time.time() - t0, 1)
            iteration.kept_design = keep_design
            if result.run_folder is not None:
                iteration.folder = f"kia/{result.run_folder.number}"
                iteration.new_skills = result.run_folder.read_meta().get(
                    "new_skills", [])
            self.iterations.append(iteration)

            if result.score > self.best_score:
                self.best_score = result.score
                self.best_individual = individual.copy()
                log.info("НОВЫЙ РЕКОРД АВТОНОМНОГО РЕЖИМА: %.0f очков", result.score)

            log.info("Итерация %d (%s): %.0f очков | %s | правок: %d%s",
                     number, iteration.folder or "—", result.score,
                     ", ".join(result.record.milestones) or "без достижений",
                     len(changes),
                     " | ракета сохранена" if keep_design else "")
            if iteration.new_skills:
                log.info("В глобальную память записаны навыки: %s",
                         ", ".join(iteration.new_skills))

            # Следующая попытка идёт с исправлениями аналитика
            self.current = next_individual
            # Раз в N итераций — обновление популяции эволюцией
            if number % self.evolve_every == 0:
                log.info("Плановое обновление популяции (эволюция)")
                # Накопленную аналитиком правку в новую популяцию НАДО
                # передать. Раньше она просто терялась: `self.current`
                # обнулялся, и следующая особь бралась из свежей популяции
                # с нуля. На замерах это видно прямо — две итерации подряд
                # стартовали с одного и того же значения:
                #
                #     итерация 6: turn_exponent 1.129 -> 1.049
                #     итерация 7: turn_exponent 1.129 -> 1.049
                #
                # То есть шесть правок шестой итерации ушли в никуда, и
                # цепочка исправлений каждый раз начиналась заново. Профиль
                # упирался в невидимый потолок, хотя аналитик исправно
                # работал.
                carried = next_individual.copy()
                carried.score = None
                self.dispatcher.optimizer.record_result(individual, result.score)
                population = self.dispatcher.optimizer.evolve()
                # Исправленная особь становится первой в новой популяции:
                # разведка новых сочетаний продолжается, но накопленное
                # знание о том, что именно надо чинить, не выбрасывается.
                if population:
                    carried.generation = population[0].generation
                    carried.index = 0
                    population[0] = carried
                self.current = None

            self.analyst.save()
            self._save_state()
            self.write_report()

        log.info("════════ АВТОНОМНЫЙ РЕЖИМ ОСТАНОВЛЕН ════════")
        self.write_report()
        return self.summary()

    # ------------------------------------------------------------------
    def _record_for_analysis(self, result) -> dict:
        rec = result.record
        data = rec.to_dict()
        telemetry = rec.telemetry_final or {}
        data["telemetry_final"] = telemetry
        data["peak_altitude"] = _find_event_value(rec.events, "peak_altitude",
                                                  telemetry.get("altitude", 0.0))
        data["peak_apoapsis"] = telemetry.get("apoapsis", 0.0)
        data["metrics"] = data.get("metrics", {})
        data["phase_log"] = data.get("phase_log", [])
        data["total_score"] = rec.score
        return data

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        runtime = (datetime.now() - self.started_at).total_seconds()
        scores = [i.score for i in self.iterations]
        return {
            "iterations": len(self.iterations),
            "runtime_h": round(runtime / 3600.0, 2),
            "best_score": self.best_score if scores else None,
            "mean_score": round(sum(scores) / len(scores), 1) if scores else None,
            "milestones_reached": sorted({m for i in self.iterations for m in i.milestones}),
            "top_failures": _top([f for i in self.iterations for f in i.failures]),
            "top_causes": _top([d["cause"] for i in self.iterations
                                for d in i.diagnoses]),
        }

    # ------------------------------------------------------------------
    def write_report(self, path: Path | None = None) -> Path:
        target = Path(path or AUTONOMY_REPORT)
        s = self.summary()
        L = ["# Автономный режим KIA", "",
             f"**Запущен:** {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}  ",
             f"**Задача:** {self.dispatcher.spec.title} — "
             f"{self.dispatcher.spec.describe()}  ",
             f"**Итераций:** {s['iterations']} за {s['runtime_h']} ч  ",
             f"**Лучший счёт:** {s['best_score'] if s['best_score'] is not None else '—'}"
             f" | средний: {s['mean_score'] if s['mean_score'] is not None else '—'}  ",
             f"**Достигнутые рубежи:** {', '.join(s['milestones_reached']) or '—'}  ",
             f"**Частые отказы:** "
             f"{', '.join(f'{k}×{v}' for k, v in s['top_failures']) or '—'}  ",
             f"**Частые причины:** "
             f"{', '.join(f'{k}×{v}' for k, v in s['top_causes']) or '—'}", "",
             "## Ход обучения", "",
             "| # | Счёт | Масса, т | ΔV | TWR | Ступ. | Итог | Диагноз | Что изменено |",
             "|---|---|---|---|---|---|---|---|---|"]
        for it in self.iterations[-60:]:
            outcome = ", ".join(it.failures) or ", ".join(it.milestones[-2:]) or "—"
            diagnosis = "; ".join(d["title"] for d in it.diagnoses)[:80]
            changes = ", ".join(f"{k.split('.')[-1]} {v[0]}→{v[1]}"
                                for k, v in it.changes.items())[:90] or "—"
            L.append(f"| {it.number} | {it.score:.0f} | {it.design.get('mass_t','—')} | "
                     f"{it.design.get('delta_v','—')} | {it.design.get('twr','—')} | "
                     f"{it.design.get('stages','—')} | {outcome} | {diagnosis} | {changes} |")
        L.append("")

        if self.iterations:
            last = self.iterations[-1]
            L += ["## Последний разбор", ""]
            for d in last.diagnoses:
                L += [f"### {d['title']}", "",
                      f"- **Основание:** {d['evidence']}",
                      f"- **Решение:** {d['action']}",
                      f"- **Правки политики:** "
                      f"{d['policy_delta'] or '—'} / профиля: {d['flight_delta'] or '—'}",
                      ""]
        if self.best_individual is not None:
            L += ["## Лучшая найденная конфигурация", "",
                  f"Счёт: **{self.best_score:.0f}**", "",
                  "```json",
                  json.dumps(self.best_individual.to_dict(), ensure_ascii=False, indent=2),
                  "```", ""]
        target.write_text("\n".join(L) + "\n", encoding="utf-8")
        return target

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "iterations": [i.to_dict() for i in self.iterations[-200:]],
            "best_score": self.best_score if self.best_score > float("-inf") else None,
            "best_individual": (self.best_individual.to_dict()
                                if self.best_individual else None),
            "current": self.current.to_dict() if self.current else None,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self.iterations = [Iteration(**i) for i in data.get("iterations", [])]
        if data.get("best_score") is not None:
            self.best_score = float(data["best_score"])
        if data.get("best_individual"):
            self.best_individual = Individual.from_dict(data["best_individual"])
        if data.get("current"):
            self.current = Individual.from_dict(data["current"])
        if self.iterations:
            log.info("Автономный режим продолжает работу: выполнено %d итераций, "
                     "рекорд %.0f", len(self.iterations), self.best_score)


# --------------------------------------------------------------------------
def _top(items: list[str], limit: int = 5) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:limit]


def _find_event_value(events: list, key: str, default: float) -> float:
    for event in events or []:
        detail = event.get("detail", {})
        if key in detail:
            try:
                return float(detail[key])
            except (TypeError, ValueError):
                continue
    return float(default or 0.0)
