"""Разбор полётов: KIA сама ставит диагноз неудаче и правит конструкцию.

Вход — запись о запуске (очки, достижения, отказы, телеметрия, фазы,
метрики качества) и геном, с которым летели. Выход — список диагнозов,
каждый из которых несёт КОНКРЕТНЫЕ правки инженерной политики и профиля
полёта. Эти правки применяются к следующей итерации напрямую
(ламарковское наследование), а не ждут случайной мутации.

Правила формулируются в терминах физики полёта: «упал при работающем
двигателе и растущей скорости → аэродинамическая неустойчивость →
добавить стабилизаторы и начинать разворот выше».
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from ..config import DATA_DIR, ROOT
from ..logging_setup import get_logger

log = get_logger("learning.diagnosis")

# На сколько градусов фактический тангаж должен отстать от программного,
# чтобы считать, что аппарат ЗАВАЛИЛСЯ, а не полез свечой. Меньше — это
# обычное отставание рулей, оно бывает и на исправном выведении.
FALL_OVER_GAP = 20.0

DIAGNOSES_FILE = DATA_DIR / "diagnoses.json"
AUTONOMY_REPORT = ROOT / "kia_autonomy_report.md"


@dataclass
class Diagnosis:
    """Одна причина неудачи и что с ней делать."""
    cause: str                       # ключ причины
    title: str                       # формулировка для человека
    evidence: str                    # на чём основан вывод
    action: str                      # что меняем и почему
    policy_delta: dict = field(default_factory=dict)   # правки DesignPolicy
    flight_delta: dict = field(default_factory=dict)   # правки FlightGenome
    severity: float = 1.0            # 0..1, вес диагноза
    vehicle_fault: bool = True       # False — ракета исправна, виноват случай/пилотаж

    def to_dict(self) -> dict:
        return asdict(self)


class FailureAnalyst:
    """Аналитик KIA: читает результат запуска и назначает исправления."""

    def __init__(self):
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def analyze(self, record: dict, design_summary: dict | None = None) -> list[Diagnosis]:
        """Главный разбор. record — то, что вернула миссия/RunRecord."""
        design_summary = design_summary or record.get("design_summary", {}) or {}
        milestones = set(record.get("milestones", []))
        failures = list(record.get("failures", []))
        telemetry = record.get("telemetry_final", {}) or {}
        metrics = record.get("metrics", {}) or {}
        phases = [p.get("phase") for p in record.get("phase_log", [])]
        events = record.get("events", []) or []
        peak_alt = float(record.get("peak_altitude", 0.0) or 0.0)
        peak_ap = float(record.get("peak_apoapsis", 0.0) or 0.0)
        verdict = record.get("verdict", {}) or {}

        out: list[Diagnosis] = []

        # --- 0. Случайный сбой, а не дефект аппарата ---
        random_faults = {"connection_lost", "unexpected_error"}
        hit = random_faults.intersection(failures)
        if hit and not {"explosion", "crash", "no_liftoff", "out_of_fuel"} & set(failures):
            out.append(Diagnosis(
                cause="random_fault",
                title="Сбой не связан с конструкцией (связь/сцена/пилотаж)",
                evidence=f"отказы: {', '.join(sorted(hit))}; "
                         f"аппарат успел набрать {peak_alt:.0f} м",
                action="ракета признана исправной — повторяю полёт на ТОЙ ЖЕ "
                       "конструкции без изменений",
                severity=0.0, vehicle_fault=False))
            return self._finish(record, out)

        # --- 1. Конструкция даже не прошла проверку ---
        if "design_rejected" in failures:
            problems = " ".join(verdict.get("problems", []))
            if "TWR" in problems:
                out.append(Diagnosis(
                    cause="design_twr",
                    title="Конструктор не смог обеспечить стартовую тяговооружённость",
                    evidence=problems[:200],
                    action="снижаю требование по запасу ΔV и разрешаю более тяжёлый "
                           "двигатель (уменьшаю штраф за массу)",
                    policy_delta={"dv_margin": -0.04, "mass_penalty": -0.25,
                                  "liftoff_twr": -0.1}))
            else:
                out.append(Diagnosis(
                    cause="design_dv",
                    title="Не хватило ΔV на выбранной элементной базе",
                    evidence=problems[:200] or "проект отклонён валидатором",
                    action="разрешаю более тяжёлую ракету: снижаю штрафы за массу "
                           "и число деталей",
                    policy_delta={"mass_penalty": -0.3, "part_penalty": -0.15,
                                  "dv_margin": -0.03}))
            return self._finish(record, out)

        # --- 2. Не оторвался от стола ---
        if "no_liftoff" in failures:
            twr = _event_detail(events, "no_liftoff", "twr", 0.0)
            out.append(Diagnosis(
                cause="no_liftoff",
                title="Ракета не оторвалась от стартового стола",
                evidence=f"замеренный TWR после зажигания {twr}",
                action="поднимаю целевой стартовый TWR и разрешаю более мощный "
                       "двигатель нижней ступени",
                policy_delta={"liftoff_twr": +0.25, "mass_penalty": -0.2}))
            return self._finish(record, out)

        # --- 3. Разрушение или падение ---
        if "explosion" in failures or "crash" in failures:
            situation = telemetry.get("situation", "")
            vertical = float(telemetry.get("vertical_speed", 0.0) or 0.0)
            fuel_left = _fuel_fraction(telemetry)
            if peak_alt > 1_000 and fuel_left > 0.15:
                # топливо ещё было, но аппарат пошёл вниз — потеря устойчивости
                out.append(Diagnosis(
                    cause="aero_instability",
                    title="Потеря устойчивости в атмосфере (аппарат развернуло потоком)",
                    evidence=f"пик высоты {peak_alt:.0f} м, затем снижение "
                             f"(вертикальная скорость {vertical:.0f} м/с) при остатке "
                             f"топлива {fuel_left*100:.0f}%",
                    action="ставлю стабилизаторы на нижнюю ступень, начинаю разворот "
                           "выше и веду его плавнее, ограничиваю тягу на максимуме напора",
                    policy_delta={"fin_level": +0.4, "liftoff_twr": +0.05},
                    flight_delta={"turn_start_altitude": +400.0,
                                  "turn_exponent": +0.12,
                                  "max_q_throttle": -0.1},
                    severity=1.0))
            elif peak_alt <= 1_000:
                out.append(Diagnosis(
                    cause="crash_on_launch",
                    title="Разрушение сразу после старта",
                    evidence=f"пик высоты {peak_alt:.0f} м, состояние «{situation}»",
                    action="увеличиваю стартовый TWR и откладываю начало разворота",
                    policy_delta={"liftoff_twr": +0.2, "fin_level": +0.3},
                    flight_delta={"turn_start_altitude": +600.0}))
            else:
                out.append(Diagnosis(
                    cause="crash_no_fuel",
                    title="Падение после выработки топлива",
                    evidence=f"пик высоты {peak_alt:.0f} м, остаток топлива "
                             f"{fuel_left*100:.0f}%",
                    action="увеличиваю запас ΔV и долю топлива верхней ступени",
                    policy_delta={"dv_margin": +0.06, "ascent_split": -0.05}))

        # --- 3б. Срыв выведения: аппарат жив, но летит не туда ---
        #
        # Отказ lost_control ставят два разных прибора в ascent.py, и
        # лечатся они ПРОТИВОПОЛОЖНО. Различаем их по тому, что записано в
        # подробностях отказа, иначе диагноз выходил «явная причина не
        # выделена», и оптимизатор не знал, какие гены трогать.
        if "lost_control" in failures:
            total_angle = _event_detail(events, "lost_control", "total_angle", 0.0)
            best_ap = _event_detail(events, "lost_control", "best_apoapsis", 0.0)
            if total_angle >= 30.0:
                # Нос ушёл от потока — это неустойчивость, лечится
                # оперением и более поздним, плавным разворотом.
                out.append(Diagnosis(
                    cause="aero_instability",
                    title="Аппарат опрокинуло потоком на выведении",
                    evidence=f"нос отклонился от потока на {total_angle:.0f}° "
                             f"на высоте "
                             f"{_event_detail(events, 'lost_control', 'altitude', 0.0):.0f} м",
                    action="ставлю стабилизаторы, начинаю разворот выше и веду "
                           "его плавнее, срезаю тягу на максимуме напора",
                    policy_delta={"fin_level": +0.4},
                    flight_delta={"turn_start_altitude": +400.0,
                                  "turn_exponent": +0.10,
                                  "max_q_throttle": -0.1},
                    severity=1.0))
            else:
                # Апоапсис падал при работающем двигателе. Причин ровно
                # две, и лечатся они ПРОТИВОПОЛОЖНО:
                #
                #   летел слишком вертикально — не набрал скорость вбок;
                #   завалился слишком рано    — нос ушёл к горизонту и ниже.
                #
                # Раньше здесь стоял только первый случай, и второй лечился
                # тем же лекарством: «разворачивайся раньше». Два правила
                # толкали профиль в одну сторону без противовеса, и он уехал
                # в пол диапазона — конец разворота с 42 500 до 26 162 м,
                # показатель кривой с 1.26 до 0.90. Ракета стала клевать
                # носом почти сразу после отрыва.
                #
                # Различаем по ФАКТИЧЕСКОМУ тангажу в момент срыва: если он
                # заметно ниже того, что требовал профиль, аппарат завалился,
                # а не полез свечой.
                pitch = _event_detail(events, "lost_control", "pitch", None)
                target = _event_detail(events, "lost_control", "target_pitch", None)
                fell_over = (pitch is not None and target is not None
                             and pitch < target - FALL_OVER_GAP)
                if fell_over:
                    out.append(Diagnosis(
                        cause="ascent_too_flat",
                        title="Аппарат завалился к горизонту раньше времени",
                        evidence=f"профиль требовал {target:.0f}°, фактический "
                                 f"тангаж {pitch:.0f}° — нос ушёл на "
                                 f"{target - pitch:.0f}° ниже программы",
                        action="разворачиваюсь позже и мягче: поднимаю показатель "
                               "кривой тангажа и конец разворота",
                        flight_delta={"turn_exponent": +0.12,
                                      "turn_end_altitude": +5_000.0,
                                      "turn_start_altitude": +100.0},
                        severity=1.0))

                    # ЕСЛИ АППАРАТ ПРИ ЭТОМ ЕЩЁ И НЕ НАБРАЛ ВЫСОТЫ — ОН
                    # ПРОСТО ТЯЖЁЛЫЙ.
                    #
                    # Правило «орбита не замкнулась» добавляет запас ΔV по
                    # +0.04 за раз, чтобы хватило на скругление. Обратного
                    # хода у него нет, и запас дорос до 1.30. Замер восьми
                    # запусков подряд:
                    #
                    #     масса 6.71 т -> апоапсис 28.3 и 80.9 км (успехи)
                    #     масса 8.68 т -> апоапсис 4.8 км (шесть провалов)
                    #
                    # Разница только в массе: потяжелевшая ракета теряет
                    # тяговооружённость и перестаёт набирать вовсе. Запас
                    # ΔV, добавленный ради скругления, съел саму
                    # возможность до этого скругления долететь.
                    if peak_ap < 0.35 * (_target_ap(record) or 80_000.0):
                        out.append(Diagnosis(
                            cause="too_heavy_to_climb",
                            title="Аппарат слишком тяжёл, чтобы набирать высоту",
                            evidence=f"пик апоапсиса всего {peak_ap:.0f} м при "
                                     f"цели {_target_ap(record):.0f} м — тяги не "
                                     f"хватает на такую массу",
                            action="снимаю лишний запас ΔV и сильнее экономлю "
                                   "массу при подборе деталей",
                            policy_delta={"dv_margin": -0.05,
                                          "mass_penalty": +0.15},
                            severity=1.0))
                else:
                    out.append(Diagnosis(
                        cause="ascent_too_steep",
                        title="Выведение слишком крутое: горизонтальной скорости нет",
                        evidence=f"апоапсис не поднялся выше {best_ap:.0f} м и пошёл "
                                 f"вниз при работающем двигателе",
                        action="разворачиваю раньше и энергичнее: уменьшаю показатель "
                               "кривой тангажа и опускаю конец разворота",
                        policy_delta={},
                        flight_delta={"turn_exponent": -0.15,
                                      "turn_end_altitude": -6_000.0,
                                      "turn_start_altitude": -80.0},
                        severity=1.0))

        # --- 4. Топливо кончилось до цели ---
        if "out_of_fuel" in failures:
            out.append(Diagnosis(
                cause="out_of_fuel",
                title="Топливо закончилось раньше расчётного",
                evidence=f"пик апоапсиса {peak_ap:.0f} м при цели "
                         f"{_target_ap(record):.0f} м",
                action="повышаю расчётный запас ΔV и передаю часть работы "
                       "верхней ступени (реальные потери выше табличных)",
                policy_delta={"dv_margin": +0.07, "ascent_split": -0.04}))

        # --- 5. Не вышел на орбиту, хотя атмосферу прошёл ---
        if ("atmosphere_exit" in milestones and "stable_orbit" not in milestones
                and "explosion" not in failures and "crash" not in failures):
            out.append(Diagnosis(
                cause="no_circularization",
                title="Атмосфера пройдена, но орбита не замкнута",
                evidence=f"пик апоапсиса {peak_ap:.0f} м, круговость "
                         f"{metrics.get('accurate_orbit', 0)}",
                action="усиливаю верхнюю ступень (выше требуемый TWR) и начинаю "
                       "ожог скругления раньше",
                policy_delta={"upper_twr": +0.12, "dv_margin": +0.04},
                flight_delta={"circularization_lead": +0.05}))

            # ПОЧЕМУ орбита не замкнулась — вопрос отдельный, и ответ на
            # него виден по периапсису. Если он ушёл глубоко в минус, пока
            # апоапсис на месте, значит аппарат ЗАБРАЛСЯ на высоту, но не
            # РАЗОГНАЛСЯ вбок: пришёл почти вертикально.
            #
            # Замер, ради которого это дописано: апоапсис 84.5 км при
            # периапсисе −182 км. Узел скругления запросил 1948 м/с вместо
            # обычных 700-900 — почти вся орбитальная скорость осталась
            # ненабранной. Правки выше (мощнее верхняя ступень, +4 % запаса
            # ΔV) такую яму не закрывают: они добирают скругление грубой
            # силой, вместо того чтобы не создавать нехватку.
            #
            # Лечение — разворачиваться раньше и заканчивать разворот ниже,
            # чтобы скорость с самого начала шла вбок, а не вверх.
            periapsis = float(telemetry.get("periapsis", 0.0) or 0.0)
            target_ap = _target_ap(record) or 80_000.0
            if periapsis < -0.5 * target_ap:
                out.append(Diagnosis(
                    cause="ascent_too_vertical",
                    title="На орбитальную высоту пришли без орбитальной скорости",
                    evidence=f"апоапсис {peak_ap:.0f} м при периапсисе "
                             f"{periapsis:.0f} м — почти вся скорость "
                             f"вертикальная",
                    action="разворачиваюсь раньше и заканчиваю разворот ниже, "
                           "чтобы тяга шла вбок, а не вверх",
                    flight_delta={"turn_end_altitude": -5_000.0,
                                  "turn_exponent": -0.08,
                                  "turn_start_altitude": -60.0},
                    severity=1.0))

        # --- 6. Слишком долгое выведение / таймаут ---
        if "timeout" in failures:
            out.append(Diagnosis(
                cause="timeout",
                title="Не уложились в отведённое время",
                evidence=f"фазы: {', '.join(phases[-3:]) or '—'}",
                action="разворачиваюсь энергичнее и меньше режу тягу на напоре",
                flight_delta={"turn_exponent": -0.1, "max_q_throttle": +0.1},
                policy_delta={"liftoff_twr": +0.1}))

        # --- 7. Севшие батареи ---
        if "out_of_electricity" in failures:
            out.append(Diagnosis(
                cause="power",
                title="Кончилось электричество",
                evidence="заряд упал ниже порога управления",
                action="ставлю солнечные панели обязательными",
                policy_delta={"solar_level": +0.5}))

        # --- 8. Узел построен, но перехвата нет ---
        if "mun_node_created" in milestones and "mun_encounter" not in milestones:
            out.append(Diagnosis(
                cause="no_encounter",
                title="Манёвр рассчитан, но перехват цели не получился",
                evidence="узел создан, патч не входит в сферу влияния цели",
                action="сдвигаю фазовый угол старта и точнее веду ожог; "
                       "конструкция не виновата — меняю только профиль пилотирования",
                flight_delta={"transfer_phase_bias": +3.0,
                              "node_execute_lead": +0.03},
                vehicle_fault=False))

        # --- 9. В SOI вошли, но не затормозились ---
        if "mun_soi_entered" in milestones and "mun_orbit" not in milestones:
            out.append(Diagnosis(
                cause="no_capture",
                title="Долетели до цели, но не смогли затормозиться на орбиту",
                evidence=f"остаток топлива {_fuel_fraction(telemetry)*100:.0f}%",
                action="увеличиваю запас ΔV и тягу перелётной ступени",
                policy_delta={"dv_margin": +0.06, "transfer_twr": +0.15}))

        # --- 10. Перегрузки и напор ---
        if float(metrics.get("low_gforce", 1.0)) < 0.35:
            out.append(Diagnosis(
                cause="high_g",
                title="Слишком большие перегрузки на выведении",
                evidence=f"метрика мягкости полёта {metrics.get('low_gforce')}",
                action="ограничиваю тягу на участке максимального напора",
                flight_delta={"max_q_throttle": -0.08},
                severity=0.5))

        # --- 11. Успех: закрепляем и оптимизируем ---
        if not failures and "stable_orbit" in milestones:
            fuel_left = _fuel_fraction(telemetry)
            if fuel_left > 0.35:
                out.append(Diagnosis(
                    cause="overbuilt",
                    title="Задача выполнена с большим запасом — ракета избыточна",
                    evidence=f"осталось {fuel_left*100:.0f}% топлива",
                    action="сокращаю запас ΔV и усиливаю экономию массы",
                    policy_delta={"dv_margin": -0.03, "mass_penalty": +0.2},
                    severity=0.4))
            else:
                out.append(Diagnosis(
                    cause="success",
                    title="Задача выполнена, параметры близки к оптимуму",
                    evidence=f"остаток топлива {fuel_left*100:.0f}%",
                    action="фиксирую политику и продолжаю точную донастройку",
                    severity=0.2))

        if not out:
            out.append(Diagnosis(
                cause="unknown",
                title="Явная причина не выделена",
                evidence=f"отказы: {', '.join(failures) or 'нет'}; "
                         f"достижения: {', '.join(sorted(milestones)) or 'нет'}",
                action="увеличиваю разброс поиска, чтобы выйти из локального минимума",
                policy_delta={"dv_margin": +0.02},
                flight_delta={"turn_exponent": +0.05},
                severity=0.3))
        return self._finish(record, out)

    # ------------------------------------------------------------------
    @staticmethod
    def should_keep_design(diagnoses: list[Diagnosis]) -> bool:
        """True — конструкция исправна, следующий полёт на той же ракете."""
        if not diagnoses:
            return True
        return all(not d.vehicle_fault for d in diagnoses)

    # ------------------------------------------------------------------
    def _finish(self, record: dict, diagnoses: list[Diagnosis]) -> list[Diagnosis]:
        entry = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "run_id": record.get("run_id"),
            "score": record.get("total_score", record.get("score")),
            "diagnoses": [d.to_dict() for d in diagnoses],
        }
        self.history.append(entry)
        for d in diagnoses:
            log.info("ДИАГНОЗ: %s | %s | %s", d.title, d.evidence[:80], d.action)
        return diagnoses

    # ------------------------------------------------------------------
    @staticmethod
    def apply(individual, diagnoses: list[Diagnosis]) -> dict:
        """Применяет правки к особи. Возвращает фактически изменённые поля."""
        from ..engineer.autodesign import DesignPolicy
        from .genome import FlightGenome

        applied: dict[str, tuple[float, float]] = {}
        p_bounds, f_bounds = DesignPolicy.bounds(), FlightGenome.bounds()

        for d in diagnoses:
            for name, delta in d.policy_delta.items():
                if name not in p_bounds:
                    continue
                lo, hi = p_bounds[name]
                before = float(getattr(individual.design, name))
                after = max(lo, min(hi, before + delta * d.severity))
                setattr(individual.design, name, after)
                applied[f"design.{name}"] = (round(before, 3), round(after, 3))
            for name, delta in d.flight_delta.items():
                if name not in f_bounds:
                    continue
                lo, hi = f_bounds[name]
                before = float(getattr(individual.flight, name))
                after = max(lo, min(hi, before + delta * d.severity))
                setattr(individual.flight, name, after)
                applied[f"flight.{name}"] = (round(before, 3), round(after, 3))
        individual.design.clamp()
        individual.flight.clamp()
        return applied

    # ------------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        target = Path(path or DIAGNOSES_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.history, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return target


# --------------------------------------------------------------------------
def _event_detail(events: list, label: str, key: str, default=0.0):
    for event in events:
        if event.get("label") == label:
            return event.get("detail", {}).get(key, default)
    return default


def _fuel_fraction(telemetry: dict) -> float:
    resources = telemetry.get("resources", {}) or {}
    lf = resources.get("LiquidFuel", {})
    if not lf or not lf.get("max"):
        return 0.0
    return float(lf.get("amount", 0.0)) / float(lf["max"])


def _target_ap(record: dict) -> float:
    telemetry = record.get("telemetry_final", {}) or {}
    return float(telemetry.get("apoapsis", 0.0) or 0.0)
