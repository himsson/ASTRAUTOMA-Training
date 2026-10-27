"""ПРОЕКТНОЕ БЮРО KIA — опыт полётов замыкается на конструктора.

Задача модуля одна: сделать так, чтобы шишки, набитые пилотом в воздухе,
меняли следующий чертёж. До сих пор две ветки системы жили порознь —
конструктор считал ΔV и TWR по формулам, пилот падал из-за аэродинамики,
и никто из них об этом друг другу не сообщал. Здесь эта петля
замыкается (closed-loop experience transfer).

ЧТО ДЕЛАЕТ МОДУЛЬ

1. ЧИТАЕТ ОПЫТ. Разбирает `kia_global_memory.json` (уроки отказов и
   подтверждённые навыки), `data/diagnoses.json` (разбор каждого полёта),
   журналы `kia_flight_log.md` и `kia/<N>/kia_flight_log.md`. Из этого
   собирается давление опыта: насколько сильно прошлые полёты требуют
   аэродинамики, тяги, лёгкости.

2. ПРАВИТ ФОРМУ. Если пилот терял скорость на сопротивлении или его
   опрокидывало на сверхзвуке, бюро режет мидель (максимальный диаметр
   ступеней), требует носовой обтекатель и ставит стабилизаторы —
   ровно в той мере, в какой это подтверждено полётами.

3. ПРОВЕРЯЕТ ЧЕРТЁЖ по многокритериальному чек-листу и возвращает
   маркеры ошибок, понятные и человеку, и обучению:

       CRITICAL_LOW_TWR   TWR < 1.1 — не оторвётся
       OVER_THRUST        TWR > 2.5 — сгорит в плотных слоях
       AERO_UNSTABLE      центр масс позади центра давления — опрокинет
       FUEL_OVERBUILT     ΔV с запасом больше 30 % — овер-инжиниринг
       POOR_MASS_RATIO    сухая масса съедает полезный импульс
       VACUUM_ENGINE_LOW  вакуумный двигатель поставлен вниз
       ATMO_ENGINE_HIGH   атмосферный тяжеловес поставлен наверх
       NO_NOSE_CONE       нет обтекателя при подтверждённых потерях на drag
       NO_FINS            нет оперения при подтверждённых опрокидываниях

4. ШТРАФУЕТ ЗА ИЗБЫТОК. Принцип разумной достаточности: ΔV сверх
   потребности миссии на 25–30 % ещё прощается, дальше награда режется.
   Это заставляет облегчать сухую массу, а не заливать топливо.

5. ЗНАЕТ АЭРОКОСМИЧЕСКУЮ БАЗУ: уравнение Циолковского, иерархию
   удельных импульсов (атмосферные тяжеловесы вниз, вакуумные наверх),
   шаблон симметричных радиальных ускорителей через разделители при
   нехватке стартового TWR.

ГДЕ РАБОТАЕТ

Только на Тренажёре+ и Тренажёре++ (ступени 2 и 3 учебной программы).
На базовом тренажёре бюро молчит: там сеть ещё учится вообще выводить
аппарат на орбиту, и разговор про мидель и обтекатели ей рано. Порог
задан константой `MIN_STAGE`, проверка — `is_enabled(stage)`.

ЧЕГО МОДУЛЬ НЕ ДЕЛАЕТ

Не запускает игру, не читает kRPC и не трогает веса нейросети. Он
работает с уже записанными логами и со списком деталей.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import sys

if __package__ in (None, ""):                      # запуск файлом напрямую
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kia_core.config import DATA_DIR, G0, ROOT
from kia_core.logging_setup import get_logger

log = get_logger("designer_pro")

# Ступень учебной программы, начиная с которой бюро включается.
# 1 — базовый тренажёр: там про аэродинамику говорить рано.
MIN_STAGE = 2

# --------------------------------------------------------------------------
# Источники полётного опыта
# --------------------------------------------------------------------------
MEMORY_FILE = ROOT / "kia_global_memory.json"
DIAGNOSES_FILE = DATA_DIR / "diagnoses.json"
FLIGHT_LOG_MD = ROOT / "kia_flight_log.md"
FLIGHT_LOG_JSON = ROOT / "kia_flight_log.json"
RUNS_DIR = ROOT / "kia"

# Маркеры отказов, которые ГОВОРЯТ ОБ АЭРОДИНАМИКЕ. Разделены по смыслу:
# от одних лечит форма корпуса, от других — оперение.
DRAG_CAUSES = {"high_g", "overbuilt", "no_circularization"}
TUMBLE_CAUSES = {"aero_instability", "crash_on_launch"}
THRUST_CAUSES = {"no_liftoff", "design_twr"}
MASS_CAUSES = {"out_of_fuel", "crash_no_fuel", "design_dv"}

# Слова в журналах пилота, по которым узнаётся та же беда. Журнал
# человекочитаемый, и это единственный способ вытащить из него факт.
DRAG_WORDS = ("сопротивлен", "напор", "max q", "max_q", "потеря скорости",
              "тормозит", "перегрев")
TUMBLE_WORDS = ("опрокид", "устойчив", "развернуло", "кувыр", "сверхзвук",
                "потеря управления", "закрутило")
THRUST_WORDS = ("не оторв", "не взлет", "twr")

# --------------------------------------------------------------------------
# Требования миссии, м/с. Совпадают с расчётом конструктора.
# --------------------------------------------------------------------------
DV_TO_ORBIT = 3400.0
DV_TRANSFER = 1000.0
DV_LANDING = 1200.0

# Разумная достаточность: запас до 25 % бесплатен, после 30 % режем.
OVERBUILD_FREE = 0.25
OVERBUILD_HARD = 0.30

# Чек-лист тяговооружённости
TWR_CRITICAL_LOW = 1.1
TWR_OVER_THRUST = 2.5

# Циолковский: ниже этого отношения масс сухая конструкция съедает импульс
MIN_MASS_RATIO = 1.8

# Вклады в нормальную силу (упрощение схемы Барроумена). Нос и корпус
# отнесены к миделю, оперение — к собственной площади.
NOSE_LIFT = 1.0
BODY_LIFT = 0.15
FIN_LIFT = 3.0

# Иерархия Isp: двигатель считается вакуумным, если у земли он теряет
# больше этой доли удельного импульса (Terrier: 345 -> 85, это 0.75)
VACUUM_ENGINE_LOSS = 0.45
# ...и атмосферным тяжеловесом, если тяга большая, а потери малые
ATMO_ENGINE_THRUST = 150.0        # кН


def is_enabled(stage: int | None) -> bool:
    """Работает ли бюро на этой ступени учебной программы."""
    try:
        return int(stage or 0) >= MIN_STAGE
    except (TypeError, ValueError):
        return False


# ==========================================================================
# 1. ЧТЕНИЕ ПОЛЁТНОГО ОПЫТА
# ==========================================================================
@dataclass
class FlightExperience:
    """Что пилот уже выяснил на практике — в числах.

    Каждое «давление» лежит в диапазоне 0..1 и означает: насколько
    настойчиво прошлые полёты требуют именно этого изменения конструкции.
    Ноль — опыта нет, единица — беда повторялась столько раз, что спорить
    не о чем.
    """
    flights: int = 0
    drag_events: int = 0
    tumble_events: int = 0
    thrust_events: int = 0
    mass_events: int = 0
    sources: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    # --- давление опыта -------------------------------------------------
    @staticmethod
    def _pressure(count: int) -> float:
        """Насыщение: первый случай уже слышен, три — почти максимум."""
        if count <= 0:
            return 0.0
        return min(1.0, 1.0 - math.exp(-0.8 * count))

    @property
    def drag_pressure(self) -> float:
        return self._pressure(self.drag_events)

    @property
    def tumble_pressure(self) -> float:
        return self._pressure(self.tumble_events)

    @property
    def thrust_pressure(self) -> float:
        return self._pressure(self.thrust_events)

    @property
    def mass_pressure(self) -> float:
        return self._pressure(self.mass_events)

    @property
    def aero_pressure(self) -> float:
        """Общее требование «займись аэродинамикой»."""
        return max(self.drag_pressure, self.tumble_pressure)

    @property
    def has_experience(self) -> bool:
        return bool(self.drag_events or self.tumble_events
                    or self.thrust_events or self.mass_events)

    def describe(self) -> str:
        if not self.has_experience:
            return ("Полётного опыта нет: бюро проектирует по чистой теории, "
                    "аэродинамических требований не предъявляет.")
        lines = [f"Разобрано полётов: {self.flights}. Что усвоено:"]
        rows = (("потери на сопротивлении", self.drag_events, self.drag_pressure),
                ("опрокидывания в атмосфере", self.tumble_events,
                 self.tumble_pressure),
                ("нехватка стартовой тяги", self.thrust_events,
                 self.thrust_pressure),
                ("нехватка ΔV / лишняя масса", self.mass_events,
                 self.mass_pressure))
        for title, count, pressure in rows:
            if count:
                lines.append(f"  • {title}: {count} раз "
                             f"(давление на проект {pressure:.0%})")
        for note in self.notes[:6]:
            lines.append(f"    — {note}")
        if self.sources:
            lines.append("  источники: " + ", ".join(self.sources))
        return "\n".join(lines)


class ExperienceReader:
    """Собирает опыт пилота из всех файлов, где он вообще остаётся."""

    def __init__(self, memory_file: Path | None = None,
                 diagnoses_file: Path | None = None,
                 flight_log: Path | None = None,
                 runs_dir: Path | None = None):
        self.memory_file = Path(memory_file or MEMORY_FILE)
        self.diagnoses_file = Path(diagnoses_file or DIAGNOSES_FILE)
        self.flight_log = Path(flight_log or FLIGHT_LOG_MD)
        self.runs_dir = Path(runs_dir or RUNS_DIR)

    # ------------------------------------------------------------------
    def read(self) -> FlightExperience:
        experience = FlightExperience()
        self._read_memory(experience)
        self._read_diagnoses(experience)
        self._read_logs(experience)
        log.debug("Полётный опыт: drag=%d tumble=%d thrust=%d mass=%d",
                  experience.drag_events, experience.tumble_events,
                  experience.thrust_events, experience.mass_events)
        return experience

    # ------------------------------------------------------------------
    def _read_memory(self, experience: FlightExperience) -> None:
        """Уроки отказов из долгосрочной памяти — самый весомый источник."""
        data = _load_json(self.memory_file)
        if not isinstance(data, dict):
            return
        lessons = data.get("failure_lessons") or {}
        if not isinstance(lessons, dict):
            return
        touched = False
        for cause, lesson in lessons.items():
            count = int((lesson or {}).get("count", 0) or 0)
            if count <= 0:
                continue
            touched = True
            if cause in TUMBLE_CAUSES:
                experience.tumble_events += count
                experience.notes.append(
                    f"память: «{cause}» {count}× — аппарат теряет устойчивость")
            elif cause in DRAG_CAUSES:
                experience.drag_events += count
                experience.notes.append(
                    f"память: «{cause}» {count}× — потери на сопротивлении")
            elif cause in THRUST_CAUSES:
                experience.thrust_events += count
                experience.notes.append(
                    f"память: «{cause}» {count}× — не хватает стартовой тяги")
            elif cause in MASS_CAUSES:
                experience.mass_events += count
        experience.flights += int(data.get("total_runs", 0) or 0)
        if touched:
            experience.sources.append(self.memory_file.name)

    # ------------------------------------------------------------------
    def _read_diagnoses(self, experience: FlightExperience) -> None:
        """Разбор каждого полёта: причина отказа названа явно."""
        records = _load_json(self.diagnoses_file)
        if not isinstance(records, list):
            return
        touched = False
        for record in records:
            if not isinstance(record, dict):
                continue
            for diagnosis in record.get("diagnoses") or []:
                if not isinstance(diagnosis, dict):
                    continue
                cause = str(diagnosis.get("cause", ""))
                if not diagnosis.get("vehicle_fault", True):
                    continue                 # сбой связи — конструкция ни при чём
                severity = float(diagnosis.get("severity", 0.5) or 0.0)
                weight = 1 if severity >= 0.5 else 0
                if cause in TUMBLE_CAUSES:
                    experience.tumble_events += max(1, weight)
                    touched = True
                elif cause in DRAG_CAUSES:
                    experience.drag_events += max(1, weight)
                    touched = True
                elif cause in THRUST_CAUSES:
                    experience.thrust_events += max(1, weight)
                    touched = True
                elif cause in MASS_CAUSES:
                    experience.mass_events += max(1, weight)
                    touched = True
        if touched:
            experience.sources.append(self.diagnoses_file.name)

    # ------------------------------------------------------------------
    def _read_logs(self, experience: FlightExperience) -> None:
        """Журналы пилота: текст, писавшийся по ходу полёта."""
        candidates = [self.flight_log]
        if self.runs_dir.is_dir():
            candidates += sorted(self.runs_dir.glob("*/kia_flight_log.md"))[-8:]
        touched = False
        for path in candidates:
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            interesting = [line for line in text.splitlines()
                           if line.startswith(("| ", "* ", "- ", "⚠", "✖"))
                           or "⚠" in line or "✖" in line]
            body = "\n".join(interesting) or text
            drag = _count_words(body, DRAG_WORDS)
            tumble = _count_words(body, TUMBLE_WORDS)
            thrust = _count_words(body, THRUST_WORDS)
            if drag or tumble or thrust:
                touched = True
                # журнал многословен: одно и то же событие попадает в него
                # десятки раз, поэтому считаем факт, а не число упоминаний
                experience.drag_events += 1 if drag else 0
                experience.tumble_events += 1 if tumble else 0
                experience.thrust_events += 1 if thrust else 0
                experience.notes.append(
                    f"журнал {path.name}: сопротивление {drag}, "
                    f"устойчивость {tumble}, тяга {thrust}")
        if touched:
            experience.sources.append("журналы полётов")


def _count_words(text: str, words: tuple) -> int:
    return sum(text.count(word) for word in words)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ==========================================================================
# 2. ГЕОМЕТРИЯ И УСТОЙЧИВОСТЬ
# ==========================================================================
@dataclass
class Layout:
    """Продольная раскладка аппарата: смещения от носа, метры (вниз — плюс)."""
    length: float = 0.0
    com: float = 0.0            # центр масс
    cop: float = 0.0            # центр аэродинамического давления
    max_diameter: float = 0.0
    frontal_area: float = 0.0
    fin_area: float = 0.0
    has_nose_cone: bool = False

    @property
    def static_margin(self) -> float:
        """Запас устойчивости: центр давления минус центр масс.

        Положительный — центр давления ПОЗАДИ центра масс, поток
        возвращает нос на место. Отрицательный — аппарат опрокидывает.
        Знак совпадает с моделью Тренажёра++ (`sim_env_pro`), иначе бюро
        и симулятор спорили бы об одном и том же аппарате.
        """
        return self.cop - self.com

    @property
    def stable(self) -> bool:
        return self.static_margin > 0.0


def build_layout(parts, densities: dict | None = None) -> Layout:
    """Раскладывает детали вдоль корпуса сверху вниз и считает ЦМ и ЦД.

    Порядок деталей — тот, в котором их выбрал конструктор: первая идёт
    в нос. Это же соглашение использует `BuilderEnv.to_blueprint`.
    """
    densities = densities or {}
    layout = Layout()
    if not parts:
        return layout

    offset = 0.0
    mass_moment = 0.0
    total_mass = 0.0
    area_moment = 0.0
    total_area = 0.0

    for part in parts:
        height = max(0.3, float(getattr(part, "height", 0.5) or 0.5))
        diameter = max(0.3, float(getattr(part, "diameter", 0.0) or 0.0))
        try:
            mass = float(part.wet_mass(densities))
        except (AttributeError, TypeError):
            mass = float(getattr(part, "dry_mass", 0.0) or 0.0)
        frontal = math.pi * (diameter / 2.0) ** 2

        # Оперение и прочая радиальная обвязка крепятся СБОКУ и не
        # удлиняют корпус: они стоят у того среза, до которого стек уже
        # дорос. Если считать их продолжением стека, ракета получается
        # длиннее, чем есть, и центр давления уезжает в хвост.
        radial = _is_fin(part)
        centre = offset if radial else offset + height / 2.0

        mass_moment += mass * centre
        total_mass += mass
        layout.max_diameter = max(layout.max_diameter, diameter)

        if _is_nose_cone(part):
            layout.has_nose_cone = True
            # Нормальная сила носа отнесена к миделю (так её и считают),
            # приложена у самого носа — то есть ДЕСТАБИЛИЗИРУЕТ. Именно
            # поэтому ракете с обтекателем оперение нужно даже сильнее,
            # чем тупоносой.
            area_moment += frontal * NOSE_LIFT * (offset + height * 0.5)
            total_area += frontal * NOSE_LIFT
        elif radial:
            # Оперение — несущая поверхность: считаем по собственной
            # площади и с большим коэффициентом, оно и стабилизирует
            fin_area = height * diameter * FIN_LIFT
            layout.fin_area += fin_area
            area_moment += fin_area * centre
            total_area += fin_area
        else:
            # Цилиндрический корпус почти нейтрален
            area_moment += frontal * BODY_LIFT * centre
            total_area += frontal * BODY_LIFT
        if not radial:
            offset += height

    layout.length = offset
    layout.com = mass_moment / total_mass if total_mass > 0 else 0.0
    layout.cop = area_moment / total_area if total_area > 0 else offset * 0.5
    layout.frontal_area = math.pi * (layout.max_diameter / 2.0) ** 2
    return layout


def _is_fin(part) -> bool:
    if getattr(part, "category", "") == "Aero":
        return not _is_nose_cone(part)
    name = f"{getattr(part, 'name', '')} {getattr(part, 'title', '')}".lower()
    return any(word in name for word in ("fin", "winglet", "стабилиз"))


def _is_nose_cone(part) -> bool:
    name = f"{getattr(part, 'name', '')} {getattr(part, 'title', '')}".lower()
    return any(word in name for word in ("nosecone", "nose cone", "cone",
                                         "обтекат", "shockcone", "aerodynamicnose"))


# ==========================================================================
# 3. ВЕРДИКТ ПО ЧЕРТЕЖУ
# ==========================================================================
@dataclass
class DesignVerdict:
    """Результат проверки чертежа: маркеры ошибок и метрики."""
    markers: list = field(default_factory=list)
    delta_v: float = 0.0
    twr: float = 0.0
    total_mass: float = 0.0
    dry_mass: float = 0.0
    mass_ratio: float = 0.0
    required_dv: float = 0.0
    overbuild: float = 0.0          # доля превышения ΔV сверх нужного
    layout: Layout = field(default_factory=Layout)
    advice: list = field(default_factory=list)
    penalty: float = 0.0            # суммарный штраф к награде конструктора
    penalties: dict = field(default_factory=dict)   # из чего он сложился

    @property
    def critical(self) -> bool:
        return any(marker in ("CRITICAL_LOW_TWR", "AERO_UNSTABLE",
                              "NO_ENGINE", "NO_COMMAND", "NO_TANK")
                   for marker in self.markers)

    @property
    def clean(self) -> bool:
        return not self.markers

    def describe(self) -> str:
        head = (f"ΔV {self.delta_v:.0f} м/с при нужных {self.required_dv:.0f}, "
                f"TWR {self.twr:.2f}, масса {self.total_mass:.2f} т, "
                f"отношение масс {self.mass_ratio:.2f}, запас устойчивости "
                f"{self.layout.static_margin:+.2f} м")
        if self.clean:
            return "ЧЕРТЁЖ ЧИСТ. " + head
        lines = ["МАРКЕРЫ: " + ", ".join(self.markers), head]
        lines += [f"  → {item}" for item in self.advice]
        if self.penalty:
            breakdown = ", ".join(f"{name} {value:+.2f}"
                                  for name, value in self.penalties.items())
            lines.append(f"  штраф к награде конструктора: {self.penalty:+.2f}"
                         + (f" ({breakdown})" if breakdown else ""))
        return "\n".join(lines)


# ==========================================================================
# 4. ПРОЕКТНОЕ БЮРО
# ==========================================================================
@dataclass
class ShapeGuidance:
    """Требования к форме, выведенные из полётного опыта."""
    max_diameter: float = 2.5
    require_nose_cone: bool = False
    min_fins: int = 0
    fin_level: float = 0.6
    reason: str = "опыта нет — ограничений не выставляю"

    def describe(self) -> str:
        return (f"мидель до {self.max_diameter:.3g} м, "
                f"обтекатель {'обязателен' if self.require_nose_cone else 'по желанию'}, "
                f"оперение ≥ {self.min_fins} шт (уровень {self.fin_level:.2f}). "
                f"{self.reason}")


class DesignerPro:
    """Проектное бюро: проверяет чертежи и учится на полётах пилота."""

    def __init__(self, experience: FlightExperience | None = None,
                 reader: ExperienceReader | None = None,
                 goal_altitude: float = 80_000.0, goal_transfer: bool = False,
                 goal_landing: bool = False, stage: int = MIN_STAGE):
        self.reader = reader or ExperienceReader()
        self.experience = experience if experience is not None else self.reader.read()
        self.goal_altitude = goal_altitude
        self.goal_transfer = goal_transfer
        self.goal_landing = goal_landing
        self.stage = int(stage)

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return is_enabled(self.stage)

    def refresh(self) -> FlightExperience:
        """Перечитывает логи: полёты могли случиться уже после запуска."""
        self.experience = self.reader.read()
        return self.experience

    # ------------------------------------------------------------------
    def required_dv(self) -> float:
        """Сколько ΔV миссии нужно на самом деле."""
        need = DV_TO_ORBIT
        # орбита выше 80 км стоит дороже, ниже — дешевле
        need += max(-400.0, min(900.0, (self.goal_altitude - 80_000.0) / 100.0))
        if self.goal_transfer:
            need += DV_TRANSFER
        if self.goal_landing:
            need += DV_LANDING
        return need

    # ------------------------------------------------------------------
    # 1. Форма корпуса из полётного опыта
    # ------------------------------------------------------------------
    def shape_guidance(self) -> ShapeGuidance:
        """Во что полётный опыт превращается на чертёжной доске."""
        guidance = ShapeGuidance()
        if not self.enabled:
            guidance.reason = (f"ступень {self.stage}: бюро включается "
                               f"с {MIN_STAGE}-й")
            return guidance
        experience = self.experience
        if not experience.has_experience:
            return guidance

        reasons = []
        # Мидель: каждый подтверждённый эпизод потерь на сопротивлении
        # сужает разрешённый диаметр. Лобовое сопротивление растёт как
        # квадрат диаметра, и это первое, чем платят за «толстую» ракету.
        drag = experience.drag_pressure
        if drag > 0.25:
            guidance.max_diameter = 1.25 if drag > 0.6 else 2.5
            reasons.append(f"потери на сопротивлении ({drag:.0%}) — режу мидель")
        if drag > 0.3:
            guidance.require_nose_cone = True
            reasons.append("нужен носовой обтекатель")

        # Оперение: лечит опрокидывание, а не сопротивление
        tumble = experience.tumble_pressure
        if tumble > 0.2:
            guidance.min_fins = 4 if tumble > 0.55 else 2
            guidance.fin_level = min(1.0, 0.6 + 0.4 * tumble)
            reasons.append(f"опрокидывания ({tumble:.0%}) — ставлю "
                           f"{guidance.min_fins} стабилизатора")
        guidance.reason = "; ".join(reasons) or "опыт есть, правок формы не требует"
        return guidance

    def apply_to_policy(self, policy) -> dict:
        """Переносит опыт в политику алгоритмического конструктора.

        Возвращает применённые правки — их удобно печатать в отчёт.
        """
        applied: dict[str, float] = {}
        if not self.enabled:
            return applied
        guidance = self.shape_guidance()
        experience = self.experience

        if guidance.min_fins and hasattr(policy, "fin_level"):
            if policy.fin_level < guidance.fin_level:
                policy.fin_level = guidance.fin_level
                applied["fin_level"] = round(guidance.fin_level, 3)
        if experience.thrust_pressure > 0.3 and hasattr(policy, "liftoff_twr"):
            raised = min(2.05, policy.liftoff_twr + 0.20 * experience.thrust_pressure)
            if raised > policy.liftoff_twr:
                policy.liftoff_twr = raised
                applied["liftoff_twr"] = round(raised, 3)
        if experience.mass_pressure > 0.3 and hasattr(policy, "dv_margin"):
            raised = min(1.30, policy.dv_margin + 0.05 * experience.mass_pressure)
            if raised > policy.dv_margin:
                policy.dv_margin = raised
                applied["dv_margin"] = round(raised, 3)
        # Овер-инжиниринг лечится не запасом, а штрафом за массу
        if experience.drag_pressure > 0.4 and hasattr(policy, "mass_penalty"):
            raised = min(2.0, policy.mass_penalty + 0.3 * experience.drag_pressure)
            if raised > policy.mass_penalty:
                policy.mass_penalty = raised
                applied["mass_penalty"] = round(raised, 3)
        if applied:
            log.info("Опыт полётов применён к политике конструктора: %s", applied)
        return applied

    # ------------------------------------------------------------------
    # 2-4. Проверка чертежа
    # ------------------------------------------------------------------
    def verify(self, parts, densities: dict | None = None,
               delta_v: float | None = None, twr: float | None = None
               ) -> DesignVerdict:
        """Многокритериальный чек-лист. Возвращает маркеры ошибок."""
        parts = list(parts or [])
        densities = densities or {}
        verdict = DesignVerdict()
        verdict.required_dv = self.required_dv()

        engines = [p for p in parts if getattr(p, "is_engine", False)]
        tanks = [p for p in parts if getattr(p, "is_tank", False)]
        if not any(getattr(p, "is_command", False) for p in parts):
            verdict.markers.append("NO_COMMAND")
            verdict.advice.append("без командного модуля аппарат неуправляем")
        if not engines:
            verdict.markers.append("NO_ENGINE")
            verdict.advice.append("нет двигателя — лететь нечем")
        if not tanks:
            verdict.markers.append("NO_TANK")
            verdict.advice.append("нет бака — двигателю нечего жечь")

        total = sum(_wet(p, densities) for p in parts)
        dry = sum(float(getattr(p, "dry_mass", 0.0) or 0.0) for p in parts)
        thrust = sum(float(getattr(p, "max_thrust", 0.0) or 0.0) for p in engines)
        verdict.total_mass = total
        verdict.dry_mass = dry
        verdict.mass_ratio = (total / dry) if dry > 0 else 0.0

        isp = 0.0
        if thrust > 0:
            isp = sum(float(getattr(e, "isp_vac", 0.0) or 0.0)
                      * float(getattr(e, "max_thrust", 0.0) or 0.0)
                      for e in engines) / thrust
        if delta_v is None:
            delta_v = (isp * G0 * math.log(total / dry)
                       if isp > 0 and dry > 0 and total > dry else 0.0)
        verdict.delta_v = delta_v
        if twr is None:
            twr = (thrust / (total * 9.81)) if total > 0 else 0.0
        verdict.twr = twr

        verdict.layout = build_layout(parts, densities)
        if not verdict.markers:
            self._check_thrust(verdict)
            self._check_stability(verdict)
            self._check_overbuild(verdict)
            self._check_tsiolkovsky(verdict)
            self._check_isp_hierarchy(verdict, engines, parts)
            self._check_shape(verdict, parts)
        verdict.penalty = self._penalty(verdict)
        return verdict

    # ------------------------------------------------------------------
    def _check_thrust(self, verdict: DesignVerdict) -> None:
        """TWR: и мало, и много — одинаково плохо."""
        if verdict.twr < TWR_CRITICAL_LOW:
            verdict.markers.append("CRITICAL_LOW_TWR")
            verdict.advice.append(
                f"TWR {verdict.twr:.2f} < {TWR_CRITICAL_LOW}: аппарат не "
                f"оторвётся. " + self.booster_template(verdict))
        elif verdict.twr > TWR_OVER_THRUST:
            verdict.markers.append("OVER_THRUST")
            verdict.advice.append(
                f"TWR {verdict.twr:.2f} > {TWR_OVER_THRUST}: разгон в плотных "
                f"слоях, аппарат встретит максимум напора на большой скорости "
                f"и сгорит или развалится. Ставьте двигатель слабее либо "
                f"добавьте топлива в ту же ступень")

    def _check_stability(self, verdict: DesignVerdict) -> None:
        """Центр масс должен быть ВПЕРЕДИ центра давления."""
        layout = verdict.layout
        if layout.length <= 0.0:
            return
        if not layout.stable:
            verdict.markers.append("AERO_UNSTABLE")
            verdict.advice.append(
                f"центр масс ({layout.com:.2f} м от носа) позади центра "
                f"давления ({layout.cop:.2f} м): запас устойчивости "
                f"{layout.static_margin:+.2f} м — гарантированное "
                f"опрокидывание. Ставьте оперение в самый низ или "
                f"переносите массу выше")

    def _check_overbuild(self, verdict: DesignVerdict) -> None:
        """Разумная достаточность: лишнее топливо — это лишняя сухая масса."""
        need = verdict.required_dv
        if need <= 0 or verdict.delta_v <= 0:
            return
        verdict.overbuild = verdict.delta_v / need - 1.0
        if verdict.overbuild > OVERBUILD_HARD:
            verdict.markers.append("FUEL_OVERBUILT")
            verdict.advice.append(
                f"ΔV {verdict.delta_v:.0f} м/с при нужных {need:.0f} — запас "
                f"{verdict.overbuild:.0%} против разрешённых "
                f"{OVERBUILD_HARD:.0%}. Это не надёжность, а лишняя сухая "
                f"масса: облегчайте ступени, а не заливайте топливо")

    def _check_tsiolkovsky(self, verdict: DesignVerdict) -> None:
        """Формула Циолковского: ΔV = Isp·g₀·ln(m_полн/m_сух).

        Логарифм беспощаден: если сухая масса велика, никакое топливо
        уже не спасает. Отношение масс ниже 1.8 означает, что конструкция
        везёт сама себя.
        """
        if verdict.mass_ratio <= 0:
            return
        if verdict.mass_ratio < MIN_MASS_RATIO:
            verdict.markers.append("POOR_MASS_RATIO")
            verdict.advice.append(
                f"отношение масс {verdict.mass_ratio:.2f} < {MIN_MASS_RATIO}: "
                f"сухая масса {verdict.dry_mass:.2f} т съедает импульс. "
                f"По Циолковскому ΔV растёт только логарифмом — снимайте "
                f"лишние детали, а не добавляйте бак")

    def _check_isp_hierarchy(self, verdict: DesignVerdict, engines,
                             parts) -> None:
        """Иерархия удельных импульсов: кому куда.

        Атмосферный тяжеловес нужен внизу — он даёт тягу там, где давление
        душит сопло. Вакуумный двигатель наверху — у него длинное сопло и
        высокий Isp в пустоте, но у земли он почти бесполезен.
        """
        if len(engines) < 2:
            return
        order = [p for p in parts if getattr(p, "is_engine", False)]
        top, bottom = order[0], order[-1]
        if _is_vacuum_engine(bottom):
            verdict.markers.append("VACUUM_ENGINE_LOW")
            verdict.advice.append(
                f"внизу стоит вакуумный двигатель «{getattr(bottom, 'name', '?')}» "
                f"(Isp {getattr(bottom, 'isp_asl', 0):.0f} у земли против "
                f"{getattr(bottom, 'isp_vac', 0):.0f} в вакууме): у стола он "
                f"теряет больше половины импульса. Вниз — атмосферный "
                f"тяжеловес, вакуумный наверх")
        if _is_atmospheric_heavy(top):
            verdict.markers.append("ATMO_ENGINE_HIGH")
            verdict.advice.append(
                f"наверху стоит тяжёлый атмосферный двигатель "
                f"«{getattr(top, 'name', '?')}» ({getattr(top, 'max_thrust', 0):.0f} кН): "
                f"в вакууме его тяга избыточна, а масса вычитается из "
                f"полезной нагрузки на всём протяжении полёта")

    def _check_shape(self, verdict: DesignVerdict, parts) -> None:
        """Требования формы, выведенные из полётов пилота."""
        guidance = self.shape_guidance()
        layout = verdict.layout
        if guidance.max_diameter and layout.max_diameter > guidance.max_diameter + 1e-6:
            verdict.markers.append("MIDSECTION_TOO_WIDE")
            verdict.advice.append(
                f"мидель {layout.max_diameter:.3g} м при разрешённых "
                f"{guidance.max_diameter:.3g}: пилот уже терял скорость на "
                f"сопротивлении, а оно растёт как квадрат диаметра")
        if guidance.require_nose_cone and not layout.has_nose_cone:
            verdict.markers.append("NO_NOSE_CONE")
            verdict.advice.append(
                "нет носового обтекателя, хотя потери на сопротивлении уже "
                "подтверждены полётами")
        if guidance.min_fins:
            fins = sum(1 for p in parts if _is_fin(p))
            if fins < guidance.min_fins:
                verdict.markers.append("NO_FINS")
                verdict.advice.append(
                    f"стабилизаторов {fins} при нужных {guidance.min_fins}: "
                    f"опрокидывание в атмосфере уже случалось")

    # ------------------------------------------------------------------
    def booster_template(self, verdict: DesignVerdict) -> str:
        """Шаблон сборки: симметричные радиальные ускорители.

        Классический ответ на нехватку стартового TWR: не менять
        центральный блок, а навесить чётное число одинаковых ускорителей
        через разделители. Симметрия обязательна — иначе тяга уводит
        аппарат с курса.
        """
        if verdict.twr <= 0 or verdict.total_mass <= 0:
            return ("Добавьте симметричные радиальные ускорители через "
                    "разделители.")
        need_thrust = TWR_CRITICAL_LOW * verdict.total_mass * 9.81
        have_thrust = verdict.twr * verdict.total_mass * 9.81
        missing = max(0.0, need_thrust - have_thrust)
        if missing <= 0:
            return ""
        # прикидываем по типовому твердотопливному ускорителю (~250 кН)
        pairs = max(1, math.ceil(missing / (2 * 250.0)))
        return (f"Не хватает {missing:.0f} кН: навесьте {2 * pairs} "
                f"симметричных ускорителя через разделители "
                f"({pairs} пар{'а' if pairs == 1 else ''}) — центральный "
                f"блок при этом менять не нужно.")

    # ------------------------------------------------------------------
    def _penalty(self, verdict: DesignVerdict) -> float:
        """Во что маркеры обходятся конструктору в награде."""
        verdict.penalties = {}
        for marker in verdict.markers:
            cost = MARKER_PENALTY.get(marker, 0.2)
            if cost:
                verdict.penalties[marker] = -cost
        # Штраф за избыток топлива — не ступенька, а наклон: чем дальше
        # за разрешённый запас, тем дороже. Так у сети есть градиент, по
        # которому можно спускаться, а не обрыв, который она обойдёт.
        if verdict.overbuild > OVERBUILD_FREE:
            excess = verdict.overbuild - OVERBUILD_FREE
            verdict.penalties["избыток ΔV"] = -min(2.0, 3.0 * excess)
        return sum(verdict.penalties.values())

    def reward(self, base_reward: float, verdict: DesignVerdict) -> float:
        """Награда конструктора с учётом чек-листа и разумной достаточности."""
        if not self.enabled:
            return base_reward
        return base_reward + verdict.penalty

    # ------------------------------------------------------------------
    def report(self, verdict: DesignVerdict) -> str:
        """Полный отчёт бюро — для журнала и для оператора."""
        lines = [
            "=" * 68,
            f"  ПРОЕКТНОЕ БЮРО KIA (ступень {self.stage}, "
            f"{'включено' if self.enabled else 'выключено'})",
            "=" * 68,
            self.experience.describe(),
            "",
            "Требования к форме: " + self.shape_guidance().describe(),
            "",
            verdict.describe(),
        ]
        return "\n".join(lines)


# Насколько дорого обходится каждый маркер. Критические ошибки стоят
# столько, чтобы никакая экономия массы их не окупала.
MARKER_PENALTY = {
    "NO_COMMAND": 1.0,
    "NO_ENGINE": 1.0,
    "NO_TANK": 1.0,
    "CRITICAL_LOW_TWR": 1.2,
    "OVER_THRUST": 0.6,
    "AERO_UNSTABLE": 1.0,
    "POOR_MASS_RATIO": 0.5,
    "VACUUM_ENGINE_LOW": 0.4,
    "ATMO_ENGINE_HIGH": 0.3,
    "MIDSECTION_TOO_WIDE": 0.4,
    "NO_NOSE_CONE": 0.3,
    "NO_FINS": 0.5,
    "FUEL_OVERBUILT": 0.0,      # считается отдельно, наклоном
}


def _wet(part, densities: dict) -> float:
    try:
        return float(part.wet_mass(densities))
    except (AttributeError, TypeError):
        return float(getattr(part, "dry_mass", 0.0) or 0.0)


def _is_vacuum_engine(part) -> bool:
    vac = float(getattr(part, "isp_vac", 0.0) or 0.0)
    asl = float(getattr(part, "isp_asl", 0.0) or 0.0)
    if vac <= 0:
        return False
    return (vac - asl) / vac > VACUUM_ENGINE_LOSS


def _is_atmospheric_heavy(part) -> bool:
    if _is_vacuum_engine(part):
        return False
    return float(getattr(part, "max_thrust", 0.0) or 0.0) >= ATMO_ENGINE_THRUST


# ==========================================================================
def build_designer(stage: int, goal_altitude: float = 80_000.0,
                   goal_transfer: bool = False, goal_landing: bool = False):
    """Бюро для указанной ступени или None, если оно там не работает."""
    if not is_enabled(stage):
        return None
    return DesignerPro(goal_altitude=goal_altitude, goal_transfer=goal_transfer,
                       goal_landing=goal_landing, stage=stage)


def main(argv: list[str] | None = None) -> int:
    """Разбор опыта и проверка текущего чертежа из командной строки."""
    from kia_core.engineer.parts_catalog import get_catalog
    from kia_core.neural.builder_env import build_palette

    argv = list(sys.argv[1:] if argv is None else argv)
    stage = int(argv[0]) if argv and argv[0].isdigit() else MIN_STAGE
    designer = DesignerPro(stage=stage)
    print(designer.experience.describe())
    print()
    print("Требования к форме: " + designer.shape_guidance().describe())
    print()
    try:
        catalog = get_catalog()
        palette = build_palette(catalog)
    except Exception as exc:                       # каталога игры может не быть
        print(f"Каталог деталей недоступен ({exc}) — проверять нечего.")
        return 0
    sample = [p for p in palette.parts if p.is_command][:1]
    sample += [p for p in palette.parts if p.is_tank][:1]
    sample += [p for p in palette.parts if p.is_engine][:1]
    verdict = designer.verify(sample, catalog.densities)
    print(designer.report(verdict))
    return 0


__all__ = ["DesignerPro", "DesignVerdict", "FlightExperience",
           "ExperienceReader", "ShapeGuidance", "Layout", "build_layout",
           "build_designer", "is_enabled", "MIN_STAGE", "MARKER_PENALTY"]


if __name__ == "__main__":
    raise SystemExit(main())
