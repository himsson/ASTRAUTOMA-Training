"""Глобальная долгосрочная память KIA: kia_global_memory.json.

Файл живёт вне циклов обучения и НИКОГДА не стирается при перезапуске
симуляции, смене цели или создании новой ракеты. В нём фиксируется то,
что система уже доказала на практике:

* НАВЫКИ (Milestone Success) — подтверждённые достижения с точными
  математическими параметрами, при которых они были получены
  («выход на орбиту при profile pitch: 250→41000 м, n=0.62, TWR 1.56»);
* УРОКИ ОТКАЗОВ — области параметров, которые уже приводили к провалу
  (например, стартовый TWR ниже 1.2 → «не оторвался от стола»);
* ЛУЧШИЕ ВЕТКИ — лучший геном для каждой глобальной цели отдельно.

Перед каждым запуском KIA читает память и:
  1. берёт подтверждённые параметры как основу (seed);
  2. сужает границы оптимизатора вокруг доказанных значений, чтобы
     алгоритм не повторял старые ошибки.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from .config import ROOT
from .logging_setup import get_logger

log = get_logger("memory")

MEMORY_FILE = ROOT / "kia_global_memory.json"

# Какие параметры фиксировать при каждом навыке
SKILL_PARAMETERS = {
    "flight": ["turn_start_altitude", "turn_end_altitude", "turn_exponent",
               "ascent_heading", "max_q_throttle", "max_q_threshold",
               "coast_throttle", "circularization_lead", "node_execute_lead",
               "transfer_phase_bias", "capture_lead"],
    "design": ["ascent_split", "liftoff_twr", "upper_twr", "transfer_twr",
               "dv_margin", "mass_penalty", "part_penalty", "fin_level",
               "solar_level", "science_level"],
}

# Навыки, после подтверждения которых профиль выведения считается «незыблемым»
ASCENT_SKILLS = {"liftoff", "supersonic", "atmosphere_exit", "stable_orbit"}

# ЧТО ИМЕННО ДОКАЗЫВАЕТ КАЖДЫЙ НАВЫК.
#
# Раньше карты не было: навык сохранял ВЕСЬ геном полёта, и любой из них
# закреплял все параметры разом. Отсюда вышла тихая, но дорогая ошибка.
#
# Навык «оторвался от стола» подтверждался в каждом полёте — отрыв
# удавался всегда. Вместе с ним закреплялся и показатель кривой тангажа
# 1.30 из того же генома, хотя отрыв от стола не говорит о гравитационном
# развороте ровно ничего. Дальше память притягивала каждого нового
# кандидата к этому «доказанному» значению на половину пути:
#
#     1.018 -> аналитик правит -0.15 -> 0.868 -> память тянет назад -> 1.009
#
# Аналитик ставил верный диагноз, правка применялась в полную силу, и всё
# равно значение не двигалось: память возвращала его обратно. Три источника
# одной и той же беды — best_per_goal, история запусков и вот этот —
# держали систему на профиле, который гарантировал падение.
#
# Теперь навык закрепляет только те параметры, которые он в самом деле
# проверил. Отрыв от стола — про тягу и удержание на столе. Про разворот
# говорит выход из атмосферы, про скругление — устойчивая орбита.
SKILL_PROVES: dict[str, dict[str, tuple[str, ...]]] = {
    "liftoff": {
        "flight": ("turn_start_altitude",),
        "design": ("liftoff_twr", "fin_level"),
    },
    "supersonic": {
        # Сверхзвук проходят при определённом ограничении напора и
        # начале разворота — это они и подтверждают.
        "flight": ("turn_start_altitude", "max_q_throttle", "max_q_threshold"),
        "design": ("liftoff_twr", "fin_level"),
    },
    "atmosphere_exit": {
        # Атмосфера пройдена — значит профиль разворота рабочий.
        "flight": ("turn_start_altitude", "turn_end_altitude", "turn_exponent",
                   "ascent_heading", "max_q_throttle", "max_q_threshold"),
        "design": ("liftoff_twr", "fin_level", "ascent_split", "dv_margin"),
    },
    "stable_orbit": {
        # Орбита замкнута — доказано всё, включая скругление.
        "flight": ("turn_start_altitude", "turn_end_altitude", "turn_exponent",
                   "ascent_heading", "max_q_throttle", "max_q_threshold",
                   "coast_throttle", "circularization_lead", "target_apoapsis"),
        "design": ("liftoff_twr", "upper_twr", "fin_level", "ascent_split",
                   "dv_margin"),
    },
}

# Порядок по силе доказательства: следующий перекрывает предыдущих. Орбита
# доказывает профиль надёжнее, чем отрыв от стола, поэтому идёт последней.
ASCENT_SKILL_ORDER = ("liftoff", "supersonic", "atmosphere_exit", "stable_orbit")


@dataclass
class Skill:
    """Подтверждённый навык с параметрами, при которых он получен."""
    key: str
    title: str
    first_achieved: str
    last_confirmed: str
    confirmations: int = 1
    best_score: float = 0.0
    goal: str = ""
    parameters: dict = field(default_factory=dict)
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


SKILL_TITLES = {
    "liftoff": "Безопасный отрыв от стартового стола",
    "cleared_tower": "Уверенный набор высоты после старта",
    "supersonic": "Проход звукового барьера без разрушения",
    "atmosphere_exit": "Выход за пределы атмосферы",
    "stable_orbit": "Стабильная круговая орбита",
    "mun_node_created": "Точный расчёт окна перелёта",
    "mun_encounter": "Траектория с перехватом цели",
    "mun_soi_entered": "Вход в сферу влияния цели",
    "mun_orbit": "Орбита целевого тела",
    "target_landing": "Мягкая посадка на цель",
    "returned_home": "Возвращение домой",
}


class GlobalMemory:
    """Постоянная база знаний KIA."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or MEMORY_FILE)
        self._lock = threading.RLock()
        self.created = datetime.now().isoformat(timespec="seconds")
        self.updated = self.created
        self.skills: dict[str, Skill] = {}
        self.failure_lessons: dict[str, dict] = {}
        self.best_per_goal: dict[str, dict] = {}
        self.total_runs: int = 0
        self.load()

    # ------------------------------------------------------------------
    # Чтение / запись
    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            log.info("Глобальная память пуста — начинаем с чистого листа")
            self.save()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            backup = self.path.with_suffix(".corrupt.json")
            log.error("Память повреждена (%s) — копия в %s", exc, backup)
            try:
                self.path.replace(backup)
            except OSError:
                pass
            return
        self.created = data.get("created", self.created)
        self.updated = data.get("updated", self.updated)
        self.total_runs = int(data.get("total_runs", 0))
        self.skills = {k: Skill.from_dict(v) for k, v in data.get("skills", {}).items()}
        self.failure_lessons = data.get("failure_lessons", {})
        self.best_per_goal = data.get("best_per_goal", {})
        log.info("Глобальная память: %d навыков, %d уроков, %d веток целей, "
                 "всего запусков %d",
                 len(self.skills), len(self.failure_lessons),
                 len(self.best_per_goal), self.total_runs)

    def save(self) -> None:
        with self._lock:
            self.updated = datetime.now().isoformat(timespec="seconds")
            payload = {
                "_comment": "Долгосрочная память KIA. Не удалять: здесь "
                            "зафиксированы подтверждённые навыки и уроки отказов.",
                "created": self.created,
                "updated": self.updated,
                "total_runs": self.total_runs,
                "skills": {k: s.to_dict() for k, s in self.skills.items()},
                "failure_lessons": self.failure_lessons,
                "best_per_goal": self.best_per_goal,
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Запись успехов
    # ------------------------------------------------------------------
    @staticmethod
    def extract_parameters(individual) -> dict:
        return {
            "flight": {name: round(float(getattr(individual.flight, name)), 4)
                       for name in SKILL_PARAMETERS["flight"]
                       if hasattr(individual.flight, name)},
            "design": {name: round(float(getattr(individual.design, name)), 4)
                       for name in SKILL_PARAMETERS["design"]
                       if hasattr(individual.design, name)},
        }

    def record_success(self, milestones: list[str], individual, score: float,
                       goal: str, evidence: str = "") -> list[str]:
        """Фиксирует достижения как подтверждённые навыки. Возвращает новые."""
        now = datetime.now().isoformat(timespec="seconds")
        params = self.extract_parameters(individual)
        fresh: list[str] = []
        with self._lock:
            for key in milestones:
                title = SKILL_TITLES.get(key, key)
                existing = self.skills.get(key)
                if existing is None:
                    self.skills[key] = Skill(
                        key=key, title=title, first_achieved=now, last_confirmed=now,
                        confirmations=1, best_score=score, goal=goal,
                        parameters=params, evidence=evidence)
                    fresh.append(key)
                    log.info("НОВЫЙ НАВЫК ЗАФИКСИРОВАН: %s (счёт %.0f)", title, score)
                else:
                    existing.confirmations += 1
                    existing.last_confirmed = now
                    if score > existing.best_score:
                        existing.best_score = score
                        existing.parameters = params
                        existing.evidence = evidence
                        log.info("Навык «%s» подтверждён с лучшим результатом %.0f",
                                 title, score)
            self.total_runs += 1
            self.save()
        return fresh

    def record_failure_lesson(self, cause: str, individual, detail: str = "") -> None:
        """Запоминает область параметров, которая уже приводила к провалу."""
        with self._lock:
            lesson = self.failure_lessons.setdefault(cause, {
                "count": 0, "examples": [], "first_seen":
                    datetime.now().isoformat(timespec="seconds")})
            lesson["count"] += 1
            lesson["last_seen"] = datetime.now().isoformat(timespec="seconds")
            snapshot = {
                "liftoff_twr": round(float(individual.design.liftoff_twr), 3),
                "dv_margin": round(float(individual.design.dv_margin), 3),
                "fin_level": round(float(individual.design.fin_level), 3),
                "turn_start_altitude": round(float(individual.flight.turn_start_altitude), 1),
                "turn_exponent": round(float(individual.flight.turn_exponent), 3),
                "detail": detail[:120],
            }
            lesson["examples"] = (lesson["examples"] + [snapshot])[-8:]
            self.save()

    # Ниже этого счёта геном не считается достижением и в память не идёт.
    #
    # Раньше порога не было вовсе, и это заперло обучение намертво. Все
    # десять первых полётов провалились, лучшим из них оказался счёт
    # −196.5 — и он лёг в память как «лучший геном ветки Kerbin/80km».
    # Дальше каждый запуск начинался с него, а в нём turn_exponent = 1.30,
    # то есть профиль, который держит нос вертикально до десяти километров
    # и гарантирует падение. Система десять полётов подряд старательно
    # воспроизводила собственную неудачу, потому что ничего лучше не
    # видела, а память подавала её как знание.
    #
    # Наименее плохой провал — это не опыт. Пока ни одна попытка не дала
    # положительного счёта, правильное поведение — искать дальше, а не
    # закрепляться.
    MIN_SCORE_TO_REMEMBER = 0.0

    def remember_best(self, goal: str, individual, score: float) -> bool:
        """Хранит лучший геном отдельно для каждой глобальной цели (ветки)."""
        with self._lock:
            if score <= self.MIN_SCORE_TO_REMEMBER:
                log.debug("Ветка «%s»: счёт %.0f не выше нуля — геном не "
                          "закрепляю, это провал, а не достижение", goal, score)
                return False
            current = self.best_per_goal.get(goal)
            if current is not None and current.get("score", float("-inf")) >= score:
                return False
            self.best_per_goal[goal] = {
                "score": score,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "genome": individual.to_dict(),
            }
            self.save()
            log.info("Ветка «%s»: новый лучший геном (счёт %.0f)", goal, score)
            return True

    # ------------------------------------------------------------------
    # Применение знаний
    # ------------------------------------------------------------------
    def has_skill(self, key: str) -> bool:
        return key in self.skills

    def branch_seed(self, goal: str):
        """Лучший известный геном для этой цели (начало новой ветки)."""
        from .learning.genome import Individual
        entry = self.best_per_goal.get(goal)
        if not entry:
            return None
        try:
            return Individual.from_dict(entry["genome"])
        except Exception as exc:
            log.debug("Геном ветки «%s» не восстановлен: %s", goal, exc)
            return None

    def _proven(self, section: str) -> dict:
        """Параметры, действительно доказанные полученными навыками.

        Собирается по всем навыкам сразу, а не по одному «самому ценному»:
        отрыв от стола доказывает своё, выход из атмосферы — своё. Каждый
        параметр берётся у САМОГО СИЛЬНОГО навыка, который его проверил
        (порядок ASCENT_SKILL_ORDER), поэтому подтверждённая орбита
        перекрывает подтверждённый отрыв, а не наоборот.
        """
        out: dict[str, float] = {}
        for key in ASCENT_SKILL_ORDER:
            skill = self.skills.get(key)
            if skill is None:
                continue
            allowed = SKILL_PROVES.get(key, {}).get(section, ())
            stored = skill.parameters.get(section, {}) or {}
            for name in allowed:
                if name in stored:
                    out[name] = stored[name]
        return out

    def proven_ascent_profile(self) -> dict:
        """Профиль выведения в той части, которая реально подтверждена."""
        return self._proven("flight")

    def proven_design_policy(self) -> dict:
        return self._proven("design")

    def apply_to(self, individual, strength: float = 1.0) -> dict:
        """Ставит подтверждённые параметры как основу для новой особи."""
        applied: dict[str, float] = {}
        flight = self.proven_ascent_profile()
        design = self.proven_design_policy()
        for name, value in flight.items():
            if hasattr(individual.flight, name):
                current = float(getattr(individual.flight, name))
                blended = current + (value - current) * strength
                setattr(individual.flight, name, blended)
                applied[f"flight.{name}"] = round(blended, 3)
        for name, value in design.items():
            if hasattr(individual.design, name):
                current = float(getattr(individual.design, name))
                blended = current + (value - current) * strength
                setattr(individual.design, name, blended)
                applied[f"design.{name}"] = round(blended, 3)
        individual.flight.clamp()
        individual.design.clamp()
        if applied:
            log.info("Из памяти применено %d подтверждённых параметров", len(applied))
        return applied

    def constrain_bounds(self, bounds: dict, section: str,
                         width: float = 0.35) -> dict:
        """Сужает границы поиска вокруг доказанных значений.

        Это и есть «блокировка старых ошибок»: после подтверждённого навыка
        оптимизатору больше не разрешено уходить далеко от рабочего режима.
        """
        proven = (self.proven_ascent_profile() if section == "flight"
                  else self.proven_design_policy())
        if not proven:
            return bounds
        narrowed = {}
        for name, (lo, hi) in bounds.items():
            value = proven.get(name)
            if value is None:
                narrowed[name] = (lo, hi)
                continue
            span = (hi - lo) * width / 2.0
            new_lo = max(lo, value - span)
            new_hi = min(hi, value + span)
            if new_hi <= new_lo:
                new_lo, new_hi = lo, hi
            narrowed[name] = (new_lo, new_hi)
        return narrowed

    def forbidden_zone(self, individual) -> str | None:
        """Проверяет, не лезет ли особь в область известных отказов.

        ЗАПРЕТ, ПОКРЫВАЮЩИЙ ВЕСЬ ДИАПАЗОН, — НЕ ЗАПРЕТ, А ЛОВУШКА.

        Здесь стояло простое `twr <= worst`, где worst — наибольший TWR
        среди записанных отказов «не оторвался». Пока такие отказы
        случались при малом TWR, правило работало: оно поднимало тягу.

        Но каждое срабатывание поднимало TWR на 0.2, и рано или поздно
        очередной отказ записывался при значении, упёртом в верхнюю
        границу диапазона (2.10). После этого условие «≤ 2.10» стало
        истинным для ЛЮБОГО кандидата: все запуски флагировались, всем
        поднимали тягу, все упирались в потолок.

        Чем это кончилось в игре: 178 м/с на высоте 104 метра при норме
        60-80. Ракета разгонялась вдвое быстрее в самом плотном воздухе,
        и её опрокидывало от любого возмущения — то пронесёт, то нет.
        Геном при этом у успешных и разрушенных запусков был почти
        одинаковый, потому что решал не он.

        Теперь урок должен оставлять место для манёвра: если он покрывает
        весь доступный диапазон, он бесполезен и игнорируется.
        """
        from .engineer.autodesign import DesignPolicy

        lesson = self.failure_lessons.get("no_liftoff")
        if lesson:
            worst = max((e.get("liftoff_twr", 0.0) for e in lesson["examples"]),
                        default=0.0)
            low, high = DesignPolicy.bounds()["liftoff_twr"]
            covers_everything = worst >= high - 1e-9
            if individual.design.liftoff_twr <= worst and not covers_everything:
                return (f"стартовый TWR {individual.design.liftoff_twr:.2f} уже "
                        f"приводил к отказу «не оторвался» (до {worst:.2f})")
            if covers_everything:
                log.debug("Урок «не оторвался» покрывает весь диапазон TWR "
                          "(до %.2f при потолке %.2f) — игнорирую, иначе он "
                          "запирает поиск целиком", worst, high)
        lesson = self.failure_lessons.get("aero_instability")
        if lesson and individual.design.fin_level < 0.3:
            return "полёты без стабилизаторов уже заканчивались потерей устойчивости"
        return None

    # ------------------------------------------------------------------
    def summary(self) -> str:
        if not self.skills and not self.failure_lessons:
            return "Память пуста: подтверждённых навыков пока нет."
        lines = [f"Подтверждённые навыки ({len(self.skills)}):"]
        for skill in sorted(self.skills.values(),
                            key=lambda s: -s.best_score):
            lines.append(f"  ✔ {skill.title} — подтверждён {skill.confirmations}× "
                         f"(лучший счёт {skill.best_score:.0f}, "
                         f"впервые {skill.first_achieved[:16]})")
        if self.failure_lessons:
            lines.append(f"Уроки отказов ({len(self.failure_lessons)}):")
            for cause, lesson in sorted(self.failure_lessons.items(),
                                        key=lambda kv: -kv[1]["count"]):
                lines.append(f"  ✖ {cause} — {lesson['count']}×")
        if self.best_per_goal:
            lines.append("Лучшие ветки по целям:")
            for goal, entry in self.best_per_goal.items():
                lines.append(f"  ▸ {goal}: {entry['score']:.0f} очков")
        profile = self.proven_ascent_profile()
        if profile:
            lines.append("Незыблемый профиль выведения:")
            lines.append("  " + ", ".join(f"{k}={v:g}" for k, v in profile.items()))
        lines.append(f"Всего запусков в памяти: {self.total_runs}")
        return "\n".join(lines)


# Единая память процесса
GLOBAL_MEMORY = GlobalMemory()
