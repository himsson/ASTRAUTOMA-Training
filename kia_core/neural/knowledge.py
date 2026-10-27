"""ЗНАНИЯ, ДОБЫТЫЕ В ТРЕНАЖЁРАХ: kia_sim_knowledge.json.

Долгосрочная память `kia_global_memory.json` наполняется только живыми
полётами в KSP — это правильно, там фиксируются навыки алгоритмической
части, подтверждённые настоящей игрой. Но у нейросети своя жизнь: она
может неделями учиться в тренажёрах, и всё это время оператор не видит
НИЧЕГО, кроме средней награды. Что именно сеть уже умеет, а что ещё нет —
неизвестно.

Этот модуль закрывает дыру. Он смотрит на исходы эпизодов и превращает
их в понятные утверждения:

    оторвался от стола           подтверждено 4 812 раз
    прошёл звуковой барьер       подтверждено 3 990 раз
    вышел за атмосферу           подтверждено   611 раз
    вышел на орбиту              подтверждено   163 раза

Это НЕ подсказки агенту: сеть про этот файл ничего не знает и наград за
него не получает. Это приборная панель для человека — и источник данных
для карты мозга (`kia_brain_map.py`).

Знания копятся отдельно по каждой ступени: «умеет выходить на орбиту в
базовом тренажёре» и «умеет то же самое в Тренажёре++» — разные
утверждения, и вторая гораздо дороже первой.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

from ..config import ROOT
from ..logging_setup import get_logger

log = get_logger("neural.knowledge")

KNOWLEDGE_FILE = ROOT / "kia_sim_knowledge.json"

# Что именно считается навыком и как это увидеть в исходе эпизода.
# Пороги — описание мира, а не цель обучения: сеть их не видит.
SKILLS: tuple[tuple[str, str, str], ...] = (
    ("liftoff", "Оторвался от стартового стола",
     "поднялся выше 100 м — двигатель работает и тяги хватает"),
    ("cleared_tower", "Уверенно набрал первый километр",
     "выше 1 км: аппарат управляем и не заваливается сразу"),
    ("supersonic", "Прошёл звуковой барьер",
     "выше 10 км: пережил максимум скоростного напора"),
    ("high_atmosphere", "Добрался до верхней атмосферы",
     "выше 40 км: разворот выполнен, набор идёт по траектории"),
    ("space", "Вышел за пределы атмосферы",
     "выше 70 км — граница космоса Кербина"),
    ("orbit", "Вышел на устойчивую орбиту",
     "периапсис выше атмосферы: аппарат не упадёт обратно"),
    ("clean_flight", "Долетел, ни во что не врезавшись",
     "эпизод закончился без разрушения аппарата"),
    ("smooth_steering", "Провёл полёт без грубого скольжения",
     "нос держался по потоку: ни одного нарушения угла атаки"),
    ("power_kept", "Не посадил батарею",
     "заряд ни разу не падал до нуля — рулил плавно"),
)

SKILL_TITLES = {key: title for key, title, _ in SKILLS}
SKILL_HINTS = {key: hint for key, _, hint in SKILLS}

ATMOSPHERE_TOP = 70_000.0


@dataclass
class SkillRecord:
    """Один навык на одной ступени."""
    key: str
    confirmations: int = 0
    attempts: int = 0
    first_seen: str = ""
    last_seen: str = ""
    best_value: float = 0.0

    @property
    def rate(self) -> float:
        return self.confirmations / self.attempts if self.attempts else 0.0

    def to_dict(self) -> dict:
        return {"key": self.key, "confirmations": self.confirmations,
                "attempts": self.attempts, "first_seen": self.first_seen,
                "last_seen": self.last_seen,
                "best_value": round(self.best_value, 1),
                "rate": round(self.rate, 4)}

    @classmethod
    def from_dict(cls, data: dict) -> "SkillRecord":
        record = cls(key=str(data.get("key", "")))
        record.confirmations = int(data.get("confirmations", 0))
        record.attempts = int(data.get("attempts", 0))
        record.first_seen = str(data.get("first_seen", ""))
        record.last_seen = str(data.get("last_seen", ""))
        record.best_value = float(data.get("best_value", 0.0))
        return record


@dataclass
class StageKnowledge:
    """Всё, что сеть показала на одной ступени."""
    stage: int
    title: str = ""
    episodes: int = 0
    steps: int = 0
    best_return: float = float("-inf")
    best_altitude: float = 0.0
    skills: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)

    def record(self, key: str, confirmed: bool, value: float = 0.0) -> None:
        skill = self.skills.get(key)
        if skill is None:
            skill = SkillRecord(key=key)
            self.skills[key] = skill
        skill.attempts += 1
        if confirmed:
            now = time.strftime("%Y-%m-%d %H:%M")
            skill.confirmations += 1
            skill.last_seen = now
            if not skill.first_seen:
                skill.first_seen = now
            skill.best_value = max(skill.best_value, value)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage, "title": self.title,
            "episodes": self.episodes, "steps": self.steps,
            "best_return": (round(self.best_return, 2)
                            if self.best_return > float("-inf") else None),
            "best_altitude": round(self.best_altitude),
            "skills": {key: value.to_dict() for key, value in self.skills.items()},
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StageKnowledge":
        entry = cls(stage=int(data.get("stage", 1)))
        entry.title = str(data.get("title", ""))
        entry.episodes = int(data.get("episodes", 0))
        entry.steps = int(data.get("steps", 0))
        best = data.get("best_return")
        entry.best_return = float(best) if best is not None else float("-inf")
        entry.best_altitude = float(data.get("best_altitude", 0.0))
        entry.skills = {key: SkillRecord.from_dict(value)
                        for key, value in (data.get("skills") or {}).items()}
        entry.failures = dict(data.get("failures") or {})
        return entry


class SimKnowledge:
    """Копилка знаний тренажёров. Пишется на диск, читается картой мозга."""

    def __init__(self, path=None):
        from pathlib import Path
        self.path = Path(path or KNOWLEDGE_FILE)
        self._lock = threading.RLock()
        self.stages: dict[int, StageKnowledge] = {}
        self.updated = ""
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.updated = str(data.get("updated", ""))
        for raw in (data.get("stages") or {}).values():
            entry = StageKnowledge.from_dict(raw)
            self.stages[entry.stage] = entry

    def save(self) -> None:
        with self._lock:
            self.updated = time.strftime("%Y-%m-%d %H:%M:%S")
            payload = {
                "_comment": "Что нейросеть умеет по итогам тренажёров. "
                            "Файл только для человека и карты мозга: сеть "
                            "его не видит и наград за него не получает.",
                "updated": self.updated,
                "stages": {str(number): entry.to_dict()
                           for number, entry in sorted(self.stages.items())},
            }
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False,
                                            indent=2), encoding="utf-8")
            temporary.replace(self.path)

    # ------------------------------------------------------------------
    def stage(self, number: int, title: str = "") -> StageKnowledge:
        entry = self.stages.get(number)
        if entry is None:
            entry = StageKnowledge(stage=number, title=title)
            self.stages[number] = entry
        elif title and not entry.title:
            entry.title = title
        return entry

    # ------------------------------------------------------------------
    def observe_episode(self, stage_number: int, info: dict,
                        episode_return: float, summary: dict | None = None,
                        title: str = "") -> None:
        """Превращает исход одного эпизода в подтверждённые навыки.

        Вызывается тренером после каждого эпизода любой копии среды.
        Работа дешёвая — арифметика по словарю, никаких обращений к диску.
        """
        with self._lock:
            entry = self.stage(stage_number, title)
            entry.episodes += 1
            entry.best_return = max(entry.best_return, float(episode_return))

            summary = summary or {}
            peak = float(summary.get("peak_altitude",
                                     info.get("altitude", 0.0)) or 0.0)
            entry.best_altitude = max(entry.best_altitude, peak)
            orbit = bool(info.get("orbit"))
            reason = str(info.get("reason", "—"))
            if not orbit:
                entry.failures[reason] = entry.failures.get(reason, 0) + 1

            entry.record("liftoff", peak > 100.0, peak)
            entry.record("cleared_tower", peak > 1_000.0, peak)
            entry.record("supersonic", peak > 10_000.0, peak)
            entry.record("high_atmosphere", peak > 40_000.0, peak)
            entry.record("space", peak > ATMOSPHERE_TOP, peak)
            entry.record("orbit", orbit, peak)
            entry.record("clean_flight", not summary.get("crashed", False), peak)
            if "aoa_violations" in summary:
                entry.record("smooth_steering",
                             int(summary.get("aoa_violations", 0)) == 0, peak)
            if "blackouts" in summary:
                entry.record("power_kept",
                             int(summary.get("blackouts", 0)) == 0, peak)

    def add_steps(self, stage_number: int, steps: int) -> None:
        with self._lock:
            self.stage(stage_number).steps += int(steps)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        if not self.stages:
            return ("Тренажёры ещё ничего не показали: обучение не "
                    "запускалось или не закончило ни одного эпизода.")
        lines = []
        for number, entry in sorted(self.stages.items()):
            best = (f"{entry.best_return:+.1f}"
                    if entry.best_return > float("-inf") else "—")
            lines.append(f"Ступень {number} ({entry.title or '?'}): "
                         f"{entry.episodes} эпизодов, потолок "
                         f"{entry.best_altitude:.0f} м, лучшая награда {best}")
            for key, _, _ in SKILLS:
                skill = entry.skills.get(key)
                if skill is None or not skill.attempts:
                    continue
                mark = "✔" if skill.confirmations else "·"
                lines.append(f"   {mark} {SKILL_TITLES[key]:38s} "
                             f"{skill.confirmations:6d} раз "
                             f"({skill.rate:.0%} попыток)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "updated": self.updated,
            "stages": {str(number): entry.to_dict()
                       for number, entry in sorted(self.stages.items())},
        }


# Единая копилка процесса
SIM_KNOWLEDGE = SimKnowledge()

__all__ = ["SimKnowledge", "SIM_KNOWLEDGE", "SKILLS", "SKILL_TITLES",
           "SKILL_HINTS", "KNOWLEDGE_FILE", "StageKnowledge", "SkillRecord"]
