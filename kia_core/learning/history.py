"""Журнал обучения: JSON-история всех попыток.

Формат файла data/history.json:
{
  "created": ISO-время,
  "runs": [ {run_id, generation, individual, genome, score, milestones, ...}, ... ],
  "stats": {...}
}
Запись атомарная (через .tmp + replace), чтобы прерывание сессии
не рушило историю.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from ..config import CONFIG
from ..logging_setup import get_logger

log = get_logger("learning.history")


@dataclass
class RunRecord:
    run_id: int
    generation: int
    individual: int
    genome: dict
    score: float
    milestones: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    terminated: bool = False
    termination_reason: str | None = None
    design_summary: dict = field(default_factory=dict)
    verdict: dict = field(default_factory=dict)
    telemetry_final: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class HistoryStore:
    """Персистентная история запусков."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or CONFIG.learning.history_file)
        self.data = self._load()

    # ------------------------------------------------------------------
    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                log.info("История загружена: %d запусков", len(data.get("runs", [])))
                return data
            except (json.JSONDecodeError, OSError) as exc:
                backup = self.path.with_suffix(f".corrupt_{int(time.time())}.json")
                log.error("История повреждена (%s), сохраняю копию в %s", exc, backup)
                try:
                    self.path.replace(backup)
                except OSError:
                    pass
        return {"created": datetime.now().isoformat(timespec="seconds"),
                "runs": [], "stats": {}}

    def save(self) -> None:
        self.data["stats"] = self.compute_stats()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.path)
        log.debug("История сохранена (%d запусков)", len(self.runs))

    # ------------------------------------------------------------------
    @property
    def runs(self) -> list[dict]:
        return self.data.setdefault("runs", [])

    def next_run_id(self) -> int:
        return len(self.runs) + 1

    def append(self, record: RunRecord) -> RunRecord:
        self.runs.append(record.to_dict())
        self.save()
        log.info("Запуск #%d записан: счёт %.1f", record.run_id, record.score)
        return record

    # ------------------------------------------------------------------
    def best_run(self) -> dict | None:
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: r.get("score", float("-inf")))

    def best_genome(self) -> dict | None:
        """Геном для затравки поиска — только с УДАЧНОГО запуска.

        Здесь стоял `best_run()`, то есть просто максимум по счёту. Пока
        хоть один полёт удачен, это верно. Но когда провалились ВСЕ,
        максимумом оказывается наименее плохой провал — и он идёт в
        родители каждому следующему кандидату.

        Чем это кончилось на живом прогоне. Девять итераций подряд давали
        ровно −147 очков. Аналитик исправно ставил диагноз «выведение
        слишком крутое» и правил показатель кривой тангажа на −0.15, но
        кандидат каждый раз скрещивался с провальным геномом (1.30), и
        правка ополовинивалась:

            1.300 -> правка -> 1.150, среднее с 1.30 = 1.225
            1.225 -> правка -> 1.075, среднее с 1.30 = 1.188
            1.188 -> правка -> 1.038, среднее с 1.30 = 1.169

        Ряд сходится к 1.15 вместо нужных ~0.55. Система вечно тянулась к
        собственной неудаче и не могла от неё уйти: каждая следующая
        поправка была вдвое слабее предыдущей.

        Провал — не достижение, и в родители он не годится. Пока нет ни
        одного запуска с положительным счётом, затравки нет вовсе, и поиск
        идёт от значений по умолчанию, где заложен рабочий профиль.
        """
        successful = [r for r in self.runs if r.get("score", 0.0) > 0.0]
        if not successful:
            return None
        best = max(successful, key=lambda r: r.get("score", float("-inf")))
        return best.get("genome")

    def last_generation(self) -> int:
        return max((r.get("generation", 0) for r in self.runs), default=-1)

    def generation_runs(self, generation: int) -> list[dict]:
        return [r for r in self.runs if r.get("generation") == generation]

    def compute_stats(self) -> dict:
        if not self.runs:
            return {"count": 0}
        scores = [r.get("score", 0.0) for r in self.runs]
        milestones: dict[str, int] = {}
        failures: dict[str, int] = {}
        for r in self.runs:
            for m in r.get("milestones", []):
                milestones[m] = milestones.get(m, 0) + 1
            for f in r.get("failures", []):
                failures[f] = failures.get(f, 0) + 1
        best = self.best_run() or {}
        return {
            "count": len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "mean_score": sum(scores) / len(scores),
            "last_score": scores[-1],
            "generations": self.last_generation() + 1,
            "best_run_id": best.get("run_id"),
            "milestone_counts": milestones,
            "failure_counts": failures,
        }

    def learning_curve(self) -> list[tuple[int, float]]:
        """(generation, лучший счёт поколения) — для графиков/анализа."""
        by_gen: dict[int, float] = {}
        for r in self.runs:
            g = r.get("generation", 0)
            by_gen[g] = max(by_gen.get(g, float("-inf")), r.get("score", 0.0))
        return sorted(by_gen.items())

    def save_best_genome(self, path: str | Path | None = None) -> Path | None:
        genome = self.best_genome()
        if genome is None:
            return None
        target = Path(path or CONFIG.learning.best_genome_file)
        best = self.best_run() or {}
        target.write_text(json.dumps(
            {"genome": genome, "score": best.get("score"),
             "run_id": best.get("run_id"),
             "milestones": best.get("milestones", []),
             "saved_at": datetime.now().isoformat(timespec="seconds")},
            ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Лучший геном сохранён: %s (счёт %.1f)", target, best.get("score", 0.0))
        return target

    def report(self) -> str:
        stats = self.compute_stats()
        if not stats.get("count"):
            return "История пуста — обучение ещё не начиналось."
        lines = [
            f"Запусков: {stats['count']} | поколений: {stats['generations']}",
            f"Лучший счёт: {stats['best_score']:.1f} (запуск #{stats['best_run_id']})",
            f"Средний счёт: {stats['mean_score']:.1f} | последний: {stats['last_score']:.1f}",
            "Достижения: " + (", ".join(f"{k}×{v}" for k, v in
                                        sorted(stats["milestone_counts"].items(),
                                               key=lambda kv: -kv[1])) or "—"),
            "Отказы: " + (", ".join(f"{k}×{v}" for k, v in
                                    sorted(stats["failure_counts"].items(),
                                           key=lambda kv: -kv[1])) or "—"),
        ]
        return "\n".join(lines)
