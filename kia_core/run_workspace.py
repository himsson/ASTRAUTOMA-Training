"""Папки запусков: kia/1, kia/2, kia/3 ...

Каждая итерация обучения получает собственную пронумерованную папку.
В ней ровно два файла этого запуска:

    kia/<N>/kia_design_report.md   — расчёт и обоснование конструкции
    kia/<N>/kia_flight_log.md      — бортовой журнал полёта

Старые папки никогда не перезаписываются: номер всегда следующий за
максимальным существующим.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR, ROOT
from .logging_setup import get_logger

log = get_logger("workspace")

RUNS_DIR = ROOT / "kia"
DESIGN_REPORT_NAME = "kia_design_report.md"
FLIGHT_LOG_NAME = "kia_flight_log.md"

# Машинные данные хранятся отдельно: в пронумерованной папке запуска лежат
# ровно два файла — отчёт конструктора и бортовой журнал.
META_DIR = DATA_DIR / "runs"


@dataclass
class RunFolder:
    number: int
    path: Path

    @property
    def design_report(self) -> Path:
        return self.path / DESIGN_REPORT_NAME

    @property
    def flight_log(self) -> Path:
        return self.path / FLIGHT_LOG_NAME

    @property
    def design_json(self) -> Path:
        return META_DIR / f"{self.number}_design.json"

    @property
    def flight_json(self) -> Path:
        return META_DIR / f"{self.number}_flight.json"

    @property
    def meta_file(self) -> Path:
        return META_DIR / f"{self.number}_run.json"

    def write_meta(self, data: dict) -> None:
        payload = {"run": self.number,
                   "at": datetime.now().isoformat(timespec="seconds")}
        payload.update(data)
        self.meta_file.parent.mkdir(parents=True, exist_ok=True)
        self.meta_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    def read_meta(self) -> dict:
        if not self.meta_file.exists():
            return {}
        try:
            return json.loads(self.meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}


class RunWorkspace:
    """Управляет каталогом kia/ и нумерацией запусков."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or RUNS_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        META_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def existing_numbers(self) -> list[int]:
        numbers = []
        for entry in self.root.iterdir():
            if entry.is_dir() and re.fullmatch(r"\d+", entry.name):
                numbers.append(int(entry.name))
        return sorted(numbers)

    def next_number(self) -> int:
        numbers = self.existing_numbers()
        return (numbers[-1] + 1) if numbers else 1

    def create(self) -> RunFolder:
        """Создаёт следующую пронумерованную папку запуска."""
        number = self.next_number()
        path = self.root / str(number)
        # На случай гонки: ищем первый свободный номер
        while path.exists():
            number += 1
            path = self.root / str(number)
        path.mkdir(parents=True)
        log.info("Папка запуска: %s", path)
        return RunFolder(number=number, path=path)

    def latest(self) -> RunFolder | None:
        numbers = self.existing_numbers()
        if not numbers:
            return None
        return RunFolder(number=numbers[-1], path=self.root / str(numbers[-1]))

    def get(self, number: int) -> RunFolder | None:
        path = self.root / str(number)
        return RunFolder(number=number, path=path) if path.exists() else None

    def all_runs(self) -> list[RunFolder]:
        return [RunFolder(number=n, path=self.root / str(n))
                for n in self.existing_numbers()]

    def summary(self, limit: int = 12) -> str:
        runs = self.all_runs()[-limit:]
        if not runs:
            return "Запусков пока нет — папка kia/ пуста."
        lines = [f"Запусков в kia/: {len(self.existing_numbers())}"]
        for run in runs:
            meta = run.read_meta()
            score = meta.get("score")
            outcome = meta.get("outcome", "")
            lines.append(f"  kia/{run.number}: "
                         f"{'счёт ' + str(round(score)) if score is not None else '—'}"
                         f"{' | ' + outcome if outcome else ''}")
        return "\n".join(lines)


WORKSPACE = RunWorkspace()
