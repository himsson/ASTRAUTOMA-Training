"""Тренажёр практики: сквозные миссии ко всем планетам на разных ракетах.

    py -3 kia_practice.py train [итераций]   дообучить навигатора практикой
    py -3 kia_practice.py show               отчёт
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from kia_core.neural import practice as P
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "train":
        report = P.train(rounds=int(sys.argv[2]) if len(sys.argv) > 2 else 24)
        print(json.dumps({k: report[k]["overall"] for k in ("before", "after", "expert_ceiling")}, indent=2))
    elif P.REPORT.exists():
        print(P.REPORT.read_text(encoding="utf-8"))
    else:
        print("Практики ещё не было")
    return 0


if __name__ == "__main__":
    sys.exit(main())
