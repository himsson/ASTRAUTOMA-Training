"""Тренажёр исполнения межпланетного перелёта (автопилот).

    py -3 kia_cruise.py train        учить для всех планет (минуты)
    py -3 kia_cruise.py show         что выучено
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from kia_core.neural import cruise_sim as C
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "train":
        C.train(generations=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif C.PROFILE_FILE.exists():
        print(C.PROFILE_FILE.read_text(encoding="utf-8"))
    else:
        print("Автопилот ещё не обучался")
    return 0


if __name__ == "__main__":
    sys.exit(main())
