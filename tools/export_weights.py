r"""Выгрузить обученные веса в библиотеку ASTRAUTOMA (по модулям).

    py -3 tools/export_weights.py                    в Desktop\ASTRAUTOMA Weights
    py -3 tools/export_weights.py --label "Мои"      своё имя версии
    py -3 tools/export_weights.py --library D:\W     другая папка

Каждая сеть — своя папка: FlightProfile, Pilot, Builder, Math, Navigator,
Knowledge. В ASTRAUTOMA: «Веса / Знания» → выбрать → установить.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from astra_weights.knowledge import export_from_training  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="KIA")
    ap.add_argument("--author", default="")
    ap.add_argument("--library", default=str(Path.home() / "Desktop" / "ASTRAUTOMA Weights"))
    args = ap.parse_args()
    for folder in export_from_training(HERE.parent, Path(args.library), args.label, args.author):
        print(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
