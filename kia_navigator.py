"""Межпланетная программа навигатора — запуск из консоли.

    py -3 kia_navigator.py train            учить (продолжает с сохранённого)
    py -3 kia_navigator.py train -w 10      число процессов-тренажёров
    py -3 kia_navigator.py exam             отчёт по всем планетам и целям
    py -3 kia_navigator.py plan Duna land 4500   план миссии (Δv 4500 м/с)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("-w", "--workers", type=int, default=10)
    t.add_argument("-b", "--batch", type=int, default=512)
    t.add_argument("-i", "--iterations", type=int, default=4000)
    sub.add_parser("exam")
    p = sub.add_parser("plan")
    p.add_argument("target")
    p.add_argument("objective", choices=("land", "low", "high"))
    p.add_argument("dv", type=float)
    args = ap.parse_args()

    from kia_core.neural import navigator as nav
    if args.cmd == "train":
        trainer = nav.NavigatorTrainer(workers=args.workers, batch=args.batch)
        try:
            report = trainer.train(max_iterations=args.iterations)
        except KeyboardInterrupt:
            report = trainer.report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.cmd == "exam":
        print(nav.REPORT_FILE.read_text(encoding="utf-8") if nav.REPORT_FILE.exists()
              else "Навигатор ещё не обучался")
    elif args.cmd == "plan":
        print(json.dumps(nav.plan_mission(args.target, args.objective, args.dv),
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
