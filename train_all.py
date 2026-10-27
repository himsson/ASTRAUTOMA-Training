r"""Одна кнопка: учится всё.

    py -3 train_all.py              пройти всё, что ещё не пройдено
    py -3 train_all.py --from pilot начать с этого блока
    py -3 train_all.py --status     что уже сделано, без обучения

Четыре блока по очереди, а не разом: каждый грузит процессор целиком, и
вместе они только мешали бы друг другу.

    1. математик    арифметика ракеты -> межпланетное
    2. конструктор  собрать то, что летает -> чертёж проходит валидатор
    3. навигатор    маршрут, окно, гравитационный манёвр
    4. пилот        пять ступеней, последняя — отказы техники в полёте

Ctrl+C останавливает всё. Веса пишутся на диск по ходу, журнал — в
data/train_all.json, поэтому следующий запуск продолжает с того места,
где остановились, а не начинает заново.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Консоль Windows по умолчанию в cp1252 и давится кириллицей. Журнал
# обучения идёт в неё сутками — пусть идёт целым.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

JOURNAL = ROOT / "data" / "train_all.json"
BLOCKS = ("math", "builder", "navigator", "pilot")
TITLES = {"math": "математик", "builder": "конструктор",
          "navigator": "навигатор", "pilot": "пилот"}


def load_journal() -> dict:
    if JOURNAL.exists():
        try:
            return json.loads(JOURNAL.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"done": [], "runs": []}


def save_journal(journal: dict) -> None:
    JOURNAL.parent.mkdir(exist_ok=True)
    JOURNAL.write_text(json.dumps(journal, ensure_ascii=False, indent=2),
                       encoding="utf-8")


def say(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {text}", flush=True)


# ==========================================================================
def run_math_and_builder(which: str) -> dict:
    """Лестницы математика и конструктора: ступень сдаётся по экзамену."""
    from kia_core.neural.trainer import NeuralTrainer
    trainer = NeuralTrainer()
    trainer.begin_session()
    journal = trainer.evolve_programs(programs=(which,))
    trainer.brain.save()
    passed = sum(1 for row in journal if row.get("passed"))
    return {"stages_passed": passed, "attempts": len(journal)}


def run_navigator() -> dict:
    """Навигатор: предобучение по точному планировщику, затем PPO."""
    from kia_core.neural.navigator import NavigatorTrainer
    trainer = NavigatorTrainer(workers=10, batch=512)
    return trainer.train()


def run_pilot() -> dict:
    """Пилот: конвейер ведёт его по ступеням сам, с экзаменом и откатом."""
    from kia_core.neural.pipeline import EvolutionPipeline
    pipeline = EvolutionPipeline()
    return pipeline.run()


RUNNERS = {
    "math": lambda: run_math_and_builder("math"),
    "builder": lambda: run_math_and_builder("builder"),
    "navigator": run_navigator,
    "pilot": run_pilot,
}


# ==========================================================================
def status() -> None:
    journal = load_journal()
    done = set(journal.get("done", []))
    for name in BLOCKS:
        mark = "пройден" if name in done else "ждёт"
        print(f"  {TITLES[name]:12} {mark}")
    for run in journal.get("runs", [])[-6:]:
        print(f"    {run['block']:10} {run['finished']}  {run.get('note', '')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=BLOCKS, default=None,
                    help="начать с этого блока, предыдущие пропустить")
    ap.add_argument("--only", choices=BLOCKS, default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--again", action="store_true",
                    help="пройти заново даже то, что отмечено пройденным")
    args = ap.parse_args()

    if args.status:
        status()
        return 0

    journal = load_journal()
    done = set(journal.get("done", []))
    blocks = list(BLOCKS)
    if args.only:
        blocks = [args.only]
    elif args.start:
        blocks = blocks[blocks.index(args.start):]

    say(f"Обучение: {', '.join(TITLES[b] for b in blocks)}")
    say("Ctrl+C останавливает. Веса пишутся по ходу, прогресс не теряется.")
    for name in blocks:
        if name in done and not args.again and not args.only:
            say(f"{TITLES[name]}: уже пройден, пропускаю")
            continue
        say(f"=== {TITLES[name]} ===")
        started = time.time()
        try:
            result = RUNNERS[name]()
        except KeyboardInterrupt:
            say(f"{TITLES[name]}: остановлено оператором")
            save_journal(journal)
            return 1
        except Exception as exc:
            say(f"{TITLES[name]}: сорвалось — {type(exc).__name__}: {exc}")
            journal.setdefault("runs", []).append({
                "block": name, "finished": time.strftime("%Y-%m-%d %H:%M"),
                "note": f"ошибка: {exc}"})
            save_journal(journal)
            continue
        hours = (time.time() - started) / 3600.0
        say(f"{TITLES[name]}: готово за {hours:.1f} ч — {result}")
        journal.setdefault("runs", []).append({
            "block": name, "finished": time.strftime("%Y-%m-%d %H:%M"),
            "hours": round(hours, 2), "note": str(result)[:200]})
        done.add(name)
        journal["done"] = sorted(done)
        save_journal(journal)

    say("Всё пройдено. Выгрузить веса в приложение: "
        "py -3 tools/export_weights.py --label \"2.0\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
