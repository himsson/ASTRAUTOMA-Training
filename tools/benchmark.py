r"""Стенд замеров: один и тот же экзамен для каждой версии весов.

    py -3 tools/benchmark.py                      прогнать все версии библиотеки
    py -3 tools/benchmark.py --only "v1.3,v2.0"   только эти
    py -3 tools/benchmark.py --table              markdown из готового json
    py -3 tools/benchmark.py --quick              короткий прогон для проверки

Результат — data/benchmark.json: по строке на версию, по колонке на
навык. Каждый замер детерминирован (фиксированные seed'ы и выключенный
шум политики), так что числа воспроизводятся и их можно сравнивать между
версиями. Таблицу для README рендерит --table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LIBRARY = Path.home() / "Desktop" / "111" / "ASTRAUTOMA Weights"
ASTRA = Path.home() / "Desktop" / "111" / "ASTRAUTOMA"

# Подпись в таблице -> папка версии в библиотеке. Сети в релизах 1.1, 1.2 и
# 1.3 совпадают байт в байт (один kia_model_best.pth), различалась только
# школа проектирования, поэтому все три идут одной строкой «1.x» и меряются
# по самому свежему набору.
VERSIONS = {"1.x": "ASTRAUTOMA v1.3"}
PENDING = ("2.0",)
OUT_FILE = ROOT / "data" / "benchmark.json"
MODULES = ("Pilot", "Builder", "Math", "Navigator")


def bench_designer(folder: Path | None, astra: Path) -> dict:
    """Проектировщик: сколько проектов приложение доводит до чертежа.

    Считается отдельным процессом в папке ASTRAUTOMA: там свой kia_core,
    новее здешнего, и смешивать их в одном интерпретаторе нельзя.
    """
    import subprocess
    script = astra / "tools" / "bench_designer.py"
    if folder is None or not script.exists():
        return {}
    school = folder / "design_school.json"
    out = subprocess.run(
        [sys.executable, str(script), "--json",
         "--school", str(school) if school.exists() else ""],
        cwd=str(astra), capture_output=True, text=True, encoding="utf-8")
    line = next((l for l in reversed(out.stdout.splitlines())
                 if l.startswith("{")), "")
    if not line:
        return {"error": (out.stderr or out.stdout)[-200:]}
    return json.loads(line)


def discover(library: Path) -> dict[str, dict[str, Path]]:
    """Версии, у которых есть хотя бы один модуль: {версия: {модуль: папка}}."""
    found: dict[str, dict[str, Path]] = {}
    for module in MODULES:
        for folder in sorted((library / module).glob("*")):
            if folder.is_dir():
                found.setdefault(folder.name, {})[module] = folder
    return found


# ==========================================================================
def bench_math(brain, folder: Path | None, tasks: int) -> dict:
    """Математик: доля задач, решённых в допуске, на каждой из 4 ступеней.

    Экзамен один и тот же для всех версий — включая ступени, до которых
    версия не доучилась. Низкое число там не ошибка замера, а ответ.
    """
    import torch
    from kia_core.neural.trainer import NeuralTrainer
    if folder is None:
        return {}
    payload = torch.load(folder / "mathematician.pt", map_location="cpu",
                         weights_only=True)
    brain.mathematician.load_state_dict(payload["state"])
    trainer = NeuralTrainer(brain=brain)
    stages = {}
    for number in (1, 2, 3, 4):
        result = trainer.evaluate_math(number, tasks=tasks)
        stages[str(number)] = round(result["accuracy"] * 100, 1)
    return {"by_stage": stages,
            "score": round(sum(stages.values()) / len(stages), 1)}


def bench_pilot(brain, folder: Path | None, episodes: int, stage: int) -> dict:
    """Пилот: доля эпизодов, закончившихся орбитой, на Тренажёре++.

    Проверка идёт со стартового стола и на полной жёсткости — тем же
    методом, что и приёмный экзамен обучения.
    """
    import torch
    from kia_core.neural.trainer import NeuralTrainer
    if folder is None:
        return {}
    payload = torch.load(folder / "pilot.pt", map_location="cpu",
                         weights_only=True)
    brain.pilot.load_state_dict(payload["state"])
    brain.stage = int(payload.get("stage", 1))
    trainer = NeuralTrainer(brain=brain)
    from kia_core.neural.curriculum import stage_by_number
    result = trainer.evaluate_pilot(episodes=episodes,
                                    stage=stage_by_number(stage))
    orbits = 100.0 * result["orbits"] / max(1, result["episodes"])
    return {"score": round(orbits, 1),
            "mean_return": round(result["mean_return"], 1),
            "crashes": result["crashes"],
            "episodes": result["episodes"],
            "trained_stage": int(payload.get("stage", 1))}


def catalog_from_cache():
    """Каталог деталей прямо из кэша, без обращения к установленной игре.

    get_catalog() ищет GameData и без неё отдаёт мини-каталог из десятка
    деталей — на нём палитра вырождается и любая версия показывает ноль.
    Замер должен зависеть только от весов, поэтому читаем кэш как есть.
    """
    import json as _json
    from kia_core.engineer.parts_catalog import (CACHE_FILE, FALLBACK_DENSITY,
                                                 PartCatalog, PartInfo)
    data = _json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    parts = {name: PartInfo.from_dict(d) for name, d in data["parts"].items()}
    return PartCatalog(parts, data.get("densities", FALLBACK_DENSITY),
                       source=data.get("source", str(CACHE_FILE)))


def bench_builder(brain, folder: Path | None, episodes: int) -> dict:
    """Конструктор: доля сборок, которые проходят валидатор чертежа."""
    import torch
    from kia_core.neural.builder_env import BuilderEnv, build_palette
    if folder is None:
        return {}
    payload = torch.load(folder / "builder.pt", map_location="cpu",
                         weights_only=True)
    brain.builder.load_state_dict(payload["state"])
    stage = int(payload.get("stage", 1))
    catalog = catalog_from_cache()
    palette = build_palette(catalog=catalog)
    env = BuilderEnv(palette=palette, catalog=catalog, stage=stage,
                     seed=20260101)
    valid = 0
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        done, total, info = False, 0.0, {}
        while not done:
            mask = env.action_mask()
            _o, _c, discrete, _lp, _v = brain.builder.act(
                obs, deterministic=True, action_mask=mask,
                update_normalizer=False)
            obs, reward, done, info = env.step(int(discrete[0].item()))
            total += reward
        evaluation = info.get("evaluation")
        valid += int(bool(evaluation is not None and evaluation.valid))
        returns.append(total)
    return {"score": round(100.0 * valid / max(1, episodes), 1),
            "mean_return": round(sum(returns) / max(1, len(returns)), 2),
            "episodes": episodes,
            "trained_stage": stage}


def bench_emergency(brain, folder: Path | None, episodes: int) -> dict:
    """Аварии: доля отказов, после которых экипаж остался жив.

    Гоняется ступень 5 программы — тот же мир, что и на четвёртой, плюс
    отказы техники. Считаются только эпизоды, где отказ действительно
    сработал и на борту были люди: на исправном полёте мерить нечего.

    Для весов, которые отказов не видели, число будет низким. Это не
    придирка стенда, а ровно та дыра, ради которой заводится ступень.
    """
    import torch
    from kia_core.neural.curriculum import stage_by_number
    if folder is None:
        return {}
    payload = torch.load(folder / "pilot.pt", map_location="cpu",
                         weights_only=True)
    brain.pilot.load_state_dict(payload["state"])
    agent = brain.pilot
    stage = stage_by_number(5)
    faced = saved = crashes = 0
    for n in range(episodes):
        env = stage.make_env(seed=90_000 + n, exploring_starts=0.0)
        env.set_difficulty(1.0)
        obs = env.reset()
        done = False
        while not done:
            mask = env.action_mask()
            _o, continuous, discrete, _lp, _v = agent.act(
                obs, deterministic=True, action_mask=mask,
                update_normalizer=False)
            from kia_core.neural.spaces import FlightAction
            action = FlightAction(
                float(continuous[0]), float(continuous[1]),
                float(continuous[2]), float(continuous[3]),
                bool(int(discrete[0].item())))
            obs, _r, done, _info = env.step(action)
        summary = env.episode_summary()
        if not summary.get("failures") or not summary.get("crew"):
            continue
        faced += 1
        saved += int(bool(summary.get("crew_saved")))
        crashes += int(not summary.get("crew_saved"))
    return {"score": round(100.0 * saved / faced, 1) if faced else None,
            "faced": faced, "saved": saved, "lost": crashes,
            "episodes": episodes}


def bench_navigator(folder: Path | None, rounds: int, workers: int,
                    batch: int) -> dict:
    """Навигатор: доля миссий, в которых выбран верный план полёта."""
    from kia_core.neural.nav_plan import build_agent, load
    from kia_core.neural.navigator import WorkerPool
    if folder is None:
        return {}
    agent = build_agent()
    stage = load(agent, folder / "navigator.pt")
    if stage is None:
        return {}
    import torch
    pool = WorkerPool(workers, batch)
    try:
        rates = []
        for _ in range(rounds):
            obs, _c, _d = pool.reset(min(stage, 3))
            with torch.no_grad():
                cont, disc, _lp, _v = agent.net.act(torch.as_tensor(obs),
                                                    deterministic=True)
            _r, info = pool.step(cont.numpy(), disc.numpy())
            rates.append(float(info["ok"].mean()))
    finally:
        pool.close()
    return {"score": round(100.0 * sum(rates) / max(1, len(rates)), 1),
            "missions": rounds * workers * batch,
            "trained_stage": int(stage)}


# ==========================================================================
def run(library: Path, astra: Path, only: list[str], quick: bool) -> dict:
    from kia_core.neural.ppo import build_brain
    sizes = dict(math_tasks=60, pilot_episodes=4, nav_rounds=1, nav_workers=2,
                 nav_batch=64, emergency_episodes=10) if quick else             dict(math_tasks=600, pilot_episodes=40, nav_rounds=6, nav_workers=6,
                 nav_batch=256, emergency_episodes=300)

    available = discover(library)
    report = {"measured": time.strftime("%Y-%m-%d %H:%M"),
              "sizes": sizes, "versions": {}}
    for label, folder_name in VERSIONS.items():
        if only and label not in only:
            continue
        mods = available.get(folder_name)
        if not mods:
            print(f"[{label}] нет папки {folder_name} в {library}")
            continue
        print(f"\n=== {label}  ({folder_name}) ===", flush=True)
        brain = build_brain(load=False)
        row = {"source": folder_name}
        checks = (
            ("pilot", lambda: bench_pilot(brain, mods.get("Pilot"),
                                          sizes["pilot_episodes"], 3)),
            ("designer", lambda: bench_designer(mods.get("Builder"), astra)),
            ("math", lambda: bench_math(brain, mods.get("Math"),
                                        sizes["math_tasks"])),
            ("emergency", lambda: bench_emergency(
                brain, mods.get("Pilot"), sizes["emergency_episodes"])),
            ("navigator", lambda: bench_navigator(
                mods.get("Navigator"), sizes["nav_rounds"],
                sizes["nav_workers"], sizes["nav_batch"])),
        )
        for name, fn in checks:
            started = time.time()
            try:
                row[name] = fn()
            except Exception as exc:
                row[name] = {"error": f"{type(exc).__name__}: {exc}"}
            got = row[name].get("score", row[name].get("error", "—"))
            print(f"  {name:10} {got}   ({time.time() - started:.0f} с)",
                  flush=True)
        report["versions"][label] = row

    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n{OUT_FILE}")
    return report


# ==========================================================================
COLUMNS = (("pilot", "Piloting"), ("designer", "Design"),
           ("navigator", "Planning"), ("emergency", "Emergencies"))


def table(report: dict) -> str:
    """Markdown: строка на версию, лучшее в колонке — зелёным.

    Зелёным красим только когда есть что с чем сравнивать: подсветка
    единственной строки ничего не сообщает.
    """
    rows = [label for label in VERSIONS if label in report["versions"]]
    rows += [label for label in PENDING if label not in report["versions"]]
    measured = [r for r in rows if r in report["versions"]]

    best = {}
    for key, _ in COLUMNS:
        scores = [report["versions"][r].get(key, {}).get("score")
                  for r in measured]
        scores = [s for s in scores if isinstance(s, (int, float))]
        best[key] = max(scores) if len(scores) > 1 else None

    lines = ["| Version | " + " | ".join(t for _, t in COLUMNS) + " |",
             "|---|" + "---|" * len(COLUMNS)]
    for label in rows:
        cells = []
        for key, _ in COLUMNS:
            score = report["versions"].get(label, {}).get(key, {}).get("score")
            if not isinstance(score, (int, float)):
                cells.append("—")
            elif score == best[key]:
                cells.append(rf"$\color{{green}}\textsf{{{score} %}}$")
            else:
                cells.append(f"{score} %")
        lines.append(f"| **{label}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(LIBRARY))
    ap.add_argument("--astra", default=str(ASTRA))
    ap.add_argument("--only", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()

    if args.table:
        report = json.loads(OUT_FILE.read_text(encoding="utf-8"))
    else:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
        report = run(Path(args.library), Path(args.astra), only,
                     args.quick)
    print()
    print(table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
