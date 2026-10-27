"""Kerbal Intelligence Agency — точка входа.

Без аргументов открывается консоль прямого диалога с ИИ (cli.py).

    python main.py                      консоль управления
    python main.py init                 проверка окружения и связи с kRPC
    python main.py design "лети на Муну" расчёт аппарата + kia_design_report.md
    python main.py craft                сгенерировать .craft в VAB
    python main.py fly                  один полёт по текущей задаче
    python main.py train -g 5           обучение (5 поколений)
    python main.py train --offline      обучение без игры
    python main.py auto                 БЕСКОНЕЧНЫЙ автономный цикл
    python main.py status               отчёт по истории
    python main.py telemetry            живой поток телеметрии
    python main.py rewards              показать прейскурант наград
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kia_core.config import CONFIG, ROOT
from kia_core.logging_setup import setup_logging


# ==========================================================================
def cmd_init(args) -> int:
    """Проверка окружения: пакеты, каталог деталей, kRPC-сервер, судно."""
    setup_logging()
    print("=" * 68)
    print("  KERBAL INTELLIGENCE AGENCY — инициализация")
    print("=" * 68)
    ok = True

    try:
        import krpc
        print(f"[OK] пакет krpc установлен (клиент {getattr(krpc, '__version__', 'n/a')})")
    except ImportError:
        print("[FAIL] пакет krpc не установлен -> pip install krpc")
        return 1

    from kia_core.agent.provision import find_ksp_root
    ksp_root = find_ksp_root()
    if ksp_root:
        print(f"[OK] Каталог KSP: {ksp_root}")
    else:
        print("[WARN] Каталог KSP не найден — задайте KIA_CRAFT_KSP_ROOT")

    from kia_core.engineer.parts_catalog import get_catalog
    catalog = get_catalog()
    stats = catalog.stats()
    print(f"[OK] Каталог деталей: {stats['total']} шт "
          f"(двигателей {stats['engines']}, баков {stats['tanks']}, "
          f"разделителей {stats['decouplers']})")

    from kia_core.engineer import AutonomousEngineer, DesignPolicy
    from kia_core.mission_spec import MissionSpec
    spec = MissionSpec.load()
    design = AutonomousEngineer().design(spec, DesignPolicy())
    print(f"[OK] Конструктор: под задачу «{spec.title}» рассчитан аппарат "
          f"{design.total_mass:.1f} т, ΔV {design.total_delta_v:.0f} м/с, "
          f"TWR {design.launch_twr:.2f}, жизнеспособен={design.viable}")

    from kia_core.learning.reward_book import REWARD_BOOK
    print(f"[OK] Прейскурант: максимум {REWARD_BOOK.max_possible():.0f} очков, "
          f"пользовательских правил {len(REWARD_BOOK.rules)}")

    from kia_core.environment.connection import ConnectionFailed, KRPCConnection
    conn = KRPCConnection()
    try:
        conn.connect()
        print(f"[OK] kRPC-сервер: {conn.info().describe()}")
        try:
            vessel = conn.space_center.active_vessel
            print(f"[OK] Активное судно: '{vessel.name}', "
                  f"деталей {len(vessel.parts.all)}, масса {vessel.mass/1000:.2f} т")
        except Exception:
            print("[WARN] Активного судна нет (вы не в полётной сцене)")
    except ConnectionFailed as exc:
        print(f"[FAIL] {exc}")
        ok = False
    finally:
        conn.close()

    print("-" * 68)
    print("Готово. Запустите `python main.py` для консоли управления."
          if ok else "Инициализация завершена с ошибками (см. выше).")
    return 0 if ok else 2


# ==========================================================================
def cmd_design(args) -> int:
    setup_logging()
    from kia_core.engineer import AutonomousEngineer, DesignPolicy, write_report
    from kia_core.mission_spec import MissionSpec, parse_command

    spec = parse_command(args.task, MissionSpec.load()) if args.task else MissionSpec.load()
    if args.task:
        spec.save()
    design = AutonomousEngineer().design(spec, DesignPolicy())
    path = write_report(design)

    print(f"Задача: {spec.describe()}")
    print(f"Требуемый ΔV: {design.budget.total:.0f} м/с")
    for s in sorted(design.stages, key=lambda x: -x.requirement.index):
        print(f"  Ступень {s.requirement.index}: {s.engine.title} + "
              f"{s.tank_count}×{s.tank.title} → ΔV {s.delta_v():.0f} "
              f"(нужно {s.requirement.dv_required:.0f}), TWR {s.twr():.2f}")
    print(f"Итог: {design.total_mass:.2f} т | ΔV {design.total_delta_v:.0f} м/с | "
          f"TWR {design.launch_twr:.2f} | деталей {design.part_count}")
    print("Вердикт:", "жизнеспособен" if design.viable else
          "ОТКЛОНЁН — " + "; ".join(design.problems))
    print(f"Отчёт: {path}")
    return 0


# ==========================================================================
def cmd_craft(args) -> int:
    setup_logging()
    from kia_core.engineer import AutonomousEngineer, CraftAssembler, DesignPolicy
    from kia_core.engineer.craft_writer import validate_craft
    from kia_core.mission_spec import MissionSpec

    # ПОКАЗЫВАЕМ РОВНО ТО, ЧТО ПОЛЕТИТ.
    #
    # Раньше `craft` строил по умолчаниям, а автономный цикл — по геному
    # эволюции. Оператор одобрял в ангаре одну ракету, а в полёте видел
    # другую: вместо 60.7 т на «Бобкэте» с четырьмя «Колотушками»
    # взлетали 74.7 т на «Мастодонте». Закреплённая конструкция — общая
    # для обеих команд.
    from kia_core.learning.genome import load_design_lock

    spec = MissionSpec.load()
    design = AutonomousEngineer().design(spec, load_design_lock() or DesignPolicy())
    assembler = CraftAssembler(design.blueprint.name)
    text = assembler.render(design.blueprint)
    problems = validate_craft(text)
    if problems:
        print("Чертёж не прошёл проверку:")
        for p in problems:
            print("  !", p)
        return 2

    out_dir = Path(args.out) if args.out else assembler.craft_directory()
    if out_dir is None:
        out_dir = ROOT / "data" / "crafts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{assembler.craft_name}.craft"
    path.write_text(text, encoding="utf-8")
    print(f"Чертёж записан: {path}")
    print(f"Деталей: {design.part_count} | масса {design.total_mass:.2f} т | "
          f"ΔV {design.total_delta_v:.0f} м/с | TWR {design.launch_twr:.2f}")
    return 0


# ==========================================================================
def cmd_fly(args) -> int:
    setup_logging()
    from kia_core.agent import Dispatcher
    dispatcher = Dispatcher(resume=True)
    result = dispatcher.run_best()
    dispatcher.disconnect()
    if result is None:
        print("Полёт не состоялся: нет связи с KSP.")
        return 2
    print(f"\nИтог полёта: {result.score:.1f} очков | "
          f"достижения: {', '.join(result.record.milestones) or '—'}")
    print(f"Журнал полёта: {ROOT / 'kia_flight_log.md'}")
    return 0


# ==========================================================================
def cmd_train(args) -> int:
    setup_logging()
    from kia_core.agent import Dispatcher
    dispatcher = Dispatcher(offline=args.offline, resume=not args.restart,
                            seed=args.seed)
    print(dispatcher.status())
    stats = dispatcher.train(generations=args.generations)
    dispatcher.disconnect()
    print("\n=== ИТОГИ ОБУЧЕНИЯ ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


# ==========================================================================
def _single_instance_or_die() -> "Path | None":
    """Не даёт запустить второй автономный цикл на ту же игру.

    ДВА ЦИКЛА В ОДНОЙ ИГРЕ — ЭТО ХАОС, А НЕ УДВОЕНИЕ СКОРОСТИ.
    Один процесс ставит ракету на стол, второй в это же время грузит
    свою; оба командуют одним kRPC. В журнале это выглядело как «No such
    vessel», взрывы на старте и TWR 1.25 вместо 2.32, а в игре — как
    четыре клиента в панели kRPC вместо двух. Оператор заметил их
    раньше, чем я.
    """
    import os
    from kia_core.config import DATA_DIR

    lock = Path(DATA_DIR) / "autonomy.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except Exception:
            pid = -1
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)          # проверка «жив ли», без вреда
                alive = True
            except OSError:
                alive = False
        if alive:
            print(f"Автономный цикл уже работает (процесс {pid}). "
                  f"Второй запускать нельзя: они подерутся за одну игру.")
            return None
        lock.unlink()                    # замок от умершего процесса
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def cmd_auto(args) -> int:
    """Бесконечный автономный цикл: проектирование → полёт → разбор → правка."""
    setup_logging()
    lock = _single_instance_or_die()
    if lock is None:
        return 2
    from kia_core.agent import AutonomousLoop, Dispatcher
    dispatcher = Dispatcher(offline=args.offline, resume=True)
    loop = AutonomousLoop(dispatcher, max_iterations=args.iterations)
    try:
        summary = loop.run()
    except KeyboardInterrupt:
        loop.stop()
        summary = loop.summary()
    finally:
        dispatcher.disconnect()
        try:
            lock.unlink()
        except OSError:
            pass
    print("\n=== АВТОНОМНЫЙ РЕЖИМ ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {ROOT / 'kia_autonomy_report.md'}")
    return 0


# ==========================================================================
def cmd_status(args) -> int:
    setup_logging()
    from kia_core.learning.history import HistoryStore
    history = HistoryStore()
    print(history.report())
    curve = history.learning_curve()
    if curve:
        print("\nКривая обучения (поколение: лучший счёт):")
        for gen, score in curve:
            bar = "█" * max(0, min(50, int((score + 1000) / 100)))
            print(f"  {gen:3d} | {score:8.1f} {bar}")
    return 0


# ==========================================================================
def cmd_rewards(args) -> int:
    setup_logging()
    from kia_core.learning.reward_book import REWARD_BOOK
    print(REWARD_BOOK.describe())
    return 0


# ==========================================================================
def cmd_telemetry(args) -> int:
    setup_logging()
    from kia_core.environment import KRPCConnection, Telemetry
    conn = KRPCConnection()
    try:
        conn.connect()
        with Telemetry(conn) as tm:
            print("Живая телеметрия (Ctrl+C — выход)\n")
            while True:
                s = tm.snapshot()
                print(f"\r[{s.situation:>12}] h={s.altitude:9.0f} м | "
                      f"Ap={s.apoapsis:10.0f} | Pe={s.periapsis:10.0f} | "
                      f"v={s.speed:7.1f} м/с | m={s.mass:7.2f} т | "
                      f"T={s.thrust:7.1f} кН | TWR={s.twr:5.2f} | ст.{s.stage}",
                      end="", flush=True)
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nОстановлено.")
        return 0
    except Exception as exc:
        print(f"\nОшибка: {exc}")
        return 2
    finally:
        conn.close()


# ==========================================================================
def cmd_console(args) -> int:
    import cli
    return cli.main([])


# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kia", description="Kerbal Intelligence Agency — автономный ИИ-агент для KSP")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="проверка окружения и связи с kRPC")
    sub.add_parser("console", help="консоль прямого диалога с ИИ")

    p_design = sub.add_parser("design", help="рассчитать аппарат под задачу")
    p_design.add_argument("task", nargs="?", help="задача, например «лети на Муну»")

    p_craft = sub.add_parser("craft", help="сгенерировать .craft файл")
    p_craft.add_argument("--out", help="каталог для записи чертежа")

    sub.add_parser("fly", help="один полёт по текущей задаче")

    p_train = sub.add_parser("train", help="цикл обучения")
    p_train.add_argument("-g", "--generations", type=int,
                         default=CONFIG.learning.max_generations)
    p_train.add_argument("--offline", action="store_true",
                         help="без KSP: оценка только конструкций")
    p_train.add_argument("--restart", action="store_true", help="эволюция заново")
    p_train.add_argument("--seed", type=int, default=None)

    p_auto = sub.add_parser("auto", help="бесконечный автономный цикл обучения")
    p_auto.add_argument("-n", "--iterations", type=int, default=None,
                        help="предел итераций (по умолчанию без предела)")
    p_auto.add_argument("--offline", action="store_true")

    sub.add_parser("status", help="отчёт по истории обучения")
    sub.add_parser("rewards", help="показать прейскурант наград")
    sub.add_parser("telemetry", help="живой поток телеметрии")
    return parser


COMMANDS = {
    "init": cmd_init, "design": cmd_design, "craft": cmd_craft, "fly": cmd_fly,
    "train": cmd_train, "status": cmd_status, "telemetry": cmd_telemetry,
    "rewards": cmd_rewards, "console": cmd_console, "auto": cmd_auto,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        return cmd_console(args)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
