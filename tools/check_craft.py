"""ПРОВЕРКА ЧЕРТЕЖА В ЖИВОЙ ИГРЕ — блокер номер один (MIGRATION 7.1).

    python tools/check_craft.py            собрать и поставить на стол
    python tools/check_craft.py --design   только посчитать, игру не трогать

Что делает. Инженерная часть считает ракету по орбитальной механике,
пишет `.craft` и просит KSP поставить её на стартовый стол. Дальше
скрипт сверяет ЧИСЛО ДЕТАЛЕЙ на столе с расчётным: игра либо собрала
аппарат целиком, либо молча выкинула часть дерева — и второе выглядит
как «площадка пустая», без единой ошибки в логе.

Почему отдельным скриптом, а не пунктом пульта: это разовая проверка
одного блокера, и она должна быть простой настолько, чтобы её результат
нельзя было истолковать двояко.

ЧТО НУЖНО: KSP запущена, сохранение загружено, в окне мода kRPC нажато
`Start server`. Проверка ЗАЙМЁТ СТАРТОВЫЙ СТОЛ — она реально ставит
аппарат на площадку.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kia_core.engineer.autodesign import DesignPolicy      # noqa: E402
from kia_core.logging_setup import setup_logging           # noqa: E402


def describe(design) -> None:
    print("\n=== ЧТО ПОСЧИТАЛ ИНЖЕНЕР ===")
    print(f"  деталей в чертеже:     {design.part_count}")
    print(f"  стартовая масса:       {design.total_mass:.2f} т")
    print(f"  полезная нагрузка:     {design.payload_mass:.3f} т")
    print(f"  запас ΔV:              {design.total_delta_v:.0f} м/с "
          f"(нужно {design.budget.total:.0f})")
    print(f"  тяга у земли:          TWR {design.launch_twr:.2f}")
    print(f"  чертёж пригоден:       {'да' if design.viable else 'НЕТ'}")
    if design.problems:
        print(f"  проблемы:              {'; '.join(design.problems)}")
    if design.warnings:
        print(f"  предупреждения:        {'; '.join(design.warnings)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_craft",
        description="Проверка беспилотного чертежа в живой KSP")
    parser.add_argument("--design", action="store_true",
                        help="только расчёт, без подключения к игре")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    setup_logging()

    from kia_core.agent.dispatcher import Dispatcher

    dispatcher = Dispatcher(resume=False)

    if args.design:
        design = dispatcher.engineer.design(dispatcher.spec, DesignPolicy())
        describe(design)
        print("\nИгра не трогалась: это был только расчёт.")
        return 0

    print("Подключаюсь к KSP...")
    if not dispatcher.connect():
        print("\n[НЕТ СВЯЗИ] KSP не отвечает. Проверь, что игра запущена,")
        print("            сохранение загружено, а в окне kRPC нажато")
        print("            `Start server` (порты 50000/50001).")
        return 2

    try:
        design = dispatcher.engineer.design(dispatcher.spec, DesignPolicy())
        describe(design)

        print("\n=== СТАВЛЮ НА СТАРТОВЫЙ СТОЛ ===")
        result = dispatcher.provisioner.provision(design.blueprint)
        print(f"  способ:   {result.strategy}")
        print(f"  файл:     {result.craft_path or '—'}")
        print(f"  итог:     {result.message}")
        if not result.ok:
            print("\n[НЕ СОБРАЛОСЬ] Аппарат на стол не встал.")
            print("  Смотреть KSP.log в корне игры: строки ShipConstruct."
                  "LoadShip и GetAttachNodeInfo.")
            return 1

        # Главная проверка: игра могла молча выкинуть часть дерева
        try:
            vessel = dispatcher.connection.space_center.active_vessel
            actual = len(vessel.parts.all)
        except Exception as exc:
            print(f"\n[НЕ ВИЖУ АППАРАТ] Судно не читается через kRPC: {exc}")
            return 1

        print("\n=== ЧТО ПОЛУЧИЛОСЬ В ИГРЕ ===")
        print(f"  деталей на столе:      {actual}")
        print(f"  деталей в чертеже:     {design.part_count}")
        if actual == design.part_count:
            print("\n[СОБРАЛОСЬ ЦЕЛИКОМ] Все детали на месте — блокер 7.1 снят.")
            return 0
        print(f"\n[СОБРАЛОСЬ ЧАСТИЧНО] Игра потеряла "
              f"{design.part_count - actual} деталей.")
        print("  Это и есть та тихая поломка, ради которой проверка "
              "написана:")
        print("  валидатор чертёж пропускает, а KSP молча обрезает дерево.")
        print("  Смотреть KSP.log: ShipConstruct.LoadShip, GetAttachNodeInfo.")
        return 1
    finally:
        dispatcher.disconnect()


if __name__ == "__main__":
    sys.exit(main())
