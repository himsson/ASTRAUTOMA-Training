"""Стенд перелёта: проверка межпланетной части без повторного взлёта.

ЗАЧЕМ. Каждая проба перелёта к Дюне стоила пятнадцати минут, из
которых шесть уходили на подъём и скругление — уже отработанные и
неизменные. Правишь расчёт окна — и снова смотришь, как ракета лезет
на 800 км. Стенд снимает эту плату: состояние на опорной орбите
сохраняется один раз, а дальше каждая проба начинается прямо с него.

Заодно чинится вторая беда: неудачная проба уводит вселенную за окно, и
следующая ждёт девятьсот суток. Загрузка сохранения возвращает время
назад вместе с аппаратом.

    py -3 tools/transfer_lab.py --запомнить   сохранить текущее состояние
    py -3 tools/transfer_lab.py               прогнать перелёт с него
    py -3 tools/transfer_lab.py --раз 5       пять проб подряд
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kia_core.logging_setup import get_logger, setup_logging   # noqa: E402
from kia_core.mission_spec import MissionSpec                  # noqa: E402

log = get_logger("lab")

SAVE_NAME = "kia_parking"


def _connect():
    """Обёртка проекта, а не голый клиент kRPC.

    Телеметрия ждёт объект с полем `conn` внутри — на голом клиенте
    проба падала с «Client object has no attribute conn».
    """
    from kia_core.environment.connection import KRPCConnection
    link = KRPCConnection()
    link.connect()
    return link


def remember(conn) -> int:
    """Запоминает текущее состояние как точку начала проб."""
    sc = conn.space_center
    v = sc.active_vessel
    pe = v.orbit.periapsis_altitude / 1000.0
    ap = v.orbit.apoapsis_altitude / 1000.0
    if v.orbit.body.name == "Kerbin" and pe < 70.0:
        log.error("Аппарат не на орбите (%.0f x %.0f км) — запоминать нечего",
                  pe, ap)
        return 2
    sc.save(SAVE_NAME)
    log.info("Состояние сохранено как «%s»: %s, орбита %.0f x %.0f км вокруг %s",
             SAVE_NAME, v.name, pe, ap, v.orbit.body.name)
    return 0


def run_once(conn, spec: MissionSpec) -> bool:
    """Одна проба: грузим сохранение и гоняем перелёт."""
    from kia_core.environment.control import ShipControl
    from kia_core.environment.telemetry import Telemetry
    from kia_core.learning.genome import FlightGenome
    from kia_core.learning.rewards import RewardSystem
    from kia_core.pilot.maneuver import ManeuverExecutor
    from kia_core.pilot.transfer import TransferPlanner

    sc = conn.space_center
    sc.load(SAVE_NAME)
    time.sleep(3.0)

    telemetry = Telemetry(conn)
    control = ShipControl(conn)
    genome = FlightGenome()
    rewards = RewardSystem()
    maneuver = ManeuverExecutor(conn, telemetry, control)
    transfer = TransferPlanner(conn, telemetry, control, maneuver, genome,
                               rewards, target=spec.target_body,
                               target_periapsis=spec.target_orbit_altitude)

    t0 = time.time()
    node = transfer.plan_transfer()
    if node is None:
        log.warning("Проба: узел не построен (%.0f с)", time.time() - t0)
        return False
    if not transfer.execute_transfer(node):
        log.warning("Проба: инжекция не добрана")
        return False
    ok = transfer.cruise_to_planet()
    log.info("Проба заняла %.0f с, вход в сферу: %s",
             time.time() - t0, "ДА" if ok else "нет")
    if ok:
        captured = transfer.capture()
        log.info("Захват: %s", "ДА" if captured else "нет")
        return captured
    return False


def main() -> int:
    setup_logging()
    conn = _connect()
    if "--запомнить" in sys.argv:
        return remember(conn)

    spec = MissionSpec.load()
    times = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--раз" and i + 1 < len(sys.argv):
            times = int(sys.argv[i + 1])

    good = 0
    for attempt in range(1, times + 1):
        log.info("=" * 60)
        log.info("ПРОБА %d из %d: перелёт к %s", attempt, times, spec.target_body)
        try:
            if run_once(conn, spec):
                good += 1
        except Exception as exc:
            log.error("Проба сорвалась: %s", exc)
    log.info("Удачных проб: %d из %d", good, times)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
