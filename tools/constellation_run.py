"""Развёртывание созвездий связи: 4 спутника у Кербина, 4 у Муны.

Запускает подряд восемь миссий, меняя только точку стояния. Каждая
миссия — обычный полёт KIA: конструктор строит спутник, пилот выводит
его на орбиту и разводит по своей фазе (см. `pilot/constellation.py`).

Порядок важен: сначала Кербин. Спутники у Кербина дают ретрансляцию,
через которую держится связь с аппаратами у Муны — без них лунное
созвездие пришлось бы разворачивать вслепую.

Запуск: py -3 -u tools/constellation_run.py
        py -3 -u tools/constellation_run.py --только-кербин
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kia_core.logging_setup import setup_logging          # noqa: E402
from kia_core.mission_spec import MissionSpec             # noqa: E402

# Журнал заведён под именем «kia»; всё, что вне этого дерева, пишет в
# пустоту — на этом уже один раз потерялся весь разбор развода по фазе.
log = logging.getLogger("kia.constellation")

SIZE = 4

KERBIN = dict(title="Созвездие связи Кербина", target_body="Kerbin",
              home_body="Kerbin", park_orbit_altitude=800_000.0,
              objectives=["orbit", "relay"], collect_science=False)

MUN = dict(title="Созвездие связи Муны", target_body="Mun",
           home_body="Kerbin", park_orbit_altitude=80_000.0,
           target_orbit_altitude=250_000.0,
           objectives=["orbit", "relay"], collect_science=False)


def fly_one(base: dict, slot: int) -> float:
    """Один запуск. Возвращает счёт полёта (0 при неудаче)."""
    from kia_core.agent import Dispatcher

    spec = MissionSpec(constellation_size=SIZE, constellation_slot=slot, **base)
    spec.save()
    phase = spec.constellation_phase
    log.info("=" * 66)
    log.info("ЗАПУСК %d из %d: %s, точка %.0f°",
             slot + 1, SIZE, base["title"], phase)
    log.info("=" * 66)

    dispatcher = Dispatcher(resume=True)
    try:
        result = dispatcher.run_best()
    finally:
        dispatcher.disconnect()

    if result is None:
        log.error("Полёт не состоялся: нет связи с KSP")
        return 0.0
    log.info("Итог: %.0f очков | %s", result.score,
             ", ".join(result.record.milestones) or "—")
    return result.score


def main() -> int:
    setup_logging()
    only_kerbin = "--только-кербин" in sys.argv
    only_mun = "--только-муна" in sys.argv

    plan: list[dict] = []
    if not only_mun:
        plan.append(KERBIN)
    if not only_kerbin:
        plan.append(MUN)

    # С КАКОГО МЕСТА ПРОДОЛЖАТЬ. Первый аппарат созвездия — опора, от
    # него отсчитываются остальные. Если он уже висит на орбите,
    # повторять слот 0 нельзя: новый спутник встанет опоре на голову.
    start = 0
    for arg in sys.argv:
        if arg.startswith("--с-слота="):
            start = int(arg.split("=", 1)[1])

    scores: list[tuple[str, int, float]] = []
    for base in plan:
        for slot in range(start, SIZE):
            start = 0          # у следующего тела начинаем сначала
            score = fly_one(base, slot)
            scores.append((base["title"], slot, score))

    log.info("")
    log.info("ИТОГ РАЗВЁРТЫВАНИЯ")
    for title, slot, score in scores:
        mark = "+" if score > 0 else "-"
        log.info("  %s %s, точка %3.0f°: %.0f очков",
                 mark, title, (360.0 / SIZE) * slot, score)
    good = sum(1 for _, _, s in scores if s > 0)
    log.info("Поставлено аппаратов: %d из %d", good, len(scores))
    return 0 if good == len(scores) else 1


if __name__ == "__main__":
    raise SystemExit(main())
