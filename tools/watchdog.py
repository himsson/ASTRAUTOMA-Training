"""Сторож полёта: смотрит за тем, что оператор видит глазами.

Подключается ВТОРЫМ клиентом kRPC и печатает только тревоги — то, ради
чего оператор просил «смотреть игру»: перегрев, падение, незажжённые
двигатели, невыброшенная связка.

Каждая проверка выведена из настоящего провала, а не придумана:

* ПУСТАЯ СТУПЕНЬ НА БОРТУ. Замер полёта 22:13: связка «Колотушек»
  выгорела за 44 с, а ракета шла на той же стадии ПЯТЬ минут, таща
  тридцать тонн мёртвого груза. Апоапсис встал на 50 км.
* НЕ ОТОРВАЛАСЬ. Полёт с TWR 0.40: двигатель работает, ракета стоит.
* ПЕРЕГРЕВ. Оператор прислал снимок горящего аппарата — в числах это
  отношение температуры обшивки к предельной.
* ПАДЕНИЕ. Вертикальная скорость ушла в минус на подъёме.

Запуск: py -3 tools/watchdog.py
"""
from __future__ import annotations

import math
import sys
import time

try:
    import krpc
except ImportError:                                    # pragma: no cover
    print("krpc не установлен", file=sys.stderr)
    raise SystemExit(1)

# Пороги. Ни один не «на глаз»: см. пояснения в шапке.
HOT_RATIO = 0.75          # доля от предельной температуры детали
DEAD_STAGE_GRACE = 12.0   # с, сколько терпим выгоревший двигатель на борту
NO_LIFT_GRACE = 8.0       # с после зажигания без набора высоты
FALL_GRACE = 6.0          # с непрерывного снижения на подъёме
SPIN_LIMIT = 15.0         # °/с — выше этого считаем раскруткой
SPIN_GRACE = 4.0          # с подряд, прежде чем поднимать тревогу


def main() -> int:
    # Журнал читаю я, а не консоль Windows: без этого кириллица
    # превращается в мусор и разбирать тревоги невозможно.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    conn = krpc.connect(name="KIA_watchdog")
    sc = conn.space_center
    print("сторож на связи", flush=True)

    last_count: int | None = None
    last_met: float | None = None
    staged_at = time.time()
    spin_since: float | None = None
    spin_error0: float | None = None
    dead_since: float | None = None
    fall_since: float | None = None
    ignite_at: float | None = None
    last_stage: int | None = None
    best_apoapsis = 0.0
    said: set[str] = set()

    def alarm(key: str, text: str) -> None:
        """Каждую беду сообщаем один раз — иначе журнал утонет в повторах."""
        if key not in said:
            said.add(key)
            print(f"!!! ТРЕВОГА: {text}", flush=True)

    while True:
        try:
            vessel = sc.active_vessel
            met = vessel.met
            flight = vessel.flight(vessel.orbit.body.reference_frame)
            altitude = flight.mean_altitude
            vertical = flight.vertical_speed
            stage = vessel.control.current_stage
            throttle = vessel.control.throttle
        except Exception:
            # Смена сцены, пересоздание судна — это не беда, ждём.
            time.sleep(2.0)
            dead_since = fall_since = ignite_at = spin_since = None
            last_stage = None
            last_count = None
            last_met = None
            best_apoapsis = 0.0
            said.clear()
            continue

        # НОВЫЙ ПОЛЁТ — НОВЫЙ СЧЁТ.
        #
        # Сброс шёл только при обрыве связи, поэтому счётчики переезжали
        # из полёта в полёт и давали ложную тревогу: «двигатели работают
        # 961 с, а аппарат не поднимается» — на деле шла уже следующая
        # ракета, только что поставленная на стол. Признак нового полёта
        # надёжный: время миссии пошло назад.
        if last_met is not None and met + 1.0 < last_met:
            dead_since = fall_since = ignite_at = spin_since = None
            last_stage = last_count = None
            best_apoapsis = 0.0
            said.clear()
            print(f"--- новый полёт ---", flush=True)
        last_met = met

        if last_stage is None or stage != last_stage:
            if last_stage is not None:
                print(f"стадия {last_stage} -> {stage} на t+{met:.0f} с, "
                      f"h={altitude:.0f} м", flush=True)
            last_stage = stage
            dead_since = None
            staged_at = time.time()

        # --- ВЗРЫВ: деталей стало меньше без отделения ступени ----------
        #
        # Оператор описывал это словами «что-то рвануло на старте». В
        # числах взрыв отличается от штатного разделения только тем, что
        # при разделении меняется номер стадии, а при взрыве — нет.
        try:
            count = len(vessel.parts.all)
            # СБРОС ОБТЕКАТЕЛЯ — НЕ ВЗРЫВ. Створки уходят без смены
            # стадии, и деталей честно становится меньше: замер полёта
            # 11:57 дал «41 вместо 42» ровно в ту секунду, когда журнал
            # писал «створки обтекателя сброшены».
            jettison = any(p.fairing is not None and p.fairing.jettisoned
                           for p in vessel.parts.all)
            if last_count is not None and count < last_count and not jettison:
                if time.time() - staged_at > 2.0:
                    alarm(f"boom{count}",
                          f"деталей стало {count} вместо {last_count} без "
                          f"смены стадии — что-то разрушено "
                          f"(t+{met:.0f} с, h={altitude:.0f} м)")
            last_count = count
        except Exception:
            pass

        now = time.time()

        # --- выгоревший двигатель всё ещё на борту -----------------------
        try:
            active = [e for e in vessel.parts.engines if e.active]
            spent = [e for e in active if not e.has_fuel]
            if spent and throttle > 0.05:
                dead_since = dead_since or now
                if now - dead_since > DEAD_STAGE_GRACE:
                    alarm("dead",
                          f"{len(spent)} выгоревших двигателей висят на борту "
                          f"{now - dead_since:.0f} с — ступень не сброшена "
                          f"(t+{met:.0f} с, h={altitude:.0f} м)")
            else:
                dead_since = None
        except Exception:
            pass

        # --- не оторвалась ----------------------------------------------
        try:
            if throttle > 0.5 and vessel.available_thrust > 1.0 and met < 60:
                ignite_at = ignite_at or now
                if altitude < 100 and now - ignite_at > NO_LIFT_GRACE:
                    twr = (vessel.available_thrust
                           / max(1e-6, vessel.mass * 9.81))
                    alarm("nolift",
                          f"двигатели работают {now - ignite_at:.0f} с, а "
                          f"аппарат не поднимается (TWR={twr:.2f})")
        except Exception:
            pass

        # --- перегрев ----------------------------------------------------
        try:
            worst, worst_name = 0.0, ""
            for part in vessel.parts.all:
                limit = part.max_skin_temperature or 0.0
                if limit <= 0:
                    continue
                ratio = (part.skin_temperature or 0.0) / limit
                if ratio > worst:
                    worst, worst_name = ratio, part.title
            if worst > HOT_RATIO:
                alarm("hot",
                      f"перегрев: {worst_name} на {worst * 100:.0f}% от "
                      f"предела (t+{met:.0f} с, h={altitude:.0f} м)")
        except Exception:
            pass

        # --- створки не сброшены за атмосферой ---------------------------
        #
        # Оператор увидел это глазами раньше любой проверки: обтекатель
        # ещё на ракете, панели не раскрыты, аппарат крутится. Всё это —
        # одно и то же: под закрытыми створками панели не видят солнца,
        # заряд садится, управление пропадает.
        try:
            if altitude > 72000:
                stuck = sum(1 for p in vessel.parts.all
                            if p.fairing is not None and not p.fairing.jettisoned)
                if stuck:
                    alarm("fairing",
                          f"обтекатель не сброшен на высоте {altitude:.0f} м "
                          f"({stuck} шт) — панели под ним не работают")
                charge = vessel.resources.amount("ElectricCharge")
                if charge < 1.0:
                    alarm("power",
                          f"заряд на нуле (t+{met:.0f} с) — аппарат "
                          f"неуправляем")
        except Exception:
            pass

        # --- раскрутка: ловим момент и обстоятельства ---------------------
        try:
            ref = vessel.orbit.body.non_rotating_reference_frame
            spin = math.degrees(math.hypot(*vessel.angular_velocity(ref)))
            ap = vessel.auto_pilot
            try:
                ap_error = float(ap.error)
            except Exception:
                ap_error = float("nan")
            if spin > SPIN_LIMIT:
                if spin_since is None:
                    spin_since = now
                    spin_error0 = ap_error
                # РАЗВОРОТ — НЕ РАСКРУТКА.
                #
                # Замер 09:03: 24 °/с при ошибке 49° — это автопилот
                # доворачивал на узел, полёт закончился орбитой 148x111 км.
                # Настоящая раскрутка отличается тем, что ошибка наведения
                # НЕ убывает: аппарат крутится мимо цели, а не к ней.
                closing = (spin_error0 is not None
                           and math.isfinite(ap_error)
                           and math.isfinite(spin_error0)
                           and ap_error < spin_error0 - 5.0)
                if now - spin_since > SPIN_GRACE and not closing:
                    ctl = vessel.control
                    alarm(f"spin{int(met/60)}",
                          f"РАСКРУТКА {spin:.0f}°/с {now-spin_since:.0f} с | "
                          f"t+{met:.0f} h={altitude:.0f} ст.{stage} "
                          f"газ={throttle:.2f} ошибка АП={ap.error:.0f}° "
                          f"команды p={ctl.pitch:+.2f} y={ctl.yaw:+.2f} "
                          f"r={ctl.roll:+.2f} SAS={ctl.sas} "
                          f"заряд={vessel.resources.amount('ElectricCharge'):.0f}")
            else:
                spin_since = None
        except Exception:
            pass

        # --- падение на подъёме ------------------------------------------
        try:
            apoapsis = vessel.orbit.apoapsis_altitude
            best_apoapsis = max(best_apoapsis, apoapsis)
            if vertical < -5.0 and altitude < 70000 and met > 20:
                fall_since = fall_since or now
                if now - fall_since > FALL_GRACE:
                    alarm("fall",
                          f"снижение {vertical:.0f} м/с на высоте "
                          f"{altitude:.0f} м (максимум апоапсиса был "
                          f"{best_apoapsis:.0f} м)")
            else:
                fall_since = None
        except Exception:
            pass

        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
