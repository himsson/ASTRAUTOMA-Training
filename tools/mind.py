"""Пульт мыслей: что ИИ видит и что решает — человеческим языком.

Оператор просил не «журнал», а рассказ от первого лица: «рассчитал
манёвр», «тяга 100», «кренит вправо — доворачиваю влево». Здесь ровно
это. Отдельный клиент kRPC, ничем не управляет — только смотрит и
говорит. Пилота не трогает и остановить полёт не может.

Запуск: py -3 tools/mind.py    (или KIA_МЫСЛИ.bat)
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

TICK = 0.5              # с между замерами
SAY_EVERY = 4.0         # с между обычными репликами: чаще — сплошная стена
TURN_LIMIT = 5.0        # °/с, ниже этого крен не стоит упоминания
POINT_LIMIT = 10.0      # ° ошибки наведения, ниже — «держу курс»

# Раскраска: ошибки красным, решения жёлтым, замеры обычным.
RED, YEL, CYN, GRY, OFF = "\033[91m", "\033[93m", "\033[96m", "\033[90m", "\033[0m"


def _bar(value: float, width: int = 10) -> str:
    """Полоска газа: видно с другого конца комнаты."""
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _side(value: float, plus: str, minus: str) -> str:
    return plus if value > 0 else minus


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"{CYN}=== ПУЛЬТ МЫСЛЕЙ KIA ==={OFF}")
    print(f"{GRY}Здесь видно, что ИИ замечает и что решает делать.")
    print(f"Окно можно закрывать в любой момент — на полёт это не влияет.{OFF}\n")

    while True:
        try:
            conn = krpc.connect(name="KIA_mind")
            break
        except Exception:
            print(f"{GRY}жду kRPC (KSP запущен? сервер включён?)…{OFF}")
            time.sleep(3.0)

    sc = conn.space_center
    said_at = 0.0
    last_stage: int | None = None
    last_met: float | None = None
    last_throttle = 0.0
    last_phrase = ""

    def say(text: str, colour: str = "") -> None:
        nonlocal last_phrase
        if text == last_phrase:
            return
        last_phrase = text
        stamp = time.strftime("%H:%M:%S")
        print(f"{GRY}{stamp}{OFF} {colour}{text}{OFF}", flush=True)

    def event(text: str, colour: str = YEL) -> None:
        """Событие говорим всегда, не глядя на паузу между репликами."""
        nonlocal last_phrase
        last_phrase = ""
        stamp = time.strftime("%H:%M:%S")
        print(f"{GRY}{stamp}{OFF} {colour}{text}{OFF}", flush=True)

    while True:
        try:
            v = sc.active_vessel
            met = v.met
            body = v.orbit.body
            frame = body.reference_frame
            fl = v.flight(frame)
            altitude = fl.mean_altitude
            vertical = fl.vertical_speed
            speed = fl.speed
            stage = v.control.current_stage
            throttle = v.control.throttle
            ap = v.auto_pilot
            ctl = v.control
            spin = math.degrees(math.hypot(*v.angular_velocity(
                body.non_rotating_reference_frame)))
            try:
                error = float(ap.error)
            except Exception:
                error = float("nan")
            charge = v.resources.amount("ElectricCharge")
            ap_alt = v.orbit.apoapsis_altitude
            pe_alt = v.orbit.periapsis_altitude
        except Exception:
            say("сменилась сцена — жду, пока появится аппарат", GRY)
            time.sleep(2.0)
            last_stage = last_met = None
            continue

        now = time.time()

        # --- новый полёт --------------------------------------------------
        if last_met is not None and met + 1.0 < last_met:
            print()
            event(f"{CYN}=== НОВЫЙ ПОЛЁТ: {v.name} ==={OFF}", "")
            last_stage = None
        last_met = met

        # --- события: их пропускать нельзя --------------------------------
        if last_stage is None:
            last_stage = stage
        elif stage != last_stage:
            event(f"отделяю ступень {last_stage} → {stage}: отработавшее "
                  f"железо только мешает (t+{met:.0f} с, {altitude/1000:.1f} км)")
            last_stage = stage

        if throttle > 0.05 and last_throttle <= 0.05:
            event(f"зажигание, тяга {throttle*100:.0f}% {_bar(throttle)}")
        elif throttle <= 0.05 and last_throttle > 0.05:
            event(f"глушу двигатель — импульс набран")
        last_throttle = throttle

        # --- обычная речь: не чаще, чем раз в SAY_EVERY --------------------
        if now - said_at < SAY_EVERY:
            time.sleep(TICK)
            continue
        said_at = now

        # Порядок реплик = порядок важности. Говорим ОДНУ, самую важную:
        # сплошной поток человек не читает, а тревогу в нём не замечает.
        if charge < 5.0:
            say(f"заряд почти на нуле ({charge:.0f}) — вот-вот потеряю "
                f"управление", RED)

        elif not math.isfinite(error):
            say(f"автопилот молчит — веду на глаз", GRY)

        elif spin > TURN_LIMIT and error > POINT_LIMIT:
            # Куда именно кренит и куда я отклоняю рули.
            pitch_word = _side(ctl.pitch, "задираю нос", "опускаю нос")
            yaw_word = _side(ctl.yaw, "довожу вправо", "довожу влево")
            say(f"нос ушёл на {error:.0f}°, крутит {spin:.0f}°/с — "
                f"{pitch_word}, {yaw_word} "
                f"(руль {ctl.pitch:+.2f}/{ctl.yaw:+.2f})",
                RED if spin > 25 else YEL)

        elif error > POINT_LIMIT:
            say(f"довожу нос до цели: осталось {error:.0f}°")

        elif throttle > 0.05:
            say(f"тяга {throttle*100:.0f}% {_bar(throttle)}, курс держу "
                f"({error:.0f}° ошибки), скорость {speed:.0f} м/с, "
                f"Ap {ap_alt/1000:.0f} км / Pe {pe_alt/1000:.0f} км")

        elif altitude < 70000 and vertical > 5:
            say(f"поднимаюсь по инерции: {altitude/1000:.1f} км, "
                f"{vertical:+.0f} м/с, апоцентр будет {ap_alt/1000:.0f} км")

        elif altitude < 70000 and vertical < -5:
            say(f"снижаюсь {vertical:.0f} м/с на {altitude/1000:.1f} км", RED)

        else:
            say(f"иду по орбите {pe_alt/1000:.0f} × {ap_alt/1000:.0f} км, "
                f"двигатель молчит, жду точки манёвра", GRY)

        time.sleep(TICK)


if __name__ == "__main__":
    raise SystemExit(main())
