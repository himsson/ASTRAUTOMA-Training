"""Живая среда KSP для нейросети: kRPC как gym-окружение.

reset()  — вывести аппарат на стартовый стол и вернуть первое наблюдение
step(a)  — применить действие сети (тяга, тангаж, рыскание, крен,
           отделение ступени), подождать шаг управления и вернуть
           (наблюдение, награда, конец эпизода, детали)

Автопилот kRPC здесь НЕ используется: сеть управляет осями напрямую,
как человек рулит с клавиатуры. Никакого гравитационного разворота по
формуле — только то, что выдала политика.
"""
from __future__ import annotations

import math
import time

from ..flight_log import FlightLog
from ..logging_setup import get_logger
from .spaces import FlightAction, encode_flight_state

log = get_logger("neural.flight_env")


class KSPFlightEnv:
    """Обёртка вокруг живой игры с интерфейсом обучения с подкреплением."""

    # Те же правила, что в Тренажёре+: сеть училась с ними, значит и в игре
    # они должны действовать, иначе выученное поведение упрётся в другой мир.
    STAGE_COOLDOWN = 2.0         # с, блокировка повторного отделения
    SPENT_STAGE_FUEL = 0.03      # доля остатка, ниже которой ступень пустая
    # Цена раскрутки по крену — те же значения, что в Тренажёре++
    # (`PrecisionAscentSimulator.ROLL_RATE_FREE` и `.ROLL_PENALTY`)
    ROLL_RATE_FREE = 10.0        # град/с, до этого не наказывается
    ROLL_PENALTY = 0.001         # за каждые 10 град/с сверх свободного, за шаг

    def __init__(self, dispatcher, goal_altitude: float = 80_000.0,
                 control_seconds: float = 1.0,
                 max_episode_seconds: float = 600.0,
                 flight_log: FlightLog | None = None):
        """control_seconds — как часто сеть принимает решение, СЕКУНД.

        Это не настройка удобства, а часть договора со средой обучения.
        Ритм задаётся ступенью программы (`Stage.decision_seconds`) и
        обязан совпадать: сеть, учившаяся решать раз в 0.3 с, в мире,
        который спрашивает её раз в секунду, ведёт аппарат втрое грубее,
        и профиль выведения рассыпается. Ровно на таком рассинхроне
        (1.0 с против 0.3 с) сегодня целый день не бралась третья
        ступень — только там он был между тренажёрами.
        """
        self.dispatcher = dispatcher
        self.goal_altitude = goal_altitude
        self.control_dt = max(0.05, float(control_seconds))
        self.max_episode_seconds = max_episode_seconds
        # фактический ритм: по нему видно, поспевает ли игра за сетью
        self.period_sum = 0.0
        self.period_count = 0
        self.period_worst = 0.0
        self.vanished = 0            # подряд идущих «деталей ноль»
        self._staging_unknown = False   # чтобы не сыпать одно и то же в лог
        self.flight_log = flight_log
        self.telemetry = None
        self.control = None
        self.steps = 0
        self.episode_return = 0.0
        self.started_wall = 0.0
        self.previous_apoapsis = 0.0
        self.previous_altitude = 0.0
        self.previous_energy = 0.0
        self.previous_momentum = 0.0
        self.initial_momentum = 0.0
        self.previous_roll = 0.0
        self.body_radius = 600_000.0
        self.body_mu = 3.5316e12
        self.peak_altitude = 0.0
        self.reached_orbit = False
        self.done = True
        self.last_info: dict = {}
        self.last_snapshot = None
        self.last_stage_wall = 0.0

    # ------------------------------------------------------------------
    def reset(self, blueprint=None) -> list[float] | None:
        """Ставит аппарат на стол и готовит телеметрию."""
        from ..environment.control import ShipControl
        from ..environment.telemetry import Telemetry

        if not self.dispatcher.connect():
            log.error("Нет связи с KSP — эпизод не начат")
            return None

        provision = self.dispatcher.provisioner.provision(blueprint)
        if not provision.ok:
            log.error("Аппарат не выведен на стол: %s", provision.message)
            return None

        try:
            self.telemetry = Telemetry(self.dispatcher.connection)
            self.control = ShipControl(self.dispatcher.connection, self.telemetry,
                                       flight_log=self.flight_log)
        except Exception as exc:
            log.error("Телеметрия недоступна: %s", exc)
            return None

        # Сеть управляет осями сама: автопилот и SAS выключены
        self.control.disengage_autopilot()
        self.control.set_sas(False)
        self.control.set_throttle(0.0)

        self.steps = 0
        self.episode_return = 0.0
        self.started_wall = time.time()
        self.reached_orbit = False
        self.done = False
        self.last_stage_wall = 0.0
        self.vanished = 0
        self.period_sum = self.period_worst = 0.0
        self.period_count = 0
        snap = self.telemetry.snapshot(detailed=True)
        self.last_snapshot = snap
        try:
            body = self.telemetry.body
            self.body_radius = float(body.equatorial_radius)
            self.body_mu = float(body.gravitational_parameter)
        except Exception:
            pass                      # значения по умолчанию — Кербин
        self.previous_apoapsis = snap.apoapsis
        self.previous_altitude = snap.altitude
        self.previous_energy = self.specific_energy(snap)
        # На столе аппарат уже вращается вместе с планетой: это его
        # стартовый момент импульса, от него и считаем шкалу
        self.initial_momentum = self.angular_momentum(snap)
        self.previous_momentum = self.initial_momentum
        self.previous_roll = snap.roll
        self.peak_altitude = snap.altitude
        if self.flight_log:
            self.flight_log.status("нейросеть приняла управление",
                                   f"цель {self.goal_altitude/1000:.0f} км")
        return self.encode(snap)

    # ------------------------------------------------------------------
    def encode(self, snap) -> list[float]:
        body = self.telemetry.body
        try:
            gravity = body.surface_gravity
            radius = body.equatorial_radius
            has_atmosphere = bool(body.has_atmosphere)
        except Exception:
            gravity, radius, has_atmosphere = 9.81, 600_000.0, True
        prograde_pitch = self._prograde_pitch(snap)
        return encode_flight_state(
            snap, self.goal_altitude, gravity, radius, has_atmosphere,
            prograde_pitch=prograde_pitch,
            angle_of_attack=self._slip_angle(snap))

    def _prograde_pitch(self, snap) -> float:
        speed = max(1e-3, snap.speed)
        return math.degrees(math.asin(
            max(-1.0, min(1.0, snap.vertical_speed / speed))))

    @staticmethod
    def _wrap180(angle: float) -> float:
        """Разница углов в градусах, приведённая к [-180, 180]."""
        return (float(angle) + 180.0) % 360.0 - 180.0

    def _slip_angle(self, snap) -> float:
        """Полное отклонение носа от потока: тангаж И курс.

        В тренажёре скольжение считается по обеим осям
        (`hypot(ошибка тангажа, ошибка рыскания)`), и наказывается тоже
        по обеим. Здесь раньше учитывался только тангаж — то есть ровно
        та ошибка, из-за которой ось рыскания оказывалась без
        последствий и уползала на упор (5.3i, 5.3j). Игра считает оба
        угла сама; если сборка kRPC их не отдаёт, остаётся прежний
        расчёт по тангажу.
        """
        if getattr(snap, "aero_angles_known", False):
            return math.hypot(snap.angle_of_attack, snap.sideslip_angle)
        return abs(snap.pitch - self._prograde_pitch(snap))

    # ------------------------------------------------------------------
    def action_mask(self) -> list[bool]:
        """Что разрешено сети прямо сейчас: [не отделять, отделить].

        Повторяет правила Тренажёра+. Отделение полной ступени выбрасывает
        работающий двигатель вместе с топливом — на прогоне 4.5 млн шагов
        политика делала это на первом же шаге и оставалась без тяги
        (MIGRATION 7.3). Блокировка на 2 секунды после отделения не даёт
        одной командой сбросить всю ракету.
        """
        if time.time() - self.last_stage_wall < self.STAGE_COOLDOWN:
            return [True, False]
        snap = self.last_snapshot
        if snap is None:
            return [True, False]
        stage_resources = snap.stage_resources or {}
        current = stage_resources.get(str(snap.stage - 1)) \
            or stage_resources.get(str(snap.stage)) or {}
        fuel = current.get("LiquidFuel") or {}
        top = fuel.get("max", 0.0) or 0.0
        if top <= 0.0:
            # НЕ ЗНАЕМ — ЗНАЧИТ НЕЛЬЗЯ. Раньше здесь стояло обратное
            # («не мешаем сети решать»), и это сводило маску на нет: в
            # живой игре чтение топлива по ступеням не срабатывало, запрет
            # превращался в разрешение, и первый же полёт кончился тем,
            # что сеть за пять секунд отстрелила от себя всё — включая
            # работающую первую ступень — и осталась с тягой 0 кН.
            #
            # Маска нужна ровно для этого случая (MIGRATION 7.3), и
            # умолчание у неё должно быть запрещающим.
            if not self._staging_unknown:
                self._staging_unknown = True
                log.warning("Остаток топлива ступени не читается — отделение "
                            "запрещено, пока не станет известно")
            return [True, False]
        self._staging_unknown = False
        return [True, (fuel.get("amount", 0.0) / top) <= self.SPENT_STAGE_FUEL]

    # ------------------------------------------------------------------
    def step(self, action: FlightAction):
        """Применяет действие и возвращает (obs, reward, done, info)."""
        if self.done or self.telemetry is None:
            return None, 0.0, True, {"reason": "эпизод завершён"}

        began = time.perf_counter()
        try:
            self.control.set_throttle(action.throttle)
            self.control.set_axes(pitch=action.pitch, yaw=action.yaw,
                                  roll=action.roll)
            if action.stage and self.action_mask()[1]:
                self.control.next_stage()
                self.last_stage_wall = time.time()
        except Exception as exc:
            log.warning("Команда не прошла: %s", exc)
            self.done = True
            return None, -5.0, True, {"reason": "связь потеряна"}

        # Ждём ОСТАТОК периода, а не полный период поверх работы. Раньше
        # здесь стоял фиксированный сон, а команды и телеметрия идут через
        # kRPC по сотне миллисекунд каждая — и заявленная «секунда» на
        # деле оказывалась секундой с четвертью. Сеть при этом уверена,
        # что мир отвечает ей ровно в срок.
        spent = time.perf_counter() - began
        if spent < self.control_dt:
            time.sleep(self.control_dt - spent)

        try:
            snap = self.telemetry.snapshot(detailed=True)
            self.last_snapshot = snap
        except Exception as exc:
            log.warning("Телеметрия оборвалась: %s", exc)
            self.done = True
            return None, -5.0, True, {"reason": "телеметрия потеряна"}

        reward, done, info = self.reward(snap)
        self.steps += 1
        self.episode_return += reward
        self.previous_apoapsis = snap.apoapsis
        self.previous_altitude = snap.altitude
        self.previous_energy = self.specific_energy(snap)
        self.previous_momentum = self.angular_momentum(snap)
        self.previous_roll = snap.roll
        self.peak_altitude = max(self.peak_altitude, snap.altitude)

        # Замер фактического ритма: если игра не поспевает за сетью, это
        # надо ВИДЕТЬ, а не догадываться по странному поведению аппарата
        period = time.perf_counter() - began
        self.period_sum += period
        self.period_count += 1
        self.period_worst = max(self.period_worst, period)
        self.done = done
        self.last_info = info
        return self.encode(snap), reward, done, info

    # ------------------------------------------------------------------
    def specific_energy(self, snap) -> float:
        """Удельная орбитальная энергия v²/2 − μ/r, Дж/кг."""
        r = max(1.0, self.body_radius + snap.altitude)
        return snap.speed * snap.speed / 2.0 - self.body_mu / r

    def target_energy(self) -> float:
        """Удельная энергия на целевой круговой орбите."""
        return -self.body_mu / (2.0 * (self.body_radius + self.goal_altitude))

    def energy_scale(self) -> float:
        """Насколько поднять энергию от старта до целевой орбиты."""
        start = -self.body_mu / self.body_radius
        return max(1.0, self.target_energy() - start)

    def angular_momentum(self, snap) -> float:
        """|r×v| по телеметрии: горизонтальная скорость на плече r."""
        r = max(1.0, self.body_radius + snap.altitude)
        return r * max(0.0, snap.horizontal_speed)

    def target_momentum(self) -> float:
        """Момент импульса на целевой круговой орбите: h = √(μ·r)."""
        return math.sqrt(self.body_mu * (self.body_radius + self.goal_altitude))

    def momentum_scale(self) -> float:
        """От стартового момента (аппарат крутится с планетой) до целевого."""
        return max(1.0, self.target_momentum() - self.initial_momentum)

    def reward(self, snap) -> tuple[float, bool, dict]:
        """Плотная награда: без неё PPO в такой длинной задаче не сдвинется.

        Формулы намеренно совпадают с тренажёрными (`sim_env.step`). Если
        в игре платить за другое, чем в тренажёре, то весь смысл учебной
        программы теряется: сеть придёт с политикой, оптимальной для
        одной функции награды, и окажется в мире с другой.
        """
        info: dict = {}
        reward = 0.0

        # Потолок на целевой энергии — как в тренажёре. Без него плотный
        # сигнал платит за уход прочь от планеты больше, чем за орбиту
        # (замер: +23.9 у вертикального выстрела против +12.0 у орбиты).
        target = self.target_energy()
        energy = min(self.specific_energy(snap), target)
        reward += (energy - min(self.previous_energy, target)) \
            / self.energy_scale() * 12.0

        # Момент импульса: то, чего у полёта строго вверх нет и быть не
        # может. Это и есть плотный сигнал «разворачивайся на горизонт».
        momentum_goal = self.target_momentum()
        momentum = min(self.angular_momentum(snap), momentum_goal)
        reward += (momentum - min(self.previous_momentum, momentum_goal)) \
            / self.momentum_scale() * 8.0

        gain = (min(snap.apoapsis, self.goal_altitude)
                - min(self.previous_apoapsis, self.goal_altitude))
        reward += gain / self.goal_altitude * 4.0

        if snap.dynamic_pressure > 30_000.0:
            reward -= (snap.dynamic_pressure - 30_000.0) / 30_000.0 * 0.05

        # нос ниже горизонта в плотных слоях — тяга в землю (как в тренажёре)
        if snap.altitude < 10_000.0 and snap.pitch < 0.0:
            reward -= min(1.0, -snap.pitch / 90.0) * 0.05

        # скольжение: штраф из Тренажёра+, те же 5° и те же коэффициенты.
        # Угол берётся полный — по тангажу И по курсу, как в тренажёре
        if snap.altitude < 15_000.0 and snap.dynamic_pressure > 1_000.0:
            excess = self._slip_angle(snap) - 5.0
            if excess > 0.0:
                load = min(1.0, snap.dynamic_pressure / 20_000.0)
                reward -= 0.25 * (excess / 10.0) * load

        # Раскрутка по крену — как в Тренажёре++. В игре крен работает
        # по-настоящему, и если за него не спрашивать, сеть придёт сюда с
        # произвольной командой по этой оси и будет вертеть аппарат.
        # Скорость крена считаем по разнице углов между шагами: отдельного
        # потока для неё нет, а шаг управления известен.
        roll_rate = abs(self._wrap180(snap.roll - self.previous_roll)) \
            / max(1e-3, self.control_dt)
        excess_roll = roll_rate - self.ROLL_RATE_FREE
        if excess_roll > 0.0:
            # веса те же, что в тренажёре, приведённые к шагу управления:
            # там штраф капает каждые 0.2 с, здесь — раз в секунду
            reward -= (self.ROLL_PENALTY * (excess_roll / 10.0)
                       * self.control_dt / 0.2)

        reward -= 0.002

        done = False
        situation = snap.situation
        atmosphere = 70_000.0
        try:
            body = self.telemetry.body
            atmosphere = body.atmosphere_depth if body.has_atmosphere else 0.0
        except Exception:
            pass

        # Разрушение подтверждается ДВАЖДЫ подряд. Одиночный ноль — это
        # почти всегда сбой чтения: ссылка на судно становится
        # недействительной после каждого отделения. Первый живой полёт
        # нейросети закончился «аппарат разрушен» через две секунды после
        # отстрела пусковых мачт, при том что ракета была цела.
        if snap.part_count <= 0:
            self.vanished += 1
        else:
            self.vanished = 0
        if self.vanished >= 2:
            reward -= 5.0
            done = True
            info["reason"] = "аппарат разрушен"
        elif situation in ("landed", "splashed") and snap.mission_time > 15:
            reward -= 5.0
            done = True
            info["reason"] = "падение"
        elif (snap.periapsis > atmosphere and snap.apoapsis > atmosphere
              and 0.0 <= snap.eccentricity < 1.0):
            reward += 50.0
            error = abs(snap.apoapsis - self.goal_altitude) / self.goal_altitude
            reward += max(0.0, 10.0 * (1.0 - error))
            self.reached_orbit = True
            done = True
            info["reason"] = "орбита достигнута"
        elif time.time() - self.started_wall > self.max_episode_seconds:
            reward -= 1.0
            done = True
            info["reason"] = "лимит времени эпизода"

        lf = snap.resources.get("LiquidFuel", {})
        if lf.get("max", 0) > 0 and lf.get("amount", 0) <= 0.1 \
                and snap.available_thrust < 1.0:
            if snap.apoapsis < self.goal_altitude * 0.9:
                reward -= 2.0
                done = True
                info["reason"] = "топливо кончилось"

        info.update({"altitude": snap.altitude, "apoapsis": snap.apoapsis,
                     "periapsis": snap.periapsis, "orbit": self.reached_orbit})
        return reward, done, info

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            if self.control is not None:
                self.control.full_stop()
            if self.telemetry is not None:
                self.telemetry.close()
        except Exception:
            pass
        self.telemetry = None
        self.control = None
        self.done = True

    def episode_summary(self) -> dict:
        summary = {
            "steps": self.steps,
            "return": round(self.episode_return, 2),
            "peak_altitude": round(self.peak_altitude),
            "orbit": self.reached_orbit,
            "reason": self.last_info.get("reason", "—"),
            "control_seconds": round(self.control_dt, 3),
        }
        if self.period_count:
            actual = self.period_sum / self.period_count
            summary["actual_seconds"] = round(actual, 3)
            summary["worst_seconds"] = round(self.period_worst, 3)
            # Отставание надо ВИДЕТЬ. Сеть уверена, что мир отвечает ей
            # ровно в срок; если игра не поспевает, аппарат ведёт себя
            # не так, как в тренажёре, и разбираться придётся вслепую.
            if actual > self.control_dt * 1.25:
                summary["ритм"] = (f"игра не поспевает: {actual:.2f} с "
                                   f"вместо {self.control_dt:.2f} с")
        return summary
