"""ТРЕНАЖЁР++ — третья ступень программы: математический клон KSP.

Последний рубеж перед живым запуском. Все скрытые механики Тренажёра+
остаются (задержка, таймаут ступеней, ветер, инерция тяги, батарея), но
физика перестаёт быть игрушечной:

* ПОЛНАЯ 3D-БАЛЛИСТИКА. Положение и скорость — трёхмерные векторы,
  планета вращается вокруг своей оси, атмосфера вращается вместе с ней.
  Появляется настоящее наклонение орбиты и азимут пуска.
* ИСТИННАЯ АТМОСФЕРА КЕРБИНА. Давление и температура берутся из
  высотных таблиц игры и интерполируются экспоненциально (давление) и
  линейно (температура), плотность считается уравнением состояния.
  Скорость звука — из температуры, сопротивление — по числу Маха с
  трансзвуковым горбом, как в стоковой кривой KSP.
* НАСТОЯЩЕЕ ПОВЕДЕНИЕ ДВИГАТЕЛЕЙ. Массовый расход постоянный, а тяга
  падает вместе с удельным импульсом в плотных слоях:
  F(p) = F_vac · Isp(p)/Isp_vac. Это ровно та формула, по которой
  считает игра.
* ЖИВОЙ ЦЕНТР МАСС. Аппарат больше не точка: у него есть длина,
  распределение масс по ступеням и центр давления. По мере выработки
  топлива центр масс ползёт, вместе с ним меняются момент инерции,
  плечо управления и запас устойчивости.
* УПРАВЛЕНИЕ МОМЕНТОМ, А НЕ УГЛОМ. Рули больше не задают скорость
  поворота напрямую: команда — это отклонение сопла и маховики, дальше
  работают моменты, инерция и аэродинамическая стабилизация. Именно так
  устроено управление в KSP при выключенном SAS — а нейросеть летает
  именно с выключенным SAS.

Таблицы и коэффициенты — описание мира, а не подсказка агенту: на вход
сети по-прежнему идут только сырые показания приборов.
"""
from __future__ import annotations

import math

from .sim_env import (BODY_DIAMETER, DRAG_MACH_KEYS, HEAT_RATIO,
                      KERBIN_ATMO_HEIGHT, KERBIN_MU, KERBIN_PRESSURE_KEYS,
                      KERBIN_RADIUS, KERBIN_SIDEREAL_DAY,
                      KERBIN_TEMPERATURE_KEYS, R_SPECIFIC, SEA_LEVEL_PA,
                      SimStage, _interpolate, _interpolate_log)
from .sim_env_plus import HardcoreAscentSimulator
from .spaces import FlightAction

# ==========================================================================
# Атмосфера, сопротивление и двигатели переехали в `sim_env.py` — теперь
# они общие для ВСЕХ ступеней. Пока таблицы жили только здесь, ранние
# ступени считали воздух экспонентой и тормозили аппарат в 2-7 раз
# сильнее: сеть училась вести одну ракету, а на третьей ступени получала
# другую. Здесь остаётся только то, чего на плоских ступенях нет.
# ==========================================================================
KERBIN_OMEGA = 2.0 * math.pi / KERBIN_SIDEREAL_DAY   # рад/с
G0 = 9.80665

FUEL_DENSITY = 1000.0         # кг/м³ смеси LiquidFuel+Oxidizer (как в игре)


# --- элементарная векторная алгебра (три компоненты, без numpy) ----------
def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _unit(a):
    n = _norm(a)
    return (0.0, 0.0, 0.0) if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def _wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


# ==========================================================================
class StageGeometry:
    """Где в аппарате что стоит: смещения от носа, метры (вниз — плюс)."""

    __slots__ = ("top", "length", "structure", "engine", "tank", "fin_area",
                 "gimbal", "nose_share")

    def __init__(self, top: float, length: float, nose_share: float,
                 fin_area: float, gimbal: float):
        self.top = top
        self.length = length
        self.structure = top + length * 0.45
        self.engine = top + length - 0.5
        self.tank = top + length * 0.45
        self.fin_area = fin_area
        self.gimbal = gimbal
        self.nose_share = nose_share      # доля сухой массы, вынесенная в нос


class PrecisionAscentSimulator(HardcoreAscentSimulator):
    """Высокоточная 3D-модель выведения: финальный «университет» перед KSP."""

    stage_id = 3
    title = "Тренажёр++"
    DEFAULT_DT = 0.1              # вращательная динамика требует мелкого шага

    # Маховики зондового ядра слабые: основной момент даёт качающееся
    # сопло, поэтому на выбеге с выключенным двигателем аппарат
    # разворачивается медленно — как в игре.
    RW_TORQUE = 600.0             # Н·м
    MAX_RATE = 45.0               # град/с, дальше конструкция не держит

    # ТРЕНИРОВОЧНЫЕ КОЛЁСА (см. `assist_rate`).
    # На Тренажёрах 1-2 команда тангажа задавала СКОРОСТЬ поворота, здесь
    # она задаёт МОМЕНТ. Это разные задачи: во втором случае угол —
    # двойной интеграл от команды, и политика, выученная на первом законе,
    # разваливается мгновенно. Первый прогон это и показал: 1 орбита на
    # 4106 эпизодов при 654 орбитах на прошлой ступени.
    # Поэтому на малой жёсткости здесь работает идеальный автопилот,
    # удерживающий заданную скорость поворота, — и его авторитет тает по
    # мере разогрева. К полной жёсткости остаётся чистое управление
    # моментом, как в игре с выключенным SAS.
    ASSIST_RATE = 25.0            # град/с при полной команде (как в Тренажёре+)

    # Нижняя граница помощи. 1.0 — колёса приварены навсегда: команда
    # ВСЕГДА задаёт скорость поворота, как на прошлых ступенях. 0.0 —
    # помощь тает вместе с разогревом и остаётся чистый момент.
    #
    # Ради этого числа физика KSP и управление моментом разведены по
    # разным ступеням. Живой прогон показал, почему это обязательно:
    # сеть, сдавшая Тренажёр+, обвалилась на Тренажёре++ за две минуты
    # (награда −42, 2 орбиты на 385 эпизодов) — причём ещё на полной
    # помощи, то есть развалило её не управление моментом, а всё
    # остальное сразу: три измерения, таблицы атмосферы, число Маха,
    # кривые двигателей, разрушение напором и втрое более частые решения.
    ASSIST_FLOOR = 0.0

    # Третье измерение тоже вводится постепенно, и это не прихоть.
    # На плоских ступенях ось рыскания не подключена НИ К ЧЕМУ: что бы
    # сеть туда ни выдавала, ничего не происходит, и градиента, который
    # научил бы её держать ноль, не возникает никогда. Замер на боевых
    # весах: политика, сдавшая Тренажёр+, держала рыскание на упоре
    # +0.98. В трёх измерениях это 24 град/с — за полёт курс уехал с 90°
    # на 187°, аппарат полетел боком, и 362 эпизода из 1483 кончились
    # опрокидыванием.
    # При YAW_FADE ось включается вместе с жёсткостью: сперва она мертва,
    # как на прошлых ступенях, и оживает по мере разогрева — тогда штраф
    # за скольжение успевает научить сеть убирать лишнюю команду.
    YAW_FADE = False

    # С какой горизонтальной скорости курс вообще что-то значит, м/с.
    #
    # История этой константы стоит того, чтобы её записать. Сначала вклад
    # ошибки курса в угол атаки гасился множителем `cos(тангаж)` — честная
    # геометрия: у вертикальной ракеты курс вырожден. Но из-за этого ось
    # рыскания у зенита оказывалась без последствий, и я поставил на этот
    # множитель пол 0.25, чтобы за неё всё-таки спрашивали.
    #
    # Это оказалось хуже болезни. Замер по шагам: ракета стоит на столе,
    # высота 2 метра, тангаж ровно 90°, скорость почти нулевая — а угол
    # атаки показывает 28°. Потому что у неподвижного аппарата «курс
    # потока» вычисляется из горизонтальной составляющей размером с шум,
    # то есть он случаен, а пол превращал этот случайный курс в
    # четверть настоящего угла. Сеть видела чудовищное скольжение на
    # старте, честно валила нос вниз — и падала на девятой секунде.
    #
    # Правильный признак не «насколько мы вертикальны», а «есть ли вообще
    # горизонтальное движение». Пока его нет, курс не значит ничего и в
    # угол атаки не входит; на развороте он входит полностью.
    HEADING_MEANINGFUL_SPEED = 50.0

    # --- крен ---
    # До сих пор ось крена была подключена к пустоте: в действиях она
    # есть, заряд тратит, а в физике не участвует ничем. Это ровно та
    # мёртвая ось, что однажды уже убила ступень через рыскание (5.3i):
    # градиента к ней нет, значение копится произвольное — а в живой KSP
    # крен работает по-настоящему, и сеть придёт туда с ним наперевес.
    # Теперь крен считается честно (маховики против демпфирования
    # потоком), виден сети через roll_sin/roll_cos и стоит награды, если
    # аппарат раскручивают.
    ROLL_TORQUE = 400.0           # Н·м, маховики зондового ядра по крену
    MAX_ROLL_RATE = 60.0          # град/с
    ROLL_DAMPING = 1.0
    # Здесь мёртвых осей нет: рыскание разворачивает аппарат по-настоящему
    # (и наказывается через скольжение), крен раскручивает его вокруг оси
    # (и наказывается ниже). Плата за «руль в пустоту» с плоских ступеней
    # тут не нужна и только мешала бы рулить.
    INERT_AXES: tuple[str, ...] = ()

    ROLL_RATE_FREE = 10.0         # град/с, до этого раскрутка не наказывается
    # За каждые 10 град/с сверх свободного, ЗА ШАГ. Число выглядит
    # крошечным, и это осознанно: штраф капает все 2100 шагов полёта, а
    # не короткое окно, как у скольжения. Замер: непрерывная раскрутка на
    # предельные 60 град/с стоит около 10 очков за полёт — заметно рядом
    # с +50 за орбиту, но не перевешивает саму задачу. При 0.02 (первая
    # прикидка) выходило −199, и это была бы уже не цена крена, а запрет
    # летать.
    ROLL_PENALTY = 0.001

    AERO_DAMPING = 1.0            # множитель аэродинамического демпфирования
    NORMAL_SLOPE = 2.0            # dCn/dα для тонкого тела, на радиан
    # BREAKUP_PRESSURE унаследован от базового тренажёра: разрушаться от
    # напора аппарат обязан одинаково на всех ступенях

    # ------------------------------------------------------------------
    # Геометрия и начальное состояние
    # ------------------------------------------------------------------
    def __init__(self, *args, assist_floor: float | None = None,
                 yaw_fade: bool | None = None, **kwargs):
        # Уровень помощи задаётся ступенью, а не жёстко классом: одна и та
        # же физика служит и «Тренажёру++» (помощь навсегда), и
        # «Тренажёру+++» (помощь тает).
        self.assist_floor = (self.ASSIST_FLOOR if assist_floor is None
                             else max(0.0, min(1.0, float(assist_floor))))
        self.yaw_fade = self.YAW_FADE if yaw_fade is None else bool(yaw_fade)
        super().__init__(*args, **kwargs)

    def random_rocket(self) -> list[SimStage]:
        stages = super().random_rocket()
        # разброс оперения: часть аппаратов почти нейтральна по
        # устойчивости, и удержать их можно только активным управлением
        self.fin_area = self.rng.uniform(2.0, 3.2)
        self.gimbal_first = self.rng.uniform(2.5, 4.0)
        self.gimbal_upper = self.rng.uniform(3.5, 5.0)
        return stages

    def build_geometry(self) -> None:
        """Раскладывает ступени вдоль корпуса: сверху — последняя, снизу — первая."""
        self.geometry: list[StageGeometry | None] = [None] * len(self.stages)
        offset = 0.0
        top_index = len(self.stages) - 1
        for index in reversed(range(len(self.stages))):
            stage = self.stages[index]
            # длина бака из объёма топлива плюс двигатель с переходником
            length = stage.capacity / (FUEL_DENSITY * self.body_area) + 1.5
            self.geometry[index] = StageGeometry(
                top=offset, length=length,
                nose_share=0.35 if index == top_index else 0.0,
                fin_area=self.fin_area if index == 0 else 0.0,
                gimbal=(self.gimbal_first if index == 0 else self.gimbal_upper))
            offset += length
        self.total_length = offset

    def on_reset(self) -> None:
        super().on_reset()
        self.body_area = math.pi * (BODY_DIAMETER / 2.0) ** 2
        self.build_geometry()

        # --- трёхмерное состояние ---
        latitude = math.radians(self.rng.uniform(-0.5, 0.5))   # широта КЦК
        self.pos = (KERBIN_RADIUS * math.cos(latitude), 0.0,
                    KERBIN_RADIUS * math.sin(latitude))
        self.vel = self.surface_velocity(self.pos)
        self.pitch = 90.0
        self.heading = 90.0                 # пуск на восток: вращение помогает
        self.roll = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        self.roll_rate = 0.0
        self.max_roll_rate = 0.0
        self.wind_lateral = 0.0
        self.control = (0.0, 0.0, 0.0)
        self.com = 0.0
        self.cop = 0.0
        self.static_margin = 0.0
        self.min_margin = 99.0
        self.max_mach = 0.0
        self.max_q = 0.0
        # состояние аэродинамики текущего шага: скоростной напор, воздушная
        # скорость, нормальная сила и эталонная площадь
        self.aero = (0.0, 0.0, 0.0, self.body_area)

    # ------------------------------------------------------------------
    def surface_velocity(self, pos) -> tuple[float, float, float]:
        """Скорость точки поверхности: планета и атмосфера вращаются вместе."""
        return _cross((0.0, 0.0, KERBIN_OMEGA), pos)

    def _randomize_airborne_start(self) -> None:
        """Случайное состояние в середине выведения — в трёх измерениях."""
        altitude = self.rng.uniform(500.0, 45_000.0)
        speed = self.rng.uniform(80.0, 2_000.0)
        flight_angle = math.radians(self.rng.uniform(5.0, 88.0))
        self.pitch = max(-90.0, min(90.0, math.degrees(flight_angle)
                                    + self.rng.uniform(-8.0, 8.0)))
        self.heading = 90.0 + self.rng.uniform(-10.0, 10.0)

        latitude = math.radians(self.rng.uniform(-1.0, 1.0))
        r = KERBIN_RADIUS + altitude
        self.pos = (r * math.cos(latitude), 0.0, r * math.sin(latitude))
        up, north, east = self.local_frame()
        horizontal = _add(_scale(north, math.cos(flight_angle)
                                 * math.cos(math.radians(self.heading))),
                          _scale(east, math.cos(flight_angle)
                                 * math.sin(math.radians(self.heading))))
        self.vel = _add(self.surface_velocity(self.pos),
                        _add(_scale(up, speed * math.sin(flight_angle)),
                             _scale(horizontal, speed)))
        self.airborne = True
        self.peak_altitude = altitude

        burned = self.rng.uniform(0.0, 0.7)
        first = self.stages[0]
        first.fuel_mass *= (1.0 - burned)
        if burned > 0.55 and len(self.stages) > 1:
            self.stage_index = 1
        self.time = self.rng.uniform(0.0, 60.0)

    # ------------------------------------------------------------------
    # Примитивы состояния в трёх измерениях
    # ------------------------------------------------------------------
    def _radius(self) -> float:
        return _norm(self.pos)

    def _speed(self) -> float:
        return _norm(self.vel)

    def _vertical_speed(self) -> float:
        return _dot(self.vel, _unit(self.pos))

    def _angular_momentum(self) -> float:
        return _norm(_cross(self.pos, self.vel))

    def _inclination(self) -> float:
        h = _cross(self.pos, self.vel)
        magnitude = _norm(h)
        if magnitude < 1e-6:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, h[2] / magnitude))))

    def local_frame(self):
        """Местная тройка: вверх, север, восток."""
        up = _unit(self.pos)
        polar = (0.0, 0.0, 1.0)
        north = _sub(polar, _scale(up, _dot(polar, up)))
        if _norm(north) < 1e-9:                 # ровно над полюсом
            north = _unit(_cross(up, (1.0, 0.0, 0.0)))
        else:
            north = _unit(north)
        east = _unit(_cross(north, up))
        return up, north, east

    def nose_vector(self):
        up, north, east = self.local_frame()
        pitch = math.radians(self.pitch)
        heading = math.radians(self.heading)
        horizontal = _add(_scale(north, math.cos(heading)),
                          _scale(east, math.sin(heading)))
        return _add(_scale(up, math.sin(pitch)),
                    _scale(horizontal, math.cos(pitch)))

    def air_velocity(self):
        """Скорость относительно вращающейся атмосферы."""
        return _sub(self.vel, self.surface_velocity(self.pos))

    def air_speed(self) -> float:
        return _norm(self.air_velocity())

    # Атмосфера (таблицы давления и температуры), число Маха, кривая Cd и
    # формула тяги F = F_vac · Isp(p)/Isp_vac — всё унаследовано от
    # базового тренажёра: мир на всех ступенях обязан быть одним.

    # ------------------------------------------------------------------
    # Массы, центр масс, момент инерции, центр давления
    # ------------------------------------------------------------------
    def mass_blocks(self):
        """[(смещение от носа, масса)] для всех оставшихся ступеней."""
        blocks = []
        for index in range(self.stage_index, len(self.stages)):
            stage = self.stages[index]
            geometry = self.geometry[index]
            dry = stage.dry_mass
            nose = dry * geometry.nose_share
            if nose > 0.0:
                blocks.append((geometry.top + 0.4, nose))
            rest = dry - nose
            blocks.append((geometry.engine, rest * 0.6))
            blocks.append((geometry.structure, rest * 0.4))
            if stage.fuel_mass > 0.0:
                blocks.append((geometry.tank, stage.fuel_mass))
        return blocks

    def mass_properties(self) -> tuple[float, float, float]:
        """(масса, центр масс от носа, момент инерции вокруг поперечной оси)."""
        blocks = self.mass_blocks()
        total = sum(m for _, m in blocks)
        if total <= 0.0:
            return 1.0, 0.0, 1.0
        com = sum(x * m for x, m in blocks) / total
        inertia = sum(m * (x - com) ** 2 for x, m in blocks)
        return total, com, max(1.0, inertia)

    def pressure_centre(self) -> tuple[float, float]:
        """(центр давления от носа, эталонная площадь).

        Нос дестабилизирует (подъёмная сила впереди центра масс),
        корпус почти нейтрален, оперение стабилизирует. Оперение стоит
        на первой ступени — вместе с ней и уходит.
        """
        span = 0.0
        fins = 0.0
        for index in range(self.stage_index, len(self.stages)):
            geometry = self.geometry[index]
            span = max(span, geometry.top + geometry.length)
            fins += geometry.fin_area
        nose_area = 0.5 * self.body_area
        body_area = 0.25 * self.body_area
        total = nose_area + body_area + fins
        centre = (nose_area * 0.3 + body_area * span * 0.5
                  + fins * max(0.0, span - 0.4)) / max(1e-6, total)
        return centre, total

    # ------------------------------------------------------------------
    def angle_of_attack(self) -> float:
        pitch_error, yaw_error = self.attitude_error()
        return math.hypot(pitch_error, yaw_error)

    def attitude_error(self) -> tuple[float, float]:
        """Отклонение носа от вектора движения: (по тангажу, по рысканию)."""
        air = self.air_velocity()
        speed = _norm(air)
        if speed < 1.0:
            return 0.0, 0.0
        up, north, east = self.local_frame()
        vertical = _dot(air, up)
        flight_pitch = math.degrees(math.asin(max(-1.0, min(1.0, vertical / speed))))
        pitch_error = self.pitch - flight_pitch

        # Курс потока имеет смысл ровно настолько, насколько есть само
        # горизонтальное движение. У стоящей на столе ракеты он случаен —
        # см. HEADING_MEANINGFUL_SPEED.
        north_speed, east_speed = _dot(air, north), _dot(air, east)
        horizontal = math.hypot(north_speed, east_speed)
        weight = min(1.0, horizontal / self.HEADING_MEANINGFUL_SPEED)
        if weight <= 0.0:
            return pitch_error, 0.0
        flight_heading = math.degrees(math.atan2(east_speed, north_speed))
        yaw_error = _wrap180(self.heading - flight_heading) * weight
        return pitch_error, yaw_error

    # ------------------------------------------------------------------
    def apply_command(self, command: FlightAction) -> None:
        """Дроссель и рули. Углы здесь не меняются — их двигают моменты."""
        self.fire_failure()
        if command.stage and self.can_separate():
            self.separate()
        elif command.stage and self.abort_available():
            # сбрасывать нечего, ракета сломана — кнопка уводит кабину
            # (см. HardcoreAscentSimulator.abort_capsule)
            self.abort_capsule()
        self.throttle = max(0.0, min(1.0, command.throttle))
        self.control = (command.pitch, command.yaw, command.roll)
        self.spend_electricity(command)
        self.blow_wind()
        self.spool_engine()
        self.leak_fuel()
        self.update_chute()

    def blow_wind(self) -> None:
        """Порыв сносит нос по обеим осям."""
        if not self.airborne:
            return
        shift = self.gust()
        self.wind_offset += shift
        self.pitch = max(-90.0, min(90.0, self.pitch + shift))
        self.wind_lateral, sideways = self.ou_gust(self.wind_lateral)
        self.heading = (self.heading + sideways * 0.6) % 360.0

    # ------------------------------------------------------------------
    def integrate(self) -> None:
        """Шаг физики: тяга, гравитация, аэродинамика и вращение."""
        altitude = self.altitude
        pressure = self.pressure_atm(altitude)
        rho = self.density(altitude)
        stage = self.current_stage

        thrust = 0.0
        if stage is not None:
            thrust_max, isp = self.thrust_and_isp(stage, pressure)
            thrust = self.burn(thrust_max, isp)

        mass, com, inertia = self.mass_properties()
        self.com = com
        cop, reference_area = self.pressure_centre()
        self.cop = cop
        self.static_margin = cop - com
        self.min_margin = min(self.min_margin, self.static_margin)

        up = _unit(self.pos)
        nose = self.nose_vector()

        # --- силы ---
        force = _scale(nose, thrust)
        force = _add(force, _scale(up, -KERBIN_MU * mass / (_norm(self.pos) ** 2)))

        air = self.air_velocity()
        air_speed = _norm(air)
        q = 0.5 * rho * air_speed * air_speed
        self.max_q = max(self.max_q, q)
        normal_force = 0.0

        if rho > 0.0 and air_speed > 1.0:
            mach = air_speed / self.speed_of_sound(altitude)
            self.max_mach = max(self.max_mach, mach)
            drag = q * self.drag_coefficient(mach) * self.body_area
            # см. AscentSimulator.integrate: потолок импульса сопротивления
            drag = min(drag, mass * air_speed / self.dt)
            force = _add(force, _scale(_unit(air), -drag))

            # нормальная сила: тело под углом атаки работает как крыло
            alpha = math.radians(min(60.0, self.angle_of_attack()))
            if alpha > 1e-4:
                normal_force = (q * reference_area * self.NORMAL_SLOPE
                                * math.sin(alpha) * math.cos(alpha))
                lateral = _sub(nose, _scale(_unit(air), _dot(nose, _unit(air))))
                if _norm(lateral) > 1e-6:
                    force = _add(force, _scale(_unit(lateral), normal_force))

        self.aero = (q, air_speed, normal_force, reference_area)
        self.rotate(thrust, inertia)

        acceleration = _scale(force, 1.0 / max(1.0, mass))
        self.vel = _add(self.vel, _scale(acceleration, self.dt))
        self.pos = _add(self.pos, _scale(self.vel, self.dt))
        self.time += self.dt

    # ------------------------------------------------------------------
    def rotate(self, thrust: float, inertia: float) -> None:
        """Вращательная динамика: моменты, инерция, демпфирование.

        Три источника момента:
          * управление  — отклонение сопла плюс маховики;
          * аэродинамика — нормальная сила на плече «центр давления минус
            центр масс»; при положительном плече она возвращает нос к
            вектору движения, при отрицательном — опрокидывает;
          * демпфирование — сопротивление вращению набегающим потоком.
        """
        q, air_speed, normal_force, reference_area = self.aero
        geometry = self.geometry[min(self.stage_index, len(self.geometry) - 1)]
        arm = max(0.3, geometry.engine - self.com)
        gimbal = math.radians(geometry.gimbal)
        pitch_cmd, yaw_cmd, _ = self.control

        control_pitch = thrust * math.sin(gimbal * pitch_cmd) * arm \
            + self.RW_TORQUE * pitch_cmd
        control_yaw = thrust * math.sin(gimbal * yaw_cmd) * arm \
            + self.RW_TORQUE * yaw_cmd

        pitch_error, yaw_error = self.attitude_error()
        total_aoa = math.hypot(pitch_error, yaw_error)
        aero_pitch = aero_yaw = 0.0
        if total_aoa > 1e-6 and normal_force > 0.0:
            restoring = normal_force * self.static_margin
            aero_pitch = -restoring * pitch_error / total_aoa
            aero_yaw = -restoring * yaw_error / total_aoa

        # Демпфирование: оперение на плече `l` от центра масс при вращении
        # с угловой скоростью ω видит добавочный угол атаки ω·l/V, и его
        # нормальная сила гасит это вращение.
        damping = 0.0
        if air_speed > 1.0 and q > 0.0:
            lever = max(0.5, abs(self.static_margin))
            damping = (self.AERO_DAMPING * q * reference_area
                       * self.NORMAL_SLOPE * lever * lever / air_speed)

        for axis in ("pitch", "yaw"):
            if axis == "pitch":
                torque = control_pitch + aero_pitch
                rate = self.pitch_rate
            else:
                torque = control_yaw + aero_yaw
                rate = self.yaw_rate
            torque -= damping * math.radians(rate)
            rate += math.degrees(torque / inertia) * self.dt
            rate = max(-self.MAX_RATE, min(self.MAX_RATE, rate))
            if axis == "pitch":
                self.pitch_rate = rate
            else:
                self.yaw_rate = rate

        self.assist_rate(pitch_cmd, yaw_cmd)
        self.spin_roll(q, air_speed, reference_area)

        self.pitch = max(-90.0, min(90.0, self.pitch + self.pitch_rate * self.dt))
        if abs(self.pitch) >= 89.999:
            self.pitch_rate = 0.0
        # Вращение вокруг СОБСТВЕННОЙ оси рыскания раскладывается на смену
        # курса и крен — по тангажу. У горизонтально летящего аппарата оно
        # целиком меняет курс; у стоящего вертикально курс почти не
        # меняется, а вращение уходит в крен. Это и есть честная
        # кинематика, и она заодно снимает две мины разом.
        #
        # Раньше здесь стояло усиление 1/cos(тангаж): у зенита команда −1
        # крутила азимут на 100 град/с — полный оборот за четыре секунды —
        # при том что нос в пространстве почти не двигался. Число
        # крутилось, физики за ним не было, а на развороте накопленный
        # мусорный курс убивал полёт.
        #
        # Теперь то же самое вращение видно там, где оно и происходит: в
        # крене. А крен раскрутку наказывает (ROLL_PENALTY), поэтому руль
        # в пустоту перестал быть бесплатным и у вертикальной ракеты.
        pitch_rad = math.radians(self.pitch)
        self.heading = (self.heading
                        + self.yaw_rate * math.cos(pitch_rad) * self.dt) % 360.0
        self.roll_rate += self.yaw_rate * math.sin(pitch_rad)
        self.roll_rate = max(-self.MAX_ROLL_RATE,
                             min(self.MAX_ROLL_RATE, self.roll_rate))

    # ------------------------------------------------------------------
    def spin_roll(self, q: float, air_speed: float,
                  reference_area: float) -> None:
        """Крен: маховики раскручивают аппарат, поток его успокаивает.

        Момент инерции вокруг ПРОДОЛЬНОЙ оси у длинного тонкого корпуса
        мал (m·r²/2 против m·L²/12 поперёк), поэтому слабые маховики
        раскручивают ракету по крену куда живее, чем по тангажу. Ровно
        так и в игре: аппарат с выключенным SAS от одной команды крена
        начинает медленно вращаться и сам не останавливается.
        """
        radius = BODY_DIAMETER / 2.0
        inertia = max(1.0, self.mass * radius * radius / 2.0)
        torque = self.ROLL_TORQUE * self.control[2]
        if air_speed > 1.0 and q > 0.0:
            # оперение сопротивляется вращению: добавочный угол атаки
            # лопасти на плече r при угловой скорости ω равен ω·r/V
            damping = (self.ROLL_DAMPING * q * reference_area
                       * self.NORMAL_SLOPE * radius * radius / air_speed)
            torque -= damping * math.radians(self.roll_rate)
        self.roll_rate += math.degrees(torque / inertia) * self.dt
        self.roll_rate = max(-self.MAX_ROLL_RATE,
                             min(self.MAX_ROLL_RATE, self.roll_rate))
        self.max_roll_rate = max(self.max_roll_rate, abs(self.roll_rate))
        self.roll = (self.roll + self.roll_rate * self.dt) % 360.0

    def shaping_reward(self, command: FlightAction) -> float:
        """Штрафы Тренажёра+ плюс цена раскрутки по крену.

        Зачем крену штраф. Для осесимметричной ракеты крен сам по себе
        почти безвреден — и именно поэтому ось остаётся мёртвой: команда
        ни на что не влияет, градиента нет, значение уползает на упор и
        выстреливает в живой игре. Цена делается явной: раскрутка сверх
        десяти градусов в секунду — это болтанка конструкции, лишний
        расход заряда и потерянное наведение. Штраф вводится вместе с
        жёсткостью, как и всё остальное на этой ступени.
        """
        penalty = super().shaping_reward(command)
        excess = abs(self.roll_rate) - self.ROLL_RATE_FREE
        if excess > 0.0:
            penalty -= self.ROLL_PENALTY * (excess / 10.0) * self.difficulty
        return penalty

    # ------------------------------------------------------------------
    def yaw_authority(self) -> float:
        """Насколько сейчас работает ось рыскания (см. YAW_FADE)."""
        return self.difficulty if self.yaw_fade else 1.0

    def assist_rate(self, pitch_cmd: float, yaw_cmd: float) -> None:
        """Тренировочные колёса: перевод с языка углов на язык моментов.

        При жёсткости 0 команда полностью задаёт скорость поворота — сеть
        оказывается в привычном ей мире Тренажёра+. При жёсткости 1
        помощь исчезает, и остаётся честная вращательная динамика. Между
        ними — сервопривод с частичным авторитетом; ровно так ведёт себя
        SAS в игре, когда ему не хватает момента.
        """
        assist = max(self.assist_floor, 1.0 - self.difficulty)
        if assist <= 0.0:
            return
        target_pitch = max(-1.0, min(1.0, pitch_cmd)) * self.ASSIST_RATE
        target_yaw = (max(-1.0, min(1.0, yaw_cmd)) * self.ASSIST_RATE
                      * self.yaw_authority())
        self.pitch_rate += (target_pitch - self.pitch_rate) * assist
        self.yaw_rate += (target_yaw - self.yaw_rate) * assist

    # ------------------------------------------------------------------
    def ground_contact(self) -> float:
        if self.altitude >= 0.0:
            if self.altitude > 5.0:
                self.airborne = True
            return 0.0

        up = _unit(self.pos)
        radial = _dot(self.vel, up)
        impact = abs(radial)
        self.pos = _scale(up, KERBIN_RADIUS)
        if radial < 0.0:
            self.vel = _sub(self.vel, _scale(up, radial))
        if not self.airborne:
            # стоим на столе: держимся вместе с поверхностью и не заваливаемся
            self.vel = self.surface_velocity(self.pos)
            self.pitch_rate = 0.0
            self.yaw_rate = 0.0
            self.roll_rate = 0.0        # на столе аппарат держат опоры
            self.pitch = max(self.pitch, 0.0)
        return impact

    # ------------------------------------------------------------------
    # Ни `check_failure`, ни `dynamic_pressure` здесь не переопределяются:
    # разрушение напором переехало в базовый тренажёр (мир один),
    # опрокидывание живёт в Тренажёре+, а формула напора наследуется и
    # сама берёт трёхмерную скорость относительно воздуха. Дубликат
    # формулы — это будущее расхождение, поэтому его тут нет.

    # ------------------------------------------------------------------
    def episode_summary(self) -> dict:
        summary = super().episode_summary()
        summary.update({
            "inclination": round(self._inclination(), 2),
            "max_mach": round(self.max_mach, 2),
            "max_q": round(self.max_q),
            "min_margin": round(self.min_margin, 2),
            "com": round(self.com, 2),
            "heading": round(self.heading, 1),
            "max_roll_rate": round(self.max_roll_rate, 1),
        })
        return summary


__all__ = ["PrecisionAscentSimulator", "KERBIN_PRESSURE_KEYS",
           "KERBIN_TEMPERATURE_KEYS", "DRAG_MACH_KEYS"]
