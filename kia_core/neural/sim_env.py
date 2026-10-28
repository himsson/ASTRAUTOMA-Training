"""Базовый тренажёр выведения — первая ступень учебной программы.

Зачем он нужен. PPO — алгоритм, требовательный к числу шагов: для
устойчивого управления ему нужны сотни тысяч, а то и миллионы
взаимодействий со средой. Один запуск в KSP идёт в реальном времени и
даёт около 6000 шагов за 5 минут; миллион шагов — это примерно две
недели непрерывных полётов. Тренажёр гоняет те же самые наблюдения и
действия в десятки тысяч раз быстрее, поэтому сеть приходит в KSP уже
не полностью случайной.

Физика здесь — это ПРАВИЛА МИРА, а не подсказка агенту: сеть по-прежнему
видит только сырые числа и не получает ни одной формулы. Симулятор
намеренно грубый (точечная масса, экспоненциальная атмосфера), поэтому
после него идут ещё два этапа: `sim_env_plus` (скрытые механики игры) и
`sim_env_pro` (высокоточный математический клон KSP).

УСТРОЙСТВО ШАГА. Чтобы следующие ступени программы не копировали код,
шаг разбит на переопределяемые части:

    filter_command(action)   что реально дошло до аппарата (задержка связи)
    apply_command(command)   отделение ступени, газ, рули
    integrate()              тяга, гравитация, сопротивление
    ground_contact()         опора на грунт
    shaping_reward(command)  дополнительные штрафы этапа
    check_failure()          дополнительные причины конца эпизода

Позиция хранится в плоскости (x, y). Всё, что зависит от размерности,
спрятано в маленькие методы (`_radius`, `_vertical_speed`,
`_angular_momentum`, `_inclination`), поэтому трёхмерный клон в
`sim_env_pro` переопределяет их, а формулы орбиты и наград наследует.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .spaces import FlightAction, encode_flight_state

# Параметры Кербина (из справочника игры — это описание мира, не знание агента)
KERBIN_MU = 3.5316e12
KERBIN_RADIUS = 600_000.0
KERBIN_ATMO_HEIGHT = 70_000.0
KERBIN_SCALE_HEIGHT = 5_600.0
KERBIN_SEA_DENSITY = 1.225
KERBIN_ROTATION_SPEED = 174.5      # м/с на экваторе
KERBIN_SIDEREAL_DAY = 21_549.425   # с, полный оборот планеты

# Ниже этого остатка ступень считается выработанной и её можно сбрасывать.
# Отделение полной ступени — не запрет физики, а грубая ошибка пилотирования:
# именно она убивала полёты на прогоне 4.5 млн шагов (см. MIGRATION 7.3).
SPENT_STAGE_FUEL = 0.03

# ==========================================================================
# АТМОСФЕРА И СОПРОТИВЛЕНИЕ — ОДНИ И ТЕ ЖЕ НА ВСЕХ СТУПЕНЯХ
# ==========================================================================
# Раньше базовый тренажёр считал воздух экспонентой и брал сопротивление
# как «площадь от массы при Cd = 1», а Тренажёр++ — по настоящим таблицам
# Кербина с числом Маха. Разница выглядела мелкой правкой, а на деле
# означала РАЗНЫЙ АППАРАТ: замер при одинаковой массе и скорости показал
# силу сопротивления в 2-7 раз больше на ранних ступенях.
#
#     1 км, 150 м/с     9.7 кН против  2.5 кН   (×3.8)
#    12 км, 700 м/с    29.7 кН против 12.6 кН   (×2.4)
#    35 км, 2000 м/с    4.0 кН против  0.5 кН   (×7.4)
#
# Сеть училась вести тормозящуюся ракету, а на третьей ступени получала
# вчетверо более скользкую. Профиль выведения — когда газовать и как
# быстро заваливать нос — переставал подходить, и политика разваливалась
# за тысячу эпизодов. Ровно это четыре прогона подряд читалось в логе как
# «разрушен скоростным напором», «опрокинуло», «падение».
#
# Поэтому мир теперь один. Ступени по-прежнему различаются механиками
# (задержка, ветер, батарея, три измерения, момент), но ВОЗДУХ И
# СОПРОТИВЛЕНИЕ у них общие — и это описание мира, а не подсказка агенту.

# Давление, кПа — ключи высотной кривой игры. Между узлами
# интерполируется экспоненциально: на этих участках атмосфера и правда
# близка к изотермической.
KERBIN_PRESSURE_KEYS: list[tuple[float, float]] = [
    (0.0, 101.325), (1000.0, 84.02916), (2000.0, 69.68138),
    (3000.0, 57.78463), (4000.0, 47.75419), (5000.0, 39.27772),
    (6000.0, 32.15662), (7000.0, 26.20419), (8000.0, 21.22641),
    (9000.0, 17.10102), (10000.0, 13.72816), (15000.0, 4.457409),
    (20000.0, 1.509558), (25000.0, 0.5303647), (30000.0, 0.1962376),
    (40000.0, 0.02979127), (50000.0, 0.005205474),
    (60000.0, 0.0009887569), (70000.0, 0.0),
]

# Температура, К. Профиль как у земной стандартной атмосферы: тропосфера
# остывает, стратосфера греется озоном, мезосфера снова стынет.
KERBIN_TEMPERATURE_KEYS: list[tuple[float, float]] = [
    (0.0, 288.15), (2500.0, 268.65), (6000.0, 250.65), (8815.0, 216.65),
    (16050.0, 216.65), (25729.0, 228.65), (37879.0, 270.65),
    (57440.0, 270.65), (70000.0, 186.95),
]

# Коэффициент сопротивления от числа Маха: околозвуковой горб — главная
# причина, по которой в KSP нельзя разгоняться у земли.
DRAG_MACH_KEYS: list[tuple[float, float]] = [
    (0.0, 0.16), (0.85, 0.19), (1.0, 0.42), (1.2, 0.45), (2.0, 0.32),
    (3.0, 0.26), (5.0, 0.22), (10.0, 0.20),
]

R_SPECIFIC = 287.053          # Дж/(кг·К), газовая постоянная воздуха
HEAT_RATIO = 1.4              # показатель адиабаты
SEA_LEVEL_PA = 101_325.0
BODY_DIAMETER = 1.25          # м, стандартный диаметр стока


def _interpolate(keys: list[tuple[float, float]], x: float) -> float:
    """Линейная интерполяция по узлам таблицы."""
    if x <= keys[0][0]:
        return keys[0][1]
    if x >= keys[-1][0]:
        return keys[-1][1]
    low, high = 0, len(keys) - 1
    while high - low > 1:                     # двоичный поиск: таблица мала,
        mid = (low + high) // 2               # но шагов симуляции миллионы
        if keys[mid][0] <= x:
            low = mid
        else:
            high = mid
    x0, y0 = keys[low]
    x1, y1 = keys[high]
    span = x1 - x0
    return y0 if span <= 0 else y0 + (y1 - y0) * (x - x0) / span


def _interpolate_log(keys: list[tuple[float, float]], x: float) -> float:
    """Экспоненциальная интерполяция: линейная по логарифму значения."""
    if x <= keys[0][0]:
        return keys[0][1]
    if x >= keys[-1][0]:
        return keys[-1][1]
    low, high = 0, len(keys) - 1
    while high - low > 1:
        mid = (low + high) // 2
        if keys[mid][0] <= x:
            low = mid
        else:
            high = mid
    x0, y0 = keys[low]
    x1, y1 = keys[high]
    if y0 <= 0.0 or y1 <= 0.0:
        span = x1 - x0
        return y0 if span <= 0 else y0 + (y1 - y0) * (x - x0) / span
    t = (x - x0) / (x1 - x0)
    return y0 * math.exp(t * math.log(y1 / y0))


@dataclass
class SimStage:
    dry_mass: float          # кг
    fuel_mass: float         # кг
    thrust_vac: float        # Н
    thrust_asl: float        # Н
    isp_vac: float
    isp_asl: float
    capacity: float = 0.0    # исходный запас топлива, кг (для доли остатка)

    @property
    def total(self) -> float:
        return self.dry_mass + self.fuel_mass

    @property
    def fuel_fraction(self) -> float:
        top = self.capacity or self.fuel_mass
        return self.fuel_mass / top if top > 0 else 0.0


@dataclass
class SimSnapshot:
    """Двойник TelemetrySnapshot: те же поля, что читает кодировщик."""
    altitude: float = 0.0
    surface_altitude: float = 0.0
    apoapsis: float = 0.0
    periapsis: float = 0.0
    time_to_apoapsis: float = 0.0
    eccentricity: float = 0.0
    inclination: float = 0.0
    speed: float = 0.0
    vertical_speed: float = 0.0
    horizontal_speed: float = 0.0
    pitch: float = 90.0
    heading: float = 90.0
    roll: float = 0.0
    dynamic_pressure: float = 0.0
    static_pressure: float = 101_325.0
    atmosphere_density: float = 1.225
    g_force: float = 1.0
    mass: float = 0.0            # тонны (как в TelemetrySnapshot)
    dry_mass: float = 0.0
    thrust: float = 0.0          # кН
    available_thrust: float = 0.0
    specific_impulse: float = 0.0
    throttle: float = 0.0
    stage: int = 0
    part_count: int = 0
    mission_time: float = 0.0
    situation: str = "pre_launch"
    body: str = "Kerbin"
    crew_count: int = 0
    resources: dict = field(default_factory=dict)
    stage_resources: dict = field(default_factory=dict)


class AscentSimulator:
    """Точечная масса с тягой, гравитацией и сопротивлением воздуха."""

    stage_id = 1
    title = "Базовый Тренажёр"
    DEFAULT_DT = 0.2

    # Мидель корпуса: тот же, что у трёхмерного клона. Аппарат на всех
    # ступенях должен быть одной и той же ракетой, иначе профиль
    # выведения, выученный на одной, не годится на другой.
    body_area = math.pi * (BODY_DIAMETER / 2.0) ** 2

    # Па, выше — конструкция не выдерживает. Тоже общий для всех ступеней.
    BREAKUP_PRESSURE = 45_000.0

    # Секунд на то, чтобы оторваться от стола. Дальше эпизод обрывается.
    #
    # Это не придирка, а лечение самой дорогой ловушки обучения. Пока
    # неудачный старт длился все 420 секунд, каждая такая попытка сыпала в
    # буфер 2100 шагов «стояния на столе» с почти нулевой наградой. Они
    # заваливали данные, критик убеждался, что стартовое состояние ничего
    # не стоит, и политика уверенно училась НЕ ВЗЛЕТАТЬ.
    #
    # Замер, который это вскрыл (ступень 1, обученная политика):
    #
    #     со стола   0 орбит из 20, средний газ 0.27
    #     в воздухе 15 орбит из 20, средний газ 0.95
    #
    # Сеть отлично летала и совершенно не умела стартовать — на столе она
    # осознанно держала газ ниже порога отрыва.
    #
    # Приём известный: обрывать эпизод сразу после провала и штрафовать в
    # точке обрыва, чтобы «шум после провала» не попадал в градиент
    # политики (ESPO, arXiv 2605.29860; та же практика в работах по
    # посадке ступеней). Ракета с TWR 1.15 отрывается за секунду, так что
    # десять — это огромный запас, а не строгость.
    LIFTOFF_DEADLINE = 10.0
    LIFTOFF_PENALTY = 3.0

    # Жёсткость среды: 0 — щадящий режим, 1 — полная. Тренер поднимает её
    # по ходу прогона (см. `Stage.warmup`). Значение живёт на классе, а не
    # ставится в `on_reset`, — иначе оно сбрасывалось бы каждым эпизодом.
    difficulty = 1.0

    def __init__(self, goal_altitude: float = 80_000.0, dt: float | None = None,
                 max_time: float = 420.0, seed: int | None = None,
                 exploring_starts: float = 0.5):
        """exploring_starts — доля эпизодов, начинающихся уже в полёте.

        Классическая проблема: пока ракета стоит на столе, награда
        одинакова для любой тяги ниже той, что оторвёт аппарат от земли.
        Поверхность награды плоская, градиента нет, и PPO не за что
        зацепиться. Если часть эпизодов начинать в воздухе — там градиент
        есть, и оценка ценности постепенно доходит назад до стартового
        стола. Приём стандартный (exploring starts), физика при этом
        не меняется.
        """
        self.goal_altitude = goal_altitude
        self.dt = self.DEFAULT_DT if dt is None else dt
        self.max_time = max_time
        self.exploring_starts = exploring_starts
        self.rng = random.Random(seed)
        self.stages: list[SimStage] = []
        self.step_info: dict = {}
        self.reset()

    # ------------------------------------------------------------------
    def random_rocket(self) -> list[SimStage]:
        """Случайная, но летающая двухступенчатая связка.

        Разброс параметров заставляет сеть учиться управлять РАЗНЫМИ
        аппаратами, а не запоминать один-единственный профиль.
        """
        # Диапазоны подобраны так, чтобы суммарный ΔV был примерно
        # 4200-5600 м/с: хватает на орбиту с запасом, но не позволяет
        # уйти на гиперболу простым «жать газ до упора».
        payload = self.rng.uniform(300.0, 700.0)
        second = SimStage(
            dry_mass=self.rng.uniform(500.0, 800.0) + payload,
            fuel_mass=self.rng.uniform(1400.0, 2000.0),
            thrust_vac=self.rng.uniform(25_000.0, 50_000.0),
            thrust_asl=0.0, isp_vac=345.0, isp_asl=180.0)
        second.thrust_asl = second.thrust_vac * 0.4
        first = SimStage(
            dry_mass=self.rng.uniform(1600.0, 2400.0),
            fuel_mass=self.rng.uniform(5000.0, 7000.0),
            thrust_vac=self.rng.uniform(140_000.0, 220_000.0),
            thrust_asl=0.0, isp_vac=310.0, isp_asl=270.0)
        first.thrust_asl = first.thrust_vac * 0.86

        # Гарантируем, что аппарат в принципе способен оторваться:
        # эпизод с TWR < 1 неразрешим, и сеть учится на нём мусору.
        launch_mass = first.total + second.total
        min_thrust = 1.15 * launch_mass * 9.81
        if first.thrust_asl < min_thrust:
            scale = min_thrust / first.thrust_asl
            first.thrust_asl *= scale
            first.thrust_vac *= scale
        for stage in (first, second):
            stage.capacity = stage.fuel_mass
        return [first, second]

    # ------------------------------------------------------------------
    def reset(self, goal_altitude: float | None = None) -> list[float]:
        if goal_altitude is not None:
            self.goal_altitude = goal_altitude
        self.stages = self.random_rocket()
        self.stage_index = 0
        self.time = 0.0
        self.x = 0.0                       # горизонтальная координата
        self.y = KERBIN_RADIUS            # расстояние от центра планеты
        # Вращение планеты помогает старту. Направление «на восток» во всём
        # симуляторе — это (-up_y, up_x), то есть при старте с (0, R) оно
        # смотрит в −x. Раньше здесь стояло +x: аппарат на столе «летел»
        # против собственного вращения со скоростью 350 м/с относительно
        # воздуха, и первое же наблюдение уходило в мусор.
        self.vx = -KERBIN_ROTATION_SPEED
        self.vy = 0.0
        self.pitch = 90.0                 # градусы от горизонта
        self.heading = 90.0               # на восток
        self.roll = 0.0
        self.throttle = 0.0               # фактическое положение дросселя
        self.done = False
        self.airborne = False
        self.crashed = False
        self.reached_orbit = False
        self.max_apoapsis = 0.0
        self.initial_fuel = sum(s.fuel_mass for s in self.stages)
        self.peak_altitude = 0.0
        self.separations = 0
        self.step_info = {}
        self.on_reset()

        if self.rng.random() < self.airborne_start_chance():
            self._randomize_airborne_start()
        return self.observation()

    def airborne_start_chance(self) -> float:
        """Доля эпизодов, начинающихся в воздухе, — тает вместе с разогревом.

        Приём exploring starts нужен в начале: пока ракета стоит на столе,
        награда одинакова для любой тяги ниже той, что оторвёт аппарат, и
        зацепиться не за что. Но экзамен спрашивают СО СТОЛА, и если
        половина обучения так и остаётся полётом «с середины», сеть
        доводит уже летящий аппарат, а стартовать не умеет.

        Живой замер обученной политики: со стола 0 орбит из 20 при газе
        0.27, в воздухе 15 из 20 при газе 0.95. Обучающая выборка должна к
        концу разогрева сойтись к тому, что спрашивают.
        """
        return self.exploring_starts * max(0.0, 1.0 - self.difficulty)

    def on_reset(self) -> None:
        """Хук для этапов программы: обнулить свои механики."""

    def set_difficulty(self, value: float) -> None:
        """Насколько строго среда наказывает за ошибки прямо сейчас.

        В базовом тренажёре ни на что не влияет: смягчать там нечего.
        В Тренажёре+ от этого зависит штраф за скольжение — сеть сначала
        получает предупреждение, а смертельный штраф только тогда, когда
        уже научилась держать нос по потоку.
        """
        self.difficulty = max(0.0, min(1.0, float(value)))

    def _randomize_airborne_start(self) -> None:
        """Случайное состояние в середине выведения."""
        altitude = self.rng.uniform(500.0, 45_000.0)
        speed = self.rng.uniform(80.0, 2_000.0)
        # угол вектора скорости к горизонту: от почти вертикали до пологого
        flight_angle = math.radians(self.rng.uniform(5.0, 88.0))
        self.pitch = math.degrees(flight_angle) + self.rng.uniform(-12.0, 12.0)
        self.pitch = max(-90.0, min(90.0, self.pitch))

        r = KERBIN_RADIUS + altitude
        self.x, self.y = 0.0, r
        up_x, up_y = 0.0, 1.0
        east_x, east_y = -up_y, up_x
        vertical = speed * math.sin(flight_angle)
        horizontal = speed * math.cos(flight_angle) + KERBIN_ROTATION_SPEED
        self.vx = up_x * vertical + east_x * horizontal
        self.vy = up_y * vertical + east_y * horizontal
        self.airborne = True
        self.peak_altitude = altitude

        # часть топлива уже израсходована
        burned = self.rng.uniform(0.0, 0.7)
        first = self.stages[0]
        first.fuel_mass *= (1.0 - burned)
        if burned > 0.55 and len(self.stages) > 1:
            self.stage_index = 1
        self.time = self.rng.uniform(0.0, 60.0)

    # ------------------------------------------------------------------
    # Примитивы состояния. Всё, что знает о размерности пространства,
    # собрано здесь: трёхмерный клон переопределяет только их.
    # ------------------------------------------------------------------
    def _radius(self) -> float:
        return math.hypot(self.x, self.y)

    def _speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def _vertical_speed(self) -> float:
        r = max(1.0, self._radius())
        return (self.vx * self.x + self.vy * self.y) / r

    def _angular_momentum(self) -> float:
        return abs(self.x * self.vy - self.y * self.vx)

    def _inclination(self) -> float:
        return 0.0                      # плоская задача: наклонение всегда 0

    @property
    def altitude(self) -> float:
        return self._radius() - KERBIN_RADIUS

    @property
    def mass(self) -> float:
        return sum(s.total for s in self.stages[self.stage_index:])

    @property
    def current_stage(self) -> SimStage | None:
        if self.stage_index < len(self.stages):
            return self.stages[self.stage_index]
        return None

    def fuel_left(self) -> float:
        return sum(s.fuel_mass for s in self.stages[self.stage_index:])

    # ------------------------------------------------------------------
    # Атмосфера и двигатели. Тренажёр++ подменяет их точными таблицами.
    # ------------------------------------------------------------------
    def pressure_pa(self, altitude: float) -> float:
        """Статическое давление по высотной таблице игры, Па."""
        if altitude >= KERBIN_ATMO_HEIGHT:
            return 0.0
        return _interpolate_log(KERBIN_PRESSURE_KEYS, max(0.0, altitude)) * 1000.0

    def pressure_atm(self, altitude: float) -> float:
        return self.pressure_pa(altitude) / SEA_LEVEL_PA

    def temperature(self, altitude: float) -> float:
        return _interpolate(KERBIN_TEMPERATURE_KEYS,
                            max(0.0, min(KERBIN_ATMO_HEIGHT, altitude)))

    def density(self, altitude: float) -> float:
        """Плотность из уравнения состояния: на уровне моря ровно 1.225."""
        if altitude >= KERBIN_ATMO_HEIGHT:
            return 0.0
        return self.pressure_pa(altitude) / (R_SPECIFIC
                                             * self.temperature(altitude))

    def speed_of_sound(self, altitude: float) -> float:
        return math.sqrt(HEAT_RATIO * R_SPECIFIC * self.temperature(altitude))

    def drag_coefficient(self, mach: float) -> float:
        return _interpolate(DRAG_MACH_KEYS, mach)

    def thrust_and_isp(self, stage: SimStage, pressure: float) -> tuple[float, float]:
        """Тяга и удельный импульс при данном давлении — по формуле игры.

        Расход топлива постоянный, а тяга падает вместе с удельным
        импульсом: F(p) = F_vac · Isp(p)/Isp_vac. Так считает KSP, и так
        должны считать все ступени программы, иначе сеть переучивается на
        другой двигатель при каждом переходе.
        """
        p = max(0.0, min(1.0, pressure))
        isp = stage.isp_vac + (stage.isp_asl - stage.isp_vac) * p
        thrust = stage.thrust_vac * (isp / stage.isp_vac)
        return thrust, isp

    # ------------------------------------------------------------------
    def orbital_elements(self) -> tuple[float, float, float]:
        """(апоапсис, периапсис, эксцентриситет) — высоты над поверхностью."""
        r = self._radius()
        v = self._speed()
        energy = v * v / 2.0 - KERBIN_MU / r
        h = self._angular_momentum()
        if abs(energy) < 1e-9:
            return 0.0, 0.0, 1.0
        a = -KERBIN_MU / (2.0 * energy)
        e_sq = 1.0 + 2.0 * energy * h * h / (KERBIN_MU ** 2)
        e = math.sqrt(max(0.0, e_sq))
        if a <= 0:                    # гиперболическая траектория
            return 1e9, a * (1 + e) - KERBIN_RADIUS, e
        return a * (1 + e) - KERBIN_RADIUS, a * (1 - e) - KERBIN_RADIUS, e

    def specific_energy(self) -> float:
        """Удельная орбитальная энергия: v²/2 − μ/r (Дж/кг)."""
        r = max(1.0, self._radius())
        v = self._speed()
        return v * v / 2.0 - KERBIN_MU / r

    def target_energy(self) -> float:
        """Удельная энергия на целевой круговой орбите, Дж/кг."""
        return -KERBIN_MU / (2.0 * (KERBIN_RADIUS + self.goal_altitude))

    def energy_scale(self) -> float:
        """Насколько энергию надо поднять от стартовой до целевой орбиты."""
        start = (KERBIN_ROTATION_SPEED ** 2) / 2.0 - KERBIN_MU / KERBIN_RADIUS
        return max(1.0, self.target_energy() - start)

    def target_momentum(self) -> float:
        """Момент импульса на целевой круговой орбите: h = r·v = √(μ·r)."""
        return math.sqrt(KERBIN_MU * (KERBIN_RADIUS + self.goal_altitude))

    def momentum_scale(self) -> float:
        """Насколько поднять момент импульса от стартового до орбитального.

        На столе аппарат уже вращается вместе с планетой, и какой-то момент
        у него есть — около 7 % от орбитального. Считаем от него.
        """
        start = KERBIN_RADIUS * KERBIN_ROTATION_SPEED
        return max(1.0, self.target_momentum() - start)

    # ------------------------------------------------------------------
    def observation(self) -> list[float]:
        return encode_flight_state(
            self.snapshot(), self.goal_altitude,
            body_gravity=KERBIN_MU / (KERBIN_RADIUS ** 2),
            body_radius=KERBIN_RADIUS, has_atmosphere=True,
            prograde_pitch=self.prograde_pitch(),
            angle_of_attack=self.angle_of_attack())

    def prograde_pitch(self) -> float:
        speed = self._speed()
        if speed < 1e-3:
            return 90.0
        vertical = self._vertical_speed()
        return math.degrees(math.asin(max(-1.0, min(1.0, vertical / speed))))

    def air_velocity(self) -> tuple[float, float]:
        """Скорость относительно атмосферы, которая вращается с планетой.

        Без этой поправки аппарат на стартовом столе «летит» со скоростью
        вращения Кербина — 175 м/с. Для орбитальной механики это верно, а
        для аэродинамики нет: угол атаки на столе получался 90°, а
        скоростной напор — 18 кПа на неподвижной ракете.
        """
        r = max(1.0, self._radius())
        up_x, up_y = self.x / r, self.y / r
        east_x, east_y = -up_y, up_x
        surface = KERBIN_ROTATION_SPEED * r / KERBIN_RADIUS
        return self.vx - surface * east_x, self.vy - surface * east_y

    def air_speed(self) -> float:
        air = self.air_velocity()
        return math.hypot(air[0], air[1])

    def angle_of_attack(self) -> float:
        """Угол между носом и набегающим потоком, градусы."""
        air = self.air_velocity()
        speed = math.hypot(air[0], air[1])
        if speed < 1.0:
            return 0.0
        r = max(1.0, self._radius())
        vertical = (air[0] * self.x + air[1] * self.y) / r
        flight = math.degrees(math.asin(max(-1.0, min(1.0, vertical / speed))))
        return abs(self.pitch - flight)

    def dynamic_pressure(self) -> float:
        """Скоростной напор q = ρv²/2, Па — по скорости относительно воздуха."""
        speed = self.air_speed()
        return 0.5 * self.density(self.altitude) * speed * speed

    def electric_fraction(self) -> float:
        """Доля оставшегося заряда. Лимит энергии появляется в Тренажёре+."""
        return 1.0

    def effective_throttle(self) -> float:
        """Фактическое положение дросселя.

        В базовом тренажёре двигатель отзывается мгновенно, поэтому
        совпадает с командой. В Тренажёре+ появляется раскрутка турбины,
        и фактическая тяга отстаёт от команды на полсекунды-секунду.
        """
        return self.throttle

    def action_mask(self) -> list[bool]:
        """Что физически разрешено сделать прямо сейчас.

        Единственная дискретная голова сети — «отделить ступень».
        Отделение полной ступени выбрасывает работающий двигатель вместе
        с топливом; в KSP это разрешено, но осмысленного применения при
        выведении не имеет, а PPO цеплялся именно за него (MIGRATION 7.3).
        Маска запрещает действие, а не наказывает за него: политика
        вообще не тратит вероятностную массу на заведомо гиблый ход.
        """
        stage = self.current_stage
        can_stage = (stage is not None
                     and self.stage_index < len(self.stages) - 1
                     and stage.fuel_fraction <= SPENT_STAGE_FUEL)
        return [True, bool(can_stage)]

    def snapshot(self) -> SimSnapshot:
        altitude = self.altitude
        apoapsis, periapsis, eccentricity = self.orbital_elements()
        speed = self._speed()
        vertical = self._vertical_speed()
        horizontal = math.sqrt(max(0.0, speed * speed - vertical * vertical))
        rho = self.density(altitude)
        stage = self.current_stage
        fuel_left = self.fuel_left()
        pressure = self.pressure_atm(altitude)
        thrust = 0.0
        isp = 0.0
        if stage is not None and stage.fuel_mass > 0:
            thrust, isp = self.thrust_and_isp(stage, pressure)

        return SimSnapshot(
            altitude=altitude, surface_altitude=altitude,
            apoapsis=apoapsis, periapsis=periapsis,
            time_to_apoapsis=max(0.0, (apoapsis - altitude) / max(1.0, abs(vertical))),
            eccentricity=min(3.0, eccentricity),
            inclination=self._inclination(),
            speed=speed, vertical_speed=vertical, horizontal_speed=horizontal,
            pitch=self.pitch, heading=self.heading, roll=self.roll,
            dynamic_pressure=self.dynamic_pressure(),
            static_pressure=pressure * 101_325.0,
            atmosphere_density=rho,
            g_force=1.0 + abs(vertical) / 100.0,
            mass=self.mass / 1000.0,
            dry_mass=sum(s.dry_mass for s in self.stages[self.stage_index:]) / 1000.0,
            thrust=thrust * self.effective_throttle() / 1000.0,
            available_thrust=thrust / 1000.0,
            specific_impulse=isp,
            throttle=self.throttle,
            stage=len(self.stages) - self.stage_index,
            part_count=10,
            mission_time=self.time,
            situation="pre_launch" if self.time < 1 else
                      ("flying" if altitude < KERBIN_ATMO_HEIGHT else "orbiting"),
            resources={
                "LiquidFuel": {"amount": fuel_left, "max": max(1.0, self.initial_fuel)},
                "Oxidizer": {"amount": fuel_left, "max": max(1.0, self.initial_fuel)},
                "ElectricCharge": {"amount": 100.0 * self.electric_fraction(),
                                   "max": 100.0},
            })

    # ==================================================================
    # ШАГ СИМУЛЯЦИИ
    # ==================================================================
    def step(self, action: FlightAction):
        """Шаг симуляции. Возвращает (observation, reward, done, info)."""
        if self.done:
            return self.observation(), 0.0, True, {"already_done": True}

        previous_apoapsis = self.orbital_elements()[0]
        previous_energy = self.specific_energy()
        previous_momentum = self._angular_momentum()
        self.step_info = {}

        command = self.filter_command(action)
        self.apply_command(command)
        self.integrate()
        impact_speed = self.ground_contact()

        altitude = self.altitude
        self.peak_altitude = max(self.peak_altitude, altitude)
        apoapsis, periapsis, eccentricity = self.orbital_elements()
        self.max_apoapsis = max(self.max_apoapsis, apoapsis)
        # --- награда ---
        # Главный плотный сигнал — прирост удельной орбитальной энергии
        # ε = v²/2 − μ/r. Это и есть «то, что делает ракета»: любой полезный
        # манёвр её повышает, любые потери (сопротивление, гравитационные)
        # понижают. Такой сигнал приходит на КАЖДОМ шаге, а не в конце, —
        # без этого PPO в задаче с наградой через тысячу шагов не сдвинется.
        # Это функция награды, а не подсказка агенту: сеть по-прежнему видит
        # только сырые числа и сама ищет, какими действиями энергию поднимать.
        reward = 0.0
        info: dict = dict(self.step_info)

        # Энергия считается ДО ЦЕЛЕВОЙ и ни на джоуль дальше. Без потолка
        # член был неограниченным, и выяснилось это дорого: замер
        # детерминированной политики после 178 млн шагов показал полёт
        # строго вверх до 739 км с уходом на гиперболу и наградой +23.9
        # по одной только энергии — ВДВОЕ больше тех +12.0, которые даёт
        # выход на орбиту. Плотный сигнал звал ракету прочь от планеты, и
        # сеть честно шла туда, куда ей платили.
        target = self.target_energy()
        energy_gain = (min(self.specific_energy(), target)
                       - min(previous_energy, target))
        reward += energy_gain / self.energy_scale() * 12.0

        # Момент импульса |r×v| — то единственное, чего у вертикального
        # выстрела нет и быть не может. Энергию можно набрать, уходя прямо
        # вверх; момент импульса набирается только разворотом на горизонт.
        # Это и есть недостающий градиент «поворачивай на восток»: он
        # приходит на каждом шаге, а не в конце, и потолок у него такой
        # же — на целевой орбите платить больше не за что.
        momentum_goal = self.target_momentum()
        momentum_gain = (min(self._angular_momentum(), momentum_goal)
                         - min(previous_momentum, momentum_goal))
        reward += momentum_gain / self.momentum_scale() * 8.0

        gain = min(apoapsis, self.goal_altitude) - min(previous_apoapsis,
                                                       self.goal_altitude)
        reward += gain / self.goal_altitude * 4.0

        q = self.dynamic_pressure()
        if q > 30_000.0:
            reward -= (q - 30_000.0) / 30_000.0 * 0.05

        # Нос ниже горизонта в плотных слоях — это тяга в землю. На прогоне
        # 4.5 млн шагов политика уезжала тангажом с +90° до −87° и прижимала
        # аппарат к столу: прирост энергии наказывал её слишком медленно.
        if altitude < 10_000.0 and self.pitch < 0.0:
            reward -= min(1.0, -self.pitch / 90.0) * 0.05

        reward -= 0.0005
        reward += self.shaping_reward(command)

        # --- условия конца эпизода ---
        # Проверки независимые: раньше они стояли цепочкой elif, и при
        # пустых баках с высоким апоапсисом лимит времени не проверялся —
        # эпизод не заканчивался никогда.
        fuel_left = self.fuel_left()
        falling = self._vertical_speed() < 0
        failure = self.check_failure()

        if self.airborne and altitude <= 0.5 and impact_speed > 12.0:
            self.done = True
            self.crashed = True
            reward -= 5.0
            info["reason"] = "падение"
        elif failure is not None:
            penalty, reason = failure
            self.done = True
            self.crashed = True
            reward -= penalty
            info["reason"] = reason
        elif (periapsis > KERBIN_ATMO_HEIGHT and apoapsis > KERBIN_ATMO_HEIGHT
              and eccentricity < 1.0):
            # эксцентриситет обязателен: без него «орбитой» засчитывалась
            # гиперболическая траектория ухода из системы
            self.done = True
            self.reached_orbit = True
            reward += 50.0
            error = abs(apoapsis - self.goal_altitude) / self.goal_altitude
            reward += max(0.0, 10.0 * (1.0 - error))
            info["reason"] = "орбита"
        elif not self.airborne and self.time >= self.LIFTOFF_DEADLINE:
            # так и не оторвался: дальше смотреть нечего, а две тысячи
            # шагов стояния на столе отравят обучение (см. LIFTOFF_DEADLINE)
            self.done = True
            reward -= self.LIFTOFF_PENALTY
            info["reason"] = "не оторвался от стола"
        elif fuel_left <= 0.0 and falling and altitude < KERBIN_ATMO_HEIGHT:
            # топливо кончилось и аппарат уже падает — дальше смотреть нечего
            self.done = True
            reward -= 2.0
            info["reason"] = "кончилось топливо"
        elif self.time >= self.max_time:
            self.done = True
            reward -= 1.0
            info["reason"] = "лимит времени"

        info.update({"altitude": altitude, "apoapsis": apoapsis,
                     "periapsis": periapsis, "orbit": self.reached_orbit})
        return self.observation(), reward, self.done, info

    # ------------------------------------------------------------------
    # Части шага. Переопределяются следующими ступенями программы.
    # ------------------------------------------------------------------
    def filter_command(self, action: FlightAction) -> FlightAction:
        """Что из решения сети реально дошло до аппарата.

        В базовом тренажёре — всё и мгновенно. В Тренажёре+ здесь живут
        задержка связи и потеря управления при разряженной батарее.
        """
        return action

    def apply_command(self, command: FlightAction) -> None:
        """Отделение ступени, дроссель, рули."""
        if command.stage and self.can_separate():
            self.separate()
        self.throttle = max(0.0, min(1.0, command.throttle))
        # команда тангажа меняет угол не мгновенно: рули имеют скорость
        self.pitch = max(-90.0, min(90.0,
                                    self.pitch + command.pitch * 25.0 * self.dt))

    def can_separate(self) -> bool:
        """Можно ли сейчас отделить ступень (правило мира, не награда)."""
        stage = self.current_stage
        return (stage is not None
                and self.stage_index < len(self.stages) - 1
                and stage.fuel_fraction <= SPENT_STAGE_FUEL)

    def separate(self) -> None:
        self.stage_index += 1
        self.separations += 1

    def burn(self, thrust_max: float, isp: float) -> float:
        """Сжигает топливо за шаг и возвращает фактическую тягу, Н."""
        stage = self.current_stage
        throttle = self.effective_throttle()
        if stage is None or stage.fuel_mass <= 0 or throttle <= 0:
            return 0.0
        thrust = thrust_max * throttle
        flow = thrust / max(1.0, isp * 9.80665)
        burned = min(stage.fuel_mass, flow * self.dt)
        stage.fuel_mass = max(0.0, stage.fuel_mass - burned)
        return thrust

    def integrate(self) -> None:
        """Один шаг физики: тяга, гравитация, сопротивление."""
        altitude = self.altitude
        rho = self.density(altitude)
        pressure = self.pressure_atm(altitude)
        stage = self.current_stage

        thrust = 0.0
        if stage is not None:
            thrust_max, isp = self.thrust_and_isp(stage, pressure)
            thrust = self.burn(thrust_max, isp)

        mass = max(1.0, self.mass)
        r = self._radius()
        up_x, up_y = self.x / r, self.y / r
        east_x, east_y = -up_y, up_x           # направление «по горизонту»

        pitch_rad = math.radians(self.pitch)
        dir_x = up_x * math.sin(pitch_rad) + east_x * math.cos(pitch_rad)
        dir_y = up_y * math.sin(pitch_rad) + east_y * math.cos(pitch_rad)

        ax = thrust * dir_x / mass
        ay = thrust * dir_y / mass

        # --- гравитация ---
        g = KERBIN_MU / (r * r)
        ax -= g * up_x
        ay -= g * up_y

        # --- сопротивление (по скорости относительно воздуха) ---
        # Cd берётся по числу Маха, площадь — миделя корпуса: те же
        # формулы, что и на старших ступенях. Раньше здесь стояло
        # «площадь от массы при Cd = 1», и аппарат тормозился в разы
        # сильнее, чем на Тренажёре++ (см. заголовок модуля).
        air_x, air_y = self.air_velocity()
        air_speed = math.hypot(air_x, air_y)
        if air_speed > 0 and rho > 0:
            mach = air_speed / self.speed_of_sound(altitude)
            drag = (0.5 * rho * air_speed * air_speed
                    * self.drag_coefficient(mach) * self.body_area)
            # Сопротивление тормозит, но никогда не разворачивает поток:
            # импульс за шаг не может превысить сам импульс аппарата.
            # Без этого потолка явная схема взрывается на всём, что
            # тормозит резко, — парашют уводил скорость в минус и обратно
            # с нарастанием, пока «напор» не убивал кабину.
            drag = min(drag, mass * air_speed / self.dt)
            ax -= drag * air_x / air_speed / mass
            ay -= drag * air_y / air_speed / mass

        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.time += self.dt

    def ground_contact(self) -> float:
        """Опора на грунт. Возвращает скорость удара, если она была.

        Пока тяга не превысила вес, аппарат стоит на столе: гравитация
        компенсируется реакцией опоры, а не роняет его «под поверхность».
        """
        if self.altitude >= 0.0:
            if self.altitude > 5.0:
                self.airborne = True
            return 0.0

        r_now = self._radius()
        up_x, up_y = self.x / r_now, self.y / r_now
        radial = self.vx * up_x + self.vy * up_y
        impact_speed = abs(radial)
        self.x, self.y = up_x * KERBIN_RADIUS, up_y * KERBIN_RADIUS
        if radial < 0.0:
            # убираем вертикальную составляющую — опора её держит
            self.vx -= radial * up_x
            self.vy -= radial * up_y
        if not self.airborne:
            # ещё не взлетали: держим горизонтальную скорость вращения
            east_x, east_y = -up_y, up_x
            tangential = self.vx * east_x + self.vy * east_y
            drift = tangential - KERBIN_ROTATION_SPEED
            self.vx -= drift * east_x
            self.vy -= drift * east_y
        return impact_speed

    # Оси, которые в плоском мире ни на что не влияют. Держать их
    # бесплатными нельзя: за что не спрашивают, то уходит на упор.
    INERT_AXES: tuple[str, ...] = ("yaw", "roll")
    INERT_PENALTY = 0.002        # за единицу команды за шаг

    def shaping_reward(self, command: FlightAction) -> float:
        """Дополнительные штрафы этапа (аэродинамика, энергия, ресурсы).

        Здесь же — плата за руль, который в этом мире не работает.

        Зачем. Плоский тренажёр не знает ни курса, ни крена: что бы сеть
        ни выдала по этим осям, ничего не происходит, награда не меняется,
        градиента нет. Значение уходит на упор — и замер это подтвердил:
        детерминированная политика держала рыскание ровно −1.00 и крен
        −0.80 на обеих плоских ступенях.

        Беда вскрывается на Тренажёре++, где ось оживает. У вертикально
        стоящей ракеты команда рыскания усилена вчетверо (1/cos тангажа),
        и −1.00 разворачивает курс со скоростью 100 градусов в секунду —
        полный оборот за четыре секунды. Дальше чудовищное скольжение,
        опрокидывание и паника с дросселем. Четыре прогона подряд ступень
        3 умирала именно так.

        Лечится не на третьей ступени, а здесь: руль тратит заряд и
        раскачивает аппарат независимо от того, есть ли от него толк.
        Цена мелкая — полностью выкрученная ось стоит около четырёх очков
        за полёт при полусотне за орбиту, — но градиент к нулю появляется,
        и на следующую ступень политика приходит с чистыми осями.
        """
        if not self.INERT_AXES:
            return 0.0
        effort = sum(abs(getattr(command, axis, 0.0))
                     for axis in self.INERT_AXES)
        return -self.INERT_PENALTY * effort * (self.dt / self.DEFAULT_DT)

    def check_failure(self) -> tuple[float, str] | None:
        """Способы угробить аппарат: (штраф, причина).

        Разрушение скоростным напором живёт здесь, а не только в
        трёхмерном клоне, по той же причине, что и общая атмосфера: мир на
        всех ступенях один. Пока предел существовал лишь с третьей
        ступени, на первых двух можно было безнаказанно разгоняться в
        плотных слоях — и ровно этому сеть и училась, а на третьей за то
        же самое её убивало. Замер эталонного профиля: 19-33 кПа на всех
        ступенях, то есть запас до предела остаётся заметный.
        """
        if self.dynamic_pressure() > self.BREAKUP_PRESSURE:
            return 6.0, "аппарат разрушен скоростным напором"
        return None

    # ------------------------------------------------------------------
    def episode_summary(self) -> dict:
        apoapsis, periapsis, _ = self.orbital_elements()
        return {
            "time": round(self.time, 1),
            "peak_altitude": round(self.peak_altitude),
            "apoapsis": round(apoapsis),
            "periapsis": round(periapsis),
            "orbit": self.reached_orbit,
            "crashed": self.crashed,
        }
