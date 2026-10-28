"""ТРЕНАЖЁР+ — вторая ступень учебной программы: скрытые механики KSP.

Физика движения здесь та же, что в базовом тренажёре (и считается так же
быстро). Меняется другое — всё, что стоит МЕЖДУ решением сети и ракетой.
Именно на этом разваливается политика, обученная в идеальном мире:
в тренажёре команда исполняется мгновенно и точно, а в игре она идёт
через сеть, дроссель раскручивается, ветер сносит нос, а батарея садится.

Шесть механик, все — правила мира, а не подсказки агенту:

1. ЗАДЕРЖКА kRPC (~100 мс). Команда доходит до аппарата не в тот же миг:
   часть шага исполняется ещё прошлое решение. Учит не полагаться на
   мгновенную обратную связь.
2. ТАЙМАУТ СТУПЕНЕЙ (2 с). После отделения стейджинг заблокирован —
   как в игре, где повторный пробел просто не проходит. Отучает
   «спамить пробел».
3. ВЕТЕР (доменная рандомизация). Коррелированный случайный снос носа
   на 1-2 градуса, сильный у земли и исчезающий с высотой. Заставляет
   держать курс постоянными микро-коррекциями, а не одной командой.
4. ШТРАФ ЗА СКОЛЬЖЕНИЕ (AoA). Угол между носом и вектором движения
   больше 5° в плотных слоях (ниже 15 км) наказывается тем сильнее, чем
   выше скоростной напор. При грубом развороте аппарат опрокидывает —
   эпизод обрывается.
5. ИНЕРЦИЯ ТЯГИ (0.5-1 с). Двигатель выходит на режим не мгновенно:
   тяга догоняет команду по экспоненте. Управлять надо на опережение.
6. ЛИМИТ ЭНЕРГИИ. Каждое движение рулей тратит заряд, причём резкие
   перекладки — в разы дороже плавных. На нуле управление отваливается
   и возвращается только по мере подзарядки от генератора двигателя.

Всё это сеть видит только через те же сырые числа: тягу, которая
отстаёт от дросселя, угол атаки, долю заряда в `ElectricCharge`.
Никаких новых входов не добавлено — веса прошлого этапа подходят
без переобучения с нуля.
"""
from __future__ import annotations

import math

from .sim_env import (KERBIN_SEA_DENSITY, SPENT_STAGE_FUEL, AscentSimulator,
                      SimStage)
from .spaces import FlightAction


class HardcoreAscentSimulator(AscentSimulator):
    """Базовый тренажёр плюс жёсткие скрытые механики реальной игры."""

    stage_id = 2
    title = "Тренажёр+"

    # --- задержка связи ---
    LAG_SECONDS = 0.10           # типичный круг kRPC: команда + подтверждение
    LAG_JITTER = (0.8, 1.4)      # разброс: сеть не бывает стабильной

    # --- блокировка стейджинга ---
    STAGE_COOLDOWN = 2.0         # с, повторное отделение не проходит

    # --- ветер ---
    WIND_TAU = 5.0               # с, время корреляции порыва
    WIND_DEG = 1.5               # град, стандартное отклонение сноса носа

    # --- инерция тяги ---
    SPOOL_MIN = 0.5              # с, самый резвый двигатель
    SPOOL_MAX = 1.0              # с, самый вялый

    # --- электричество ---
    # Ёмкость подняли вдвое после разбора обвала политики на 112-м
    # обновлении: при 100 единицах порыв ветра оборачивался паникой,
    # полным разрядом за 4 секунды и падением. Восемь-десять секунд —
    # это уже время на то, чтобы выровнять курс, а не приговор.
    # Пороги brownout/dead остались абсолютными: восстановление после
    # обесточивания должно занимать столько же, сколько раньше.
    EC_CAPACITY = 200.0          # условных единиц заряда
    EC_HOLD = 1.0                # ед/с при полностью отклонённых рулях
    EC_MOVE = 4.0                # ед на единицу ПЕРЕКЛАДКИ руля
    EC_ALTERNATOR = 4.0          # ед/с от генератора работающего двигателя
    EC_SOLAR = 0.4               # ед/с от панелей при выключенном двигателе
    EC_BROWNOUT = 25.0           # ниже этого заряда управление слабеет
    EC_DEAD = 2.0                # ниже этого зонд гаснет совсем

    # --- аэродинамика ---
    # Пары «строгое значение / щадящее»: между ними среда двигается по
    # `difficulty`. Смысл в том, чтобы сеть сначала получала ПРЕДУПРЕЖДЕНИЕ
    # (широкий допуск, мелкий штраф, опрокидывание только на явном боку),
    # а полную строгость — уже освоив, как держать нос по потоку.
    AOA_LIMIT = 5.0              # град, за пределом начинается штраф
    AOA_LIMIT_EASY = 12.0        # град, допуск в начале обучения
    AOA_DENSE_ALTITUDE = 15_000.0
    AOA_PENALTY = 0.25           # за 10° перебора при полном напоре
    AOA_PENALTY_EASY = 0.05      # впятеро мягче на первых порах
    AOA_TUMBLE = 55.0            # град, дальше аппарат уже не выправить
    AOA_TUMBLE_EASY = 80.0       # в начале обучения прощаем почти боком
    TUMBLE_PRESSURE = 12_000.0   # Па, при меньшем напоре не опрокинет

    # --- отказы ---
    # Выключены по умолчанию: это отдельная ступень программы, а не
    # фон. Включение здесь поменяло бы смысл всех прежних замеров.
    FAILURES = False

    # ------------------------------------------------------------------
    def __init__(self, *args, mechanics: bool = True,
                 failures: bool | None = None,
                 failure_rate: float | None = None,
                 crew_share: float = 0.5, **kwargs):
        """mechanics=False — тот же мир, но с идеальным интерфейсом.

        Скрытые механики (задержка команд, ветер, батарея, раскрутка
        двигателя, таймаут отделения) — это про то, что стоит МЕЖДУ
        решением сети и ракетой, а не про физику полёта. Первая ступень
        программы отключает именно их: аппарат, воздух и аэродинамика
        остаются настоящими, а управление отзывается мгновенно и точно.

        Разделение важно. Пока первая ступень была вдобавок ещё и плоской,
        сеть приходила на трёхмерную с привычками, которые там убивали:
        замер показал 0 орбит после перехода против 164 орбит у той же
        ступени, выученной с чистого листа. Мир должен быть один, меняться
        должно только то, что мешает управлять.
        """
        self.mechanics = bool(mechanics)
        if failures is not None:
            self.FAILURES = bool(failures)
        # Доля отказов и доля полётов с экипажем задаются ступенью.
        # На ступени, где учат спасать, ломается каждый полёт и люди на
        # борту всегда: иначе градиент размазывается по исправным рейсам.
        self.failure_rate = failure_rate
        self.crew_share = float(crew_share)
        super().__init__(*args, **kwargs)
        if not self.mechanics:
            self.LAG_SECONDS = 0.0        # команда доходит мгновенно
            self.WIND_DEG = 0.0           # штиль
            self.SPOOL_MIN = 1e-3         # двигатель выходит на режим сразу
            self.SPOOL_MAX = 1e-3
            self.STAGE_COOLDOWN = 0.0     # отделение без блокировки
            self.EC_HOLD = 0.0            # батарея не садится
            self.EC_MOVE = 0.0
            self.on_reset()               # перечитать постоянные с новыми

    def on_reset(self) -> None:
        self.held = FlightAction(0.0, 0.0, 0.0, 0.0, False)
        self.pending_stage = False
        self.stage_ready_at = 0.0
        self.actual_throttle = 0.0
        self.spool_tau = self.rng.uniform(self.SPOOL_MIN, self.SPOOL_MAX)
        self.wind_state = 0.0
        self.wind_offset = 0.0        # накопленный снос, только для отчёта
        self.electric = self.EC_CAPACITY
        self.min_electric = self.EC_CAPACITY
        self.blackouts = 0
        self.control_lost = False
        self.last_deflection = 0.0
        self.requested = 0.0
        self.aoa_violations = 0
        self.stage_blocked = 0
        self.tumbled = False
        self.aborted = False
        self.chute_open = False
        self.abandoned = 0.0
        self.body_area = AscentSimulator.body_area
        self.BREAKUP_PRESSURE = AscentSimulator.BREAKUP_PRESSURE
        self.roll_failure()

    # ------------------------------------------------------------------
    # 7. Отказы в полёте
    # ------------------------------------------------------------------
    # Порог мягкой посадки — тот же, по которому базовый тренажёр считает
    # удар смертельным. Ниже него касание грунта переживают и аппарат,
    # и люди.
    SOFT_TOUCHDOWN = 12.0

    # ------------------------------------------------------------------
    # Отделяемая кабина
    # ------------------------------------------------------------------
    # Без неё ступень «Аварии» была нечестной задачей. Замер: скриптовый
    # пилот с идеальным тормозным импульсом спасал 0 экипажей из 34 —
    # ракета на спуске разрушалась скоростным напором в 30 случаях из 34.
    # Дело не в политике: корпус с лобовой площадью ракеты, падающий с
    # сорока километров, проходит 45 кПа раньше, чем успевает затормозить.
    # В самой игре экипаж в такой ситуации спасает не манёвр, а кнопка
    # аварийного увода: кабина уходит от ракеты, тормозит щитом и садится
    # под куполом. Этого в тренажёре не было — значит, учили невозможному.
    #
    # Числа подобраны по кораблям игры: кабина Mk1-3 с щитом весит около
    # полутора тонн, купол Mk16 держит её у земли на шести метрах в
    # секунду, а щит терпит вчетверо больший напор, чем голый корпус.
    CAPSULE_MASS = 1500.0          # кг — кабина, щит, купол
    CAPSULE_AREA = 3.1              # м² — щит диаметром два метра
    CAPSULE_CD = 1.2                # тупое тело тормозит, а не режет воздух
    CAPSULE_BREAKUP = 180_000.0    # Па — щит держит вчетверо больше корпуса
    CHUTE_CD = 1.0
    CHUTE_TERMINAL = 6.0           # м/с — с какой скоростью купол сажает
    CHUTE_ALTITUDE = 2_500.0       # м — ниже этого купол раскрывается
    CHUTE_MAX_Q = 15_000.0         # Па — выше купол рвёт потоком

    @property
    def chute_area(self) -> float:
        """Площадь купола — из той скорости, с какой он должен сажать.

        Задавать площадь в квадратных метрах бессмысленно: она зависит от
        модели сопротивления, а проверяема здесь одна величина — скорость
        касания. Поэтому площадь считается из неё, а не подбирается.
        """
        weight = self.CAPSULE_MASS * 9.81
        v = self.CHUTE_TERMINAL
        return 2.0 * weight / (KERBIN_SEA_DENSITY * self.CHUTE_CD * v * v)

    def abort_capsule(self) -> None:
        """Увести кабину с аварийной ракеты.

        Всё, что ниже кабины, остаётся позади: масса падает до полутора
        тонн, тяги больше нет и не будет, зато щит держит вход, а ниже
        двух с половиной километров раскрывается купол. Дальше от пилота
        не требуется ничего — и в этом вся суть кнопки увода.
        """
        if self.aborted:
            return
        self.aborted = True
        # Что осталось на ракете, то и потеряно: за это платят деньгами.
        self.abandoned = max(0.0, self.mass - self.CAPSULE_MASS) / 1000.0
        self.stages = [SimStage(dry_mass=self.CAPSULE_MASS, fuel_mass=0.0,
                                thrust_vac=0.0, thrust_asl=0.0,
                                isp_vac=1.0, isp_asl=1.0)]
        self.stage_index = 0
        self.body_area = self.CAPSULE_AREA
        self.BREAKUP_PRESSURE = self.CAPSULE_BREAKUP
        self.throttle = 0.0
        # Трёхмерный тренажёр считает моменты по геометрии ступеней —
        # её тоже надо пересобрать, иначе кабина полетит с плечами
        # выброшенной ракеты.
        if hasattr(self, "build_geometry"):
            self.build_geometry()

    def update_chute(self) -> None:
        """Купол: раскрывается низко и только если поток не рвёт его."""
        if not self.aborted or self.chute_open:
            return
        if self.altitude > self.CHUTE_ALTITUDE:
            return
        if self.dynamic_pressure() > self.CHUTE_MAX_Q:
            return
        self.chute_open = True
        self.body_area = self.chute_area

    def step(self, action):
        """Шаг плюс две развязки, которых нет на ступенях без отказов.

        Мягкая посадка. В обычном выведении касание грунта не исход:
        ракета либо вышла на орбиту, либо разбилась. Когда двигатель
        отказал на сороковой секунде, орбиты уже не будет, и «посадил
        живым» — это и есть лучший возможный конец. Поэтому здесь он
        становится терминальным и оплачиваемым.

        Пустые баки. Базовый тренажёр обрывает эпизод, как только топливо
        кончилось и аппарат пошёл вниз: дальше ему смотреть нечего. Здесь
        есть на что — цел ли экипаж к моменту касания. Поэтому полёт
        продолжается до земли, а штраф за обрыв возвращается.
        """
        obs, reward, done, info = super().step(action)
        if not self.FAILURES:
            return obs, reward, done, info

        if done and info.get("reason") == "кончилось топливо":
            self.done = False
            done = False
            reward += 2.0                 # тот самый штраф за обрыв
            info.pop("reason", None)
            # Проверки конца эпизода в базовом тренажёре идут цепочкой
            # elif, и на пустых баках до лимита времени дело не доходит.
            # Сняв обрыв по топливу, лимит приходится держать самому —
            # иначе падение длится бесконечно.
            if self.time >= self.max_time:
                self.done = True
                done = True
                info["reason"] = "лимит времени"

        # Кабина вышла на установившийся спуск под куполом: дальше
        # смотреть нечего, исход решён. Досиживать эти несколько минут
        # полезного сигнала не даёт, а стоит десяти тысяч шагов.
        if (not done and self.aborted and self.chute_open
                and abs(self._vertical_speed()) < 2.0 * self.CHUTE_TERMINAL):
            self.done = True
            done = True
            self.landed_safely = True
            info["reason"] = "кабина под куполом"
            reward += self.crew_reward(saved=True)
            return obs, reward, done, info

        # Орбита с людьми на борту — тоже спасённый экипаж, и платить
        # за неё надо не меньше, чем за посадку под куполом. Пока
        # премия шла только за посадку, увод кабины стоил сети столько
        # же, сколько выполненный полёт.
        if done and self.reached_orbit:
            reward += self.crew_reward(saved=True)
        elif not done and self.airborne and self.altitude <= 0.5:
            self.done = True
            done = True
            self.landed_safely = True
            info["reason"] = "мягкая посадка"
            reward += self.crew_reward(saved=True)
        elif done and not self.reached_orbit and not self.landed_safely:
            # Лимит времени — это конец замера, а не конец полёта: аппарат
            # ещё летит, и чем кончится, никто не знает. Платить за него
            # как за погибший экипаж значит наказывать политику за приговор,
            # которого не выносили. Такой эпизод просто не считается.
            if info.get("reason") != "лимит времени":
                reward += self.crew_reward(saved=False)
            else:
                info["crew_saved"] = None
        return obs, reward, done, info

    def crew_reward(self, saved: bool) -> float:
        """Цена исхода. Люди несопоставимы ни с какой экономией массы.

        Беспилотный аппарат — это деньги: за спасённое железо платят,
        за потерянное вычитают, и счёт идёт в тоннах. Экипаж в эту
        шкалу не кладётся: +60 за живых против −120 за погибших больше
        любой экономии, какую может дать риск. Сеть, которой выгодно
        поберечь ступень ценой людей, — это ровно то, чего быть не должно.
        """
        value = 60.0 if saved else -120.0
        if not self.crew:
            mass = max(0.0, self.mass) / 1000.0    # тонны
            value = 2.0 * mass if saved else -1.0 * mass
        # Брошенная ракета — это деньги, и платить за неё приходится
        # даже тогда, когда люди дома. Иначе увод кабины обходится сети
        # даром, и она уводит её при любой поломке: замер показал 43
        # увода на 45 аварий, из них больше половины — отказы, с
        # которыми ракета ещё долетела бы. Цена в тоннах заведомо меньше
        # цены экипажа, так что порядок решений не меняется: сначала
        # люди, и только потом железо.
        return value - self.abandoned

    def roll_failure(self) -> None:
        """Бросок на отказ для этого эпизода (см. neural/failures.py)."""
        from .failures import FailureState, roll
        self.fault = FailureState()
        self.pending_failure = None
        self.landed_safely = False
        # Можно ли было после этого отказа долететь. Ставится в момент
        # срабатывания, по устройству мира, а не по тому, что потом
        # сделал пилот: иначе критерий «не бросай исправную ракету»
        # обходится тем, что ракету бросают не попробовав.
        self.recoverable = False
        # Экипаж в половине эпизодов: сеть должна одинаково уверенно
        # вести и беспилотник, где можно рискнуть железом, и пилотируемый
        # аппарат, где рисковать нечем.
        self.crew = self.FAILURES and self.rng.random() < self.crew_share
        if not self.mechanics or not self.FAILURES:
            return
        # Ожидаемая длительность выведения — ориентир, когда «интересно»
        # ломаться. Точность тут не нужна: отказ всё равно раскидывается
        # по окну в несколько минут.
        self.pending_failure = roll(self.rng, self.difficulty, 300.0,
                                    rate=self.failure_rate)

    def fire_failure(self) -> None:
        """Пора ли сработать отказу. Зовётся каждый шаг до физики."""
        from .failures import apply
        pending = self.pending_failure
        if pending is None or pending.fired or self.time < pending.at:
            return
        stage = self.current_stage
        apply(pending, self.fault, stage.fuel_mass if stage else 0.0)
        last = self.stage_index >= len(self.stages) - 1
        if pending.kind in ("engine_out", "thrust_loss", "fuel_leak"):
            # Сломалась камера или потёк бак — это принадлежит горящей
            # ступени, а не всей ракете. Сбросишь её — поломка уйдёт с ней.
            self.fault.stage_bound = self.stage_index
            # Поправимо ровно тогда, когда есть что сбросить.
            self.recoverable = not last
        elif pending.kind == "stage_stuck":
            # Залипший сепаратор смертелен, пока внизу осталась ступень:
            # мёртвый груз не вывести. На последней он ничего не меняет.
            self.recoverable = last
        else:
            # Мёртвый генератор и уведённые рули полёт не отменяют.
            self.recoverable = True
        if not self.fault.alternator:
            # генератор мёртв: дальше только то, что осталось в батарее
            self.EC_ALTERNATOR = 0.0

    def leak_fuel(self) -> None:
        """Течь: топливо уходит мимо камеры, тяга его не получает."""
        stage = self.current_stage
        if stage is None or self.fault.leak_per_second <= 0.0:
            return
        stage.fuel_mass = max(0.0, stage.fuel_mass
                              - self.fault.leak_per_second * self.dt)

    # ------------------------------------------------------------------
    # 1. Задержка связи и 6. потеря управления при разряде
    # ------------------------------------------------------------------
    def filter_command(self, action: FlightAction) -> FlightAction:
        """Что реально дошло до аппарата за этот шаг.

        Команда уходит в игру и возвращается подтверждением примерно за
        100 мс. Значит, доля шага `lag/dt` исполняется ещё ПРОШЛЫМ
        решением. Смешиваем старую и новую команду по времени — это
        честная модель кусочно-постоянного сигнала с задержкой, а не
        «размытие» команды.
        """
        lag = self.LAG_SECONDS * self.rng.uniform(*self.LAG_JITTER)
        stale = min(1.0, max(0.0, lag / self.dt))
        fresh = 1.0 - stale
        held = self.held

        def mix(old: float, new: float) -> float:
            return old * stale + new * fresh

        throttle = mix(held.throttle, action.throttle)
        pitch = mix(held.pitch, action.pitch)
        yaw = mix(held.yaw, action.yaw)
        roll = mix(held.roll, action.roll)

        # Заряд тратится по ЗАПРОШЕННОМУ отклонению, а не по фактическому:
        # маховик honestly пытается отработать команду и тянет ток даже
        # тогда, когда крутящего момента уже не хватает. Иначе разряд сам
        # себя ограничивал бы и до нуля дело не доходило никогда.
        self.requested = abs(pitch) + 0.5 * abs(yaw) + 0.5 * abs(roll)

        # Разряженная батарея = мёртвые сервоприводы и замерший дроссель:
        # зонд перестаёт принимать команды, и всё остаётся как было.
        # Авторитет возвращается постепенно, по мере подзарядки.
        authority = (self.electric - self.EC_DEAD) / (self.EC_BROWNOUT
                                                      - self.EC_DEAD)
        authority = min(1.0, max(0.0, authority))
        lost = authority <= 0.0
        if lost and not self.control_lost:
            self.blackouts += 1
        self.control_lost = lost
        throttle = self.throttle + authority * (throttle - self.throttle)
        pitch *= authority
        yaw *= authority
        roll *= authority

        # Увод рулей: заклинивший привод тянет в свою сторону, и команда
        # складывается с этим уводом, а не заменяет его.
        if self.fault.control_bias:
            pitch = max(-1.0, min(1.0, pitch + self.fault.control_bias))

        # Отделение — событие, а не уровень: оно либо успело за этот шаг,
        # либо переносится на следующий.
        stage = self.pending_stage
        if action.stage and authority > 0.0:
            if lag <= self.dt:
                stage = True
                self.pending_stage = False
            else:
                self.pending_stage = True
        elif stage:
            self.pending_stage = False

        self.held = action
        return FlightAction(throttle, pitch, yaw, roll, stage)

    # ------------------------------------------------------------------
    # 2. Таймаут ступеней
    # ------------------------------------------------------------------
    def dead_stage(self) -> bool:
        """Ступень, которая уже не тянет и годится только на сброс.

        Обычное правило мира — ступень отделяют пустой: везти полный бак
        вниз незачем, и сеть не должна учиться выбрасывать топливо. При
        мёртвой камере правило обратное: бак полон, но тяги с него не
        будет никогда, и каждая его тонна — это мёртвый груз. В игре
        такую ступень сбрасывают вместе с топливом, и здесь тоже.

        Порог по тяге, а не по нулю: двигатель, потерявший половину, ещё
        работает, и выбрасывать его — ошибка. Решает сеть, мир только
        разрешает.
        """
        return (self.fault.stage_bound == self.stage_index
                and self.fault.engine_scale < 0.35)

    def can_separate(self) -> bool:
        if self.fault.stage_stuck:
            self.stage_blocked += 1
            return False
        if self.time < self.stage_ready_at:
            self.stage_blocked += 1
            return False
        if self.dead_stage():
            return (self.current_stage is not None
                    and self.stage_index < len(self.stages) - 1)
        return super().can_separate()

    def separate(self) -> None:
        super().separate()
        self.fault.separated(self.stage_index)
        self.stage_ready_at = self.time + self.STAGE_COOLDOWN
        # межступенчатый разъём тоже питается от батареи
        self.electric = max(0.0, self.electric - 5.0)

    def action_mask(self) -> list[bool]:
        allowed = super().action_mask()
        if self.time < self.stage_ready_at:
            allowed[1] = False
        # На сломанной ракете та же кнопка означает увод кабины, и
        # разрешена она тогда, когда обычное отделение невозможно, —
        # иначе маска запрещала бы единственный способ спастись.
        if self.abort_available():
            allowed[1] = True
        return allowed

    def drag_coefficient(self, mach: float) -> float:
        if self.chute_open:
            return self.CHUTE_CD
        if self.aborted:
            return self.CAPSULE_CD
        return super().drag_coefficient(mach)

    def abort_available(self) -> bool:
        """Стала ли кнопка отделения кнопкой увода кабины.

        Правило одно: на сломанной ракете, когда сбрасывать нечего,
        отделение означает увод. Пока под ногами есть отработавшая
        ступень, кнопка работает по прямому назначению — сбросить её и
        лететь дальше; как только сбрасывать нечего, та же команда
        уводит кабину.

        Так остаются обе возможности сразу, и выбирает между ними сеть.
        Привязывать увод к виду отказа было ошибкой: при уводе рулей или
        севшей батарее кнопка оставалась недоступной, и экипаж на
        сорвавшемся полёте было уже нечем спасать — замер показал восемь
        таких безвыходных потерь из тридцати шести.

        «Сбрасывать нечего» здесь означает устройство ракеты, а не
        секундомер: пауза межступенчатого разъёма отделение запрещает,
        но ступень под ногами от этого никуда не делась. Считать такую
        паузу безвыходностью значило бы открыть увод кабины на пару
        секунд после каждого отделения — и сеть научилась бы ждать их.
        """
        if not (self.FAILURES and self.fault.broken and not self.aborted
                and self.airborne and not self.reached_orbit):
            return False
        if self.stage_index < len(self.stages) - 1 and not self.fault.stage_stuck:
            # Ступень под ногами есть — кнопка сбрасывает её, и это
            # правильный ход. Кроме одного случая: полёт уже кончился,
            # аппарат пошёл вниз или его опрокинуло. Ступенями этого не
            # исправить, а запереть людей на падающей ракете ради
            # экономии железа нельзя.
            if self._vertical_speed() > 0.0 and not self.tumbled:
                return False
            return True
        return not self.can_separate()

    # ------------------------------------------------------------------
    # 3. Ветер, 5. инерция тяги, 6. расход заряда
    # ------------------------------------------------------------------
    def apply_command(self, command: FlightAction) -> None:
        self.fire_failure()
        # Команда отделения на сломанной ракете, когда сбрасывать нечего,
        # и есть увод кабины. Отдельной кнопки у сети нет и не будет:
        # набор действий обязан совпадать с тем, что уходит в игру.
        if command.stage and self.abort_available():
            self.abort_capsule()
        super().apply_command(command)
        self.spend_electricity(command)
        self.blow_wind()
        self.spool_engine()
        self.leak_fuel()
        self.update_chute()

    def spend_electricity(self, command: FlightAction) -> None:
        """Заряд тратится на удержание рулей и — втрое дороже — на перекладку.

        Плавное управление держится на генераторе двигателя бесконечно.
        Дёрганья от упора до упора съедают батарею за пару секунд, и
        аппарат остаётся без управления в самый неподходящий момент.
        """
        deflection = self.requested
        change = abs(deflection - self.last_deflection)
        self.last_deflection = deflection

        drain = self.EC_HOLD * deflection * self.dt + self.EC_MOVE * change
        burning = self.effective_throttle() > 0.01 and self.fuel_left() > 0.0
        regen = (self.EC_ALTERNATOR if burning else self.EC_SOLAR) * self.dt
        self.electric = max(0.0, min(self.EC_CAPACITY,
                                     self.electric - drain + regen))
        self.min_electric = min(self.min_electric, self.electric)

    def ou_gust(self, state: float) -> tuple[float, float]:
        """Порыв ветра: процесс Орнштейна-Уленбека, градусы сноса.

        Белый шум сеть просто усредняет и игнорирует. Порыв, который
        держится несколько секунд, приходится реально парировать — и
        именно этого требует живая атмосфера Кербина. Возвращается
        (новое состояние, ПРИРАЩЕНИЕ сноса за шаг): сам снос не должен
        накапливаться случайным блужданием, он держится в пределах пары
        градусов и слабеет вместе с плотностью воздуха.
        """
        dt, tau = self.dt, self.WIND_TAU
        updated = state + (-state * dt / tau
                           + self.WIND_DEG * math.sqrt(2.0 * dt / tau)
                           * self.rng.gauss(0.0, 1.0))
        air = min(1.0, self.density(self.altitude) / KERBIN_SEA_DENSITY)
        return updated, (updated - state) * air

    def gust(self) -> float:
        self.wind_state, shift = self.ou_gust(self.wind_state)
        return shift

    def blow_wind(self) -> None:
        if not self.airborne:
            return                      # на столе аппарат держат опоры
        shift = self.gust()
        self.wind_offset += shift
        self.pitch = max(-90.0, min(90.0, self.pitch + shift))

    def spool_engine(self) -> None:
        """Тяга догоняет команду по экспоненте с постоянной 0.5-1 с."""
        alpha = 1.0 - math.exp(-self.dt / self.spool_tau)
        self.actual_throttle += (self.throttle - self.actual_throttle) * alpha
        self.actual_throttle = max(0.0, min(1.0, self.actual_throttle))

    def effective_throttle(self) -> float:
        # Отказ двигателя входит сюда же: мёртвая камера не даёт ни тяги,
        # ни тока с генератора — ровно как в игре.
        return self.actual_throttle * self.fault.engine_scale

    def electric_fraction(self) -> float:
        return self.electric / self.EC_CAPACITY

    # ------------------------------------------------------------------
    # 4. Аэродинамическое скольжение
    # ------------------------------------------------------------------
    def ease(self, strict: float, easy: float) -> float:
        """Значение между щадящим и строгим по текущей жёсткости среды."""
        return easy + (strict - easy) * self.difficulty

    def shaping_reward(self, command: FlightAction) -> float:
        # плата за неработающие оси — общая для всех плоских ступеней
        penalty = super().shaping_reward(command)
        altitude = self.altitude
        q = self.dynamic_pressure()

        if altitude < self.AOA_DENSE_ALTITUDE and q > 1_000.0:
            limit = self.ease(self.AOA_LIMIT, self.AOA_LIMIT_EASY)
            excess = self.angle_of_attack() - limit
            if excess > 0.0:
                self.aoa_violations += 1
                load = min(1.0, q / 20_000.0)
                weight = self.ease(self.AOA_PENALTY, self.AOA_PENALTY_EASY)
                penalty -= weight * (excess / 10.0) * load

        if self.electric <= 0.0:
            penalty -= 0.02             # обесточенный аппарат неуправляем
        return penalty

    def check_failure(self) -> tuple[float, str] | None:
        """Опрокидывание: нос поперёк потока при большом напоре.

        В KSP это выглядит как мгновенный кувырок и разрушение — здесь
        эпизод просто обрывается с крупным штрафом.
        """
        failure = super().check_failure()      # разрушение напором — общее
        if failure is not None:
            return failure
        # Кабина тупым концом вперёд — это её штатное положение, а не
        # потеря устойчивости: «правильного» угла атаки у неё нет.
        if self.aborted:
            return None
        limit = self.ease(self.AOA_TUMBLE, self.AOA_TUMBLE_EASY)
        if (self.dynamic_pressure() > self.TUMBLE_PRESSURE
                and self.angle_of_attack() > limit):
            self.tumbled = True
            return 8.0, "аппарат опрокинуло потоком"
        return None

    # ------------------------------------------------------------------
    def episode_summary(self) -> dict:
        summary = super().episode_summary()
        summary.update({
            **self.fault.summary(),
            "crew": self.crew,
            "crew_saved": bool(self.reached_orbit or self.landed_safely)
                          if self.crew else None,
            "landed_safely": self.landed_safely,
            "min_electric": round(self.min_electric, 1),
            "blackouts": self.blackouts,
            "aoa_violations": self.aoa_violations,
            "stage_blocked": self.stage_blocked,
            "wind_offset": round(self.wind_offset, 2),
            "tumbled": self.tumbled,
            "recoverable": self.recoverable,
        })
        return summary


__all__ = ["HardcoreAscentSimulator", "SimStage", "SPENT_STAGE_FUEL"]
