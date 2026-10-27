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
                 failures: bool | None = None, **kwargs):
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
        self.roll_failure()

    # ------------------------------------------------------------------
    # 7. Отказы в полёте
    # ------------------------------------------------------------------
    # Порог мягкой посадки — тот же, по которому базовый тренажёр считает
    # удар смертельным. Ниже него касание грунта переживают и аппарат,
    # и люди.
    SOFT_TOUCHDOWN = 12.0

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

        if not done and self.airborne and self.altitude <= 0.5:
            self.done = True
            done = True
            self.landed_safely = True
            info["reason"] = "мягкая посадка"
            reward += self.crew_reward(saved=True)
        elif done and not self.reached_orbit and not self.landed_safely:
            reward += self.crew_reward(saved=False)
        return obs, reward, done, info

    def crew_reward(self, saved: bool) -> float:
        """Цена исхода. Люди несопоставимы ни с какой экономией массы.

        Беспилотный аппарат — это деньги: за спасённое железо платят,
        за потерянное вычитают, и счёт идёт в тоннах. Экипаж в эту
        шкалу не кладётся: +60 за живых против −120 за погибших больше
        любой экономии, какую может дать риск. Сеть, которой выгодно
        поберечь ступень ценой людей, — это ровно то, чего быть не должно.
        """
        if self.crew:
            return 60.0 if saved else -120.0
        mass = max(0.0, self.mass) / 1000.0        # тонны
        return 2.0 * mass if saved else -1.0 * mass

    def roll_failure(self) -> None:
        """Бросок на отказ для этого эпизода (см. neural/failures.py)."""
        from .failures import FailureState, roll
        self.fault = FailureState()
        self.pending_failure = None
        self.landed_safely = False
        # Экипаж в половине эпизодов: сеть должна одинаково уверенно
        # вести и беспилотник, где можно рискнуть железом, и пилотируемый
        # аппарат, где рисковать нечем.
        self.crew = self.FAILURES and self.rng.random() < 0.5
        if not self.mechanics or not self.FAILURES:
            return
        # Ожидаемая длительность выведения — ориентир, когда «интересно»
        # ломаться. Точность тут не нужна: отказ всё равно раскидывается
        # по окну в несколько минут.
        self.pending_failure = roll(self.rng, self.difficulty, 300.0)

    def fire_failure(self) -> None:
        """Пора ли сработать отказу. Зовётся каждый шаг до физики."""
        from .failures import apply
        pending = self.pending_failure
        if pending is None or pending.fired or self.time < pending.at:
            return
        stage = self.current_stage
        apply(pending, self.fault, stage.fuel_mass if stage else 0.0)
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
    def can_separate(self) -> bool:
        if self.fault.stage_stuck:
            self.stage_blocked += 1
            return False
        if self.time < self.stage_ready_at:
            self.stage_blocked += 1
            return False
        return super().can_separate()

    def separate(self) -> None:
        super().separate()
        self.stage_ready_at = self.time + self.STAGE_COOLDOWN
        # межступенчатый разъём тоже питается от батареи
        self.electric = max(0.0, self.electric - 5.0)

    def action_mask(self) -> list[bool]:
        allowed = super().action_mask()
        if self.time < self.stage_ready_at:
            allowed[1] = False
        return allowed

    # ------------------------------------------------------------------
    # 3. Ветер, 5. инерция тяги, 6. расход заряда
    # ------------------------------------------------------------------
    def apply_command(self, command: FlightAction) -> None:
        self.fire_failure()
        super().apply_command(command)
        self.spend_electricity(command)
        self.blow_wind()
        self.spool_engine()
        self.leak_fuel()

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
        })
        return summary


__all__ = ["HardcoreAscentSimulator", "SimStage", "SPENT_STAGE_FUEL"]
