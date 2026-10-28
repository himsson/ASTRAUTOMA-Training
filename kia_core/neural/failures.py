"""ОТКАЗЫ В ПОЛЁТЕ — седьмая механика Тренажёра+.

Остальные шесть механик (задержка, ветер, батарея, раскрутка двигателя,
таймаут отделения, скольжение) делают мир неудобным, но честным: он
ведёт себя одинаково от начала до конца эпизода. Отказ устроен иначе —
это событие, которое меняет правила посреди полёта:

    engine_out    двигатель текущей ступени гаснет насовсем
    thrust_loss   тяга падает до доли номинала (прогар сопла)
    stage_stuck   ступень не отделяется, сколько ни командуй
    power_loss    генератор мёртв, заряд только из батареи
    fuel_leak     топливо уходит мимо камеры сгорания
    control_bias  рули уводит в одну сторону

Ничего из этого сеть не получает отдельным входом. Отказ виден ровно
так же, как в игре: тяга в приборах упала, ступень не ушла, заряд тает.
Распознать происходящее — часть задачи, а не подарок к ней.

Почему вообще: политика, обученная в мире без отказов, на отказе ведёт
себя как ни в чём не бывало — продолжает програму тангажа с мёртвым
двигателем и теряет аппарат. Решение принимается за секунды, и учить
ему надо там же, где оно принимается.

Экипаж. У аппарата появляется признак `crew`, и цена аварии зависит от
него: потерять беспилотный зонд — минус деньги, потерять экипаж —
минус, несопоставимый ни с какой экономией. Отсюда и порядок действий,
которого мы ждём от сети: сначала увести людей на спасаемую траекторию,
и только потом беречь ступени.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Доля эпизодов с отказом при полной жёсткости. Выше половины делать
# нельзя: сеть начнёт считать отказ нормой и возить лишний запас на
# каждый полёт, а на исправных ракетах это прямой проигрыш по массе.
FAILURE_RATE = 0.35

# Когда отказ случается: доля от ожидаемой длительности выведения.
# Раньше 5 % — это ещё стартовый стол, позже 90 % — уже почти орбита,
# и там любой отказ дёшев. Интересное лежит между.
WINDOW = (0.05, 0.90)

KINDS = ("engine_out", "thrust_loss", "stage_stuck", "power_loss",
         "fuel_leak", "control_bias")

# Веса подобраны по тому, что реально случается в KSP: прогары и
# заглохшие двигатели — обычное дело, залипший сепаратор — редкость.
WEIGHTS = (0.26, 0.24, 0.12, 0.14, 0.14, 0.10)


@dataclass
class Failure:
    """Один отказ: что, когда и насколько сильно."""
    kind: str
    at: float                  # секунда полёта, когда это случится
    severity: float = 1.0      # 0..1 — насколько всё плохо
    fired: bool = False

    def describe(self) -> str:
        return f"{self.kind} на {self.at:.0f} с (сила {self.severity:.2f})"


@dataclass
class FailureState:
    """Что сейчас сломано. Читается физикой, не агентом."""
    engine_scale: float = 1.0      # множитель тяги: 0 — двигатель мёртв
    stage_stuck: bool = False      # сепаратор не срабатывает
    alternator: bool = True        # генератор жив
    leak_per_second: float = 0.0   # кг/с мимо камеры
    control_bias: float = 0.0      # постоянный увод рулей, доля команды
    # Номер ступени, вместе с которой уйдёт поломка. Прогар сопла, мёртвая
    # камера и течь принадлежат той ступени, которая горела: сбросишь её —
    # следующая запустится целой, и в игре это первое, что делает пилот.
    # Залипший сепаратор, мёртвый генератор и уведённые рули остаются с
    # аппаратом: они не в баке.
    stage_bound: int | None = None
    history: list = field(default_factory=list)

    def separated(self, stage_index: int) -> bool:
        """Ступень ушла — ушла ли с ней поломка. True, если полегчало."""
        if self.stage_bound is None or stage_index <= self.stage_bound:
            return False
        self.engine_scale = 1.0
        self.leak_per_second = 0.0
        self.stage_bound = None
        return True

    @property
    def broken(self) -> bool:
        return (self.engine_scale < 1.0 or self.stage_stuck
                or not self.alternator or self.leak_per_second > 0.0
                or self.control_bias != 0.0)

    def summary(self) -> dict:
        return {"failures": list(self.history), "broken": self.broken,
                "engine_scale": round(self.engine_scale, 3)}


def roll(rng: random.Random, difficulty: float, flight_time: float,
         rate: float | None = None) -> Failure | None:
    """Бросок на отказ для одного эпизода.

    difficulty 0 — отказов нет вообще (так устроена первая ступень
    программы и все прежние замеры), 1 — полная доля. Долю задаёт
    ступень: на обычной это FAILURE_RATE, на той, где учат спасать
    экипаж, — единица.
    """
    share = FAILURE_RATE if rate is None else float(rate)
    if difficulty <= 0.0 or rng.random() > share * difficulty:
        return None
    kind = rng.choices(KINDS, weights=WEIGHTS, k=1)[0]
    at = flight_time * rng.uniform(*WINDOW)
    # Слабый отказ на лёгкой жёсткости, полный — на строгой: сеть
    # сначала встречает падение тяги на четверть, а не мгновенную смерть
    # двигателя.
    severity = rng.uniform(0.35, 1.0) * (0.4 + 0.6 * difficulty)
    return Failure(kind=kind, at=at, severity=min(1.0, severity))


def apply(failure: Failure, state: FailureState, stage_fuel: float) -> str:
    """Переводит сработавший отказ в состояние мира. Возвращает описание."""
    s = failure.severity
    if failure.kind == "engine_out":
        state.engine_scale = 0.0
    elif failure.kind == "thrust_loss":
        state.engine_scale = min(state.engine_scale, 1.0 - 0.8 * s)
    elif failure.kind == "stage_stuck":
        state.stage_stuck = True
    elif failure.kind == "power_loss":
        state.alternator = False
    elif failure.kind == "fuel_leak":
        # за минуту утекает от трети до всего остатка ступени
        state.leak_per_second = max(0.0, stage_fuel) * (0.3 + 0.7 * s) / 60.0
    elif failure.kind == "control_bias":
        state.control_bias = 0.35 * s * (1 if failure.at % 2 < 1 else -1)
    failure.fired = True
    state.history.append(failure.describe())
    return failure.describe()
