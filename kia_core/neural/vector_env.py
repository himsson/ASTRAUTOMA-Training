"""Пачка тренажёров, работающих одновременно.

ЗАЧЕМ. Замер на боевой конфигурации показал, что 94 % времени обучения
уходило не на физику и не на градиенты, а на обращения к сети по одному
наблюдению: 3.3 миллисекунды на решение при том, что сама физика шага
занимает около 30 микросекунд. Сеть маленькая, и почти всё это время —
накладные расходы: собрать тензор, вызвать слои, вернуть результат в
python. Расход этот ПОСТОЯННЫЙ и почти не зависит от того, спрашиваем мы
про одну ракету или про шестнадцать.

Отсюда решение: держать несколько независимых тренажёров и спрашивать
сеть сразу обо всех. Шестнадцать ракет за один вызов — это те же 3 мс,
но уже на шестнадцать решений.

ЧТО ЭТО НЕ МЕНЯЕТ. Физика, награды и правила остаются ровно теми же:
каждая копия — обычный симулятор ступени со своим случайным аппаратом.
Копии независимы, между собой не общаются и завершаются в разное время;
закончившаяся немедленно перезапускается, чтобы пачка не простаивала.

ПОЧЕМУ БЕЗ ОТДЕЛЬНЫХ ПРОЦЕССОВ. Шаг физики стоит десятки микросекунд —
дешевле, чем переслать наблюдение между процессами. Многопроцессность
здесь только добавила бы задержек и съела бы ядра, которые нужны
человеку за компьютером.
"""
from __future__ import annotations

from ..logging_setup import get_logger
from .spaces import FlightAction

log = get_logger("neural.vector_env")

# Сколько копий держать по умолчанию. Больше — быстрее, но каждая копия
# добавляет работы процессору; на этом числе выигрыш от пачки уже почти
# исчерпан, а машина остаётся отзывчивой.
DEFAULT_ENVS = 16


class VectorSimEnv:
    """N независимых копий одного тренажёра с общим интерфейсом."""

    def __init__(self, stage, count: int = DEFAULT_ENVS,
                 goal_altitude: float = 80_000.0, seed: int | None = None,
                 exploring_starts: float = 0.5):
        self.stage = stage
        self.count = max(1, int(count))
        self.goal_altitude = goal_altitude
        base = 0 if seed is None else int(seed)
        # Разные зёрна: иначе все копии проживут одинаковую жизнь и пачка
        # будет стоить как одна среда
        self.envs = [stage.make_env(goal_altitude=goal_altitude,
                                    seed=base + index,
                                    exploring_starts=exploring_starts)
                     for index in range(self.count)]
        self.episode_returns = [0.0] * self.count
        self.episode_steps = [0] * self.count
        # Счётчик шагов последнего вызова step(). Заводится сразу: в
        # совместном режиме CPU+GPU пачка идёт через step_range(), и
        # тренер, спросив last_steps до первого step(), получал
        # AttributeError вместо числа.
        self.last_steps = 0

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.count

    def reset(self) -> list:
        self.episode_returns = [0.0] * self.count
        self.episode_steps = [0] * self.count
        return [env.reset() for env in self.envs]

    def action_mask(self) -> list:
        return [env.action_mask() for env in self.envs]

    def set_difficulty(self, value: float) -> None:
        for env in self.envs:
            env.set_difficulty(value)

    # ------------------------------------------------------------------
    def step_range(self, start: int, stop: int, actions: list,
                   action_repeat: int = 1):
        """Шаг только для копий [start, stop). Возвращает те же четыре списка.

        Нужен совместному режиму: пока видеокарта считает действия для
        хвоста пачки, процессор успевает просчитать физику головы. Без
        такого разделения шаг просто ждал бы видеокарту.
        """
        observations: list = []
        rewards: list = []
        dones: list = []
        infos: list = []
        steps = 0

        for offset, index in enumerate(range(start, stop)):
            env = self.envs[index]
            action = actions[offset]
            reward = 0.0
            done = False
            info: dict = {}
            observation = None
            for repeat in range(action_repeat):
                step_action = action if repeat == 0 else FlightAction(
                    action.throttle, action.pitch, action.yaw, action.roll,
                    False)
                observation, step_reward, done, info = env.step(step_action)
                reward += step_reward
                steps += 1
                if done:
                    break

            self.episode_returns[index] += reward
            self.episode_steps[index] += 1
            if done:
                info = dict(info)
                info["episode_return"] = self.episode_returns[index]
                info["episode_steps"] = self.episode_steps[index]
                # Итог эпизода снимаем ДО перезапуска: по нему копилка
                # знаний понимает, что именно сеть сумела в этом полёте
                info["summary"] = env.episode_summary()
                self.episode_returns[index] = 0.0
                self.episode_steps[index] = 0
                observation = env.reset()

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return observations, rewards, dones, infos, steps

    def step(self, actions: list, action_repeat: int = 1):
        """Шаг всей пачки. Возвращает (наблюдения, награды, концы, детали).

        Решение держится `action_repeat` шагов, как и в одиночной среде;
        отделение ступени выполняется только на первом из них. Копия,
        закончившая эпизод, тут же перезапускается — в наблюдении уже
        стоит начало нового полёта, а итог прошлого лежит в `info`.
        """
        observations: list = []
        rewards: list = []
        dones: list = []
        infos: list = []
        steps = 0

        for index, env in enumerate(self.envs):
            action = actions[index]
            reward = 0.0
            done = False
            info: dict = {}
            observation = None
            for repeat in range(action_repeat):
                step_action = action if repeat == 0 else FlightAction(
                    action.throttle, action.pitch, action.yaw, action.roll,
                    False)
                observation, step_reward, done, info = env.step(step_action)
                reward += step_reward
                steps += 1
                if done:
                    break

            self.episode_returns[index] += reward
            self.episode_steps[index] += 1
            if done:
                info = dict(info)
                info["episode_return"] = self.episode_returns[index]
                info["episode_steps"] = self.episode_steps[index]
                # Итог эпизода снимаем ДО перезапуска: по нему копилка
                # знаний понимает, что именно сеть сумела в этом полёте
                info["summary"] = env.episode_summary()
                self.episode_returns[index] = 0.0
                self.episode_steps[index] = 0
                observation = env.reset()

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        self.last_steps = steps
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return (f"{self.count} копий среды «{self.stage.title}», "
                f"решение раз в {self.stage.decision_seconds:.1f} с")


__all__ = ["VectorSimEnv", "DEFAULT_ENVS"]
