"""Циклы обучения нейросети KIA.

Режимы:

* `train_stage`      — ступень учебной программы: перенести веса
                       предыдущей ступени и доучить под новые правила
                       (см. neural/curriculum.py);
* `pretrain_pilot`   — базовый тренажёр без переноса весов;
* `train_builder`    — сеть-конструктор на палитре деталей из игры;
* `train_in_ksp`     — дообучение в живой KSP (медленно, зато по-настоящему).

Все полётные режимы гоняют один и тот же цикл `run_flight_training`:
собрать пакет шагов → посчитать преимущества → несколько эпох
градиентного спуска → сохранить веса в kia_model.pth. Отличаются они
только средой и тем, как долго держится одно решение сети.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import DATA_DIR
from ..logging_setup import get_logger
from .builder_env import FINISH_ACTION, BuilderEnv, build_palette
from .ppo import Brain, PPOConfig, RolloutBuffer, build_brain
from .sim_env import AscentSimulator
from .spaces import FlightAction

log = get_logger("neural.trainer")

METRICS_FILE = DATA_DIR / "neural_metrics.json"

# Как часто сбрасывать веса на диск, секунды. Чаще смысла нет, а папка
# проекта синхронизируется OneDrive — лишние записи он гоняет в облако.
SAVE_INTERVAL = 30.0


def log_std_ceiling(agent, stage_number: int) -> float:
    """Потолок разброса действий для ступени — но без раздувания назад.

    Потолок задан по ступеням: на первой он широкий (σ 0.368), потому что
    там учатся с нуля и разведка необходима, на последней узкий (σ 0.135),
    потому что момент требует точности. Беда начинается, когда на ПЕРВУЮ
    ступень возвращается уже обученная политика — при откате или при
    прогоне программы заново. Потолок поднимается вдвое, а бонус за
    энтропию, по закону из MIGRATION 5.3d, всегда прижимает разброс к
    потолку. Сошедшаяся политика получает обратно вдвое больше шума.

    Замер, ради которого это написано (06.08, 12:50). Веса вернулись на
    ступень 1 с σ 0.165, потолок поднялся до 0.368 — и σ поползла вверх:

        12:49  σ 0.266   доля орбит за окно  8 %
        12:50  σ 0.278                      14 %
        12:50  σ 0.293                       2 %
        12:50  σ 0.295                       0 %

    При этом БЕЗ ШУМА та же политика давала 11 орбит из 40 (28 %). То
    есть разваливалась не политика, а обучающие эпизоды вокруг неё — и
    экзамен, который считает долю орбит по ним, стал недостижим.

    Поэтому: у мозга с историей обучения потолок не может оказаться выше
    того разброса, который у него уже есть. Сузиться — пожалуйста,
    раздуться обратно — нет. Свежая сеть (обновлений ноль) правила не
    замечает и получает полную разведку своей ступени.
    """
    from .ppo import log_std_max_for_stage

    ceiling = log_std_max_for_stage(stage_number)
    net = getattr(agent, "net", None)
    if net is None or net.log_std is None or agent.stats.updates <= 0:
        return ceiling
    current = float(net.log_std.max().item())
    return min(ceiling, current)


@dataclass
class TrainingReport:
    mode: str
    iterations: int = 0
    episodes: int = 0
    steps: int = 0
    mean_return: float = 0.0
    best_return: float = float("-inf")
    successes: int = 0
    wall_time: float = 0.0
    stage: int = 0
    envs: int = 1
    axis_means: list = field(default_factory=list)
    difficulty: float = 1.0
    error: str = ""
    stop_reason: str = ""
    updates: list = field(default_factory=list)
    failures: dict = field(default_factory=dict)

    def describe(self) -> str:
        if self.error:
            return f"{self.mode}: не начато — {self.error}"
        best = self.best_return if self.best_return > float("-inf") else 0.0
        text = (f"{self.mode}: {self.episodes} эпизодов, {self.steps} шагов, "
                f"средняя награда {self.mean_return:.2f}, лучшая {best:.2f}, "
                f"успехов {self.successes}, {self.wall_time:.0f} с")
        if self.failures:
            top = sorted(self.failures.items(), key=lambda item: -item[1])[:4]
            text += "\n  чаще всего срывалось: " + ", ".join(
                f"{reason} ×{count}" for reason, count in top)
        return text


class NeuralTrainer:
    """Тренер: держит мозг, гоняет среды, обновляет веса, пишет метрики."""

    def __init__(self, brain: Brain | None = None, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        self.brain = brain or build_brain(self.config)
        self.stop_flag = threading.Event()
        self.last_report: TrainingReport | None = None
        self.watchdog = None          # дежурный: сводка раз в пять минут

    def stop(self) -> None:
        self.stop_flag.set()

    def begin_session(self, pipeline=None) -> None:
        """Начало новой работы по команде оператора: снимаем флаг остановки.

        Раньше флаг снимался внутри самого цикла обучения — и это был
        тихий способ проглотить команду «остановить». Конвейер заходит в
        тренера десятки раз подряд; «стоп», нажатый между двумя заходами,
        снимался следующим же заходом, и обучение продолжалось как ни в
        чём не бывало. Теперь флаг снимает только тот, кто ЗАПУСКАЕТ
        работу: пульт, командная строка, конвейер в начале run().

        Здесь же заступает на смену дежурный (`neural/watchdog.py`) —
        он раз в несколько минут пишет одну строку о том, идёт всё как
        надо или уже нет.
        """
        self.stop_flag.clear()
        self.start_watchdog(pipeline)

    def start_watchdog(self, pipeline=None) -> None:
        """Поднимает дежурного (один на процесс). Выключается KIA_WATCHDOG=0."""
        from .watchdog import Watchdog

        if self.watchdog is None:
            self.watchdog = Watchdog(self, pipeline=pipeline)
            if self.watchdog.start():
                period = self.watchdog.period
                often = (f"{period:.0f} с" if period < 60
                         else f"{period / 60:.0f} мин")
                log.info("Дежурный на смене: сводка раз в %s "
                         "(выключить — переменная среды KIA_WATCHDOG=0)", often)
        elif pipeline is not None:
            self.watchdog.pipeline = pipeline

    @property
    def running(self) -> bool:
        return not self.stop_flag.is_set()

    # ==================================================================
    # Учебная программа: три ступени тренажёра
    # ==================================================================
    def train_stage(self, stage, total_steps: int | None = None,
                    rollout: int | None = None, goal_altitude: float = 80_000.0,
                    progress=None, promote: bool = True,
                    on_episode=None, difficulty_floor: float = 0.0) -> TrainingReport:
        """Обучение на указанной ступени программы.

        promote=True означает «перед стартом сделать снимок весов прошлой
        ступени и продолжить обучение с них». Именно этим ступень
        отличается от отдельного прогона: сеть не начинает заново, а
        доучивается под новые правила мира.
        """
        from .curriculum import prepare

        if promote:
            promotion = prepare(stage, self.brain)
            for line in promotion.messages:
                log.info("[%s] %s", stage.title, line)
            if not promotion.ok:
                report = TrainingReport(mode=stage.title)
                report.error = "; ".join(promotion.messages)
                self.last_report = report
                return report

        from .vector_env import VectorSimEnv

        envs = VectorSimEnv(stage, count=stage.envs,
                            goal_altitude=goal_altitude)
        return self.run_vector_training(
            envs, total_steps=total_steps or stage.default_steps,
            rollout=rollout or stage.rollout,
            action_repeat=stage.action_repeat, label=stage.title,
            stage_number=stage.number, warmup=stage.warmup, progress=progress,
            on_episode=on_episode, difficulty_floor=difficulty_floor)

    def pretrain_pilot(self, total_steps: int = 200_000, rollout: int = 2048,
                       goal_altitude: float = 80_000.0, action_repeat: int = 5,
                       progress=None) -> TrainingReport:
        """Базовый тренажёр (первая ступень программы) без переноса весов."""
        from .curriculum import stage_by_number
        from .vector_env import VectorSimEnv

        stage = stage_by_number(1)
        envs = VectorSimEnv(stage, count=stage.envs, goal_altitude=goal_altitude)
        return self.run_vector_training(
            envs, total_steps=total_steps, rollout=rollout,
            action_repeat=action_repeat, label="Базовый Тренажёр",
            stage_number=1, progress=progress)

    # ------------------------------------------------------------------
    def run_vector_training(self, envs, total_steps: int, rollout: int,
                            action_repeat: int, label: str,
                            stage_number: int = 1, warmup: float = 0.0,
                            progress=None, on_episode=None,
                            difficulty_floor: float = 0.0) -> TrainingReport:
        """Цикл PPO по ПАЧКЕ тренажёров.

        Раньше здесь стоял цикл по одной среде, и 94 % времени уходило не
        на физику и не на градиенты, а на обращения к сети по одному
        наблюдению — 3.3 мс на решение при цене шага физики в 30 мкс.
        Расход на вызов сети почти не зависит от размера пачки, поэтому
        шестнадцать ракет обходятся почти как одна.

        action_repeat — сколько шагов среды держится одно решение сети.
        Это не подсказка, а приём разведки. При независимом шуме на каждом
        шаге средняя тяга всегда около половины, ракета не отрывается, и
        сеть не видит, что бывает иначе. Удерживая решение секунду, агент
        хотя бы иногда даёт длинный полный газ — и получает сигнал.

        warmup — доля прогона, за которую среда набирает полную жёсткость.

        difficulty_floor — жёсткость, ниже которой опускаться нельзя.
        Разогрев вводит новые механики ступени и проходится ОДИН раз; при
        повторном заходе на ту же ступень сеть их уже видела, и откат
        среды в щадящий режим превращает мир в качели, за которыми
        политика гоняется вместо того, чтобы учиться.

        on_episode(награда, info, отчёт) вызывается после каждого эпизода
        ЛЮБОЙ копии. Если вернёт непустую строку — обучение
        останавливается с этой причиной. Через этот хук автономный
        конвейер ловит момент сдачи экзамена и момент тупика.
        """
        from . import brain_map
        from .knowledge import SIM_KNOWLEDGE as knowledge
        from .ppo import (VectorRolloutBuffer, entropy_coef_for_stage,
                          learning_rate_for_stage, log_std_max_for_stage)

        agent = self.brain.pilot
        count = len(envs)
        # Один заход буфера — это rollout решений на КАЖДУЮ копию, то есть
        # rollout x копий x action_repeat шагов среды. Отжиг LR считаем по
        # этому же числу, иначе расписание разъедется с реальной длиной.
        per_pass = max(1, rollout * count * action_repeat)
        planned = max(10, total_steps // per_pass)
        agent.config.total_updates_planned = planned
        # ...и отсчитываться от начала ЭТОГО прогона, а не от рождения мозга
        agent.config.anneal_baseline = agent.stats.updates

        rate = learning_rate_for_stage(stage_number)
        if abs(rate - agent.config.learning_rate) > 1e-12:
            log.info("[%s] скорость обучения %.1e (была %.1e): поверх готовых "
                     "навыков шаг градиента должен быть мельче", label, rate,
                     agent.config.learning_rate)
            agent.config.learning_rate = rate
            for group in agent.optimizer.param_groups:
                group["lr"] = rate

        bonus = entropy_coef_for_stage(stage_number)
        if abs(bonus - agent.config.entropy_coef) > 1e-12:
            log.info("[%s] бонус за энтропию %.4f (был %.4f): точное "
                     "управление требует, чтобы политика сошлась, а не "
                     "разбрасывалась", label, bonus, agent.config.entropy_coef)
            agent.config.entropy_coef = bonus

        # Потолок разброса действий: чем точнее управление на ступени,
        # тем ближе выбранное действие должно быть к среднему. Иначе
        # обучается распределение, а не политика (см. ppo.STAGE_LOG_STD_MAX).
        # Обратно разброс не раздувается — см. log_std_ceiling.
        stage_ceiling = log_std_max_for_stage(stage_number)
        ceiling = log_std_ceiling(agent, stage_number)
        # Сообщать только о ЗАМЕТНОМ сужении. Веса лежат во float32, и
        # сравнение «строго меньше» срабатывало на разнице в седьмом знаке:
        # в лог шла бессмыслица «оставлен узким: sigma 0.368 при потолке
        # 0.368». Порог в один процент отделяет настоящее сужение от шума
        # округления.
        if math.exp(ceiling) < math.exp(stage_ceiling) * 0.99:
            log.info("[%s] разброс действий оставлен узким: sigma %.3f "
                     "(потолок ступени %.3f). Политика уже сошлась — "
                     "раздувать её обратно значит учить шум, а не полёт",
                     label, math.exp(ceiling), math.exp(stage_ceiling))
        if abs(ceiling - agent.net.LOG_STD_MAX) > 1e-12:
            log.info("[%s] потолок разброса действий %.2f (sigma %.3f), был %.2f",
                     label, ceiling, math.exp(ceiling), agent.net.LOG_STD_MAX)
            agent.net.LOG_STD_MAX = ceiling
        agent.net.clamp_log_std()

        self.brain.stage = stage_number
        mask_dim = agent.discrete_sizes[0] if agent.discrete_sizes else 0
        buffer = VectorRolloutBuffer(agent.obs_dim, agent.continuous_dim,
                                     len(agent.discrete_sizes), rollout,
                                     agent.device, envs=count,
                                     mask_dim=mask_dim)
        report = TrainingReport(mode=label, stage=stage_number)
        report.envs = count
        # отчёт публикуется сразу: пульт показывает ход обучения по нему,
        # а не только итог
        self.last_report = report
        started = time.time()
        observations = envs.reset()
        stop_reason = ""
        saved_steps = 0
        last_save = 0.0
        axis_sum = torch.zeros(agent.continuous_dim, device=agent.device)
        axis_count = 0
        # флаг остановки здесь НЕ снимается: это делает только тот, кто
        # запускает работу (см. begin_session)
        log.info("[%s] %s", label, envs.describe())
        brain_map.write(self.brain, report=report, knowledge=knowledge)

        while report.steps < total_steps and self.running and not stop_reason:
            # Жёсткость среды поднимается по ходу прогона: сначала штраф за
            # скольжение предупреждает, к концу разогрева — наказывает всерьёз
            ramp = (1.0 if warmup <= 0.0 else
                    min(1.0, report.steps / (warmup * total_steps)))
            report.difficulty = max(min(1.0, max(0.0, difficulty_floor)), ramp)
            envs.set_difficulty(report.difficulty)
            buffer.size = 0
            buffer.has_masks = False
            while not buffer.full and self.running and not stop_reason:
                # Маска запрещает заведомо гиблые ходы (сброс полной
                # ступени, стейджинг в блокировке). Она же уходит в буфер:
                # обновление PPO обязано видеть то же распределение.
                masks = envs.action_mask()
                head = agent.hybrid_cpu_count(count)

                def step_head(action, head=head):
                    """Физика головы пачки — считается в тени видеокарты.

                    Пока GPU занят своей долей наблюдений, процессору
                    незачем простаивать: он прогоняет физику тех копий,
                    решения по которым уже посчитал сам.
                    """
                    if head >= count:
                        return None
                    rows = action[0][:head].tolist()
                    picks = action[1][:head].tolist()
                    return envs.step_range(
                        0, head,
                        [FlightAction.decode(row, pick)
                         for row, pick in zip(rows, picks)],
                        action_repeat=action_repeat)

                (obs_t, continuous, discrete, logp, value,
                 done_head) = agent.act_many(observations, action_masks=masks,
                                             while_waiting=step_head)

                if done_head is None:
                    decoded = [FlightAction.decode(row, choice)
                               for row, choice in zip(continuous.tolist(),
                                                      discrete.tolist())]
                    observations, rewards, dones, infos = envs.step(
                        decoded, action_repeat=action_repeat)
                    report.steps += envs.last_steps
                else:
                    # голову уже просчитали, осталось догнать хвост
                    tail_rows = continuous[head:].tolist()
                    tail_picks = discrete[head:].tolist()
                    tail = envs.step_range(
                        head, count,
                        [FlightAction.decode(row, pick)
                         for row, pick in zip(tail_rows, tail_picks)],
                        action_repeat=action_repeat)
                    observations = done_head[0] + tail[0]
                    rewards = done_head[1] + tail[1]
                    dones = done_head[2] + tail[2]
                    infos = done_head[3] + tail[3]
                    report.steps += done_head[4] + tail[4]
                buffer.add_batch(obs_t, continuous, discrete, logp, value,
                                 rewards, dones, masks=masks)
                # Средняя команда по каждой оси. Прибор дешёвый, а ловит
                # целый класс ошибок: ось, которая на этой ступени ни на
                # что не влияет, спокойно упирается в единицу и выстрелит
                # на следующей. Так рыскание +0.98 развернуло аппарат на
                # 97° и убило Тренажёр++ (раздел 5.3i в MIGRATION).
                axis_sum += continuous.mean(dim=0)
                axis_count += 1

                for done, info in zip(dones, infos):
                    if not done:
                        continue
                    episode_return = float(info.get("episode_return", 0.0))
                    agent.record_episode(episode_return)
                    report.episodes += 1
                    report.best_return = max(report.best_return, episode_return)
                    if info.get("orbit"):
                        report.successes += 1
                    else:
                        reason = info.get("reason", "—")
                        report.failures[reason] = report.failures.get(reason, 0) + 1
                    knowledge.observe_episode(
                        stage_number, info, episode_return,
                        summary=info.get("summary"), title=label)
                    if on_episode is not None:
                        stop_reason = on_episode(episode_return, info, report) or ""

            with torch.no_grad():
                tail = torch.as_tensor(observations, dtype=torch.float32,
                                       device=agent.device)
                last_value = agent.net.value(tail)
            metrics = agent.update(buffer, last_value)
            report.iterations += 1
            report.updates.append(metrics)
            report.mean_return = agent.mean_return(20)
            if report.episodes >= 20:
                self.brain.save_best(report.mean_return)

            # Сохраняемся не чаще раза в полминуты. Проект лежит в
            # папке OneDrive: запись мегабайтного файла каждые двенадцать
            # секунд заставила бы синхронизацию гонять его всю ночь.
            due = (time.time() - last_save) > SAVE_INTERVAL
            if (report.iterations % 5 == 0 and due) or report.steps >= total_steps:
                last_save = time.time()
                # sigma в логе не для красоты: именно её невидимость стоила
                # ночного прогона. Разброс действий стоял на потолке, а по
                # одной энтропии это не читалось
                sigma = (math.exp(float(agent.net.log_std.max().item()))
                         if agent.net.log_std is not None else 0.0)
                speed = report.steps / max(1e-6, time.time() - started)
                axes = (axis_sum / max(1, axis_count)).tolist()
                report.axis_means = [round(value, 3) for value in axes]
                axis_sum = torch.zeros_like(axis_sum)
                axis_count = 0
                log.info("[%s] шаг %d/%d | эпизодов %d | средняя награда "
                         "%.2f | орбит %d | KL %.4f | энтропия %.2f | "
                         "разброс sigma %.3f | жёсткость %.0f%% | LR %.1e | "
                         "%.0f шаг/с",
                         label, report.steps, total_steps, report.episodes,
                         report.mean_return, report.successes,
                         metrics.get("approx_kl", 0.0),
                         metrics.get("entropy", 0.0),
                         sigma, report.difficulty * 100.0,
                         metrics.get("learning_rate", 0.0), speed)
                log.info("[%s] средние команды: газ %+.2f | тангаж %+.2f | "
                         "рыскание %+.2f | крен %+.2f", label, *report.axis_means)
                self.brain.save()
                self.write_metrics()
                knowledge.add_steps(stage_number, report.steps - saved_steps)
                saved_steps = report.steps
                knowledge.save()
                brain_map.write(self.brain, report=report, knowledge=knowledge)
            if progress is not None:
                progress(report)

        report.wall_time = time.time() - started
        report.stop_reason = stop_reason
        self.brain.save()
        self.write_metrics()
        knowledge.add_steps(stage_number, report.steps - saved_steps)
        knowledge.save()
        brain_map.write(self.brain, report=report, knowledge=knowledge)
        self.last_report = report
        log.info("Обучение на ступени «%s» завершено%s. %s", label,
                 f" ({stop_reason})" if stop_reason else "", report.describe())
        return report


    # ==================================================================
    # Конструктор: сборка ракеты
    # ==================================================================
    def train_builder(self, episodes: int | None = None, rollout: int | None = None,
                      goal_altitude: float = 80_000.0,
                      goal_transfer: bool = False, stage=None,
                      progress=None) -> TrainingReport:
        from .programs import BUILDER_STAGES, prepare

        agent = self.brain.builder
        if agent is None:
            raise RuntimeError("Сеть-конструктор не создана")
        if stage is None:
            number = int(self.brain.stages.get("builder", 1))
            stage = next((s for s in BUILDER_STAGES if s.number == number),
                         BUILDER_STAGES[0])
        for line in prepare("builder", stage, self.brain):
            log.info("[конструктор] %s", line)
        episodes = int(episodes or stage.default_steps)
        rollout = int(rollout or stage.rollout)
        # Ступень программы передаётся в среду сборки: со «Сборки+»
        # подключается проектное бюро и меняющиеся цели, со «Сборки++» —
        # проверка собираемости и расчёт ΔV по ступеням.
        env = BuilderEnv(goal_altitude=goal_altitude, goal_transfer=goal_transfer,
                         stage=stage.number)
        buffer = RolloutBuffer(agent.obs_dim, 0, 1, rollout, agent.device,
                               mask_dim=agent.discrete_sizes[0])
        report = TrainingReport(mode=f"конструктор: {stage.title}",
                                stage=stage.number)
        started = time.time()
        last_save = started
        obs = env.reset()
        episode_return = 0.0

        while report.episodes < episodes and self.running:
            buffer.clear()
            while not buffer.full and self.running:
                mask = env.action_mask()
                obs_t, continuous, discrete, logp, value = agent.act(
                    obs, action_mask=mask)
                action = int(discrete[0].item())
                next_obs, reward, done, info = env.step(action)
                buffer.add(obs_t, continuous, discrete, logp, value, reward, done,
                           mask=mask)
                episode_return += reward
                report.steps += 1
                obs = next_obs
                if done:
                    agent.record_episode(episode_return)
                    report.episodes += 1
                    report.best_return = max(report.best_return, episode_return)
                    evaluation = info.get("evaluation")
                    if evaluation is not None and evaluation.valid:
                        report.successes += 1
                    elif evaluation is not None and evaluation.problems:
                        # Чем именно сборка не годится — это и есть самое
                        # полезное в отчёте: «нет питания» и «не собирается
                        # в .craft» лечатся совершенно по-разному.
                        reason = str(evaluation.problems[0])[:48]
                        report.failures[reason] = report.failures.get(reason, 0) + 1
                    episode_return = 0.0
                    obs = env.reset()
                    if report.episodes >= episodes:
                        break

            with torch.no_grad():
                last_value = float(agent.net.value(agent.observe(obs)).item())
            metrics = agent.update(buffer, last_value)
            report.iterations += 1
            report.mean_return = agent.mean_return(20)
            if report.episodes >= 20:
                self.brain.save_best(report.mean_return)
            if report.iterations % 5 == 0:
                log.info("[конструктор] эпизодов %d/%d | средняя награда %.2f | "
                         "годных сборок %d", report.episodes, episodes,
                         report.mean_return, report.successes)
                if time.time() - last_save > SAVE_INTERVAL:
                    self.brain.save()
                    last_save = time.time()
            if progress is not None:
                progress(report)

        report.wall_time = time.time() - started
        self.brain.save()
        self.write_metrics()
        self.last_report = report
        log.info("Обучение конструктора завершено. %s", report.describe())
        return report

    # ==================================================================
    # Математик: счёт манёвров
    # ==================================================================
    def train_math(self, stage=None, episodes: int | None = None,
                   rollout: int | None = None, progress=None) -> TrainingReport:
        """Обучение счёту — прямой регрессией по точному эталону.

        Почему НЕ обучение с подкреплением, как у пилота и конструктора.
        Там награда — единственный источник знания о том, что хорошо: как
        правильно вести ракету, заранее не известно никому. Здесь известно
        точно. Правильный ответ есть у нас в руках ДО того, как сеть что-то
        сказала, и добывать его случайными пробами — значит выкидывать его.

        Замер, который это решил: 30 000 эпизодов PPO на первой ступени
        дали точность 2.2 %. Награда вида exp(-ошибка/допуск) при ошибке в
        сто раз численно равна нулю — сеть не получала вообще никакого
        направления, куда двигаться, и блуждала.

        Эталон приходит из `engineer/rocket_math` — тех самых функций,
        которыми считает полётный код. Поэтому «сдал экзамен» здесь
        означает буквально «повторяет игровые формулы с заданной
        точностью», а не «научился чему-то похожему».
        """
        from .math_env import MathEnv
        from .programs import MATH_STAGES, prepare

        agent = self.brain.mathematician
        if agent is None:
            raise RuntimeError("Сеть-математик не создана")
        if stage is None:
            number = int(self.brain.stages.get("math", 1))
            stage = next((s for s in MATH_STAGES if s.number == number),
                         MATH_STAGES[0])
        for line in prepare("math", stage, self.brain):
            log.info("[математик] %s", line)

        episodes = int(episodes or stage.default_steps)
        batch = int(rollout or stage.rollout)
        env = MathEnv(stage=stage.number, tolerance=stage.tolerance)
        report = TrainingReport(mode=f"математик: {stage.title}",
                                stage=stage.number)
        started = time.time()
        last_save = started
        net = agent.net
        # Отдельный оптимизатор: у агента свой, настроенный под PPO с очень
        # малым шагом. Регрессии нужен шаг побольше, а трогать общий нельзя —
        # он же обслуживает пилота через тот же класс.
        optimizer = torch.optim.Adam(net.parameters(), lr=1.5e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, episodes // batch), eta_min=5e-5)
        done_tasks = 0

        while done_tasks < episodes and self.running:
            observations, targets = [], []
            for _ in range(batch):
                observations.append(env.reset())
                targets.append(env.task.target())
            obs_t = torch.as_tensor(observations, dtype=torch.float32,
                                    device=agent.device)
            target_t = torch.as_tensor(targets, dtype=torch.float32,
                                       device=agent.device)
            net.normalizer.update(obs_t)
            mu = torch.tanh(net.mu_head(net.features(obs_t)))
            loss = torch.nn.functional.mse_loss(mu, target_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            net.clamp_log_std()

            done_tasks += batch
            report.iterations += 1
            report.episodes = done_tasks
            report.steps = done_tasks
            agent.stats.updates += 1
            agent.stats.total_steps += batch
            if report.iterations % 25 == 0:
                exam = self.evaluate_math(stage.number, tasks=300)
                report.mean_return = exam["accuracy"]
                log.info("[математик] %s: задач %d/%d | ошибка %.4f | "
                         "решено %.1f %%", stage.title, done_tasks, episodes,
                         float(loss.item()), exam["accuracy"] * 100.0)
                # Писать веса на каждый отчёт нельзя: итерация здесь
                # занимает доли секунды, и за ночь файл в два мегабайта
                # ушёл бы на диск сотни тысяч раз. Отчёт частый, запись
                # редкая — как в полётном цикле.
                if time.time() - last_save > SAVE_INTERVAL:
                    self.brain.save()
                    last_save = time.time()
            if progress is not None:
                progress(report)

        exam = self.evaluate_math(stage.number, tasks=800)
        report.mean_return = exam["accuracy"]
        report.difficulty = exam["accuracy"]
        report.successes = int(exam["accuracy"] * 800)
        report.wall_time = time.time() - started
        self.brain.save()
        self.write_metrics()
        self.last_report = report
        log.info("Обучение математика завершено: %s, точность %.1f %% "
                 "при допуске %.0f %%", stage.title,
                 exam["accuracy"] * 100.0, stage.tolerance * 100.0)
        for key, row in exam["tasks"].items():
            log.info("    %-22s решено %4d из %4d (%.0f %%), средняя ошибка "
                     "%.4f", key, row["solved"], row["total"],
                     100.0 * row["solved"] / max(1, row["total"]),
                     row["mean_error"])
            if row["solved"] < row["total"] * 0.8:
                report.failures[key] = row["total"] - row["solved"]
        return report

    def evaluate_math(self, stage_number: int | None = None,
                      tasks: int = 400) -> dict:
        """Экзамен без обучения: сколько задач сеть решает в допуске."""
        from .math_env import MathEnv
        from .programs import MATH_STAGES

        agent = self.brain.mathematician
        if agent is None:
            raise RuntimeError("Сеть-математик не создана")
        number = int(stage_number or self.brain.stages.get("math", 1))
        stage = next((s for s in MATH_STAGES if s.number == number),
                     MATH_STAGES[0])
        env = MathEnv(stage=number, tolerance=stage.tolerance, seed=12345)
        obs = env.reset()
        per_task: dict[str, list] = {}
        for _ in range(tasks):
            _obs_t, continuous, _d, _lp, _v = agent.act(
                obs, deterministic=True, update_normalizer=False)
            answer = [float(v) for v in continuous.squeeze(0).tolist()]
            obs, _reward, _done, info = env.step(answer)
            row = per_task.setdefault(info["task"], [0, 0, 0.0])
            row[0] += 1
            row[1] += int(info["solved"])
            row[2] += info["error"]
        return {
            "stage": number,
            "title": stage.title,
            "tolerance": stage.tolerance,
            "accuracy": env.accuracy,
            "tasks": {key: {"total": row[0], "solved": row[1],
                            "mean_error": row[2] / max(1, row[0])}
                      for key, row in sorted(per_task.items())},
        }

    # ==================================================================
    # Автономный проход обеих программ
    # ==================================================================
    def evolve_programs(self, programs=("math", "builder"),
                        attempts_per_stage: int = 3,
                        progress=None) -> list[dict]:
        """Проводит конструктора и математика по их лестницам без человека.

        На каждой ступени делается до `attempts_per_stage` заходов. Если
        экзамен сдан — переход на следующую. Если нет — ещё заход с тем же
        бюджетом: ступень не пропускается никогда, потому что следующая
        начинается с весов текущей, и пускать туда неготовую сеть значит
        учить её поверх мусора.

        Возвращает журнал: что делали, чем кончилось.
        """
        from .programs import stages_of

        journal: list[dict] = []
        for program in programs:
            for stage in stages_of(program):
                passed = False
                for attempt in range(1, attempts_per_stage + 1):
                    if not self.running:
                        return journal
                    log.info("=== %s / %s: заход %d из %d ===", program,
                             stage.title, attempt, attempts_per_stage)
                    if program == "math":
                        report = self.train_math(stage=stage, progress=progress)
                        score = report.difficulty
                        passed = score >= stage.pass_mean
                        detail = f"точность {score * 100:.1f} %"
                    else:
                        report = self.train_builder(stage=stage,
                                                    progress=progress)
                        score = report.mean_return
                        passed = (score >= stage.pass_mean
                                  and report.successes >= stage.pass_count)
                        detail = (f"средняя {score:+.2f}, годных "
                                  f"{report.successes}")
                    journal.append({
                        "program": program, "stage": stage.number,
                        "title": stage.title, "attempt": attempt,
                        "passed": passed, "score": round(float(score), 4),
                        "detail": detail,
                        "wall_time": round(report.wall_time, 1)})
                    log.info("%s / %s: %s (%s)", program, stage.title,
                             "СДАНО" if passed else "не сдано", detail)
                    if passed:
                        break
                if not passed:
                    log.warning("%s: ступень «%s» не сдана за %d заходов — "
                                "дальше не иду, следующая ступень учится "
                                "поверх этой", program, stage.title,
                                attempts_per_stage)
                    break
        return journal

    # ==================================================================
    # Живая KSP
    # ==================================================================
    def train_in_ksp(self, dispatcher, episodes: int = 20,
                     goal_altitude: float = 80_000.0, rollout: int = 1024,
                     use_neural_builder: bool = False,
                     flight_log=None, progress=None) -> TrainingReport:
        """Дообучение в игре: каждый эпизод — реальный запуск."""
        from .curriculum import STAGES, stage_by_number
        from .flight_env import KSPFlightEnv

        agent = self.brain.pilot
        # Ритм решений берётся из ступени, на которой мозг учился, а не
        # задаётся отдельно. Иначе получается тот самый разрыв, который
        # весь день ломал третью ступень: сеть училась решать раз в 0.3 с,
        # а живая игра спрашивала её раз в секунду — и та же команда
        # поворачивала аппарат втрое слабее.
        number = min(max(1, int(getattr(self.brain, "stage", 1))), len(STAGES))
        stage = stage_by_number(number)
        log.info("[KSP] ритм решений %.2f с — как на ступени %d (%s)",
                 stage.decision_seconds, stage.number, stage.title)
        env = KSPFlightEnv(dispatcher, goal_altitude=goal_altitude,
                           control_seconds=stage.decision_seconds,
                           max_episode_seconds=stage.max_time,
                           flight_log=flight_log)
        mask_dim = agent.discrete_sizes[0] if agent.discrete_sizes else 0
        buffer = RolloutBuffer(agent.obs_dim, agent.continuous_dim,
                               len(agent.discrete_sizes), rollout, agent.device,
                               mask_dim=mask_dim)
        report = TrainingReport(mode="KSP")
        started = time.time()

        for episode in range(episodes):
            if not self.running:
                break
            blueprint = self._blueprint_for(dispatcher, goal_altitude,
                                            use_neural_builder)
            obs = env.reset(blueprint)
            if obs is None:
                log.error("Эпизод %d не начался", episode + 1)
                break

            episode_return = 0.0
            buffer.clear()
            done = False
            while not done and self.running:
                mask = env.action_mask()
                obs_t, continuous, discrete, logp, value = agent.act(
                    obs, action_mask=mask)
                action = FlightAction.decode(continuous.tolist(), discrete.tolist())
                next_obs, reward, done, info = env.step(action)
                buffer.add(obs_t, continuous, discrete, logp, value, reward, done,
                           mask=mask)
                episode_return += reward
                report.steps += 1
                if next_obs is not None:
                    obs = next_obs
                if buffer.full:
                    break

            summary = env.episode_summary()
            agent.record_episode(episode_return)
            report.episodes += 1
            report.best_return = max(report.best_return, episode_return)
            if summary.get("orbit"):
                report.successes += 1

            with torch.no_grad():
                last_value = 0.0 if done else float(
                    agent.net.value(agent.observe(obs)).item())
            metrics = agent.update(buffer, last_value)
            report.iterations += 1
            report.mean_return = agent.mean_return(10)
            self.brain.save_best(report.mean_return)
            log.info("[KSP] эпизод %d/%d: награда %.2f | %s | шагов %d | "
                     "потолок %d м", episode + 1, episodes, episode_return,
                     summary.get("reason", "—"), summary.get("steps", 0),
                     summary.get("peak_altitude", 0))
            env.close()
            self.brain.save()
            self.write_metrics()
            if progress is not None:
                progress(report)

        report.wall_time = time.time() - started
        self.last_report = report
        log.info("Обучение в KSP завершено. %s", report.describe())
        return report

    # ------------------------------------------------------------------
    def _blueprint_for(self, dispatcher, goal_altitude: float,
                       use_neural_builder: bool):
        """Чертёж для эпизода: от сети-конструктора или от инженера."""
        if use_neural_builder and self.brain.builder is not None:
            # Ступень берётся КОНСТРУКТОРСКАЯ, а не пилотская. Раньше поле
            # было одно на всех, и это сходилось само собой; с появлением
            # отдельных программ пилот на четвёртой ступени заставлял бы
            # конструктора работать по правилам, которых тот не проходил.
            stage = int(self.brain.stages.get("builder", 1))
            env = BuilderEnv(goal_altitude=goal_altitude, stage=stage)
            obs = env.reset()
            done = False
            while not done:
                mask = env.action_mask()
                _, _, discrete, _, _ = self.brain.builder.act(
                    obs, deterministic=True, action_mask=mask)
                obs, _, done, info = env.step(int(discrete[0].item()))
            blueprint = env.to_blueprint()
            if blueprint is not None:
                log.info("Чертёж собран нейросетью: %s",
                         ", ".join(p.name for p in env.chosen))
                return blueprint
            log.warning("Сеть собрала несобираемую конструкцию — "
                        "беру расчётный чертёж инженера")
        design = dispatcher.engineer.design(dispatcher.spec)
        return design.blueprint

    # ------------------------------------------------------------------
    def write_metrics(self, path: Path | None = None) -> Path:
        target = Path(path or METRICS_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pilot": self.brain.pilot.stats.to_dict(),
            "builder": (self.brain.builder.stats.to_dict()
                        if self.brain.builder else None),
            "config": self.config.to_dict(),
            "device": str(self.brain.pilot.device),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return target

    # ------------------------------------------------------------------
    def evaluate_crew(self, stage, episodes: int = 120) -> dict:
        """Экзамен ступени «Аварии»: сколько экипажей вернулось живыми.

        Считаются только те эпизоды, где отказ ДЕЙСТВИТЕЛЬНО сработал и
        на борту были люди. Остальные к делу не относятся: на исправной
        ракете спасать некого.

        Почему это отдельный замер. Обычный экзамен ступени считает
        орбиты, а орбиты на этой ступени дают те две трети полётов, где
        ничего не сломалось. Первый прогон так и закончился: ступень
        «сдана» за 47 секунд при средней награде −5.22, то есть экипажи
        гибли, а орбит хватало. Орбитами аварийную подготовку не измерить.
        """
        from .spaces import FlightAction

        agent = self.brain.pilot
        faced = saved = 0
        flyable = flown = 0
        for n in range(episodes):
            env = stage.make_env(goal_altitude=80_000.0, seed=500_000 + n,
                                 exploring_starts=0.0)
            env.set_difficulty(1.0)
            obs = env.reset()
            done = False
            info: dict = {}
            while not done and self.running:
                mask = env.action_mask()
                _o, continuous, discrete, _lp, _v = agent.act(
                    obs, deterministic=True, action_mask=mask,
                    update_normalizer=False)
                obs, _r, done, info = env.step(FlightAction(
                    float(continuous[0]), float(continuous[1]),
                    float(continuous[2]), float(continuous[3]),
                    bool(int(discrete[0].item()))))
            if not self.running:
                break
            summary = env.episode_summary()
            if not summary.get("failures") or not summary.get("crew"):
                continue
            # Замер оборвали на полпути — чем кончился полёт, неизвестно.
            # Такой эпизод не свидетельство ни за, ни против.
            if info.get("reason") == "лимит времени":
                continue
            faced += 1
            saved += int(bool(summary.get("crew_saved")))
            # Отказ, после которого двигатель ещё тянет, — это не повод
            # бросать ракету: полёт можно довести. Считаем такие отдельно,
            # иначе экзамен сдаётся уводом кабины на каждой поломке.
            if summary.get("engine_scale", 0.0) >= 0.5:
                flyable += 1
                flown += int(bool(summary.get("orbit")))
        return {"faced": faced, "saved": saved,
                "rate": (saved / faced) if faced else 0.0,
                "flyable": flyable, "flown": flown,
                "flown_rate": (flown / flyable) if flyable else 1.0}

    def evaluate_pilot(self, episodes: int = 10,
                       goal_altitude: float = 80_000.0, stage=None) -> dict:
        """Прогон без обучения: проверка, чему научилась сеть.

        stage — ступень программы, на которой проверять. По умолчанию та,
        на которой мозг обучался последней: проверять политику Тренажёра++
        на базовой физике бессмысленно, она отвечает на другие вопросы.

        Проверка идёт СО СТАРТОВОГО СТОЛА и на ПОЛНОЙ жёсткости. Оба
        условия обязательны, иначе экзамен меряет не то:

        * `exploring_starts` в обучении бросает половину эпизодов уже в
          воздухе, на 500-45000 м и с готовой скоростью. Для разведки это
          приём, для экзамена — подмена задачи: «долететь до орбиты
          с 40 км и 2 км/с» умеет и та политика, которая не умеет
          взлетать. Здесь стартовых подарков нет;
        * жёсткость 1.0 ставится явно. Значение живёт в атрибуте
          экземпляра, а не в классе, и «по умолчанию единица» держится
          только до первой правки — полагаться на это в приёмной комиссии
          нельзя.
        """
        from .curriculum import STAGES, stage_by_number

        agent = self.brain.pilot
        if stage is None:
            number = min(max(1, getattr(self.brain, "stage", 1)), len(STAGES))
            stage = stage_by_number(number)
        env = stage.make_env(goal_altitude=goal_altitude, exploring_starts=0.0)
        env.set_difficulty(1.0)
        returns, orbits, peaks, crashes = [], 0, [], 0
        reasons: dict = {}
        for _ in range(episodes):
            obs = env.reset()
            total, done, info = 0.0, False, {}
            while not done:
                mask = env.action_mask()
                # normalizer не трогаем: проверка не имеет права менять
                # статистику входов, по которой живёт обучение
                _, continuous, discrete, _, _ = agent.act(
                    obs, deterministic=True, action_mask=mask,
                    update_normalizer=False)
                action = FlightAction.decode(continuous.tolist(), discrete.tolist())
                for repeat in range(stage.action_repeat):
                    step_action = action if repeat == 0 else FlightAction(
                        action.throttle, action.pitch, action.yaw, action.roll,
                        False)
                    obs, reward, done, info = env.step(step_action)
                    total += reward
                    if done:
                        break
            returns.append(total)
            peaks.append(env.peak_altitude)
            orbits += 1 if env.reached_orbit else 0
            crashes += 1 if env.crashed else 0
            reason = info.get("reason", "—")
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "stage": stage.number,
            "stage_title": stage.title,
            "episodes": episodes,
            "mean_return": sum(returns) / len(returns),
            "best_return": max(returns),
            "orbits": orbits,
            "crashes": crashes,
            "mean_peak_altitude": sum(peaks) / len(peaks),
            "reasons": reasons,
        }
