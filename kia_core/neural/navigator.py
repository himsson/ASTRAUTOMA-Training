"""НАВИГАТОР: межпланетная программа — все планеты KSP, все цели, связь.

Четвёртая сеть KIA (после пилота, конструктора и математика). Учится
планировать миссию к любой планете: окно вылета, время перелёта,
захват или аэрозахват, коррекция курса, отказ при нехватке Δv,
доставка ретрансляторов, когда без них не будет связи.

КАК УЧИТСЯ

1. Со знаниями. На каждой ступени сначала предобучение по точному
   планировщику (перебор окон × Ламберт — то, что математик уже умеет
   считать): сеть сразу получает правильные ответы, а не ищет их
   наугад. Это правило проекта: где есть точный ответ, учить по нему.
2. Потом PPO в тренажёре: разброс двигателя, коридор аэрозахвата,
   случайная потеря связи — то, где точной формулы нет.
3. Все планеты и цели идут ОДНОВРЕМЕННО: десять процессов-тренажёров
   на процессоре, в каждой пачке вперемешку Мохо, Ив, Дюна, Дрес, Джул,
   Илу, посадка, низкая и высокая орбита.
4. Экзамен — по КАЖДОЙ паре «планета + цель» отдельно: ≥ 95 % верных
   решений. Сдал — ступень переключается сама, веса ступени
   сохраняются снимком. После третьей ступени программа закончена.

ВЕСА
    kia_model_navigator.pth        рабочие
    kia_model_navigator_best.pth   лучшие по экзамену
    kia_model_navigator_stageN.pth снимок после сдачи ступени N
Файл содержит только тензоры и числа — грузится с weights_only=True,
то есть безопасно, даже если веса скачаны у другого человека.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from ..config import DATA_DIR, ROOT
from ..logging_setup import get_logger

log = get_logger("neural.navigator")

from .nav_plan import BEST_FILE, MODEL_FILE   # noqa: E402
LOG_FILE = DATA_DIR / "navigator_log.txt"
REPORT_FILE = DATA_DIR / "navigator_report.json"

STAGE_TITLES = {1: "Навигатор", 2: "Навигатор+", 3: "Навигатор++"}
PASS_RATE = 0.95
EXAM_WINDOW = 400          # последних миссий на каждую пару «планета + цель»


def stage_file(n: int) -> Path:
    return ROOT / f"kia_model_navigator_stage{n}.pth"


# ==========================================================================
# Процессы-тренажёры
# ==========================================================================
def _worker(conn, batch: int, seed: int) -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .nav_env import NavigatorEnv
    env = NavigatorEnv(batch=batch, stage=1, seed=seed)
    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            env.set_stage(payload)
            obs = env.reset()
            cont, disc = env.encode_expert()
            conn.send((obs, cont, disc))
        elif cmd == "step":
            cont, disc = payload
            rewards, info = env.step(cont, disc)
            conn.send((rewards, {k: v for k, v in info.items()
                                 if k in ("ok", "success", "target", "objective",
                                          "abort", "burned", "comm_lost",
                                          "relays", "relays_needed", "total",
                                          "expert_total")}))
        elif cmd == "close":
            conn.close()
            return


class WorkerPool:
    def __init__(self, workers: int, batch: int):
        ctx = mp.get_context("spawn")
        self.pipes = []
        self.procs = []
        for i in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, batch, 1000 + i), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)

    def reset(self, stage: int):
        for p in self.pipes:
            p.send(("reset", stage))
        parts = [p.recv() for p in self.pipes]
        return (np.concatenate([x[0] for x in parts]),
                np.concatenate([x[1] for x in parts]),
                np.concatenate([x[2] for x in parts]))

    def step(self, cont: np.ndarray, disc: np.ndarray):
        chunks = np.array_split(np.arange(len(cont)), len(self.pipes))
        for p, idx in zip(self.pipes, chunks):
            p.send(("step", (cont[idx], disc[idx])))
        parts = [p.recv() for p in self.pipes]
        rewards = np.concatenate([x[0] for x in parts])
        info = {k: np.concatenate([x[1][k] for x in parts]) for k in parts[0][1]}
        return rewards, info

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)


from .nav_plan import build_agent, load, plan_mission, save   # noqa: E402,F401


# ==========================================================================
class NavigatorTrainer:
    def __init__(self, workers: int = 10, batch: int = 512, echo=print):
        self.workers = workers
        self.batch = batch
        self.echo = echo
        self.agent = build_agent()
        loaded = load(self.agent)
        self.stage = loaded or 1
        self.best_rate = 0.0
        self.running = True
        self.history: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=EXAM_WINDOW))

    def say(self, text: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        self.echo(line)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ------------------------------------------------------------------
    def pretrain(self, pool: WorkerPool, rounds: int = 12, epochs: int = 30) -> float:
        """Предобучение по точному планировщику («знания»).

        Замер: 50 000 миссий × 30 эпох — сеть повторяет учителя на 100 %
        миссий первой ступени. Скорость обучения здесь выше, чем у PPO:
        это обычная регрессия, ей не нужна осторожность."""
        net, opt = self.agent.net, self.agent.optimizer
        obs_all, cont_all, disc_all = [], [], []
        for _ in range(rounds):
            obs, cont, disc = pool.reset(self.stage)
            obs_all.append(obs)
            cont_all.append(cont)
            disc_all.append(disc)
        obs = torch.as_tensor(np.concatenate(obs_all))
        cont = torch.as_tensor(np.concatenate(cont_all))
        disc = torch.as_tensor(np.concatenate(disc_all), dtype=torch.long)
        net.normalizer.update(obs)
        for group in opt.param_groups:
            group["lr"] = 1e-3
        n = len(obs)
        loss_value = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 1024):
                idx = perm[i:i + 1024]
                h = net.features(obs[idx])
                mu = torch.tanh(net.mu_head(h))
                loss = ((mu - cont[idx]) ** 2).mean() * 4.0
                for k, head in enumerate(net.discrete_heads):
                    loss = loss + torch.nn.functional.cross_entropy(head(h), disc[idx, k])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                loss_value = float(loss.detach())
        for group in opt.param_groups:
            group["lr"] = self.agent.config.learning_rate
        with torch.no_grad():
            # После предобучения разброс действий — самый узкий: ищем
            # улучшения рядом со знаниями, а не во всём пространстве
            net.log_std.fill_(net.LOG_STD_MIN)
        net.clamp_log_std()
        return loss_value

    def evaluate(self, pool: WorkerPool) -> float:
        """Экзаменационный раунд: сеть отвечает без шума, как в игре."""
        obs, _, _ = pool.reset(self.stage)
        with torch.no_grad():
            cont, disc, _, _ = self.agent.net.act(torch.as_tensor(obs), deterministic=True)
        _, info = pool.step(cont.numpy(), disc.numpy())
        for t, o, ok in zip(info["target"], info["objective"], info["ok"]):
            self.history[(int(t), int(o))].append(bool(ok))
        return float(info["ok"].mean())

    # ------------------------------------------------------------------
    def exam(self) -> tuple[float, dict]:
        from .nav_env import OBJECTIVES, STAGE_RULES
        from .solar import GAS_GIANTS, PLANETS
        rules = STAGE_RULES[min(self.stage, 3)]
        table = {}
        worst = 1.0
        for ti, planet in enumerate(PLANETS):
            for obj in rules.objectives:
                oi = OBJECTIVES.index(obj)
                if planet in GAS_GIANTS and obj == "land":
                    continue
                h = self.history[(ti, oi)]
                rate = (sum(h) / len(h)) if len(h) >= EXAM_WINDOW // 2 else 0.0
                table[f"{planet}/{obj}"] = round(rate, 3)
                worst = min(worst, rate)
        return worst, table

    def train(self, max_iterations: int = 4000) -> dict:
        from .ppo import VectorRolloutBuffer
        from .nav_env import CONTINUOUS_DIM, DISCRETE_SIZES, OBS_DIM
        pool = WorkerPool(self.workers, self.batch)
        total = self.workers * self.batch
        self.say(f"Навигатор: {self.workers} тренажёров × {self.batch} миссий, "
                 f"ступень {self.stage} «{STAGE_TITLES[self.stage]}»")
        try:
            loss = self.pretrain(pool)
            self.say(f"Предобучение по знаниям (Ламберт, коридоры, связь): ошибка {loss:.3f}")
            self.exam_now(pool)
            it = 0
            recent = deque(maxlen=20)
            while self.running and it < max_iterations and self.stage <= 3:
                it += 1
                # Два раунда миссий на обновление: PPO проекта не учится
                # на пачке из одного шага
                buf = VectorRolloutBuffer(OBS_DIM, CONTINUOUS_DIM, len(DISCRETE_SIZES),
                                          capacity=2, device=self.agent.device, envs=total)
                for _ in range(2):
                    obs, _, _ = pool.reset(self.stage)
                    obs_t, cont, disc, logp, value, _ = self.agent.act_many(obs)
                    rewards, info = pool.step(cont.cpu().numpy(), disc.cpu().numpy())
                    buf.add_batch(obs_t, cont, disc, logp, value, rewards, np.ones(total))
                metrics = self.agent.update(buf, last_value=0.0)
                recent.append(float(rewards.mean()))
                self.agent.record_episode(float(rewards.mean()))
                if it % 5 == 0:
                    exam_rate = self.evaluate(pool)
                if it % 10 == 0:
                    worst, table = self.exam()
                    self.say(f"ст.{self.stage} итер. {it}: награда {np.mean(recent):+.2f}, "
                             f"экзамен {exam_rate * 100:.1f} %, худшая пара "
                             f"{worst * 100:.1f} %, KL {metrics.get('approx_kl', 0):.4f}")
                    save(self.agent, self.stage)
                    if worst > self.best_rate:
                        self.best_rate = worst
                        save(self.agent, self.stage, BEST_FILE)
                    if worst >= PASS_RATE:
                        self.promote(table)
                        if self.stage > 3:
                            break
                        loss = self.pretrain(pool)
                        self.say(f"Предобучение ступени {self.stage}: ошибка {loss:.3f}")
                        self.exam_now(pool)
            return self.report()
        finally:
            pool.close()

    def exam_now(self, pool: WorkerPool) -> None:
        """Экзамен сразу после предобучения: если знаний уже хватает,
        ступень сдаётся без PPO (замер: PPO поверх предобучения только
        портил решения — 96 % → 84 %). Может сдать несколько ступеней подряд."""
        while self.stage <= 3:
            for _ in range(3):
                self.evaluate(pool)
            worst, table = self.exam()
            self.say(f"Экзамен после предобучения, ступень {self.stage}: худшая пара "
                     f"{worst * 100:.1f} %")
            if worst < PASS_RATE:
                save(self.agent, self.stage, BEST_FILE)
                self.best_rate = worst
                return
            self.promote(table)
            if self.stage > 3:
                return
            loss = self.pretrain(pool)
            self.say(f"Предобучение ступени {self.stage}: ошибка {loss:.3f}")

    def promote(self, table: dict) -> None:
        save(self.agent, self.stage, stage_file(self.stage))
        self.say(f"ЭКЗАМЕН СДАН: ступень {self.stage} «{STAGE_TITLES[self.stage]}» — "
                 + ", ".join(f"{k} {v * 100:.0f}%" for k, v in table.items()))
        self.stage += 1
        self.best_rate = 0.0
        self.history.clear()
        if self.stage > 3:
            save(self.agent, 3)
            save(self.agent, 3, BEST_FILE)
            self.say("Межпланетная программа пройдена целиком")
        else:
            save(self.agent, self.stage)
            self.say(f"Переход на ступень {self.stage} «{STAGE_TITLES[self.stage]}»")

    def report(self) -> dict:
        worst, table = self.exam()
        data = {"stage": min(self.stage, 3), "finished": self.stage > 3,
                "worst_pair": worst, "pairs": table,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        REPORT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data


