"""ТРЕНАЖЁР ПРАКТИКИ: сквозные миссии ко всем планетам, разные ракеты.

Навигатор сдал экзамены по планированию, автопилот — по исполнению.
Здесь они летают ВМЕСТЕ, как в игре, и миссии идут вперемешку:

    случайная планета и цель (посадка, низкая или высокая орбита);
    случайная ракета: тяговооружённость (от вялой 0.15 до бодрой 1.2),
      запас Δv 0.7…1.8 от нужного, есть ли теплозащита, сколько
      парашютов, какая антенна, тяга при посадке в местной гравитации;
    случайные настройки связи в мире.

Миссия проходит три этапа:
    1. навигатор решает: лететь или отказаться, окно, захват, связь;
    2. тренажёр исполнения проводит перелёт с выученными для этой
       планеты параметрами и НАСТОЯЩЕЙ для этой ракеты длиной ожогов
       (вялая ракета теряет больше на уходе и захвате);
    3. посадка: хватит ли тяги в местной гравитации; в атмосфере —
       хватит ли парашютов (на Дюне воздух разрежен, нужно больше
       куполов или доводка двигателем).

Лучшие по итогам практики решения дообучают навигатора (перебор
вариантов + регрессия; PPO здесь только портил: 97.6 % → 92 % за
20 итераций, замер 21.09). Веса
сохраняются только если результат практики вырос.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict

import numpy as np
import torch

from ..config import DATA_DIR, ROOT
from .nav_env import NavigatorEnv, OBJECTIVES
from .solar import PLANETS

REPORT = DATA_DIR / "practice_report.json"
LOG = DATA_DIR / "practice_log.txt"


class PracticeEnv(NavigatorEnv):
    """Навигатор + исполнение + посадка, ракеты случайные."""

    def __init__(self, batch: int = 512, seed: int | None = None):
        super().__init__(batch=batch, stage=3, seed=seed)
        from .cruise_sim import PROFILE_FILE, planets
        self.cruise_planets = planets()
        data = json.loads(PROFILE_FILE.read_text(encoding="utf-8")) if PROFILE_FILE.exists() else {}
        self.cruise_profile = data.get("planets", {})

    def reset(self) -> np.ndarray:
        super().reset()
        n, rng = self.batch, self.rng
        c = self.ctx
        c["twr"] = rng.uniform(0.15, 1.2, n)                       # вакуумная, для манёвров
        c["land_twr"] = rng.uniform(0.6, 4.0, n)                   # в гравитации цели
        c["n_chutes"] = np.where(c["chutes"], rng.integers(1, 7, n), 0)
        # разные ракеты: запас Δv шире, чем в экзамене
        need = self.expert["total"]
        c["dv"] = need * rng.uniform(0.7, 1.8, n)
        self.expert["abort"] = (c["dv"] < need) | self.expert["comm_impossible"]
        return self.observe()

    def _params(self, planet: str, aero: bool) -> dict:
        from .cruise_sim import NAMES, PARAMS
        block = self.cruise_profile.get(planet, {})
        prof = block.get("aero" if aero else "propulsive") or block.get("propulsive") or {}
        out = {k: np.float64(prof.get(k, sum(PARAMS[k]) / 2)) for k in NAMES}
        out["threshold_frac"] = 10 ** (-3 + 3 * out["threshold"])
        return out

    def step(self, continuous, discrete):
        from .cruise_sim import simulate
        rewards, info = super().step(continuous, discrete)
        c, n = self.ctx, self.batch
        a = self.decode(continuous, discrete)
        names = [PLANETS[t] for t in c["target"]]
        flew = info["success"].copy()                 # план выполним на бумаге
        exec_ok = np.ones(n, bool)
        extra = np.zeros(n)
        for name in set(names):
            for aero in (False, True):
                m = np.array([t == name for t in names]) & (a["aero"] == aero) & flew
                if not m.any():
                    continue
                p = self.cruise_planets[name]
                params = self._params(name, aero)
                rng = np.random.default_rng(self.rng.integers(1 << 31))
                # Правило то же, что у навигатора: без теплозащиты аэрозахват
                # допустим только при медленном подлёте (v∞ < 2.5 км/с)
                shield = c["heatshield"][m] | (self.expert["vinf"][m] < 2_500.0)
                s, e = simulate(p, params, int(m.sum()), rng, aero, heatshield=shield)
                # поправка на тягу конкретной ракеты: вялая теряет больше
                twr = c["twr"][m]
                e = e * (1.0 + np.maximum(0.0, 0.4 - twr) * 3.0)
                exec_ok[m] = s
                extra[m] = e
        fuel_ok = info["total"] + extra <= c["dv"]
        # посадка
        land = c["objective"] == 0
        atm = np.array([self.S[t].atmosphere > 0 for t in names])
        thin = np.array([0 < self.S[t].atmosphere < 60_000 for t in names])
        chutes = c["n_chutes"]
        vac_ok = c["land_twr"] >= 1.3
        atm_ok = (chutes >= np.where(thin, 3, 1)) | (c["land_twr"] >= 1.8)
        land_ok = ~land | np.where(atm, atm_ok, vac_ok)
        success = flew & exec_ok & fuel_ok & land_ok
        rocket_limited = flew & exec_ok & fuel_ok & ~land_ok      # ракета не умеет сесть — не вина навигатора
        reward = rewards.copy()
        reward = np.where(flew & ~(exec_ok & fuel_ok), -5.0, reward)
        reward = np.where(rocket_limited, -1.0, reward)
        info = dict(info)
        info.update(success=success, exec_ok=exec_ok, fuel_ok=fuel_ok, land_ok=land_ok,
                    rocket_limited=rocket_limited,
                    ok=np.where(info["right_abort"], info["abort"], success | rocket_limited))
        return reward.astype(np.float32), info


# ==========================================================================
def improve(env: PracticeEnv, cont: np.ndarray, disc: np.ndarray, repeats: int = 3):
    """Лучшее решение для каждой миссии перебором вариантов в практике.

    Варианты вокруг решения учителя: наоборот лететь/отказаться,
    аэрозахват вместо двигателя (и обратно), захват выше/ниже, на один
    ретранслятор больше. Каждый проигрывается `repeats` раз — исполнение
    случайно, — и берётся тот, что в среднем даёт лучший исход. Учитель
    получает небольшую фору: сменить решение можно только если другой
    вариант заметно лучше."""
    cands = [(cont, disc)]

    def variant(fn):
        c, d = cont.copy(), disc.copy()
        fn(c, d)
        cands.append((c, d))

    variant(lambda c, d: d.__setitem__((slice(None), 0), 1 - disc[:, 0]))
    variant(lambda c, d: d.__setitem__((slice(None), 3), 1 - disc[:, 3]))
    variant(lambda c, d: c.__setitem__((slice(None), 2), np.clip(cont[:, 2] + 0.15, -1, 1)))
    variant(lambda c, d: c.__setitem__((slice(None), 2), np.clip(cont[:, 2] - 0.15, -1, 1)))
    variant(lambda c, d: d.__setitem__((slice(None), 1), np.minimum(4, disc[:, 1] + 1)))
    scores = np.zeros((len(cands), len(cont)))
    for i, (c, d) in enumerate(cands):
        for _ in range(repeats):
            r, _ = env.step(c, d)
            scores[i] += r / repeats
    scores[0] += 0.3                                   # фора учителю
    best = scores.argmax(0)
    out_c = np.stack([cands[b][0][j] for j, b in enumerate(best)])
    out_d = np.stack([cands[b][1][j] for j, b in enumerate(best)])
    return out_c.astype(np.float32), out_d


def _worker(conn, batch: int, seed: int) -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    env = PracticeEnv(batch=batch, seed=seed)
    keep = ("ok", "success", "target", "objective", "abort", "right_abort", "exec_ok", "fuel_ok",
            "land_ok", "rocket_limited", "comm_lost")
    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            obs = env.reset()
            cont, disc = env.encode_expert()
            conn.send((obs, cont, disc))
        elif cmd == "step":
            r, info = env.step(*payload)
            conn.send((r, {k: info[k] for k in keep}))
        elif cmd == "improve":
            obs = env.reset()
            cont, disc = improve(env, *env.encode_expert(), repeats=payload or 3)
            conn.send((obs, cont, disc))
        else:
            return


class Pool:
    def __init__(self, workers: int, batch: int):
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for i in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, batch, 5000 + i), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)

    def reset(self):
        for p in self.pipes:
            p.send(("reset", None))
        parts = [p.recv() for p in self.pipes]
        return tuple(np.concatenate([x[i] for x in parts]) for i in range(3))

    def step(self, cont, disc):
        chunks = np.array_split(np.arange(len(cont)), len(self.pipes))
        for p, idx in zip(self.pipes, chunks):
            p.send(("step", (cont[idx], disc[idx])))
        parts = [p.recv() for p in self.pipes]
        r = np.concatenate([x[0] for x in parts])
        info = {k: np.concatenate([x[1][k] for x in parts]) for k in parts[0][1]}
        return r, info

    def improve(self, repeats: int = 3):
        for p in self.pipes:
            p.send(("improve", repeats))
        parts = [p.recv() for p in self.pipes]
        return tuple(np.concatenate([x[i] for x in parts]) for i in range(3))

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
            except Exception:
                pass


# ==========================================================================
def evaluate(agent, pool: Pool, rounds: int = 4, expert: bool = False) -> dict:
    """Практика без шума: сколько сквозных миссий удалось, по парам."""
    stats = defaultdict(list)
    totals = defaultdict(int)
    for _ in range(rounds):
        obs, ec, ed = pool.reset()
        if expert:
            cont, disc = ec, ed
        else:
            with torch.no_grad():
                c, d, _, _ = agent.net.act(torch.as_tensor(obs), deterministic=True)
            cont, disc = c.numpy(), d.numpy()
        _, info = pool.step(cont, disc)
        for t, o, ok in zip(info["target"], info["objective"], info["ok"]):
            stats[f"{PLANETS[t]}/{OBJECTIVES[o]}"].append(bool(ok))
        for k in ("ok", "success", "abort", "exec_ok", "fuel_ok", "land_ok", "rocket_limited"):
            totals[k] += int(np.sum(info[k]))
        totals["n"] += len(info["ok"])
    pairs = {k: round(float(np.mean(v)), 3) for k, v in sorted(stats.items())}
    n = totals["n"]
    return {"overall": totals["ok"] / n, "flown_success": totals["success"] / n,
            "aborted": totals["abort"] / n, "rocket_limited": totals["rocket_limited"] / n,
            "worst_pair": min(pairs.values()), "pairs": pairs, "missions": n}


def train(rounds: int = 24, workers: int = 10, batch: int = 512, echo=print) -> dict:
    """Практика: улучшенные перебором решения → обучение навигатора на них.

    PPO поверх почти идеального навигатора только шумел (97.6 % → 92 %
    за 20 итераций, замер 21.09). Здесь сеть учится на решениях, которые
    практика показала лучшими, — это обычная регрессия, она не портит."""
    from .nav_plan import BEST_FILE, MODEL_FILE, build_agent, load, save

    def say(text):
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        echo(line)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    agent = build_agent()
    load(agent)
    backup = ROOT / "kia_model_navigator_before_practice.pth"
    if not backup.exists() and MODEL_FILE.exists():
        backup.write_bytes(MODEL_FILE.read_bytes())
    pool = Pool(workers, batch)
    try:
        base = evaluate(agent, pool)
        ceiling = evaluate(agent, pool, expert=True)
        say(f"Практика до: верно {base['overall'] * 100:.1f} %, худшая пара {base['worst_pair'] * 100:.1f} % "
            f"(точный планировщик: {ceiling['overall'] * 100:.1f} %)")
        O, C, D = [], [], []
        for r in range(1, rounds + 1):
            obs, cont, disc = pool.improve()
            O.append(obs)
            C.append(cont)
            D.append(disc)
            say(f"раунд {r}/{rounds}: разыграно {len(obs) * 18} вариантов миссий")
        obs = torch.as_tensor(np.concatenate(O))
        cont = torch.as_tensor(np.concatenate(C))
        disc = torch.as_tensor(np.concatenate(D), dtype=torch.long)
        net, opt = agent.net, agent.optimizer
        for group in opt.param_groups:
            group["lr"] = 3e-4
        n = len(obs)
        for epoch in range(20):
            perm = torch.randperm(n)
            for i in range(0, n, 1024):
                idx = perm[i:i + 1024]
                h = net.features(obs[idx])
                loss = ((torch.tanh(net.mu_head(h)) - cont[idx]) ** 2).mean() * 4.0
                for k, head in enumerate(net.discrete_heads):
                    loss = loss + torch.nn.functional.cross_entropy(head(h), disc[idx, k])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.clamp_log_std()
        final = evaluate(agent, pool, rounds=6)
        kept = final["overall"] > base["overall"]
        if kept:
            save(agent, 3)
            save(agent, 3, BEST_FILE)
        else:
            load(agent)
        say(f"Практика после: верно {final['overall'] * 100:.1f} %, худшая пара {final['worst_pair'] * 100:.1f} % — "
            + ("веса обновлены" if kept else "хуже прежних, оставлены прежние веса"))
        report = {"trained_at": time.strftime("%Y-%m-%d %H:%M"), "before": base, "after": final,
                  "expert_ceiling": ceiling, "kept": kept, "missions_practised": n}
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        pool.close()
