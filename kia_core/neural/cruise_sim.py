"""ТРЕНАЖЁР ИСПОЛНЕНИЯ МЕЖПЛАНЕТНОГО ПЕРЕЛЁТА (автопилот).

Навигатор решает, КУДА и КОГДА лететь. Этот тренажёр учит автопилот
ВЫПОЛНИТЬ перелёт с настоящими ошибками — так, как это случается в игре:

  * импульс ухода длится минуты, а узел считается мгновенным: часть ожога
    до узла, часть после. Чем длиннее ожог относительно витка, тем больше
    потеря и ошибка направления — и тем важнее, какую долю ожога начать
    ДО узла (`eject_lead`);
  * ошибка скорости ухода за месяцы пути превращается в промах мимо
    планеты;
  * промах известен неточно: в начале пути оценка грубая, у цели точная.
    Поправка рано — дешёвая, но неточная; поздно — точная, но дорогая.
    Автопилот выбирает, когда править (`corr1`, `corr2`) и при каком
    промахе вообще не трогать курс (`threshold`);
  * у планеты с атмосферой — коридор аэрозахвата: слишком глубоко —
    сгорел, слишком высоко — пролетел мимо (`aero_depth`). Без атмосферы —
    торможение двигателем на высоте с поправкой `cap_bias`.

Обучение — метод кросс-энтропии: популяция параметров, лучшие 12 %
задают следующее поколение. Для каждой планеты и способа захвата свои
параметры. Всё векторизовано: одно поколение — десятки тысяч перелётов.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np

from ..config import DATA_DIR
from .solar import HOME, PLANETS, SolarSystem

PROFILE_FILE = DATA_DIR / "interplanetary_profile.json"
G0 = 9.80665

# Параметры исполнения и их пределы
PARAMS = {
    "eject_lead": (0.3, 0.7),       # доля ожога ухода ДО узла
    "corr1": (0.05, 0.85),          # когда первая коррекция (доля пути)
    "corr2": (0.85, 0.99),          # когда вторая
    "corr3": (0.99, 0.9995),        # третья — у самой сферы действия (долгие перелёты)
    "threshold": (0.0, 1.0),        # порог промаха для коррекции (доля радиуса цели, лог-шкала)
    "cap_bias": (-0.4, 2.0),        # поправка высоты захвата к плановой (доля): выше — безопаснее
    "aero_depth": (0.25, 0.8),      # глубина перицентра аэрозахвата (доля атмосферы)
    "capture_lead": (0.3, 0.7),     # доля ожога захвата ДО перицентра
}
NAMES = tuple(PARAMS)


@dataclass
class Planet:
    name: str
    radius: float
    mu: float
    atmosphere: float
    terrain: float
    low: float
    tof: float
    v_inf: float
    ejection: float


def planets() -> dict[str, Planet]:
    S = SolarSystem()
    k = S[HOME]
    out = {}
    for name in PLANETS:
        b = S[name]
        ej, vinf, tof = S.hohmann(name)
        out[name] = Planet(name, b.radius, b.mu, b.atmosphere, b.terrain, b.low_orbit(),
                           tof, vinf, ej)
    return out


def decode(u: np.ndarray) -> dict[str, np.ndarray]:
    """[0,1]^n → физические значения параметров."""
    out = {}
    for i, name in enumerate(NAMES):
        lo, hi = PARAMS[name]
        out[name] = lo + (hi - lo) * np.clip(u[..., i], 0, 1)
    out["threshold_frac"] = 10 ** (-3 + 3 * out["threshold"])    # 0.001 … 1 радиуса
    return out


def simulate(p: Planet, params: dict, n: int, rng: np.random.Generator, aero: bool,
             heatshield: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """n перелётов с разными аппаратами. Возвращает (успех, лишняя Δv м/с)."""
    k_park_period = 2 * math.pi * math.sqrt((600_000 + 80_000) ** 3 / 3.5316e12)
    twr = rng.uniform(0.2, 1.2, n)                       # аппараты бывают вялые и бодрые
    accel = twr * G0
    burn = p.ejection / accel                             # длительность ожога ухода, с
    frac = burn / k_park_period                           # доля витка
    lead = params["eject_lead"]
    # Потери и ошибка направления растут с долей витка и перекосом центровки
    steer_loss = p.ejection * (0.15 * frac ** 2 + 0.6 * frac * (lead - 0.5) ** 2)
    dir_err = p.ejection * (0.002 + 0.05 * frac * np.abs(lead - 0.5)) * np.abs(rng.normal(1, 0.35, n))
    # Промах у цели (м): ошибка скорости × время пути, раскрытая геометрией
    miss = dir_err * p.tof * 0.6 * np.sign(rng.normal(0, 1, n))
    extra = steer_loss.copy()
    thr = params["threshold_frac"] * p.radius

    for key in ("corr1", "corr2", "corr3"):
        f = params[key]
        remaining = p.tof * (1 - f)
        # Оценка промаха: чем дальше от цели, тем грубее
        est_sigma = np.abs(miss) * 0.35 * (1 - f) + 2e4 * (1 - f) + 500.0
        est = miss + rng.normal(0, 1, n) * est_sigma
        do = np.abs(est) > thr
        dv = np.abs(est) / np.maximum(remaining, 1.0)
        exec_err = dv * 0.01 * rng.normal(0, 1, n) * remaining
        miss = np.where(do, miss - est + exec_err, miss)
        extra = extra + np.where(do, dv + 0.5, 0.0)       # 0.5 м/с — разворот и доводка

    # Промах → ошибка высоты перицентра (фокусировка гравитацией цели)
    focus = 1.0 / math.sqrt(1 + 2 * p.mu / (p.radius * p.v_inf ** 2))
    pe_err = miss * focus
    if aero:
        plan_alt = params["aero_depth"] * p.atmosphere
        pe = plan_alt + pe_err
        depth = pe / max(p.atmosphere, 1.0)
        burned = (depth < 0.3) | ((depth < 0.45) & ~heatshield) | (pe < p.terrain)
        skipped = depth > 0.72
        success = ~burned & ~skipped
        # после захвата — подъём перицентра и скругление (мало)
        extra = extra + 30.0 + np.abs(depth - 0.5) * 60.0
    else:
        plan_alt = p.low * (1 + params["cap_bias"])
        pe = plan_alt + pe_err
        crashed = pe < max(p.terrain, p.atmosphere) + 2_000.0
        # цена исправления высоты до рабочей орбиты
        rp, rg = p.radius + np.maximum(pe, 1.0), p.radius + p.low
        v_hyp = np.sqrt(p.v_inf ** 2 + 2 * p.mu / rp)
        ideal = math.sqrt(p.v_inf ** 2 + 2 * p.mu / rg) - math.sqrt(p.mu / rg)
        capture = v_hyp - np.sqrt(p.mu / rp)
        a = (rp + rg) / 2
        move = np.abs(np.sqrt(p.mu * (2 / rp - 1 / a)) - np.sqrt(p.mu / rp)) + \
            np.abs(np.sqrt(p.mu / rg) - np.sqrt(p.mu * (2 / rg - 1 / a)))
        cap_burn = capture / accel
        cap_loss = capture * 0.3 * (cap_burn / (2 * math.pi * np.sqrt(rp ** 3 / p.mu))) ** 2 \
            * (1 + 8 * (params["capture_lead"] - 0.5) ** 2)
        extra = extra + np.maximum(capture + move - ideal, 0) + cap_loss
        success = ~crashed
    return success, extra


def score(success: np.ndarray, extra: np.ndarray) -> float:
    return float(success.mean() * 10.0 - np.mean(np.where(success, extra, 0.0)) / 50.0)


def train_planet(p: Planet, aero: bool, rng: np.random.Generator, generations: int = 40,
                 population: int = 96, episodes: int = 600, echo=print) -> dict:
    dim = len(NAMES)
    mean = np.full(dim, 0.5)
    std = np.full(dim, 0.25)
    best = (-1e9, None, None)
    for gen in range(generations):
        cand = np.clip(mean + std * rng.normal(0, 1, (population, dim)), 0, 1)
        scores = np.zeros(population)
        for i in range(population):
            params = decode(cand[i])
            s, e = simulate(p, params, episodes, rng, aero)
            scores[i] = score(s, e)
        elite = cand[np.argsort(scores)[-population // 8:]]
        mean = elite.mean(0)
        std = np.maximum(elite.std(0), 0.02)
        if scores.max() > best[0]:
            best = (scores.max(), cand[np.argmax(scores)], None)
    # Экзамен лучших параметров на свежих перелётах
    params = decode(best[1])
    s, e = simulate(p, params, 20_000, np.random.default_rng(12345), aero)
    result = {k: float(params[k]) for k in NAMES}
    result.update(mode="aero" if aero else "propulsive", success=float(s.mean()),
                  extra_dv=float(np.mean(e[s])) if s.any() else None,
                  passed=bool(s.mean() >= 0.95))
    # Для сравнения: те же перелёты без обучения (середина всех диапазонов)
    base = decode(np.full(dim, 0.5))
    sb, eb = simulate(p, base, 20_000, np.random.default_rng(12345), aero)
    result["untrained_success"] = float(sb.mean())
    result["untrained_extra_dv"] = float(np.mean(eb[sb])) if sb.any() else None
    return result


def train(echo=print, generations: int = 40) -> dict:
    rng = np.random.default_rng(7)
    out = {"trained_at": time.strftime("%Y-%m-%d %H:%M"), "planets": {}}
    for name, p in planets().items():
        modes = [False] + ([True] if p.atmosphere > 0 else [])
        out["planets"][name] = {}
        for aero in modes:
            t0 = time.time()
            r = train_planet(p, aero, rng, generations=generations)
            out["planets"][name][r["mode"]] = r
            echo(f"{name:6} {r['mode']:10} успех {r['success'] * 100:5.1f} % "
                 f"(без обучения {r['untrained_success'] * 100:5.1f} %), "
                 f"лишняя Δv {r['extra_dv'] or 0:6.0f} м/с "
                 f"(без обучения {r['untrained_extra_dv'] or 0:6.0f}) — "
                 f"{'СДАН' if r['passed'] else 'не сдан'}  [{time.time() - t0:.0f} с]")
        out["planets"][name]["passed"] = any(v.get("passed") for v in out["planets"][name].values()
                                             if isinstance(v, dict))
    PROFILE_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
