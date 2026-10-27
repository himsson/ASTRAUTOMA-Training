"""СНИМОК МОЗГА для живой карты: data/brain_map.json.

Тренер каждые несколько секунд сбрасывает сюда состояние сети, а
страница `kia_brain_map.html` его рисует. Ничего секретного и никакой
магии: всё, что видно на карте, берётся из настоящих весов.

ЧТО ЗА «НЕЙРОНЫ» НА КАРТЕ

Рисовать 78 тысяч весов бессмысленно — получится каша. Поэтому карта
показывает три честных слоя:

    ВХОДЫ (40 кружков)   ровно те признаки, что видит сеть. У каждого
                         своё имя, текущее среднее и разброс, которые
                         сеть насчитала сама (нормализатор), и ВЕС
                         ВНИМАНИЯ — суммарная величина связей этого
                         входа с первым слоем. Чем толще связь, тем
                         больше решение зависит от этого прибора.

    ЯДРО (2 слоя)        внутренние слои показаны сводкой: сколько
                         нейронов живо (не залипло в нуле), какой у
                         слоя средний вес. Отдельные нейроны скрытого
                         слоя не имеют смысла по одиночке — врать про
                         них карта не будет.

    ВЫХОДЫ (5 кружков)   тяга, тангаж, рыскание, крен, отделение
                         ступени. У каждого текущий разброс действий
                         (σ): чем он меньше, тем увереннее сеть.

    НАВЫКИ               отдельный слой: что сеть уже показала в
                         тренажёрах (`kia_sim_knowledge.json`).

ВЕС ВНИМАНИЯ — ЧТО ЭТО ТОЧНО

Сумма модулей весов первого слоя по данному входу, поделённая на
среднее по всем входам. Единица — «внимание как у среднего прибора»,
два — «вдвое важнее среднего». Это не «важность признака» в строгом
смысле (для неё нужны абляции), а именно величина связей: показывает,
куда сеть вообще смотрит.
"""
from __future__ import annotations

import json
import math
import time

from ..config import DATA_DIR
from ..logging_setup import get_logger

log = get_logger("neural.brain_map")

MAP_FILE = DATA_DIR / "brain_map.json"

# Человеческие имена входов: на карте должно быть понятно без словаря
FEATURE_TITLES = {
    "altitude": "Высота над уровнем моря",
    "surface_altitude": "Высота над поверхностью",
    "apoapsis": "Апоапсис — верхняя точка орбиты",
    "periapsis": "Периапсис — нижняя точка орбиты",
    "time_to_apoapsis": "Время до апоапсиса",
    "eccentricity": "Эксцентриситет орбиты",
    "inclination": "Наклонение орбиты",
    "speed": "Полная скорость",
    "vertical_speed": "Вертикальная скорость",
    "horizontal_speed": "Горизонтальная скорость",
    "pitch_sin": "Тангаж (синус)",
    "pitch_cos": "Тангаж (косинус)",
    "heading_sin": "Курс (синус)",
    "heading_cos": "Курс (косинус)",
    "roll_sin": "Крен (синус)",
    "roll_cos": "Крен (косинус)",
    "prograde_pitch_sin": "Направление движения (синус)",
    "prograde_pitch_cos": "Направление движения (косинус)",
    "angle_of_attack": "Угол атаки — нос против потока",
    "dynamic_pressure": "Скоростной напор",
    "static_pressure": "Атмосферное давление",
    "atmosphere_density": "Плотность воздуха",
    "g_force": "Перегрузка",
    "mass": "Полная масса аппарата",
    "dry_mass": "Сухая масса",
    "thrust": "Текущая тяга",
    "available_thrust": "Доступная тяга",
    "specific_impulse": "Удельный импульс двигателя",
    "throttle": "Положение дросселя",
    "stage": "Сколько ступеней осталось",
    "part_count": "Число деталей",
    "liquid_fuel_fraction": "Остаток топлива",
    "oxidizer_fraction": "Остаток окислителя",
    "electric_fraction": "Заряд батареи",
    "mission_time": "Время полёта",
    "body_gravity": "Сила тяжести планеты",
    "body_radius": "Радиус планеты",
    "has_atmosphere": "Есть ли атмосфера",
    "goal_altitude": "Заданная высота орбиты",
    "goal_progress": "Насколько близко к цели",
}

OUTPUT_TITLES = [
    ("throttle", "Дроссель", "насколько открыть газ"),
    ("pitch", "Тангаж", "поднять или опустить нос"),
    ("yaw", "Рыскание", "довернуть влево или вправо"),
    ("roll", "Крен", "повернуть аппарат вокруг оси"),
    ("stage", "Отделение ступени", "сбросить отработавшую ступень"),
]


def _tensor_stats(tensor) -> dict:
    values = tensor.detach().float().flatten()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std().item()) if values.numel() > 1 else 0.0,
        "max": float(values.abs().max().item()),
    }


def snapshot(brain, report=None, knowledge=None, extra: dict | None = None) -> dict:
    """Собирает состояние мозга в словарь, готовый к отрисовке."""
    import torch
    from .spaces import FLIGHT_FEATURES

    agent = brain.pilot
    net = agent.net
    with torch.no_grad():
        first = net.trunk[0].weight.detach().float()      # (hidden, входы)
        attention = first.abs().sum(dim=0)                # по каждому входу
        average = float(attention.mean().item()) or 1.0
        normalizer = net.normalizer
        means = normalizer.mean.detach().float().tolist()
        variances = normalizer.var.detach().float().tolist()

        inputs = []
        for index, (name, scale) in enumerate(FLIGHT_FEATURES):
            weight = float(attention[index].item()) / average
            inputs.append({
                "id": f"in:{name}",
                "name": name,
                "title": FEATURE_TITLES.get(name, name),
                "attention": round(weight, 3),
                "mean": round(means[index], 4) if index < len(means) else 0.0,
                "spread": round(math.sqrt(max(0.0, variances[index])), 4)
                if index < len(variances) else 0.0,
                "scale": scale,
            })

        layers = []
        linear = [module for module in net.trunk if hasattr(module, "weight")]
        for depth, module in enumerate(linear, start=1):
            weights = module.weight.detach().float()
            alive = int((weights.abs().sum(dim=1) > 1e-4).sum().item())
            layers.append({
                "id": f"layer:{depth}",
                "title": f"Внутренний слой {depth}",
                "neurons": int(weights.shape[0]),
                "alive": alive,
                "stats": {k: round(v, 4)
                          for k, v in _tensor_stats(weights).items()},
            })

        sigma = []
        if net.log_std is not None:
            clamped = net.log_std.detach().clamp(net.LOG_STD_MIN,
                                                 net.LOG_STD_MAX)
            sigma = [round(math.exp(value), 4) for value in clamped.tolist()]

        # Средняя команда по оси — прибор против «мёртвых осей»: если
        # выход на этой ступени ни на что не влияет, он копит произвольное
        # значение и выстрелит на следующей (см. MIGRATION 5.3i)
        axes = list(getattr(report, "axis_means", []) or []) if report else []
        outputs = []
        for index, (key, title, hint) in enumerate(OUTPUT_TITLES):
            outputs.append({
                "id": f"out:{key}",
                "name": key,
                "title": title,
                "hint": hint,
                "sigma": sigma[index] if index < len(sigma) else None,
                "command": axes[index] if index < len(axes) else None,
            })

    stats = agent.stats
    data = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": getattr(brain, "stage", 1),
        "device": str(agent.device),
        "weights": net.parameter_count(),
        "training": {
            "updates": stats.updates,
            "decisions": stats.total_steps,
            "episodes": stats.episodes,
            "best_return": (round(stats.best_return, 2)
                            if stats.best_return > float("-inf") else None),
            "mean_return": round(agent.mean_return(20), 2),
            "recent": stats.last_returns[-60:],
        },
        "inputs": inputs,
        "layers": layers,
        "outputs": outputs,
        "knowledge": knowledge.to_dict() if knowledge is not None else {},
    }
    if report is not None:
        data["run"] = {
            "stage": report.stage,
            "title": report.mode,
            "steps": report.steps,
            "episodes": report.episodes,
            "orbits": report.successes,
            "difficulty": round(report.difficulty, 3),
            "envs": getattr(report, "envs", 1),
            "failures": dict(report.failures),
            "mean_return": round(report.mean_return, 2),
        }
    if extra:
        data.update(extra)
    return data


def write(brain, report=None, knowledge=None, path=None,
          extra: dict | None = None):
    """Сохраняет снимок на диск. Пишется атомарно — карта не поймает обрывок."""
    target = path or MAP_FILE
    try:
        payload = snapshot(brain, report=report, knowledge=knowledge,
                           extra=extra)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False),
                             encoding="utf-8")
        temporary.replace(target)
        return target
    except Exception as exc:                    # карта не должна ронять счёт
        log.debug("Снимок мозга не записан: %s", exc)
        return None


__all__ = ["snapshot", "write", "MAP_FILE", "FEATURE_TITLES", "OUTPUT_TITLES"]
