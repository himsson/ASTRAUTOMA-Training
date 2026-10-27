"""ТРЁХМЕРНАЯ КАРТА МОЗГА: данные для kia_brain3d.html.

Плоская карта (`brain_map.py`) — рабочий прибор: по ней удобно ловить
упор по осям и залипший разброс действий. Эта — про другое: показать
мозг как объём, в который можно залететь.

ДВА СЛОЯ, И ЭТО ПРИНЦИПИАЛЬНО

    СНАРУЖИ — НАВЫКИ. Кружки с настоящими названиями: «оторвался от
    стартового стола», «прошёл звуковой барьер», «вышел на орбиту».
    Они берутся из `kia_sim_knowledge.json`, то есть из того, что сеть
    РЕАЛЬНО показала в тренажёрах, и появляются по мере обучения. На
    чистом мозге не появится ни одного — останется только ядро.
    Линии между ними — логические зависимости: не пройдя звуковой
    барьер, не окажешься в верхней атмосфере.

    ВНУТРИ — НЕЙРОНЫ. Настоящие: 40 входов, два внутренних слоя по 256
    и 5 выходов. Связи — самые сильные веса. Сюда можно залететь сквозь
    оболочку ядра.

ЧЕСТНО ПРО ПОДПИСИ НЕЙРОНОВ

У скрытого нейрона нет собственного смысла: это число, а не «детектор
высоты». Единственное, что можно сказать о нём правдиво, — на какие
входы он сильнее всего опирается и насколько влияет на решение. Так
подписи и построены. Никаких «этот нейрон отвечает за посадку».

КОГДА НЕЙРОН СЧИТАЕТСЯ ПРОСНУВШИМСЯ

Архитектура сети фиксирована с рождения: все 512 скрытых нейронов
существуют с первой секунды. Но при рождении они ни на что не влияют —
головы политики инициализируются с усилением 0.01, то есть почти нулём.
Влияние появляется только по мере обучения, и его можно измерить:

    влияние(нейрон второго слоя) = сумма |весов| в головы решений
    влияние(нейрон первого слоя) = сумма |весов| во второй слой,
                                   взвешенная влиянием тех нейронов

Это распространение величин связей, а не строгая мера важности (для неё
нужны абляции). Но оно честно отвечает на вопрос «участвует ли этот
нейрон в решении вообще» — а именно это и значит «проснулся».

Порог берётся не с потолка. Рядом собирается ЧИСТАЯ СЕТЬ той же
архитектуры, и её влияние — это уровень «нейрон существует, но ничего
не решает». Замер на живых весах:

    слой 1: медиана влияния 15.7 против 0.054 у чистой сети  (×290)
    слой 2: медиана влияния 0.40 против 0.0030 у чистой      (×134)
    голова политики: средний |вес| 0.0854 против 0.00050     (×170)

Разрыв в сотню раз, поэтому граница проводится уверенно: проснулся тот,
кто влияет хотя бы в несколько раз сильнее новорождённого. У чистого
мозга по этому правилу не просыпается ни один — на карте остаётся голое
ядро, ровно как и должно быть.
"""
from __future__ import annotations

import json
import math
import time

from ..config import DATA_DIR
from ..logging_setup import get_logger
from .brain_map import FEATURE_TITLES, OUTPUT_TITLES
from .knowledge import SKILL_HINTS, SKILL_TITLES, SKILLS

log = get_logger("neural.brain3d")

MAP3D_FILE = DATA_DIR / "brain3d.json"

# Логические зависимости навыков: что без чего не бывает. Это описание
# задачи, а не результат обучения, — линии на карте должны означать
# «сначала одно, потом другое», иначе граф превращается в украшение.
SKILL_PARENTS: dict[str, tuple[str, ...]] = {
    "liftoff": (),
    "cleared_tower": ("liftoff",),
    "supersonic": ("cleared_tower",),
    "high_atmosphere": ("supersonic",),
    "space": ("high_atmosphere",),
    "orbit": ("space",),
    "clean_flight": ("liftoff",),
    "smooth_steering": ("cleared_tower",),
    "power_kept": ("cleared_tower",),
}

# Навык считается ОСВОЕННЫМ, а не просто случайно показанным, если он
# подтверждался хотя бы в этой доле попыток. Одна орбита на тысячу
# эпизодов — это удача разброса действий, а не умение; ровно на такой
# подмене проект однажды потерял ночь (MIGRATION 5.3d).
SKILL_MASTERED_RATE = 0.35
SKILL_SEEN_MIN = 1              # хотя бы раз подтверждён — уже виден

# Сколько связей рисовать между соседними слоями. Всех 78 тысяч не нужно:
# получится сплошная каша, в которой ничего не разглядеть.
EDGES_PER_LAYER = 420

# Во сколько раз влияние нейрона должно превысить уровень новорождённой
# сети, чтобы считать его проснувшимся. Замеренный разрыв — сто с лишним
# раз, так что четвёрка стоит с большим запасом от обеих границ: чистый
# мозг не даёт ложных срабатываний, обученный не теряет работающих.
AWAKE_BIRTH_FACTOR = 4.0

# Зависимости разделов математики: что на чём стоит. Как и у навыков
# полёта, это описание предмета, а не результат обучения: не зная
# круговой скорости, нечего делать в гомановском переходе.
MATH_PARENTS: dict[str, tuple[str, ...]] = {
    "circular_speed": (),
    "gravity_at": (),
    "orbital_period": ("circular_speed",),
    "tsiolkovsky": (),
    "twr": ("gravity_at",),
    "burn_time": ("tsiolkovsky",),
    "vis_viva": ("circular_speed", "orbital_period"),
    "raise_apoapsis": ("vis_viva",),
    "circularize": ("vis_viva",),
    "hohmann": ("raise_apoapsis", "circularize"),
    "hohmann_time": ("orbital_period",),
    "apoapsis_from_state": ("vis_viva",),
    "phase_angle": ("hohmann_time",),
    "synodic_period": ("orbital_period",),
    "plane_change": ("circular_speed",),
    "escape_dv": ("circular_speed",),
    "capture_dv": ("escape_dv",),
    "hyperbolic_excess": ("escape_dv",),
}

# Умения конструктора. Это НЕ список пожеланий: каждое проверяется на
# сборках, которые сеть делает прямо сейчас, и доля успешных — и есть
# оценка. Порядок от простого к сложному.
DESIGN_SKILLS: tuple[tuple[str, str, str], ...] = (
    ("assembles", "Собирает годный аппарат",
     "в сборке есть ядро, бак и двигатель, и она не отвергнута проверкой"),
    ("liftoff_twr", "Отрывает от стола",
     "тяговооружённость выше единицы — иначе ракета не тронется"),
    ("has_power", "Помнит про энергию",
     "беспилотник без заряда теряет управление на второй минуте"),
    ("has_link", "Помнит про связь",
     "без антенны зонд неуправляем вне зоны видимости"),
    ("enough_dv", "Набирает запас скорости",
     "ΔV покрывает задачу, а не только красиво выглядит"),
    ("buildable", "Чертёж открывается в игре",
     "сборка переводится в .craft и проходит валидатор"),
    ("frugal", "Не расточительствует",
     "не больше десяти деталей: каждая лишняя — масса"),
)

DESIGN_TITLES = {key: title for key, title, _ in DESIGN_SKILLS}
DESIGN_HINTS = {key: hint for key, _, hint in DESIGN_SKILLS}

# Сколько сборок и задач берётся на замер для карты. Больше не нужно:
# карта показывает порядок величины, а не приёмочный экзамен (для него
# есть пункт «Экзамен математика» и `kia_neural.py math-eval`).
MAP_MATH_TASKS = 180
MAP_DESIGN_BUILDS = 32
# Замер живёт столько секунд. Карта перечитывается каждые две секунды, и
# гонять сеть по три сотни раз на каждый кадр — верный способ отобрать
# процессор у самого обучения.
MAP_PROBE_TTL = 90.0

# Геометрия сцены (мировые единицы, их же видит страница)
CORE_RADIUS = 18.0              # оболочка ядра: внутрь неё залетают
SKILL_SHELL = (34.0, 48.0, 62.0, 76.0)   # радиусы колец по ступеням
MATH_SHELL = 96.0               # оболочка математики
DESIGN_SHELL = 114.0            # оболочка конструктора
MISSION_RADIUS = 132.0          # оболочка навыков ЖИВОЙ игры
MISSION_NODE_RADIUS = 3.4       # узел полётного навыка
LAYER_X = (-12.0, -4.0, 4.0, 12.0)       # входы, слой 1, слой 2, выходы
LAYER_SPREAD = (9.0, 7.5, 7.5, 5.0)      # радиус диска каждого слоя

GOLDEN = math.pi * (3.0 - math.sqrt(5.0))    # золотой угол


def _disc_point(index: int, total: int, radius: float) -> tuple[float, float]:
    """Точка на диске по спирали Фибоначчи — равномерно и без случайности.

    Позиции обязаны быть УСТОЙЧИВЫМИ: карта перечитывается каждые две
    секунды, и если раскладка будет случайной, картинка задрожит.
    """
    if total <= 1:
        return 0.0, 0.0
    step = math.sqrt((index + 0.5) / total) * radius
    angle = index * GOLDEN
    return step * math.cos(angle), step * math.sin(angle)


def _sphere_point(index: int, total: int, radius: float) -> tuple[float, float, float]:
    """Точка на сфере по той же спирали: кольцо навыков вокруг ядра."""
    if total <= 1:
        return 0.0, radius, 0.0
    y = 1.0 - 2.0 * (index + 0.5) / total
    ring = math.sqrt(max(0.0, 1.0 - y * y))
    angle = index * GOLDEN
    return (radius * ring * math.cos(angle), radius * y,
            radius * ring * math.sin(angle))


# ==========================================================================
def _skill_nodes(knowledge: dict, current_stage: int) -> tuple[list, list]:
    """Внешний слой: что сеть умеет, по ступеням."""
    from .curriculum import STAGES

    nodes: list[dict] = []
    edges: list[dict] = []
    stages = (knowledge or {}).get("stages") or {}

    for stage in STAGES:
        entry = stages.get(str(stage.number)) or {}
        skills = entry.get("skills") or {}
        shell = SKILL_SHELL[min(stage.number, len(SKILL_SHELL)) - 1]
        placed: dict[str, dict] = {}
        visible = [key for key, _, _ in SKILLS
                   if int((skills.get(key) or {}).get("confirmations", 0))
                   >= SKILL_SEEN_MIN]
        for order, key in enumerate(visible):
            record = skills.get(key) or {}
            confirmations = int(record.get("confirmations", 0))
            rate = float(record.get("rate", 0.0))
            x, y, z = _sphere_point(order, max(1, len(visible)), shell)
            node = {
                "id": f"skill:{stage.number}:{key}",
                "kind": "skill",
                "stage": stage.number,
                "title": SKILL_TITLES.get(key, key),
                "hint": SKILL_HINTS.get(key, ""),
                "detail": (f"ступень {stage.number} · {stage.title}\n"
                           f"подтверждено {confirmations} раз "
                           f"({rate:.0%} попыток)"),
                "strength": round(min(1.0, rate / SKILL_MASTERED_RATE), 3),
                "mastered": rate >= SKILL_MASTERED_RATE,
                "current": stage.number == current_stage,
                "pos": [round(x, 2), round(y, 2), round(z, 2)],
            }
            nodes.append(node)
            placed[key] = node

        for key, node in placed.items():
            for parent in SKILL_PARENTS.get(key, ()):
                if parent in placed:
                    edges.append({"a": placed[parent]["id"], "b": node["id"],
                                  "w": round(node["strength"], 3),
                                  "kind": "skill"})
    return nodes, edges



# ==========================================================================
# НАСТОЯЩИЕ ПОЛЁТЫ — ОТДЕЛЬНАЯ ОБОЛОЧКА.
#
# На карте были только ступени тренажёра, и оператор справедливо спросил,
# где же знания о полётах на Муну и Дюну. Их там не было вовсе: навыки
# живой игры копятся в `kia_global_memory.json` (орбита, перехват, захват,
# посадка), а карта строилась только по учебной программе сети.
# ==========================================================================
MISSION_TITLES = {
    "liftoff": "Отрыв от стола",
    "cleared_tower": "Уход от башни",
    "supersonic": "Звуковой барьер",
    "atmosphere_exit": "Выход из атмосферы",
    "stable_orbit": "Орбита Кербина",
    "science_collected": "Сбор науки",
    "viable_design": "Жизнеспособный проект",
    "mun_node_created": "Расчёт окна к Муне",
    "mun_encounter": "Перехват Муны",
    "mun_soi_entered": "Вход в сферу Муны",
    "mun_orbit": "Орбита Муны",
    "target_landing": "Мягкая посадка",
}
# Порядок = путь полёта: от стола до грунта другого тела.
MISSION_ORDER = list(MISSION_TITLES)


def _mission_nodes() -> tuple[list, list]:
    """Кольцо навыков ЖИВОЙ игры вокруг ядра."""
    # `Path` в этом модуле не импортирован — первая версия падала молча
    # на первой же строке и отдавала пустое кольцо.
    from pathlib import Path as _Path
    try:
        from ..memory import MEMORY_FILE
        raw = json.loads(_Path(MEMORY_FILE).read_text(encoding="utf-8"))
    except Exception as exc:
        log.debug("Память полётов не прочитана: %s", exc)
        return [], []
    skills = raw.get("skills") or {}
    known = [k for k in MISSION_ORDER if k in skills]
    known += [k for k in skills if k not in MISSION_TITLES]
    nodes, edges = [], []
    total = max(1, len(MISSION_ORDER))
    for idx, key in enumerate(known):
        record = skills.get(key) or {}
        title = MISSION_TITLES.get(key) or record.get("title") or key
        best = float(record.get("best_score") or 0.0)
        pos = _sphere_point(idx, total, MISSION_RADIUS)
        node = {
            "id": f"mission:{key}", "kind": "mission", "title": title,
            "hint": "проверено в живой игре",
            "detail": (f"впервые {str(record.get('first_achieved') or '')[:10]}"
                       + (f" · лучший счёт {best:.0f}" if best else "")),
            "strength": 1.0, "awake": True, "mastered": True,
            "pos": list(pos), "radius": MISSION_NODE_RADIUS,
        }
        # Связей с ядром не делаем: ядро лежит отдельным полем снимка, а
        # не среди узлов, и самопроверка справедливо ловит такую связь как
        # ведущую в никуда.
        nodes.append(node)
    for a, b in zip(known, known[1:]):
        edges.append({"a": f"mission:{a}", "b": f"mission:{b}",
                      "w": 0.6, "kind": "mission"})
    return nodes, edges

# ==========================================================================
# Замеры двух других сетей. Обе оболочки строятся по ТОМУ ЖЕ правилу, что
# и навыки полёта: показывается не то, что сеть должна уметь, а то, что
# она показала на свежей выборке.
# ==========================================================================
_PROBE_CACHE: dict[str, tuple[float, object]] = {}


def _cached(key: str, builder):
    """Замер с коротким сроком годности — карту обновляют раз в две секунды."""
    now = time.time()
    hit = _PROBE_CACHE.get(key)
    if hit is not None and now - hit[0] < MAP_PROBE_TTL:
        return hit[1]
    try:
        value = builder()
    except Exception as exc:
        log.debug("Замер «%s» не удался: %s", key, exc)
        value = {}
    _PROBE_CACHE[key] = (now, value)
    return value


def _math_probe(brain) -> dict:
    """Доля задач каждого раздела, решённых в допуске своей ступени."""
    from .math_env import MathEnv, TOLERANCE_BY_STAGE

    agent = getattr(brain, "mathematician", None)
    if agent is None:
        return {}
    stage = int((getattr(brain, "stages", None) or {}).get("math", 1))
    env = MathEnv(stage=stage, tolerance=TOLERANCE_BY_STAGE.get(stage, 0.05),
                  seed=20250806)
    obs = env.reset()
    rows: dict[str, list] = {}
    for _ in range(MAP_MATH_TASKS):
        _o, continuous, _d, _lp, _v = agent.act(obs, deterministic=True,
                                                update_normalizer=False)
        answer = [float(v) for v in continuous.squeeze(0).tolist()]
        obs, _reward, _done, info = env.step(answer)
        row = rows.setdefault(info["task"], [0, 0, 0.0, info["title"]])
        row[0] += 1
        row[1] += int(info["solved"])
        row[2] += info["error"]
    return {"stage": stage, "tolerance": env.tolerance, "rows": rows}


def _design_probe(brain) -> dict:
    """Что сеть-конструктор реально выдаёт прямо сейчас."""
    from .builder_env import BuilderEnv, FINISH_ACTION

    agent = getattr(brain, "builder", None)
    if agent is None:
        return {}
    stage = int((getattr(brain, "stages", None) or {}).get("builder", 1))
    env = BuilderEnv(stage=stage, seed=20250806)
    counts = {key: [0, 0] for key, _, _ in DESIGN_SKILLS}

    for _ in range(MAP_DESIGN_BUILDS):
        obs = env.reset()
        done = False
        info: dict = {}
        while not done:
            mask = env.action_mask()
            _o, _c, discrete, _lp, _v = agent.act(obs, action_mask=mask,
                                                  update_normalizer=False)
            action = int(discrete[0].item())
            obs, _reward, done, info = env.step(action)
        evaluation = info.get("evaluation")
        if evaluation is None:
            continue
        chosen = env.chosen
        need = 3400.0 + (1000.0 if env.goal_transfer else 0.0) \
            + (1200.0 if env.goal_landing else 0.0)
        outcome = {
            "assembles": evaluation.valid,
            "liftoff_twr": evaluation.twr >= 1.0,
            "has_power": any(p.resources.get("ElectricCharge", 0) > 0
                             for p in chosen),
            "has_link": any(p.is_antenna for p in chosen),
            "enough_dv": evaluation.delta_v >= need,
            "buildable": not env._buildability_problems(),
            "frugal": evaluation.part_count <= 10,
        }
        for key, ok in outcome.items():
            counts[key][0] += 1
            counts[key][1] += int(ok)
    return {"stage": stage, "rows": counts}


def _math_nodes(brain) -> tuple[list, list]:
    """Оболочка математики: по узлу на раздел, размер — по освоенности."""
    probe = _cached("math", lambda: _math_probe(brain))
    rows = (probe or {}).get("rows") or {}
    if not rows:
        return [], []
    stage = int(probe.get("stage", 1))
    tolerance = float(probe.get("tolerance", 0.05))

    order = [key for key in MATH_PARENTS if key in rows]
    order += [key for key in rows if key not in MATH_PARENTS]
    nodes: list[dict] = []
    placed: dict[str, dict] = {}
    for index, key in enumerate(order):
        total, solved, error_sum, title = rows[key]
        rate = solved / total if total else 0.0
        x, y, z = _sphere_point(index, max(1, len(order)), MATH_SHELL)
        node = {
            "id": f"math:{key}",
            "kind": "math",
            "stage": stage,
            "title": title,
            "hint": "раздел математики манёвров",
            "detail": (f"решено {solved} из {total} задач "
                       f"({rate:.0%}) при допуске {tolerance:.0%}\n"
                       f"средняя ошибка {error_sum / max(1, total) * 100:.2f} %\n"
                       f"эталон — те же формулы, которыми считает полётный код"),
            "strength": round(rate, 3),
            "mastered": rate >= 0.8,
            "current": True,
            "pos": [round(x, 2), round(y, 2), round(z, 2)],
        }
        nodes.append(node)
        placed[key] = node

    edges = []
    for key, node in placed.items():
        for parent in MATH_PARENTS.get(key, ()):
            if parent in placed:
                edges.append({"a": placed[parent]["id"], "b": node["id"],
                              "w": round(node["strength"], 3), "kind": "skill"})
    return nodes, edges


def _design_nodes(brain) -> tuple[list, list]:
    """Оболочка конструктора: умения, проверенные на свежих сборках."""
    probe = _cached("design", lambda: _design_probe(brain))
    rows = (probe or {}).get("rows") or {}
    if not rows or not any(total for total, _ in rows.values()):
        return [], []
    stage = int(probe.get("stage", 1))

    nodes: list[dict] = []
    order = [key for key, _, _ in DESIGN_SKILLS if key in rows]
    for index, key in enumerate(order):
        total, good = rows[key]
        rate = good / total if total else 0.0
        x, y, z = _sphere_point(index, max(1, len(order)), DESIGN_SHELL)
        nodes.append({
            "id": f"design:{key}",
            "kind": "design",
            "stage": stage,
            "title": DESIGN_TITLES.get(key, key),
            "hint": DESIGN_HINTS.get(key, ""),
            "detail": (f"выполнено в {good} из {total} сборок ({rate:.0%})\n"
                       f"замер сделан сетью-конструктором на ступени {stage}"),
            "strength": round(rate, 3),
            "mastered": rate >= 0.8,
            "current": True,
            "pos": [round(x, 2), round(y, 2), round(z, 2)],
        })
    # Порядок умений — это и есть их зависимость: не собрав годный
    # аппарат, нечего говорить про экономию деталей.
    edges = [{"a": nodes[i]["id"], "b": nodes[i + 1]["id"],
              "w": round(nodes[i + 1]["strength"], 3), "kind": "skill"}
             for i in range(len(nodes) - 1)]
    return nodes, edges


# ==========================================================================
def _influence(net):
    """Насколько каждый нейрон участвует в РЕШЕНИИ. См. заголовок модуля."""
    import torch

    with torch.no_grad():
        heads = []
        if net.mu_head is not None:
            heads.append(net.mu_head.weight.detach().abs())
        for head in net.discrete_heads:
            heads.append(head.weight.detach().abs())
        if not heads:
            raise RuntimeError("у сети нет голов решений")
        # влияние нейронов последнего внутреннего слоя
        last = torch.cat(heads, dim=0).sum(dim=0).float()

        linear = [m for m in net.trunk if hasattr(m, "weight")]
        influences = [last]
        for module in reversed(linear[1:]):
            weights = module.weight.detach().abs().float()   # (выход, вход)
            influences.append((weights * influences[-1].unsqueeze(1)).sum(dim=0))
        influences.reverse()          # теперь по порядку слоёв
        return [value.cpu() for value in influences]


_BIRTH_CACHE: dict[tuple, list[float]] = {}


def _birth_influence(net) -> list[float]:
    """Уровень «нейрон есть, но ничего не решает» — по чистой сети-двойнику.

    Собирается сеть той же архитектуры со свежей инициализацией, и с неё
    снимается медианное влияние по каждому слою. Это и есть точка отсчёта:
    порог не приходится подбирать руками, он сам подстроится под любую
    архитектуру, если однажды поменяется размер слоёв.
    """
    from .networks import ActorCritic

    linear = [m for m in net.trunk if hasattr(m, "weight")]
    key = (net.obs_dim, net.continuous_dim, tuple(net.discrete_sizes),
           int(linear[0].weight.shape[0]), len(linear))
    cached = _BIRTH_CACHE.get(key)
    if cached is not None:
        return cached
    twin = ActorCritic(net.obs_dim, net.continuous_dim, net.discrete_sizes,
                       hidden=key[3], depth=key[4])
    levels = [float(values.float().median().item())
              for values in _influence(twin)]
    _BIRTH_CACHE[key] = levels
    return levels


def _top_edges(weights, source_ids: list[str], target_ids: list[str],
               limit: int, alive_source=None, alive_target=None) -> list[dict]:
    """Самые сильные связи между двумя слоями."""
    import torch

    with torch.no_grad():
        matrix = weights.detach().abs().float()          # (цель, источник)
        if alive_target is not None:
            matrix = matrix * alive_target.unsqueeze(1)
        if alive_source is not None:
            matrix = matrix * alive_source.unsqueeze(0)
        flat = matrix.flatten()
        count = min(int(limit), flat.numel())
        if count <= 0:
            return []
        values, indices = torch.topk(flat, count)
        peak = float(values[0].item()) or 1.0
        columns = matrix.shape[1]
        edges = []
        for value, index in zip(values.tolist(), indices.tolist()):
            if value <= 0.0:
                break
            target = index // columns
            source = index % columns
            edges.append({"a": source_ids[source], "b": target_ids[target],
                          "w": round(value / peak, 3), "kind": "wire"})
        return edges


def _neuron_nodes(brain) -> tuple[list, list]:
    """Внутренний слой: настоящие нейроны сети и их связи."""
    import torch

    from .spaces import FLIGHT_FEATURES

    net = brain.pilot.net
    linear = [m for m in net.trunk if hasattr(m, "weight")]
    influences = _influence(net)
    birth = _birth_influence(net)
    nodes: list[dict] = []
    edges: list[dict] = []

    with torch.no_grad():
        first = linear[0].weight.detach().abs().float()      # (скрытый, вход)
        attention = first.sum(dim=0)
        average = float(attention.mean().item()) or 1.0

        # --- входы: у них настоящие имена ---
        input_ids = []
        for index, (name, _) in enumerate(FLIGHT_FEATURES):
            x, y = _disc_point(index, len(FLIGHT_FEATURES), LAYER_SPREAD[0])
            node_id = f"in:{name}"
            input_ids.append(node_id)
            share = float(attention[index].item()) / average
            nodes.append({
                "id": node_id, "kind": "input", "layer": 0,
                "title": FEATURE_TITLES.get(name, name),
                "hint": "прибор, который видит сеть",
                "detail": (f"вес внимания {share:.2f} — во столько раз "
                           f"связи этого входа сильнее среднего"),
                "strength": round(min(1.0, share / 2.5), 3),
                "awake": True,
                "pos": [LAYER_X[0], round(y, 2), round(x, 2)],
            })

        # --- скрытые слои ---
        layer_ids: list[list[str]] = []
        alive_masks = []
        for depth, module in enumerate(linear):
            influence = influences[depth]
            peak = float(influence.max().item()) or 1.0
            threshold = AWAKE_BIRTH_FACTOR * max(1e-12, birth[depth])
            neurons = int(module.weight.shape[0])
            ids = []
            mask = torch.zeros(neurons)
            incoming = module.weight.detach().abs().float()   # (нейрон, вход)
            for index in range(neurons):
                value = float(influence[index].item())
                share = value / peak
                awake = value >= threshold
                mask[index] = 1.0 if awake else 0.0
                node_id = f"h{depth + 1}:{index}"
                ids.append(node_id)
                if not awake:
                    continue                     # спящих не рисуем вовсе
                x, y = _disc_point(index, neurons, LAYER_SPREAD[depth + 1])
                if depth == 0:
                    top = torch.topk(incoming[index],
                                     min(3, incoming.shape[1])).indices.tolist()
                    names = [FEATURE_TITLES.get(FLIGHT_FEATURES[i][0],
                                                FLIGHT_FEATURES[i][0])
                             for i in top]
                    detail = "сильнее всего опирается на:\n· " + "\n· ".join(names)
                else:
                    detail = ("собирает сигнал первого слоя; влияние на "
                              "решение — доля от самого влиятельного нейрона")
                nodes.append({
                    "id": node_id, "kind": "neuron", "layer": depth + 1,
                    "title": f"Нейрон {depth + 1}.{index}",
                    "hint": f"участвует в решении на {share:.0%} от сильнейшего",
                    "detail": (f"{detail}\nвлияние в {value / threshold * AWAKE_BIRTH_FACTOR:.0f} "
                               f"раз выше, чем у нейрона необученной сети"),
                    "strength": round(share, 3),
                    "awake": True,
                    "pos": [LAYER_X[depth + 1], round(y, 2), round(x, 2)],
                })
            layer_ids.append(ids)
            alive_masks.append(mask)

        # --- выходы: решения ---
        sigma = []
        if net.log_std is not None:
            clamped = net.log_std.detach().clamp(net.LOG_STD_MIN, net.LOG_STD_MAX)
            sigma = [math.exp(value) for value in clamped.tolist()]
        output_ids = []
        for index, (key, title, hint) in enumerate(OUTPUT_TITLES):
            x, y = _disc_point(index, len(OUTPUT_TITLES), LAYER_SPREAD[3])
            node_id = f"out:{key}"
            output_ids.append(node_id)
            spread = (f"разброс действий σ {sigma[index]:.3f}"
                      if index < len(sigma) else "дискретное решение")
            nodes.append({
                "id": node_id, "kind": "output", "layer": 3,
                "title": title, "hint": hint, "detail": spread,
                "strength": 1.0, "awake": True,
                "pos": [LAYER_X[3], round(y, 2), round(x, 2)],
            })

        # --- связи ---
        edges += _top_edges(linear[0].weight, input_ids, layer_ids[0],
                            EDGES_PER_LAYER, alive_target=alive_masks[0])
        for depth in range(1, len(linear)):
            edges += _top_edges(linear[depth].weight, layer_ids[depth - 1],
                                layer_ids[depth], EDGES_PER_LAYER,
                                alive_source=alive_masks[depth - 1],
                                alive_target=alive_masks[depth])
        head_rows = []
        if net.mu_head is not None:
            head_rows.append(net.mu_head.weight.detach())
        for head in net.discrete_heads:
            # у дискретной головы две строки на одно решение — берём разницу:
            # именно она и определяет выбор
            rows = head.weight.detach()
            head_rows.append((rows[1:2] - rows[0:1]) if rows.shape[0] > 1 else rows)
        heads = torch.cat(head_rows, dim=0)
        edges += _top_edges(heads, layer_ids[-1], output_ids,
                            EDGES_PER_LAYER // 2,
                            alive_source=alive_masks[-1])

    return nodes, edges


# ==========================================================================
def build(brain, knowledge=None, report=None) -> dict:
    """Полный снимок для страницы: ядро, навыки, нейроны, связи."""
    from .curriculum import STAGES

    stage = int(getattr(brain, "stage", 1) or 1)
    knowledge_data = knowledge.to_dict() if knowledge is not None else {}
    skills, skill_edges = _skill_nodes(knowledge_data, stage)
    # Две другие сети живут в том же файле весов и учатся своим программам.
    # Не показывать их — значит показывать треть мозга и называть это мозгом.
    math_nodes, math_edges = _math_nodes(brain)
    design_nodes, design_edges = _design_nodes(brain)
    neurons, wires = _neuron_nodes(brain)
    mission_nodes, mission_edges = _mission_nodes()

    stats = brain.pilot.stats
    awake = sum(1 for node in neurons if node["kind"] == "neuron")
    total_hidden = sum(int(m.weight.shape[0])
                       for m in brain.pilot.net.trunk if hasattr(m, "weight"))

    core = {
        "id": "core", "kind": "core", "title": "Мозг KIA",
        "hint": "залетай внутрь: там настоящие нейроны",
        "detail": (f"{brain.pilot.net.parameter_count():,} весов · "
                   f"ступень {stage} · обновлений {stats.updates}"
                   ).replace(",", " "),
        "strength": 1.0, "awake": True, "pos": [0.0, 0.0, 0.0],
        "radius": CORE_RADIUS,
    }

    stages = dict(getattr(brain, "stages", None) or {})
    return {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "stage_title": next((s.title for s in STAGES if s.number == stage), ""),
        "core": core,
        "nodes": skills + mission_nodes + math_nodes + design_nodes + neurons,
        "edges": skill_edges + mission_edges + math_edges + design_edges + wires,
        "counts": {
            "skills": len(skills),
            "missions": len(mission_nodes),
            "math": len(math_nodes),
            "math_mastered": sum(1 for n in math_nodes if n["mastered"]),
            "design": len(design_nodes),
            "design_mastered": sum(1 for n in design_nodes if n["mastered"]),
            "awake": awake,
            "hidden": total_hidden,
            "inputs": sum(1 for n in neurons if n["kind"] == "input"),
            "outputs": sum(1 for n in neurons if n["kind"] == "output"),
        },
        "programs": {
            "math_stage": int(stages.get("math", 1)),
            "builder_stage": int(stages.get("builder", 1)),
        },
        "training": {
            "updates": stats.updates,
            "decisions": stats.total_steps,
            "episodes": stats.episodes,
            "mean_return": round(brain.pilot.mean_return(20), 2),
            "best_return": (round(stats.best_return, 2)
                            if stats.best_return > float("-inf") else None),
        },
        "geometry": {"core_radius": CORE_RADIUS,
                     "shells": list(SKILL_SHELL) + [MATH_SHELL, DESIGN_SHELL]},
    }


def write(brain, knowledge=None, path=None):
    """Сохраняет снимок атомарно — страница не поймает обрывок файла."""
    target = path or MAP3D_FILE
    try:
        payload = build(brain, knowledge=knowledge)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False),
                             encoding="utf-8")
        temporary.replace(target)
        return target
    except Exception as exc:              # карта не должна ронять ничего
        log.debug("Трёхмерный снимок не записан: %s", exc)
        return None


__all__ = ["build", "write", "MAP3D_FILE", "SKILL_PARENTS", "MATH_PARENTS",
           "DESIGN_SKILLS", "AWAKE_BIRTH_FACTOR", "CORE_RADIUS", "SKILL_SHELL",
           "MATH_SHELL", "DESIGN_SHELL"]
