"""Школа конструктора: учимся строить по чужим хорошим аппаратам.

Берём все чертежи, какие есть у игрока:
    • мастерская Steam      steamapps/workshop/content/220200/<id>/*.craft
    • стандартные           <KSP>/Ships/VAB
    • свои                  <KSP>/saves/*/Ships/VAB
и оставляем только те, где КАЖДАЯ деталь есть в установленной игре.

Каждый чертёж относится к одному из видов:
    rocket     ракета-носитель (стартует вертикально, TWR > 1)
    rover      ровер (колёса, своё питание)
    base       база на поверхности (опоры/колёса, жилые модули, без носителя)
    station    станция (стыковочные узлы, экипаж или модули, без носителя)
    probe      телескоп, спутник, зонд (наука, антенна, питание)
    plane      самолёт — отбрасывается (так просил игрок)

Из ракет выучиваем, КАК их строят люди:
    число ступеней по Δv, TWR на старте и у верхних ступеней, раскладку Δv
    по ступеням, отношения масс ступеней, какие двигатели на какой роли,
    сколько ускорителей и какую долю старта они несут, сужение диаметра
    вверх, обтекатели, носовые конусы на ускорителях, симметрию.
Из остальных видов — состав: сколько колёс, батарей, панелей, антенн,
узлов, фар, мест экипажа и приборов приходится на тонну аппарата.

Результат — data/design_school.json. Это знание и есть «веса» модуля
Builder: ASTRAUTOMA по нему выбирает число ступеней, TWR, раскладку Δv,
двигатели и оборудование, а для ровера, базы, станции и телескопа —
их состав.

    python tools/design_school.py [--astra <папка ASTRAUTOMA>] [--ksp <папка KSP>]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "design_school.json"
DEFAULT_ASTRA = Path.home() / "Desktop" / "ASTRAUTOMA"
DEFAULT_KSP = Path(r"D:\SteamLibrary\steamapps\common\Kerbal Space Program")

JET = ("IntakeAir",)
SKIP_NAMES = ("kia_autogen", "kia_neural", "автосохран")


# ==========================================================================
def sources(ksp: Path) -> list[tuple[str, Path]]:
    out = []
    ws = ksp.parent.parent / "workshop" / "content" / "220200"
    for f in sorted(ws.glob("*/*.craft")):
        out.append(("workshop", f))
    for f in sorted((ksp / "Ships" / "VAB").glob("*.craft")):
        out.append(("stock", f))
    for f in sorted((ksp / "Ships" / "SPH").glob("*.craft")):
        out.append(("stock", f))          # среди них есть роверы
    for f in sorted((ksp / "saves").glob("*/Ships/*/*.craft")):
        if any(s in f.name.lower() for s in SKIP_NAMES):
            continue
        out.append(("player", f))
    for f in sorted((ROOT / "data" / "craft_library").glob("*/*/*.craft")):
        out.append((f.parent.parent.name, f))        # скачанные: github, …
    return out


def modules_of(info) -> set[str]:
    return set(getattr(info, "modules", []) or [])


def features(vessel, lookup) -> dict:
    """Счётчики деталей по назначению."""
    c = Counter()
    sizes = []
    for p in vessel.parts:
        info = lookup(p.name)
        mods = modules_of(info) if info else set()
        n = p.name.lower()
        cat = (getattr(info, "category", "") or "").lower()
        if "ModuleWheelBase" in mods or "wheel" in n:
            if "gear" in n or "landing" in n:
                c["gear"] += 1
            else:
                c["wheel"] += 1
        if "ModuleDockingNode" in mods:
            c["docking"] += 1
        if "ModuleLiftingSurface" in mods or "ModuleControlSurface" in mods:
            if cat in ("aero",) and "fin" not in n and "nose" not in n:
                c["wing"] += 1
            else:
                c["fin"] += 1
        if "ModuleScienceExperiment" in mods:
            c["science"] += 1
        if "ModuleDeployableSolarPanel" in mods:
            c["solar"] += 1
        if "ModuleGenerator" in mods or "ModuleResourceConverter" in mods and "fuelCell" in n:
            c["generator"] += 1
        if "ModuleResourceHarvester" in mods or "ModuleResourceConverter" in mods and "isru" in n:
            c["isru"] += 1
        if "ModuleLight" in mods:
            c["light"] += 1
        if "ModuleDataTransmitter" in mods:
            c["antenna"] += 1
        if "ModuleReactionWheel" in mods:
            c["reaction_wheel"] += 1
        if "ModuleRCS" in mods or "ModuleRCSFX" in mods:
            c["rcs"] += 1
        if "ModuleProceduralFairing" in mods:
            c["fairing"] += 1
        if "nose" in n or "cone" in n:
            c["nose"] += 1
        if p.leg:
            c["leg"] += 1
        if p.parachute:
            c["parachute"] += 1
        if p.command:
            c["command"] += 1
        c["crew_capacity"] += p.crew_capacity
        if p.resources_max.get("ElectricCharge", 0) > 0 and not p.command:
            c["battery"] += 1
        if p.is_engine:
            props = getattr(p, "propellants", ())
            if any(j in props for j in JET):
                c["jet"] += 1
            elif p.solid:
                c["srb"] += 1
            else:
                c["liquid"] += 1
        if p.name.lower().startswith(("fuel", "rockomax", "size", "mk")) and p.resources_max:
            sizes.append(p.name)
        if cat in ("utility", "science") and p.crew_capacity >= 1 and not p.command:
            c["habitat"] += 1
    c["parts"] = len(vessel.parts)
    return dict(c)


def clean_name(name: str) -> str:
    """Стоковые чертежи подписаны меткой перевода: «#autoLOC_501226 //#autoLOC_501226 = Kerbal X»."""
    m = re.search(r"=\s*(.+)$", name)
    return m.group(1).strip() if name.startswith("#autoLOC") and m else name


def classify(path: Path, vessel, f: dict, stages) -> str:
    name = (clean_name(vessel.name) + " " + path.stem).lower()
    folder = path.parent.name.upper()
    twr = [s.thrust_asl / max(s.m0 * 9.81, 1e-6) for s in stages[:2]]
    liftoff_twr = max(twr, default=0.0)
    dv = sum(s.dv_vac for s in stages)
    engines = f.get("liquid", 0) + f.get("srb", 0)
    words = lambda *ws: any(w in name for w in ws)            # noqa: E731
    if f.get("jet", 0) >= 1 and f.get("wing", 0) >= 2:
        return "plane"
    if folder == "SPH" and f.get("wing", 0) >= 2 and f.get("wheel", 0) < 3:
        return "plane"
    if words("sr-71", "blackbird", "ssto", "spaceplane", "самолёт", "самолет", "aeris", "stratolaunch") \
            and f.get("wing", 0) >= 2:
        return "plane"
    if f.get("wheel", 0) >= 3 and dv < 2500:
        return "base" if (f.get("crew_capacity", 0) >= 4 or words("base", "база", "colony", "колон")) else "rover"
    if dv >= 2500 and liftoff_twr >= 1.0 and engines >= 1:
        return "rocket"
    if f.get("leg", 0) >= 3 and dv >= 800 and engines >= 1:
        return "lander"
    if words("base", "база", "colony", "колон", "outpost"):
        return "base"
    if words("telescope", "телескоп", "satellite", "спутник", "probe", "зонд", "mirror", "sputnik", "luna") \
            or (f.get("crew_capacity", 0) == 0 and (f.get("science", 0) or f.get("antenna", 0))):
        return "probe"
    if f.get("docking", 0) >= 1 or words("station", "станц") or f.get("crew_capacity", 0) >= 3:
        return "station"
    if f.get("leg", 0) >= 3:
        return "base"
    return "probe"


# ==========================================================================
def rocket_facts(vessel, stages, lookup, f: dict) -> dict | None:
    """Как устроена ракета: ступени снизу вверх."""
    st_ = [s for s in stages if s.dv_vac > 50]
    if not st_:
        return None
    total = sum(s.dv_vac for s in st_)
    g = 9.81
    twr0 = st_[0].thrust_asl / max(st_[0].m0 * g, 1e-6)
    uppers = [s.thrust_vac / max(s.m0 * g, 1e-6) for s in st_[1:]]
    # роль двигателей по ступени активации
    by_stage = defaultdict(list)
    for p in vessel.parts:
        if p.is_engine and p.activate_stage >= 0:
            by_stage[p.activate_stage].append(p)
    order = sorted(by_stage, reverse=True)          # первыми срабатывают старшие istg
    engines = []
    for i, s in enumerate(order):
        es = by_stage[s]
        kinds = Counter(e.name for e in es)
        engines.append({"role": "first" if i == 0 else ("second" if i == 1 else "upper"),
                        "engines": dict(kinds), "count": len(es),
                        "solid": sum(1 for e in es if e.solid)})
    srb_first = engines[0]["solid"] if engines else 0
    liquid_first = (engines[0]["count"] - srb_first) if engines else 0
    boosters = srb_first if liquid_first else 0
    if not boosters and engines and engines[0]["count"] >= 2 and len(engines) >= 2:
        # жидкие ускорители: те же двигатели, что и у ядра, отпадают раньше ядра
        boosters = 0
    ratios = [st_[i].m0 / max(st_[i + 1].m0, 1e-6) for i in range(len(st_) - 1)]
    return {"stages": len(st_), "dv_total": total, "dv_split": [s.dv_vac / total for s in st_],
            "twr0": twr0, "twr_upper": uppers, "mass_ratios": ratios,
            "payload_fraction": st_[-1].m1 / max(st_[0].m0, 1e-6),
            "mass": st_[0].m0, "boosters": boosters, "engines": engines,
            "fairing": f.get("fairing", 0) > 0, "nose": f.get("nose", 0), "fins": f.get("fin", 0),
            "parts": f.get("parts", 0)}


def q(values, p):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def band(values) -> dict | None:
    values = [x for x in values if x is not None and math.isfinite(x)]
    if not values:
        return None
    return {"p25": round(q(values, 0.25), 4), "median": round(q(values, 0.5), 4),
            "p75": round(q(values, 0.75), 4), "n": len(values)}


DV_CLASSES = (("orbit", 0, 5_200), ("moon", 5_200, 8_000), ("deep", 8_000, 1e9))


def learn_rockets(rows: list[dict]) -> dict:
    out = {"classes": {}}
    for name, lo, hi in DV_CLASSES:
        rs = [r for r in rows if lo <= r["dv_total"] < hi]
        if not rs:
            continue
        stages = Counter(r["stages"] for r in rs)
        n_st = stages.most_common(1)[0][0]
        split = []
        for i in range(n_st):
            split.append(band([r["dv_split"][i] for r in rs if r["stages"] == n_st]))
        out["classes"][name] = {
            "n": len(rs), "dv_range": [lo, hi], "stages": dict(stages), "typical_stages": n_st,
            "twr0": band([r["twr0"] for r in rs]),
            "twr_second": band([r["twr_upper"][0] for r in rs if r["twr_upper"]]),
            "twr_upper": band([r["twr_upper"][-1] for r in rs if len(r["twr_upper"]) >= 2]),
            "dv_split": split,
            "mass_ratio": band([x for r in rs for x in r["mass_ratios"]]),
            "payload_fraction": band([r["payload_fraction"] for r in rs]),
            "boosters_share": round(sum(1 for r in rs if r["boosters"]) / len(rs), 3),
            "boosters": dict(Counter(r["boosters"] for r in rs if r["boosters"])),
            "fairing_share": round(sum(1 for r in rs if r["fairing"]) / len(rs), 3),
            "fins_share": round(sum(1 for r in rs if r["fins"]) / len(rs), 3),
            "parts": band([r["parts"] for r in rs]),
        }
    # двигатели по ролям: что люди ставят на первую, вторую и верхние ступени
    roles = defaultdict(Counter)
    for r in rows:
        for e in r["engines"]:
            for name, k in e["engines"].items():
                roles[e["role"]][name] += k
    out["engines_by_role"] = {role: dict(c.most_common(15)) for role, c in roles.items()}
    # ступени против Δv: простая регрессия «сколько км/с несёт ступень»
    pts = [(r["dv_total"], r["stages"]) for r in rows]
    if len(pts) >= 3:
        out["dv_per_stage"] = round(st.median(dv / n for dv, n in pts), 1)
    return out


def learn_composition(rows: list[dict]) -> dict:
    """Состав аппарата: сколько чего на аппарат (медиана и разброс)."""
    keys = ("wheel", "leg", "battery", "solar", "generator", "antenna", "docking", "light",
            "reaction_wheel", "rcs", "science", "crew_capacity", "command", "habitat", "isru",
            "parachute", "parts")
    out = {"n": len(rows), "per_craft": {}}
    for k in keys:
        b = band([r["f"].get(k, 0) for r in rows])
        if b:
            out["per_craft"][k] = b
    out["mass"] = band([r["mass"] for r in rows])
    out["examples"] = [r["name"] for r in rows[:12]]
    return out


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--astra", type=Path, default=DEFAULT_ASTRA)
    ap.add_argument("--ksp", type=Path, default=DEFAULT_KSP)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    sys.path.insert(0, str(a.astra))
    from astra import craft as C                        # разбор .craft и каталог деталей игры

    t0 = time.time()
    seen = set()
    taken, skipped = [], Counter()
    for origin, path in sources(a.ksp):
        try:
            v = C.parse_craft(path)
        except Exception:
            skipped["unreadable"] += 1
            continue
        key = (v.name.strip().lower(), len(v.parts))
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        if v.unknown_parts:
            skipped["missing parts"] += 1
            continue
        if len(v.parts) < 3:
            skipped["too small"] += 1
            continue
        stages = v.stages()
        f = features(v, C._lookup)
        kind = classify(path, v, f, stages)
        if kind == "plane":
            skipped["plane"] += 1
            continue
        row = {"name": clean_name(v.name), "origin": origin, "kind": kind, "file": str(path), "mass": v.mass, "f": f}
        if kind == "rocket":
            facts = rocket_facts(v, stages, C._lookup, f)
            if facts is None:
                skipped["no stages"] += 1
                continue
            row.update(facts)
        taken.append(row)

    by_kind = defaultdict(list)
    for r in taken:
        by_kind[r["kind"]].append(r)
    school = {
        "format": "astra-design-school-1",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": {"used": len(taken), "skipped": dict(skipped),
                    "by_kind": {k: len(v) for k, v in by_kind.items()},
                    "by_origin": dict(Counter(r["origin"] for r in taken))},
        "rocket": learn_rockets(by_kind["rocket"]),
        "rover": learn_composition(by_kind["rover"]),
        "base": learn_composition(by_kind["base"]),
        "station": learn_composition(by_kind["station"]),
        "probe": learn_composition(by_kind["probe"]),
        "lander": learn_composition(by_kind["lander"]),
        "crafts": [{"name": r["name"], "kind": r["kind"], "origin": r["origin"],
                    "mass": round(r["mass"], 2), "parts": r["f"].get("parts", 0),
                    **({"stages": r["stages"], "dv": round(r["dv_total"]), "twr0": round(r["twr0"], 2),
                        "boosters": r["boosters"]} if r["kind"] == "rocket" else {})}
                   for r in taken],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(school, ensure_ascii=False, indent=1), encoding="utf-8")

    d = school["dataset"]
    print(f"Школа конструктора: {d['used']} чертежей за {time.time() - t0:.1f} с")
    print("  по видам:", d["by_kind"])
    print("  откуда:  ", d["by_origin"])
    print("  отброшено:", d["skipped"])
    for name, c in school["rocket"]["classes"].items():
        print(f"  ракеты «{name}»: {c['n']} шт., ступеней обычно {c['typical_stages']}, "
              f"TWR старта {c['twr0']['median']:.2f}, 2-й ступени "
              f"{(c['twr_second'] or {}).get('median', 0):.2f}, ускорители у {c['boosters_share'] * 100:.0f} %")
    print("  сохранено:", a.out)


if __name__ == "__main__":
    main()
