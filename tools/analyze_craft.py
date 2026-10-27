"""Разбор готового чертежа: что в нём есть и почему он летает.

Зачем это нужно. KIA до сих пор училась только на СВОИХ ракетах, а они
все происходят от одного расчётного шаблона: ядро, баки, двигатель,
оперение. Когда такой аппарат оказывается неуправляемым, система может
крутить только те ручки, которые у неё есть, и заметить отсутствующую
деталь она не в состоянии — ей неоткуда узнать, что бывает иначе.

Готовый чертёж, который заведомо летает, снимает это ограничение. Здесь
он раскладывается на инженерные величины, сопоставимые с тем, что считает
собственный конструктор:

    масса, ступени, тяговооружённость по ступеням;
    ЧЕМ аппарат управляется — маховики, RCS, качание сопла, рули;
    аэродинамика — обтекатели, оперение, мидель;
    состав по назначению деталей.

Запуск:
    python tools/analyze_craft.py                     все чертежи сохранения
    python tools/analyze_craft.py "Наполлон"          один
    python tools/analyze_craft.py --compare           рядом с тем, что строит KIA
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kia_core.config import CONFIG                                  # noqa: E402
from kia_core.engineer.parts_catalog import PartInfo, get_catalog   # noqa: E402
from kia_core.engineer.parts_db import get_body                     # noqa: E402

KERBIN_G = 9.81


# ==========================================================================
@dataclass
class PlacedPart:
    """Одна деталь чертежа: что это и где стоит."""
    part_id: str
    name: str
    info: PartInfo | None
    pos: tuple[float, float, float]
    istg: int
    dstg: int
    surface: bool

    @property
    def title(self) -> str:
        return self.info.title if self.info else self.name


@dataclass
class CraftReport:
    """Инженерная сводка по чертежу."""
    name: str
    parts: list[PlacedPart] = field(default_factory=list)
    dry_mass: float = 0.0
    wet_mass: float = 0.0

    # управление
    reaction_wheels: int = 0
    rcs_blocks: int = 0
    gimbal_engines: int = 0
    max_gimbal: float = 0.0
    control_surfaces: int = 0
    fins: int = 0

    # аэродинамика и компоновка
    nose_cones: int = 0
    fairings: int = 0
    max_diameter: float = 0.0
    length: float = 0.0

    # двигатели
    engines: list[PartInfo] = field(default_factory=list)
    boosters: list[PartInfo] = field(default_factory=list)
    stages: int = 0
    decouplers: int = 0
    radial_decouplers: int = 0

    # прочее
    crewed: bool = False
    parachutes: int = 0
    solar: int = 0
    batteries: int = 0
    antennas: int = 0
    science: int = 0
    ladders: int = 0
    legs: int = 0
    wheels: int = 0
    expansion_parts: int = 0

    @property
    def liftoff_thrust(self) -> float:
        """Тяга у земли всего, что зажигается на старте (кН)."""
        return sum(e.thrust_at(1.0) for e in self.engines_at_launch)

    engines_at_launch: list[PartInfo] = field(default_factory=list)

    @property
    def liftoff_twr(self) -> float:
        if self.wet_mass <= 0:
            return 0.0
        return self.liftoff_thrust * 1000.0 / (self.wet_mass * 1000.0 * KERBIN_G)

    @property
    def slenderness(self) -> float:
        """Отношение длины к максимальному диаметру."""
        return self.length / self.max_diameter if self.max_diameter else 0.0


# ==========================================================================
def parse_craft(path: Path, catalog) -> CraftReport:
    text = path.read_text(encoding="utf-8", errors="ignore")
    name = re.search(r"^\s*ship\s*=\s*(.+)$", text, re.M)
    report = CraftReport(name=(name.group(1).strip() if name else path.stem))

    for block in re.findall(r"PART\s*\{(.*?)\n\}", text, re.S):
        pid = re.search(r"^\s*part\s*=\s*(.+)$", block, re.M)
        pos = re.search(r"^\s*pos\s*=\s*(.+)$", block, re.M)
        if not (pid and pos):
            continue
        raw = pid.group(1).strip()
        part_name = raw.rsplit("_", 1)[0]
        info = catalog.get(part_name.replace(".", "_")) or catalog.get(part_name)
        istg = re.search(r"^\s*istg\s*=\s*(-?\d+)", block, re.M)
        dstg = re.search(r"^\s*dstg\s*=\s*(-?\d+)", block, re.M)
        try:
            xyz = tuple(float(v) for v in pos.group(1).split(","))
        except ValueError:
            continue
        report.parts.append(PlacedPart(
            part_id=raw, name=part_name, info=info, pos=xyz,
            istg=int(istg.group(1)) if istg else -1,
            dstg=int(dstg.group(1)) if dstg else 0,
            surface="srfN" in block))

    densities = catalog.densities
    heights = [p.pos[1] for p in report.parts]
    if heights:
        report.length = max(heights) - min(heights)
    launch_stage = max((p.istg for p in report.parts), default=0)

    for p in report.parts:
        info = p.info
        if info is None:
            continue
        report.dry_mass += info.dry_mass
        report.wet_mass += info.wet_mass(densities)
        report.max_diameter = max(report.max_diameter, info.diameter)
        if "SquadExpansion" in (info.cfg_path or ""):
            report.expansion_parts += 1

        mods = info.modules
        if "ModuleReactionWheel" in mods:
            report.reaction_wheels += 1
        if "ModuleRCSFX" in mods or "ModuleRCS" in mods:
            report.rcs_blocks += 1
        if "ModuleControlSurface" in mods:
            report.control_surfaces += 1
        elif "ModuleLiftingSurface" in mods and info.category == "Aero":
            report.fins += 1
        if info.gimbal > 0:
            report.gimbal_engines += 1
            report.max_gimbal = max(report.max_gimbal, info.gimbal)
        if info.is_engine:
            (report.boosters if info.is_solid_booster
             else report.engines).append(info)
            if p.istg == launch_stage:
                report.engines_at_launch.append(info)
        if info.is_decoupler:
            if len(info.stack_nodes) >= 2:
                report.decouplers += 1
            else:
                report.radial_decouplers += 1
        if info.is_parachute:
            report.parachutes += 1
        if info.is_solar:
            report.solar += 1
        if info.is_battery:
            report.batteries += 1
        if info.is_antenna:
            report.antennas += 1
        if info.is_science:
            report.science += 1
        if info.crew_capacity > 0:
            report.crewed = True
        lowered = f"{info.name} {info.title}".lower()
        if "fairing" in lowered or "обтекат" in lowered:
            report.fairings += 1
        elif (info.bottom_node and not info.top_node
              and info.category in ("Aero", "Structural", "Payload")
              and ("cone" in lowered or "конус" in lowered)):
            report.nose_cones += 1
        if "ladder" in lowered:
            report.ladders += 1
        if "ModuleWheelBase" in mods:
            (report.wheels if "wheel" in lowered else report.legs).__iadd__ \
                if False else None
            if "wheel" in lowered or "колес" in lowered:
                report.wheels += 1
            else:
                report.legs += 1

    report.stages = len({p.istg for p in report.parts if p.istg >= 0})
    return report


# ==========================================================================
def describe(report: CraftReport) -> str:
    r = report
    lines = [f"═══ {r.name} ═══"]
    lines.append(f"  деталей {len(r.parts)}, масса {r.wet_mass:.1f} т "
                 f"(сухая {r.dry_mass:.1f} т), длина {r.length:.1f} м, "
                 f"мидель {r.max_diameter:.2f} м, вытянутость {r.slenderness:.1f}")
    lines.append(f"  стадий {r.stages}, разделителей {r.decouplers} стековых "
                 f"+ {r.radial_decouplers} радиальных")

    if r.engines or r.boosters:
        thrust = r.liftoff_thrust
        lines.append(f"  ТЯГА: на старте {thrust:.0f} кН -> "
                     f"стартовый TWR {r.liftoff_twr:.2f}")
        seen = {}
        for e in r.engines + r.boosters:
            seen[e.title] = seen.get(e.title, 0) + 1
        for title, count in sorted(seen.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {count} x {title}")

    control = []
    if r.reaction_wheels:
        control.append(f"маховиков {r.reaction_wheels}")
    if r.rcs_blocks:
        control.append(f"блоков RCS {r.rcs_blocks}")
    if r.gimbal_engines:
        control.append(f"качающихся сопел {r.gimbal_engines} (до {r.max_gimbal:.0f}°)")
    if r.control_surfaces:
        control.append(f"управляемых рулей {r.control_surfaces}")
    if r.fins:
        control.append(f"стабилизаторов {r.fins}")
    lines.append("  УПРАВЛЕНИЕ: " + (", ".join(control) or "НЕТ"))

    aero = []
    if r.fairings:
        aero.append(f"обтекателей {r.fairings}")
    if r.nose_cones:
        aero.append(f"носовых конусов {r.nose_cones}")
    lines.append("  АЭРОДИНАМИКА: " + (", ".join(aero) or "ничего"))

    extra = []
    for label, value in (("парашютов", r.parachutes), ("панелей", r.solar),
                         ("батарей", r.batteries), ("антенн", r.antennas),
                         ("приборов", r.science), ("опор", r.legs),
                         ("колёс", r.wheels), ("лестниц", r.ladders)):
        if value:
            extra.append(f"{label} {value}")
    if r.crewed:
        extra.append("С ЭКИПАЖЕМ")
    if r.expansion_parts:
        extra.append(f"из дополнения {r.expansion_parts}")
    if extra:
        lines.append("  ПРОЧЕЕ: " + ", ".join(extra))
    return "\n".join(lines)


# ==========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Разбор чертежей KSP")
    parser.add_argument("pattern", nargs="?", default="",
                        help="часть имени чертежа")
    parser.add_argument("--dir", default="", help="папка с чертежами")
    parser.add_argument("--learn", action="store_true",
                        help="снять нормы с образцов и сохранить их")
    args = parser.parse_args(argv)

    catalog = get_catalog()
    folder = Path(args.dir) if args.dir else (
        Path(CONFIG.craft.save_directory) / "Ships" / "VAB"
        if getattr(CONFIG.craft, "save_directory", "") else None)
    if folder is None or not folder.exists():
        print(f"Папка с чертежами не найдена: {folder}")
        return 2

    files = sorted(folder.glob("*.craft"))
    if args.pattern:
        files = [f for f in files if args.pattern.lower() in f.stem.lower()]
    if not files:
        print("Чертежей не найдено")
        return 1

    for path in files:
        try:
            print(describe(parse_craft(path, catalog)))
            print()
        except Exception as exc:
            print(f"═══ {path.stem} ═══\n  не разобран: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
