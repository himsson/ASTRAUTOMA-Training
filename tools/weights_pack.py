"""Пакеты весов ИИ: как делиться обученным ИИ через GitHub и не только.

ФОРМАТ .kiaw — обычный zip, внутри:

    manifest.json          что это за веса: имя, автор, дата, чему обучены
    brain.pt               пилот + конструктор + математик (PyTorch, только тензоры)
    navigator.pt           навигатор межпланетной программы (если обучен)
    genome.json            профиль полёта: параметры разворота, тяги, манёвров
    knowledge/*.json       знания тренажёров, навыки, математика

КАК ПОЛЬЗОВАТЬСЯ
    Скачал чужой .kiaw → положил в папку ASTRAUTOMA\\weights → в меню
    «Веса ИИ» выбрал его → Enter. Всё: приложение летает на этих весах.
    Прежние веса перед заменой сохраняются в weights\\_previous.

БЕЗОПАСНОСТЬ
    Файлы PyTorch (.pt/.pth) умеют хранить исполняемый код. Поэтому при
    установке каждый .pt грузится ТОЛЬКО в режиме weights_only=True —
    допускаются тензоры и числа, код не выполнится. Имена файлов внутри
    архива проверяются по белому списку: пакет не может записать ничего
    за пределы своих мест.
"""
from __future__ import annotations

import io
import json
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

FORMAT = "kia-weights-1"
MAX_SIZE = 200 * 1024 * 1024          # пакет больше 200 МБ — подозрительно

# Куда что ставится (относительно корня приложения)
INSTALL_MAP = {
    "brain.pt": ["kia_model.pth", "kia_model_best.pth"],
    "navigator.pt": ["kia_model_navigator.pth"],
    "genome.json": ["data/best_genome.json"],
    "knowledge/kia_global_memory.json": ["kia_global_memory.json"],
    "knowledge/kia_math_skills.json": ["kia_math_skills.json"],
    "knowledge/kia_sim_knowledge.json": ["kia_sim_knowledge.json"],
}
ALLOWED = set(INSTALL_MAP) | {"manifest.json", "README.md"}


@dataclass
class PackInfo:
    path: Path
    manifest: dict
    files: list[str]
    error: str = ""

    @property
    def name(self) -> str:
        return self.manifest.get("name") or self.path.stem

    def describe(self) -> list[str]:
        m = self.manifest
        lines = [f"{m.get('name', self.path.stem)}  ·  автор {m.get('author', '—')}  ·  "
                 f"{m.get('created', '—')}"]
        if m.get("description"):
            lines.append(m["description"])
        t = m.get("trained", {})
        if t.get("pilot_score") is not None:
            lines.append(f"пилот: лучший счёт {t['pilot_score']:.0f}, вехи: "
                         + ", ".join(t.get("milestones", [])[:6]))
        if t.get("navigator"):
            nav = t["navigator"]
            lines.append(f"навигатор: ступень {nav.get('stage')}"
                         + (" (программа пройдена)" if nav.get("finished") else ""))
        lines.append("состав: " + ", ".join(self.files))
        return lines


# ==========================================================================
def _check_torch(data: bytes, name: str) -> None:
    import torch
    try:
        torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{name}: не прошёл безопасную загрузку ({exc})") from exc


def inspect(path: Path) -> PackInfo:
    """Читает описание пакета, ничего не устанавливая."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_SIZE:
            return PackInfo(path, {}, [], "пакет больше 200 МБ")
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            bad = [n for n in names if n not in ALLOWED]
            if bad:
                return PackInfo(path, {}, names, f"посторонние файлы: {', '.join(bad[:3])}")
            manifest = json.loads(z.read("manifest.json").decode("utf-8"))
            if manifest.get("format") != FORMAT:
                return PackInfo(path, manifest, names, f"неизвестный формат {manifest.get('format')}")
            return PackInfo(path, manifest, [n for n in names if n != "manifest.json"])
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        return PackInfo(path, {}, [], f"не читается: {exc}")


def install(path: Path, root: Path) -> PackInfo:
    """Проверяет и ставит пакет. Прежние файлы — в weights/_previous."""
    info = inspect(path)
    if info.error:
        raise ValueError(info.error)
    root = Path(root)
    with zipfile.ZipFile(path) as z:
        payload = {name: z.read(name) for name in info.files if name in INSTALL_MAP}
    for name, data in payload.items():
        if name.endswith(".pt"):
            _check_torch(data, name)
        else:
            json.loads(data.decode("utf-8"))
    backup = root / "weights" / "_previous"
    backup.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        for target in INSTALL_MAP[name]:
            dest = root / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.copy2(dest, backup / dest.name)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
    (root / "weights" / "active.json").write_text(
        json.dumps({"pack": path.name, "name": info.name,
                    "installed": time.strftime("%Y-%m-%d %H:%M:%S")},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def active(root: Path) -> dict:
    try:
        return json.loads((Path(root) / "weights" / "active.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def packs(root: Path) -> list[PackInfo]:
    folder = Path(root) / "weights"
    folder.mkdir(exist_ok=True)
    return [inspect(p) for p in sorted(folder.glob("*.kiaw"))]


# ==========================================================================
def export(root: Path, out: Path, name: str, author: str = "", description: str = "",
           prefer_best: bool = True) -> Path:
    """Собирает пакет из папки с обученным ИИ (обучающей или ASTRAUTOMA)."""
    root = Path(root)
    sources = {
        "brain.pt": [root / "kia_model_best.pth", root / "kia_model.pth"] if prefer_best
        else [root / "kia_model.pth"],
        "navigator.pt": [root / "kia_model_navigator_best.pth", root / "kia_model_navigator.pth"],
        "genome.json": [root / "data" / "best_genome.json"],
        "knowledge/kia_global_memory.json": [root / "kia_global_memory.json"],
        "knowledge/kia_math_skills.json": [root / "kia_math_skills.json"],
        "knowledge/kia_sim_knowledge.json": [root / "kia_sim_knowledge.json"],
    }
    chosen = {}
    for arc, candidates in sources.items():
        for src in candidates:
            if src.exists():
                data = src.read_bytes()
                if arc.endswith(".pt"):
                    _check_torch(data, src.name)
                chosen[arc] = data
                break
    if "brain.pt" not in chosen:
        raise FileNotFoundError(f"В {root} нет весов kia_model*.pth")

    trained = {}
    try:
        genome = json.loads(chosen.get("genome.json", b"{}").decode("utf-8"))
        trained["pilot_score"] = genome.get("score")
        trained["milestones"] = genome.get("milestones", [])
    except json.JSONDecodeError:
        pass
    report = root / "data" / "navigator_report.json"
    if report.exists():
        try:
            trained["navigator"] = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    manifest = {"format": FORMAT, "name": name, "author": author,
                "description": description, "created": time.strftime("%Y-%m-%d %H:%M"),
                "game": "Kerbal Space Program 1.12", "trained": trained,
                "files": sorted(chosen)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for arc, data in chosen.items():
            z.writestr(arc, data)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Пакеты весов ИИ (.kiaw)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="собрать пакет из папки с обученным ИИ")
    e.add_argument("root")
    e.add_argument("out")
    e.add_argument("--name", required=True)
    e.add_argument("--author", default="")
    e.add_argument("--description", default="")
    i = sub.add_parser("info", help="показать, что внутри пакета")
    i.add_argument("pack")
    s = sub.add_parser("install", help="поставить пакет в папку приложения")
    s.add_argument("pack")
    s.add_argument("root")
    args = ap.parse_args()
    if args.cmd == "export":
        print(export(Path(args.root), Path(args.out), args.name, args.author, args.description))
    elif args.cmd == "info":
        info = inspect(Path(args.pack))
        print("\n".join(info.describe()) if not info.error else f"ОШИБКА: {info.error}")
    else:
        print("\n".join(install(Path(args.pack), Path(args.root)).describe()))
