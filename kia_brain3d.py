"""Трёхмерная живая карта мозга KIA.

    BRAIN3D.bat                      запустить и открыть браузер
    python kia_brain3d.py --port 8740 --no-browser

Поднимает локальный сервер и открывает `kia_brain3d.html`. Страница
показывает мозг как объём: снаружи кольца НАВЫКОВ, которые появляются по
мере обучения, внутри ядра — настоящие НЕЙРОНЫ сети со связями.
Колесом мыши влетаешь внутрь, наведением смотришь, что это за узел.

ЧЕМ ОТЛИЧАЕТСЯ ОТ BRAIN_MAP.bat. Плоская карта — рабочий прибор: по ней
ловят упор по осям и залипший разброс действий, и она осталась на месте.
Эта — про устройство и рост: что у сети уже есть и что появилось.

ПОЧЕМУ СЕРВЕР САМ ЧИТАЕТ ВЕСА, А НЕ ЖДЁТ ТРЕНЕРА. Так карта работает и
без обучения: открыл — увидел текущее состояние мозга. Тренер при этом
не трогается вовсе, то есть карта физически не может ни замедлить счёт,
ни его уронить. Файл весов перечитывается только когда он изменился.

Сервер слушает 127.0.0.1 — снаружи к нему не подключиться, интернет не
нужен, ничего не устанавливается.
"""
from __future__ import annotations

import argparse
import http.server
import json
import socket
import socketserver
import sys
import threading
import time
import webbrowser
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PAGE = "kia_brain3d.html"
DEFAULT_PORT = 8740
REFRESH_SECONDS = 2.0


class SnapshotMaker:
    """Держит мозг в памяти и обновляет снимок, когда веса изменились."""

    def __init__(self):
        self.brain = None
        self.knowledge = None
        self.stamp = 0.0
        self.error = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def prepare(self) -> bool:
        try:
            from kia_core.neural.knowledge import SIM_KNOWLEDGE
            from kia_core.neural.ppo import build_brain
        except Exception as exc:
            self.error = f"нейромодуль недоступен: {exc}"
            return False
        try:
            self.brain = build_brain(load=True)
            self.knowledge = SIM_KNOWLEDGE
        except Exception as exc:
            self.error = f"мозг не собрался: {exc}"
            return False
        return True

    # ------------------------------------------------------------------
    def weights_stamp(self) -> float:
        from kia_core.neural.ppo import MODEL_FILE
        try:
            return Path(MODEL_FILE).stat().st_mtime
        except OSError:
            return 0.0

    def refresh(self, force: bool = False) -> bool:
        """Перечитывает веса и переписывает снимок, если файл изменился."""
        from kia_core.neural import brain3d
        from kia_core.neural.ppo import MODEL_FILE

        with self._lock:
            stamp = self.weights_stamp()
            if not force and stamp <= self.stamp:
                return False
            self.stamp = stamp
            try:
                if stamp:
                    self.brain.load(MODEL_FILE)
                if self.knowledge is not None:
                    self.knowledge.load()
                brain3d.write(self.brain, knowledge=self.knowledge)
                return True
            except Exception as exc:
                self.error = f"снимок не собрался: {exc}"
                return False

    def watch(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception:            # карта не имеет права падать
                pass
            time.sleep(REFRESH_SECONDS)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Файловый сервер без болтовни и без кэша: карта обязана быть свежей."""

    def log_message(self, fmt, *args):        # noqa: A003 — подпись из базы
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def free_port(preferred: int) -> int:
    for candidate in range(preferred, preferred + 20):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    return 0


def serve(port: int = DEFAULT_PORT, open_browser: bool = True,
          block: bool = True):
    if not (ROOT / PAGE).exists():
        raise FileNotFoundError(f"нет файла карты: {ROOT / PAGE}")

    maker = SnapshotMaker()
    ready = maker.prepare()
    if ready:
        maker.refresh(force=True)
        threading.Thread(target=maker.watch, daemon=True,
                         name="kia-brain3d-watch").start()

    port = free_port(port)
    handler = partial(QuietHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    port = server.server_address[1]
    address = f"http://127.0.0.1:{port}/{PAGE}"

    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name="kia-brain3d")
    thread.start()
    if open_browser:
        webbrowser.open(address)

    if block:
        print("=" * 68)
        print("  ТРЁХМЕРНАЯ КАРТА МОЗГА KIA")
        print("=" * 68)
        print(f"  Адрес: {address}")
        print("  Колесо мыши — влететь внутрь и обратно.")
        print("  Мышь с зажатой кнопкой — повернуть, правая кнопка — сдвинуть.")
        print("  Наведение на узел — что это такое.")
        print(f"  Обновляется каждые {REFRESH_SECONDS:.0f} с, пока идёт обучение.")
        print("  Закрыть — Ctrl+C (на обучение это никак не влияет).")
        if not ready:
            print()
            print(f"  ВНИМАНИЕ: {maker.error}")
            print("  Карта поднимется, но останется пустой.")
        print()
        try:
            thread.join()
        except KeyboardInterrupt:
            print("\n  Карта закрыта.")
            server.shutdown()
    return server, address


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kia_brain3d", description="Трёхмерная карта мозга KIA")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true",
                        help="не открывать браузер автоматически")
    parser.add_argument("--dump", action="store_true",
                        help="только собрать снимок и выйти")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.dump:
        maker = SnapshotMaker()
        if not maker.prepare():
            print(maker.error)
            return 1
        maker.refresh(force=True)
        from kia_core.neural.brain3d import MAP3D_FILE
        data = json.loads(Path(MAP3D_FILE).read_text(encoding="utf-8"))
        print(f"снимок: {MAP3D_FILE}")
        print(f"узлов {len(data['nodes'])}, связей {len(data['edges'])}, "
              f"счётчики {data['counts']}")
        return 0

    try:
        serve(port=args.port, open_browser=not args.no_browser)
    except OSError as exc:
        print(f"Карта не поднялась: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
