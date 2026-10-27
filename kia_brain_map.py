"""Живая карта мозга KIA в браузере.

    python kia_brain_map.py          поднять карту и открыть браузер
    python kia_brain_map.py --port 8765 --no-browser

Поднимает крошечный локальный сервер и открывает `kia_brain_map.html`.
Страница каждые две секунды перечитывает `data/brain_map.json`, который
тренер обновляет по ходу обучения, — поэтому карта живая: кружки меняют
цвет и размер прямо во время счёта.

ПОЧЕМУ СЕРВЕР, А НЕ ПРОСТО ФАЙЛ. Открытый двойным кликом html лежит на
`file://`, а оттуда браузер запрещает читать соседние файлы (политика
источников). Локальный сервер снимает вопрос и не требует ни интернета,
ни установки чего-либо: всё уже есть в стандартной библиотеке python.

Сервер слушает только петлевой адрес 127.0.0.1 — снаружи к нему не
подключиться.
"""
from __future__ import annotations

import argparse
import http.server
import socket
import socketserver
import sys
import threading
import webbrowser
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAGE = "kia_brain_map.html"
DEFAULT_PORT = 8731


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Тот же файловый сервер, но без болтовни в консоль."""

    def log_message(self, fmt, *args):        # noqa: A003 - подпись из базы
        pass

    def end_headers(self):
        # Карта обязана быть свежей: браузер не должен кэшировать снимок
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def free_port(preferred: int) -> int:
    """Свободный порт: предпочтительный, иначе любой соседний."""
    for candidate in range(preferred, preferred + 20):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    return 0                                   # пусть система выберет сама


def serve(port: int = DEFAULT_PORT, open_browser: bool = True,
          block: bool = True):
    """Поднимает карту. Возвращает (сервер, адрес)."""
    if not (ROOT / PAGE).exists():
        raise FileNotFoundError(f"нет файла карты: {ROOT / PAGE}")

    port = free_port(port)
    handler = partial(QuietHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    port = server.server_address[1]
    address = f"http://127.0.0.1:{port}/{PAGE}"

    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name="kia-brain-map")
    thread.start()
    if open_browser:
        webbrowser.open(address)

    if block:
        print("=" * 68)
        print("  КАРТА МОЗГА KIA")
        print("=" * 68)
        print(f"  Адрес: {address}")
        print("  Обновляется каждые 2 секунды, пока идёт обучение.")
        print("  Закрыть — Ctrl+C (на обучение это никак не влияет).")
        print()
        snapshot = ROOT / "data" / "brain_map.json"
        if not snapshot.exists():
            print("  Снимка мозга ещё нет: запустите обучение (пульт -> 0),")
            print("  карта наполнится сама через несколько секунд.")
            print()
        try:
            thread.join()
        except KeyboardInterrupt:
            print("\n  Карта закрыта.")
            server.shutdown()
    return server, address


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kia_brain_map", description="Живая карта мозга KIA")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true",
                        help="не открывать браузер автоматически")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        serve(port=args.port, open_browser=not args.no_browser)
    except OSError as exc:
        print(f"Карта не поднялась: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
