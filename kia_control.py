"""Пульт KIA. Запуск: LAUNCH_KIA.bat или python kia_control.py"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kia_core.agent.autonomy import AutonomousLoop                      # noqa: E402
from kia_core.agent.dispatcher import Dispatcher                        # noqa: E402
from kia_core.config import CONFIG, DATA_DIR, ROOT                      # noqa: E402
from kia_core.engineer.autodesign import DesignPolicy                   # noqa: E402
from kia_core.learning.reward_book import REWARD_BOOK, RULE_CATALOG     # noqa: E402
from kia_core.logging_setup import setup_logging                        # noqa: E402
from kia_core.memory import GLOBAL_MEMORY                               # noqa: E402
from kia_core.mission_spec import MissionSpec, parse_command            # noqa: E402
from kia_core.run_workspace import WORKSPACE                            # noqa: E402

BANNER = "kia\n"

MENU = """
  0  эволюция        вся программа в симуляторах, без игры
  1  kRPC
  2  автообучение    бесконечный цикл с полётами
  3  настройки
  4  логи
  5  сброс
  6  память
  7  нейросеть
  8  манифест
  9  выход
"""

NEURAL_MENU = """
  пилот         1 базовый   2 плюс     3 про      4 что это
  конструктор  17 ступени                        20 что это
  математика   18 ступени  19 экзамен
  навигатор    21 ступени  22 экзамен

   5 мозг          11 лучшие веса
   6 прогон        12 откат к ступени
   7 живая KSP     13 стереть веса
   8 конструктор   14 знания тренажёров
   9 палитра       15 карта мозга
  10 стоп          16 CPU+GPU
   0 назад
"""

GOALS = {
    "1": ("Орбита Кербина 80 км", "выйди на орбиту 80 км"),
    "2": ("Орбита Кербина 120 км", "выйди на орбиту 120 км"),
    "3": ("Полёт к Муне (орбита 30 км)", "лети на Муну орбита 30 км"),
    "4": ("Посадка на Муну", "сядь на Мун"),
    "5": ("Посадка на Муну с возвращением", "сядь на Мун и вернись"),
    "6": ("Облёт Минмуса", "облёт Минмуса"),
}


# ==========================================================================
def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def pause() -> None:
    ask("\n  enter ")


_EOF_COUNT = 0
_EOF_LIMIT = 5


def ask(prompt: str, default: str = "") -> str:
    """Ввод с защитой от «мёртвого» stdin.

    Если пульт запущен с закрытым или пустым вводом (конвейер, служба),
    input() мгновенно возвращает EOF. Без счётчика меню в этом случае
    крутится вхолостую и жжёт процессор — такое уже случалось.
    """
    global _EOF_COUNT
    try:
        # BOM и невидимые символы прилетают из файлов и буфера обмена;
        # без очистки «7» не совпадает с пунктом меню
        answer = input(prompt).replace("﻿", "").replace("​", "").strip()
    except (EOFError, KeyboardInterrupt):
        _EOF_COUNT += 1
        if _EOF_COUNT >= _EOF_LIMIT:
            print("\n  Ввод недоступен — пульт завершает работу.")
            raise SystemExit(0)
        return default
    _EOF_COUNT = 0
    return answer or default


def enter_pressed(timeout: float = 0.0) -> bool:
    """Нажат ли Enter, без блокировки и без отдельного потока-читателя.

    Отдельный поток с input() отменить невозможно: он остаётся висеть на
    stdin и перехватывает следующий ввод оператора. Поэтому клавиатура
    опрашивается напрямую.
    """
    end = time.time() + max(0.0, timeout)
    try:
        import msvcrt                      # Windows
        while True:
            while msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\r", "\n"):
                    return True
            if time.time() >= end:
                return False
            time.sleep(0.05)
    except ImportError:
        import select                      # POSIX
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            sys.stdin.readline()
            return True
        return False


# ==========================================================================
class ControlPanel:
    """Текстовый пульт: связывает оператора с ядром KIA."""

    def __init__(self):
        self.log = setup_logging()
        self.spec = MissionSpec.load()
        self.dispatcher = Dispatcher(spec=self.spec, resume=True)
        self.loop: AutonomousLoop | None = None
        self.worker: threading.Thread | None = None
        self._trainer = None
        self._pipeline = None

    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def spawn(self, name: str, fn) -> str:
        """Запускает длительное задание в фоне, чтобы меню не блокировалось.

        Исключение внутри задания не должно ронять пульт — оно уходит
        в лог, а оператор видит сообщение и может посмотреть подробности
        в logs/.
        """
        if self.busy:
            return ("  Уже выполняется задание. Сначала остановите его "
                    "(пункт 7 подменю или пункт 5 главного меню).")

        def wrapper():
            try:
                fn()
            except Exception as exc:
                self.log.error("Задание «%s» упало: %s\n%s", name, exc,
                               traceback.format_exc())
                print(f"\n  ошибка: Задание «{name}» прервано: {exc}")

        self.worker = threading.Thread(target=wrapper, name=name, daemon=True)
        self.worker.start()
        return f"  пошло: «{name}» работает в фоне."

    def header(self) -> str:
        memory = GLOBAL_MEMORY
        skills = len(memory.skills)
        runs = len(WORKSPACE.existing_numbers())
        state = "обучение идёт" if self.busy else "простой"
        if self._pipeline is not None and self.busy and self._pipeline.running:
            info = self._pipeline.status()
            state = (f"эволюция, ступень {info['stage']}, орбит "
                     f"{info['orbits']}/{self._pipeline.config.promote_orbits}, "
                     f"средняя {info['mean_return']:+.1f}")
        return "\n".join([
            f"  цель    {self.spec.title} — "
            f"{self.spec.describe().replace('цель: ', '', 1)}",
            f"  статус  {state}",
            f"  kia/    {runs} запусков, {skills} навыков, "
            f"{len(REWARD_BOOK.milestones)}/{len(REWARD_BOOK.failures)}"
            f"/{len(REWARD_BOOK.rules)} наград/штрафов/правил",
        ])

    # ==================================================================
    # 0. Полная автономная эволюция (только тренажёры)
    # ==================================================================
    def action_evolution(self) -> None:
        """Конвейер: вся программа обучения без человека и без игры."""
        clear()
        from kia_core.neural import is_available
        print("  эволюция")
        print()
        if not is_available():
            print("  PyTorch не установлен — конвейеру не на чем считать.")
            print("    pip install --force-reinstall torch "
                  "--index-url https://download.pytorch.org/whl/cu126")
            pause()
            return

        from kia_core.neural.pipeline import (MANIFEST_FILE, PipelineConfig,
                                              complete_file)
        from kia_core.neural.curriculum import STAGES

        config = PipelineConfig()
        print("  ступени:  "
              + ", ".join(f"{s.number} {s.title}" for s in STAGES))
        print(f"  экзамен:  средняя > {config.promote_reward:+.0f} за "
              f"{config.window} эпизодов и {config.promote_orbits} орбит")
        print(f"  тупик:    средняя < {config.collapse_reward:+.0f} и без роста "
              f"{config.collapse_patience} эпизодов -> откат, lr вдвое")
        print(f"  хроника:  {MANIFEST_FILE.name}")
        print("  игра не нужна, всё считается в тренажёрах")
        print()
        print("  архивы:   " + ", ".join(
            f"{s.number} {'есть' if complete_file(s.number).exists() else 'нет'}"
            for s in STAGES))
        print()

        if self.busy:
            print("  Уже выполняется задание. Сначала остановите его")
            print("  (пункт 7 -> 10) — два обучения затрут друг другу веса.")
            pause()
            return

        from kia_core.neural.pipeline import EvolutionPipeline
        trainer = self._ensure_trainer()
        recorded = max(1, min(len(STAGES), int(trainer.brain.stage or 1)))
        print(f"  Веса числятся на ступени {recorded}.")
        print("  Начать можно с любой: веса при этом не подменяются, меняется")
        print("  только «класс», в котором они учатся. Спуститься вниз имеет")
        print("  смысл, если политика разучилась летать — на первой ступени")
        print("  это восстанавливается быстрее всего.")
        raw = ask(f"\n  С какой ступени начать (Enter — {recorded}): ",
                  str(recorded))
        try:
            start_stage = max(1, min(len(STAGES), int(raw)))
        except ValueError:
            return

        answer = ask("  Запустить эволюцию? (д/Н): ", "н")
        if not answer.lower().startswith(("д", "y")):
            return

        self._pipeline = EvolutionPipeline(
            trainer=trainer, config=config, start_stage=start_stage,
            goal_altitude=self.spec.park_orbit_altitude)
        pipeline = self._pipeline

        def job():
            summary = pipeline.run()
            print("\n" + pipeline.describe())
            self.log.info("Конвейер эволюции завершён: %s", summary)

        print(self.spawn("автономная эволюция", job))
        time.sleep(2.0)
        self.evolution_monitor()

    def evolution_monitor(self) -> None:
        """Наблюдение за конвейером: ступень, орбиты, экзамен."""
        print()
        print("  Наблюдение. Enter — назад в меню, эволюция продолжится.")
        print()
        last = ""
        while self.busy:
            pipeline = self._pipeline
            if pipeline is not None:
                state = pipeline.status()
                line = (f"  ступень {state['stage']} | орбит "
                        f"{state['orbits']}/{pipeline.config.promote_orbits} "
                        f"({state['rate']:.0%} за окно) | эпизодов "
                        f"{state['episodes']} | средняя "
                        f"{state['mean_return']:+.1f} | шагов "
                        f"{state['steps']} | переводов {state['promotions']} | "
                        f"откатов {state['rollbacks']}")
                if line != last:
                    print(line)
                    last = line
            if enter_pressed(timeout=5.0):
                return
        print()
        print("  Конвейер остановлен.")
        pause()

    # ==================================================================
    # 8. Манифест эволюции
    # ==================================================================
    def action_manifest(self) -> None:
        clear()
        from kia_core.neural.pipeline import MANIFEST_FILE
        print("  манифест")
        print()
        if not MANIFEST_FILE.exists():
            print(f"  Файла {MANIFEST_FILE.name} ещё нет: конвейер")
            print("  (пункт 0 главного меню) ни разу не запускался.")
            pause()
            return
        lines = MANIFEST_FILE.read_text(encoding="utf-8").splitlines()
        tail = lines[-60:]
        if len(lines) > len(tail):
            print(f"  ...показаны последние {len(tail)} строк из {len(lines)}")
            print()
        for line in tail:
            print("  " + line)
        print()
        print(f"  Файл целиком: {MANIFEST_FILE}")
        pause()

    # ==================================================================
    # 1. Связь с kRPC
    # ==================================================================
    def action_connect(self) -> None:
        clear()
        print("  kRPC")
        print(f"\n  Адрес сервера: {CONFIG.connection.address}:"
              f"{CONFIG.connection.rpc_port} (поток {CONFIG.connection.stream_port})")
        print("  Подключаюсь...\n")

        if not self.dispatcher.connect():
            print("  ошибка: Сервер kRPC не отвечает.")
            print("\n  Что проверить:")
            print("   • KSP запущена и загружено сохранение (не главное меню)")
            print("   • В игре открыто окно kRPC и нажата кнопка Start server")
            print(f"   • Порты {CONFIG.connection.rpc_port}/"
                  f"{CONFIG.connection.stream_port} свободны")
            pause()
            return

        conn = self.dispatcher.connection
        info = conn.info()
        print(f"  ok: {info.describe()}")
        print(f"  Игровая сцена:   {conn.game_scene()}")
        print(f"  Время игры (UT): {info.ut:.0f} с")

        try:
            vessel = conn.space_center.active_vessel
            print(f"  Активное судно:  {vessel.name}")
            print(f"    деталей {len(vessel.parts.all)}, "
                  f"масса {vessel.mass/1000:.2f} т, "
                  f"экипаж {vessel.crew_count}, "
                  f"ступень {vessel.control.current_stage}")
            print(f"    состояние: {str(vessel.situation).split('.')[-1]}")
        except Exception:
            print("  Активного судна нет (вы вне полётной сцены) — это нормально:")
            print("  KIA сама выведет аппарат на стартовый стол при запуске обучения.")

        save = self.dispatcher.provisioner.detect_save_name()
        print(f"\n  Сохранение KSP:  {save}")
        print(f"  Каталог игры:    {CONFIG.craft.ksp_root or 'не найден'}")
        from kia_core.engineer.parts_catalog import get_catalog
        stats = get_catalog().stats()
        print(f"  Каталог деталей: {stats['total']} шт "
              f"(двигателей {stats['engines']}, баков {stats['tanks']})")
        pause()

    # ==================================================================
    # 2. Автономное обучение
    # ==================================================================
    def action_autonomous(self) -> None:
        clear()
        print("  автообучение")
        if self.busy:
            print("\n  Обучение уже идёт.")
            self.monitor_loop()
            return

        print(f"\n  Цель: {self.spec.title} — {self.spec.describe()}")
        print(f"  Ветка эволюции: {self.dispatcher.goal_key()}")
        if GLOBAL_MEMORY.skills:
            print(f"  KIA стартует не с нуля: в памяти {len(GLOBAL_MEMORY.skills)} "
                  f"подтверждённых навыков")
        print("\n  Цикл: расчёт → сборка → старт → полёт → разбор → правка → снова")
        print("  Каждый запуск пишется в отдельную папку kia/<номер>/")
        print("\n  Остановить можно в любой момент клавишей Ctrl+C или из меню.")

        limit = ask("\n  Предел итераций (Enter — без предела): ", "")
        try:
            max_iterations = int(limit) if limit else None
        except ValueError:
            max_iterations = None

        if not self.dispatcher.connect():
            print("\n  ошибка: Нет связи с KSP — обучение не запущено.")
            print("  Сначала пункт 1 меню.")
            pause()
            return

        self.loop = AutonomousLoop(self.dispatcher, max_iterations=max_iterations)

        def job():
            try:
                self.loop.run()
            except Exception as exc:
                self.log.error("Автономный цикл упал: %s\n%s", exc,
                               traceback.format_exc())

        self.worker = threading.Thread(target=job, name="autonomy", daemon=True)
        self.worker.start()
        print("\n  пошло: Можно закрывать глаза — KIA работает сама.")
        time.sleep(1.5)
        self.monitor_loop()

    def monitor_loop(self) -> None:
        """Живое наблюдение; Enter — выход в меню, цикл продолжает работу."""
        print()
        print("  Наблюдение. Enter — назад в меню, обучение продолжится.")
        print()
        last = ""
        while self.busy:
            loop = self.loop
            if loop and loop.iterations:
                it = loop.iterations[-1]
                line = (f"  итерация {it.number} [{it.folder}] счёт {it.score:.0f} | "
                        f"{', '.join(it.failures) or ', '.join(it.milestones[-2:]) or '—'}")
            else:
                line = "  идёт первая итерация..."
            from kia_core.flight_log import FLIGHT_LOG
            output = f"{line}\n  сейчас: {FLIGHT_LOG.current_status}"
            if output != last:
                print(output)
                last = output
            if enter_pressed(timeout=3.0):
                return
        print()
        print("  Цикл завершён.")
        pause()

    def action_stop_learning(self) -> None:
        if self.loop is not None and self.loop.running:
            self.loop.stop()
            print("\n  Остановка после текущей итерации...")
        else:
            print("\n  Обучение не запущено.")

    # ==================================================================
    # 3. Настройки
    # ==================================================================
    def action_settings(self) -> None:
        while True:
            clear()
            print("  настройки")
            print(self.header())
            print("""
    1  Сменить глобальную цель
    2  Изменить награду или штраф
    3  Включить/выключить правило качества
    4  Показать весь прейскурант
    5  Сбросить прейскурант к значениям по умолчанию
    6  Лимит времени на одну попытку
    0  Назад
""")
            choice = ask("  > ")
            if choice == "0" or not choice:
                return
            if choice == "1":
                self.settings_goal()
            elif choice == "2":
                self.settings_price()
            elif choice == "3":
                self.settings_rule()
            elif choice == "4":
                clear()
                print(REWARD_BOOK.describe())
                pause()
            elif choice == "5":
                REWARD_BOOK.reset()
                print("\n  Прейскурант возвращён к значениям по умолчанию.")
                time.sleep(1.2)
            elif choice == "6":
                self.settings_time_limit()

    def settings_goal(self) -> None:
        clear()
        print("  цель\n")
        print("  Каждая цель — отдельная эволюционная ветка ракеты:")
        print("  накопленный опыт по другой цели не мешает, а навыки")
        print("  из глобальной памяти переносятся.\n")
        for key, (title, _) in GOALS.items():
            print(f"    {key}  {title}")
        print("    7  Своя формулировка (например: «облёт Дюны без экипажа»)")
        choice = ask("\n  Цель: ")
        if choice in GOALS:
            command = GOALS[choice][1]
        elif choice == "7":
            command = ask("  Опишите задачу словами: ")
            if not command:
                return
        else:
            return
        self.spec = parse_command(command, self.spec)
        self.spec.crewed = False          # KIA строит только беспилотники
        self.spec.save()
        self.dispatcher.set_spec(self.spec)
        if self.loop is not None:
            self.loop.goal = self.dispatcher.goal_key()
        print(f"\n  Цель принята: {self.spec.title} — {self.spec.describe()}")
        print(f"  Ветка эволюции: {self.dispatcher.goal_key()}")
        pause()

    def settings_price(self) -> None:
        clear()
        print("  цены\n")
        print("  Доступные события (можно писать по-русски):")
        print("    достижения: орбита, отрыв, космос, узел, перехват, сои, муна,")
        print("                посадка, наука, возвращение")
        print("    отказы:     взрыв, падение, топливо, электричество, таймаут,")
        print("                управление, связь\n")
        name = ask("  Что меняем: ")
        if not name:
            return
        raw = ask("  Новое значение (например 800 или -2000): ")
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("  Это не число.")
            pause()
            return
        ok, message = REWARD_BOOK.set_value(name, value)
        print(f"\n  {message}")
        if ok:
            print("  Изменение действует немедленно, в том числе для идущего полёта.")
        pause()

    def settings_rule(self) -> None:
        clear()
        print("  правила качества\n")
        for key, (description, _) in RULE_CATALOG.items():
            active = REWARD_BOOK.rules.get(key)
            mark = f"{active.weight:+.0f} очков" if active else "выключено"
            print(f"    {key:16s} {description:45s} [{mark}]")
        name = ask("\n  Какое правило настроить (Enter — назад): ")
        if not name:
            return
        raw = ask("  Цена в очках (0 — выключить): ")
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("  Это не число.")
            pause()
            return
        if value == 0:
            ok, message = REWARD_BOOK.remove_rule(name)
        else:
            ok, message = REWARD_BOOK.set_value(name, value)
        print(f"\n  {message}")
        pause()

    def settings_time_limit(self) -> None:
        clear()
        print(f"  Текущий лимит на попытку: {self.spec.max_wall_time/60:.0f} минут")
        raw = ask("  Новый лимит в минутах (Enter — не менять): ")
        if not raw:
            return
        try:
            self.spec.max_wall_time = max(120.0, float(raw) * 60.0)
            self.spec.save()
            print(f"  Лимит: {self.spec.max_wall_time/60:.0f} минут")
        except ValueError:
            print("  Это не число.")
        pause()

    # ==================================================================
    # 4. Логи и отчёты
    # ==================================================================
    def action_reports(self) -> None:
        clear()
        print("  логи")
        runs = WORKSPACE.all_runs()
        if not runs:
            print("\n  Запусков ещё не было — папка kia/ пуста.")
            print("  Запустите обучение (пункт 2 меню).")
            pause()
            return

        print("\n" + WORKSPACE.summary(limit=15))
        choice = ask(f"\n  Номер запуска (Enter — последний, {runs[-1].number}): ",
                     str(runs[-1].number))
        try:
            number = int(choice)
        except ValueError:
            return
        run = WORKSPACE.get(number)
        if run is None:
            print("  Такого запуска нет.")
            pause()
            return

        while True:
            clear()
            meta = run.read_meta()
            print(f"  kia/{run.number}")
            if meta:
                print(f"  Цель: {meta.get('goal', '—')}")
                print(f"  Счёт: {meta.get('score', '—')} | итог: {meta.get('outcome', '—')}")
                print(f"  Аппарат: {meta.get('mass_t', '—')} т, "
                      f"ΔV {meta.get('delta_v', '—')} м/с, TWR {meta.get('twr', '—')}")
                if meta.get("new_skills"):
                    print(f"  Новые навыки: {', '.join(meta['new_skills'])}")
            print("""
    1  Отчёт конструктора  (kia_design_report.md)
    2  Журнал полёта       (kia_flight_log.md)
    3  Оба файла подряд
    0  Назад
""")
            choice = ask("  > ")
            if choice == "0" or not choice:
                return
            if choice in ("1", "3"):
                self.show_file(run.design_report, "отчёт конструктора")
            if choice in ("2", "3"):
                self.show_file(run.flight_log, "журнал полёта")

    @staticmethod
    def show_file(path: Path, title: str) -> None:
        clear()
        print(f"  {title}: {path}")
        if not path.exists():
            print("\n  Файл не найден.")
            pause()
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        page = 45
        for start in range(0, len(lines), page):
            for line in lines[start:start + page]:
                print("  " + line)
            if start + page < len(lines):
                more = ask(f"\n  --- показано {start + page} из {len(lines)} строк, "
                           f"Enter — дальше, q — выход --- ")
                if more.lower().startswith("q"):
                    return
                clear()
        pause()

    # ==================================================================
    # 5. Экстренный сброс
    # ==================================================================
    def action_reset(self) -> None:
        clear()
        print("  сброс")
        print("""
    1  Отменить текущий полёт в игре (Revert Flight)
    2  Очистить историю обучения (data/*.json)
    3  Очистить папки запусков kia/1, kia/2, ...
    4  Полный сброс: пункты 1 + 2 + 3
    5  СТЕРЕТЬ ГЛОБАЛЬНУЮ ПАМЯТЬ (все подтверждённые навыки)
    0  Назад
""")
        choice = ask("  Пункт: ")
        if choice == "0" or not choice:
            return

        if choice in ("1", "4"):
            self.revert_flight()
        if choice in ("2", "4"):
            self.clear_history()
        if choice in ("3", "4"):
            self.clear_runs()
        if choice == "5":
            confirm = ask("\n  Память — это весь накопленный опыт KIA.\n"
                          "  Введите СТЕРЕТЬ для подтверждения: ")
            if confirm.strip().upper() == "СТЕРЕТЬ":
                path = GLOBAL_MEMORY.path
                if path.exists():
                    path.unlink()
                GLOBAL_MEMORY.skills.clear()
                GLOBAL_MEMORY.failure_lessons.clear()
                GLOBAL_MEMORY.best_per_goal.clear()
                GLOBAL_MEMORY.total_runs = 0
                GLOBAL_MEMORY.save()
                print("  Глобальная память стёрта.")
            else:
                print("  Отменено — память не тронута.")
        pause()

    def revert_flight(self) -> None:
        print("\n  Отмена полёта в игре...")
        if self.loop is not None and self.loop.running:
            self.loop.stop()
        self.dispatcher.abort()
        if not self.dispatcher.connect():
            print("  ! Нет связи с KSP — отменить полёт нечем.")
            return
        try:
            from kia_core.environment.control import ShipControl
            control = ShipControl(self.dispatcher.connection)
            control.full_stop()
            if control.revert_to_launch():
                print("  ok: Полёт отменён, аппарат возвращён на стартовый стол.")
            else:
                print("  ! Игра не приняла Revert Flight "
                      "(возможно, вы не в полётной сцене).")
        except Exception as exc:
            print(f"  ! Ошибка отмены: {exc}")

    @staticmethod
    def clear_history() -> None:
        removed = []
        for name in ("history.json", "optimizer_state.json", "best_genome.json",
                     "autonomy_state.json", "diagnoses.json"):
            path = DATA_DIR / name
            if path.exists():
                path.unlink()
                removed.append(name)
        print(f"\n  История обучения очищена: {', '.join(removed) or 'файлов не было'}")
        print("  Глобальная память (kia_global_memory.json) НЕ тронута.")

    @staticmethod
    def clear_runs() -> None:
        runs = WORKSPACE.all_runs()
        for run in runs:
            shutil.rmtree(run.path, ignore_errors=True)
        print(f"\n  Удалено папок запусков: {len(runs)}")

    # ==================================================================
    # 6. Глобальная память
    # ==================================================================
    def action_memory(self) -> None:
        clear()
        print("  память")
        print(f"  Файл: {GLOBAL_MEMORY.path}")
        print("  Этот файл переживает перезапуски, смену цели и сброс истории.\n")
        print(GLOBAL_MEMORY.summary())

        if not GLOBAL_MEMORY.skills and not GLOBAL_MEMORY.failure_lessons:
            # Пустая память — это не поломка, а закономерность: её
            # наполняют только НАСТОЯЩИЕ полёты. Без объяснения оператор
            # решает, что пункт меню сломан.
            print()
            print("  пусто: сюда пишут только реальные полёты в KSP,")
            print("  тренажёры не пишут. Наполняют: пункт 2, пункт 7->7,")
            print("  python main.py auto. Нужна запущенная игра.")
        else:
            profile = GLOBAL_MEMORY.proven_ascent_profile()
            if profile:
                print("\n  Эти параметры KIA считает доказанными и берёт за основу")
                print("  в каждом новом поколении, а оптимизатору запрещено уходить")
                print("  от них далеко — так система не повторяет старых ошибок.")
        pause()

    # ==================================================================
    # 7. Нейросеть (PPO на PyTorch)
    # ==================================================================
    def action_neural(self) -> None:
        from kia_core.neural import is_available
        if not is_available():
            clear()
            print("  нейросеть")
            print()
            print("  PyTorch не установлен. Команда установки с поддержкой GPU:")
            print("    pip install --force-reinstall torch "
                  "--index-url https://download.pytorch.org/whl/cu126")
            pause()
            return

        while True:
            clear()
            from kia_core.neural.device import inspect
            info = inspect()
            print(f"  {info.describe()}")
            if self._trainer is not None:
                stats = self._trainer.brain.pilot.stats
                print(f"  {stats.updates} обновлений, {stats.total_steps} шагов, "
                      f"{stats.episodes} эпизодов, средняя "
                      f"{self._trainer.brain.pilot.mean_return(20):+.2f}")
            print(f"  {'обучение идёт' if self.busy else 'простой'}")
            print()
            self.neural_ladder()
            print(NEURAL_MENU)
            choice = ask("  > ")
            if choice == "0" or not choice:
                return
            if choice in ("1", "2", "3"):
                self.neural_stage(int(choice))
            elif choice == "4":
                self.neural_program()
            elif choice == "5":
                self.neural_info()
            elif choice == "6":
                self.neural_eval()
            elif choice == "7":
                self.neural_ksp()
            elif choice == "8":
                self.neural_builder()
            elif choice == "9":
                self.neural_palette()
            elif choice == "10":
                if self._pipeline is not None and self._pipeline.running:
                    self._pipeline.stop()
                    print("\n  Останавливаю конвейер эволюции "
                          "после текущего пакета...")
                    time.sleep(1.5)
                elif self._trainer is not None:
                    self._trainer.stop()
                    print("\n  Остановка после текущего пакета...")
                    time.sleep(1.5)
                else:
                    print("\n  Обучение не запущено.")
                    time.sleep(1.0)
            elif choice == "11":
                self.neural_restore_best()
            elif choice == "12":
                self.neural_rollback()
            elif choice == "13":
                self.neural_reset()
            elif choice == "14":
                self.neural_knowledge()
            elif choice == "15":
                self.neural_brain_map()
            elif choice == "16":
                self.neural_hybrid()
            elif choice == "17":
                self.program_stage("builder")
            elif choice == "18":
                self.program_stage("math")
            elif choice == "19":
                self.math_exam()
            elif choice == "20":
                self.programs_description()
            elif choice == "21":
                self.navigator_train()
            elif choice == "22":
                self.navigator_exam()

    # ------------------------------------------------------------------
    def _ensure_trainer(self):
        if self._trainer is None:
            from kia_core.neural.trainer import NeuralTrainer
            print("\n  Загружаю мозг (первый раз — несколько секунд)...")
            self._trainer = NeuralTrainer()
        return self._trainer

    def neural_info(self) -> None:
        clear()
        trainer = self._ensure_trainer()
        print("  мозг")
        print(trainer.brain.summary())
        print()
        print(f"  Что сеть знает до обучения: "
              f"{ROOT / 'KIA_BASELINE_KNOWLEDGE.md'}")
        pause()

    # ------------------------------------------------------------------
    # Учебная программа: три ступени тренажёра
    # ------------------------------------------------------------------
    def neural_ladder(self) -> None:
        """Лестница обучения: что пройдено, что впереди."""
        from kia_core.neural.curriculum import progress
        current = self._trainer.brain.stage if self._trainer is not None else None
        print("  лестница")
        for status in progress(current):
            stage = status.stage
            mark = f"[{status.mark}]"
            print(f"    {stage.number}. {stage.title:18s} {mark:12s} "
                  f"{status.checkpoint.describe()}")

    def neural_program(self) -> None:
        clear()
        from kia_core.neural.curriculum import describe_program
        print("  ступени")
        print()
        print("  sim-to-real: мир усложняется по ступеням, каждая стартует")
        print("  с весов предыдущей. Снимок на переходе — kia_model_stageN.pth,")
        print("  откат пунктом 12.")
        print()
        print(describe_program())
        pause()

    def neural_stage(self, number: int) -> None:
        """Запуск обучения на ступени программы."""
        clear()
        from kia_core.neural.curriculum import (checkpoint_info, prepare,
                                                stage_by_number)
        stage = stage_by_number(number)
        print(f"  обучение: {stage.title}")
        print()
        print(stage.describe())
        print()
        if stage.backup is not None:
            info = checkpoint_info(stage.backup)
            if info.exists:
                print(f"  Снимок прошлой ступени уже есть: {info.describe()}")
            else:
                print(f"  Перед стартом будет сделана резервная копия весов")
                print(f"  в {stage.backup.name} — прошлая ступень не потеряется.")
            print()
        print("  Тренажёр не трогает игру: ни одного реального запуска,")
        print("  только счёт. Обучение конечно — оно остановится на")
        print("  заданном числе шагов.")
        print()
        raw = ask(f"  Сколько шагов (Enter — {stage.default_steps}): ",
                  str(stage.default_steps))
        try:
            steps = int(raw)
        except ValueError:
            return
        if self.busy:
            print("\n  Уже идёт задание. Сначала остановите его (пункт 10).")
            pause()
            return

        trainer = self._ensure_trainer()
        if stage.backup is not None:
            # проверяем возможность перехода ДО запуска фонового задания,
            # иначе оператор увидит «запущено» и тишину в логах
            probe = prepare(stage, brain=trainer.brain, dry_run=True)
            print()
            for line in probe.messages:
                print(f"  {'нельзя' if not probe.ok else '·'} {line}")
            if not probe.ok:
                pause()
                return
            if ask("\n  Начинать? (Д/н): ", "д").lower().startswith("н"):
                return

        def job():
            trainer.begin_session()
            report = trainer.train_stage(
                stage, total_steps=steps,
                goal_altitude=self.spec.park_orbit_altitude)
            self.log.info("Ступень %d завершена: %s", stage.number,
                          report.describe())

        print(self.spawn(f"обучение: {stage.title}", job))
        time.sleep(2.0)
        self.neural_monitor()


    def neural_builder(self) -> None:
        clear()
        print("  конструктор")
        print()
        print("  Сеть выбирает детали по одной из палитры каталога игры.")
        print("  Награда — за то, что реально получилось: ΔV, TWR, масса.")
        print()
        raw = ask("  Сколько эпизодов сборки (Enter — 3000): ", "3000")
        try:
            episodes = int(raw)
        except ValueError:
            return
        trainer = self._ensure_trainer()

        def job():
            trainer.begin_session()
            trainer.train_builder(episodes=episodes,
                                  goal_altitude=self.spec.park_orbit_altitude,
                                  goal_transfer=self.spec.needs_transfer)

        print(self.spawn("обучение конструктора", job))
        time.sleep(1.5)
        self.neural_monitor()

    # ------------------------------------------------------------------
    def program_stage(self, program: str) -> None:
        """Запуск ступени программы конструктора или математика."""
        from kia_core.neural.programs import stages_of

        clear()
        titles = {"builder": "программа конструктора",
                  "math": "программа математика"}
        print(f"  {titles[program]}")
        print()
        trainer = self._ensure_trainer()
        reached = int(trainer.brain.stages.get(program, 1))
        stages = stages_of(program)
        for stage in stages:
            mark = ("ПРОЙДЕНА" if reached > stage.number else
                    "в работе" if reached == stage.number else "не начата")
            print(f"    {stage.number}  {stage.title:12s} {stage.summary:45s} "
                  f"[{mark}]")
        print()
        raw = ask(f"  Ступень (Enter — текущая, {reached}): ", str(reached))
        try:
            stage = next(s for s in stages if s.number == int(raw))
        except (ValueError, StopIteration):
            print("  Нет такой ступени.")
            time.sleep(1.2)
            return
        default = str(stage.default_steps)
        raw_n = ask(f"  Сколько задач/эпизодов (Enter — {default}): ", default)
        try:
            amount = int(raw_n)
        except ValueError:
            return

        def job():
            trainer.begin_session()
            if program == "math":
                trainer.train_math(stage=stage, episodes=amount)
            else:
                trainer.train_builder(stage=stage, episodes=amount,
                                      goal_altitude=self.spec.park_orbit_altitude,
                                      goal_transfer=self.spec.needs_transfer)

        print(self.spawn(f"{titles[program].lower()}: {stage.title}", job))
        time.sleep(1.5)
        self.neural_monitor()

    def math_exam(self) -> None:
        """Что математик знает по разделам — без обучения."""
        clear()
        trainer = self._ensure_trainer()
        current = int(trainer.brain.stages.get("math", 1))
        raw = ask(f"  На какой ступени проверять (Enter — {current}): ",
                  str(current))
        try:
            number = int(raw)
        except ValueError:
            return
        print("\n  Считаю...")
        result = trainer.evaluate_math(number, tasks=600)
        clear()
        print(f"  экзамен матан: {result['title']}")
        print(f"  Решено {result['accuracy'] * 100:.1f} % задач в допуске "
              f"{result['tolerance'] * 100:.0f} %")
        print()
        print(f"  {'раздел':24s} {'решено':>11s}  {'средняя ошибка':>14s}")
        for key, row in result["tasks"].items():
            share = 100.0 * row["solved"] / max(1, row["total"])
            print(f"  {key:24s} {row['solved']:4d}/{row['total']:<4d} "
                  f"{share:4.0f}% {row['mean_error'] * 100:13.2f} %")
        print()
        print("  Эталон берётся из тех же формул, которыми считает полётный")
        print("  код. Раздел ниже 80 % — сеть его ещё не знает.")
        pause()

    def programs_description(self) -> None:
        from kia_core.neural.programs import describe_program
        clear()
        print(describe_program("builder"))
        print()
        print(describe_program("math"))
        print()
        pause()

    def neural_eval(self) -> None:
        clear()
        from kia_core.neural.curriculum import STAGES, stage_by_number
        print("  ПРОВЕРКА ПОЛИТИКИ (без обучения)")
        print()
        trainer = self._ensure_trainer()
        print("  На какой ступени проверять:")
        for stage in STAGES:
            print(f"    {stage.number}  {stage.title}")
        default = str(trainer.brain.stage)
        raw = ask(f"  Ступень (Enter — {default}, где мозг учился): ", default)
        try:
            stage = stage_by_number(int(raw))
        except (ValueError, KeyError):
            return
        print(f"\n  Гоняю {stage.title}...")
        result = trainer.evaluate_pilot(
            episodes=10, goal_altitude=self.spec.park_orbit_altitude,
            stage=stage)
        print()
        print(f"  Ступень:          {result['stage_title']}")
        print(f"  Эпизодов:         {result['episodes']}")
        print(f"  Средняя награда:  {result['mean_return']:+.2f}")
        print(f"  Лучшая награда:   {result['best_return']:+.2f}")
        print(f"  Достигнуто орбит: {result['orbits']}")
        print(f"  Разбилось:        {result['crashes']}")
        print(f"  Средний потолок:  {result['mean_peak_altitude']:.0f} м")
        print(f"  Чем кончалось:    " + ", ".join(
            f"{reason} ×{count}"
            for reason, count in sorted(result["reasons"].items(),
                                        key=lambda item: -item[1])))
        print()
        print("  Для необученной сети норма: 0 орбит, награда около -2.")
        pause()

    def neural_ksp(self) -> None:
        clear()
        print("  живая KSP")
        print()
        print("  Каждый эпизод — реальный запуск. Автопилот выключен:")
        print("  осями управляет нейросеть напрямую.")
        print("  Это медленно: один эпизод — минуты реального времени.")
        print()
        if not self.dispatcher.connect():
            print("  ошибка: Нет связи с KSP. Сначала пункт 1 главного меню.")
            pause()
            return
        raw = ask("  Сколько запусков (Enter — 5): ", "5")
        try:
            episodes = int(raw)
        except ValueError:
            return
        neural_build = ask("  Ракету собирает нейросеть? (д/Н): ", "н")
        trainer = self._ensure_trainer()
        use_neural = neural_build.lower().startswith(("д", "y"))

        def job():
            trainer.begin_session()
            trainer.train_in_ksp(
                self.dispatcher, episodes=episodes,
                goal_altitude=self.spec.park_orbit_altitude,
                use_neural_builder=use_neural,
                flight_log=self.dispatcher.flight_log)

        print(self.spawn("нейрообучение в KSP", job))
        time.sleep(1.5)
        self.neural_monitor()

    def neural_monitor(self) -> None:
        print()
        print("  Наблюдение. Enter — назад в меню, обучение продолжится.")
        print()
        last = ""
        while self.busy:
            trainer = self._trainer
            if trainer is not None:
                stats = trainer.brain.pilot.stats
                best = (stats.best_return
                        if stats.best_return > float("-inf") else 0.0)
                report = trainer.last_report
                orbits = f" | орбит {report.successes}" if report else ""
                line = (f"  ступень {trainer.brain.stage} | решений "
                        f"{stats.total_steps} | обновлений "
                        f"{stats.updates} | эпизодов {stats.episodes} | "
                        f"средняя {trainer.brain.pilot.mean_return(20):+.2f} | "
                        f"рекорд {best:+.1f}{orbits}")
                if line != last:
                    print(line)
                    last = line
            if enter_pressed(timeout=3.0):
                return
        print()
        print("  Обучение завершено.")
        pause()

    def neural_palette(self) -> None:
        clear()
        from kia_core.neural.builder_env import build_palette
        print("  палитра")
        print("  Из этих деталей сеть собирает ракету. Порядок фиксирован,")
        print("  иначе сохранённые веса потеряли бы смысл.")
        print()
        print(build_palette().describe())
        pause()

    def neural_restore_best(self) -> None:
        clear()
        from kia_core.neural.ppo import BEST_MODEL_FILE
        print("  лучшие веса")
        print()
        if self.busy:
            print("  Сначала остановите обучение (пункт 7), иначе оно затрёт файл.")
            pause()
            return
        if not BEST_MODEL_FILE.exists():
            print(f"  Файла {BEST_MODEL_FILE.name} нет — лучшая версия ещё")
            print("  не сохранялась (нужно минимум 20 эпизодов обучения).")
            pause()
            return
        trainer = self._ensure_trainer()
        print(f"  Сейчас в работе: средняя за 20 = "
              f"{trainer.brain.pilot.mean_return(20):+.2f}")
        if trainer.brain.restore_best():
            print(f"  Восстановлено: средняя за 20 = "
                  f"{trainer.brain.pilot.mean_return(20):+.2f}")
        else:
            print("  Не удалось восстановить, подробности в logs/.")
        pause()

    def neural_rollback(self) -> None:
        """Вернуть в работу веса пройденной ступени."""
        clear()
        from kia_core.neural.curriculum import (STAGES, checkpoint_info,
                                                rollback, stage_by_number)
        print("  откат к ступени")
        print()
        if self.busy:
            print("  Сначала остановите обучение (пункт 10), иначе оно")
            print("  затрёт восстановленные веса своим следующим сохранением.")
            pause()
            return
        available = []
        for index, stage in enumerate(STAGES):
            following = STAGES[index + 1] if index + 1 < len(STAGES) else None
            snapshot = following.backup if following else None
            if snapshot is None:
                continue
            info = checkpoint_info(snapshot)
            print(f"    {stage.number}  {stage.title:20s} {info.describe()}")
            if info.exists:
                available.append(stage.number)
        print()
        if not available:
            print("  Снимков ещё нет: они появляются при переходе на")
            print("  следующую ступень. Пока откатываться не к чему.")
            pause()
            return
        raw = ask("  Ступень (Enter — отмена): ")
        if not raw.strip():
            return
        try:
            stage = stage_by_number(int(raw))
        except (ValueError, KeyError):
            print("  Нет такой ступени.")
            pause()
            return
        trainer = self._ensure_trainer()
        print(f"\n  Сейчас в работе: ступень {trainer.brain.stage}, "
              f"средняя за 20 = {trainer.brain.pilot.mean_return(20):+.2f}")
        if rollback(stage, trainer.brain):
            print(f"  Восстановлено: ступень {trainer.brain.stage}, "
                  f"средняя за 20 = {trainer.brain.pilot.mean_return(20):+.2f}")
        else:
            print("  Не удалось восстановить, подробности в logs/.")
        pause()

    def neural_knowledge(self) -> None:
        """Что сеть подтвердила в тренажёрах — в отличие от живой памяти."""
        clear()
        from kia_core.neural.knowledge import KNOWLEDGE_FILE, SimKnowledge
        print("  знания тренажёров")
        print()
        print("  Это не глобальная память (пункт 6 главного меню): туда")
        print("  попадают только настоящие полёты в KSP. Здесь — то, что")
        print("  нейросеть показала на счёте, без единого запуска игры.")
        print()
        knowledge = SimKnowledge()          # свежее чтение с диска
        print(knowledge.summary())
        print()
        print(f"  Файл: {KNOWLEDGE_FILE}")
        print("  Наполняется сам по ходу обучения. Видно в реальном времени")
        print("  на карте мозга — пункт 15.")
        pause()

    def neural_brain_map(self) -> None:
        """Поднимает живую карту мозга в браузере."""
        clear()
        print("  карта мозга")
        print()
        print("  слева приборы, в середине слои, справа решения.")
        print("  обновление раз в 2 с. то же самое — BRAIN_MAP.bat")
        print()
        try:
            from kia_brain_map import serve
            server, address = serve(open_browser=True, block=False)
            self._brain_map = server
            print(f"  Открыто: {address}")
            print("  Окно браузера можно закрыть в любой момент — на обучение")
            print("  это не влияет.")
        except Exception as exc:
            print(f"  ошибка: Карта не поднялась: {exc}")
        pause()

    def navigator_train(self) -> None:
        """Межпланетная программа: все планеты, все цели, связь — одним прогоном."""
        clear()
        print("  навигатор")
        print()
        print("  Мохо, Ив, Дюна, Дрес, Джул, Илу — посадка и орбиты.")
        print("  3 ступени, 10 процессов на CPU, Ctrl+C — стоп")
        print("  (сохраняется каждые 10 итераций).")
        print()
        from kia_core.neural.navigator import NavigatorTrainer
        trainer = NavigatorTrainer(workers=10, batch=512)
        try:
            trainer.train()
        except KeyboardInterrupt:
            trainer.report()
            print()
            print("  Остановлено. Веса сохранены.")
        pause()

    def navigator_exam(self) -> None:
        clear()
        from kia_core.neural.navigator import REPORT_FILE
        print("  экзамен навигатора")
        if not REPORT_FILE.exists():
            print()
            print("  Навигатор ещё не обучался.")
        else:
            data = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
            print()
            print(f"  Ступень {data['stage']}"
                  + ("  — программа пройдена" if data.get("finished") else ""))
            for pair, rate in data.get("pairs", {}).items():
                print(f"    {pair:18} {rate * 100:5.1f} %")
        pause()

    def neural_hybrid(self) -> None:
        """Переключатель совместного счёта на процессоре и видеокарте."""
        clear()
        from kia_core.neural.hybrid import configured_share, save_share
        print("  CPU+GPU")
        print()
        print("  замер на этой машине:")
        print("    только CPU ........ 11 623 шаг/с")
        print("    65/35 CPU/GPU .....  6 161 шаг/с")
        print("    поровну ...........  6 259 шаг/с")
        print()
        print("  совместный режим медленнее почти вдвое: GPU упирается не в")
        print("  счёт, а в python-обвязку запуска ядер, она держит GIL.")
        print()
        current = configured_share()
        print(f"  Сейчас: доля видеокарты {current:.0%}"
              + ("  (выключено, самый быстрый режим)" if current <= 0 else ""))
        print()
        print("    0  выключить — считает только процессор")
        print("    1  35 % видеокарте (процессор берёт основное)")
        print("    2  50 % видеокарте (поровну)")
        print()
        answer = ask("  Выбор (Enter — оставить как есть): ")
        mapping = {"0": 0.0, "1": 0.35, "2": 0.5}
        if answer in mapping:
            share = save_share(mapping[answer])
            print(f"\n  Записано: доля видеокарты {share:.0%}.")
            print("  Вступит в силу при следующем запуске обучения.")
            if self._trainer is not None:
                self._trainer.brain.pilot.enable_hybrid(share)
        pause()

    def neural_reset(self) -> None:
        clear()
        from kia_core.neural.ppo import MODEL_FILE
        print("  стереть веса")
        print()
        if not MODEL_FILE.exists():
            print("  Файла весов нет — мозг и так чистый.")
            pause()
            return
        print(f"  Будет удалён {MODEL_FILE}")
        print("  Весь опыт обучения нейросети пропадёт безвозвратно.")
        answer = ask("\n  Введите СТЕРЕТЬ для подтверждения: ")
        if answer.strip().upper() == "СТЕРЕТЬ":
            MODEL_FILE.unlink()
            self._trainer = None
            print("  Веса удалены.")
        else:
            print("  Отменено.")
        pause()

    # ==================================================================
    def run(self) -> int:
        while True:
            clear()
            print(BANNER)
            print(self.header())
            print(MENU)
            if self.busy:
                print("  (обучение идёт в фоне — пункт 2 покажет ход,"
                      " пункт 5 остановит)")
            choice = ask("  > ")
            if choice == "0":
                self.action_evolution()
            elif choice == "1":
                self.action_connect()
            elif choice == "2":
                self.action_autonomous()
            elif choice == "3":
                self.action_settings()
            elif choice == "4":
                self.action_reports()
            elif choice == "5":
                self.action_reset()
            elif choice == "6":
                self.action_memory()
            elif choice == "7":
                self.action_neural()
            elif choice == "8":
                self.action_manifest()
            elif choice in ("9", "q", "exit", "выход"):
                if self.loop is not None and self.loop.running:
                    print("\n  Останавливаю автономный цикл...")
                    self.loop.stop()
                if self._pipeline is not None and self._pipeline.running:
                    print("\n  Останавливаю конвейер эволюции...")
                    self._pipeline.stop()
                self.dispatcher.disconnect()
                print("\n  Работа пульта завершена. Данные сохранены.\n")
                return 0
            elif choice == "":
                continue
            else:
                print("  Нет такого пункта.")
                time.sleep(0.8)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        return ControlPanel().run()
    except KeyboardInterrupt:
        print("\n\n  Прервано оператором.\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
