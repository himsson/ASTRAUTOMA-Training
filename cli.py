"""Консоль прямого управления KIA — диалог оператора с ИИ.

Оператор пишет команды обычным языком, не трогая код:

    > лети на Муну
    > выйди на орбиту 100 км
    > повысь штраф за взрыв до -2000
    > дай +500 за ровный шаг тангажа
    > проект            — рассчитать аппарат и показать отчёт
    > старт             — выполнить миссию (полёт идёт в фоне)
    > обучение 5        — пять поколений эволюции
    > автономно         — бесконечный цикл: сама строит, летает, чинит
    > лог               — что ИИ делает прямо сейчас
    > стоп              — прервать текущий полёт

Полёт выполняется в фоновом потоке, поэтому цены наград и задачу можно
менять прямо во время полёта — они действуют немедленно.
"""
from __future__ import annotations

import re
import sys
import threading
import traceback
from pathlib import Path

from kia_core.agent.autonomy import AutonomousLoop
from kia_core.agent.dispatcher import Dispatcher
from kia_core.config import CONFIG, ROOT
from kia_core.engineer.autodesign import DesignPolicy
from kia_core.flight_log import FLIGHT_LOG
from kia_core.learning.reward_book import REWARD_BOOK, RULE_CATALOG
from kia_core.logging_setup import setup_logging
from kia_core.mission_spec import MissionSpec, parse_command

BANNER = r"""
 _  _____ _
| |/ /_ _/ \       KERBAL INTELLIGENCE AGENCY
| ' / | | _ |      консоль прямого управления
| . \ | | | |      наберите «помощь» для списка команд
|_|\_\___|_|_|     «выход» — завершить сессию
"""

HELP = """
КОМАНДЫ (пишите как удобно, ИИ разберёт фразу):

  ЗАДАЧА
    лети на Муну                     задать цель полёта
    выйди на орбиту 100 км           орбита у Кербина
    сядь на Мун и вернись            посадка + возвращение
    облёт Минмуса без экипажа        пролёт беспилотником
    задача                           показать текущую задачу

  ЦЕНЫ И ПРАВИЛА (действуют сразу, в том числе во время полёта)
    повысь штраф за взрыв до -2000   изменить штраф
    награда за орбиту 800            изменить награду
    дай +500 за ровный шаг тангажа   включить правило качества
    убери правило тангаж             выключить правило
    награды                          показать весь прейскурант
    сбрось награды                   вернуть значения по умолчанию

  РАБОТА
    проект                           рассчитать аппарат, отчёт в kia_design_report.md
    старт                            один полёт по текущей задаче
    обучение 5                       5 поколений эволюции
    автономно                        БЕСКОНЕЧНЫЙ цикл без участия человека:
                                     проектирование → сборка → полёт →
                                     разбор отказа → правка чертежа → снова
    автономно 20                     то же, но ровно 20 итераций
    стоп                             прервать текущий полёт или автономный режим
    разбор                           последний диагноз и внесённые правки
    статус                           состояние системы и истории
    лог [N]                          последние N записей журнала полёта
    телеметрия                       текущие показания с судна
    каталог                          сколько деталей прочитано из игры
    связь                            проверить соединение с KSP
    выход                            закончить работу
"""


class Console:
    """Диалоговая оболочка над диспетчером."""

    def __init__(self):
        self.log = setup_logging()
        self.spec = MissionSpec.load()
        self.dispatcher = Dispatcher(spec=self.spec, resume=True)
        self.worker: threading.Thread | None = None
        self.loop: AutonomousLoop | None = None
        self.running = True

    # ------------------------------------------------------------------
    def busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def spawn(self, name: str, fn) -> str:
        if self.busy():
            return "Уже выполняется задание. «стоп» — прервать, «лог» — посмотреть ход."
        def wrapper():
            try:
                fn()
            except Exception as exc:
                self.log.error("Задание '%s' упало: %s\n%s", name, exc,
                               traceback.format_exc())
        self.worker = threading.Thread(target=wrapper, name=name, daemon=True)
        self.worker.start()
        return f"Задание «{name}» запущено в фоне. Консоль свободна: «лог», «стоп», «статус»."

    # ==================================================================
    # Обработчики команд
    # ==================================================================
    def cmd_help(self, _: str) -> str:
        return HELP

    def cmd_task_show(self, _: str) -> str:
        s = self.spec
        return (f"Текущая задача: {s.title}\n  {s.describe()}\n"
                f"  опорная орбита: {s.park_orbit_altitude/1000:.0f} км | "
                f"орбита у цели: {s.target_orbit_altitude/1000:.0f} км\n"
                f"  цели: {', '.join(s.objectives)} | "
                f"запас ΔV ×{s.dv_margin:.2f} | "
                f"лимит на попытку {s.max_wall_time/60:.0f} мин")

    def cmd_set_task(self, text: str) -> str:
        self.spec = parse_command(text, self.spec)
        self.dispatcher.set_spec(self.spec)
        return ("Задача принята.\n" + self.cmd_task_show("") +
                "\nДальше: «проект» — посчитать аппарат, «старт» — лететь.")

    # ------------------------------------------------------------------
    def cmd_rewards(self, _: str) -> str:
        return REWARD_BOOK.describe()

    def cmd_reset_rewards(self, _: str) -> str:
        REWARD_BOOK.reset()
        return "Прейскурант возвращён к значениям по умолчанию."

    def cmd_set_reward(self, text: str) -> str:
        """«повысь штраф за взрыв до -2000», «дай +500 за ровный шаг тангажа»."""
        numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?", text)
        if not numbers:
            return "Не вижу числа. Пример: «штраф за взрыв -2000» или «+500 за тангаж»."
        value = float(numbers[-1].replace(",", "."))

        lowered = text.lower().replace("ё", "е")
        # знак: «штраф» без явного минуса означает отрицательное значение
        if "штраф" in lowered and value > 0 and "+" not in text:
            value = -value

        subject = self._extract_subject(text)
        if not subject:
            return ("Не понял, за что менять цену. Пример: "
                    "«штраф за взрыв до -2000», «награда за орбиту 800».")
        ok, message = REWARD_BOOK.set_value(subject, value)
        if not ok:
            return message + "\n" + self._rules_hint()
        return message + "  (изменение действует немедленно)"

    def cmd_remove_rule(self, text: str) -> str:
        subject = self._extract_subject(text)
        ok, message = REWARD_BOOK.remove_rule(subject or text)
        return message

    def _extract_subject(self, text: str) -> str:
        """Вытаскивает предмет команды: «...за взрыв до -2000» -> «взрыв»."""
        lowered = text.lower().replace("ё", "е")
        match = re.search(r"\bза\s+(.+?)(?:\s+(?:до|на|в)\b|\s*[-+]?\d|$)", lowered)
        subject = match.group(1).strip() if match else ""
        for noise in ("штраф", "награду", "награда", "правило", "очков", "очка", "балл"):
            subject = subject.replace(noise, " ")
        subject = re.sub(r"\s+", " ", subject).strip(" .,:;")
        if not subject:
            words = [w for w in re.findall(r"[а-яa-z_]+", lowered)
                     if w not in ("повысь", "понизь", "измени", "поставь", "дай",
                                  "сделай", "штраф", "награда", "награду", "за",
                                  "до", "на", "очков", "правило", "убери", "выключи")]
            subject = " ".join(words[:2])
        return subject

    def _rules_hint(self) -> str:
        return "Доступные правила качества: " + ", ".join(
            f"{k} ({v[0]})" for k, v in RULE_CATALOG.items())

    # ------------------------------------------------------------------
    def cmd_design(self, _: str) -> str:
        design = self.dispatcher.design_only(DesignPolicy())
        lines = [f"РАСЧЁТ ПОД ЗАДАЧУ: {self.spec.describe()}",
                 f"Требуется ΔV: {design.budget.total:.0f} м/с "
                 f"(из {len(design.budget.legs)} участков)",
                 ""]
        for leg in design.budget.legs:
            lines.append(f"  {leg.name}: {leg.dv:.0f} м/с")
        lines.append("")
        for s in sorted(design.stages, key=lambda x: -x.requirement.index):
            lines.append(
                f"  Ступень {s.requirement.index}: {s.engine.title} + "
                f"{s.tank_count}×{s.tank.title} → ΔV {s.delta_v():.0f} м/с "
                f"(нужно {s.requirement.dv_required:.0f}), TWR {s.twr():.2f}")
        lines.append("")
        lines.append(f"ИТОГ: масса {design.total_mass:.2f} т | ΔV "
                     f"{design.total_delta_v:.0f} м/с | TWR старта "
                     f"{design.launch_twr:.2f} | деталей {design.part_count}")
        lines.append("Вердикт: " + ("проект жизнеспособен" if design.viable
                                    else "ОТКЛОНЁН — " + "; ".join(design.problems)))
        for w in design.warnings:
            lines.append(f"  замечание: {w}")
        lines.append(f"Полный отчёт с формулами: {ROOT / 'kia_design_report.md'}")
        return "\n".join(lines)

    def cmd_launch(self, _: str) -> str:
        def job():
            result = self.dispatcher.run_best()
            if result is None:
                print("\n[KIA] Полёт не состоялся: нет связи с KSP.")
            else:
                print(f"\n[KIA] Полёт завершён: {result.score:.0f} очков | "
                      f"достижения: {', '.join(result.record.milestones) or '—'}")
        return self.spawn("полёт", job)

    def cmd_train(self, text: str) -> str:
        numbers = re.findall(r"\d+", text)
        generations = int(numbers[0]) if numbers else 3
        def job():
            stats = self.dispatcher.train(generations=generations)
            print(f"\n[KIA] Обучение окончено. Лучший счёт: "
                  f"{stats.get('best_score', 0):.0f} за {stats.get('count', 0)} запусков")
        return self.spawn(f"обучение ({generations} поколений)", job)

    def cmd_autonomous(self, text: str) -> str:
        numbers = re.findall(r"\d+", text)
        limit = int(numbers[0]) if numbers else None
        self.loop = AutonomousLoop(self.dispatcher, max_iterations=limit)

        def job():
            summary = self.loop.run()
            print(f"\n[KIA] Автономный режим завершён: "
                  f"{summary['iterations']} итераций, "
                  f"лучший счёт {summary.get('best_score')}")

        started = self.spawn("автономный режим", job)
        return "\n".join([
            started,
            "ИИ работает сам: считает ракету, собирает, летит, разбирает отказ, "
            "правит чертёж и повторяет.",
            f"Отчёт по итерациям: {ROOT / 'kia_autonomy_report.md'}",
            "«разбор» — последний диагноз, «стоп» — остановить.",
        ])

    def cmd_diagnosis(self, _: str) -> str:
        loop = self.loop
        if loop is None or not loop.iterations:
            return "Разборов пока нет — запустите «старт» или «автономно»."
        it = loop.iterations[-1]
        lines = [f"Итерация {it.number}: {it.score:.0f} очков | "
                 f"{', '.join(it.failures) or ', '.join(it.milestones) or '—'}",
                 f"Аппарат: {it.design.get('mass_t')} т, ΔV {it.design.get('delta_v')}, "
                 f"TWR {it.design.get('twr')}, ступеней {it.design.get('stages')}"]
        for d in it.diagnoses:
            lines += [f"  ДИАГНОЗ: {d['title']}",
                      f"    основание: {d['evidence']}",
                      f"    решение:   {d['action']}"]
        if it.changes:
            lines.append("  ПРАВКИ В СЛЕДУЮЩУЮ ВЕРСИЮ:")
            for key, (before, after) in it.changes.items():
                lines.append(f"    {key}: {before} → {after}")
        summary = loop.summary()
        lines.append(f"Всего итераций: {summary['iterations']} | "
                     f"рекорд {summary.get('best_score')}")
        return "\n".join(lines)

    def cmd_stop(self, _: str) -> str:
        if self.loop is not None and self.loop.running:
            self.loop.stop()
            return "Автономный режим останавливается после текущей итерации."
        if not self.busy():
            return "Сейчас ничего не выполняется."
        self.dispatcher.abort()
        return "Команда на прерывание отправлена — полёт остановится на ближайшей проверке."

    def cmd_status(self, _: str) -> str:
        return self.dispatcher.status()

    def cmd_log(self, text: str) -> str:
        numbers = re.findall(r"\d+", text)
        count = int(numbers[0]) if numbers else 15
        tail = FLIGHT_LOG.tail(count)
        return tail + f"\n\nПолный журнал: {ROOT / 'kia_flight_log.md'}"

    def cmd_telemetry(self, _: str) -> str:
        d = self.dispatcher
        if not d.connect():
            return "Нет связи с KSP."
        try:
            from kia_core.environment import Telemetry
            with Telemetry(d.connection) as tm:
                s = tm.snapshot()
            return (f"{s.body} | {s.situation}\n"
                    f"  высота {s.altitude:,.0f} м (над рельефом {s.surface_altitude:,.0f})\n"
                    f"  Ap {s.apoapsis:,.0f} м | Pe {s.periapsis:,.0f} м | "
                    f"e={s.eccentricity:.3f}\n"
                    f"  скорость {s.speed:.1f} м/с | вертикальная {s.vertical_speed:+.1f}\n"
                    f"  масса {s.mass:.2f} т | тяга {s.thrust:.0f}/{s.available_thrust:.0f} кН"
                    f" | TWR {s.twr:.2f}\n"
                    f"  ступень {s.stage} | деталей {s.part_count} | экипаж {s.crew_count}\n"
                    f"  топливо: " + ", ".join(
                        f"{k} {v['amount']:.0f}/{v['max']:.0f}"
                        for k, v in s.resources.items() if v.get("max", 0) > 0))
        except Exception as exc:
            return f"Телеметрия недоступна: {exc}"

    def cmd_catalog(self, _: str) -> str:
        from kia_core.engineer.parts_catalog import get_catalog
        catalog = get_catalog()
        stats = catalog.stats()
        lines = [f"Каталог деталей: {stats['total']} шт из {stats['source']}",
                 f"  двигателей {stats['engines']} | баков {stats['tanks']} | "
                 f"разделителей {stats['decouplers']} | капсул {stats['pods']}",
                 f"  диаметры: {', '.join(f'{d:g} м' for d in catalog.diameters())}"]
        return "\n".join(lines)

    def cmd_connect(self, _: str) -> str:
        if self.dispatcher.connect():
            info = self.dispatcher.connection.info()
            return f"Связь есть. {info.describe()}"
        return ("Связи нет. Запустите KSP, откройте окно kRPC и нажмите Start Server "
                f"(ожидается {CONFIG.connection.address}:{CONFIG.connection.rpc_port}).")

    def cmd_exit(self, _: str) -> str:
        self.running = False
        if self.busy():
            self.dispatcher.abort()
        self.dispatcher.disconnect()
        return "Работа завершена."

    # ==================================================================
    # Разбор строки оператора
    # ==================================================================
    def dispatch(self, text: str) -> str:
        # ﻿ — BOM: прилетает первой строкой, когда команды подают через конвейер
        raw = text.replace("﻿", "").strip()
        if not raw:
            return ""
        lowered = raw.lower().replace("ё", "е")

        rules = [
            (r"^(помощь|help|\?|команды)", self.cmd_help),
            (r"^(выход|quit|exit|стоп работа)", self.cmd_exit),
            (r"^(награды|прейскурант|цены)", self.cmd_rewards),
            (r"^(сбрось|сброс|обнули).*(награ|цен|прейскурант)", self.cmd_reset_rewards),
            (r"^(убери|выключи|удали).*(правил|награ|штраф)", self.cmd_remove_rule),
            (r"(штраф|награ|очк|балл|правил|дай|начисл|повысь|понизь|снизь|увеличь)",
             self.cmd_set_reward),
            (r"^(проект|рассчитай|расчет|расчёт|спроектируй|конструируй)", self.cmd_design),
            (r"^(старт|запуск|лети сейчас|поехали|взлет|взлёт|полет|полёт)", self.cmd_launch),
            (r"^(обучение|учись|тренируй|эволюция)", self.cmd_train),
            (r"^(автономно|автономный|ночной режим|бесконечн|сам[аи]? работай|"
             r"работай сам)", self.cmd_autonomous),
            (r"^(разбор|диагноз|анализ|почему)", self.cmd_diagnosis),
            (r"^(стоп|отмена|прерви|прекрати)", self.cmd_stop),
            (r"^(статус|состояние|как дела)", self.cmd_status),
            (r"^(лог|журнал|что делаешь|что сейчас)", self.cmd_log),
            (r"^(телеметрия|показания|приборы)", self.cmd_telemetry),
            (r"^(каталог|детали|запчасти)", self.cmd_catalog),
            (r"^(связь|подключись|коннект|соединение)", self.cmd_connect),
            (r"^(задача|цель\??$|что за задача)", self.cmd_task_show),
            (r"(лети|летим|слетай|курс на|цель|орбит|посад|сядь|облет|облёт|"
             r"вернись|миссия)", self.cmd_set_task),
        ]
        for pattern, handler in rules:
            if re.search(pattern, lowered):
                try:
                    return handler(raw)
                except Exception as exc:
                    self.log.error("Ошибка команды: %s\n%s", exc, traceback.format_exc())
                    return f"Команда не выполнена: {exc}"
        return ("Не понял команду. Наберите «помощь».\n"
                "Примеры: «лети на Муну», «орбита 120 км», "
                "«штраф за взрыв -2000», «дай +500 за ровный шаг тангажа», «старт».")

    # ------------------------------------------------------------------
    def repl(self) -> int:
        # Команды, поданные через конвейер, приходят в кодировке системы —
        # приводим ввод к UTF-8, иначе кириллица превращается в мусор.
        if not sys.stdin.isatty():
            try:
                sys.stdin.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
        print(BANNER)
        print(f"Задача: {self.spec.title} — {self.spec.describe()}")
        print(f"Прейскурант: {len(REWARD_BOOK.milestones)} наград, "
              f"{len(REWARD_BOOK.failures)} штрафов, "
              f"{len(REWARD_BOOK.rules)} пользовательских правил")
        print(f"Журнал полёта: {ROOT / 'kia_flight_log.md'}")
        print(f"Отчёт конструктора: {ROOT / 'kia_design_report.md'}")
        print("-" * 68)

        while self.running:
            try:
                line = input("\nKIA> ")
            except (EOFError, KeyboardInterrupt):
                print()
                self.cmd_exit("")
                break
            answer = self.dispatch(line)
            if answer:
                print(answer)
        return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    console = Console()
    if argv:
        # неинтерактивный режим: python cli.py "лети на Муну"
        print(console.dispatch(" ".join(argv)))
        if console.busy():
            console.worker.join()
        return 0
    return console.repl()


if __name__ == "__main__":
    sys.exit(main())
