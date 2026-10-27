"""ДЕЖУРНЫЙ: короткая сводка о ходе обучения раз в несколько минут.

Зачем. Обучение сыплет в консоль поток технических строк, и по нему
невозможно понять главное — идёт всё как надо или уже час крутится
впустую. За одну ночь это стоило проекту 178 млн шагов: конвейер честно
писал «попытка 60», а вопрос «это нормально?» задать было некому.

Дежурный отвечает на него сам, тремя цветами:

    ЗЕЛЁНЫЙ  ШТАТНО     считает, эпизоды идут, орбиты набираются
    ЖЁЛТЫЙ   ВНИМАНИЕ   считает, но результата нет: орбит ноль, ось на
                        упоре, награда стоит, ступень не сдаётся
    КРАСНЫЙ  СЛОМАЛОСЬ  счёт замер, эпизоды не заканчиваются, ступень
                        не запустилась, веса перестали сохраняться

Жёлтый и красный называют ПРИЧИНУ и говорят, куда смотреть. Это не
украшение: все проверки взяты из тех разборов, которые уже случались, —
упор по оси (5.3i), нулевая доля орбит при растущей награде (5.3j),
замерший счёт.

Дежурный ничего не решает и ни во что не вмешивается: только читает
отчёт тренера и печатает строку. Ошибка внутри него не может уронить
обучение — весь цикл обёрнут и молчит при любой неожиданности.
"""
from __future__ import annotations

import math
import os
import threading
import time

from ..logging_setup import STATUS, get_logger

log = get_logger("neural.watchdog")

# Раз в сколько секунд подавать голос. Пять минут — достаточно редко,
# чтобы не мешать чтению лога, и достаточно часто, чтобы не уйти спать
# при застрявшем обучении.
DEFAULT_PERIOD = 300.0
# Первая сводка приходит раньше остальных: оператор должен сразу увидеть,
# что дежурный на месте, а не гадать пять минут.
FIRST_DELAY = 60.0

# --- пороги, при которых загорается жёлтый ---
NO_ORBIT_EPISODES = 400        # эпизодов на ступени без единой орбиты
AXIS_RAIL = 0.85               # |средняя команда| выше — ось на упоре
SLOW_SHARE = 0.5               # скорость упала ниже доли от лучшей
STALL_EPISODES = 300           # награда не растёт столько эпизодов
MANY_ATTEMPTS = 3              # столько заходов на одну ступень подряд

AXIS_NAMES = ("газ", "тангаж", "рыскание", "крен")


class Watchdog:
    """Смотрит на отчёт тренера со стороны и раз в N минут пишет сводку."""

    def __init__(self, trainer, pipeline=None, period: float | None = None):
        self.trainer = trainer
        self.pipeline = pipeline
        self.period = float(period if period is not None else _period_from_env())
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.started = 0.0
        # что было в прошлый раз — по этому и видно, движется ли счёт
        # Живость меряется по НАКОПИТЕЛЬНЫМ счётчикам мозга. Счётчик в
        # отчёте тренера обнуляется на каждом новом заходе на ступень, и
        # первая же версия дежурного на этом обожглась: в 12:55:03 было
        # 1 022 241 шаг, в 12:55:18 начался новый заход с 204 476 — и
        # дежурный объявил «счёт замер», хотя счёт шёл 14 тыс. шаг/с.
        self.last_alive = -1
        self.last_alive_episodes = -1
        self.last_steps = -1
        self.last_rollbacks = 0
        self.best_speed = 0.0
        self.ticks = 0

    # ------------------------------------------------------------------
    def start(self) -> bool:
        if self.thread is not None and self.thread.is_alive():
            return False
        if self.period <= 0:
            return False                     # выключен через переменную среды
        self.started = time.time()
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name="kia-watchdog")
        self.thread.start()
        return True

    def stop(self) -> None:
        self.stop_flag.set()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        if self.stop_flag.wait(min(FIRST_DELAY, self.period)):
            return
        while not self.stop_flag.is_set():
            try:
                self.report()
            except Exception as exc:          # дежурный не имеет права падать
                log.debug("Дежурный споткнулся: %s", exc)
            if self.stop_flag.wait(self.period):
                return

    def report(self) -> tuple[str, str]:
        """Собирает сводку и печатает её нужным цветом."""
        level, text = self.assess()
        self.ticks += 1
        if level == "red":
            log.error("%s", text)
        elif level == "yellow":
            log.warning("%s", text)
        else:
            log.log(STATUS, "%s", text)
        return level, text

    # ==================================================================
    def snapshot(self) -> dict:
        """Всё, что известно о ходе прямо сейчас. Только чтение."""
        report = getattr(self.trainer, "last_report", None)
        exam = getattr(self.pipeline, "exam", None) if self.pipeline else None
        # накопительные счётчики: растут всю жизнь мозга и не сбрасываются
        # между заходами — по ним и видно, жив ли счёт
        stats = getattr(getattr(self.trainer, "brain", None), "pilot", None)
        stats = getattr(stats, "stats", None)
        data = {
            "alive_steps": int(getattr(stats, "total_steps", 0) or 0),
            "alive_episodes": int(getattr(stats, "episodes", 0) or 0),
            "error": getattr(report, "error", "") if report else "",
            "title": getattr(report, "mode", "") if report else "",
            "stage": getattr(report, "stage", 0) if report else 0,
            "steps": int(getattr(report, "steps", 0) or 0) if report else 0,
            "episodes": int(getattr(report, "episodes", 0) or 0) if report else 0,
            "orbits": int(getattr(report, "successes", 0) or 0) if report else 0,
            "mean": float(getattr(report, "mean_return", 0.0) or 0.0) if report else 0.0,
            "difficulty": float(getattr(report, "difficulty", 1.0) or 1.0) if report else 1.0,
            "axes": list(getattr(report, "axis_means", []) or []) if report else [],
            "failures": dict(getattr(report, "failures", {}) or {}) if report else {},
            "wall": time.time() - (self.started or time.time()),
        }
        if exam is not None:
            data.update({
                "exam_episodes": exam.episodes,
                "exam_orbits": exam.orbits,
                "exam_rate": exam.orbit_rate,
                "exam_recent": sum(exam.recent_orbits),
                "exam_mean": exam.mean_return,
                "exam_stalled": exam.stalled,
                "attempts": getattr(self.pipeline, "attempts", 0),
                "rollbacks": getattr(self.pipeline, "rollbacks", 0),
                "finished": getattr(self.pipeline, "finished", False),
            })
        return data

    # ------------------------------------------------------------------
    def assess(self) -> tuple[str, str]:
        """Вердикт и текст. Разбито по причинам, а не по ощущению."""
        now = self.snapshot()
        moment = time.strftime("%H:%M")
        stage = now["stage"] or getattr(self.trainer.brain, "stage", 0)
        title = now["title"] or "обучение"
        head = f"[{moment}] ступень {stage} ({title})"

        # --- живость: по накопительным счётчикам, они не обнуляются ---
        alive = now["alive_steps"]
        first = self.last_alive < 0
        grew = alive - self.last_alive if not first else alive
        episodes_grew = (now["alive_episodes"] - self.last_alive_episodes
                         if self.last_alive_episodes >= 0
                         else now["alive_episodes"])

        # --- скорость: по счётчику захода, но только если он не сбросился ---
        steps = now["steps"]
        step_delta = steps - self.last_steps if self.last_steps >= 0 else 0
        speed = (step_delta / self.period
                 if step_delta > 0 and self.period > 0 else 0.0)
        if speed > self.best_speed:
            self.best_speed = speed

        # запоминаем ДО ранних выходов, иначе следующая сводка соврёт
        self.last_alive = alive
        self.last_alive_episodes = now["alive_episodes"]
        self.last_steps = steps
        rollback_now = now.get("rollbacks", 0) - self.last_rollbacks
        self.last_rollbacks = now.get("rollbacks", 0)

        # ================= КРАСНЫЙ: сломалось =========================
        if now["error"]:
            return "red", f"СЛОМАЛОСЬ · {head} · ступень не запустилась: {now['error']}"
        if not math.isfinite(now["mean"]):
            return "red", (f"СЛОМАЛОСЬ · {head} · средняя награда стала не "
                           f"числом — расчёт разошёлся, обучение надо "
                           f"остановить (пульт → 7 → 10)")
        if now.get("finished"):
            return "green", (f"ЗАВЕРШЕНО · {head} · программа пройдена "
                             f"целиком, финальная модель записана")
        if not first and grew <= 0:
            return "red", (f"СЛОМАЛОСЬ · {head} · счёт замер: за "
                           f"{self.period / 60:.0f} мин решений не прибавилось "
                           f"(стоит на {alive:,}). Проверь, жив ли процесс"
                           .replace(",", " "))
        if not first and episodes_grew <= 0 and grew > 0:
            return "red", (f"СЛОМАЛОСЬ · {head} · решения идут, а эпизоды не "
                           f"заканчиваются: {now['alive_episodes']} как было. "
                           f"Похоже, эпизод не может завершиться")
        # Проверки «веса давно не сохранялись» здесь намеренно НЕТ. Она
        # выглядела разумной, но ловила пустоту: если запись на диск не
        # проходит, тренер падает с ошибкой и счёт останавливается — а это
        # уже поймано строкой выше. Зато сама она краснела всякий раз,
        # когда обучение просто не запущено. Ложная тревога стоит доверия
        # ко всем остальным, поэтому правило удалено, а не смягчено.

        # ================= ЖЁЛТЫЙ: тупит =============================
        rate = now.get("exam_rate")
        exam_episodes = now.get("exam_episodes", 0)
        recent = now.get("exam_recent", None)
        troubles = []

        if exam_episodes >= NO_ORBIT_EPISODES and recent == 0:
            troubles.append(f"орбит нет за последние {exam_episodes} эпизодов")
        elif rate is not None and rate == 0.0 and exam_episodes >= NO_ORBIT_EPISODES:
            troubles.append(f"доля орбит за окно — ноль ({exam_episodes} эпизодов)")

        rail = [f"{AXIS_NAMES[i]} {value:+.2f}"
                for i, value in enumerate(now["axes"][:len(AXIS_NAMES)])
                if abs(value) >= AXIS_RAIL]
        if rail:
            troubles.append("ось на упоре: " + ", ".join(rail))

        if now.get("exam_stalled", 0) >= STALL_EPISODES:
            troubles.append(f"награда не растёт {now['exam_stalled']} эпизодов")
        if now.get("attempts", 0) >= MANY_ATTEMPTS:
            troubles.append(f"ступень не сдаётся, заход {now['attempts'] + 1}")
        if rollback_now > 0:
            troubles.append(f"был откат на ступень назад ({now['rollbacks']}-й)")
        if (not first and self.best_speed > 0
                and speed < self.best_speed * SLOW_SHARE):
            troubles.append(f"счёт замедлился: {speed:.0f} против "
                            f"{self.best_speed:.0f} шаг/с")

        if troubles:
            top = self._top_failure(now)
            tail = f" · чаще всего срывалось: {top}" if top else ""
            return "yellow", (f"ВНИМАНИЕ · {head} · " + " · ".join(troubles)
                              + tail + "\n         куда смотреть: MIGRATION "
                              "5.6 — прогон без шума (kia_neural.py eval -s "
                              f"{stage}), средние команды по осям, причины срыва")

        # ================= ЗЕЛЁНЫЙ: штатно ===========================
        parts = [f"ШТАТНО · {head}"]
        if rate is not None:
            parts.append(f"орбит {now['exam_orbits']} ({rate:.0%} за окно)")
        elif now["orbits"]:
            parts.append(f"орбит {now['orbits']}")
        parts.append(f"эпизодов {now['episodes']}")
        parts.append(f"средняя {now['mean']:+.1f}")
        if speed > 0:
            parts.append(f"{speed / 1000:.1f} тыс. шаг/с")
        if now["difficulty"] < 0.999:
            parts.append(f"жёсткость {now['difficulty']:.0%}")
        parts.append(f"в работе {now['wall'] / 60:.0f} мин")
        return "green", " · ".join(parts)

    # ------------------------------------------------------------------
    @staticmethod
    def _top_failure(now: dict) -> str:
        failures = now.get("failures") or {}
        if not failures:
            return ""
        reason, count = max(failures.items(), key=lambda item: item[1])
        return f"{reason} ×{count}"


# ==========================================================================
def _period_from_env() -> float:
    """Период сводок. KIA_WATCHDOG=0 выключает дежурного совсем."""
    raw = os.environ.get("KIA_WATCHDOG")
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            return DEFAULT_PERIOD
        if value <= 0:
            return 0.0
        # маленькие числа удобно трактовать как минуты, большие — как секунды
        return value * 60.0 if value <= 60 else value
    return DEFAULT_PERIOD


__all__ = ["Watchdog", "DEFAULT_PERIOD", "FIRST_DELAY", "AXIS_RAIL",
           "NO_ORBIT_EPISODES"]
