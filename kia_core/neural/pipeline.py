"""АВТОНОМНЫЙ КОНВЕЙЕР ЭВОЛЮЦИИ KIA.

Проводит нейросеть по всей учебной программе без человека:

    Базовый Тренажёр -> Тренажёр+ -> Тренажёр++ -> готовая модель

ЖИВАЯ ИГРА В ЭТОТ ЦИКЛ НЕ ВХОДИТ. Конвейер работает строго внутри трёх
симуляторов и ни при каких условиях не запускает KSP: обучение может
идти ночью, при закрытой игре, и не может испортить ни сохранение, ни
стартовый стол. Дообучение в KSP остаётся отдельным ручным пунктом
пульта.

УСТРОЙСТВО

Конвейер — это надстройка над тренером, а не второй тренер. Он смотрит
на каждый закончившийся эпизод и принимает одно из трёх решений:

* ЭКЗАМЕН СДАН — за скользящее окно в 50 эпизодов средняя награда выше
  +20 И на ступени набрано 100 успешных орбит. Веса уходят в архивный
  чекпоинт `kia_model_stageN_complete.pth`, конвейер переводит сеть на
  следующую ступень. После третьей ступени пишется `kia_model_final.pth`
  и работа заканчивается.

* ТУПИК — средняя награда за окно ниже −40 и не растёт 300 эпизодов
  подряд. Это policy collapse: политика развалилась и сама уже не
  выберется. Конвейер стирает испорченные веса, поднимает архив
  предыдущей ступени, откатывает сеть на ступень назад ровно на 200 000
  шагов среды и возвращает её обратно — уже с вдвое меньшей скоростью
  обучения. Каждая новая неудача на той же ступени снижает её ещё вдвое.

* ПРОДОЛЖАЕМ — ничего не произошло, обучение идёт дальше.

Каждое изменение уровня попадает в `kia_evolution_manifest.txt` с
отметкой времени и сопровождается звуковым сигналом: оператор слышит
прогресс, даже отойдя от экрана.

ПОЧЕМУ ИМЕННО ТАК

Порог «100 орбит» намеренно считается по ступени, а не по окну: окно
показывает, что сеть летает СЕЙЧАС стабильно, счётчик орбит — что она
делала это достаточно много раз, а не поймала удачную серию. Оба
условия обязательны, иначе экзамен сдаётся случайностью.

Откат именно на ступень назад, а не «продолжить как есть», потому что
провалившаяся ступень уже показала, что сеть к ней не готова: ей нужно
не больше попыток в том же месте, а закрепление предыдущего материала.
"""
from __future__ import annotations

import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ROOT
from ..logging_setup import get_logger
from . import curriculum as program

log = get_logger("neural.pipeline")

# Доля эпизодов с отказом на ступени «Аварии» — нужна, чтобы
# прикинуть, сколько всего полётов прогнать ради нужного числа
# аварий (см. neural/failures.py).
from .failures import FAILURE_RATE as FAILURE_SHARE
# Доля полётов с экипажем на борту (см. sim_env_plus.roll_failure).
CREW_SHARE = 0.5

MANIFEST_FILE = ROOT / "kia_evolution_manifest.txt"
FINAL_MODEL_FILE = ROOT / "kia_model_final.pth"

ALARM_BANNER = ("=== ДЕГРАДАЦИЯ НАВЫКОВ. СИСТЕМА ОПРЕДЕЛИЛА ТУПИК. "
                "ДЕЛАЮ CD-ROLLBACK ===")


def complete_file(stage_number: int) -> Path:
    """Архив весов, сдавших экзамен ступени."""
    return ROOT / f"kia_model_stage{int(stage_number)}_complete.pth"


# ==========================================================================
@dataclass
class PipelineConfig:
    """Правила перевода и отката. Меняются одной строкой."""

    # --- экзамен ---
    window: int = 50                    # эпизодов в скользящем окне
    promote_reward: float = 20.0        # средняя награда за окно
    promote_orbits: int = 100           # успешных орбит на ступени
    # Второй путь к экзамену — ДОЛЯ ОРБИТ в окне.
    #
    # Средняя награда несравнима между ступенями: каждая следующая
    # добавляет штрафы (скольжение, батарея, опрокидывание), и хвост
    # неудачных эпизодов тянет среднюю вниз даже у хорошей политики.
    # Замер на живом прогоне: 1082 орбиты за 3029 эпизодов (36 % успеха),
    # а средняя болтается от −17 до +13 и порога +20 не берёт никогда.
    # Доля орбит от ступени не зависит и говорит ровно то, что нужно:
    # сеть выводит аппарат на орбиту стабильно, а не поймала серию.
    #
    # Пропуском наверх это не делает: за обоими путями всё равно стоит
    # проверка БЕЗ ШУМА, и она решает.
    promote_rate: float = 0.35          # доля орбит в окне

    # --- подтверждение экзамена без шума ---
    # Скользящее окно считает награду ПО ОБУЧАЮЩИМ эпизодам, а в них
    # действия берутся из распределения — со всем разбросом. Политика
    # может набрать орбиты одним лишь шумом: именно так и вышло на
    # прогоне 615 обновлений, где сеть «сдала» две ступени, а
    # детерминированно летела строго вверх. Поэтому перед переводом
    # обязателен прогон БЕЗ шума.
    # Десяти эпизодов мало, и это не придирка, а арифметика. При пороге
    # «3 орбиты из 10» политика с истинной вероятностью 15 % проходит
    # проверку примерно в одном случае из пяти. А конвейер пересдаёт
    # экзамен снова и снова — значит, рано или поздно он проходит его по
    # удаче. Так и вышло: ступень 4 была объявлена сданной с 3/10, а
    # независимая проверка на двадцати эпизодах дала 1/20, то есть 5 %.
    #
    # Двадцать эпизодов с порогом 30 % разделяют куда честнее:
    # политика на 15 % проходит их примерно в одном случае из двадцати, а
    # политика на 35 % — почти всегда.
    confirm_episodes: int = 20          # эпизодов детерминированной проверки
    confirm_orbits: int = 6             # 30 % — столько же, сколько на окне
    confirm_cooldown: int = 300         # эпизодов до следующей попытки

    # --- тупик ---
    collapse_reward: float = -40.0      # средняя награда за окно
    collapse_patience: int = 300        # эпизодов без роста подряд
    # Застой — это не обвал. Награда может держаться на −15 и никуда не
    # двигаться: политика не разваливается, но и не учится. За такое
    # конвейер раньше не цеплялся и ходил кругами по 3 млн шагов.
    stagnation_episodes: int = 1500     # эпизодов на ступени без результата
    stagnation_orbits: int = 5          # меньше этого числа орбит = застой
    rollback_steps: int = 200_000       # шагов среды на дообучение
    lr_penalty: float = 0.5             # во сколько раз резать LR
    # Сколько раз подряд откатываться с одной и той же ступени, прежде чем
    # позвать человека. Откат лечит ОДИН сорт беды — политика испортилась и
    # ей надо вспомнить прошлый материал. Если это не помогло трижды,
    # причина другая (мёртвая ось, непроходимая механика, слишком много
    # нового за раз), и её надо искать по разделу 5.6 MIGRATION, а не
    # добивать скорость обучения до пола.
    max_rollbacks_per_stage: int = 3

    # --- границы работы ---
    # Предохранитель на весь конвейер. Считать надо от СКОРОСТИ, а не от
    # ощущения «много»: после ускорения до ~16 тыс. шаг/с прежние 60 млн
    # сгорали за два часа, и ночной прогон вставал сам собой. 400 млн при
    # средних 8 тыс. шаг/с — это около четырнадцати часов работы.
    max_steps: int = 400_000_000
    beep: bool = True

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "promote_reward": self.promote_reward,
            "promote_orbits": self.promote_orbits,
            "promote_rate": self.promote_rate,
            "collapse_reward": self.collapse_reward,
            "collapse_patience": self.collapse_patience,
            "stagnation_episodes": self.stagnation_episodes,
            "stagnation_orbits": self.stagnation_orbits,
            "max_rollbacks_per_stage": self.max_rollbacks_per_stage,
            "confirm_episodes": self.confirm_episodes,
            "confirm_orbits": self.confirm_orbits,
            "rollback_steps": self.rollback_steps,
            "lr_penalty": self.lr_penalty,
            "max_steps": self.max_steps,
        }


# ==========================================================================
class StageExam:
    """Приёмная комиссия одной попытки: считает эпизоды и выносит вердикт.

    Живёт ровно столько, сколько длится попытка сдать ступень. Счётчик
    орбит НЕ обнуляется между заходами тренера внутри одной попытки —
    иначе 100 орбит не набрались бы никогда.
    """

    def __init__(self, config: PipelineConfig, watch_collapse: bool = True):
        self.config = config
        self.watch_collapse = watch_collapse
        self.window: deque = deque(maxlen=config.window)
        self.window_orbits: deque = deque(maxlen=config.window)
        # Орбиты за ПОСЛЕДНИЕ stagnation_episodes эпизодов. Отдельно от
        # накопительного счётчика: застой — это «нет результата СЕЙЧАС», а
        # накопленные орбиты только растут и через сотню эпизодов
        # выключали проверку навсегда. Реальный случай: 76 орбит набрано в
        # начале ночи, следующие 120 000 эпизодов — ни одной, а тупик так
        # и не объявился, потому что 76 не меньше пяти.
        self.recent_orbits: deque = deque(maxlen=max(1, config.stagnation_episodes))
        self.orbit_rate = 0.0
        self.orbits = 0
        self.episodes = 0
        self.best_mean = float("-inf")
        self.stalled = 0
        self.verdict = ""
        self.mean_return = 0.0
        self.cooldown = 0            # эпизодов до следующей попытки экзамена
        self.rejections = 0          # сколько раз проверка без шума провалилась

    # ------------------------------------------------------------------
    def reject_promotion(self, cooldown: int) -> None:
        """Экзамен не подтвердился без шума — учимся дальше.

        Пауза растёт с каждым отказом. Причина в живом наблюдении: когда
        орбиты в обучении делает разброс действий, условие экзамена по
        обучающим эпизодам выполняется СНОВА через те же триста эпизодов —
        и конвейер начинает гонять проверку каждую минуту, отдавая ей
        пятую часть времени счёта. Один отказ — случайность, шестой подряд
        означает, что политике нужно не переспросить, а поучиться.
        """
        self.rejections += 1
        self.verdict = ""
        self.cooldown = max(1, int(cooldown) * min(8, self.rejections))

    # ------------------------------------------------------------------
    def observe(self, episode_return: float, info: dict, report) -> str:
        """Хук тренера: вызывается после каждого эпизода.

        Возвращает непустую строку, если обучение пора остановить.
        """
        self.episodes += 1
        self.window.append(float(episode_return))
        reached = bool(info.get("orbit"))
        self.window_orbits.append(reached)
        self.recent_orbits.append(reached)
        if reached:
            self.orbits += 1

        if self.cooldown > 0:
            self.cooldown -= 1
        if len(self.window) < self.window.maxlen:
            return ""                    # окно ещё не набрано, судить рано
        self.mean_return = sum(self.window) / len(self.window)

        # --- экзамен ---
        self.orbit_rate = (sum(1 for value in self.window_orbits if value)
                           / len(self.window_orbits)
                           if self.window_orbits else 0.0)
        stable = (self.mean_return > self.config.promote_reward
                  or self.orbit_rate >= self.config.promote_rate)
        if (self.cooldown <= 0 and stable
                and self.orbits >= self.config.promote_orbits):
            self.verdict = "promote"
            return "экзамен сдан"

        # --- тупик ---
        # «Не растёт» считается по лучшему среднему за окно: пока политика
        # хоть иногда улучшается, тупиком это не считается, каким бы низким
        # ни было текущее значение.
        if self.mean_return > self.best_mean + 1e-9:
            self.best_mean = self.mean_return
            self.stalled = 0
        else:
            self.stalled += 1

        if (self.watch_collapse
                and self.mean_return < self.config.collapse_reward
                and self.stalled >= self.config.collapse_patience):
            self.verdict = "collapse"
            return "политика в тупике"

        # --- застой ---
        # Награда может стоять на месте, не проваливаясь: сеть не рушится,
        # но и не движется. Формально это не обвал, а по сути тот же тупик,
        # только тихий — и без этой проверки конвейер ходит кругами часами.
        #
        # Считать надо орбиты ЗА ПОСЛЕДНЕЕ ОКНО, а не за всю попытку.
        # С накопительным счётчиком проверка выключалась навсегда после
        # первых же пяти случайных орбит: реальный прогон 178 млн шагов
        # простоял 60 заходов при «орбит 76/100 (0 % за окно)» и ни разу
        # не объявил тупик.
        #
        # Условия «награда не растёт» здесь намеренно НЕТ. Средняя награда
        # на ступени 3 гуляет от +3 до +13 сама по себе, и любой всплеск
        # обнулял бы терпение: при 0 % орбит за окно это не признак жизни,
        # а шум. Полторы тысячи эпизодов почти без орбит — приговор
        # независимо от того, как в это время колебалась награда.
        if (self.watch_collapse
                and self.episodes >= self.config.stagnation_episodes
                and sum(self.recent_orbits) < self.config.stagnation_orbits):
            self.verdict = "collapse"
            return "застой: ступень не поддаётся"
        return ""

    def describe(self) -> str:
        text = (f"эпизодов {self.episodes}, орбит {self.orbits}/"
                f"{self.config.promote_orbits} ({self.orbit_rate:.0%} за "
                f"окно, {sum(self.recent_orbits)} за последние "
                f"{len(self.recent_orbits)}), средняя за {len(self.window)} = "
                f"{self.mean_return:+.1f}")
        if self.rejections:
            text += f", проверок без шума провалено {self.rejections}"
        return text


# ==========================================================================
@dataclass
class PipelineEvent:
    """Одна запись в манифесте эволюции."""
    kind: str
    stage: int
    text: str
    at: float = field(default_factory=time.time)

    def line(self) -> str:
        return f"[{time.strftime('%H:%M', time.localtime(self.at))}] {self.text}"


class EvolutionPipeline:
    """Полный автономный цикл обучения внутри трёх тренажёров."""

    def __init__(self, trainer=None, config: PipelineConfig | None = None,
                 goal_altitude: float = 80_000.0, start_stage: int | None = None):
        from .trainer import NeuralTrainer
        self.config = config or PipelineConfig()
        self.trainer = trainer or NeuralTrainer()
        self.goal_altitude = goal_altitude
        # С какой ступени начинать. По умолчанию — с той, что записана в
        # весах. Задаётся явно, когда политику нужно переучить снизу: так
        # было после ловушки разброса действий, когда веса числились на
        # третьей ступени, а летать разучились совсем.
        self.start_stage = start_stage
        self.stop_flag = threading.Event()
        self.events: list[PipelineEvent] = []
        self.started = 0.0
        self.total_steps = 0
        self.promotions = 0
        self.rollbacks = 0
        self.finished = False
        self.exam: StageExam | None = None
        self.stage_number = 1
        self.attempts = 0
        # Сколько раз подряд откатывались с каждой ступени. Обнуляется
        # успешным переводом наверх: значит, лечение сработало.
        self.stage_rollbacks: dict[int, int] = {}

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return not self.stop_flag.is_set()

    def stop(self) -> None:
        """Останавливает и конвейер, и текущее обучение."""
        self.stop_flag.set()
        self.trainer.stop()

    # ------------------------------------------------------------------
    # Манифест и сигналы
    # ------------------------------------------------------------------
    def record(self, kind: str, stage: int, text: str, beep: bool = True) -> None:
        event = PipelineEvent(kind=kind, stage=stage, text=text)
        self.events.append(event)
        try:
            with MANIFEST_FILE.open("a", encoding="utf-8") as handle:
                handle.write(event.line() + "\n")
        except OSError as exc:
            log.warning("Манифест не пишется: %s", exc)
        log.info("[конвейер] %s", text)
        print(f"\n  {event.line()}")
        if beep and self.config.beep:
            # Короткий сигнал спикера: слышно, что уровень сменился,
            # даже если оператор отошёл от экрана
            print("\a", end="", flush=True)

    def open_manifest(self) -> None:
        header = (f"\n{'=' * 70}\n"
                  f"  АВТОНОМНАЯ ЭВОЛЮЦИЯ KIA — запуск "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"  экзамен: средняя > {self.config.promote_reward:+.1f} за "
                  f"{self.config.window} эпизодов и "
                  f"{self.config.promote_orbits} орбит\n"
                  f"  тупик:   средняя < {self.config.collapse_reward:+.1f} и "
                  f"без роста {self.config.collapse_patience} эпизодов\n"
                  f"  живая KSP в цикл не входит\n"
                  f"{'=' * 70}\n")
        try:
            with MANIFEST_FILE.open("a", encoding="utf-8") as handle:
                handle.write(header)
        except OSError as exc:
            log.warning("Манифест не создаётся: %s", exc)

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Гоняет программу до финальной модели или до остановки оператором."""
        self.stop_flag.clear()
        # дежурному нужен доступ к экзамену: доля орбит и застой видны
        # только отсюда
        self.trainer.begin_session(self)
        self.started = time.time()
        self.finished = False
        self.open_manifest()

        requested = self.start_stage or self.trainer.brain.stage or 1
        self.stage_number = max(1, min(len(program.STAGES), int(requested)))
        if self.start_stage and self.start_stage != self.trainer.brain.stage:
            # веса остаются те же, меняется только «класс», в котором
            # они учатся: prepare увидит совпадение и ничего не подменит
            self.trainer.brain.stage = self.stage_number
            self.trainer.brain.best_mean = float("-inf")
        self.record("start", self.stage_number,
                    f"Старт конвейера со ступени {self.stage_number} "
                    f"({program.stage_by_number(self.stage_number).title})",
                    beep=False)

        while self.running and self.total_steps < self.config.max_steps:
            stage = program.stage_by_number(self.stage_number)
            verdict = self.attempt(stage)

            if verdict == "promote":
                if not self.promote(stage):
                    break
            elif verdict == "collapse":
                if not self.rollback(stage):
                    break
            elif verdict == "stopped":
                break
            # "budget" — ступень не сдана за отведённые шаги, идём по кругу

        if not self.finished:
            self.record("stop", self.stage_number,
                        f"Конвейер остановлен на ступени {self.stage_number}. "
                        f"{self.progress_line()}", beep=False)
        if getattr(self.trainer, "watchdog", None) is not None:
            self.trainer.watchdog.stop()
        return self.summary()

    # ------------------------------------------------------------------
    def attempt(self, stage) -> str:
        """Одна попытка сдать ступень. Возвращает вердикт."""
        # Застой ловим ВЕЗДЕ, включая первую ступень. Откатываться с неё
        # действительно некуда — но это довод позвать человека, а не
        # довод молчать: без этой проверки конвейер честно ходил кругами
        # всю ночь и к утру выдавал «попытка 60» вместо тревоги.
        self.exam = StageExam(self.config, watch_collapse=True)
        self.attempts = 0
        log.info("[конвейер] ступень %d (%s): попытка сдать экзамен",
                 stage.number, stage.title)

        while self.running and self.total_steps < self.config.max_steps:
            # Разогрев жёсткости — это ввод НОВЫХ механик ступени, и
            # проходится он один раз. На повторном заходе сеть их уже
            # видела: сбрасывать среду обратно в щадящий режим значит
            # заставлять политику гоняться за качающимся миром. Именно
            # так и вышло на прогоне 178 млн шагов — 60 заходов, и в
            # каждом жёсткость заново ползла с нуля до единицы.
            report = self.trainer.train_stage(
                stage, total_steps=stage.default_steps,
                goal_altitude=self.goal_altitude,
                on_episode=self.exam.observe,
                difficulty_floor=1.0 if self.attempts else 0.0)
            self.total_steps += report.steps

            if report.error:
                self.record("error", stage.number,
                            f"Ступень {stage.number} не запустилась: "
                            f"{report.error}", beep=False)
                return "stopped"
            if self.exam.verdict == "promote" and not self.confirm(stage):
                continue                 # проверка без шума провалена, учимся
            if self.exam.verdict:
                return self.exam.verdict
            if not self.running:
                return "stopped"

            # Бюджет захода исчерпан, экзамен не сдан — заходим ещё раз.
            # Это не ошибка: ступень просто требует больше шагов, чем
            # заложено по умолчанию. Но в хронике это должно быть видно,
            # иначе после ночи работы непонятно, крутился конвейер
            # впустую или честно учился.
            self.attempts += 1
            self.record("budget", stage.number,
                        f"Ступень {stage.number} не сдана за "
                        f"{report.steps} шагов ({self.exam.describe()}). "
                        f"Захожу заново, попытка {self.attempts + 1}",
                        beep=False)
        return "stopped" if not self.running else "budget"

    # ------------------------------------------------------------------
    def confirm(self, stage) -> bool:
        """Проверка политики БЕЗ шума перед переводом на следующую ступень.

        Скользящее окно считает награду по обучающим эпизодам, где действия
        берутся из распределения. Политика может набрать сотню орбит одним
        разбросом, оставаясь при этом нерабочей: детерминированно она пойдёт
        строго вверх. Такое уже случилось — две ступени были «сданы» именно
        так. Поэтому перевод подтверждается прогоном с выключенным шумом.
        """
        config = self.config
        if config.confirm_episodes <= 0:
            return True
        result = self.trainer.evaluate_pilot(
            episodes=config.confirm_episodes, goal_altitude=self.goal_altitude,
            stage=stage)
        orbits = int(result.get("orbits", 0))
        if orbits >= config.confirm_orbits:
            if not self.confirm_crew(stage):
                return False
            self.record("confirm", stage.number,
                        f"Проверка без шума: {orbits}/"
                        f"{config.confirm_episodes} орбит, награда "
                        f"{result.get('mean_return', 0.0):+.1f} — экзамен "
                        f"засчитан", beep=False)
            return True
        self.exam.reject_promotion(config.confirm_cooldown)
        self.record("reject", stage.number,
                    f"Экзамен НЕ засчитан: без шума только {orbits}/"
                    f"{config.confirm_episodes} орбит (нужно "
                    f"{config.confirm_orbits}), награда "
                    f"{result.get('mean_return', 0.0):+.1f}. Орбиты делал "
                    f"разброс действий, а не политика — учимся дальше",
                    beep=False)
        return False

    def confirm_crew(self, stage) -> bool:
        """Второй вопрос ступени «Аварии»: а экипаж-то живой?

        Орбиты на этой ступени набираются теми полётами, где ничего не
        сломалось, — их две трети. Поэтому орбитного экзамена мало: он
        один раз уже пропустил политику, у которой средняя награда была
        −5.22, то есть люди гибли. Здесь спрашивается прямо: из аварий,
        которые действительно случились, сколько экипажей вернулось.
        """
        exam = getattr(stage, "crew_exam", None)
        if not exam:
            return True
        need_faced, need_rate, need_flown = exam
        # Нужная авария — это отказ И экипаж на борту. Доли берём у самой
        # ступени, а не из констант модуля: ступень их переопределяет, и
        # разъехавшаяся оценка выборки стоила бы завала на пустом месте.
        options = dict(stage.env_options)
        share = (options.get("failure_rate", FAILURE_SHARE)
                 * options.get("crew_share", CREW_SHARE))
        share = max(0.05, share)
        result = self.trainer.evaluate_crew(
            stage, episodes=int(need_faced / share * 1.3))
        faced, rate = result["faced"], result["rate"]
        flown = result["flown_rate"]
        if faced >= need_faced and rate >= need_rate and flown >= need_flown:
            self.record("confirm", stage.number,
                        f"Аварийный экзамен: спасено {result['saved']} из "
                        f"{faced} экипажей ({rate:.0%}), доведено до орбиты "
                        f"{result['flown']} из {result['flyable']} "
                        f"поправимых ({flown:.0%}) — засчитано",
                        beep=False)
            return True
        self.exam.reject_promotion(self.config.confirm_cooldown)
        self.record("reject", stage.number,
                    f"Аварийный экзамен НЕ сдан: спасено "
                    f"{result['saved']} из {faced} экипажей ({rate:.0%}, "
                    f"нужно {need_rate:.0%} и не меньше {need_faced} "
                    f"аварий), доведено до орбиты {result['flown']} из "
                    f"{result['flyable']} поправимых ({flown:.0%}, нужно "
                    f"{need_flown:.0%}) — учимся дальше", beep=False)
        return False

    # ------------------------------------------------------------------
    def promote(self, stage) -> bool:
        """Экзамен сдан: архивируем веса и переводим на следующую ступень."""
        archive = complete_file(stage.number)
        self.trainer.brain.save(archive)
        self.promotions += 1
        self.stage_rollbacks.pop(stage.number, None)   # лечение сработало
        exam = self.exam

        if stage.number >= len(program.STAGES):
            self.trainer.brain.save(FINAL_MODEL_FILE)
            self.finished = True
            self.record(
                "final", stage.number,
                f"ЭВОЛЮЦИЯ ЗАВЕРШЕНА. Ступень {stage.number} "
                f"({stage.title}) сдана: орбит {exam.orbits}, награда "
                f"{exam.mean_return:+.1f}. Финальная модель: "
                f"{FINAL_MODEL_FILE.name}")
            return False

        following = program.stage_by_number(stage.number + 1)
        self.record(
            "promote", following.number,
            f"Успешный переход на Ступень {following.number}. "
            f"Орбит: {exam.orbits}, Награда: {exam.mean_return:+.1f}")

        # Дальше едут именно те веса, которые сдали экзамен. Штатный
        # `curriculum.prepare` копирует «лучшее среднее», а это не одно и
        # то же: лучшее среднее могло быть снято раньше, до того как сеть
        # набрала сотню орбит. Поэтому снимок ступени делаем из архива.
        if following.backup is not None and not following.backup.exists():
            shutil.copy2(archive, following.backup)
            log.info("[конвейер] снимок ступени: %s -> %s", archive.name,
                     following.backup.name)
        self.trainer.brain.stage = following.number
        self.trainer.brain.best_mean = float("-inf")
        self.trainer.brain.save()
        self.stage_number = following.number
        return True

    # ------------------------------------------------------------------
    def rollback(self, stage) -> bool:
        """Тупик: откат на ступень назад с дообучением и снижением LR."""
        from .ppo import scale_stage_learning_rate

        exam = self.exam
        print(f"\n  {ALARM_BANNER}")
        if self.config.beep:
            print("\a", end="", flush=True)
        self.rollbacks += 1
        done = self.stage_rollbacks.get(stage.number, 0) + 1
        self.stage_rollbacks[stage.number] = done

        # С первой ступени откатываться некуда: предыдущего материала не
        # существует. Значит, дело не в том, что политика забыла старое, —
        # ступень не берётся сама по себе, и лечится это не конвейером.
        if stage.number <= 1:
            self.record(
                "give_up", stage.number,
                f"Базовая ступень не берётся: {exam.describe()}. "
                f"Откатываться некуда — нужен разбор по разделу 5.6 "
                f"MIGRATION: прогон без шума (eval -n 10 -s 1), средние "
                f"команды по осям, причины срыва. Конвейер остановлен",
                beep=True)
            return False

        # Откат лечит испорченную политику. Если он не помог столько раз
        # подряд — беда в другом месте, и продолжать значит просто добить
        # скорость обучения до пола и крутиться там всю ночь.
        if done > self.config.max_rollbacks_per_stage:
            self.record(
                "give_up", stage.number,
                f"Ступень {stage.number} ({stage.title}) не берётся: "
                f"{done - 1} отката подряд не помогли. {exam.describe()}. "
                f"Конвейер остановлен — нужен разбор по разделу 5.6 "
                f"MIGRATION (прогон без шума, средние команды по осям, "
                f"причины срыва)", beep=True)
            return False

        previous = program.stage_by_number(max(1, stage.number - 1))
        archive = complete_file(previous.number)
        fallback = Path(previous.backup or program.BEST_MODEL_FILE)
        source = archive if archive.exists() else fallback

        # Проверяем, есть ли куда откатываться, ДО того как что-то стирать:
        # иначе неудачный откат оставил бы систему вообще без весов
        if not source.exists():
            self.record("error", stage.number,
                        f"Откат невозможен: нет ни {archive.name}, ни "
                        f"{fallback.name}. Конвейер остановлен", beep=False)
            return False

        self.record(
            "rollback", stage.number,
            f"ТУПИК на Ступени {stage.number} ({stage.title}): награда "
            f"{exam.mean_return:+.1f}, без роста {exam.stalled} эпизодов. "
            f"Откат на Ступень {previous.number}, дообучение "
            f"{self.config.rollback_steps} шагов")

        # 1. стираем поломанные текущие веса
        for path in (program.MODEL_FILE, program.BEST_MODEL_FILE):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Не удалось стереть %s: %s", path, exc)

        # 2. поднимаем архив предыдущей ступени
        shutil.copy2(source, program.MODEL_FILE)
        if not self.trainer.brain.load(program.MODEL_FILE):
            self.record("error", stage.number,
                        f"Архив {source.name} не читается — конвейер "
                        f"остановлен", beep=False)
            return False
        self.trainer.brain.stage = previous.number
        self.trainer.brain.best_mean = float("-inf")

        # 3. режем скорость обучения провалившейся ступени
        rate = scale_stage_learning_rate(stage.number, self.config.lr_penalty)
        self.record("lr", stage.number,
                    f"Скорость обучения Ступени {stage.number} снижена до "
                    f"{rate:.2e} — следующая попытка будет осторожнее",
                    beep=False)

        # 4. дообучение на предыдущей ступени ровно заданное число шагов
        drill = StageExam(self.config, watch_collapse=False)
        report = self.trainer.train_stage(
            previous, total_steps=self.config.rollback_steps,
            goal_altitude=self.goal_altitude, on_episode=drill.observe,
            difficulty_floor=1.0)   # прошлую ступень сеть уже сдала на полной
        self.total_steps += report.steps
        self.record("drill", previous.number,
                    f"Дообучение на Ступени {previous.number} закончено: "
                    f"{drill.describe()}", beep=False)

        if not self.running:
            return False

        # 5. возвращаемся на ту ступень, где случился тупик — С ТЕМИ
        # весами, которые только что дообучились. Звать `prepare` здесь
        # нельзя: он подтянул бы снимок ступени и стёр результат
        # дообучения, ради которого весь откат и затевался.
        self.trainer.brain.stage = stage.number
        self.trainer.brain.best_mean = float("-inf")
        self.trainer.brain.save()
        self.stage_number = stage.number
        self.record("retry", stage.number,
                    f"Повторная попытка Ступени {stage.number} "
                    f"({stage.title})", beep=False)
        return True

    # ------------------------------------------------------------------
    def progress_line(self) -> str:
        exam = self.exam
        if exam is None:
            return "экзамен ещё не начинался"
        return exam.describe()

    def status(self) -> dict:
        """Короткая сводка для наблюдения из пульта."""
        exam = self.exam
        return {
            "stage": self.stage_number,
            # total_steps растёт только между заходами тренера, поэтому
            # к нему добавляется прогресс текущего — иначе в наблюдении
            # счётчик выглядит замёрзшим
            "steps": self.total_steps + (self.trainer.last_report.steps
                                         if self.trainer.last_report else 0),
            "promotions": self.promotions,
            "rollbacks": self.rollbacks,
            "finished": self.finished,
            "orbits": exam.orbits if exam else 0,
            "rate": exam.orbit_rate if exam else 0.0,
            "episodes": exam.episodes if exam else 0,
            "mean_return": exam.mean_return if exam else 0.0,
            "stalled": exam.stalled if exam else 0,
        }

    def summary(self) -> dict:
        return {
            "finished": self.finished,
            "stage": self.stage_number,
            "steps": self.total_steps,
            "promotions": self.promotions,
            "rollbacks": self.rollbacks,
            "wall_time": time.time() - self.started,
            "events": [event.line() for event in self.events],
            "final_model": (str(FINAL_MODEL_FILE)
                            if FINAL_MODEL_FILE.exists() else None),
            "manifest": str(MANIFEST_FILE),
        }

    def describe(self) -> str:
        data = self.summary()
        hours = data["wall_time"] / 3600.0
        lines = [
            "=" * 70,
            "  ОТЧЁТ О ЗАВЕРШЕНИИ ЭВОЛЮЦИИ" if data["finished"]
            else "  ОТЧЁТ КОНВЕЙЕРА (работа не завершена)",
            "=" * 70,
            f"  Ступень на момент остановки: {data['stage']}",
            f"  Пройдено шагов среды:        {data['steps']}",
            f"  Переводов вверх:             {data['promotions']}",
            f"  Откатов назад:               {data['rollbacks']}",
            f"  Времени в работе:            {hours:.1f} ч",
        ]
        if data["final_model"]:
            lines.append(f"  Финальная модель:            {data['final_model']}")
        lines.append(f"  Манифест эволюции:           {data['manifest']}")
        if self.events:
            lines.append("")
            lines.append("  Хроника:")
            lines += [f"    {event.line()}" for event in self.events]
        return "\n".join(lines)


__all__ = ["EvolutionPipeline", "PipelineConfig", "StageExam",
           "MANIFEST_FILE", "FINAL_MODEL_FILE", "complete_file",
           "ALARM_BANNER"]
