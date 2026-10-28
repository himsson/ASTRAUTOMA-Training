"""УЧЕБНАЯ ПРОГРАММА KIA: три ступени предобучения пилота.

Проблема, которую решает эта программа, называется sim-to-real gap.
Политика, выученная в идеальном мире, разваливается в реальном: там, где
тренажёр исполнял команду мгновенно и точно, живая игра отвечает через
сто миллисекунд, двигатель раскручивается полсекунды, а ветер сносит нос.
Сеть, никогда этого не видевшая, продолжает управлять «как привыкла» и
теряет аппарат на первых же секундах.

Лекарство — постепенное усложнение среды (curriculum learning). Каждая
ступень начинается НЕ с чистого листа, а с весов предыдущей: сеть уже
умеет летать и лишь доучивается под новые правила.

    1. Базовый Тренажёр   быстрая точечная физика — научиться выводить
                          аппарат на орбиту в принципе
    2. Тренажёр+          та же скорость, но со скрытыми механиками KSP:
                          задержка связи, таймаут ступеней, ветер, угол
                          атаки, инерция тяги, лимит энергии
    3. Тренажёр++         честная 3D-баллистика, таблицы атмосферы
                          Кербина, кривые двигателей, живой центр масс —
                          но управление ещё прежнее
    4. Тренажёр+++        рули задают момент, а не угол; решения втрое
                          чаще. Последний рубеж перед живой игрой

За одну ступень должно меняться ОДНО. Первая попытка совместила новую
физику и новый закон управления в одной ступени — политика, уверенно
сдавшая Тренажёр+, обвалилась там за две минуты.

Веса каждой ступени сохраняются перед переходом на следующую:

    kia_model.pth          текущее обучение
    kia_model_best.pth     лучший результат ТЕКУЩЕЙ ступени
    kia_model_stage1.pth   снимок после Базового Тренажёра
    kia_model_stage2.pth   снимок после Тренажёра+
    kia_model_stage3.pth   снимок после Тренажёра++

Откатиться на любую ступень можно, не потеряв ничего: снимки не
перезаписываются автоматически, а только по явной команде оператора.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ROOT
from ..logging_setup import get_logger

log = get_logger("neural.curriculum")

MODEL_FILE = ROOT / "kia_model.pth"
BEST_MODEL_FILE = ROOT / "kia_model_best.pth"
STAGE1_FILE = ROOT / "kia_model_stage1.pth"
STAGE2_FILE = ROOT / "kia_model_stage2.pth"
STAGE3_FILE = ROOT / "kia_model_stage3.pth"
STAGE4_FILE = ROOT / "kia_model_stage4.pth"


# ==========================================================================
@dataclass(frozen=True)
class Stage:
    """Одна ступень учебной программы."""
    number: int
    key: str
    title: str
    summary: str
    lessons: tuple[str, ...]
    module: str
    factory: str
    dt: float
    action_repeat: int
    rollout: int
    default_steps: int
    max_time: float
    # Доля прогона, за которую среда добирает полную жёсткость. Пока идёт
    # разогрев, штраф за скольжение работает как предупреждение, а не как
    # приговор: сеть успевает нащупать, чем его гасить, прежде чем он
    # начнёт перевешивать всю остальную награду.
    warmup: float = 0.0
    # Сколько копий среды крутить одновременно. Сеть спрашивают сразу обо
    # всех — расход на вызов почти не зависит от размера пачки, поэтому
    # это главный рычаг скорости обучения. Ограничение сверху — процессор:
    # каждая копия честно считает свою физику.
    envs: int = 16
    # Дополнительные настройки среды этой ступени. Через них одна и та же
    # физика служит двум разным ступеням: с приваренными «тренировочными
    # колёсами» и без них.
    env_options: tuple = ()
    # снимок весов, который делается ПЕРЕД началом этой ступени
    backup: Path | None = None
    # какие веса берутся за основу, если рабочего файла нет
    source: Path | None = None
    # Экзамен по спасению экипажа: (сколько аварий пережить,
    # какую долю экипажей спасти). Задаётся только там, где
    # ступень про аварии и есть: орбитами её не измерить.
    crew_exam: tuple | None = None

    # ------------------------------------------------------------------
    def make_env(self, goal_altitude: float = 80_000.0, seed: int | None = None,
                 exploring_starts: float = 0.5):
        """Создаёт среду этой ступени (импорт ленивый — torch не нужен)."""
        import importlib
        module = importlib.import_module(f".{self.module}", __package__)
        env_class = getattr(module, self.factory)
        return env_class(goal_altitude=goal_altitude, dt=self.dt,
                         max_time=self.max_time, seed=seed,
                         exploring_starts=exploring_starts,
                         **dict(self.env_options))

    @property
    def decision_seconds(self) -> float:
        """Сколько игрового времени держится одно решение сети."""
        return self.dt * self.action_repeat

    def describe(self) -> str:
        import math
        from .ppo import learning_rate_for_stage, log_std_max_for_stage
        lines = [f"  {self.number}. {self.title} — {self.summary}"]
        lines += [f"       • {lesson}" for lesson in self.lessons]
        lines.append(f"       шаг {self.dt:.2f} с × {self.action_repeat} = "
                     f"решение раз в {self.decision_seconds:.1f} с, "
                     f"{self.envs} копий среды одновременно")
        lines.append(f"       скорость обучения "
                     f"{learning_rate_for_stage(self.number):.1e}, "
                     f"разброс действий σ "
                     f"{math.exp(log_std_max_for_stage(self.number)):.3f}")
        if self.warmup > 0.0:
            lines.append(f"       первые {self.warmup * 100:.0f} % прогона — "
                         f"щадящий режим: штраф за скольжение работает как "
                         f"предупреждение")
        return "\n".join(lines)


# ВСЕ ступени живут в ОДНОМ мире — трёхмерном клоне KSP. Так было не
# всегда, и это стоило проекту дня работы: первые две ступени считали
# плоскую задачу, третья — трёхмерную, и сеть, отлично сдавшая плоские,
# на третьей давала НОЛЬ орбит. Замер, который всё расставил по местам:
# та же третья ступень, обученная С ЧИСТОГО ЛИСТА, за четыре минуты дала
# 164 орбиты из 1098 эпизодов. Ступень была проходима всегда — непроходим
# был ПЕРЕХОД в неё: в плоском мире нет ни курса, ни крена, и сеть
# приходила с мёртвыми осями на упоре и с двумя парами входов, которые
# никогда не менялись.
#
# Теперь ступени различаются только тем, что мешает управлять:
#
#     1. мир и аэродинамика, интерфейс идеальный
#     2. + скрытые механики игры (задержка, ветер, батарея, инерция)
#     3. + втрое более частые решения
#     4. + управление моментом вместо угла
STAGES: tuple[Stage, ...] = (
    Stage(
        number=1, key="base", title="Тренажёр",
        summary="полная физика KSP при идеальном управлении",
        lessons=(
            "оторваться от стола и не завалить нос",
            "поднимать орбитальную энергию, а не просто высоту",
            "набирать момент импульса: орбита — это скорость вбок",
            "3D с самого начала: курс и крен настоящие",
            "держать нос по потоку, иначе опрокинет",
            "сбрасывать ступень, когда она выработана",
        ),
        module="sim_env_pro", factory="PrecisionAscentSimulator",
        dt=0.2, action_repeat=5, rollout=512, envs=16,
        # интерфейс идеальный: команда исполняется мгновенно и точно
        env_options=(("assist_floor", 1.0), ("yaw_fade", False),
                     ("mechanics", False)),
        # 1.5 млн шагов и разогрев 35 % — ровно то, на чём сеть выучила
        # этот мир с чистого листа за четыре минуты (164 орбиты)
        default_steps=1_500_000, max_time=420.0, warmup=0.35,
        backup=None, source=None,
    ),
    Stage(
        number=2, key="plus", title="Тренажёр+",
        summary="скрытые механики реальной игры",
        lessons=(
            "задержка команд kRPC около 100 мс",
            "повторное отделение ступени заблокировано на 2 с",
            "ветер сносит нос на 1-2°, нужны микро-коррекции",
            "двигатель выходит на режим 0.5-1 с: рулить надо на опережение",
            "рули тратят заряд, дёрганья обесточивают аппарат",
        ),
        module="sim_env_pro", factory="PrecisionAscentSimulator",
        dt=0.2, action_repeat=5, rollout=512, envs=16,
        env_options=(("assist_floor", 1.0), ("yaw_fade", False),
                     ("mechanics", True)),
        default_steps=2_000_000, max_time=420.0, warmup=0.35,
        backup=STAGE1_FILE, source=BEST_MODEL_FILE,
    ),
    Stage(
        number=3, key="pro", title="Тренажёр++",
        summary="решения втрое чаще — как в живой игре",
        lessons=(
            "решение раз в 0.3 с вместо секунды",
            "одна и та же команда поворачивает аппарат втрое слабее",
            "профиль выведения приходится вести мелкими поправками",
            "руки прежние: команда всё ещё задаёт скорость поворота",
        ),
        module="sim_env_pro", factory="PrecisionAscentSimulator",
        # Меняется РОВНО ОДНО — ритм. Это не мелочь: при переходе на 0.3 с
        # одна и та же команда поворачивает аппарат втрое слабее, и весь
        # профиль выведения приходится перекладывать.
        dt=0.1, action_repeat=3, rollout=384, envs=16,
        env_options=(("assist_floor", 1.0), ("yaw_fade", False),
                     ("mechanics", True)),
        default_steps=3_000_000, max_time=420.0, warmup=0.35,
        backup=STAGE2_FILE, source=BEST_MODEL_FILE,
    ),
    Stage(
        number=4, key="max", title="Тренажёр+++",
        summary="управление моментом — как в игре с выключенным SAS",
        lessons=(
            "рули задают МОМЕНТ, а не угол: работает инерция аппарата",
            "угол — двойной интеграл от команды, ошибки накапливаются",
            "качающееся сопло сильнее маховиков — на выбеге аппарат вял",
            "запас устойчивости меняется по мере выработки топлива",
        ),
        module="sim_env_pro", factory="PrecisionAscentSimulator",
        dt=0.1, action_repeat=3, rollout=384, envs=16,
        # Разогрев длинный не для мягкости штрафов, а потому что за него
        # происходит смена ЗАКОНА УПРАВЛЕНИЯ: с «команда = скорость
        # поворота» на «команда = момент».
        env_options=(("assist_floor", 0.0), ("yaw_fade", False),
                     ("mechanics", True)),
        default_steps=4_000_000, max_time=420.0, warmup=0.60,
        backup=STAGE3_FILE, source=BEST_MODEL_FILE,
    ),
    Stage(
        number=5, key="fail", title="Аварии",
        summary="тот же мир, но техника ломается посреди полёта",
        lessons=(
            "двигатель может погаснуть или потерять тягу в любой момент",
            "ступень может не отделиться — план придётся менять на лету",
            "генератор отказывает: заряда хватит на считанные манёвры",
            "течь в баке: топлива меньше, чем показывал расчёт",
            "решение принимается за секунды, переигрывать его некогда",
        ),
        module="sim_env_pro", factory="PrecisionAscentSimulator",
        dt=0.1, action_repeat=3, rollout=384, envs=16,
        # Мир тот же, что на четвёртой ступени, добавлена ровно одна
        # механика. Иначе нельзя понять, с чем именно сеть не справилась:
        # с отказом или с очередной сменой правил вокруг него.
        # Доля отказов поднята с обычных 0.35, а экипаж посажен почти в
        # каждый полёт намеренно. В первом прогоне авария с людьми на
        # борту случалась в одном полёте из шести, и на этой шестой части
        # сеть должна была выучить поведение, которого не видела больше
        # нигде. Здесь таких полётов — половина; исправных по-прежнему
        # хватает, чтобы орбитальный экзамен имел смысл.
        env_options=(("assist_floor", 0.0), ("yaw_fade", False),
                     ("mechanics", True), ("failures", True),
                     ("failure_rate", 0.5), ("crew_share", 0.8)),
        # Спуск под куполом с полусотни километров в 420 секунд не
        # укладывается: эпизод обрывался на полпути, и это засчитывалось
        # как погибший экипаж.
        default_steps=6_000_000, max_time=600.0, warmup=0.50,
        backup=STAGE4_FILE, source=BEST_MODEL_FILE,
        # Девять экипажей из десяти домой, и не меньше чем на сорока
        # авариях. Порог не с потолка: пока в тренажёре не было
        # отделяемой кабины, спасти не удавалось НИКОГО — замер
        # скриптового пилота с идеальным тормозным импульсом дал 0 из
        # 34. С кабиной тот же замер на нынешних весах даёт 98 % по
        # безвыходным отказам. Значит, девять из десяти — это строгая,
        # но взятая планка, а не пожелание.
        # Третье число — про деньги. Экзамен из двух первых сдаётся
        # уводом кабины на каждой поломке: замер показал 44 увода на 44
        # аварии, в том числе 26 таких, где двигатель тянул на полную и
        # ракета долетела бы. Поэтому отдельно спрашивается: из отказов,
        # после которых тяга осталась, сколько полётов всё же доведено
        # до орбиты. Экипаж по-прежнему важнее — порог по нему выше.
        crew_exam=(40, 0.90, 0.50),
    ),
)

FIRST_STAGE = STAGES[0]
LAST_STAGE = STAGES[-1]


# ==========================================================================
def stage_by_number(number: int) -> Stage:
    for stage in STAGES:
        if stage.number == number:
            return stage
    raise KeyError(f"нет ступени №{number}")


def stage_by_key(key: str) -> Stage:
    for stage in STAGES:
        if stage.key == key:
            return stage
    raise KeyError(f"нет ступени «{key}»")


def resolve(reference: str | int) -> Stage:
    """Ступень по номеру или по короткому имени."""
    if isinstance(reference, int):
        return stage_by_number(reference)
    text = str(reference).strip().lower()
    if text.isdigit():
        return stage_by_number(int(text))
    return stage_by_key(text)


# ==========================================================================
@dataclass
class CheckpointInfo:
    """Что известно о файле весов без его загрузки."""
    path: Path
    exists: bool = False
    size: int = 0
    saved_at: str = ""

    def describe(self) -> str:
        if not self.exists:
            return f"{self.path.name}: нет"
        return (f"{self.path.name}: {self.size / 1024:.0f} КБ, "
                f"сохранён {self.saved_at}")


def checkpoint_info(path: Path | None) -> CheckpointInfo:
    if path is None:
        return CheckpointInfo(Path("—"))
    info = CheckpointInfo(Path(path))
    if info.path.exists():
        stat = info.path.stat()
        info.exists = True
        info.size = stat.st_size
        info.saved_at = time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(stat.st_mtime))
    return info


@dataclass
class StageStatus:
    """Состояние одной ступени для пульта."""
    stage: Stage
    checkpoint: CheckpointInfo
    passed: bool = False
    started: bool = False
    current: bool = False

    @property
    def mark(self) -> str:
        if self.passed:
            return "ПРОЙДЕНО"
        if self.current or self.started:
            return "в работе"
        return "не начата"


def progress(current_stage: int | None = None) -> list[StageStatus]:
    """Лестница обучения: что пройдено, что впереди.

    Ступень считается ПРОЙДЕННОЙ, если существует снимок весов, который
    делается при переходе на следующую (или если мозг уже учится дальше).
    «В работе» — если на этой ступени лежат текущие веса. Номер ступени
    хранится в самих весах (`Brain.stage`) и передаётся сюда снаружи,
    чтобы не грузить torch ради одной строки меню.
    """
    statuses = []
    trained = BEST_MODEL_FILE.exists() or MODEL_FILE.exists()
    for index, stage in enumerate(STAGES):
        following = STAGES[index + 1] if index + 1 < len(STAGES) else None
        snapshot = following.backup if following else None
        passed = bool(snapshot and snapshot.exists())
        if current_stage is not None and current_stage > stage.number:
            passed = True
        started = trained and (current_stage == stage.number
                               or (current_stage is None
                                   and stage.number == 1))
        statuses.append(StageStatus(
            stage=stage,
            checkpoint=checkpoint_info(snapshot or BEST_MODEL_FILE),
            passed=passed, started=started,
            current=(current_stage == stage.number)))
    return statuses


# ==========================================================================
@dataclass
class Promotion:
    """Отчёт о переходе на новую ступень."""
    stage: Stage
    backed_up: Path | None = None
    loaded_from: Path | None = None
    messages: list[str] = field(default_factory=list)
    ok: bool = True

    def describe(self) -> str:
        lines = [f"Переход на ступень {self.stage.number}: {self.stage.title}"]
        lines += [f"  {line}" for line in self.messages]
        return "\n".join(lines)


def prepare(stage: Stage, brain=None, overwrite_backup: bool = False,
            dry_run: bool = False) -> Promotion:
    """Готовит веса к обучению на указанной ступени.

    Порядок важен и защищает от потери результата:

    1. снимок текущих лучших весов в файл ступени (kia_model_stage1.pth
       или kia_model_stage2.pth) — если снимка ещё нет;
    2. загрузка этого снимка в рабочий мозг, чтобы обучение продолжилось
       с достигнутого, а не с нуля;
    3. пометка мозга номером новой ступени.

    Возврат к предыдущей ступени ничего не ломает: снимок остаётся на
    месте и повторно не перезаписывается (если только оператор явно не
    попросил `overwrite_backup`).

    dry_run=True только проверяет выполнимость перехода и не трогает
    ни одного файла — этим пользуется пульт, чтобы не запускать фоновое
    задание, которое всё равно упадёт.
    """
    report = Promotion(stage=stage)
    reached = int(getattr(brain, "stage", 0) or 0)
    continuing = reached >= stage.number

    if continuing:
        report.messages.append(
            f"мозг уже на ступени {reached} — продолжаем обучение, "
            f"веса не подменяю")
    elif reached and reached < stage.number - 1:
        report.messages.append(
            f"внимание: ступень {stage.number - 1} пропущена "
            f"(мозг прошёл только {reached}) — переход всё равно выполняю")

    if stage.backup is not None and not continuing:
        source = BEST_MODEL_FILE if BEST_MODEL_FILE.exists() else MODEL_FILE
        if not source.exists():
            report.ok = False
            report.messages.append(
                f"нет весов предыдущей ступени ({BEST_MODEL_FILE.name} и "
                f"{MODEL_FILE.name} отсутствуют) — сначала пройдите "
                f"ступень {stage.number - 1}")
            return report
        if stage.backup.exists() and not overwrite_backup:
            report.messages.append(
                f"снимок {stage.backup.name} уже есть "
                f"({checkpoint_info(stage.backup).saved_at}) — не трогаю")
        elif dry_run:
            report.messages.append(
                f"будет сделана резервная копия {source.name} -> "
                f"{stage.backup.name}")
        else:
            shutil.copy2(source, stage.backup)
            report.backed_up = stage.backup
            report.messages.append(
                f"резервная копия {source.name} -> {stage.backup.name}")
            log.info("Веса ступени сохранены: %s -> %s", source, stage.backup)

    if dry_run:
        return report                  # проверка выполнимости: мозг не трогаем

    if brain is not None:
        if not continuing:
            loaded = _load_start_weights(stage, brain, report)
            if loaded is not None:
                report.loaded_from = loaded
        brain.stage = stage.number
        if not continuing:
            # Лучший результат прошлой ступени не должен мешать новой:
            # среды разные, награды несопоставимы.
            #
            # А вот при ПРОДОЛЖЕНИИ той же ступени планку сбрасывать
            # нельзя. Конвейер заходит на ступень заново каждый раз, когда
            # кончается бюджет шагов (на живом прогоне — шестьдесят раз
            # подряд), и с обнулённой планкой первое же обновление любого
            # захода объявляло себя «лучшим» и затирало kia_model_best.pth.
            # Файл переставал быть лучшей версией ступени и становился
            # просто последней.
            brain.best_mean = float("-inf")
    return report


def _load_start_weights(stage: Stage, brain, report: Promotion) -> Path | None:
    """Берёт стартовые веса для ступени: снимок, лучшие или текущие."""
    candidates = []
    if stage.backup is not None:
        candidates.append(stage.backup)
    if stage.source is not None:
        candidates.append(stage.source)
    candidates.append(MODEL_FILE)

    for path in candidates:
        if path.exists() and brain.load(path):
            report.messages.append(f"веса загружены из {path.name}: "
                                   f"{brain.pilot.stats.total_steps} шагов, "
                                   f"средняя за 20 "
                                   f"{brain.pilot.mean_return(20):+.2f}")
            return path
    report.messages.append("готовых весов нет — обучение начнётся с нуля")
    return None


def rollback(stage: Stage, brain) -> bool:
    """Возвращает в работу снимок весов указанной ступени."""
    snapshot = None
    for candidate in STAGES:
        if candidate.number == stage.number + 1:
            snapshot = candidate.backup
    if snapshot is None or not snapshot.exists():
        log.warning("Снимка ступени %d нет", stage.number)
        return False
    if not brain.load(snapshot):
        return False
    brain.stage = stage.number
    brain.save(MODEL_FILE)
    log.info("Восстановлены веса ступени %d из %s", stage.number, snapshot.name)
    return True


def describe_program() -> str:
    """Текст программы обучения целиком — для пульта и документации."""
    lines = ["УЧЕБНАЯ ПРОГРАММА (curriculum learning)", ""]
    for stage in STAGES:
        lines.append(stage.describe())
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = ["Stage", "STAGES", "FIRST_STAGE", "LAST_STAGE", "stage_by_number",
           "stage_by_key", "resolve", "progress", "prepare", "rollback",
           "checkpoint_info", "describe_program", "MODEL_FILE",
           "BEST_MODEL_FILE", "STAGE1_FILE", "STAGE2_FILE", "STAGE3_FILE",
           "STAGE4_FILE"]
