"""Командная строка нейросетевого мозга KIA.

    python kia_neural.py info               устройство, размер сети, счётчики
    python kia_neural.py evolve             АВТОНОМНЫЙ КОНВЕЙЕР: вся
                                            программа без человека и без игры
    python kia_neural.py stages             учебная программа и что пройдено
    python kia_neural.py bench              где быстрее считать сеть
    python kia_neural.py stage 2 -n 2000000 обучение на ступени программы
    python kia_neural.py plus -n 2000000    то же: Тренажёр+ (ступень 2)
    python kia_neural.py pro  -n 3000000    то же: Тренажёр++ (ступень 3)
    python kia_neural.py pretrain -n 500000 базовый тренажёр без переноса весов
    python kia_neural.py programs           программы конструктора и математика
    python kia_neural.py builder -s 2       обучение конструктора на ступени
    python kia_neural.py math -s 1          тренажёр математики
    python kia_neural.py math-eval -s 1     экзамен математика по разделам
    python kia_neural.py evolve-mind        обе лестницы целиком, без человека
    python kia_neural.py eval -n 10         прогон политики без обучения
    python kia_neural.py fly -n 5           дообучение в живой KSP
    python kia_neural.py palette            какие детали видит конструктор
    python kia_neural.py best               вернуть лучшие веса (kia_model_best.pth)
    python kia_neural.py rollback 1         вернуть снимок ступени
    python kia_neural.py renorm             здоровье входов сети (и сброс
                                            статистики, если её убили выбросы)
    python kia_neural.py reset              стереть веса и начать с нуля

Ступени идут строго по порядку: каждая начинается с весов предыдущей и
делает их резервную копию (kia_model_stage1.pth, kia_model_stage2.pth).
Подробности — в kia_core/neural/curriculum.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kia_core.logging_setup import setup_logging      # noqa: E402
from kia_core.neural import is_available              # noqa: E402


def require_torch() -> bool:
    if is_available():
        return True
    print("PyTorch не установлен. Установка с поддержкой видеокарты:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu126")
    return False


# ==========================================================================
def cmd_info(args) -> int:
    from kia_core.neural.device import inspect
    from kia_core.neural.ppo import build_brain

    info = inspect()
    print("=" * 68)
    print("  НЕЙРОСЕТЕВОЙ МОЗГ KIA")
    print("=" * 68)
    print(f"  {info.describe()}")
    if info.kind != "cuda" and not info.build_has_cuda:
        print("  Установлена CPU-сборка torch. Для GPU нужна CUDA-сборка:")
        print("    pip install --force-reinstall torch "
              "--index-url https://download.pytorch.org/whl/cu126")
    elif info.kind != "cuda":
        print("  Видеокарта есть, но счёт идёт на процессоре — так быстрее")
        print("  для сети такого размера. Проверить: kia_neural.py bench")
    print()
    brain = build_brain()
    print(brain.summary())
    return 0


def cmd_pretrain(args) -> int:
    from kia_core.neural.ppo import PPOConfig
    from kia_core.neural.trainer import NeuralTrainer

    updates = max(10, args.steps // max(1, args.rollout))
    trainer = NeuralTrainer(config=PPOConfig(total_updates_planned=updates))
    print(trainer.brain.summary())
    print(f"\nПредобучение на тренажёре: {args.steps} шагов, "
          f"пакет {args.rollout}\n")
    report = trainer.pretrain_pilot(total_steps=args.steps, rollout=args.rollout,
                                    goal_altitude=args.altitude)
    print("\n" + report.describe())
    print(json.dumps(trainer.evaluate_pilot(episodes=10), ensure_ascii=False,
                     indent=2))
    return 0


# ==========================================================================
def cmd_stages(args) -> int:
    """Учебная программа: что за ступени и что уже пройдено."""
    from kia_core.neural.curriculum import describe_program, progress

    print("=" * 68)
    print("  " + describe_program().splitlines()[0])
    print("=" * 68)
    print()
    for line in describe_program().splitlines()[1:]:
        print(line)
    print()
    print("  ЛЕСТНИЦА ОБУЧЕНИЯ")
    for status in progress():
        print(f"    {status.stage.number}. {status.stage.title:20s} "
              f"{status.mark:10s} {status.checkpoint.describe()}")
    return 0


def cmd_stage(args) -> int:
    """Обучение на ступени программы с переносом весов предыдущей."""
    from kia_core.neural.curriculum import resolve
    from kia_core.neural.trainer import NeuralTrainer

    try:
        stage = resolve(args.stage)
    except KeyError as exc:
        print(exc)
        return 2

    steps = args.steps or stage.default_steps
    trainer = NeuralTrainer()
    print("=" * 68)
    print(f"  СТУПЕНЬ {stage.number}: {stage.title.upper()}")
    print("=" * 68)
    print(stage.describe())
    print()
    print(f"  Шагов: {steps}, пакет {stage.rollout}")
    print()
    report = trainer.train_stage(stage, total_steps=steps,
                                 goal_altitude=args.altitude)
    print("\n" + report.describe())
    if report.error:
        return 2
    print(json.dumps(trainer.evaluate_pilot(episodes=10, stage=stage),
                     ensure_ascii=False, indent=2))
    return 0


def cmd_bench(args) -> int:
    """Замер: где эта сеть считается быстрее — на процессоре или на GPU."""
    import time
    import torch
    from kia_core.neural.device import limit_threads
    from kia_core.neural.networks import ActorCritic
    from kia_core.neural.spaces import CONTINUOUS_DIM, FLIGHT_OBS_DIM

    limit_threads()
    print("=" * 68)
    print("  ГДЕ БЫСТРЕЕ СЧИТАТЬ СЕТЬ ПИЛОТА")
    print("=" * 68)
    print("  Замеряется полный вызов: собрать тензор, пройти сеть, вернуть")
    print("  результат в python. Именно так это происходит в обучении.")
    print()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    batches = [1, 8, 16, 32, 256]
    print(f"  {'устройство':<12}" + "".join(f"{b:>12}" for b in batches))
    best: dict = {}
    for device in devices:
        net = ActorCritic(FLIGHT_OBS_DIM, CONTINUOUS_DIM, (2,), hidden=256,
                          depth=2).to(device)
        row = []
        for batch in batches:
            sample = torch.randn(batch, FLIGHT_OBS_DIM, device=device)
            for _ in range(20):
                net.act(sample)
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            repeats = 200
            for _ in range(repeats):
                continuous, _, _, _ = net.act(sample)
                continuous.tolist()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) / repeats
            rate = batch / elapsed
            best[(device, batch)] = rate
            row.append(f"{rate:>12,.0f}".replace(",", " "))
        print(f"  {device:<12}" + "".join(row))
    print()
    print("  Числа — решений в секунду. Если процессор впереди, значит сеть")
    print("  слишком мала для видеокарты: время уходит не на счёт, а на")
    print("  запуск ядер и ожидание ответа.")
    if len(devices) > 1:
        cpu = best[("cpu", 16)]
        gpu = best[("cuda", 16)]
        print()
        print(f"  На рабочем размере пачки (16): процессор {cpu:,.0f} против "
              f"{gpu:,.0f} у GPU".replace(",", " "))
        print(f"  Выбран: {'процессор' if cpu >= gpu else 'GPU'}. "
              f"Переопределить — KIA_NEURAL_DEVICE=cuda|cpu")
    return 0


def cmd_evolve(args) -> int:
    """Полный автономный конвейер: вся программа без человека и без игры."""
    from kia_core.neural.pipeline import EvolutionPipeline, PipelineConfig

    config = PipelineConfig(beep=not args.silent)
    if args.max_steps:
        config.max_steps = args.max_steps
    pipeline = EvolutionPipeline(config=config, goal_altitude=args.altitude,
                                 start_stage=args.stage or None)
    print("=" * 68)
    print("  АВТОНОМНАЯ ЭВОЛЮЦИЯ KIA")
    print("=" * 68)
    print("  Живая игра в цикл не входит: только три тренажёра.")
    print(f"  Экзамен: средняя > {config.promote_reward:+.1f} за "
          f"{config.window} эпизодов и {config.promote_orbits} орбит.")
    print(f"  Тупик:   средняя < {config.collapse_reward:+.1f} без роста "
          f"{config.collapse_patience} эпизодов -> откат на ступень назад.")
    print("  Остановить: Ctrl+C (веса и манифест сохранятся).")
    print()
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n  Останавливаю конвейер...")
        pipeline.stop()
    print("\n" + pipeline.describe())
    return 0 if pipeline.finished else 1


def cmd_rollback(args) -> int:
    """Вернуть в работу снимок весов указанной ступени."""
    from kia_core.neural.curriculum import resolve, rollback
    from kia_core.neural.ppo import build_brain

    try:
        stage = resolve(args.stage)
    except KeyError as exc:
        print(exc)
        return 2
    brain = build_brain()
    if rollback(stage, brain):
        print(f"Веса ступени {stage.number} ({stage.title}) возвращены "
              f"в работу: средняя за 20 = {brain.pilot.mean_return(20):+.2f}")
        return 0
    print(f"Снимка ступени {stage.number} нет — откатываться не к чему.")
    return 1


def cmd_builder(args) -> int:
    from kia_core.neural.programs import resolve
    from kia_core.neural.trainer import NeuralTrainer
    trainer = NeuralTrainer()
    stage = resolve("builder", args.stage) if args.stage else None
    report = trainer.train_builder(episodes=args.episodes or None, stage=stage,
                                   goal_altitude=args.altitude)
    print("\n" + report.describe())
    return 0


def cmd_math(args) -> int:
    from kia_core.neural.programs import resolve
    from kia_core.neural.trainer import NeuralTrainer
    trainer = NeuralTrainer()
    stage = resolve("math", args.stage) if args.stage else None
    report = trainer.train_math(stage=stage, episodes=args.episodes or None)
    print("\n" + report.describe())
    print(f"Точность в допуске: {report.difficulty * 100:.1f} %")
    return 0


def cmd_math_eval(args) -> int:
    from kia_core.neural.trainer import NeuralTrainer
    trainer = NeuralTrainer()
    stage = int(args.stage) if str(args.stage).isdigit() else None
    result = trainer.evaluate_math(stage, tasks=args.tasks)
    print(f"\n{result['title']}: решено {result['accuracy'] * 100:.1f} % "
          f"при допуске {result['tolerance'] * 100:.0f} %\n")
    print(f"  {'раздел':24s} {'решено':>12s}  {'средняя ошибка':>14s}")
    for key, row in result["tasks"].items():
        share = 100.0 * row["solved"] / max(1, row["total"])
        print(f"  {key:24s} {row['solved']:4d}/{row['total']:<4d} {share:5.0f}% "
              f"{row['mean_error'] * 100:13.2f} %")
    return 0


def cmd_evolve_mind(args) -> int:
    """Конструктор и математик проходят свои лестницы сами."""
    from kia_core.neural.trainer import NeuralTrainer
    trainer = NeuralTrainer()
    trainer.begin_session()
    programs = tuple(p.strip() for p in args.programs.split(",") if p.strip())
    journal = trainer.evolve_programs(programs=programs,
                                      attempts_per_stage=args.attempts)
    print()
    print(f"  {'программа':12s} {'ступень':16s} {'заход':>6s} {'итог':>10s}  подробности")
    for row in journal:
        print(f"  {row['program']:12s} {row['title']:16s} "
              f"{row['attempt']:>6d} {'СДАНО' if row['passed'] else 'не сдано':>10s}"
              f"  {row['detail']} ({row['wall_time']:.0f} с)")
    return 0


def cmd_programs(args) -> int:
    from kia_core.neural.programs import describe_program
    from kia_core.neural.ppo import build_brain
    print(describe_program("builder"))
    print()
    print(describe_program("math"))
    brain = build_brain()
    print(f"\nСейчас: конструктор на ступени {brain.stages.get('builder', 1)}, "
          f"математик на ступени {brain.stages.get('math', 1)}")
    return 0


def cmd_eval(args) -> int:
    from kia_core.neural.curriculum import resolve
    from kia_core.neural.trainer import NeuralTrainer
    trainer = NeuralTrainer()
    stage = resolve(args.stage) if args.stage else None
    result = trainer.evaluate_pilot(episodes=args.episodes,
                                    goal_altitude=args.altitude, stage=stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_fly(args) -> int:
    from kia_core.agent.dispatcher import Dispatcher
    from kia_core.neural.trainer import NeuralTrainer

    dispatcher = Dispatcher(resume=True)
    if not dispatcher.connect():
        print("Нет связи с KSP: запустите игру и включите сервер kRPC.")
        return 2
    trainer = NeuralTrainer()
    report = trainer.train_in_ksp(dispatcher, episodes=args.episodes,
                                  goal_altitude=args.altitude,
                                  use_neural_builder=args.neural_builder,
                                  flight_log=dispatcher.flight_log)
    dispatcher.disconnect()
    print("\n" + report.describe())
    return 0


def cmd_palette(args) -> int:
    from kia_core.neural.builder_env import build_palette
    palette = build_palette()
    print("Палитра деталей, из которой выбирает сеть-конструктор:")
    print(palette.describe())
    return 0


def cmd_best(args) -> int:
    """Вернуть в работу веса с лучшим средним результатом."""
    from kia_core.neural.ppo import BEST_MODEL_FILE, build_brain
    brain = build_brain()
    if not BEST_MODEL_FILE.exists():
        print(f"Файла {BEST_MODEL_FILE.name} нет: лучшая версия ещё не сохранялась.")
        return 1
    print(f"Текущие веса: средняя за 20 = {brain.pilot.mean_return(20):+.2f}")
    if brain.restore_best():
        print(f"Восстановлено из {BEST_MODEL_FILE.name}: "
              f"средняя за 20 = {brain.pilot.mean_return(20):+.2f}")
        return 0
    print("Не удалось восстановить.")
    return 1


def cmd_renorm(args) -> int:
    """Показать здоровье входов и, по команде, обнулить их статистику.

    Нормализатор входов накопительный и лежит в чекпоинте. Один сорт
    выбросов способен убить признак навсегда: гиперболический апоапсис
    1e9 давал признак 10 000 при рабочем диапазоне 0-1, дисперсия
    улетала в десятки тысяч, и весь полезный диапазон умещался в
    0.00005 сигмы. Сам по себе такой признак не выправится — при 42 млн
    наблюдений в статистике новый миллион сдвигает среднее на проценты.
    """
    import shutil

    from kia_core.config import DATA_DIR
    from kia_core.neural.ppo import MODEL_FILE, build_brain

    brain = build_brain()
    if not MODEL_FILE.exists():
        print("Весов нет — обнулять нечего.")
        return 1

    health = sorted(brain.normalizer_health(), key=lambda item: item[1])
    print("\nЗдоровье входов (сколько сигм занимает рабочий диапазон "
          "признака):")
    for name, span in health[:8]:
        mark = "МЁРТВ" if span < 0.01 else ("сжат" if span < 0.1 else "жив")
        print(f"  {name:<24} {span:>12.5f}   {mark}")
    dead = [name for name, span in health if span < 0.01]
    if not dead:
        print("\nЗадавленных входов нет — обнулять не требуется.")
        return 0

    print(f"\nЗадавлено входов: {len(dead)} ({', '.join(dead)}).")
    if not args.yes:
        answer = input("Обнулить статистику входов? Введите ДА: ")
        if answer.strip().upper() != "ДА":
            print("Отменено.")
            return 0

    backup = DATA_DIR / "old_weights" / f"{MODEL_FILE.stem}_before_renorm.pth"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_FILE, backup)
    brain.reset_normalizer()
    brain.save()
    print(f"Копия прежних весов: {backup}")
    print("Статистика входов обнулена. Веса сохранены. Учить заново со "
          "ступени 1: python kia_neural.py evolve")
    return 0


def cmd_reset(args) -> int:
    from kia_core.neural.ppo import MODEL_FILE
    if not MODEL_FILE.exists():
        print("Весов и так нет — мозг чистый.")
        return 0
    answer = input(f"Стереть {MODEL_FILE.name} и весь накопленный опыт? "
                   f"Введите СТЕРЕТЬ: ")
    if answer.strip().upper() == "СТЕРЕТЬ":
        MODEL_FILE.unlink()
        print("Веса удалены. Следующий запуск начнётся с чистого листа.")
    else:
        print("Отменено.")
    return 0


# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kia_neural", description="Нейросетевой мозг KIA (PPO на PyTorch)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="устройство, размер сети, счётчики обучения")

    p_pre = sub.add_parser("pretrain", help="предобучение пилота на тренажёре")
    p_pre.add_argument("-n", "--steps", type=int, default=500_000)
    p_pre.add_argument("-r", "--rollout", type=int, default=8192)
    p_pre.add_argument("-a", "--altitude", type=float, default=80_000.0)

    sub.add_parser("stages", help="учебная программа и что уже пройдено")
    sub.add_parser("bench", help="где быстрее считать сеть: CPU или GPU")

    p_stage = sub.add_parser("stage", help="обучение на ступени программы")
    p_stage.add_argument("stage", help="номер ступени (1/2/3) или имя "
                                       "(base/plus/pro)")
    p_stage.add_argument("-n", "--steps", type=int, default=0)
    p_stage.add_argument("-a", "--altitude", type=float, default=80_000.0)

    for key, title in (("plus", "Тренажёр+ (ступень 2)"),
                       ("pro", "Тренажёр++ (ступень 3)")):
        shortcut = sub.add_parser(key, help=title)
        shortcut.add_argument("-n", "--steps", type=int, default=0)
        shortcut.add_argument("-a", "--altitude", type=float, default=80_000.0)
        shortcut.set_defaults(stage=key)

    p_evolve = sub.add_parser("evolve",
                              help="автономный конвейер по всей программе")
    p_evolve.add_argument("-a", "--altitude", type=float, default=80_000.0)
    p_evolve.add_argument("-s", "--stage", type=int, default=0,
                          help="с какой ступени начать (по умолчанию — с той, "
                               "что записана в весах)")
    p_evolve.add_argument("--max-steps", type=int, default=0,
                          dest="max_steps",
                          help="предохранитель на весь конвейер")
    p_evolve.add_argument("--silent", action="store_true",
                          help="без звукового сигнала при смене уровня")

    p_back = sub.add_parser("rollback", help="вернуть снимок ступени в работу")
    p_back.add_argument("stage", help="номер ступени (1/2) или имя (base/plus)")

    p_build = sub.add_parser("builder", help="обучение сети-конструктора")
    p_build.add_argument("-n", "--episodes", type=int, default=0,
                         help="0 — ориентир ступени")
    p_build.add_argument("-a", "--altitude", type=float, default=80_000.0)
    p_build.add_argument("-s", "--stage", default="",
                         help="ступень сборки: 1/2/3 или build/build_plus/build_pro")

    p_math = sub.add_parser("math", help="тренажёр математики")
    p_math.add_argument("-n", "--episodes", type=int, default=0,
                        help="0 — ориентир ступени")
    p_math.add_argument("-s", "--stage", default="",
                        help="ступень: 1/2/3 или math/math_plus/math_pro")

    p_math_eval = sub.add_parser("math-eval", help="экзамен математика")
    p_math_eval.add_argument("-s", "--stage", default="")
    p_math_eval.add_argument("-n", "--tasks", type=int, default=600)

    sub.add_parser("programs", help="программы конструктора и математика")

    p_mind = sub.add_parser(
        "evolve-mind",
        help="конструктор и математик проходят свои лестницы сами")
    p_mind.add_argument("--programs", default="math,builder",
                        help="какие программы гнать, через запятую")
    p_mind.add_argument("--attempts", type=int, default=3,
                        help="сколько заходов на ступень до отказа")

    p_eval = sub.add_parser("eval", help="прогон политики без обучения")
    p_eval.add_argument("-n", "--episodes", type=int, default=10)
    p_eval.add_argument("-a", "--altitude", type=float, default=80_000.0)
    p_eval.add_argument("-s", "--stage", default="",
                        help="на какой ступени проверять (по умолчанию — "
                             "на той, где мозг учился последней)")

    p_fly = sub.add_parser("fly", help="дообучение в живой KSP")
    p_fly.add_argument("-n", "--episodes", type=int, default=5)
    p_fly.add_argument("-a", "--altitude", type=float, default=80_000.0)
    p_fly.add_argument("--neural-builder", action="store_true",
                       help="ракету собирает сеть, а не расчётный конструктор")

    sub.add_parser("palette", help="детали, доступные сети-конструктору")
    sub.add_parser("best", help="вернуть веса с лучшим результатом")
    p_renorm = sub.add_parser(
        "renorm", help="здоровье входов сети и обнуление их статистики")
    p_renorm.add_argument("-y", "--yes", action="store_true",
                          help="не спрашивать подтверждения")

    sub.add_parser("reset", help="стереть веса")
    return parser


COMMANDS = {"info": cmd_info, "pretrain": cmd_pretrain, "builder": cmd_builder,
            "eval": cmd_eval, "fly": cmd_fly, "palette": cmd_palette,
            "best": cmd_best, "reset": cmd_reset, "stages": cmd_stages,
            "renorm": cmd_renorm,
            "stage": cmd_stage, "plus": cmd_stage, "pro": cmd_stage,
            "rollback": cmd_rollback, "evolve": cmd_evolve,
            "bench": cmd_bench, "math": cmd_math, "math-eval": cmd_math_eval,
            "programs": cmd_programs, "evolve-mind": cmd_evolve_mind}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if not require_torch():
        return 1
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
