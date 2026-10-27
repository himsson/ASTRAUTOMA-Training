"""Самопроверка KIA без запуска игры.

Проверяет расчётное ядро, каталог деталей, генератор чертежей, разбор
команд оператора, прейскурант наград, журнал полёта и цикл обучения.

Запуск:  python tools/selftest.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kia_core.engineer import rocket_math as rm                              # noqa: E402
from kia_core.engineer.autodesign import (AutonomousEngineer, DesignPolicy,  # noqa: E402
                                     compute_budget, plan_stages)
from kia_core.engineer.craft_writer import CraftAssembler, validate_craft    # noqa: E402
from kia_core.engineer.parts_catalog import get_catalog                      # noqa: E402
from kia_core.engineer.parts_db import get_body                              # noqa: E402
from kia_core.flight_log import FlightLog                                    # noqa: E402
from kia_core.learning.genome import Individual, random_individual           # noqa: E402
from kia_core.learning.history import HistoryStore, RunRecord                # noqa: E402
from kia_core.learning.optimizer import GeneticOptimizer                     # noqa: E402
from kia_core.learning.diagnosis import FailureAnalyst                       # noqa: E402
from kia_core.learning.reward_book import RewardBook, FlightMetrics          # noqa: E402
from kia_core.learning.rewards import Failure, Milestone, RewardSystem       # noqa: E402
from kia_core.mission_spec import MissionSpec, parse_command                 # noqa: E402
from kia_core.memory import GlobalMemory                                     # noqa: E402
from kia_core.run_workspace import RunWorkspace                              # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {label}" + (f" — {detail}" if detail else ""))
    else:
        FAILED += 1
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# ==========================================================================
def test_math() -> None:
    print("\n[1] Ракетная математика")
    dv = rm.delta_v(345, 6.12, 2.12)
    check("Циолковский: dV ≈ 3587 м/с", close(dv, 3587, 15), f"{dv:.1f} м/с")
    check("mass_after_burn обратна dV", close(rm.mass_after_burn(6.12, dv, 345), 2.12, 0.02))
    check("TWR Swivel/10 т ≈ 2.19", close(rm.twr(215.0, 10.0), 2.19, 0.02))
    kerbin = get_body("Kerbin")
    check("g Кербина ≈ 9.81", close(kerbin.surface_gravity, 9.81, 0.02))
    v80 = rm.circular_speed(kerbin, 80_000)
    check("круговая скорость 80 км ≈ 2279 м/с", close(v80, 2279, 5), f"{v80:.1f}")
    r1 = kerbin.radius + 80_000
    r2 = get_body("Mun").orbit_radius
    dv1, dv2, tof = rm.hohmann_transfer(kerbin, r1, r2)
    check("инжекция к Муне ≈ 860 м/с", close(dv1, 860, 25), f"{dv1:.1f}")
    check("время перелёта ≈ 7.4 ч", close(tof, 26_700, 900), f"{tof/3600:.2f} ч")
    phase = math.degrees(rm.transfer_phase_angle(kerbin, r1, r2)) % 360
    check("фазовый угол ≈ 111°", close(phase, 111, 4), f"{phase:.1f}°")


# ==========================================================================
def test_catalog() -> None:
    print("\n[2] Каталог деталей из установки игры")
    catalog = get_catalog()
    stats = catalog.stats()
    check("каталог не пуст", stats["total"] > 0, f"{stats['total']} деталей")
    if stats["total"] == 0:
        return
    check("есть ЖРД", stats["engines"] > 0, f"{stats['engines']}")
    check("есть баки", stats["tanks"] > 0, f"{stats['tanks']}")
    check("есть разделители", stats["decouplers"] > 0, f"{stats['decouplers']}")
    check("есть командные модули", stats["pods"] > 0, f"{stats['pods']}")

    engines = catalog.engines(diameter=1.25)
    check("двигатели 1.25 м найдены", len(engines) > 0,
          ", ".join(e.name for e in engines[:3]))
    check("у двигателей есть Isp", all(e.isp_vac > 0 for e in engines))
    check("реактивные двигатели отфильтрованы",
          all(not e.is_air_breathing for e in engines))
    tanks = catalog.tanks(diameter=1.25)
    check("баки 1.25 м найдены", len(tanks) > 0, ", ".join(t.name for t in tanks[:3]))
    check("у баков есть узлы top/bottom",
          all(t.top_node and t.bottom_node for t in tanks))
    sample = tanks[0] if tanks else None
    if sample:
        check("имя для .craft заменяет '_' на '.'",
              sample.craft_id_name == sample.name.replace("_", "."),
              f"{sample.name} -> {sample.craft_id_name}")


# ==========================================================================
def test_mission_spec() -> None:
    print("\n[3] Разбор команд оператора")
    spec = parse_command("лети на Муну")
    check("«лети на Муну» -> цель Mun", spec.target_body == "Mun")
    spec = parse_command("выйди на орбиту 100 км")
    check("«орбиту 100 км» -> Кербин, 100 км",
          spec.target_body == "Kerbin" and close(spec.park_orbit_altitude, 100_000, 1),
          f"{spec.target_body}, {spec.park_orbit_altitude:.0f} м")
    spec = parse_command("сядь на Мун и вернись")
    check("посадка и возвращение распознаны",
          spec.needs_landing and spec.needs_return, ", ".join(spec.objectives))
    spec = parse_command("облёт Минмуса без экипажа")
    check("облёт без экипажа", spec.target_body == "Minmus" and not spec.crewed
          and not spec.needs_capture, spec.describe())


# ==========================================================================
def test_budget_and_design() -> None:
    print("\n[4] Автономный расчёт и подбор деталей")
    catalog = get_catalog()
    if not catalog.parts:
        check("каталог доступен", False, "нет деталей — проектирование пропущено")
        return

    spec = parse_command("лети на Муну, орбита 30 км")
    budget = compute_budget(spec, margin=1.10)
    check("бюджет содержит выведение и перелёт", len(budget.legs) >= 3,
          f"{len(budget.legs)} участков")
    check("суммарный ΔV в разумных пределах", 4_500 < budget.total < 7_500,
          f"{budget.total:.0f} м/с")
    check("у каждого участка есть формула",
          all(leg.formula and leg.numbers for leg in budget.legs))

    policy = DesignPolicy()
    reqs = plan_stages(spec, budget, policy)
    check("ступени распределены", len(reqs) >= 2, f"{len(reqs)} ступени")
    check("сумма ΔV ступеней ≈ бюджету",
          close(sum(r.dv_required for r in reqs), budget.total, budget.total * 0.02),
          f"{sum(r.dv_required for r in reqs):.0f} / {budget.total:.0f}")
    check("стартовой ступени задан TWR > 1",
          reqs[0].min_twr > 1.0, f"{reqs[0].min_twr:.2f}")

    design = AutonomousEngineer(catalog).design(spec, policy)
    check("проект жизнеспособен", design.viable, "; ".join(design.problems) or "без проблем")
    check("ΔV покрывает бюджет", design.total_delta_v >= design.budget.total,
          f"{design.total_delta_v:.0f} ≥ {design.budget.total:.0f}")
    # Потолок поднят с 2.5 до 2.7 по требованию оператора: связка обязана
    # быть из ЧЕТЫРЁХ ускорителей ради симметрии по обеим осям, а четыре
    # дают тяги чуть больше необходимого. Замер живых полётов: удачные
    # выведения шли при 2.28…2.52, разрушений от перетяга не было.
    check("стартовый TWR в норме", 1.15 <= design.launch_twr <= 2.7,
          f"{design.launch_twr:.2f}")
    check("каждая ступень обоснована",
          all(len(s.rationale) >= 5 for s in design.stages))
    check("массы ступеней убывают снизу вверх",
          all(a.wet_mass > b.wet_mass for a, b in zip(design.stages, design.stages[1:])))
    check("между ступенями стоят разделители",
          all(s.decoupler is not None for s in design.stages if s.requirement.index > 0))
    check("детали подобраны перебором каталога",
          all(s.candidates_examined > 10 for s in design.stages),
          f"{design.stages[0].candidates_examined} вариантов у 1-й ступени")

    # тяжёлая задача должна давать более тяжёлую ракету
    heavy = AutonomousEngineer(catalog).design(
        parse_command("сядь на Мун и вернись"), policy)
    if heavy.viable:
        check("посадочная задача тяжелее орбитальной",
              heavy.total_mass > design.total_mass,
              f"{heavy.total_mass:.1f} т против {design.total_mass:.1f} т")
    else:
        check("посадочная задача просчитана", True, "решения нет — отмечено в отчёте")


# ==========================================================================
def test_craft() -> None:
    print("\n[5] Генератор чертежей .craft")
    catalog = get_catalog()
    if not catalog.parts:
        check("каталог доступен", False)
        return
    design = AutonomousEngineer(catalog).design(parse_command("лети на Муну"),
                                                DesignPolicy())
    text = CraftAssembler("KIA_SelfTest", seed=1).render(design.blueprint)
    problems = validate_craft(text, catalog)
    check("чертёж проходит валидацию", not problems, "; ".join(problems[:2]))
    check("число блоков PART = числу деталей",
          text.count("PART\n{") == design.part_count,
          f"{text.count('PART')} / {design.part_count}")
    check("attN в формате KSP 1.12 (6 полей)",
          all(len(line.split(",", 1)[1].split("_")) == 6
              for line in text.splitlines() if line.strip().startswith("attN")))
    check("детали не висят в воздухе: у всех есть привязка",
          text.count("attN = ") + text.count("srfN = ") >= design.part_count - 1)
    check("разделители присутствуют",
          any("ecoupler" in line or "eparator" in line
              for line in text.splitlines() if line.strip().startswith("part =")))
    check("двигатель прицеплен к баку",
          any(line.strip().startswith("attN = top,fuelTank")
              or "attN = top," in line for line in text.splitlines()))

    # битый чертёж должен отлавливаться
    broken = text.replace("_0|", "|", 1)
    check("валидатор ловит битый attN", bool(validate_craft(broken, catalog)))


# ==========================================================================
def test_rewards() -> None:
    print("\n[6] Прейскурант и награды")
    with tempfile.TemporaryDirectory() as tmp:
        book = RewardBook(Path(tmp) / "book.json")
        base = book.milestone("stable_orbit")
        ok, msg = book.set_value("взрыв", -2000)
        check("«взрыв» -> explosion", ok and book.failure("explosion") == -2000, msg)
        ok, msg = book.set_value("орбита", 800)
        check("«орбита» -> stable_orbit", ok and book.milestone("stable_orbit") == 800,
              f"было {base:.0f}")
        ok, msg = book.set_value("ровный шаг тангажа", 500)
        check("правило тангажа добавлено", ok and "smooth_pitch" in book.rules, msg)
        book2 = RewardBook(Path(tmp) / "book.json")
        check("изменения сохраняются на диск",
              book2.failure("explosion") == -2000 and "smooth_pitch" in book2.rules)

        rewards = RewardSystem(book=book2, spec=MissionSpec())
        rewards.award(Milestone.STABLE_ORBIT)
        check("начислена изменённая цена", close(rewards.total, 800, 1e-6),
              f"{rewards.total:.0f}")
        rewards.award(Milestone.STABLE_ORBIT)
        check("повторное достижение не начисляется", close(rewards.total, 800, 1e-6))
        rewards.penalise(Failure.EXPLOSION)
        check("взрыв обрывает сессию и стоит -2000",
              rewards.terminated and close(rewards.total, -1200, 1e-6),
              f"{rewards.total:.0f}")

        # правила качества
        metrics = FlightMetrics()
        for i in range(30):
            metrics.pitch_samples.append((float(i), 90.0 - i * 3.0))
        applied = book2.evaluate_rules(metrics)
        smooth = next((a for a in applied if a[0] == "smooth_pitch"), None)
        check("ровный тангаж даёт высокое качество",
              smooth is not None and smooth[1] > 0.9, f"{smooth[1]:.2f}" if smooth else "—")
        metrics_rough = FlightMetrics()
        for i in range(30):
            metrics_rough.pitch_samples.append((float(i), 90.0 - (i % 5) * 12.0))
        rough = next((a for a in book2.evaluate_rules(metrics_rough)
                      if a[0] == "smooth_pitch"), None)
        check("рваный тангаж оценивается ниже ровного",
              rough is not None and smooth is not None and rough[1] < smooth[1],
              f"{rough[1]:.2f} против {smooth[1]:.2f}")


def test_telemetry_rewards() -> None:
    print("\n[7] Оценка телеметрии")
    from kia_core.environment.telemetry import TelemetrySnapshot
    spec = parse_command("лети на Муну")
    r = RewardSystem(spec=spec)
    r.evaluate_telemetry(TelemetrySnapshot(
        body="Kerbin", altitude=80_000, apoapsis=80_500, periapsis=79_500,
        speed=2280, situation="orbiting",
        resources={"ElectricCharge": {"amount": 100, "max": 100}}))
    check("орбита Кербина распознана", Milestone.STABLE_ORBIT in r.achieved,
          f"счёт {r.total:.0f}")
    r.evaluate_telemetry(TelemetrySnapshot(
        body="Mun", altitude=30_000, apoapsis=32_000, periapsis=28_000,
        eccentricity=0.05, situation="orbiting"))
    check("орбита Муны распознана", Milestone.MUN_ORBIT in r.achieved,
          f"счёт {r.total:.0f}")


# ==========================================================================
def test_flight_log() -> None:
    print("\n[8] Журнал полёта")
    with tempfile.TemporaryDirectory() as tmp:
        fl = FlightLog(Path(tmp) / "log.md", Path(tmp) / "log.json")
        fl.bind_clock(lambda: (1000.0, 42.0))
        fl.start("Орбита Муны", "KIA_Test", {"total_mass_t": 14.6})
        fl.status("Вывод на орбиту")
        fl.fact("Отделена 1 ступень", "тяга 60 кН")
        fl.reward("Достигнуто: stable_orbit", 500)
        fl.finish({"total_score": 650, "milestones": ["stable_orbit"], "failures": []})
        text = (Path(tmp) / "log.md").read_text(encoding="utf-8")
        check("файл создан и содержит статус", "Текущий статус" in text)
        check("есть раздел «Что сделано»", "Что сделано" in text)
        check("факт записан с временем", "Отделена 1 ступень" in text)
        check("UT игры попал в журнал", "1000" in text)
        check("итог записан", "650" in text)
        check("хвост журнала читается", "Отделена 1 ступень" in fl.tail(10))

        fl.start("Новая миссия", "KIA_Test2")
        text2 = (Path(tmp) / "log.md").read_text(encoding="utf-8")
        check("новый запуск пересоздаёт журнал",
              "Отделена 1 ступень" not in text2 and "Новая миссия" in text2)


# ==========================================================================
def test_learning_loop() -> None:
    print("\n[9] Конвейер обучения")
    with tempfile.TemporaryDirectory() as tmp:
        store = HistoryStore(Path(tmp) / "history.json")
        opt = GeneticOptimizer(seed=42)
        opt.cfg.population_size = 4
        opt.seed_population([Individual()])
        check("популяция сформирована", len(opt.population) == 4)

        for generation in range(3):
            for ind in opt.population:
                score = 1000 - abs(ind.flight.turn_start_altitude - 1000.0)
                opt.record_result(ind, score)
                store.append(RunRecord(run_id=store.next_run_id(),
                                       generation=generation, individual=ind.index,
                                       genome=ind.to_dict(), score=score))
            opt.evolve()

        check("история накопила 12 запусков", len(store.runs) == 12)
        check("оптимизатор дошёл до поколения 3", opt.generation == 3)
        best_error = abs(opt.best.flight.turn_start_altitude - 1000.0)
        first_error = abs(opt.history[0].best_individual.flight.turn_start_altitude - 1000.0)
        check("эволюция не ухудшила решение", best_error <= first_error,
              f"ошибка {first_error:.0f} -> {best_error:.0f} м")
        state = Path(tmp) / "state.json"
        opt.save_state(state)
        opt2 = GeneticOptimizer(seed=1)
        check("состояние восстанавливается",
              opt2.load_state(state) and opt2.generation == 3)


def test_genome_bounds() -> None:
    print("\n[10] Границы генома")
    import random
    from kia_core.engineer.autodesign import DesignPolicy as DP
    from kia_core.learning.genome import FlightGenome, jitter_individual
    rng = random.Random(5)
    ok = True
    for _ in range(200):
        ind = random_individual(rng)
        for section, bounds in ((ind.design, DP.bounds()),
                                (ind.flight, FlightGenome.bounds())):
            for name, (lo, hi) in bounds.items():
                if not (lo <= getattr(section, name) <= hi):
                    ok = False
    check("200 случайных особей внутри границ", ok)
    child = jitter_individual(Individual(), rng, scale=0.9)
    ok2 = all(lo <= getattr(child.flight, n) <= hi
              for n, (lo, hi) in FlightGenome.bounds().items())
    ok2 &= all(lo <= getattr(child.design, n) <= hi
               for n, (lo, hi) in DP.bounds().items())
    check("мутация не выходит за границы", ok2)


# ==========================================================================
def test_lambert() -> None:
    print("\n[11] Задача Ламберта")
    kerbin = get_body("Kerbin")
    mun = get_body("Mun")
    r_park = kerbin.radius + 80_000
    a_t = (r_park + mun.orbit_radius) / 2.0
    tof = math.pi * math.sqrt(a_t ** 3 / kerbin.mu)
    dv_hohmann, _, _ = rm.hohmann_transfer(kerbin, r_park, mun.orbit_radius)
    v_circ = math.sqrt(kerbin.mu / r_park)

    angle = math.radians(175.0)
    r1 = (r_park, 0.0, 0.0)
    r2 = (mun.orbit_radius * math.cos(angle), 0.0, -mun.orbit_radius * math.sin(angle))
    solution = rm.lambert(r1, r2, tof, kerbin.mu)
    check("решение найдено", solution is not None)
    if solution is None:
        return
    dv = math.sqrt(sum(c * c for c in solution[0])) - v_circ
    check("сходится к гомановскому импульсу", close(dv, dv_hohmann, 5.0),
          f"Ламберт {dv:.1f} против Гомана {dv_hohmann:.1f} м/с")
    fast = rm.lambert(r1, r2, tof * 0.7, kerbin.mu)
    if fast:
        dv_fast = math.sqrt(sum(c * c for c in fast[0])) - v_circ
        check("быстрый перелёт дороже", dv_fast > dv, f"{dv_fast:.0f} > {dv:.0f} м/с")
    dv_needed = rm.lambert_injection_dv(r1, (0.0, 0.0, -v_circ), r2, tof, kerbin.mu)
    check("импульс считается вектором", dv_needed is not None and dv_needed[0] > 0,
          f"{dv_needed[0]:.0f} м/с" if dv_needed else "—")


def test_diagnosis() -> None:
    print("\n[12] Разбор отказов и правка чертежа")
    analyst = FailureAnalyst()

    # падение при остатке топлива -> аэродинамическая неустойчивость
    record = {
        "milestones": ["viable_design", "liftoff", "cleared_tower"],
        "failures": ["crash"],
        "peak_altitude": 7800.0,
        "peak_apoapsis": 7900.0,
        "telemetry_final": {"situation": "landed", "vertical_speed": -180.0,
                            "resources": {"LiquidFuel": {"amount": 300, "max": 500}}},
        "metrics": {}, "events": [], "phase_log": [{"phase": "ascent"}],
    }
    d = analyst.analyze(record)
    causes = [x.cause for x in d]
    check("падение с топливом -> потеря устойчивости", "aero_instability" in causes,
          ", ".join(causes))
    fin = next((x for x in d if x.cause == "aero_instability"), None)
    check("назначены стабилизаторы", fin is not None and fin.policy_delta.get("fin_level", 0) > 0)
    check("разворот отложен выше",
          fin is not None and fin.flight_delta.get("turn_start_altitude", 0) > 0)

    # не оторвался
    d2 = analyst.analyze({"milestones": [], "failures": ["no_liftoff"],
                          "events": [{"label": "no_liftoff", "detail": {"twr": 0.9}}],
                          "telemetry_final": {}, "metrics": {}})
    check("не взлетел -> поднять TWR",
          any(x.policy_delta.get("liftoff_twr", 0) > 0 for x in d2))

    # топливо кончилось
    d3 = analyst.analyze({"milestones": ["liftoff", "atmosphere_exit"],
                          "failures": ["out_of_fuel"], "peak_apoapsis": 60_000,
                          "telemetry_final": {"resources": {"LiquidFuel": {"amount": 0, "max": 500}}},
                          "metrics": {}})
    check("нет топлива -> увеличить запас ΔV",
          any(x.policy_delta.get("dv_margin", 0) > 0 for x in d3))

    # долетели, но не затормозились
    d4 = analyst.analyze({"milestones": ["stable_orbit", "mun_node_created",
                                         "mun_encounter", "mun_soi_entered"],
                          "failures": [], "telemetry_final": {}, "metrics": {}})
    check("нет захвата -> усилить перелётную ступень",
          any(x.cause == "no_capture" for x in d4))

    # успех с запасом -> облегчение
    d5 = analyst.analyze({"milestones": ["stable_orbit", "crew_survived"],
                          "failures": [],
                          "telemetry_final": {"resources": {"LiquidFuel": {"amount": 400, "max": 500}}},
                          "metrics": {}})
    check("избыточный запас -> экономия массы",
          any(x.cause == "overbuilt" for x in d5))

    # правки применяются и не выходят за границы
    from kia_core.engineer.autodesign import DesignPolicy as DP
    from kia_core.learning.genome import FlightGenome
    ind = Individual()
    before = ind.design.fin_level
    changes = FailureAnalyst.apply(ind, d)
    check("правки применены к особи", len(changes) > 0, f"{len(changes)} параметров")
    check("стабилизация выросла", ind.design.fin_level > before,
          f"{before:.2f} -> {ind.design.fin_level:.2f}")
    ok = all(lo <= getattr(ind.design, n) <= hi for n, (lo, hi) in DP.bounds().items())
    ok &= all(lo <= getattr(ind.flight, n) <= hi
              for n, (lo, hi) in FlightGenome.bounds().items())
    check("после правок геном внутри границ", ok)

    # десять разборов подряд не выводят политику за пределы
    for _ in range(10):
        FailureAnalyst.apply(ind, analyst.analyze(record))
    ok2 = all(lo <= getattr(ind.design, n) <= hi for n, (lo, hi) in DP.bounds().items())
    check("десять итераций правок стабильны", ok2,
          f"fin_level={ind.design.fin_level:.2f}, TWR={ind.design.liftoff_twr:.2f}")


def test_autonomy_loop() -> None:
    print("\n[13] Автономный цикл (офлайн)")
    from kia_core.agent.autonomy import AutonomousLoop
    from kia_core.agent.dispatcher import Dispatcher
    import tempfile as tf
    from kia_core.config import DATA_DIR

    sandbox = Path(tf.mkdtemp())
    dispatcher = Dispatcher(offline=True, resume=False, seed=7)
    dispatcher.history = HistoryStore(sandbox / "history.json")
    # тест не должен засорять рабочую папку kia/ настоящими запусками
    dispatcher.workspace = RunWorkspace(sandbox / "kia")
    from kia_core.memory import GlobalMemory
    dispatcher.memory = GlobalMemory(sandbox / "memory.json")
    loop = AutonomousLoop(dispatcher, max_iterations=3)
    loop.iterations = []
    summary = loop.run()
    check("цикл выполнил итерации", summary["iterations"] >= 3,
          f"{summary['iterations']}")
    check("у каждой итерации есть диагноз",
          all(it.diagnoses for it in loop.iterations))
    check("каждая итерация вносит правки",
          all(it.changes for it in loop.iterations[:-1]) or len(loop.iterations) == 1)
    report = loop.write_report(sandbox / "autonomy_report.md")
    text = report.read_text(encoding="utf-8")
    check("отчёт автономного режима написан", "Ход обучения" in text)
    check("в отчёте есть разбор", "Последний разбор" in text)


# ==========================================================================
def test_uncrewed_design() -> None:
    print("\n[14] Беспилотная конструкция")
    catalog = get_catalog()
    if not catalog.parts:
        check("каталог доступен", False)
        return
    spec = parse_command("лети на Муну с экипажем")
    check("запрос экипажа отклонён архитектурой", not spec.crewed)
    design = AutonomousEngineer(catalog).design(spec, DesignPolicy())
    bp = design.blueprint
    check("корневая деталь — командный модуль", bp.pod.is_command, bp.pod.name)
    check("мест для экипажа нет", bp.pod.crew_capacity == 0,
          f"crew_capacity={bp.pod.crew_capacity}")
    check("у ядра есть свой запас энергии",
          bp.pod.resources.get("ElectricCharge", 0) > 0,
          f"{bp.pod.resources.get('ElectricCharge', 0):.0f} ЕЭ")
    parts = bp.all_parts()
    check("есть источник энергии в составе",
          any(p.resources.get("ElectricCharge", 0) > 0 for p in parts))
    check("есть солнечная панель", any(p.is_solar for p in parts))
    check("есть антенна", any(p.is_antenna for p in parts))
    check("парашют не занимает верхний узел ядра",
          bp.parachute is None or bp.parachute.bottom_node is not None)

    text = CraftAssembler("KIA_Uncrewed", seed=4).render(bp)
    problems = validate_craft(text, catalog)
    check("чертёж беспилотника валиден", not problems, "; ".join(problems[:2]))
    first_part = next(l for l in text.splitlines() if l.strip().startswith("part ="))
    check("командный модуль записан первым (корень дерева)",
          bp.pod.craft_id_name in first_part, first_part.strip())
    check("версия KSP в заголовке",
          any(l.startswith("version = ") and l.split("=")[1].strip()[0].isdigit()
              for l in text.splitlines()))

    # валидатор обязан ловить именно «пустую площадку»
    broken = text.replace("	link = ", "	linkX = ", 1)
    problems = validate_craft(broken, catalog)
    check("валидатор ловит разрыв дерева (пустая площадка)",
          any("не связаны с корнем" in p for p in problems),
          problems[0][:60] if problems else "не поймано")


def test_run_workspace() -> None:
    print("\n[15] Папки запусков kia/<N>/")
    with tempfile.TemporaryDirectory() as tmp:
        workspace = RunWorkspace(Path(tmp) / "kia")
        first = workspace.create()
        check("первая папка — kia/1", first.number == 1, first.path.name)
        first.design_report.write_text("отчёт", encoding="utf-8")
        first.flight_log.write_text("журнал", encoding="utf-8")
        first.write_meta({"score": 123, "outcome": "orbit"})

        second = workspace.create()
        check("вторая папка — kia/2", second.number == 2)
        check("старые данные на месте",
              first.design_report.read_text(encoding="utf-8") == "отчёт")
        check("в папке ровно нужные файлы",
              first.design_report.exists() and first.flight_log.exists(),
              ", ".join(sorted(p.name for p in first.path.iterdir())))
        check("последняя папка определяется", workspace.latest().number == 2)
        check("метаданные читаются", first.read_meta().get("score") == 123)

        third = RunWorkspace(Path(tmp) / "kia").create()
        check("новый экземпляр продолжает нумерацию", third.number == 3)


def test_global_memory() -> None:
    print("\n[16] Глобальная долгосрочная память")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kia_global_memory.json"
        memory = GlobalMemory(path)
        ind = Individual()
        ind.flight.turn_start_altitude = 900.0
        ind.flight.turn_exponent = 0.62
        ind.design.liftoff_twr = 1.62

        fresh = memory.record_success(["liftoff", "stable_orbit"], ind, 850.0,
                                      "Kerbin/80km", "тестовый запуск")
        check("навыки записаны", set(fresh) == {"liftoff", "stable_orbit"},
              ", ".join(fresh))
        check("параметры успеха сохранены",
              memory.skills["stable_orbit"].parameters["flight"]["turn_exponent"] == 0.62)

        reloaded = GlobalMemory(path)
        check("память переживает перезапуск процесса",
              reloaded.has_skill("stable_orbit") and reloaded.total_runs == 1)

        # применение к новой особи
        fresh_ind = Individual()
        fresh_ind.flight.turn_start_altitude = 250.0
        applied = reloaded.apply_to(fresh_ind, strength=1.0)
        check("подтверждённый профиль переносится в новое поколение",
              abs(fresh_ind.flight.turn_start_altitude - 900.0) < 1.0,
              f"{fresh_ind.flight.turn_start_altitude:.0f} м")
        check("применены и параметры конструктора",
              abs(fresh_ind.design.liftoff_twr - 1.62) < 0.01)

        # сужение границ вокруг доказанного
        from kia_core.learning.genome import FlightGenome
        bounds = FlightGenome.bounds()
        narrowed = reloaded.constrain_bounds(bounds, "flight", width=0.3)
        lo, hi = narrowed["turn_exponent"]
        check("границы сужены вокруг доказанного значения",
              lo <= 0.62 <= hi and (hi - lo) < (bounds["turn_exponent"][1]
                                                - bounds["turn_exponent"][0]),
              f"[{lo:.2f}; {hi:.2f}] вместо [{bounds['turn_exponent'][0]:.2f}; "
              f"{bounds['turn_exponent'][1]:.2f}]")

        # уроки отказов блокируют известную ошибку
        bad = Individual()
        bad.design.liftoff_twr = 1.30
        reloaded.record_failure_lesson("no_liftoff", bad, "TWR 0.9")
        risky = Individual()
        risky.design.liftoff_twr = 1.25
        check("память предупреждает о старой ошибке",
              reloaded.forbidden_zone(risky) is not None,
              (reloaded.forbidden_zone(risky) or "")[:60])
        safe = Individual()
        safe.design.liftoff_twr = 1.9
        safe.design.fin_level = 0.8
        check("безопасная особь не блокируется",
              reloaded.forbidden_zone(safe) is None)

        # ветки по целям
        reloaded.remember_best("Mun/80km", ind, 1200.0)
        reloaded.remember_best("Kerbin/80km", ind, 800.0)
        check("ветки целей раздельные", len(reloaded.best_per_goal) == 2,
              ", ".join(reloaded.best_per_goal))
        seed = reloaded.branch_seed("Mun/80km")
        check("геном ветки восстанавливается", seed is not None
              and abs(seed.flight.turn_exponent - 0.62) < 1e-6)
        check("сводка памяти читаема", "Подтверждённые навыки" in reloaded.summary())


# ==========================================================================
def test_neural_core() -> None:
    print("\n[17] Нейросеть: архитектура и действия")
    from kia_core.neural import is_available
    if not is_available():
        check("PyTorch установлен", False, "нейромодуль пропущен")
        return
    import torch
    from kia_core.neural.device import inspect as device_inspect
    from kia_core.neural.networks import ActorCritic, RunningNormalizer
    from kia_core.neural.spaces import (CONTINUOUS_DIM, FLIGHT_OBS_DIM,
                                        FlightAction)

    info = device_inspect()
    check("устройство определено", info.available, info.describe()[:70])

    net = ActorCritic(FLIGHT_OBS_DIM, CONTINUOUS_DIM, (2,), hidden=64, depth=2)
    check("сеть построена", net.parameter_count() > 1000,
          f"{net.parameter_count()} весов")

    obs = torch.randn(4, FLIGHT_OBS_DIM)
    continuous, discrete, logp, value = net.act(obs)
    check("форма непрерывных действий", tuple(continuous.shape) == (4, CONTINUOUS_DIM),
          str(tuple(continuous.shape)))
    check("действия в границах [-1,1]",
          bool((continuous.abs() <= 1.0 + 1e-6).all()))
    check("дискретное действие в диапазоне",
          bool(((discrete >= 0) & (discrete < 2)).all()))
    check("критик выдаёт число на состояние", tuple(value.shape) == (4,))

    logp2, entropy, value2 = net.evaluate(obs, continuous, discrete)
    check("оценка действий совпадает по форме", tuple(logp2.shape) == (4,))
    check("энтропия положительна", bool((entropy > 0).all()))

    action = FlightAction.decode([2.0, -3.0, 0.0, 0.5], [1])
    check("тяга переводится в [0,1]", 0.0 <= action.throttle <= 1.0,
          f"{action.throttle:.2f}")
    check("оси обрезаются", action.pitch == -1.0 and action.roll == 0.5)
    check("флаг отделения читается", action.stage is True)

    norm = RunningNormalizer(3)
    for _ in range(50):
        norm.update(torch.randn(16, 3) * 5.0 + 2.0)
    scaled = norm(torch.full((1, 3), 2.0))
    check("нормализация сдвигает среднее к нулю",
          bool(scaled.abs().max() < 1.0), f"{scaled.abs().max().item():.2f}")

    # Выброс в накопительной статистике убивает признак навсегда: рабочий
    # диапазон схлопывается в тысячные доли сигмы, и сеть перестаёт
    # различать 0 и 1 по этому входу. Так в живых весах умерли апоапсис,
    # периапсис и «прогресс к цели» — из-за апоапсиса 1e9 на гиперболе.
    poisoned = RunningNormalizer(1)
    for _ in range(200):
        poisoned.update(torch.rand(16, 1))
    poisoned.update(torch.full((16, 1), 10_000.0))
    span = 1.0 / float(poisoned.var[0]) ** 0.5
    check("один выброс давит признак в тысячные доли сигмы", span < 0.01,
          f"рабочий диапазон {span:.5f} сигмы")
    poisoned.reset()
    check("сброс статистики возвращает признак к жизни",
          abs(float(poisoned.mean[0])) < 1e-9
          and abs(float(poisoned.var[0]) - 1.0) < 1e-9)

    # Тот же выброс на входе кодировщика — обрезается до разумного, и это
    # общая защита для тренажёра и живой KSP: обе гоняют один encoder.
    from kia_core.neural.sim_env import SimSnapshot
    from kia_core.neural.spaces import (FLIGHT_FEATURES,
                                        OBSERVED_APOAPSIS_LIMIT,
                                        OBSERVED_FEATURE_LIMIT,
                                        encode_flight_state)
    names = [name for name, _ in FLIGHT_FEATURES]
    hyperbolic = SimSnapshot(altitude=300_000.0, apoapsis=1e9,
                             periapsis=-6_123_322.0, eccentricity=1.4,
                             time_to_apoapsis=1e9 / 3.0, speed=2_910.0)
    vector = encode_flight_state(hyperbolic, 80_000.0, body_gravity=9.81,
                                 body_radius=600_000.0, has_atmosphere=True)
    worst = max(abs(value) for value in vector)
    check("гиперболический апоапсис не проходит в наблюдение сырым",
          worst <= OBSERVED_APOAPSIS_LIMIT + 1e-6
          and vector[names.index("apoapsis")] < 10.0,
          f"самый крупный признак {worst:.1f} "
          f"(апоапсис {vector[names.index('apoapsis')]:.1f}, "
          f"прогресс {vector[names.index('goal_progress')]:.1f}, "
          f"время до апоапсиса "
          f"{vector[names.index('time_to_apoapsis')]:.1f})")

    # Последняя линия обороны: что бы ни пришло из телеметрии, признак не
    # уходит за предел. Проверяем на заведомом мусоре во всех полях.
    garbage = SimSnapshot(altitude=1e12, apoapsis=1e12, periapsis=-1e12,
                          speed=1e9, mass=1e9, dynamic_pressure=1e12,
                          time_to_apoapsis=1e12, g_force=1e6)
    vector = encode_flight_state(garbage, 80_000.0, body_gravity=9.81,
                                 body_radius=600_000.0, has_atmosphere=True)
    check("мусор в телеметрии не проходит в наблюдение",
          max(abs(value) for value in vector) <= OBSERVED_FEATURE_LIMIT + 1e-6,
          f"предел ±{OBSERVED_FEATURE_LIMIT:.0f}, максимум "
          f"{max(abs(v) for v in vector):.1f}")


def test_neural_ppo() -> None:
    print("\n[18] Нейросеть: обучение PPO")
    from kia_core.neural import is_available
    if not is_available():
        check("PyTorch установлен", False)
        return
    import torch
    from kia_core.neural.ppo import PPOAgent, PPOConfig, RolloutBuffer

    agent = PPOAgent(obs_dim=6, continuous_dim=2, discrete_sizes=(2,),
                     config=PPOConfig(epochs=4, minibatch_size=32),
                     hidden=32, name="тест")

    # Задача-игрушка: награда выше, когда первое действие близко к +1.
    buffer = RolloutBuffer(6, 2, 1, 256, agent.device)
    before = [p.detach().clone() for p in agent.net.parameters()]
    for _ in range(256):
        obs_vec = [0.5] * 6
        obs, continuous, discrete, logp, value = agent.act(obs_vec)
        reward = float(continuous[0].item())
        buffer.add(obs, continuous, discrete, logp, value, reward, False)
    metrics = agent.update(buffer)
    check("шаг обучения выполнен", "policy_loss" in metrics,
          f"KL={metrics.get('approx_kl', 0):.4f}")
    after = list(agent.net.parameters())
    changed = any(not torch.allclose(a, b) for a, b in zip(before, after))
    check("веса изменились градиентным спуском", changed)
    check("счётчики обновились", agent.stats.updates == 1
          and agent.stats.total_steps == 256)

    # политика должна сдвинуться в сторону награды
    _, continuous_after, _, _, _ = agent.act([0.5] * 6, deterministic=True)
    for _ in range(6):
        buffer.clear()
        for _ in range(256):
            obs_vec = [0.5] * 6
            obs, continuous, discrete, logp, value = agent.act(obs_vec)
            buffer.add(obs, continuous, discrete, logp, value,
                       float(continuous[0].item()), False)
        agent.update(buffer)
    _, final_action, _, _, _ = agent.act([0.5] * 6, deterministic=True)
    check("политика двигается к награде",
          float(final_action[0].item()) > float(continuous_after[0].item()),
          f"{continuous_after[0].item():+.3f} -> {final_action[0].item():+.3f}")

    # GAE на простом примере
    buffer2 = RolloutBuffer(1, 0, 1, 3, agent.device)
    for reward in (1.0, 1.0, 1.0):
        buffer2.add(torch.zeros(1, device=agent.device), None,
                    torch.zeros(1, dtype=torch.long, device=agent.device),
                    torch.tensor(0.0, device=agent.device),
                    torch.tensor(0.0, device=agent.device), reward, False)
    advantages, returns = buffer2.compute_returns(0.0, gamma=1.0, lam=1.0)
    check("GAE считает накопленную награду",
          abs(float(returns[0].item()) - 3.0) < 1e-5,
          f"{float(returns[0].item()):.2f} (ожидалось 3.00)")

    # --- пачка сред: преимущества считаются по каждой копии отдельно ---
    # Если склеить копии в одну ленту, награда одной ракеты потечёт в
    # оценку другой. Ошибка молчаливая: обучение просто станет считать по
    # чужим данным. Поэтому сверяемся с одиночным буфером напрямую.
    from kia_core.neural.ppo import VectorRolloutBuffer
    single = RolloutBuffer(1, 0, 1, 4, agent.device)
    vector = VectorRolloutBuffer(1, 0, 1, 4, agent.device, envs=2)
    zero_obs = torch.zeros(1, device=agent.device)
    zero_act = torch.zeros(1, dtype=torch.long, device=agent.device)
    scalar = torch.tensor(0.0, device=agent.device)
    rewards_a = [1.0, 2.0, 3.0, 4.0]
    ends_a = [False, True, False, False]
    for reward, done in zip(rewards_a, ends_a):
        single.add(zero_obs, None, zero_act, scalar, scalar, reward, done)
    single_adv, single_ret = single.compute_returns(0.0, 0.99, 0.95)
    for index, (reward, done) in enumerate(zip(rewards_a, ends_a)):
        vector.add_batch(torch.zeros(2, 1, device=agent.device), None,
                         torch.zeros(2, 1, dtype=torch.long, device=agent.device),
                         torch.zeros(2, device=agent.device),
                         torch.zeros(2, device=agent.device),
                         [reward, -reward], [done, False])
    vector_adv, _ = vector.compute_returns([0.0, 0.0], 0.99, 0.95)
    first_copy = vector_adv.reshape(4, 2)[:, 0]
    check("в пачке сред преимущества совпадают с одиночным расчётом",
          torch.allclose(first_copy, single_adv, atol=1e-5),
          f"{[round(v, 3) for v in first_copy.tolist()]} против "
          f"{[round(v, 3) for v in single_adv.tolist()]}")
    second_copy = vector_adv.reshape(4, 2)[:, 1]
    check("конец эпизода в одной копии не рвёт ленту соседней",
          not torch.allclose(second_copy, -first_copy, atol=1e-3),
          "у копий разные обрывы — значит расчёт идёт раздельно")
    check("пачка считает переходы как шаги × копии",
          vector.transitions == 8, f"{vector.transitions}")

    # --- ловушка разброса действий -----------------------------------
    # Самая дорогая ошибка проекта: ограничитель log_std стоял только на
    # выходе (torch.clamp), а он обнуляет градиент за пределами диапазона.
    # Параметр однажды вылез за потолок и застрял там навсегда — сеть
    # 615 обновлений подряд не могла уменьшить разброс действий и так и
    # не научилась летать без шума. Эти проверки не дают повторить.
    net = agent.net
    with torch.no_grad():
        net.log_std.fill_(5.0)              # загоняем далеко за потолок
    net.clamp_log_std()
    check("разброс действий возвращается в диапазон В САМОМ параметре",
          float(net.log_std.max().item()) <= net.LOG_STD_MAX + 1e-6,
          f"log_std {float(net.log_std.max().item()):+.3f} при потолке "
          f"{net.LOG_STD_MAX}")

    net.zero_grad(set_to_none=True)
    normal, _, _ = net.distributions(agent.observe([0.5] * 6))
    normal.log_prob(torch.zeros_like(normal.mean)).sum().backward()
    gradient = float(net.log_std.grad.abs().max().item())
    check("градиент до разброса действий доходит (ловушка снята)",
          gradient > 1e-9, f"|d/d log_std| = {gradient:.4f}")

    # обновление не должно выпускать параметр за потолок
    buffer.clear()
    for _ in range(64):
        obs, continuous, discrete, logp, value = agent.act([0.5] * 6)
        buffer.add(obs, continuous, discrete, logp, value, 1.0, False)
    agent.update(buffer)
    check("после шага оптимизатора разброс остаётся в диапазоне",
          net.LOG_STD_MIN - 1e-6 <= float(net.log_std.min().item())
          and float(net.log_std.max().item()) <= net.LOG_STD_MAX + 1e-6,
          f"log_std {float(net.log_std.min().item()):+.3f}..."
          f"{float(net.log_std.max().item()):+.3f}")

    # старый испорченный чекпоинт должен чиниться при загрузке
    broken = PPOAgent(6, 2, (2,), PPOConfig(), hidden=32, name="испорченный")
    with torch.no_grad():
        broken.net.log_std.fill_(0.201)     # ровно как в реальном файле
    payload = broken.state_dict()
    repaired = PPOAgent(6, 2, (2,), PPOConfig(), hidden=32, name="починенный")
    repaired.load_state_dict(payload)
    check("испорченный чекпоинт чинится при загрузке",
          float(repaired.net.log_std.max().item()) <= repaired.net.LOG_STD_MAX + 1e-6,
          f"+0.201 -> {float(repaired.net.log_std.max().item()):+.3f}")


def test_neural_checkpoint() -> None:
    print("\n[19] Нейросеть: сохранение опыта")
    from kia_core.neural import is_available
    if not is_available():
        check("PyTorch установлен", False)
        return
    import torch
    from kia_core.neural.ppo import Brain, PPOAgent, PPOConfig

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kia_model.pth"
        agent = PPOAgent(6, 2, (2,), PPOConfig(), hidden=32, name="сохранение")
        agent.stats.updates = 7
        agent.stats.total_steps = 1234
        agent.stats.episodes = 42
        agent.record_episode(15.5)
        brain = Brain(agent, None, path)
        brain.save()
        check("файл весов создан", path.exists(),
              f"{path.stat().st_size // 1024} КБ")

        restored_agent = PPOAgent(6, 2, (2,), PPOConfig(), hidden=32,
                                  name="восстановление")
        restored = Brain(restored_agent, None, path)
        check("веса загружены", restored.load())
        check("счётчики восстановлены",
              restored_agent.stats.updates == 7
              and restored_agent.stats.total_steps == 1234
              and restored_agent.stats.episodes == 43,   # 42 + записанный эпизод
              f"обновлений {restored_agent.stats.updates}, "
              f"шагов {restored_agent.stats.total_steps}, "
              f"эпизодов {restored_agent.stats.episodes}")
        check("рекорд награды сохранён",
              abs(restored_agent.stats.best_return - 15.5) < 1e-6)
        same = all(torch.allclose(a, b) for a, b in
                   zip(agent.net.parameters(), restored_agent.net.parameters()))
        check("веса совпадают до последнего числа", same)

        # несовместимая размерность не должна ломать запуск
        other = PPOAgent(8, 2, (2,), PPOConfig(), hidden=32, name="другой")
        loaded = Brain(other, None, path).load()
        check("смена входа обрабатывается без падения", loaded is False)


def test_neural_environments() -> None:
    print("\n[20] Нейросеть: среды обучения")
    from kia_core.neural import is_available
    if not is_available():
        check("PyTorch установлен", False)
        return
    from kia_core.neural.builder_env import (FINISH_ACTION, BuilderEnv,
                                             build_palette)
    from kia_core.neural.sim_env import AscentSimulator
    from kia_core.neural.spaces import FLIGHT_OBS_DIM, FlightAction

    env = AscentSimulator(seed=3, exploring_starts=0.0)
    obs = env.reset()
    check("наблюдение нужной длины", len(obs) == FLIGHT_OBS_DIM,
          f"{len(obs)} чисел")
    check("значения нормированы",
          all(abs(v) < 50 for v in obs), f"max={max(abs(v) for v in obs):.1f}")

    def run(policy, steps=2500):
        e = AscentSimulator(seed=3, exploring_starts=0.0)
        e.reset()
        total = 0.0
        for i in range(steps):
            _, reward, done, _ = e.step(policy(e, i))
            total += reward
            if done:
                break
        return total, e

    def good(e, i):
        """Эталонный профиль в три этапа: подъём, выбег, скругление.

        Нужен, чтобы проверить: задача в тренажёре решаема в принципе.
        Сеть этого профиля не видит и учится сама.
        """
        from kia_core.neural.sim_env import KERBIN_ATMO_HEIGHT
        h = e.altitude
        apoapsis, periapsis, _ = e.orbital_elements()
        if periapsis < KERBIN_ATMO_HEIGHT and h > KERBIN_ATMO_HEIGHT * 0.95:
            throttle, target = 1.0, 0.0          # горизонтальный ожог
        elif apoapsis < e.goal_altitude:
            throttle = 1.0
            target = 90.0 if h < 1000 else max(5.0, 90.0 * (1.0 - (h / 45000.0) ** 0.6))
        else:
            throttle, target = 0.0, 0.0          # выбег до космоса
        return FlightAction(throttle, max(-1, min(1, (target - e.pitch) / 8.0)),
                            0, 0,
                            e.stages[e.stage_index].fuel_mass <= 0
                            and e.stage_index == 0)

    def idle(e, i):
        return FlightAction(0, 0, 0, 0, False)

    good_return, good_env = run(good)
    idle_return, idle_env = run(idle)
    check("грамотный профиль награждается больше бездействия",
          good_return > idle_return + 10,
          f"{good_return:.1f} против {idle_return:.1f}")

    # Ракеты в тренажёре случайные, и один жёсткий профиль тангажа
    # подходит не каждой. Проверяем, что задача в принципе решаема:
    # хотя бы на части аппаратов этот профиль выводит на орбиту.
    orbits = 0
    for seed in range(8):
        e = AscentSimulator(seed=seed, exploring_starts=0.0)
        e.reset()
        for i in range(2500):
            _, _, done, _ = e.step(good(e, i))
            if done:
                break
        orbits += 1 if e.reached_orbit else 0
    check("тренажёр решаем: эталонный профиль выводит на орбиту", orbits >= 6,
          f"орбит {orbits} из 8 аппаратов")
    check("без тяги аппарат стоит на столе", idle_env.peak_altitude < 1.0)

    # конструктор
    catalog_ok = True
    try:
        palette = build_palette()
    except Exception as exc:
        catalog_ok = False
        palette = None
        check("палитра собрана", False, str(exc)[:60])
    if catalog_ok:
        check("палитра фиксированного размера", len(palette) > 0,
              f"{len(palette)} деталей")
        builder = BuilderEnv()
        builder.reset()
        mask = builder.action_mask()
        allowed = [i for i, ok in enumerate(mask) if ok]
        check("первым шагом разрешено только командное ядро",
              all(palette.parts[i].is_command for i in allowed if i < len(palette)),
              f"{len(allowed)} вариантов")
        check("закончить пустую сборку нельзя", not mask[FINISH_ACTION])

        core_index = allowed[0]
        builder.step(core_index)
        mask2 = builder.action_mask()
        check("второе ядро запрещено",
              not any(mask2[i] for i, p in enumerate(palette.parts)
                      if p.is_command))
        _, reward, done, info = builder.step(FINISH_ACTION)
        check("оценка сборки возвращается", done and "evaluation" in info)
        check("сборка без двигателя признана негодной",
              not info["evaluation"].valid, info["evaluation"].describe()[:60])


# ==========================================================================
def reference_pilot(env, cascade: bool = False):
    """Эталонный профиль выведения — проверка, что задача решаема.

    Сеть этого профиля не видит: он нужен только тесту, чтобы отличить
    «политика не научилась» от «в среде вообще нельзя выйти на орбиту».
    cascade=True — вариант для Тренажёра++, где рули задают момент, а не
    угол: цель по тангажу превращается в цель по угловой скорости.
    """
    from kia_core.neural.sim_env import KERBIN_ATMO_HEIGHT
    from kia_core.neural.spaces import FlightAction

    altitude = env.altitude
    apoapsis, periapsis, _ = env.orbital_elements()
    if periapsis < KERBIN_ATMO_HEIGHT and altitude > KERBIN_ATMO_HEIGHT * 0.95:
        throttle, target = 1.0, 0.0                # горизонтальный ожог
    elif apoapsis < env.goal_altitude:
        throttle = 1.0
        target = (90.0 if altitude < 1000 else
                  max(5.0, 90.0 * (1.0 - (altitude / 45000.0) ** 0.6)))
    else:
        throttle, target = 0.0, 0.0                # выбег до космоса

    stage = env.stages[env.stage_index].fuel_mass <= 1.0 and env.stage_index == 0
    if not cascade:
        return FlightAction(throttle, max(-1, min(1, (target - env.pitch) / 8.0)),
                            0, 0, stage)

    pitch_error, _ = env.attitude_error()
    if env.dynamic_pressure() > 2_000.0:
        flight = env.pitch - pitch_error
        target = max(flight - 4.0, min(flight + 4.0, target))
    rate_target = max(-6.0, min(6.0, 0.5 * (target - env.pitch)))
    command = 0.12 * (rate_target - env.pitch_rate)
    drift = (env.heading - 90.0 + 180) % 360 - 180
    yaw = -0.10 * env.yaw_rate - 0.03 * drift
    return FlightAction(throttle, max(-1, min(1, command)),
                        max(-1, min(1, yaw)), 0, stage)


def fly_reference(stage, seed: int, cascade: bool = False,
                  difficulty: float | None = None):
    """Прогоняет эталонный профиль по ступени так же, как это делает тренер."""
    from kia_core.neural.spaces import FlightAction
    env = stage.make_env(seed=seed, exploring_starts=0.0)
    if difficulty is not None:
        env.set_difficulty(difficulty)
    env.reset()
    done = False
    total = 0.0
    while not done and env.time < env.max_time:
        action = reference_pilot(env, cascade=cascade)
        for repeat in range(stage.action_repeat):
            step = action if repeat == 0 else FlightAction(
                action.throttle, action.pitch, action.yaw, action.roll, False)
            _, reward, done, _ = env.step(step)
            total += reward
            if done:
                break
    return env, total


def test_curriculum() -> None:
    """[22] Учебная программа: три ступени тренажёра."""
    print("\n[22] Учебная программа: ступени и скрытые механики")
    from kia_core.neural import is_available
    if not is_available():
        check("PyTorch установлен", False)
        return
    from kia_core.neural.curriculum import (STAGES, checkpoint_info, prepare,
                                            progress, resolve)
    from kia_core.neural.sim_env import AscentSimulator, SimStage
    from kia_core.neural.sim_env_plus import HardcoreAscentSimulator
    from kia_core.neural.sim_env_pro import PrecisionAscentSimulator
    from kia_core.neural.spaces import FLIGHT_OBS_DIM, FlightAction

    check("ступеней ровно четыре", len(STAGES) == 4,
          ", ".join(s.title for s in STAGES))
    check("ступень находится по номеру и по имени",
          resolve(2).key == "plus" and resolve("pro").number == 3
          and resolve("max").number == 4)
    check("порядок снимков весов задан",
          STAGES[0].backup is None
          and STAGES[1].backup.name == "kia_model_stage1.pth"
          and STAGES[2].backup.name == "kia_model_stage2.pth"
          and STAGES[3].backup.name == "kia_model_stage3.pth")
    # Ради этого разделения всё и затевалось: за одну ступень должно
    # меняться ОДНО. Живой прогон показал, что смена физики и смены
    # закона управления разом политику убивает.
    check("физика и закон управления разведены по разным ступеням",
          dict(STAGES[2].env_options).get("assist_floor") == 1.0
          and dict(STAGES[3].env_options).get("assist_floor") == 0.0,
          "Тренажёр++ — привычные руки, Тренажёр+++ — момент")
    # ГЛАВНОЕ ПРАВИЛО ПРОГРАММЫ: мир один, трёхмерный, с первой ступени.
    # Пока первые две ступени были плоскими, сеть приходила на третью с
    # мёртвыми осями на упоре и давала ноль орбит — при том что та же
    # ступень с чистого листа выучивалась за четыре минуты (164 орбиты).
    check("все ступени в одном трёхмерном мире",
          all(stage.module == "sim_env_pro"
              and stage.factory == "PrecisionAscentSimulator"
              for stage in STAGES),
          ", ".join(sorted({stage.module for stage in STAGES})))
    check("курс и крен живые с первой ступени",
          all(dict(stage.env_options).get("yaw_fade") is False
              for stage in STAGES),
          "плавно вводить нечего — они работают везде")
    check("интерфейс управления портится ровно на второй ступени",
          dict(STAGES[0].env_options).get("mechanics") is False
          and all(dict(stage.env_options).get("mechanics") is True
                  for stage in STAGES[1:]),
          "первая — идеальные команды, дальше задержка, ветер, батарея")

    # САМОЕ ВАЖНОЕ ПРО СТУПЕНИ: мир у них ОДИН. Ступени различаются
    # механиками, но воздух, сопротивление и двигатель обязаны совпадать
    # число в число. Пока таблицы атмосферы жили только в Тренажёре++, а
    # ранние ступени считали воздух экспонентой, сила сопротивления на них
    # была в 2-7 раз больше: сеть училась вести одну ракету, а на третьей
    # ступени получала другую — и разваливалась четыре прогона подряд.
    worlds = [AscentSimulator(seed=1, exploring_starts=0.0),
              HardcoreAscentSimulator(seed=1, exploring_starts=0.0),
              PrecisionAscentSimulator(seed=1, exploring_starts=0.0)]
    probe = SimStage(dry_mass=2000, fuel_mass=6000, thrust_vac=200_000,
                     thrust_asl=172_000, isp_vac=310, isp_asl=270,
                     capacity=6000)
    air_gap = drag_gap = thrust_gap = 0.0
    for altitude, speed in ((0, 100), (5000, 300), (12000, 700),
                            (20000, 1200), (35000, 2000)):
        densities, drags, thrusts = [], [], []
        for world in worlds:
            rho = world.density(altitude)
            mach = speed / world.speed_of_sound(altitude)
            densities.append(rho)
            drags.append(0.5 * rho * speed * speed
                         * world.drag_coefficient(mach) * world.body_area)
            thrusts.append(world.thrust_and_isp(
                probe, world.pressure_atm(altitude))[0])
        air_gap = max(air_gap, max(densities) - min(densities))
        drag_gap = max(drag_gap, max(drags) - min(drags))
        thrust_gap = max(thrust_gap, max(thrusts) - min(thrusts))
    check("воздух одинаков на всех ступенях", air_gap < 1e-9,
          f"расхождение плотности {air_gap:.2e} кг/м³")
    check("сопротивление одинаково на всех ступенях", drag_gap < 1e-6,
          f"расхождение силы {drag_gap:.2e} Н")
    check("двигатель одинаков на всех ступенях", thrust_gap < 1e-6,
          f"расхождение тяги {thrust_gap:.2e} Н")
    check("мидель корпуса общий", len({w.body_area for w in worlds}) == 1,
          f"{worlds[0].body_area:.3f} м²")

    # НЕУДАЧНЫЙ СТАРТ ОБРЫВАЕТСЯ СРАЗУ. Пока он длился все 420 секунд,
    # каждая такая попытка сыпала в буфер 2100 шагов стояния на столе, и
    # политика уверенно училась не взлетать: замер дал 0 орбит из 20 со
    # стола при газе 0.27 против 15 из 20 в воздухе при газе 0.95.
    idle = worlds[0]
    idle.reset()
    idle.set_difficulty(1.0)
    steps, done, reason = 0, False, ""
    while not done and steps < 5000:
        _, _, done, info = idle.step(FlightAction(0.0, 0.0, 0.0, 0.0, False))
        steps += 1
        reason = info.get("reason", "")
    check("не оторвался — эпизод обрывается, а не тянется до лимита",
          done and reason == "не оторвался от стола"
          and idle.time <= idle.LIFTOFF_DEADLINE + idle.dt,
          f"оборвано на {idle.time:.0f} с ({steps} шагов) вместо "
          f"{idle.max_time:.0f} с")

    # Обучающая выборка обязана сойтись к тому, что спрашивают на
    # экзамене: старт со стола. Иначе сеть учится доводить уже летящий
    # аппарат и не умеет стартовать.
    fading = AscentSimulator(seed=2, exploring_starts=0.5)
    fading.set_difficulty(0.0)
    start = fading.airborne_start_chance()
    fading.set_difficulty(1.0)
    finish = fading.airborne_start_chance()
    check("старты в воздухе тают вместе с разогревом",
          start > 0.4 and finish == 0.0,
          f"{start:.0%} в начале -> {finish:.0%} к полной жёсткости")

    # НИ ОДНА ОСЬ НЕ ДОЛЖНА БЫТЬ БЕСПЛАТНОЙ. Замер, ради которого это
    # написано: детерминированная политика держала рыскание ровно −1.00 и
    # крен −0.80 на плоских ступенях — там они ни на что не влияют. На
    # Тренажёре++ та же команда при вертикальном аппарате разворачивает
    # курс на 100 град/с (усиление 1/cos тангажа), и ступень умирала
    # четыре прогона подряд именно от этого.
    straight = FlightAction(1.0, 0.2, 0.0, 0.0, False)
    pinned = FlightAction(1.0, 0.2, 1.0, 1.0, False)
    for world, title in ((worlds[0], "Базовый"), (worlds[1], "Тренажёр+")):
        world.reset()
        world.set_difficulty(1.0)
        cost = (world.shaping_reward(straight)
                - world.shaping_reward(pinned))
        check(f"{title}: руль в пустоту стоит награды", cost > 1e-6,
              f"{cost:.4f} за шаг, около "
              f"{cost * world.max_time / world.dt:.1f} за полёт")
    pro_world = worlds[2]
    pro_world.reset()
    pro_world.set_difficulty(1.0)
    check("на Тренажёре++ платы за руль в пустоту нет — оси там живые",
          not pro_world.INERT_AXES,
          "рыскание наказывается скольжением, крен — раскруткой")

    # Кинематика рыскания: вращение вокруг собственной оси раскладывается
    # по тангажу. У вертикальной ракеты оно уходит в КРЕН (курс у зенита
    # вырожден), у летящей полого — в КУРС. Раньше здесь стояло усиление
    # 1/cos(тангаж), и у зенита команда крутила азимут на 100 град/с при
    # почти неподвижном носе — мина, срабатывавшая на развороте.
    from kia_core.neural.spaces import FlightAction as _FA
    hard_yaw = _FA(1.0, 0.0, 1.0, 0.0, False)
    for index in (0, 2):
        env = STAGES[index].make_env(seed=5, exploring_starts=0.0)
        env.set_difficulty(1.0)
        env.reset()
        for _ in range(120):
            env.step(hard_yaw)
        vertical_roll = abs(env.roll_rate)
        check(f"ступень {index + 1}: у зенита рыскание уходит в крен",
              vertical_roll > 5.0 and abs(env.heading - 90.0) < 45.0,
              f"крен {vertical_roll:.0f} град/с, курс сместился на "
              f"{abs(env.heading - 90.0):.0f}°")

        env = STAGES[index].make_env(seed=5, exploring_starts=0.0)
        env.set_difficulty(1.0)
        env.reset()
        env.pitch = 10.0                     # почти горизонтальный полёт
        env.airborne = True
        for _ in range(120):
            env.step(hard_yaw)
        check(f"ступень {index + 1}: у горизонта рыскание меняет курс",
              abs(env.heading - 90.0) > 20.0,
              f"курс ушёл на {abs(env.heading - 90.0):.0f}°")

    # ГЛАВНОЕ ПРО МЁРТВЫЕ ОСИ: за что не спрашивают, то уползает на упор.
    # Обе оси должны СТОИТЬ награды, иначе их значение — произвольный шум,
    # который выстрелит там, где ось работает (5.3i, 5.3j).
    def fly_pinned(yaw: float, roll: float, seed: int = 4) -> float:
        env = STAGES[2].make_env(seed=seed, exploring_starts=0.0)
        env.set_difficulty(1.0)
        env.reset()
        done, total = False, 0.0
        while not done and env.time < env.max_time:
            action = _FA(1.0, 1.0, yaw, roll, env.action_mask()[1])
            for repeat in range(STAGES[2].action_repeat):
                step = action if repeat == 0 else _FA(
                    action.throttle, action.pitch, action.yaw, action.roll,
                    False)
                _, reward, done, _ = env.step(step)
                total += reward
                if done:
                    break
        return total

    clean = fly_pinned(0.0, 0.0)
    pinned_yaw = fly_pinned(1.0, 0.0)
    pinned_roll = fly_pinned(0.0, 1.0)
    check("рыскание на упоре стоит награды даже у вертикальной ракеты",
          clean - pinned_yaw > 5.0,
          f"чисто {clean:+.1f} против {pinned_yaw:+.1f} при упоре")
    check("крен на упоре стоит награды", clean - pinned_roll > 3.0,
          f"чисто {clean:+.1f} против {pinned_roll:+.1f} при упоре")
    check("цена крена не перевешивает саму задачу",
          clean - pinned_roll < 25.0,
          f"раскрутка обходится в {clean - pinned_roll:.1f} очка "
          f"при +50 за орбиту")

    spun = STAGES[2].make_env(seed=4, exploring_starts=0.0)
    spun.set_difficulty(1.0)
    spun.reset()
    for _ in range(200):
        spun.step(_FA(1.0, 0.0, 0.0, 1.0, False))
    check("крен теперь физика, а не пустая ось",
          abs(spun.roll_rate) > 5.0 and spun.roll != 0.0,
          f"скорость крена {spun.roll_rate:.1f} град/с, угол "
          f"{spun.roll:.0f}°")
    # За одну ступень должно меняться ОДНО. Ритм решений — отдельная
    # ступень: при переходе на 0.3 с та же команда поворачивает аппарат
    # втрое слабее, и профиль выведения приходится перекладывать целиком.
    check("ритм решений меняется ровно один раз, на третьей ступени",
          STAGES[0].decision_seconds == STAGES[1].decision_seconds
          and STAGES[2].decision_seconds < STAGES[1].decision_seconds
          and STAGES[3].decision_seconds == STAGES[2].decision_seconds,
          " -> ".join(f"{s.decision_seconds:.1f} с" for s in STAGES))
    check("закон управления меняется ровно один раз, на четвёртой",
          all(dict(s.env_options).get("assist_floor") == 1.0
              for s in STAGES[:3])
          and dict(STAGES[3].env_options).get("assist_floor") == 0.0,
          "до неё команда задаёт скорость поворота, на ней — момент")

    # --- наблюдение одинаковой длины на всех ступенях -------------------
    # если оно разъедется, веса прошлой ступени просто не загрузятся
    for stage in STAGES:
        env = stage.make_env(seed=1, exploring_starts=0.0)
        obs = env.reset()
        check(f"ступень {stage.number}: наблюдение {FLIGHT_OBS_DIM} чисел",
              len(obs) == FLIGHT_OBS_DIM, f"{len(obs)}")
        check(f"ступень {stage.number}: значения конечны и нормированы",
              all(abs(v) < 50 for v in obs),
              f"max={max(abs(v) for v in obs):.1f}")

    # --- решаемость каждой ступени эталонным профилем -------------------
    # На ступенях 1-3 команда задаёт скорость поворота, на четвёртой к
    # концу разогрева — момент, поэтому там нужен каскадный регулятор.
    for stage, cascade in ((STAGES[0], False), (STAGES[1], False),
                           (STAGES[2], False), (STAGES[3], True)):
        orbits = 0
        for seed in range(6):
            env, _ = fly_reference(stage, seed, cascade=cascade)
            orbits += 1 if env.reached_orbit else 0
        check(f"ступень {stage.number} решаема: эталон выводит на орбиту",
              orbits >= 4, f"орбит {orbits} из 6 аппаратов ({stage.title})")

    # Начало последней ступени обязано быть проходимо ПРЕЖНИМИ руками:
    # именно ради этого помощь по управлению стартует с единицы
    orbits = 0
    for seed in range(4):
        env, _ = fly_reference(STAGES[3], seed, cascade=False, difficulty=0.0)
        orbits += 1 if env.reached_orbit else 0
    check("Тренажёр+++ на старте проходится профилем прошлой ступени",
          orbits >= 3, f"орбит {orbits} из 4 при жёсткости 0 %")

    # --- маска действий: отделение полной ступени запрещено -------------
    env = AscentSimulator(seed=2, exploring_starts=0.0)
    env.reset()
    check("полную ступень сбросить нельзя", env.action_mask() == [True, False])
    env.step(FlightAction(1.0, 0.0, 0.0, 0.0, True))
    check("команда отделения на старте проигнорирована", env.stage_index == 0)
    env.stages[0].fuel_mass = 0.0
    check("выработанную ступень сбросить можно", env.action_mask()[1])

    # --- Тренажёр+: задержка связи --------------------------------------
    plus = HardcoreAscentSimulator(seed=3, exploring_starts=0.0)
    plus.reset()
    first = plus.filter_command(FlightAction(1.0, 0.0, 0.0, 0.0, False))
    second = plus.filter_command(FlightAction(1.0, 0.0, 0.0, 0.0, False))
    check("задержка kRPC: команда доходит не сразу", first.throttle < 0.95,
          f"дошло {first.throttle:.2f} из 1.00")
    check("устоявшаяся команда доходит целиком", second.throttle > 0.99)

    # --- Тренажёр+: таймаут ступеней -------------------------------------
    plus = HardcoreAscentSimulator(seed=4, exploring_starts=0.0)
    plus.reset()
    plus.stages[0].fuel_mass = 0.0
    plus.step(FlightAction(1.0, 0.0, 0.0, 0.0, True))
    check("после отделения стейджинг заблокирован",
          not plus.action_mask()[1] and plus.stage_ready_at >= plus.time,
          f"блокировка до {plus.stage_ready_at:.1f} с")

    # --- Тренажёр+: инерция тяги -----------------------------------------
    plus = HardcoreAscentSimulator(seed=5, exploring_starts=0.0)
    plus.reset()
    for _ in range(2):
        plus.step(FlightAction(1.0, 0.0, 0.0, 0.0, False))
    check("тяга отстаёт от дросселя (spool-up)",
          0.05 < plus.actual_throttle < 0.9,
          f"команда 1.00, фактическая {plus.actual_throttle:.2f}")

    # --- Тренажёр+: ветер -------------------------------------------------
    peaks = []
    for seed in range(6):
        plus = HardcoreAscentSimulator(seed=seed, exploring_starts=0.0)
        plus.reset()
        plus.airborne = True
        peak = 0.0
        for _ in range(600):
            plus.blow_wind()
            peak = max(peak, abs(plus.wind_offset))
        peaks.append(peak)
    check("ветер сносит нос, но не уносит его совсем",
          all(0.3 < peak < 8.0 for peak in peaks),
          f"максимальный снос {min(peaks):.1f}-{max(peaks):.1f}°")

    # --- Тренажёр+: угол атаки -------------------------------------------
    plus = HardcoreAscentSimulator(seed=7, exploring_starts=0.0)
    plus.reset()
    plus.airborne = True
    plus.x, plus.y = 0.0, 604_000.0            # 4 км над поверхностью
    plus.vx = -174.5                           # восток: (-up_y, up_x)
    plus.vy = 350.0                            # вертикальный набор
    plus.pitch = plus.pitch - plus.angle_of_attack()   # нос точно по потоку
    straight = plus.shaping_reward(FlightAction(1.0, 0, 0, 0, False))
    plus.pitch -= 25.0
    skidding = plus.shaping_reward(FlightAction(1.0, 0, 0, 0, False))
    check("скольжение в плотных слоях штрафуется", skidding < straight - 0.1,
          f"прямо {straight:+.3f}, боком {skidding:+.3f}")

    # --- Тренажёр+: электричество ----------------------------------------
    def flail(amplitude: float, hold: int = 5, seconds: float = 90.0):
        """Гоняет рулями и возвращает (остаток заряда, секунда обесточивания)."""
        env = HardcoreAscentSimulator(seed=8, exploring_starts=0.0)
        env.reset()
        dead = None
        for i in range(int(seconds / env.dt)):
            pitch = amplitude * (1 if (i // hold) % 2 else -1)
            env.step(FlightAction(1.0, pitch, 0.0, 0.0, False))
            if dead is None and env.control_lost:
                dead = env.time
        return env.min_electric, dead

    smooth, _ = flail(0.05)
    check("плавное управление не сажает батарею",
          smooth > HardcoreAscentSimulator.EC_CAPACITY * 0.5,
          f"осталось {smooth:.0f} из {HardcoreAscentSimulator.EC_CAPACITY:.0f}")
    left, dead_at = flail(1.0)
    check("хаотичные дёрганья обнуляют заряд", left <= 0.01, f"осталось {left:.1f}")
    # После обвала политики на 112-м обновлении ёмкость подняли вдвое:
    # четырёх секунд на исправление ошибки пилоту не хватало
    check("на панику у пилота есть время выровняться, а не 4 секунды",
          dead_at is not None and dead_at >= 8.0,
          f"обесточивание на {dead_at:.1f} с непрерывной паники"
          if dead_at else "не обесточился вовсе")

    # --- Тренажёр+: щадящий режим в начале обучения ----------------------
    def skid_penalty(difficulty: float, aoa: float):
        env = HardcoreAscentSimulator(seed=7, exploring_starts=0.0)
        env.reset()
        env.airborne = True
        env.set_difficulty(difficulty)
        env.x, env.y = 0.0, 604_000.0
        env.vx, env.vy = -174.5, 350.0
        env.pitch = env.pitch - env.angle_of_attack() + aoa
        return (env.shaping_reward(FlightAction(1.0, 0, 0, 0, False)),
                env.check_failure())

    easy_penalty, easy_fail = skid_penalty(0.0, 30.0)
    hard_penalty, _ = skid_penalty(1.0, 30.0)
    check("в начале обучения скольжение — предупреждение, а не приговор",
          easy_penalty > hard_penalty * 0.5,
          f"щадяще {easy_penalty:+.3f}, строго {hard_penalty:+.3f}")
    _, hard_fail = skid_penalty(1.0, 60.0)
    _, easy_fail_60 = skid_penalty(0.0, 60.0)
    check("щадящий режим не опрокидывает аппарат на 60°",
          easy_fail_60 is None and hard_fail is not None,
          "строгий режим на 60° опрокидывает, щадящий — нет")
    check("полная жёсткость возвращает исходные значения",
          abs(hard_penalty - (-0.625)) < 0.01, f"{hard_penalty:+.3f}")

    # решаемость обязана сохраняться в обоих режимах, иначе щадящий режим
    # учит одному миру, а строгий — другому
    for difficulty in (0.0, 1.0):
        orbits = 0
        for seed in range(4):
            env = STAGES[1].make_env(seed=seed, exploring_starts=0.0)
            env.set_difficulty(difficulty)
            env.reset()
            done = False
            while not done and env.time < env.max_time:
                action = reference_pilot(env)
                for repeat in range(STAGES[1].action_repeat):
                    step = action if repeat == 0 else FlightAction(
                        action.throttle, action.pitch, action.yaw,
                        action.roll, False)
                    _, _, done, _ = env.step(step)
                    if done:
                        break
            orbits += 1 if env.reached_orbit else 0
        check(f"Тренажёр+ решаем при жёсткости {difficulty * 100:.0f} %",
              orbits >= 3, f"орбит {orbits} из 4")

    # --- Тренажёр++: атмосфера по таблицам --------------------------------
    pro = PrecisionAscentSimulator(seed=9, exploring_starts=0.0)
    pro.reset()
    check("плотность на уровне моря как в игре",
          abs(pro.density(0.0) - 1.225) < 0.005, f"{pro.density(0.0):.4f} кг/м³")
    check("скорость звука на уровне моря 340 м/с",
          abs(pro.speed_of_sound(0.0) - 340.3) < 1.0,
          f"{pro.speed_of_sound(0.0):.1f} м/с")
    check("атмосфера кончается на 70 км", pro.density(70_000.0) == 0.0)
    check("плотность падает монотонно",
          all(pro.density(h) > pro.density(h + 1000.0)
              for h in range(0, 65_000, 5_000)))
    check("околозвуковой горб сопротивления на месте",
          pro.drag_coefficient(1.2) > pro.drag_coefficient(0.5) * 2
          and pro.drag_coefficient(1.2) > pro.drag_coefficient(4.0),
          f"Cd: 0.5M={pro.drag_coefficient(0.5):.2f}, "
          f"1.2M={pro.drag_coefficient(1.2):.2f}, "
          f"4M={pro.drag_coefficient(4.0):.2f}")

    # --- Тренажёр++: двигатели --------------------------------------------
    first_stage = pro.stages[0]
    sea, isp_sea = pro.thrust_and_isp(first_stage, 1.0)
    vacuum, isp_vac = pro.thrust_and_isp(first_stage, 0.0)
    flow_sea = sea / (isp_sea * 9.80665)
    flow_vac = vacuum / (isp_vac * 9.80665)
    check("в плотных слоях тяга и Isp падают",
          sea < vacuum and isp_sea < isp_vac,
          f"{sea/1000:.0f} кН/{isp_sea:.0f} с против "
          f"{vacuum/1000:.0f} кН/{isp_vac:.0f} с")
    check("массовый расход постоянный (модель KSP)",
          abs(flow_sea - flow_vac) < 1e-6, f"{flow_sea:.2f} кг/с")

    # --- Тренажёр++: центр масс -------------------------------------------
    pro = PrecisionAscentSimulator(seed=10, exploring_starts=0.0)
    pro.reset()
    capacity = pro.stages[0].fuel_mass
    _, com_full, inertia_full = pro.mass_properties()
    cop_full, _ = pro.pressure_centre()
    pro.stages[0].fuel_mass = 0.0
    _, com_empty, inertia_empty = pro.mass_properties()
    check("центр масс смещается по мере выработки топлива",
          abs(com_full - com_empty) > 0.3,
          f"{com_full:.2f} м -> {com_empty:.2f} м при {capacity:.0f} кг топлива")
    check("момент инерции падает вместе с массой",
          inertia_empty < inertia_full,
          f"{inertia_full:.0f} -> {inertia_empty:.0f} кг·м²")
    check("на старте аппарат аэродинамически устойчив",
          cop_full > com_full,
          f"запас устойчивости {cop_full - com_full:+.2f} м")

    # --- Тренажёр++: 3D ---------------------------------------------------
    pro = PrecisionAscentSimulator(seed=11, exploring_starts=0.0)
    pro.reset()
    check("состояние трёхмерное", len(pro.pos) == 3 and len(pro.vel) == 3)
    # на плоских ступенях восток должен смотреть туда же, куда в 3D:
    # при рассогласовании аппарат на столе «летел» против вращения
    flat = AscentSimulator(seed=11, exploring_starts=0.0)
    flat.reset()
    check("на плоских ступенях на столе тоже штиль", flat.air_speed() < 1.0,
          f"воздушная скорость {flat.air_speed():.2f} м/с")
    check("атмосфера вращается вместе с планетой: на столе штиль",
          pro.air_speed() < 1.0 and pro.dynamic_pressure() < 1.0,
          f"воздушная скорость {pro.air_speed():.2f} м/с")
    check("орбитальная скорость на столе — это вращение Кербина",
          170.0 < pro._speed() < 180.0, f"{pro._speed():.1f} м/с")
    env, _ = fly_reference(STAGES[2], seed=0, cascade=True)
    check("наклонение орбиты считается", env._inclination() < 15.0,
          f"{env._inclination():.2f}° при пуске на восток")

    # --- переход между ступенями ------------------------------------------
    check("снимок ступени 1 описывается без загрузки весов",
          checkpoint_info(STAGES[1].backup).path.name
          == "kia_model_stage1.pth")
    # Проверяем ЛОГИКУ лестницы, а не то, какие снимки лежат на диске
    # сегодня: раньше тест требовал «ступень 3 не пройдена» и покраснел в
    # тот день, когда программа впервые была пройдена целиком.
    ladder = progress(current_stage=2)
    check("лестница обучения отмечает пройденное",
          ladder[0].passed and ladder[1].current,
          " | ".join(f"{s.stage.number}:{s.mark}" for s in ladder))
    check("выше текущей ступень считается пройденной только по снимку",
          all(status.passed == bool(STAGES[index + 1].backup
                                    and STAGES[index + 1].backup.exists())
              for index, status in enumerate(ladder[2:], start=2)
              if index + 1 < len(STAGES)),
          "признак — существование снимка весов, а не номер ступени")
    before = STAGES[1].backup.exists()
    dry_run = prepare(STAGES[1], brain=None, dry_run=True)
    check("переход на ступень 2 проверяется, ничего не трогая",
          STAGES[1].backup.exists() == before,
          "; ".join(dry_run.messages)[:70])

    # --- живая KSP платит за то же, что тренажёр --------------------------
    # Если награда в игре считается иначе, чем в тренажёре, вся программа
    # обучения теряет смысл: сеть придёт с политикой под другую задачу.
    import time as _time
    from kia_core.neural.flight_env import KSPFlightEnv
    from kia_core.neural.sim_env import SimSnapshot

    live = KSPFlightEnv(None, goal_altitude=80_000.0)
    live.started_wall = _time.time()
    live.done = False

    # РИТМ РЕШЕНИЙ В ИГРЕ ОБЯЗАН СОВПАДАТЬ С ТЕМ, НА ЧЁМ УЧИЛИСЬ.
    # Сеть, решающая раз в 0.3 с, в мире с ответом раз в секунду ведёт
    # аппарат втрое грубее. Ровно этот разрыв (только между тренажёрами)
    # весь день ломал третью ступень; в живой игре он был заложен по
    # умолчанию: control_hz = 1.0 против 0.3 с у финальной модели.
    trainer_source = (Path(__file__).resolve().parent.parent / "kia_core"
                      / "neural" / "trainer.py").read_text(encoding="utf-8")
    ksp_body = trainer_source.split("def train_in_ksp", 1)[-1]
    check("ритм решений в игре берётся из ступени программы",
          "control_seconds=stage.decision_seconds" in ksp_body,
          "единый источник правды — Stage.decision_seconds")
    for stage in STAGES:
        env = KSPFlightEnv(None, control_seconds=stage.decision_seconds)
        check(f"ступень {stage.number}: игра спрашивает раз в "
              f"{stage.decision_seconds:.1f} с",
              abs(env.control_dt - stage.decision_seconds) < 1e-9)

    live_source = (Path(__file__).resolve().parent.parent / "kia_core"
                   / "neural" / "flight_env.py").read_text(encoding="utf-8")
    check("период выдерживается с учётом времени на kRPC",
          "self.control_dt - spent" in live_source
          and "time.sleep(self.control_dt)" not in live_source,
          "ждём остаток периода, а не период поверх работы")
    check("фактический ритм попадает в отчёт о полёте",
          "actual_seconds" in live_source and "не поспевает" in live_source,
          "отставание видно, а не угадывается")

    # СИСТЕМЫ ОТСЧЁТА. Самый дорогой сорт расхождения: одно и то же слово
    # означает в игре и в тренажёре разные вещи, и заметить это можно
    # только замером. Первый живой полёт нейросети погиб именно на этом —
    # телеметрия отдавала тангаж 15° у ракеты, стоящей строго вертикально.
    telemetry_source = (Path(__file__).resolve().parent.parent / "kia_core"
                        / "environment" / "telemetry.py").read_text(
                            encoding="utf-8")
    streams = telemetry_source.split("def _open_streams", 1)[-1].split(
        "def get(", 1)[0]
    check("углы читаются из ПОВЕРХНОСТНОЙ системы отсчёта",
          all(f'self._add("{name}", self.srf_flight' in streams
              for name in ("pitch", "heading", "roll")),
          "иначе «тангаж 90» в тренажёре и «15» в игре — про одну ракету")
    check("скорости читаются из НЕВРАЩАЮЩЕЙСЯ системы",
          all(f'self._add("{name}", self.orbital_flight' in streams
              for name in ("speed", "vertical_speed", "horizontal_speed")),
          "тренажёр считает орбитальную: на столе 175 м/с, а не ноль")
    check("перепривязка обновляет обе системы",
          telemetry_source.count("self.srf_flight = ") >= 2
          and telemetry_source.count("self.orbital_flight = ") >= 2,
          "после смены судна углы и скорости не должны читаться со старого")

    def snapshot(**fields):
        snap = SimSnapshot(altitude=5_000.0, apoapsis=6_000.0, periapsis=-600_000.0,
                           speed=400.0, vertical_speed=380.0, pitch=72.0,
                           dynamic_pressure=15_000.0, part_count=10,
                           mission_time=40.0, situation="flying",
                           resources={"LiquidFuel": {"amount": 500.0, "max": 1000.0}})
        for name, value in fields.items():
            setattr(snap, name, value)
        return snap

    def live_reward(**fields) -> float:
        """Награда за шаг из базового состояния в изменённое."""
        base = snapshot()
        live.previous_apoapsis = base.apoapsis
        live.previous_altitude = base.altitude
        live.previous_energy = live.specific_energy(base)
        live.reached_orbit = False
        reward, _, _ = live.reward(snapshot(**fields))
        return reward

    aligned = live_reward()                    # нос по вектору движения
    skidding = live_reward(pitch=40.0)         # скольжение 32°
    nose_down = live_reward(pitch=-40.0, vertical_speed=-100.0)
    check("в живой KSP скольжение штрафуется так же, как в Тренажёре+",
          skidding < aligned - 0.1, f"прямо {aligned:+.3f}, боком {skidding:+.3f}")
    check("в живой KSP завал носа штрафуется", nose_down < aligned,
          f"{nose_down:+.3f} против {aligned:+.3f}")
    # разгон без изменения угла атаки: и скорость, и вертикальная растут
    faster = live_reward(speed=600.0, vertical_speed=570.0)
    # +200 м/с на высоте 5 км — это 0.36 награды и в игре, и в тренажёре
    check("в живой KSP прирост орбитальной энергии вознаграждается",
          0.3 < faster - aligned < 0.45, f"{faster:+.3f} против {aligned:+.3f}")
    check("маска стейджинга работает и в игре без телеметрии",
          live.action_mask() == [True, False])

    # --- скорость обучения и разведка по ступеням -------------------------
    from kia_core.neural.ppo import (PPOAgent, PPOConfig, RolloutBuffer,
                                     entropy_coef_for_stage,
                                     learning_rate_for_stage)
    # Проверяем ЗАМЫСЕЛ — таблицу в модуле. Действующие значения проверять
    # нельзя: конвейер законно режет скорость провалившейся ступени вдвое
    # за откат и кладёт поправку в data/stage_learning_rate.json. После
    # трёх откатов на третьей ступени она честно оказывается ниже
    # четвёртой, и самопроверка краснела бы из-за того, что сторож сработал.
    from kia_core.neural.ppo import STAGE_LEARNING_RATE
    rates = [STAGE_LEARNING_RATE[n] for n in (1, 2, 3, 4)]
    check("на доучивании шаг градиента мельче, чем на обучении с нуля",
          all(a >= b for a, b in zip(rates, rates[1:])) and rates[0] > rates[-1],
          " -> ".join(f"{value:.1e}" for value in rates))
    active = [learning_rate_for_stage(n) for n in (1, 2, 3, 4)]
    lowered = [f"ступень {n}: {STAGE_LEARNING_RATE[n]:.1e} -> {value:.1e}"
               for n, value in zip((1, 2, 3, 4), active)
               if abs(value - STAGE_LEARNING_RATE[n]) > 1e-12]
    check("действующие скорости видны оператору", True,
          "; ".join(lowered) if lowered
          else "поправок нет, работают исходные значения")
    bonuses = [entropy_coef_for_stage(n) for n in (1, 2, 3, 4)]
    check("бонус за разведку падает к точным ступеням",
          all(a >= b for a, b in zip(bonuses, bonuses[1:]))
          and bonuses[0] > bonuses[-1],
          " -> ".join(f"{value:.4f}" for value in bonuses))

    # Потолок разброса действий обязан сужаться: пока σ велика, обучается
    # распределение вокруг политики, а не сама политика. Замер на боевых
    # весах: при σ 0.50 детерминированный прогон давал 0 орбит из 12 при
    # 654 «сданных» орбитах в обучении.
    from kia_core.neural.ppo import log_std_max_for_stage
    ceilings = [log_std_max_for_stage(n) for n in (1, 2, 3, 4)]
    check("потолок разброса действий сужается к точным ступеням",
          all(a > b for a, b in zip(ceilings, ceilings[1:])),
          " -> ".join(f"σ {math.exp(v):.3f}" for v in ceilings))
    check("на последней ступени выбранное действие близко к среднему",
          math.exp(ceilings[-1]) < 0.2, f"σ {math.exp(ceilings[-1]):.3f}")
    # Разогрев есть у всех ступеней: даже на первой мир настоящий, и
    # штраф за скольжение должен сначала предупреждать, а наказывать —
    # когда сеть уже нащупала, чем его гасить.
    check("каждая ступень начинается с разогрева",
          all(0.0 < s.warmup < 1.0 for s in STAGES),
          "разогрев ступеней: " + ", ".join(
              f"{s.number}:{s.warmup:.0%}" for s in STAGES))
    check("разогрев длиннее там, где меняется закон управления",
          STAGES[3].warmup > STAGES[0].warmup,
          f"{STAGES[3].warmup:.0%} против {STAGES[0].warmup:.0%}")

    # Отжиг LR обязан отсчитываться от начала прогона. Иначе продолженное
    # обучение стартует сразу с конца расписания: 97 прошлых обновлений
    # против 10 запланированных — и скорость падает в десять раз.
    agent = PPOAgent(6, 2, (2,), PPOConfig(total_updates_planned=10),
                     hidden=16, name="отжиг")
    agent.stats.updates = 97
    agent.config.anneal_baseline = 97
    buffer = RolloutBuffer(6, 2, 1, 32, agent.device, mask_dim=2)
    for _ in range(32):
        observation, continuous, discrete, logp, value = agent.act([0.1] * 6)
        buffer.add(observation, continuous, discrete, logp, value, 0.5, False)
    first = agent.update(buffer)["learning_rate"]
    check("отжиг LR считается от начала прогона, а не от рождения мозга",
          abs(first - agent.config.learning_rate) < 1e-9,
          f"{first:.2e} при плане 10 обновлений и 97 прошлых")


# ==========================================================================
def test_pipeline() -> None:
    """[23] Автономный конвейер: экзамены, тупик, откат."""
    print("\n[23] Автономный конвейер эволюции")
    from kia_core.neural import is_available
    if not is_available():
        check("PyTorch установлен", False)
        return
    import tempfile
    from kia_core.neural import ppo
    from kia_core.neural.curriculum import STAGES
    from kia_core.neural.pipeline import (ALARM_BANNER, EvolutionPipeline,
                                          FINAL_MODEL_FILE, MANIFEST_FILE,
                                          PipelineConfig, StageExam,
                                          complete_file)

    config = PipelineConfig()
    check("правила экзамена как заданы",
          config.window == 50 and config.promote_reward == 20.0
          and config.promote_orbits == 100,
          f"окно {config.window}, награда {config.promote_reward:+.1f}, "
          f"орбит {config.promote_orbits}")
    check("правила тупика как заданы",
          config.collapse_reward == -40.0 and config.collapse_patience == 300
          and config.rollback_steps == 200_000 and config.lr_penalty == 0.5,
          f"{config.collapse_reward:+.1f} / {config.collapse_patience} эпизодов "
          f"/ {config.rollback_steps} шагов / LR ×{config.lr_penalty}")
    check("имена архивных чекпоинтов",
          complete_file(2).name == "kia_model_stage2_complete.pth"
          and FINAL_MODEL_FILE.name == "kia_model_final.pth",
          f"{complete_file(1).name}, {FINAL_MODEL_FILE.name}")
    check("манифест эволюции в корне проекта",
          MANIFEST_FILE.name == "kia_evolution_manifest.txt")
    check("текст тревоги дословный",
          ALARM_BANNER == "=== ДЕГРАДАЦИЯ НАВЫКОВ. СИСТЕМА ОПРЕДЕЛИЛА ТУПИК. "
                          "ДЕЛАЮ CD-ROLLBACK ===")

    # --- пороги перевода --------------------------------------------------
    exam = StageExam(config)
    for _ in range(config.promote_orbits - 1):
        exam.observe(25.0, {"orbit": True}, None)
    check("99 орбит при хорошей награде — ещё не экзамен", not exam.verdict,
          f"орбит {exam.orbits}, вердикт «{exam.verdict or '—'}»")
    exam.observe(25.0, {"orbit": True}, None)
    check("сотая орбита закрывает экзамен", exam.verdict == "promote",
          f"орбит {exam.orbits}, средняя {exam.mean_return:+.1f}")

    # Второй путь: доля орбит в окне. Средняя награда несравнима между
    # ступенями — на Тренажёре+ живой прогон дал 36 % орбит при средней,
    # болтающейся около нуля, и порог +20 был недостижим в принципе.
    exam = StageExam(config)
    for index in range(400):
        exam.observe(-5.0, {"orbit": index % 5 == 0}, None)   # 20 % орбит
    check("низкая доля орбит не переводит, даже когда их много всего",
          not exam.verdict,
          f"орбит {exam.orbits}, доля {exam.orbit_rate:.0%} при пороге "
          f"{config.promote_rate:.0%}")
    exam = StageExam(config)
    for index in range(400):
        exam.observe(-5.0, {"orbit": index % 2 == 0}, None)   # 50 % орбит
    check("устойчивая доля орбит переводит и при слабой средней",
          exam.verdict == "promote",
          f"доля {exam.orbit_rate:.0%}, средняя {exam.mean_return:+.1f}")

    # --- обнаружение тупика ------------------------------------------------
    exam = StageExam(config)
    fired = next((index + 1 for index in range(2000)
                  if exam.observe(-50.0, {}, None)), None)
    check("тупик объявляется после окна и терпения",
          exam.verdict == "collapse"
          and fired == config.window + config.collapse_patience,
          f"на {fired}-м эпизоде при окне {config.window} и терпении "
          f"{config.collapse_patience}")

    exam = StageExam(config)
    for index in range(600):
        exam.observe(-50.0 + index * 0.01, {}, None)   # плохо, но растёт
    check("растущая награда тупиком не считается", not exam.verdict,
          f"средняя {exam.mean_return:+.1f}, простой {exam.stalled}")

    # Дообучение при откате идёт с выключенным сторожем: там ровно
    # заданное число шагов, судить о ступени по ним нельзя
    exam = StageExam(config, watch_collapse=False)
    for _ in range(1000):
        exam.observe(-90.0, {}, None)
    check("с выключенным сторожем вердикт не выносится", not exam.verdict)

    # А вот на первой ступени застой теперь ловится — не ради отката
    # (откатываться некуда), а ради того, чтобы позвать человека. Без
    # этого конвейер честно ходил кругами всю ночь.
    pipeline_source = (Path(__file__).resolve().parent.parent / "kia_core"
                       / "neural" / "pipeline.py").read_text(encoding="utf-8")
    check("на первой ступени застой ловится и останавливает конвейер",
          "watch_collapse=True" in pipeline_source
          and "Базовая ступень не берётся" in pipeline_source,
          "вместо молчаливого круга — тревога и остановка")

    # --- снижение скорости обучения ---------------------------------------
    original = ppo.LR_OVERRIDE_FILE
    with tempfile.TemporaryDirectory() as folder:
        ppo.LR_OVERRIDE_FILE = Path(folder) / "lr.json"
        try:
            base = ppo.STAGE_LEARNING_RATE[2]
            first = ppo.scale_stage_learning_rate(2, config.lr_penalty)
            second = ppo.scale_stage_learning_rate(2, config.lr_penalty)
            check("каждый провал режет скорость обучения вдвое",
                  abs(first - base / 2) < 1e-12 and abs(second - base / 4) < 1e-12,
                  f"{base:.1e} -> {first:.1e} -> {second:.1e}")
            check("понижение переживает перезапуск (лежит на диске)",
                  ppo.LR_OVERRIDE_FILE.exists()
                  and abs(ppo.learning_rate_for_stage(2) - second) < 1e-12)
            ppo.reset_stage_learning_rate(2)
            check("сброс возвращает исходную скорость",
                  abs(ppo.learning_rate_for_stage(2) - base) < 1e-12,
                  f"{ppo.learning_rate_for_stage(2):.1e}")
        finally:
            ppo.LR_OVERRIDE_FILE = original

    # --- живая игра исключена ---------------------------------------------
    # Требование жёсткое: конвейер не имеет права трогать KSP. Проверяем
    # это по исходнику, а не по обещаниям в документации.
    source = (Path(__file__).resolve().parent.parent / "kia_core" / "neural"
              / "pipeline.py").read_text(encoding="utf-8")
    forbidden = [word for word in ("flight_env", "KSPFlightEnv", "train_in_ksp",
                                   "dispatcher", "krpc")
                 if word in source]
    check("конвейер не обращается к живой игре ни одной строкой",
          not forbidden, ", ".join(forbidden) if forbidden else
          "ни flight_env, ни dispatcher, ни krpc")

    stages_used = {stage.module for stage in STAGES}
    check("все ступени конвейера — симуляторы, живой игры среди них нет",
          stages_used == {"sim_env_pro"},
          ", ".join(sorted(stages_used)))
    check("финальная модель пишется после последней ступени",
          complete_file(len(STAGES)).name
          == f"kia_model_stage{len(STAGES)}_complete.pth")

    # --- сборка конвейера без запуска обучения ----------------------------
    check("конвейер собирается и отдаёт состояние",
          hasattr(EvolutionPipeline, "run")
          and hasattr(EvolutionPipeline, "rollback")
          and hasattr(EvolutionPipeline, "promote")
          and hasattr(EvolutionPipeline, "status")
          and hasattr(EvolutionPipeline, "confirm"))

    # --- застой ловится отдельно от обвала --------------------------------
    # Награда −15 не проваливается ниже порога обвала, но и никуда не идёт.
    # Раньше конвейер ходил по такому кругу часами (реальный случай:
    # 4106 эпизодов, 1 орбита, 7 млн шагов впустую).
    exam = StageExam(config)
    fired = next((index + 1 for index in range(4000)
                  if exam.observe(-15.0, {}, None)), None)
    check("застой без обвала тоже признаётся тупиком",
          exam.verdict == "collapse" and fired == config.stagnation_episodes,
          f"на {fired}-м эпизоде при пороге {config.stagnation_episodes} "
          f"и < {config.stagnation_orbits} орбит")
    exam = StageExam(config)
    for index in range(4000):
        exam.observe(-15.0, {"orbit": index % 30 == 0}, None)
    check("при регулярных орбитах застой не объявляется",
          not exam.verdict, f"орбит {exam.orbits}")

    # Главный случай, ради которого детектор переписан. Накопительный
    # счётчик орбит выключал проверку навсегда: набрав однажды пять орбит,
    # ступень уже не могла быть признана застойной, сколько бы десятков
    # тысяч пустых эпизодов ни прошло. Живой прогон простоял так 60
    # заходов и 178 млн шагов при «орбит 76/100 (0 % за окно)».
    exam = StageExam(config)
    for _ in range(80):
        exam.observe(25.0, {"orbit": True}, None)     # удачное начало
    fired = next((index + 1 for index in range(20_000)
                  if exam.observe(8.0, {"orbit": False}, None)), None)
    check("сухая полоса после удачного начала — тоже застой",
          exam.verdict == "collapse"
          and fired is not None and fired <= config.stagnation_episodes,
          f"накоплено орбит {exam.orbits}, за последние "
          f"{len(exam.recent_orbits)} — {sum(exam.recent_orbits)}, "
          f"тупик на {fired}-м пустом эпизоде")

    # Колебания награды не должны отменять этот вердикт: при нуле орбит
    # всплеск средней — шум, а не признак жизни. На живом прогоне средняя
    # гуляла от +3.6 до +12.8 и обнуляла терпение снова и снова.
    exam = StageExam(config)
    fired = next((index + 1 for index in range(20_000)
                  if exam.observe(4.0 + (index % 97) * 0.1,
                                  {"orbit": False}, None)), None)
    check("качели награды не спасают ступень без орбит",
          exam.verdict == "collapse" and fired == config.stagnation_episodes,
          f"тупик на {fired}-м эпизоде, средняя {exam.mean_return:+.1f}")

    check("предел откатов с одной ступени задан",
          config.max_rollbacks_per_stage >= 1,
          f"{config.max_rollbacks_per_stage} подряд, дальше зовём человека")

    # Проверка без шума должна отличать 15 % от 30 %, а не гадать. При
    # «3 из 10» слабая политика проходит примерно в одном случае из пяти —
    # и проходит, потому что экзамен пересдаётся много раз. Именно так
    # ступень 4 была объявлена сданной с 3/10 при истинных 5 % (1/20).
    from math import comb
    def pass_chance(rate, episodes, need):
        return sum(comb(episodes, k) * rate ** k * (1 - rate) ** (episodes - k)
                   for k in range(need, episodes + 1))

    weak = pass_chance(0.15, config.confirm_episodes, config.confirm_orbits)
    good = pass_chance(0.35, config.confirm_episodes, config.confirm_orbits)
    check("слабая политика почти не проходит проверку без шума",
          weak < 0.10, f"политика на 15 % проходит с вероятностью {weak:.0%}")
    check("хорошая политика проверку проходит",
          good > 0.60, f"политика на 35 % проходит с вероятностью {good:.0%}")
    check("порог проверки совпадает с порогом по доле орбит",
          abs(config.confirm_orbits / config.confirm_episodes
              - config.promote_rate) < 0.06,
          f"{config.confirm_orbits}/{config.confirm_episodes} против "
          f"{config.promote_rate:.0%} за окно")

    # Экзамен без шума обязан идти со стартового стола: половина обучающих
    # эпизодов начинается уже в воздухе (exploring starts), и с такими
    # подарками «3 орбиты из 10» ничего не доказывают — взлетать политика
    # при этом может не уметь вовсе.
    trainer_source = (Path(__file__).resolve().parent.parent / "kia_core"
                      / "neural" / "trainer.py").read_text(encoding="utf-8")
    evaluate_body = trainer_source.split("def evaluate_pilot", 1)[-1]
    check("проверка без шума стартует со стола и на полной жёсткости",
          "exploring_starts=0.0" in evaluate_body
          and "set_difficulty(1.0)" in evaluate_body
          and "update_normalizer=False" in evaluate_body,
          "старт с земли, difficulty 1.0, нормализатор не трогаем")

    # Жёсткость среды не должна падать обратно в ноль на повторном заходе:
    # разогрев вводит механики ступени один раз.
    check("повторный заход на ступень идёт на полной жёсткости",
          "difficulty_floor=1.0 if self.attempts else 0.0" in pipeline_source)

    # Флаг остановки снимает только тот, кто запускает работу. Пока это
    # делал сам цикл обучения, «стоп», нажатый между двумя заходами
    # конвейера, снимался следующим же заходом и пропадал.
    check("цикл обучения не снимает флаг остановки сам",
          "stop_flag.clear()" not in trainer_source.split(
              "def run_vector_training", 1)[-1],
          "это делает begin_session по команде оператора")
    check("тренер умеет начинать сессию явно",
          "def begin_session" in trainer_source)

    # Живая KSP обязана платить за то же, что тренажёр (раздел 5.4a):
    # потолок энергии, момент импульса, скольжение по обеим осям, крен.
    live_source = (Path(__file__).resolve().parent.parent / "kia_core"
                   / "neural" / "flight_env.py").read_text(encoding="utf-8")
    check("в живой KSP награда согласована с тренажёром",
          all(word in live_source for word in
              ("target_energy", "target_momentum", "_slip_angle",
               "ROLL_PENALTY")),
          "энергия с потолком, момент импульса, полное скольжение, крен")

    # --- экзамен подтверждается прогоном без шума -------------------------
    exam = StageExam(config)
    for _ in range(config.promote_orbits):
        exam.observe(25.0, {"orbit": True}, None)
    check("экзамен сдан по обучающим эпизодам", exam.verdict == "promote")
    exam.reject_promotion(config.confirm_cooldown)
    check("отказ по проверке без шума снимает вердикт и ставит паузу",
          not exam.verdict and exam.cooldown == config.confirm_cooldown
          and exam.rejections == 1)

    # Пауза растёт с каждым отказом: иначе при орбитах «от шума» конвейер
    # гоняет десятиэпизодную проверку каждую минуту и теряет на ней пятую
    # часть счёта. Живое наблюдение — шесть отказов подряд за шесть минут.
    stubborn = StageExam(config)
    for number in range(1, 5):
        stubborn.reject_promotion(config.confirm_cooldown)
        expected = config.confirm_cooldown * min(8, number)
        if stubborn.cooldown != expected:
            break
    check("каждый следующий отказ откладывает пересдачу дальше",
          stubborn.cooldown == config.confirm_cooldown * 4
          and stubborn.rejections == 4,
          f"после четырёх отказов пауза {stubborn.cooldown} эпизодов")
    for _ in range(config.confirm_cooldown - 1):
        exam.observe(25.0, {"orbit": True}, None)
    check("во время паузы экзамен не пересдаётся", not exam.verdict,
          f"осталось {exam.cooldown} эпизодов паузы")
    exam.observe(25.0, {"orbit": True}, None)
    exam.observe(25.0, {"orbit": True}, None)
    check("после паузы экзамен можно пересдать", exam.verdict == "promote")


# ==========================================================================
class _FakePart:
    """Двойник PartInfo для проверок бюро без каталога игры."""

    def __init__(self, name, dry_mass, height=1.0, diameter=1.25, fuel=0.0,
                 thrust=0.0, isp_vac=0.0, isp_asl=0.0, kind="tank",
                 category=""):
        self.name = self.title = name
        self.dry_mass = dry_mass
        self.height = height
        self.diameter = diameter
        self.fuel = fuel
        self.max_thrust = thrust
        self.isp_vac = isp_vac
        self.isp_asl = isp_asl
        self.kind = kind
        self.category = category

    is_engine = property(lambda self: self.kind == "engine")
    is_tank = property(lambda self: self.kind == "tank")
    is_command = property(lambda self: self.kind == "command")

    def wet_mass(self, densities=None):
        return self.dry_mass + self.fuel


def _fake_rocket(fuel=4.5, thrust=95.0, diameter=1.25, fins=0, nose=False):
    parts = []
    if nose:
        parts.append(_FakePart("noseCone", 0.03, 0.6, diameter, category="Aero"))
    parts.append(_FakePart("probeCore", 0.04, 0.3, diameter, kind="command"))
    parts.append(_FakePart("fuelTank", fuel * 0.111, 1.9, diameter, fuel=fuel))
    parts.append(_FakePart("engine", 1.25, 1.5, diameter, thrust=thrust,
                           isp_vac=310, isp_asl=270, kind="engine"))
    for index in range(fins):
        parts.append(_FakePart(f"fin{index}", 0.05, 0.8, 0.4, category="Aero"))
    return parts


def test_designer_pro() -> None:
    """[24] Проектное бюро: опыт полётов замыкается на чертёж."""
    print("\n[24] Проектное бюро (pro_modules/designer_pro)")
    import json
    import tempfile
    from pro_modules.designer_pro import (DesignerPro, ExperienceReader,
                                          MIN_STAGE, build_designer,
                                          build_layout, is_enabled)

    check("бюро работает только с Тренажёра+",
          not is_enabled(1) and is_enabled(2) and is_enabled(3)
          and build_designer(1) is None and build_designer(2) is not None,
          f"порог включения — ступень {MIN_STAGE}")

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)

        def reader(memory=None, log_text=""):
            (root / "memory.json").write_text(json.dumps(memory or {}),
                                              encoding="utf-8")
            (root / "diag.json").write_text("[]", encoding="utf-8")
            (root / "flight.md").write_text(log_text, encoding="utf-8")
            return ExperienceReader(memory_file=root / "memory.json",
                                    diagnoses_file=root / "diag.json",
                                    flight_log=root / "flight.md",
                                    runs_dir=root / "нет")

        # --- 1. без опыта бюро ничего не требует ------------------------
        blank = DesignerPro(reader=reader(), stage=2)
        check("без полётного опыта требований к форме нет",
              not blank.experience.has_experience
              and blank.shape_guidance().min_fins == 0)
        check("на базовой ступени награда не трогается",
              DesignerPro(reader=reader(), stage=1).reward(
                  5.0, blank.verify(_fake_rocket(), {})) == 5.0)

        # --- 2. опыт сопротивления -> мидель и обтекатель ---------------
        drag = DesignerPro(reader=reader(
            {"total_runs": 7,
             "failure_lessons": {"high_g": {"count": 3},
                                 "overbuilt": {"count": 2}}}), stage=2)
        guidance = drag.shape_guidance()
        check("потери на сопротивлении режут мидель и требуют обтекатель",
              guidance.max_diameter <= 1.25 and guidance.require_nose_cone,
              guidance.describe()[:80])
        wide = drag.verify(_fake_rocket(diameter=2.5, fins=4), {})
        slim = drag.verify(_fake_rocket(diameter=1.25, fins=4, nose=True), {})
        check("толстая ракета получает маркеры формы",
              "MIDSECTION_TOO_WIDE" in wide.markers
              and "NO_NOSE_CONE" in wide.markers, ", ".join(wide.markers))
        check("обтекаемая ракета их не получает и выгоднее по награде",
              not slim.markers and slim.penalty > wide.penalty,
              f"штраф {slim.penalty:+.2f} против {wide.penalty:+.2f}")

        # --- 3. опыт опрокидываний -> оперение --------------------------
        tumble = DesignerPro(reader=reader(
            {"total_runs": 5,
             "failure_lessons": {"aero_instability": {"count": 4}}}), stage=2)
        check("опрокидывания требуют стабилизаторов",
              tumble.shape_guidance().min_fins >= 4,
              tumble.shape_guidance().describe()[:80])
        bare = tumble.verify(_fake_rocket(fins=0), {})
        finned = tumble.verify(_fake_rocket(fins=4), {})
        check("без оперения — маркер, с оперением — чисто",
              "NO_FINS" in bare.markers and "NO_FINS" not in finned.markers)

        # --- 4. опыт читается прямо из журнала пилота -------------------
        from_log = DesignerPro(reader=reader(log_text=(
            "| 12:01 | ⚠ | максимальный напор превышен, аппарат тормозит |\n"
            "| 12:02 | ✖ | аппарат опрокинуло на сверхзвуке |\n")), stage=2)
        check("опыт вычитывается из журнала полёта",
              from_log.experience.drag_events > 0
              and from_log.experience.tumble_events > 0,
              from_log.experience.describe().splitlines()[0])

    # --- 5. чек-лист тяговооружённости ---------------------------------
    designer = DesignerPro(experience=None, stage=2)
    designer.experience = type(designer.experience)()      # чистый опыт
    low = designer.verify(_fake_rocket(thrust=60.0), {})
    fine = designer.verify(_fake_rocket(thrust=95.0), {})
    high = designer.verify(_fake_rocket(thrust=200.0), {})
    check("TWR ниже 1.1 -> CRITICAL_LOW_TWR",
          "CRITICAL_LOW_TWR" in low.markers, f"TWR {low.twr:.2f}")
    check("TWR выше 2.5 -> OVER_THRUST",
          "OVER_THRUST" in high.markers, f"TWR {high.twr:.2f}")
    check("нормальный TWR не порождает маркеров тяги",
          not {"CRITICAL_LOW_TWR", "OVER_THRUST"} & set(fine.markers),
          f"TWR {fine.twr:.2f}")
    check("при нехватке тяги предлагается шаблон радиальных ускорителей",
          any("ускорител" in item and "симметричн" in item
              for item in low.advice))

    # --- 6. устойчивость ------------------------------------------------
    bare = build_layout(_fake_rocket(fins=0), {})
    finned = build_layout(_fake_rocket(fins=4), {})
    check("оперение переносит центр давления назад и стабилизирует",
          not bare.stable and finned.stable,
          f"запас {bare.static_margin:+.2f} м -> {finned.static_margin:+.2f} м")
    check("неустойчивый аппарат помечается AERO_UNSTABLE",
          "AERO_UNSTABLE" in designer.verify(_fake_rocket(fins=0), {}).markers)

    # --- 7. разумная достаточность --------------------------------------
    lean = designer.verify(_fake_rocket(fuel=4.5, thrust=400.0), {})
    fat = designer.verify(_fake_rocket(fuel=40.0, thrust=400.0), {})
    check("запас ΔV в пределах нормы не штрафуется",
          "FUEL_OVERBUILT" not in lean.markers,
          f"запас {lean.overbuild:+.0%}")
    check("избыток ΔV помечается и стоит награды",
          "FUEL_OVERBUILT" in fat.markers
          and fat.penalties.get("избыток ΔV", 0.0) < -0.5,
          f"запас {fat.overbuild:+.0%}, штраф "
          f"{fat.penalties.get('избыток ΔV', 0.0):+.2f}")
    ramp = [designer.verify(_fake_rocket(fuel=fuel, thrust=400.0), {})
            .penalties.get("избыток ΔV", 0.0) for fuel in (9.0, 18.0, 40.0)]
    check("штраф за избыток растёт плавно, а не ступенькой",
          ramp[0] > ramp[1] > ramp[2],
          " -> ".join(f"{value:+.2f}" for value in ramp))

    # --- 8. иерархия Isp и Циолковский ----------------------------------
    core = _FakePart("probeCore", 0.04, 0.3, kind="command")
    terrier = _FakePart("terrier", 0.5, 0.8, thrust=60.0, isp_vac=345,
                        isp_asl=85, kind="engine")
    mainsail = _FakePart("mainsail", 6.0, 2.0, thrust=1500.0, isp_vac=310,
                         isp_asl=285, kind="engine")
    upper = _FakePart("tankUpper", 0.5, 1.2, fuel=4.5)
    lower = _FakePart("tankLower", 2.0, 3.0, fuel=18.0)
    right = designer.verify([core, terrier, upper, mainsail, lower], {})
    wrong = designer.verify([core, mainsail, upper, terrier, lower], {})
    check("вакуумный двигатель внизу — маркер иерархии Isp",
          "VACUUM_ENGINE_LOW" in wrong.markers
          and "VACUUM_ENGINE_LOW" not in right.markers,
          ", ".join(wrong.markers) or "—")
    heavy = [core, _FakePart("tank", 8.0, 1.9, fuel=4.0),
             _FakePart("engine", 1.25, 1.5, thrust=400.0, isp_vac=310,
                       isp_asl=270, kind="engine")]
    verdict = designer.verify(heavy, {})
    check("сухая масса, съедающая импульс, ловится Циолковским",
          "POOR_MASS_RATIO" in verdict.markers,
          f"отношение масс {verdict.mass_ratio:.2f}")

    # --- 9. перенос опыта в политику конструктора -----------------------
    from kia_core.engineer.autodesign import DesignPolicy
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "memory.json").write_text(json.dumps(
            {"total_runs": 9, "failure_lessons": {
                "aero_instability": {"count": 4}, "no_liftoff": {"count": 3}}}),
            encoding="utf-8")
        (root / "diag.json").write_text("[]", encoding="utf-8")
        (root / "flight.md").write_text("", encoding="utf-8")
        transfer = DesignerPro(reader=ExperienceReader(
            memory_file=root / "memory.json", diagnoses_file=root / "diag.json",
            flight_log=root / "flight.md", runs_dir=root / "нет"), stage=3)
        policy = DesignPolicy()
        before = (policy.fin_level, policy.liftoff_twr)
        applied = transfer.apply_to_policy(policy)
        check("опыт переносится в политику алгоритмического конструктора",
              policy.fin_level > before[0] and policy.liftoff_twr > before[1],
              ", ".join(f"{k}={v}" for k, v in applied.items()))

    # --- 10. подключение к среде сборки ---------------------------------
    from kia_core.neural.builder_env import BuilderEnv
    try:
        base_env = BuilderEnv(stage=1)
        pro_env = BuilderEnv(stage=2)
    except Exception as exc:
        check("среда сборки принимает ступень", False, str(exc)[:60])
    else:
        check("среда сборки подключает бюро только со ступени 2",
              base_env.designer is None and pro_env.designer is not None)

    # --- 11. модуль не трогает игру -------------------------------------
    # Проверяем ИМПОРТЫ, а не текст: слово «kRPC» встречается в описании
    # модуля именно потому, что он к нему не обращается.
    import ast
    source = (Path(__file__).resolve().parent.parent / "pro_modules"
              / "designer_pro.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = sorted(name for name in imported
                       if any(word in name.lower() for word in
                              ("krpc", "dispatcher", "connection", "control",
                               "provision", "autonomy")))
    check("бюро не импортирует ничего, что открывает игру",
          not forbidden, ", ".join(forbidden) if forbidden
          else f"{len(imported)} импортов, все безопасные")


# ==========================================================================
def _check_against_stock_craft(catalog) -> None:
    """Правило «куда наружу» против штатных чертежей KSP.

    Берутся Ships/**/*.craft из установки игры, оттуда — детали,
    прикреплённые поверхностно к ЦИЛИНДРИЧЕСКОМУ баку. Для такой детали
    направление наружу известно точно: от оси родителя к детали. С ним и
    сверяется то, что считает `outward_axis`.

    Два фильтра, и оба обязательны.

    По ЦИЛИНДРУ. Без него в выборку попадают панели на носовых конусах и
    стойки на крыльях, у которых «наружу» вообще не определено, и верное
    правило показывает 47 % совпадений.

    По СОСТАВУ. Проверяются только те детали, которые KIA сама вешает
    радиально. Остальное — солнечные панели, RCS, шасси, стойки,
    топливопроводы — игрок ставит под произвольным углом (панель
    разворачивают к солнцу, а не наружу), и требовать от них правила
    бессмысленно. Правило и не обязано их описывать: KIA их не ставит.
    """
    import re
    from kia_core.engineer.craft_writer import _rotate, outward_axis

    interesting = set()
    for group in (catalog.fins(), catalog.launch_clamps(),
                  catalog.radial_decouplers(), catalog.solid_boosters(),
                  catalog.batteries(), catalog.antennas(), catalog.science(),
                  catalog.parachutes()):
        interesting.update(part.name for part in group)

    gamedata = Path(catalog.source or "")
    # Только VAB: там ракеты стоят вертикально, и «наружу» — это честная
    # горизонталь от оси. У самолёта из SPH фюзеляж лежит, и то же самое
    # оперение крепится вдоль корпуса — эталон там неприменим.
    ships = gamedata.parent / "Ships" / "VAB" if gamedata.name == "GameData" else None
    files = sorted(ships.glob("*.craft")) if ships and ships.exists() else []
    if not files:
        check("правило сверено с чертежами KSP", True,
              "игра не найдена — проверка пропущена")
        return

    agree = disagree = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        parts = {}
        for block in re.findall(r"PART\s*\{(.*?)\n\}", text, re.S):
            name = re.search(r"^\s*part\s*=\s*(.+)$", block, re.M)
            pos = re.search(r"^\s*pos\s*=\s*(.+)$", block, re.M)
            rot = re.search(r"^\s*rot\s*=\s*(.+)$", block, re.M)
            if not (name and pos and rot):
                continue
            srf = re.search(r"^\s*srfN\s*=\s*(.+)$", block, re.M)
            try:
                parts[name.group(1).strip()] = {
                    "pos": [float(v) for v in pos.group(1).split(",")],
                    "rot": [float(v) for v in rot.group(1).split(",")],
                    "srf": srf.group(1).strip() if srf else None,
                    "name": name.group(1).strip().rsplit("_", 1)[0]}
            except ValueError:
                continue
        for part in parts.values():
            if not part["srf"]:
                continue
            fields = part["srf"].split(",")
            parent = parts.get(fields[1].strip()) if len(fields) > 1 else None
            if parent is None:
                continue
            host = catalog.get(parent["name"].replace(".", "_"))
            info = catalog.get(part["name"].replace(".", "_"))
            if host is None or info is None or not host.is_stack_tank:
                continue
            if info.name not in interesting:
                continue
            offset = [part["pos"][i] - parent["pos"][i] for i in range(3)]
            span = math.hypot(offset[0], offset[2])
            if span < host.diameter / 2 * 0.6:
                continue                       # не сбоку, а сверху или снизу
            actual = math.degrees(math.atan2(offset[2], offset[0]))
            got = _rotate(part["rot"], outward_axis(info))
            predicted = math.degrees(math.atan2(got[2], got[0]))
            if abs(((predicted - actual + 180) % 360) - 180) < 25:
                agree += 1
            else:
                disagree += 1

    total = agree + disagree
    check("правило «куда наружу» совпадает с чертежами самой игры",
          total > 20 and agree / total > 0.95,
          f"совпало {agree} из {total} деталей на цилиндрических баках")


def test_radial_geometry() -> None:
    """[26] Геометрия радиальной навески: ускорители, мачты, оперение."""
    print("\n[26] Радиальная навеска и обтекаемость")
    from kia_core.engineer.craft_writer import (BOOSTER_GAP, Blueprint,
                                                BoosterBuild, DECOUPLER_ARM,
                                                StageBuild)

    catalog = get_catalog()

    # --- поворот детали должен смотреть НАРУЖУ на всех четырёх углах ---
    # Здесь было две ошибки подряд. Сначала знак поворота: детали на 90 и
    # 270 градусах разворачивало на 180, а на 0 и 180 всё выглядело верно,
    # поэтому глазами баг ловился только наполовину. Потом выяснилось, что
    # «наружу» у каждой детали своя сторона — у оперения это −X, у мачты
    # −Z, у ускорителя +Z, — и общее допущение «всегда +X» разворачивало
    # оперение внутрь бака. Правило снято с штатных чертежей KSP.
    from kia_core.engineer.craft_writer import _rotate, outward_axis

    def outward(part) -> tuple[float, float, float]:
        return _rotate(part.rotation, outward_axis(part.info))

    engine = catalog.engines(diameter=1.25)[0]
    tank = catalog.tanks(diameter=1.25)[0]
    pod = catalog.probe_cores()[0]
    fin = catalog.fins(steerable=False)[0]
    blueprint = Blueprint(pod=pod, stages=[StageBuild(engine=engine,
                                                      tanks=[tank], index=0)],
                          fins=[fin] * 4, name="KIA_Test_Radial")
    assembler = CraftAssembler(catalog=catalog, seed=1)
    placed = assembler.assemble(blueprint)
    fins = [p for p in placed if p.info.name == fin.name]
    check("оперение поставлено симметрично", len(fins) == 4,
          f"{len(fins)} шт")
    worst = 0.0
    for part in fins:
        radius = math.hypot(part.position[0], part.position[2])
        if radius < 1e-9:
            continue
        want = (part.position[0] / radius, 0.0, part.position[2] / radius)
        got = outward(part)
        worst = max(worst, math.hypot(got[0] - want[0], got[2] - want[2]))
    check("радиальная деталь развёрнута НАРУЖУ, а не внутрь корпуса",
          worst < 1e-6, f"худшее расхождение {worst:.2e}")

    # Правило «куда деталь смотрит наружу» снято с чертежей самой игры.
    # Если оно поедет, все радиальные детали встанут боком, а тесты выше
    # этого не заметят: они проверяют согласованность, а не правильность.
    clamp = catalog.launch_clamps()[0]
    srb = next(p for p in catalog.solid_boosters() if p.diameter > 0)
    check("наружная сторона берётся из геометрии детали, а не одна на всех",
          outward_axis(fin) == (-1.0, 0.0, 0.0)
          and outward_axis(clamp) == (0.0, 0.0, -1.0)
          and outward_axis(srb) == (0.0, 0.0, 1.0),
          f"оперение {outward_axis(fin)}, мачта {outward_axis(clamp)}, "
          f"ускоритель {outward_axis(srb)}")

    # Узел крепления обязан лечь НА ОБШИВКУ несущей детали. Раньше туда
    # ставилось начало координат, и мачта, у которой захват вынесен на
    # 0.8 м, оказывалась внутри бака — это было видно на снимке из игры.
    clamped = Blueprint(pod=pod, stages=[StageBuild(engine=engine,
                                                    tanks=[tank], index=0)],
                        clamps=[clamp] * 4, name="KIA_Test_Clamps")
    asm_c = CraftAssembler(catalog=catalog, seed=4)
    placed_c = asm_c.assemble(clamped)
    host = min((p for p in placed_c if p.info.is_stack_tank),
               key=lambda p: p.position[1])
    skin = max(0.35, host.info.diameter / 2.0)
    worst_node = 0.0
    for part in [p for p in placed_c if p.info.name == clamp.name]:
        node = _rotate(part.rotation, part.info.surface_node or (0.0, 0.0, 0.0))
        node_r = math.hypot(part.position[0] + node[0],
                            part.position[2] + node[2])
        worst_node = max(worst_node, abs(node_r - skin))
    check("узел крепления мачты ложится на обшивку, а не внутрь бака",
          worst_node < 0.02, f"отклонение {worst_node:.3f} м при обшивке "
                             f"{skin:.3f} м")
    body_r = math.hypot(*[[p for p in placed_c
                           if p.info.name == clamp.name][0].position[i]
                          for i in (0, 2)])
    # Корпус мачты не имеет права опускаться ниже аппарата. Игра ставит
    # судно так, чтобы нижняя точка касалась стола; если ею оказывается
    # мачта, ракета спавнится подвешенной и садится на захваты с
    # перекосом — на снимке из игры она стояла заметно криво.
    bodies = [p.position[1] for p in placed_c if p.info.name == clamp.name]
    others = [p.position[1] for p in placed_c if p.info.name != clamp.name]
    check("корпус мачты не свисает ниже самой ракеты",
          min(bodies) >= min(others) - 0.01,
          f"мачты с {min(bodies):.2f}, аппарат с {min(others):.2f}")

    check("корпус мачты при этом стоит СНАРУЖИ корпуса",
          body_r > skin + 0.5, f"корпус на {body_r:.2f} м, обшивка {skin:.2f} м")

    # --- сверка правила с чертежами самой игры ---
    # Единственный настоящий эталон здесь — то, как детали расставляет KSP.
    # Если игра не установлена, проверка пропускается: остальные тесты от
    # неё не зависят.
    _check_against_stock_craft(catalog)

    # --- ускорители: чётность, вынос, стадии, дерево ---
    boosters = catalog.solid_boosters()
    check("каталог отличает ускорители от башни спасения и сепаратрона",
          bool(boosters) and all(p.category == "Engine" and p.diameter > 0
                                 for p in boosters)
          and not any("LaunchEscape" in p.name for p in boosters),
          f"{len(boosters)} шт")
    check("каталог находит радиальные разделители и носовые конусы",
          bool(catalog.radial_decouplers()) and bool(catalog.nose_cones()),
          f"{len(catalog.radial_decouplers())} разделителей, "
          f"{len(catalog.nose_cones())} конусов")

    srb = next(p for p in boosters if p.diameter > 0 and p.top_node)
    decoupler = catalog.radial_decouplers()[0]
    cone = catalog.nose_cones()[0]
    pack = BoosterBuild(booster=srb, decoupler=decoupler, nose=cone, count=4)
    bp2 = Blueprint(pod=pod, stages=[StageBuild(engine=engine, tanks=[tank],
                                                index=0)],
                    boosters=pack, nose=cone, name="KIA_Test_Boosters")
    asm2 = CraftAssembler(catalog=catalog, seed=2)
    placed2 = asm2.assemble(bp2)
    srbs = [p for p in placed2 if p.info.name == srb.name]
    check("ускорителей навешено ровно столько, сколько заказано",
          len(srbs) == 4, f"{len(srbs)} шт")

    core = next(p for p in placed2 if p.info.is_stack_tank)
    # ЭТАЛОН — ЧУЖОЙ ЧЕРТЁЖ, А НЕ УЗЛОВОЙ ДИАМЕТР.
    #
    # Прежние проверки мерили от `diameter` (размер стыковочного узла), а
    # у баков Making History он в полтора раза больше настоящей обшивки:
    # 2.5 против 1.875. Из-за этого связка уезжала на полметра от
    # корпуса, висла на коротком разделителе, и её рвало на старте.
    #
    # Замер по `УРНА-3`: бак обшивкой 1.250, ускоритель BACC центром на
    # 1.951 при своей обшивке 0.625 — зазор между бортами 0.076 м.
    core_r = max(0.35, core.info.hull_diameter / 2)
    srb_r = max(0.25, srb.hull_diameter / 2)
    body_radii = [math.hypot(p.position[0], p.position[2]) for p in srbs]
    gaps = [r - srb_r - core_r for r in body_radii]
    check("ускоритель стоит вплотную к корпусу, как в чертежах игры",
          all(0.02 <= g <= 0.25 for g in gaps),
          f"зазор между бортами {gaps[0]:.3f} м (эталон игры 0.076)")
    check("корпус ускорителя не пересекает центральный блок",
          all(g > 0 for g in gaps),
          f"зазор {gaps[0]:.3f} м")
    check("ускорители висят на разделителях, а не прямо на баке",
          all(p.parent is not None and p.parent.info.is_decoupler for p in srbs))
    check("ускоритель зажигается раньше, чем сбрасывается",
          all(p.stage > p.decouple_stage for p in srbs),
          f"istg {srbs[0].stage}, dstg {srbs[0].decouple_stage}")
    engines_placed = [p for p in placed2 if p.info.is_engine
                      and not p.info.is_solid_booster]
    check("ускорители зажигаются вместе с нижним двигателем",
          bool(engines_placed)
          and srbs[0].stage == max(p.stage for p in engines_placed))
    text = asm2.render(bp2)
    check("чертёж с ускорителями проходит валидатор .craft",
          not validate_craft(text), "; ".join(validate_craft(text)[:2]))

    # --- расчёт связки в конструкторе ---
    engineer = AutonomousEngineer(catalog)
    spec = MissionSpec(title="Мун: посадка", target_body="Mun",
                       home_body="Kerbin", objectives=["land"],
                       target_orbit_altitude=80_000)
    plain = engineer.design(spec, DesignPolicy(booster_level=0.0))
    # Связка считается только для блока, которому НЕ ХВАТАЕТ тяги. С тех
    # пор как подбор двигателя требует качающегося сопла, solver находит
    # вариант с запасом, и через design() эта ветка не вызывается вовсе.
    # Поэтому логика подбора проверяется напрямую: собирается заведомо
    # слабая ступень, и у неё запрашивается связка.
    from kia_core.engineer.autodesign import StageRequirement, StageSolution

    weak_engine = min((e for e in catalog.engines(diameter=1.25)
                       if e.max_thrust > 0), key=lambda e: e.max_thrust)
    big_tank = max(catalog.tanks(diameter=1.25), key=lambda t: t.fuel_units())
    weak_req = StageRequirement(index=0, role="старт", dv_required=2000.0,
                                min_twr=1.55, gravity=9.81, pressure=0.45,
                                twr_body="Kerbin")
    weak = StageSolution(requirement=weak_req, engine=weak_engine,
                         tank=big_tank, tank_count=6, decoupler=None,
                         payload_mass=2.0)
    weak._tank_fuel = big_tank.resource_mass(catalog.densities)
    weak_policy = DesignPolicy(booster_level=0.7, allow_boosters=True,
                               liftoff_twr=1.55)
    pack = engineer.choose_boosters(spec, weak_policy, weak)
    check("слабому блоку связка нужна",
          weak.twr() < 1.55 and pack is not None,
          f"блок даёт TWR {weak.twr():.2f} при цели 1.55")

    boosted = engineer.design(spec, DesignPolicy(booster_level=0.7,
                                                 allow_boosters=True,
                                                 liftoff_twr=2.10))
    check("без разрешения ускорители не появляются",
          plain.boosters is None)
    # Рубильник включён обратно: причина прежних четырёх провалов найдена
    # не в связке, а в условии отстрела (см. `should_stage`). Стадии
    # связки сверены со штатным чертежом `УРНА-3`: ускоритель зажигается
    # вместе с центральным двигателем, радиальный разделитель сбрасывает
    # его следующей стадией.
    off = engineer.design(spec, DesignPolicy(booster_level=0.0,
                                             liftoff_twr=1.25))
    check("на нулевом уровне связка не навязывается",
          off.boosters is None or off.boosters.count == 0,
          "ускорители появились при booster_level=0")
    check("рубильник по умолчанию открыт",
          DesignPolicy().allow_boosters,
          "ускорители снова разрешены после разбора стадий")
    if boosted.boosters is not None:
        check("ускоритель зажигается вместе с центральным двигателем",
              boosted.boosters.count % 2 == 0,
              "связка обязана быть чётной и симметричной")
    check("связка чётная и симметричная",
          pack is not None and pack.count % 2 == 0,
          f"{pack.count if pack else 0} шт")
    if pack is not None:
        check("ускорители поднимают стартовый TWR, а не топят его",
              pack.twr > pack.core_twr,
              f"{pack.core_twr:.2f} -> {pack.twr:.2f}")
        check("каждый ускоритель вносит вклад в плюс",
              pack.thrust_each > 1.55 * 9.81 * pack.wet_each,
              f"тяга {pack.thrust_each:.0f} кН при массе {pack.wet_each:.2f} т")
    check("обтекатель ставится при старте с атмосферного тела",
          boosted.blueprint.nose is not None)
    vacuum = engineer.design(
        MissionSpec(title="старт с Муны", target_body="Mun", home_body="Mun",
                    target_orbit_altitude=20_000), DesignPolicy())
    check("в вакууме обтекатель не берётся — это мёртвый груз",
          vacuum.blueprint.nose is None)


def test_math_trainer() -> None:
    """[27] Тренажёр математики: эталон, кодировка, перенос в игру."""
    print("\n[27] Тренажёр математики")
    import random
    from kia_core.neural.math_env import (MATH_OBS_DIM, MathEnv, TASK_BUILDERS,
                                          TOLERANCE_BY_STAGE, tasks_of_stage)
    from kia_core.neural.programs import (BUILDER_STAGES, MATH_STAGES,
                                          describe_program, resolve, stages_of)

    rng = random.Random(0)
    check("задачи разложены по трём ступеням",
          all(tasks_of_stage(n) for n in (1, 2, 3)),
          f"{len(tasks_of_stage(1))}/{len(tasks_of_stage(2))}/"
          f"{len(tasks_of_stage(3))} задач")

    outside = 0
    for slot, builder in enumerate(TASK_BUILDERS):
        for _ in range(400):
            task = builder(rng, slot)
            for index, truth in enumerate(task.answers):
                if not (task.low[index] <= truth <= task.high[index]):
                    outside += 1
    check("эталон всегда лежит внутри объявленного диапазона ответа",
          outside == 0, f"вне диапазона {outside}")

    # Кодировка «ответ -> выход сети -> ответ» обязана быть обратимой:
    # на ней держится обучение регрессией.
    worst = 0.0
    for slot, builder in enumerate(TASK_BUILDERS):
        for _ in range(200):
            task = builder(rng, slot)
            restored = task.decode(task.target())
            worst = max(worst, max(task.errors(restored)))
    check("эталон переводится в выход сети и обратно без потери",
          worst < 1e-6, f"худшая ошибка {worst:.2e}")

    env = MathEnv(stage=1, seed=3)
    check("длина наблюдения постоянна и совпадает с объявленной",
          len(env.reset()) == MATH_OBS_DIM, f"{MATH_OBS_DIM}")
    env.reset()
    _obs, reward, done, info = env.step(env.task.target())
    check("точный ответ даёт максимум награды и засчитывается",
          done and info["solved"] and reward > 1.4,
          f"награда {reward:.2f}, ошибка {info['error']:.1e}")
    env2 = MathEnv(stage=1, seed=4)
    env2.reset()
    _o, bad_reward, _d, bad_info = env2.step([1.0, 1.0])
    check("грубо неверный ответ наказывается",
          bad_reward < reward and not bad_info["solved"],
          f"{bad_reward:.2f} против {reward:.2f}")

    check("допуск ужесточается от ступени к ступени",
          TOLERANCE_BY_STAGE[1] > TOLERANCE_BY_STAGE[2] > TOLERANCE_BY_STAGE[3],
          " > ".join(f"{TOLERANCE_BY_STAGE[n]:.0%}" for n in (1, 2, 3)))
    check("ступень 3 повторяет задачи прошлых ступеней",
          any(task_slot in tasks_of_stage(1) + tasks_of_stage(2)
              for task_slot in [MathEnv(stage=3, seed=s)._pick_slot()
                                for s in range(60)]),
          "иначе сеть забывает пройденное")

    check("обе программы описываются и разрешаются по имени и номеру",
          len(stages_of("math")) == 3 and len(stages_of("builder")) == 3
          and resolve("math", "2").number == 2
          and resolve("builder", "build_pro").number == 3
          and "Матан" in describe_program("math"))
    check("снимки ступеней у программ разные",
          len({s.backup for s in MATH_STAGES if s.backup}
              | {s.backup for s in BUILDER_STAGES if s.backup}) == 4)

    # --- перенос в игру ---
    from kia_core.neural.math_brain import MathBrain
    brain = MathBrain(tolerance=0.05, path=Path(tempfile.gettempdir())
                      / "kia_math_skills_test.json")
    result = brain.verify("circular_speed", 2246.0,
                          [3531600000000.0 / 3.5e12, 1.0, 0.8, 0, 0, 0])
    check("мост в игру считает ответ сети и копит статистику",
          result.get("checked") and "guess" in result,
          f"ошибка {result.get('error', 0) * 100:.1f} %")
    check("статистика раздела растёт от проверки к проверке",
          brain.skills["circular_speed"].checks == 1
          and "circular_speed" in brain.report())
    exact_only = brain.verify("такого раздела нет", 1.0, [0] * 6)
    check("неизвестный раздел не ломает полёт, а просто не проверяется",
          exact_only["checked"] is False and exact_only["exact"] == 1.0)


def test_builder_stages() -> None:
    """[28] Ступени конструктора: цели, обвязка, собираемость."""
    print("\n[28] Ступени сети-конструктора")
    import random
    from kia_core.neural.builder_env import BuilderEnv, FINISH_ACTION

    goals = set()
    env2 = BuilderEnv(stage=2, seed=11)
    for _ in range(40):
        env2.reset()
        goals.add((env2.goal_altitude, env2.goal_transfer, env2.goal_landing))
    check("со ступени 2 задача меняется от эпизода к эпизоду",
          len(goals) > 3, f"{len(goals)} разных задач")

    env1 = BuilderEnv(stage=1, seed=11)
    goals1 = set()
    for _ in range(20):
        env1.reset()
        goals1.add((env1.goal_altitude, env1.goal_transfer, env1.goal_landing))
    check("на ступени 1 цель одна — сеть учится ставить детали",
          len(goals1) == 1)

    # Одна и та же ГОДНАЯ сборка на разных ступенях: спрос обязан расти.
    # Берём заведомо рабочий набор, иначе сравнение выродится в «ноль
    # против нуля» и ничего не проверит.
    catalog0 = get_catalog()
    palette = BuilderEnv(stage=1, catalog=catalog0, seed=5).palette.parts
    wanted = []
    for predicate in (lambda p: p.is_command,
                      lambda p: p.is_stack_tank,
                      lambda p: p.is_engine and p.is_liquid_engine):
        found = next((i for i, p in enumerate(palette) if predicate(p)), None)
        if found is not None:
            wanted.append(found)

    def run(stage: int) -> float:
        env = BuilderEnv(stage=stage, catalog=catalog0, seed=5)
        env.reset(goal_altitude=80_000.0, goal_transfer=False,
                  goal_landing=False)
        total = 0.0
        for action in wanted + [FINISH_ACTION]:
            _o, reward, done, _i = env.step(action)
            total += reward
            if done:
                break
        return total

    plain, strict = run(1), run(3)
    check("одна и та же сборка на ступени 3 оценивается строже",
          len(wanted) == 3 and strict < plain,
          f"{plain:+.2f} -> {strict:+.2f}")

    scores = {}
    for stage in (1, 2, 3):
        env = BuilderEnv(stage=stage, seed=7)
        rng = random.Random(9)
        valid = 0
        for _ in range(120):
            env.reset()
            done = False
            info = {}
            while not done:
                mask = env.action_mask()
                action = rng.choice([i for i, ok in enumerate(mask) if ok])
                _o, _r, done, info = env.step(action)
            evaluation = info.get("evaluation")
            valid += int(evaluation is not None and evaluation.valid)
        scores[stage] = valid
    check("каждая следующая ступень пропускает меньше случайных сборок",
          scores[1] > scores[2] > scores[3],
          f"годных {scores[1]} -> {scores[2]} -> {scores[3]} из 120")

    # ΔV по ступеням: одноступенчатая и двухступенчатая сборки одинаковой
    # массы не могут иметь одинаковый запас скорости.
    catalog = get_catalog()
    env3 = BuilderEnv(stage=3, catalog=catalog, seed=1)
    env3.reset()
    core = catalog.probe_cores()[0]
    tank = catalog.tanks(diameter=1.25)[0]
    engine = catalog.engines(diameter=1.25)[0]
    env3.chosen = [core, tank, engine]
    env3._absorb(core)
    staged = env3._staged_performance(catalog.densities)
    check("расчёт по ступеням выдаёт ΔV и TWR отдельно",
          staged is not None and staged[0] > 0 and staged[1] > 0,
          f"ΔV {staged[0]:.0f} м/с, TWR {staged[1]:.2f}" if staged else "нет")

    env_bad = BuilderEnv(stage=3, catalog=catalog, seed=1)
    env_bad.reset()
    env_bad.chosen = [core]
    problems = env_bad._buildability_problems()
    check("несобираемый набор деталей ступень 3 не засчитывает",
          bool(problems), problems[0] if problems else "")


def test_autostage_one_at_a_time() -> None:
    """[33] Автостейджинг делит по одной ступени, а не пачкой."""
    print("\n[33] Отделение ступеней")
    from types import SimpleNamespace
    from kia_core.environment.control import ShipControl

    class FakeVessel:
        """Судно, у которого сразу после отделения тяги ещё нет.

        Именно так ведёт себя игра: насосы раскручиваются, двигатель
        выходит на режим около секунды, и всё это время остатки по новой
        группе разделения читаются нулевыми.
        """

        def __init__(self):
            self.control = SimpleNamespace(current_stage=3, throttle=1.0,
                                           activate_next_stage=self._stage)
            self.stagings = 0
            self.available_thrust = 0.0

        def _stage(self):
            self.stagings += 1
            self.control.current_stage -= 1
            return []

    vessel = FakeVessel()
    control = ShipControl.__new__(ShipControl)
    control.vessel = vessel
    control.connection = None
    control.should_stage = lambda: vessel.control.current_stage > 0
    control.set_throttle = lambda value: setattr(vessel.control, "throttle", value)
    control.next_stage = vessel._stage

    import time as _time
    started = _time.perf_counter()
    staged = control.autostage()
    elapsed = _time.perf_counter() - started

    check("за один вызов отделяется ровно одна ступень",
          staged and vessel.stagings == 1,
          f"отделений {vessel.stagings} (цикл while делал их подряд, пока "
          f"не дойдёт до нуля)")
    check("новой ступени дают ожить перед следующим решением",
          elapsed >= 1.0, f"пауза {elapsed:.2f} с")
    check("тяга возвращается после отделения",
          vessel.control.throttle == 1.0)

    vessel.control.current_stage = 0
    check("ниже минимальной стадии не делим",
          not control.autostage(), "иначе уйдём в отстрел парашюта")


def test_periapsis_shaping() -> None:
    """[32] Счёт различает «почти орбита» и «просто подпрыгнул»."""
    print("\n[32] Оценка приближения к орбите")
    from kia_core.learning.rewards import RewardSystem

    spec = MissionSpec(title="орбита", target_body="Kerbin",
                       home_body="Kerbin", park_orbit_altitude=80_000)

    # Весь суборбитальный диапазон раньше был плоским: shape_orbit_quality
    # возвращает ноль при любом отрицательном периапсисе, а он отрицателен
    # во всех полётах до самой орбиты. Аппарат, которому остался один
    # манёвр, и аппарат, который просто подпрыгнул, получали одинаково.
    near = RewardSystem(spec)
    near.shape_periapsis(80_000.0, -138_000.0, 80_000.0)
    far = RewardSystem(spec)
    far.shape_periapsis(80_000.0, -585_000.0, 80_000.0)
    check("почти замкнутая орбита ценится выше глубоко суборбитальной",
          near.total > far.total,
          f"Pe -138 км: {near.total:.0f} очков против {far.total:.0f} при -585 км")

    low = RewardSystem(spec)
    check("без набранной высоты за периапсис не платят",
          low.shape_periapsis(20_000.0, -138_000.0, 80_000.0) == 0.0,
          "иначе выгодно летать низко и ровно, не выходя в космос")

    ratchet = RewardSystem(spec)
    ratchet.shape_periapsis(80_000.0, -200_000.0, 80_000.0)
    before = ratchet.total
    ratchet.shape_periapsis(80_000.0, -300_000.0, 80_000.0)
    check("платим только за улучшение, а не каждый тик",
          ratchet.total == before,
          "иначе награда капала бы всю миссию и заглушила вехи")

    closed = RewardSystem(spec)
    check("замкнутой орбите эта оценка уже не нужна",
          closed.shape_periapsis(80_000.0, 75_000.0, 80_000.0) == 0.0,
          "там работает stable_orbit и оценка круговости")


def test_reference_designs() -> None:
    """[31] Обучение на образцах: нормы снимаются с готовых чертежей."""
    print("\n[31] Нормы, снятые с образцов")
    import tempfile
    from kia_core.engineer import reference_designs as ref
    from kia_core.engineer.craft_writer import Blueprint, CraftAssembler, StageBuild

    catalog = get_catalog()
    empty = ref.ReferenceNorms()
    check("без образцов нормы ничего не требуют",
          not empty.known and not empty.wants_rcs(1000.0)
          and not empty.wants_fairing(1000.0),
          "конструктор должен работать и без папки с чертежами")

    # Собираем «образец» из собственного сборщика и учимся на нём: так
    # проверка не зависит от того, что лежит у оператора в сохранении.
    folder = Path(tempfile.mkdtemp())
    pod = catalog.probe_cores()[0]
    engine = max(catalog.engines(diameter=1.25), key=lambda e: e.gimbal)
    tank = catalog.tanks(diameter=1.25)[0]
    wheel = catalog.reaction_wheels()[0]
    # Не меньше пяти деталей: разборщик отсекает пустышки и
    # автосохранения, и четырёхдетальный образец в выборку не попадёт.
    blueprint = Blueprint(pod=pod, radial_payload=[wheel, wheel],
                          stages=[StageBuild(engine=engine,
                                             tanks=[tank, tank, tank], index=0)],
                          name="Образец-эталон")
    (folder / "Образец.craft").write_text(
        CraftAssembler(craft_name="Образец-эталон", catalog=catalog,
                       seed=1).render(blueprint),
        encoding="utf-8")
    norms = ref.learn(folder, catalog)
    check("нормы снимаются с чертежа и помнят, на чём учились",
          norms.known and norms.sample_count == 1 and norms.sample_names,
          f"образцов {norms.sample_count}")
    check("свои чертежи в образцы не берутся",
          not any(n.startswith("KIA") for n in norms.sample_names),
          "иначе система училась бы на собственных ошибках")

    saved = ref.save(norms, folder / "нормы.json")
    again = ref.load(saved, refresh=True)
    check("нормы переживают запись и чтение",
          again.sample_count == norms.sample_count
          and again.wheels_light == norms.wheels_light)
    ref.load(refresh=True)      # вернуть общий кэш на место

    # Главное, ради чего всё затевалось: двигатель нижней ступени обязан
    # иметь качающееся сопло, иначе рулить в атмосфере нечем.
    engineer = AutonomousEngineer(catalog)
    spec = MissionSpec(title="орбита", target_body="Kerbin",
                       home_body="Kerbin", target_orbit_altitude=80_000)
    design = engineer.design(spec, DesignPolicy())
    check("двигатель нижней ступени рулит соплом",
          design.stages and design.stages[0].engine.gimbal > 0,
          f"{design.stages[0].engine.title} — качание "
          f"{design.stages[0].engine.gimbal:.1f}°" if design.stages else "нет ступеней")

    airless = engineer.design(
        MissionSpec(title="с Муны", target_body="Mun", home_body="Mun",
                    target_orbit_altitude=20_000), DesignPolicy())
    check("в вакууме качание не требуется — рулить нечему",
          bool(airless.stages),
          "требование не должно делать задачу нерешаемой без атмосферы")


def test_ascent_failures() -> None:
    """[30] Живой полёт замечает срыв выведения, а не только пустой бак."""
    print("\n[30] Отказы выведения в живой игре")
    from types import SimpleNamespace
    from kia_core.learning.rewards import Failure, RewardSystem
    from kia_core.learning.genome import FlightGenome
    from kia_core.pilot import ascent as asc
    from kia_core.pilot import ascent as ASC

    class FakeControl:
        def __init__(self):
            self.throttle = 1.0

        def set_throttle(self, value):
            self.throttle = value

    def make_pilot():
        rewards = RewardSystem()
        control = FakeControl()
        pilot = asc.AscentAutopilot(
            connection=SimpleNamespace(space_center=None),
            telemetry=SimpleNamespace(), control=control,
            maneuver=None, flight_genome=FlightGenome(),
            rewards=rewards, target_apoapsis=80_000.0, flight_log=None)
        return pilot, rewards, control

    def snap(**kw):
        base = dict(mission_time=0.0, altitude=5_000.0, apoapsis=7_000.0,
                    speed=250.0, vertical_speed=50.0, dynamic_pressure=9_000.0,
                    angle_of_attack=0.0, sideslip_angle=0.0,
                    aero_angles_known=True)
        base.update(kw)
        return SimpleNamespace(**base)

    # --- опрокидывание ---
    pilot, rewards, control = make_pilot()
    fired = [pilot._tumbling(snap(mission_time=t, angle_of_attack=70.0))
             for t in (0.0, 1.0, 2.0, 5.0)]
    check("опрокидывание ловится только после выдержки, а не с первого кадра",
          fired == [False, False, False, True], f"{fired}")
    check("отказ записан как потеря управления и с полным счётом",
          any(e.label == Failure.LOST_CONTROL.value for e in rewards.events)
          and rewards.total < -300, f"счёт {rewards.total:.0f}")
    check("двигатель при этом глушится",
          control.throttle == 0.0)

    pilot, _r, _c = make_pilot()
    calm = [pilot._tumbling(snap(mission_time=t, angle_of_attack=5.0))
            for t in (0.0, 5.0, 10.0)]
    check("ровный полёт опрокидыванием не считается", not any(calm))

    pilot, _r, _c = make_pilot()
    vacuum = [pilot._tumbling(snap(mission_time=t, angle_of_attack=120.0,
                                   dynamic_pressure=5.0))
              for t in (0.0, 5.0, 10.0)]
    check("в вакууме угол атаки ничего не значит и отказом не считается",
          not any(vacuum), "иначе каждый разворот на орбите был бы аварией")

    pilot, _r, _c = make_pilot()
    blind = [pilot._tumbling(snap(mission_time=t, angle_of_attack=90.0,
                                  aero_angles_known=False))
             for t in (0.0, 5.0, 10.0)]
    check("без углов от kRPC детектор молчит, а не гадает", not any(blind))

    # Закрутило по КУРСУ: угол атаки почти нулевой, а аппарат летит боком.
    # По одному тангажу такой срыв не виден вовсе.
    pilot, _r, _c = make_pilot()
    yawed = [pilot._tumbling(snap(mission_time=t, angle_of_attack=3.0,
                                  sideslip_angle=75.0))
             for t in (0.0, 1.0, 2.0, 5.0)]
    check("срыв по рысканию ловится так же, как по тангажу",
          yawed == [False, False, False, True],
          "аппарат с нулевым углом атаки может лететь боком")

    # --- сорванное выведение: апоапсис падает ---
    pilot, rewards, control = make_pilot()
    rising = [pilot._ascent_hopeless(snap(mission_time=t, apoapsis=a))
              for t, a in ((0.0, 3_000.0), (5.0, 6_000.0), (10.0, 9_000.0))]
    check("растущий апоапсис срывом не считается", not any(rising))
    falling = [pilot._ascent_hopeless(snap(mission_time=t, apoapsis=a))
               for t, a in ((15.0, 8_000.0), (20.0, 7_000.0), (30.0, 6_500.0))]
    check("падающий апоапсис обрывает полёт, а не тянет его до конца топлива",
          falling[-1] and not falling[0],
          f"{falling}")
    check("срыв выведения тоже идёт в счёт",
          any(e.label == Failure.LOST_CONTROL.value for e in rewards.events))

    # В отказ обязан попасть фактический тангаж: без него «завалился рано»
    # и «лез свечой» неотличимы, и аналитик лечит их одним лекарством —
    # так профиль и уехал в пол диапазона.
    detail = next(e.detail for e in rewards.events
                  if e.label == Failure.LOST_CONTROL.value)
    check("в отказе записан фактический тангаж и требуемый профилем",
          "pitch" in detail and "target_pitch" in detail,
          f"записано: {sorted(detail)}")

    # --- поводок по потоку ---
    # Профиль тангажа не знает, куда аппарат летит на самом деле. На
    # сверхзвуке в плотном воздухе он выворачивал нос против потока, и
    # аппарат опрокидывало — при оперении, уже упёртом в потолок.
    pilot, _r, _c = make_pilot()

    def leash(q, pitch, aoa, wanted):
        return pilot.airflow_leash(
            snap(dynamic_pressure=q, pitch=pitch, angle_of_attack=aoa), wanted)

    # летит под 60° (тангаж 74 минус угол атаки 14), профиль требует 25°
    tight = leash(18_000.0, 74.0, 14.0, 25.0)
    loose = leash(400.0, 74.0, 14.0, 25.0)
    check("при сильном напоре нос держится у потока",
          abs(tight - 60.0) <= ASC.AIRFLOW_LEASH_MIN + 1e-6,
          f"скомандовано {tight:.1f}° при потоке 60°")
    check("при разреженном воздухе ограничение почти не мешает",
          loose < tight, f"{loose:.1f}° против {tight:.1f}°")
    check("поводок затягивается плавно, а не ступенькой",
          leash(2_000.0, 74.0, 14.0, 25.0) < leash(6_000.0, 74.0, 14.0, 25.0)
          < tight)
    check("если команда и так по потоку, поводок её не трогает",
          abs(leash(18_000.0, 74.0, 14.0, 58.0) - 58.0) < 1e-6)
    # Поводок не имеет права следовать за падением: вектор скорости
    # зависит от того, куда аппарат направили, и слепое следование за ним
    # даёт положительную обратную связь. В игре это выглядело как команда
    # 30° на трёх километрах при профиле 89° — ракета легла и разбилась.
    falling = [leash(9_000.0, 74.0, aoa, 89.0) for aoa in (20.0, 40.0, 60.0)]
    check("поводок не тянет команду за падающим вектором скорости",
          all(v >= 89.0 - ASC.AIRFLOW_LEASH_BACKSTOP - 1e-6 for v in falling),
          f"при углах атаки 20/40/60° команды {[round(v) for v in falling]}")

    # Пока нос идёт ПО ПОТОКУ, поводку вмешиваться не во что. Он этого и
    # не делал бы, если бы не упирался в собственный ограничитель каждую
    # секунду: команда выходила на 20° ниже профиля весь полёт, аппарат
    # опускал нос, вектор скорости шёл следом — и ракета сама
    # разворачивала себя к земле при НУЛЕВОМ угле атаки.
    aligned = [leash(9_000.0, ang, 0.0, ang) for ang in (90.0, 86.0, 80.0, 77.0)]
    check("при полёте носом по потоку поводок молчит",
          all(abs(v - ang) < 1e-6 for v, ang in zip(aligned, (90.0, 86.0, 80.0, 77.0))),
          f"команды {[round(v) for v in aligned]} при профиле 90/86/80/77°")
    check("порог включения ниже угла, который считается срывом",
          0 < ASC.AIRFLOW_LEASH_ACTIVATE < ASC.TUMBLE_ANGLE,
          f"включается с {ASC.AIRFLOW_LEASH_ACTIVATE:.0f}°, "
          f"срыв с {ASC.TUMBLE_ANGLE:.0f}°")

    check("без углов от игры автопилот рулит как прежде",
          pilot.airflow_leash(
              SimpleNamespace(aero_angles_known=False), 33.0) == 33.0,
          "ограничение не должно ломать полёт там, где данных нет")

    # --- главное: петля до конструктора замкнута ---
    from pro_modules.designer_pro import TUMBLE_WORDS
    pilot, _r, _c = make_pilot()
    written = []
    pilot._fact = lambda text, detail="": written.append(f"{text} {detail}")
    for t in (0.0, 1.0, 2.0, 5.0):
        pilot._tumbling(snap(mission_time=t, angle_of_attack=70.0))
    text = " ".join(written).lower()
    check("запись об опрокидывании написана словами, которые читает бюро",
          any(word in text for word in TUMBLE_WORDS),
          "иначе конструктор не узнает, что надо добавить оперения")


def test_brain3d_programs() -> None:
    """[29] Трёхмерная карта показывает все три знания, а не одно."""
    print("\n[29] Математика и ракетостроение на карте мозга")
    from kia_core.neural import brain3d
    from kia_core.neural.ppo import build_brain

    brain3d._PROBE_CACHE.clear()
    brain = build_brain(load=False)
    data = brain3d.build(brain)
    counts = data["counts"]
    kinds = {node["kind"] for node in data["nodes"]}

    check("на карте появились узлы математики",
          "math" in kinds and counts["math"] > 0,
          f"{counts['math']} разделов")
    check("на карте появились узлы ракетостроения",
          "design" in kinds and counts["design"] > 0,
          f"{counts['design']} умений")
    check("ступени обеих программ выведены отдельно от пилотской",
          data["programs"]["math_stage"] >= 1
          and data["programs"]["builder_stage"] >= 1
          and data["stage"] >= 1)

    shells = data["geometry"]["shells"]
    check("у математики и конструктора свои оболочки, не смешанные с полётом",
          brain3d.MATH_SHELL in shells and brain3d.DESIGN_SHELL in shells
          and brain3d.MATH_SHELL > max(brain3d.SKILL_SHELL)
          and brain3d.DESIGN_SHELL > brain3d.MATH_SHELL)

    math_nodes = [n for n in data["nodes"] if n["kind"] == "math"]
    design_nodes = [n for n in data["nodes"] if n["kind"] == "design"]
    check("у каждого узла знания есть измеренная доля, а не просто галочка",
          all(0.0 <= n["strength"] <= 1.0 for n in math_nodes + design_nodes)
          and all("решено" in n["detail"] for n in math_nodes)
          and all("сборок" in n["detail"] for n in design_nodes))
    check("необученный мозг не выдаёт себя за знающего",
          all(not n["mastered"] for n in math_nodes),
          f"освоено {counts['math_mastered']} из {counts['math']} у чистой сети")

    ids = {node["id"] for node in data["nodes"]}
    check("все связи ведут в существующие узлы",
          all(edge["a"] in ids and edge["b"] in ids for edge in data["edges"]),
          f"{len(data['edges'])} связей")
    check("зависимости разделов математики ведут от простого к сложному",
          brain3d.MATH_PARENTS["hohmann"] and brain3d.MATH_PARENTS["vis_viva"]
          and not brain3d.MATH_PARENTS["circular_speed"])

    # Замер не имеет права стоить времени на каждом кадре: карта
    # перечитывается раз в две секунды.
    import time as _time
    started = _time.perf_counter()
    brain3d.build(brain)
    cached = _time.perf_counter() - started
    check("повторная сборка карты идёт из кэша замеров",
          cached < 0.25, f"{cached * 1000:.0f} мс")

    page = (Path(__file__).resolve().parent.parent / "kia_brain3d.html")
    text = page.read_text(encoding="utf-8")
    check("страница знает про новые виды узлов и подписывает их",
          "math:" in text and "design:" in text
          and "Математика" in text and "Ракетостроение" in text)


def test_interfaces() -> None:
    """Все self.метод(), вызванные в классах интерфейса, должны существовать.

    Такая ошибка уже случалась: в пульт скопировали вызов self.spawn(),
    который был определён только в другом классе. Падало в рантайме и
    только при выборе конкретного пункта меню.
    """
    print("\n[21] Целостность интерфейсов")
    import ast

    for filename, classname in (("kia_control.py", "ControlPanel"),
                                ("cli.py", "Console")):
        source = (Path(__file__).resolve().parent.parent / filename)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        target = next((node for node in ast.walk(tree)
                       if isinstance(node, ast.ClassDef) and node.name == classname),
                      None)
        if target is None:
            check(f"{filename}: класс {classname} найден", False)
            continue

        defined = {node.name for node in target.body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assigned = set()
        for node in ast.walk(target):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "self" and isinstance(node.ctx, ast.Store):
                assigned.add(node.attr)

        missing = set()
        for node in ast.walk(target):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                    and func.value.id == "self":
                if func.attr not in defined and func.attr not in assigned:
                    missing.add(func.attr)
        check(f"{filename}: все вызовы self.* определены",
              not missing,
              ", ".join(sorted(missing)) if missing else f"{len(defined)} методов")

        # обработчики пунктов меню обязаны быть на месте
        handlers = {node.attr for node in ast.walk(target)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and node.attr.startswith(("action_", "cmd_", "neural_",
                                              "settings_"))}
        absent = {name for name in handlers if name not in defined}
        check(f"{filename}: обработчики меню на месте", not absent,
              ", ".join(sorted(absent)) if absent else f"{len(handlers)} обработчиков")



# ==========================================================================
class _FakeExam:
    """Двойник приёмной комиссии для проверок дежурного."""

    def __init__(self, episodes=0, orbits=0, rate=0.0, recent=0, mean=0.0,
                 stalled=0):
        self.episodes = episodes
        self.orbits = orbits
        self.orbit_rate = rate
        self.recent_orbits = [True] * recent
        self.mean_return = mean
        self.stalled = stalled


class _FakePipeline:
    def __init__(self, exam=None, attempts=0, rollbacks=0, finished=False):
        self.exam = exam
        self.attempts = attempts
        self.rollbacks = rollbacks
        self.finished = finished


class _FakeStats:
    """Накопительные счётчики мозга — по ним дежурный судит о живости."""

    def __init__(self, total_steps=0, episodes=0):
        self.total_steps = total_steps
        self.episodes = episodes


class _FakeTrainer:
    class brain:                    # noqa: N801 — двойник, а не класс проекта
        stage = 1

        class pilot:
            stats = _FakeStats()

    def __init__(self, report=None, alive=None):
        self.last_report = report
        if alive is not None:
            self.brain.pilot.stats = alive


def test_watchdog() -> None:
    """[26] Дежурный: зелёное, жёлтое, красное."""
    print("\n[26] Дежурный по обучению")
    from kia_core.neural import is_available
    if not is_available():
        check("PyTorch установлен", False)
        return
    from kia_core.neural.trainer import TrainingReport
    from kia_core.neural.watchdog import DEFAULT_PERIOD, Watchdog

    def report(**kwargs):
        item = TrainingReport(mode=kwargs.pop("mode", "Базовый Тренажёр"))
        for key, value in kwargs.items():
            setattr(item, key, value)
        return item

    def verdict(first, second=None, pipeline=None, between=None,
                alive=(1_000, 100), alive_then=(2_000_000, 5_000)):
        """Две сводки подряд. alive — накопительные счётчики мозга."""
        stats = _FakeStats(*alive)
        dog = Watchdog(_FakeTrainer(first, alive=stats), pipeline=pipeline,
                       period=300.0)
        dog.assess()                       # первая сводка — точка отсчёта
        if second is not None:
            dog.trainer.last_report = second
        if alive_then is not None:
            stats.total_steps, stats.episodes = alive_then
        if between is not None:
            between(pipeline)
        return dog.assess()

    check("сводка раз в пять минут по умолчанию", DEFAULT_PERIOD == 300.0,
          f"{DEFAULT_PERIOD / 60:.0f} мин, KIA_WATCHDOG=0 выключает")

    # --- разброс действий не раздувается обратно ---
    # Живой случай 06.08: веса вернулись на ступень 1 с σ 0.165, потолок
    # первой ступени 0.368 поднял его обратно, и доля орбит в обучении
    # упала 14 % -> 0 % при 28 % без шума. Обучались не полёты, а шум.
    import math as _math

    import torch

    from kia_core.neural.ppo import PPOAgent, PPOConfig
    from kia_core.neural.trainer import log_std_ceiling

    from kia_core.neural.ppo import STAGE_LOG_STD_MAX
    fresh = PPOAgent(6, 2, (2,), PPOConfig(), hidden=32, name="свежий")
    check("свежая сеть получает разведку своей ступени целиком",
          abs(log_std_ceiling(fresh, 1) - STAGE_LOG_STD_MAX[1]) < 1e-9,
          f"sigma {_math.exp(log_std_ceiling(fresh, 1)):.3f} на первой ступени")

    seasoned = PPOAgent(6, 2, (2,), PPOConfig(), hidden=32, name="обученный")
    seasoned.stats.updates = 5929
    with torch.no_grad():
        seasoned.net.log_std.fill_(-1.8)          # σ 0.165, как после ступени 3
    kept = log_std_ceiling(seasoned, 1)
    check("сошедшаяся политика не раздувается на первой ступени",
          abs(kept - (-1.8)) < 1e-5,          # веса в float32, точность своя
          f"sigma осталась {_math.exp(kept):.3f} вместо "
          f"{_math.exp(-1.0):.3f}")
    check("сузиться по-прежнему можно",
          abs(log_std_ceiling(seasoned, 4) - (-2.0)) < 1e-5,
          f"на четвёртой ступени sigma {_math.exp(log_std_ceiling(seasoned, 4)):.3f}")

    # --- красное: сломалось ---
    level, _ = verdict(report(error="нет весов предыдущей ступени"))
    check("ступень не запустилась — красный", level == "red")
    frozen = report(steps=1_204_480, episodes=900, mean_return=5.0)
    level, text = verdict(frozen, frozen, alive=(500_000, 900),
                          alive_then=(500_000, 900))
    check("счёт замер — красный", level == "red", text.split("·")[-1].strip()[:52])
    level, _ = verdict(report(steps=100_000, episodes=42),
                       report(steps=400_000, episodes=42),
                       alive=(100_000, 42), alive_then=(400_000, 42))
    check("решения идут, а эпизоды не заканчиваются — красный", level == "red")
    level, _ = verdict(report(steps=1000, episodes=10, mean_return=float("nan")),
                       report(steps=2000, episodes=20, mean_return=float("nan")))
    check("награда стала не числом — красный", level == "red")

    # Ровно тот ложный сигнал, что случился 06.08 в 12:52. Счётчик шагов
    # в отчёте обнуляется при каждом новом заходе на ступень (было
    # 1 022 241, стало 204 476), и первая версия дежурного объявила
    # «счёт замер», хотя счёт шёл 14 тыс. шаг/с. Живость меряется по
    # накопительным счётчикам мозга — они не обнуляются никогда.
    level, text = verdict(report(steps=1_022_241, episodes=592, mean_return=1.0),
                          report(steps=204_476, episodes=112, mean_return=3.5),
                          pipeline=_FakePipeline(
                              _FakeExam(episodes=200, rate=0.2, recent=30)),
                          alive=(3_000_000, 5_000),
                          alive_then=(3_400_000, 5_200))
    check("новый заход на ступень не считается остановкой", level == "green",
          "счётчик отчёта обнулился, но мозг продолжает считать")

    # --- жёлтое: тупит. Все поводы взяты из настоящих разборов ---
    dead = _FakePipeline(_FakeExam(episodes=900, rate=0.0, recent=0))
    level, text = verdict(report(steps=1000, episodes=100),
                          report(steps=900_000, episodes=900, mean_return=-3.0),
                          pipeline=dead)
    check("орбит нет за сотни эпизодов — жёлтый", level == "yellow",
          text.split("·")[2].strip()[:46] if level == "yellow" else text[:46])
    rail = _FakePipeline(_FakeExam(episodes=200, rate=0.2, recent=40))
    level, text = verdict(report(steps=1000, episodes=100),
                          report(steps=900_000, episodes=300, mean_return=4.0,
                                 axis_means=[0.2, 0.3, -0.91, 0.0]),
                          pipeline=rail)
    check("ось на упоре — жёлтый", level == "yellow",
          "именно этот прибор поймал мёртвое рыскание (5.3i)")
    stuck = _FakePipeline(_FakeExam(episodes=600, rate=0.2, recent=40,
                                    stalled=400))
    level, _ = verdict(report(steps=1000, episodes=100),
                       report(steps=900_000, episodes=600, mean_return=2.0),
                       pipeline=stuck)
    check("награда не растёт сотни эпизодов — жёлтый", level == "yellow")
    tries = _FakePipeline(_FakeExam(episodes=300, rate=0.2, recent=30),
                          attempts=3)
    level, _ = verdict(report(steps=1000, episodes=100),
                       report(steps=900_000, episodes=300, mean_return=6.0),
                       pipeline=tries)
    check("ступень не сдаётся много заходов — жёлтый", level == "yellow")
    rolled = _FakePipeline(_FakeExam(episodes=300, rate=0.2, recent=30))
    level, _ = verdict(report(steps=1000, episodes=100),
                       report(steps=900_000, episodes=300, mean_return=6.0),
                       pipeline=rolled,
                       between=lambda p: setattr(p, "rollbacks", p.rollbacks + 1))
    check("откат между сводками — жёлтый", level == "yellow")

    # --- зелёное ---
    good = _FakePipeline(_FakeExam(episodes=612, orbits=34, rate=0.18,
                                   recent=34, mean=8.4))
    level, text = verdict(report(steps=1000, episodes=100),
                          report(steps=4_000_000, episodes=612, mean_return=8.4,
                                 axis_means=[0.13, 0.41, -0.19, 0.04]),
                          pipeline=good)
    check("нормальный ход — зелёный", level == "green", text[:64])
    check("в зелёной строке есть числа, а не одно слово",
          "орбит" in text and "эпизодов" in text and "шаг/с" in text)

    # --- устойчивость: дежурный не имеет права ронять обучение ---
    dog = Watchdog(_FakeTrainer(None), period=300.0)
    level, _ = dog.assess()
    check("без отчёта тренера не падает", level in ("green", "yellow", "red"))
    source = (Path(__file__).resolve().parent.parent / "kia_core" / "neural"
              / "watchdog.py").read_text(encoding="utf-8")
    # У дежурного есть свой флаг остановки — им он гасит СЕБЯ. Запрещено
    # трогать чужое: тренера, веса, оптимизатор.
    forbidden = ("brain.save", "train_stage", "trainer.stop", "optimizer",
                 "trainer.brain.save", "run_vector_training")
    touched = [word for word in forbidden if word in source]
    check("дежурный ничего не меняет в обучении", not touched,
          ", ".join(touched) if touched
          else "только читает отчёт и печатает строку")


def test_brain3d() -> None:
    """[25] Трёхмерная карта мозга."""
    print("\n[25] Трёхмерная карта мозга")
    from kia_core.neural import is_available
    if not is_available():
        check("PyTorch установлен", False)
        return
    from kia_core.neural import brain3d
    from kia_core.neural.ppo import build_brain

    root = Path(__file__).resolve().parent.parent

    # Главное обещание карты: у чистого мозга снаружи пусто. Если оно
    # сломается, карта начнёт показывать знания там, где их нет.
    fresh = build_brain(load=False)
    empty = brain3d.build(fresh, knowledge=None)
    check("у чистого мозга нет ни навыков, ни проснувшихся нейронов",
          empty["counts"]["skills"] == 0 and empty["counts"]["awake"] == 0,
          f"навыков {empty['counts']['skills']}, проснулось "
          f"{empty['counts']['awake']} из {empty['counts']['hidden']}")
    check("но органы чувств и руки на месте всегда",
          empty["counts"]["inputs"] == 40 and empty["counts"]["outputs"] == 5,
          "40 входов и 5 выходов — это не знания, это устройство")

    # Порог пробуждения меряется по сети-двойнику, а не подбирается руками
    from kia_core.neural.brain3d import _birth_influence, _influence
    birth = _birth_influence(fresh.pilot.net)
    live = [float(v.float().median().item()) for v in _influence(fresh.pilot.net)]
    close = all(abs(a - b) < max(a, b) * 0.6 for a, b in zip(birth, live))
    check("уровень «нейрон ничего не решает» берётся с чистой сети", close,
          f"двойник {birth[0]:.4f}, сам {live[0]:.4f} — один порядок")

    trained = build_brain(load=True)
    if trained.loaded_from_disk:
        data = brain3d.build(trained, knowledge=None)
        awake = data["counts"]["awake"]
        check("у обученного мозга нейроны просыпаются", awake > 0,
              f"проснулось {awake} из {data['counts']['hidden']}")
        ids = [node["id"] for node in data["nodes"]]
        check("узлы не дублируются", len(ids) == len(set(ids)),
              f"{len(ids)} узлов")
        known = set(ids)
        broken = [edge for edge in data["edges"]
                  if edge["a"] not in known or edge["b"] not in known]
        check("все связи ведут в существующие узлы", not broken,
              f"{len(data['edges'])} связей")
        stable = brain3d.build(trained, knowledge=None)
        same = all(a["pos"] == b["pos"] for a, b in
                   zip(data["nodes"], stable["nodes"]))
        check("раскладка устойчива между снимками", same,
              "иначе картинка дрожала бы каждые две секунды")

    # Ярлык: кириллица и LF в .bat ломают cmd.exe (MIGRATION, раздел 8)
    launcher = (root / "BRAIN3D.bat").read_bytes()
    check("BRAIN3D.bat существует и только в ASCII",
          launcher and not any(byte > 127 for byte in launcher))
    check("BRAIN3D.bat с переводами строк CRLF",
          launcher.count(b"\r\n") > 10
          and launcher.count(b"\n") == launcher.count(b"\r\n"),
          f"{launcher.count(b'\r\n')} строк")
    check("ярлык запускает трёхмерную карту",
          b"kia_brain3d.py" in launcher and b"kia_brain_map.py" not in launcher)
    check("страница карты на месте", (root / "kia_brain3d.html").exists())

    # Карта не имеет права трогать обучение: она читает веса и только
    page = (root / "kia_brain3d.py").read_text(encoding="utf-8")
    check("карта не вмешивается в обучение",
          "train_stage" not in page and "run_vector_training" not in page,
          "только чтение весов и копилки знаний")


def sandbox_working_files() -> Path:
    """Все рабочие файлы KIA — во временную папку на время самопроверки.

    САМОПРОВЕРКА ПЕРЕЗАПИСАЛА НАСТОЯЩИЕ ДАННЫЕ. 21.09.2026 тест
    автономного цикла прятал в песочницу историю, память и папку kia/,
    но не состояние цикла, разборы, отчёт, метаданные прогонов и лучший
    геном: лучший профиль пилота (счёт 9789) заменился тестовым (192),
    autonomy_state.json, diagnoses.json и data/runs/1–3 — тоже.

    Теперь до первого теста каждый путь, куда код KIA умеет писать,
    переводится во временную папку. Существующие файлы копируются туда
    же: тесты читают то же, что и раньше, но пишут только в копию.
    """
    import shutil
    from kia_core import config as cfg_mod
    from kia_core import flight_log, memory, mission_spec, run_workspace
    from kia_core.agent import autonomy, dispatcher
    from kia_core.engineer import design_report, parts_catalog, reference_designs
    from kia_core.learning import diagnosis, reward_book

    box = Path(tempfile.mkdtemp(prefix="kia_selftest_"))

    def moved(path) -> Path:
        path = Path(path)
        target = box / path.name
        if path.exists() and not target.exists():
            shutil.copy2(path, target)
        return target

    autonomy.STATE_FILE = moved(autonomy.STATE_FILE)
    diagnosis.AUTONOMY_REPORT = autonomy.AUTONOMY_REPORT = moved(diagnosis.AUTONOMY_REPORT)
    diagnosis.DIAGNOSES_FILE = moved(diagnosis.DIAGNOSES_FILE)
    dispatcher.OPTIMIZER_STATE = moved(dispatcher.OPTIMIZER_STATE)
    design_report.REPORT_MD = moved(design_report.REPORT_MD)
    design_report.REPORT_JSON = moved(design_report.REPORT_JSON)
    flight_log.LOG_MD = moved(flight_log.LOG_MD)
    flight_log.LOG_JSON = moved(flight_log.LOG_JSON)
    mission_spec.SPEC_FILE = moved(mission_spec.SPEC_FILE)
    reference_designs.NORMS_FILE = moved(reference_designs.NORMS_FILE)
    parts_catalog.CACHE_FILE = moved(parts_catalog.CACHE_FILE)
    reward_book.BOOK_FILE = moved(reward_book.BOOK_FILE)
    reward_book.REWARD_BOOK.path = moved(reward_book.REWARD_BOOK.path)
    memory.MEMORY_FILE = moved(memory.MEMORY_FILE)
    memory.GLOBAL_MEMORY.path = moved(memory.GLOBAL_MEMORY.path)
    cfg_mod.CONFIG.learning.best_genome_file = str(moved(cfg_mod.CONFIG.learning.best_genome_file))
    cfg_mod.CONFIG.learning.history_file = str(box / "history.json")
    run_workspace.META_DIR = box / "runs"
    run_workspace.WORKSPACE = run_workspace.RunWorkspace(box / "kia")
    return box


def main() -> int:
    print("=" * 68)
    print("  KIA — самопроверка (без запуска игры)")
    print("=" * 68)
    box = sandbox_working_files()
    print(f"  рабочие файлы подменены копиями в {box}")
    for test in (test_math, test_catalog, test_mission_spec, test_budget_and_design,
                 test_craft, test_rewards, test_telemetry_rewards, test_flight_log,
                 test_learning_loop, test_genome_bounds, test_lambert,
                 test_diagnosis, test_autonomy_loop, test_uncrewed_design,
                 test_run_workspace, test_global_memory, test_neural_core,
                 test_neural_ppo, test_neural_checkpoint,
                 test_neural_environments, test_curriculum, test_pipeline,
                 test_watchdog, test_brain3d, test_designer_pro,
                 test_radial_geometry, test_math_trainer, test_builder_stages,
                 test_autostage_one_at_a_time, test_periapsis_shaping,
                 test_reference_designs,
                 test_ascent_failures,
                 test_brain3d_programs,
                 test_interfaces):
        test()
    print("\n" + "=" * 68)
    print(f"  Пройдено: {PASSED} | Провалено: {FAILED}")
    print("=" * 68)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
