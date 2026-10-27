# ASTRAUTOMA Training

The training side of [ASTRAUTOMA](https://github.com/himsson/ASTRAUTOMA). This is what produces
the weights the app loads — simulators, the PPO loop, the exams and the benchmark stand.

You do not need this to fly. Download the weights from the main repository and you are done. You
need this if you want to train your own, see how the numbers in the benchmark table were measured,
or read how it works.

The game is not required. Four of the five networks train entirely in simulators; only the final
fine-tuning runs against a live KSP.

## Install

```
pip install -r requirements.txt
```

Python 3.10+ and PyTorch. A GPU is not used: these networks are small enough that the CPU is
faster — measured, see the panel's CPU+GPU screen.

## The panel

Run `LAUNCH_KIA.bat`, or `python kia_control.py`.

```
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
```

Item **7** opens the networks: the five pilot stages, the designer and mathematician ladders, the
navigator, the exams, weight rollback. Item **0** runs the whole programme unattended, without the
game.

The interface is in Russian.

## What gets trained

| Network | Where | What it decides |
|---|---|---|
| Pilot | `sim_env*.py` | throttle, pitch, yaw, roll, staging |
| Designer | `builder_env.py` | which parts go on the rocket |
| Mathematician | `math_env.py` | manoeuvre arithmetic |
| Navigator | `nav_env.py` | route, window, gravity assist |

The pilot climbs five stages, each starting from the previous one's weights:

1. **Тренажёр** — the world is real, the controls are not: commands land instantly and exactly.
2. **Тренажёр+** — what sits between the decision and the rocket: 100 ms of kRPC lag, staging
   lockout, wind, thrust spool-up, an electric budget, a penalty for flying sideways.
3. **Тренажёр++** — 3D ballistics, a rotating planet and atmosphere, KSP's own pressure and
   temperature tables, Mach drag with the transonic hump, a centre of mass that moves as the
   tanks drain.
4. **Тренажёр+++** — the controls command torque, not angle, exactly as the game does with SAS
   off.
5. **Аварии** — the same world, but the hardware breaks mid-flight: engines die or lose thrust,
   a stage refuses to separate, the alternator dies, a tank leaks, a servo jams off-centre. Half
   the flights carry crew, and crew is worth more than any amount of saved hardware.

A stage is passed on its exam — mean reward over a window and a count of clean orbits. Fail it
badly enough and the pipeline rolls back to the previous stage's snapshot and continues at half
the learning rate. The chronicle is in `kia_evolution_manifest.txt`.

## Training a long run

```
python kia_neural.py evolve       # the whole programme, unattended, no game
python kia_neural.py stage 5      # one stage
python kia_navigator.py           # the navigator, 10 CPU processes
```

Stop with Ctrl+C: weights are written every 30 seconds and the stage picks up where it stopped.

## Benchmarks

```
python tools/benchmark.py               # every version in the weights library
python tools/benchmark.py --quick       # small samples, for checking the stand itself
python tools/benchmark.py --table       # re-render the table from data/benchmark.json
```

The same exam for every version of the weights, with fixed seeds and the policy's noise switched
off, so the numbers reproduce. The designer is measured by `tools/bench_designer.py` in the main
repository — it runs the real designer, not a copy.

## Exporting to the app

```
python tools/export_weights.py --label "my weights"
```

Splits the trained files into the module folders the app imports (**Weights / Knowledge → I**).

## License

[Source-available](LICENSE), not open source. Read it, run it, train with it, fork it to send a
pull request. Redistributing it, publishing a fork as its own project, or selling it is not
allowed.
