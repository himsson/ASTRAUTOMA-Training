# ASTRAUTOMA Training

*[Русская версия](README.ru.md)*

This is where [ASTRAUTOMA](https://github.com/himsson/ASTRAUTOMA) learns. If you just want to
fly, you do not need any of it — download the trained weights from the main repository and you
are done. This folder is for training your own, checking the numbers in the benchmark table, or
reading how the thing works inside.

The game is not required. Four of the five networks never see KSP at all: they live in
simulators, and the simulators are written to lie as little as possible. A live game is needed
only at the very end, for the final fine-tuning.

## Getting started

```
pip install -r requirements.txt
py -3 train_all.py
```

That is it. You can go to bed: the mathematician, the designer, the navigator and the pilot train
one after another, weights are written to disk as it goes, and the journal lands in
`data/train_all.json`. Ctrl+C stops it; the next run picks up where you left off instead of
starting over.

If you want less than everything:

```
py -3 train_all.py --status           what is already done
py -3 train_all.py --only pilot       the pilot alone
py -3 train_all.py --from navigator   the navigator and onwards
```

The GPU is not used, and that is not an oversight: these networks are small, and the CPU is
faster on them. The measurement is in the panel, on the CPU+GPU screen.

## The panel

If you would rather drive by hand than press one button, there is a panel: `LAUNCH_KIA.bat` or
`python kia_control.py`. Item **7** holds every network separately — stages, exams, weight
rollback. Item **0** is what `train_all.py` does, but for the pilot only. The interface is in
Russian.

## Who learns what

Four networks, each with its own job.

**The mathematician** does manoeuvre arithmetic — Tsiolkovsky, vis-viva, Hohmann, phase angles.
It learns quickly and, honestly, is not used by the app: these problems have exact answers and
the program computes them with formulas. The network is more of a check that the simulator's
world is not lying.

**The designer** picks the parts that go on the rocket. It has the hardest job: the reward only
arrives at the end of the build, and "put a tank on with no engine" produces the same gradient as
"forgot the parachute".

**The navigator** decides where and when to fly: the route, the window, whether a gravity assist
is worth catching. It starts by copying an exact planner and then improves on its own. This one
really does run in the app.

**The pilot** flies. Five stages, each starting from the previous one's weights:

1. **Тренажёр.** The world is already real — atmosphere, thrust, mass — but the controls are
   perfect: commands land instantly and exactly.
2. **Тренажёр+.** Everything between the decision and the rocket appears: 100 ms of kRPC lag, a
   two-second staging lockout, wind, engine spool-up, an electric budget for the controls and a
   penalty for flying sideways. This is usually where a policy trained in a perfect world falls
   apart.
3. **Тренажёр++.** The physics stops being a toy: 3D ballistics, a rotating planet and
   atmosphere, pressure and temperature from the game's own altitude tables, drag by Mach number
   with the transonic hump, and a centre of mass that moves as the tanks drain.
4. **Тренажёр+++.** The controls command torque, not angle — exactly as KSP does with SAS off.
   The angle becomes a double integral of the command, and errors start to accumulate.
5. **Аварии.** The same world, but the hardware breaks mid-flight: engines die or lose thrust, a
   stage refuses to separate, the alternator dies, a tank leaks, a servo jams. Half the flights
   carry crew, and crew is worth more than any amount of saved hardware — +60 for a live crew
   against −120 for a lost one, while an uncrewed craft is counted in tonnes.

A stage is passed on its exam: mean reward over a window and a count of clean orbits. If the
policy goes bad, the pipeline rolls back to the previous stage's snapshot by itself and continues
at half the learning rate. What was promoted and what was rolled back is in
`kia_evolution_manifest.txt`.

## Measuring

```
py -3 tools/benchmark.py            every version in the weights library
py -3 tools/benchmark.py --quick    small samples, to check the stand itself
py -3 tools/benchmark.py --table    re-render the table from the json
```

The same exam for every version: fixed seeds, the policy's noise switched off, the normalizer
left alone. The numbers reproduce and can be compared between versions — which was the whole
point. The designer is measured by a script in the main repository, because that is where the
real designer lives and a copy would lie.

## Handing the weights to the app

```
py -3 tools/export_weights.py --label "2.0"
```

Splits what was trained into the module folders the app imports through **Weights / Knowledge →
I**.

## License

[The source is open](LICENSE) to read, but this is not open source. Read it, run it, train your
own with it, fork it to send a pull request — please do. Redistributing it, publishing a fork as
a project of its own, or selling it — no.
