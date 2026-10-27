"""Оптимизатор: гибрид генетического алгоритма и локального поиска.

Выбран ГА, потому что пространство поиска смешанное: дискретные гены
конструкции (типы двигателей, число баков) + непрерывные гены профиля
полёта. Градиент по такому пространству недоступен, а оценка одной особи
дорогая (один реальный запуск в KSP) — ГА с малой популяцией и элитизмом
здесь эффективнее.

Схема поколения:
    элита (без изменений) -> потомки (турнирный отбор + BLX-кроссовер +
    гауссова мутация) -> случайные особи для разнообразия.
Дополнительно: если поколение не улучшило рекорд, радиус мутации растёт
(адаптивный отжиг), если улучшило — сужается вокруг лидера.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..config import CONFIG
from ..engineer.autodesign import DesignPolicy
from ..logging_setup import get_logger
from .genome import FlightGenome, Individual, jitter_individual, random_individual

log = get_logger("learning.optimizer")


@dataclass
class GenerationReport:
    generation: int
    best_score: float
    mean_score: float
    best_individual: Individual
    improved: bool
    mutation_scale: float

    def describe(self) -> str:
        arrow = "↑" if self.improved else "·"
        return (f"Поколение {self.generation} {arrow} лучший={self.best_score:.1f} "
                f"средний={self.mean_score:.1f} σ={self.mutation_scale:.3f}")


class GeneticOptimizer:
    """Эволюционный поиск по геному (конструкция + профиль полёта)."""

    def __init__(self, cfg=None, seed: int | None = None):
        self.cfg = cfg or CONFIG.learning
        self.rng = random.Random(seed if seed is not None else self.cfg.random_seed)
        self.generation = 0
        self.population: list[Individual] = []
        self.evaluated: list[Individual] = []
        self.best: Individual | None = None
        self.mutation_scale = self.cfg.mutation_scale
        self.history: list[GenerationReport] = []

    # ------------------------------------------------------------------
    def seed_population(self, seeds: list[Individual] | None = None) -> list[Individual]:
        """Начальная популяция: затравки, их вариации и одна случайная особь.

        Раньше все места сверх затравок заполнялись равномерно случайными
        особями по всему диапазону. В тренажёре это разумно — там особь
        стоит доли секунды. В живой игре она стоит МИНУТ реального полёта,
        и цена ошибки совсем другая.

        Замер, из-за которого это переписано. Показатель кривой тангажа
        разыгрывался равномерно в (0.25, 1.30). При 1.3 профиль выведения
        держит нос почти вертикально до десяти километров:

            высота    1 км   3 км   5 км   10 км
            n = 1.30   90°    89°    87°    81°
            n = 0.55   85°    74°    68°    56°

        Такая ракета не набирает горизонтальной скорости, зависает на
        семи километрах и падает. Полёт гарантированно провальный, и
        узнаём мы об этом через полторы минуты живого времени. Три подряд
        полёта закончились так, счёт −197, −204, −204.

        Значения по умолчанию в геноме — не случайный набор, а рабочий
        профиль выведения. Начинать поиск, выбросив это знание, значит
        тратить первые десятки живых полётов на то, чтобы заново узнать:
        ракета не должна лететь вертикально вверх.

        Поэтому популяция теперь строится ВОКРУГ затравок, и лишь одно
        место остаётся под полностью случайную особь — чтобы поиск не
        заперся в окрестности умолчаний.
        """
        self.population = []
        seeds = seeds or []
        for i, ind in enumerate(seeds[: self.cfg.population_size]):
            clone = ind.copy()
            clone.generation, clone.index, clone.score = 0, i, None
            self.population.append(clone)

        # Одна случайная особь на популяцию — разведка дальних углов
        # пространства. При крошечной популяции даже её не берём.
        wild = 1 if self.cfg.population_size >= 4 and self.population else 0
        base_pool = list(self.population) or [Individual()]
        while len(self.population) < self.cfg.population_size - wild:
            base = base_pool[len(self.population) % len(base_pool)]
            self.population.append(
                jitter_individual(base, self.rng,
                                  scale=self.cfg.mutation_scale * 2.0,
                                  generation=0, index=len(self.population)))
        while len(self.population) < self.cfg.population_size:
            self.population.append(
                random_individual(self.rng, 0, len(self.population)))
        log.info("Начальная популяция: %d особей (%d затравок, %d вариаций, "
                 "%d случайных)", len(self.population), len(seeds),
                 max(0, len(self.population) - len(seeds) - wild), wild)
        return self.population

    # ------------------------------------------------------------------
    def record_result(self, individual: Individual, score: float) -> None:
        individual.score = score
        self.evaluated.append(individual)
        if self.best is None or score > (self.best.score or float("-inf")):
            self.best = individual.copy()
            log.info("Новый рекорд: %.1f — %s", score, individual.describe())

    # ------------------------------------------------------------------
    # Операторы
    # ------------------------------------------------------------------
    def tournament(self, pool: list[Individual], k: int = 3) -> Individual:
        contenders = self.rng.sample(pool, min(k, len(pool)))
        return max(contenders, key=lambda i: i.score if i.score is not None else float("-inf"))

    def crossover(self, a: Individual, b: Individual, generation: int,
                  index: int) -> Individual:
        """BLX-α по обеим секциям генома (политика конструктора + профиль полёта)."""
        child = Individual(design=DesignPolicy(), flight=FlightGenome(),
                           generation=generation, index=index,
                           parents=(a.index, b.index))
        alpha = 0.25
        for section_name, bounds in (("design", DesignPolicy.bounds()),
                                     ("flight", FlightGenome.bounds())):
            target = getattr(child, section_name)
            sa, sb = getattr(a, section_name), getattr(b, section_name)
            for name, (lo, hi) in bounds.items():
                va, vb = getattr(sa, name), getattr(sb, name)
                low, high = min(va, vb), max(va, vb)
                span = (high - low) * alpha
                value = self.rng.uniform(low - span, high + span)
                setattr(target, name, max(lo, min(hi, value)))
        return child

    def mutate(self, ind: Individual) -> Individual:
        for section_name, bounds in (("design", DesignPolicy.bounds()),
                                     ("flight", FlightGenome.bounds())):
            section = getattr(ind, section_name)
            for name, (lo, hi) in bounds.items():
                if self.rng.random() < self.cfg.mutation_rate:
                    value = getattr(section, name)
                    value += self.rng.gauss(0.0, (hi - lo) * self.mutation_scale)
                    setattr(section, name, max(lo, min(hi, value)))
        return ind

    # ------------------------------------------------------------------
    def evolve(self) -> list[Individual]:
        """Формирует следующее поколение из оценённых особей."""
        scored = [i for i in self.population if i.score is not None]
        if not scored:
            # Заново — но снова ВОКРУГ известного хорошего, а не наугад.
            # Полный сброс в случайность выбрасывал рабочий профиль
            # выведения ровно так же, как это делала начальная популяция.
            log.warning("Нет оценённых особей — популяция пересобирается "
                        "вокруг лучшего известного генома")
            self.generation += 1
            base = self.best.copy() if self.best is not None else Individual()
            base.score = None
            self.population = [base] + [
                jitter_individual(base, self.rng,
                                  scale=self.cfg.mutation_scale * 2.0,
                                  generation=self.generation, index=i)
                for i in range(1, self.cfg.population_size)]
            return self.population

        scored.sort(key=lambda i: i.score, reverse=True)
        best_score = scored[0].score
        mean_score = sum(i.score for i in scored) / len(scored)
        prev_best = self.history[-1].best_score if self.history else float("-inf")
        improved = best_score > prev_best

        # Адаптивный радиус мутации
        if improved:
            self.mutation_scale = max(0.03, self.mutation_scale * 0.85)
        else:
            self.mutation_scale = min(0.45, self.mutation_scale * 1.25)

        self.history.append(GenerationReport(
            generation=self.generation, best_score=best_score, mean_score=mean_score,
            best_individual=scored[0].copy(), improved=improved,
            mutation_scale=self.mutation_scale))
        log.info(self.history[-1].describe())

        self.generation += 1
        next_pop: list[Individual] = []

        # 1) Элита переносится без изменений
        for i, elite in enumerate(scored[: self.cfg.elite_count]):
            clone = elite.copy()
            clone.generation, clone.index, clone.score = self.generation, i, None
            next_pop.append(clone)

        # 2) Потомки
        n_random = 1 if self.cfg.population_size >= 6 else 0
        target = self.cfg.population_size - n_random
        while len(next_pop) < target:
            idx = len(next_pop)
            if len(scored) >= 2 and self.rng.random() < self.cfg.crossover_rate:
                parent_a = self.tournament(scored)
                parent_b = self.tournament(scored)
                child = self.crossover(parent_a, parent_b, self.generation, idx)
            else:
                parent = self.tournament(scored)
                child = jitter_individual(parent, self.rng, self.mutation_scale,
                                          self.generation, idx)
            next_pop.append(self.mutate(child))

        # 3) Немного свежей крови против преждевременной сходимости
        while len(next_pop) < self.cfg.population_size:
            next_pop.append(random_individual(self.rng, self.generation, len(next_pop)))

        self.population = next_pop
        return self.population

    # ------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "generation": self.generation,
            "mutation_scale": self.mutation_scale,
            "best": self.best.to_dict() if self.best else None,
            "population": [i.to_dict() for i in self.population],
            "history": [{"generation": h.generation, "best": h.best_score,
                         "mean": h.mean_score, "improved": h.improved}
                        for h in self.history],
        }

    def save_state(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        log.debug("Состояние оптимизатора сохранено: %s", p)
        return p

    def load_state(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
        self.generation = int(data.get("generation", 0))
        self.mutation_scale = float(data.get("mutation_scale", self.cfg.mutation_scale))
        self.population = [Individual.from_dict(d) for d in data.get("population", [])]
        if data.get("best"):
            self.best = Individual.from_dict(data["best"])
        log.info("Состояние оптимизатора восстановлено: поколение %d", self.generation)
        return True
