import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


class GeneticAlgorithm:
    def __init__(
        self,
        population_size=20,
        generations=15,
        mutation_rate=0.1,
        random_state=42
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _initialize_population(self, n_features):
        return np.random.randint(
            0, 2, size=(self.population_size, n_features)
        )

    def _fitness(self, X, y, chromosome):
        if chromosome.sum() == 0:
            return 0

        X_selected = X[:, chromosome == 1]

        _, counts = np.unique(y, return_counts=True)
        min_class = counts.min()
        cv_splits = min(5, min_class)

        if cv_splits < 2:
            return 0

        skf = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=self.random_state
        )

        model = LogisticRegression(max_iter=1000)

        scores = cross_val_score(
            model,
            X_selected,
            y,
            cv=skf,
            scoring="accuracy"
        )

        return scores.mean()

    def _select(self, population, fitness_scores):
        probs = fitness_scores / fitness_scores.sum()
        idx = np.random.choice(
            range(len(population)),
            size=self.population_size,
            p=probs
        )
        return population[idx]

    def _crossover(self, parent1, parent2):
        point = np.random.randint(1, len(parent1))
        child = np.concatenate([
            parent1[:point],
            parent2[point:]
        ])
        return child

    def _mutate(self, chromosome):
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def fit(self, X, y):
        n_features = X.shape[1]
        population = self._initialize_population(n_features)

        best_chromosome = None
        best_score = -1

        for _ in range(self.generations):
            fitness_scores = np.array([
                self._fitness(X, y, chrom)
                for chrom in population
            ])

            if fitness_scores.max() > best_score:
                best_score = fitness_scores.max()
                best_chromosome = population[
                    fitness_scores.argmax()
                ]

            selected = self._select(population, fitness_scores)

            children = []
            for i in range(0, self.population_size, 2):
                p1, p2 = selected[i], selected[i + 1]
                c1 = self._mutate(self._crossover(p1, p2))
                c2 = self._mutate(self._crossover(p2, p1))
                children.extend([c1, c2])

            population = np.array(children)

        selected_features = np.where(best_chromosome == 1)[0]
        return selected_features, best_score
