# ga_optimizer.py

import numpy as np

# 1. Population
def init_population(pop_size, n_features):
    return np.random.randint(0, 2, (pop_size, n_features))

# 2. Selection
def tournament_selection(population, fitness_scores, k=3):
    idx = np.random.choice(len(population), k)
    return population[idx[np.argmax(fitness_scores[idx])]]

# 3. Crossover
def crossover(p1, p2, p=0.8):
    if np.random.rand() > p:
        return p1.copy(), p2.copy()

    point = np.random.randint(1, len(p1)-1)
    return (
        np.concatenate([p1[:point], p2[point:]]),
        np.concatenate([p2[:point], p1[point:]])
    )

# 4. Mutation
def mutation(chromosome, p=0.01):
    for i in range(len(chromosome)):
        if np.random.rand() < p:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# 5. GA Engine
def run_ga(X, y, fitness_fn,
           pop_size=30, generations=40):
    n_features = X.shape[1]
    population = init_population(pop_size, n_features)

    for _ in range(generations):
        fitness_scores = np.array([
            fitness_fn(ch, X, y) for ch in population
        ])

        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitness_scores)
            p2 = tournament_selection(population, fitness_scores)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutation(c1))
            new_pop.append(mutation(c2))

        population = np.array(new_pop[:pop_size])

    best_idx = np.argmax(fitness_scores)
    return population[best_idx]
