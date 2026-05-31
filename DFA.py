import time

import numpy as np


# Classify civilizations
def classify_civilizations(pop, fitness):
    sorted_idx = np.argsort(fitness)
    highest = [sorted_idx[0]]
    n = len(pop)
    advanced = sorted_idx[1:int(0.3 * n)]
    normal = sorted_idx[int(0.3 * n):int(0.8 * n)]
    low = sorted_idx[int(0.8 * n):]
    return highest, advanced, normal, low


# Update highest civilization — Equation (1) and (2)
def update_highest(pop, fobj, fitness, t, max_iter):
    i = 0
    step = (1 / max_iter) * (2 * t / max_iter)  # Eq. (1)
    if np.random.rand() < 0.5:
        U = np.random.uniform(-1, 1, pop.shape[1])
        new_pos = pop[i] + np.random.rand() * step * U  # Eq. (2)
    else:
        j = np.random.randint(1, len(pop))
        new_pos = pop[i] + np.random.rand() * step * (pop[j] - pop[i])  # Eq. (2)
    new_fit = fobj(new_pos)
    if new_fit < fitness[i]:
        pop[i] = new_pos
        fitness[i] = new_fit
    return pop, fitness


# Update advanced civilizations — Equation (3)
def update_advanced(pop, fobj, fitness, highest_idx):
    b = 1
    for i in range(1, int(0.3 * len(pop))):
        if np.random.rand() < 0.2:
            j = np.random.randint(int(0.3 * len(pop)), int(0.8 * len(pop)))
        else:
            j = highest_idx[0]
        l = np.random.uniform(-1, 1)
        D = np.abs(pop[j] - pop[i])
        new_pos = pop[i] + D * np.exp(b * l) * np.cos(2 * np.pi * l)  # Eq. (3)
        new_fit = fobj(new_pos)
        if new_fit < fitness[i]:
            pop[i] = new_pos
            fitness[i] = new_fit
    return pop, fitness


# Update normal civilizations — Equation (4)
def update_normal(pop, fobj, fitness):
    b = 1
    for i in range(int(0.3 * len(pop)), int(0.8 * len(pop))):
        j = np.random.randint(0, int(0.3 * len(pop)))
        l = np.random.uniform(-1, 1)
        D = np.abs(pop[j] - pop[i])
        new_pos = pop[i] + D * np.exp(b * l) * np.cos(2 * np.pi * l)  # Eq. (4)
        new_fit = fobj(new_pos)
        if new_fit < fitness[i]:
            pop[i] = new_pos
            fitness[i] = new_fit
    return pop, fitness


# Update low civilizations — Equation (5) and (6)
def update_low(pop, fobj, fitness, lb, ub):
    for i in range(int(0.8 * len(pop)), len(pop)):
        if np.random.rand() < 0.5:
            pop[i] = np.random.uniform(lb[0], ub[1], pop.shape[1])  # Eq. (5)
        else:
            elite = pop[:int(0.3 * len(pop))]
            centroid = np.mean(elite, axis=0)
            step = np.random.rand()
            pop[i] = centroid + np.random.rand() * (centroid - pop[i]) * step  # Eq. (6)
        fitness[i] = fobj(pop[i])
    return pop, fitness


# Main DFA loop
def DFA(pop, fobj, lb, ub, max_iter):
    n, dim = pop.shape[0], pop.shape[1] 
    fitness = fobj(pop[:])
    best_solution = pop[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(max_iter):
        highest, advanced, normal, low = classify_civilizations(pop, fitness)
        pop, fitness = update_highest(pop, fobj, fitness, t, max_iter)
        pop, fitness = update_advanced(pop, fobj, fitness, highest)
        pop, fitness = update_normal(pop, fobj, fitness)
        pop, fitness = update_low(pop, fobj, fitness, lb, ub)

        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = pop[np.argmin(fitness)]
        Convergence_curve[t] = best_fitness
        t = t + 1
    best_fitness = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct
    return best_fitness, Convergence_curve, best_solution, ct
