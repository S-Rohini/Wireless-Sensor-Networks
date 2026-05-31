import time

import numpy as np


def COA(SearchAgents, fitness, lowerbound, upperbound, Max_iterations):
    N, dimension = SearchAgents.shape[0], SearchAgents.shape[1]
    X = SearchAgents

    fit = np.array([fitness(ind) for ind in X])

    best_so_far = np.zeros(Max_iterations)
    average = np.zeros(Max_iterations)
    ct = time.time()
    # --- MAIN LOOP ---
    for t in range(1, Max_iterations + 1):
        # update the best candidate solution
        best = np.min(fit)
        location = np.argmin(fit)

        if t == 1:
            Xbest = X[location, :].copy()
            fbest = best
        elif best < fbest:
            fbest = best
            Xbest = X[location, :].copy()

        # --- Phase 1: Hunting and attacking strategy (Exploration Phase) ---
        for i in range(SearchAgents.shape[0] // 2):
            iguana = Xbest
            I = int(np.round(1 + np.random.rand()))
            X_P1 = X[i, :] + np.random.rand() * (iguana - I * X[i, :])  # Eq. (4)
            X_P1 = np.maximum(X_P1, lowerbound)
            X_P1 = np.minimum(X_P1, upperbound)

            F_P1 = fitness(X_P1)
            if F_P1[i] < fit[i]:
                X[i, :] = X_P1[i]
                fit[i] = F_P1[i]

        for i in range(SearchAgents.shape[0] // 2):
            iguana = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)  # Eq. (5)
            F_HL = fitness(iguana)
            I = int(np.round(1 + np.random.rand()))

            if fit[i] > F_HL[i]:
                X_P1 = X[i, :] + np.random.rand() * (iguana - I * X[i, :])  # Eq. (6)
            else:
                X_P1 = X[i, :] + np.random.rand() * (X[i, :] - iguana)  # Eq. (6)

            X_P1 = np.maximum(X_P1, lowerbound)
            X_P1 = np.minimum(X_P1, upperbound)

            F_P1 = fitness(X_P1)
            if F_P1[i] < fit[i]:
                X[i, :] = X_P1[i]
                fit[i] = F_P1[i]

        # --- Phase 2: Escaping from predators (Exploitation Phase) ---
        for i in range(SearchAgents.shape[0]):
            LO_LOCAL = lowerbound / t  # Eq. (9)
            HI_LOCAL = upperbound / t  # Eq. (10)

            X_P2 = X[i, :] + (1 - 2 * np.random.rand()) * (
                        LO_LOCAL + np.random.rand() * (HI_LOCAL - LO_LOCAL))  # Eq. (8)
            X_P2 = np.maximum(X_P2, LO_LOCAL)
            X_P2 = np.minimum(X_P2, HI_LOCAL)

            F_P2 = fitness(X_P2)
            if F_P2[i] < fit[i]:
                X[i, :] = X_P2[i]
                fit[i] = F_P2[i]

        # Save history
        best_so_far[t - 1] = fbest
        average[t - 1] = np.mean(fit)

    Best_score = fbest
    Best_pos = Xbest
    COA_curve = best_so_far
    ct = time.time() - ct
    return Best_score, COA_curve, Best_pos, ct
