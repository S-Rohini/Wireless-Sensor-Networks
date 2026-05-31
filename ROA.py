import time

import numpy as np


def _ensure_bounds(x, lb, ub):
    """Clip x to bounds elementwise (supports vectors)."""
    return np.minimum(np.maximum(x, lb), ub)


# Revolution Optimization Algorithm (ROA)
def ROA(X, objective, lb, ub, max_iter):
    pop_size, dim = X.shape[0], X.shape[1]

    rng = np.random.default_rng(5)

    # ------------ Initialization (Eq. (1),(2)) ------------
    # Evaluate initial fitnesses (Eq. (3))
    F = np.array([objective(ind.copy()) for ind in X])
    best_idx = np.argmin(F)
    leader = X[best_idx].copy()  # L in paper
    best_f = F[best_idx]

    Convergence_curve = np.zeros((max_iter, 1))

    ct = time.time()
    for t in range(max_iter):
        # store previous iteration positions & fitnesses
        X_prev = X.copy()
        F_prev = F.copy()

        # ------ Phase 1: Revolution ideology (Eq. (4),(5)) ------
        # x_i_j^P1 = (1 - t/T) * x_i_j + (t/T) * L_j
        alpha = t / max_iter
        factor = 1.0 - alpha
        # vectorized update for all individuals:
        X_p1 = factor * X + alpha * leader  # shape (pop_size, dim)

        # clip to bounds
        X_p1 = _ensure_bounds(X_p1, lb, ub)

        # Evaluate and accept if better (Eq. (5))
        F_p1 = np.array([objective(xi) for xi in X_p1])
        improved = F_p1 < F
        X[improved] = X_p1[improved]
        F[improved] = F_p1[improved]

        # update leader if needed
        idx = np.argmin(F)
        if F[idx] < best_f:
            best_f = F[idx]
            leader = X[idx].copy()

        # ------ Phase 2: Revolutionary movement — exploration (Eq. (6),(7)) ------
        # The paper gives: x_i_j^P2 = x_i_j^P2 + r*(L_j - I * x_i_j^P2)
        # (The left and right side reference to x^P2 is interpreted here as:
        #  compute candidate = X + r*(L - I*X), where I \in {1,2} randomly)
        I_choices = rng.integers(1, 3, size=pop_size)  # 1 or 2 for each member
        r_vals = rng.random((pop_size, 1))
        # compute candidate positions
        X_p2 = X + r_vals * (leader - (I_choices.reshape(-1, 1) * X))
        X_p2 = _ensure_bounds(X_p2, lb, ub)
        F_p2 = np.array([objective(xi) for xi in X_p2])
        improved2 = F_p2 < F
        X[improved2] = X_p2[improved2]
        F[improved2] = F_p2[improved2]

        # update leader if needed
        idx = np.argmin(F)
        if F[idx] < best_f:
            best_f = F[idx]
            leader = X[idx].copy()

        # ------ Phase 3: Increasing self-awareness — exploitation (Eq. (8),(9)) ------
        # Eq.(8) given piecewise in the paper:
        # x_i_j^P3 = x_i_j + r*(x_i_j_old - x_i_j)   if F_i_old < F_i
        #          = x_i_j + r*(x_i_j - x_i_j_old)   else
        # where F_i_old is fitness at (t-1) and F_i is current fitness.
        r3 = rng.random((pop_size, 1))
        # condition array: True if previous fitness < current fitness
        cond = (F_prev < F)
        # for those where cond True:
        X_p3 = np.empty_like(X)
        # move towards old position
        X_p3[cond] = X[cond] + r3[cond] * (X_prev[cond] - X[cond])
        # else move away a little (or refine in other direction)
        X_p3[~cond] = X[~cond] + r3[~cond] * (X[~cond] - X_prev[~cond])

        X_p3 = _ensure_bounds(X_p3, lb, ub)
        F_p3 = np.array([objective(xi) for xi in X_p3])
        improved3 = F_p3 < F
        X[improved3] = X_p3[improved3]
        F[improved3] = F_p3[improved3]

        # Final leader update for this iteration
        idx = np.argmin(F)
        if F[idx] < best_f:
            best_f = F[idx]
            leader = X[idx].copy()

        Convergence_curve[t] = best_f
        t = t + 1
    best_f = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct

    return best_f, Convergence_curve, leader, ct
