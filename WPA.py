import time

import numpy as np


def _ensure_bounds(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


# Waterwheel Plant Algorithm (WWPA)
def WPA(X, objective, lb, ub, max_iter):
    pop_size, dim = X.shape[0], X.shape[1]
    rng = np.random.default_rng(5)
    seed = None
    r_prob = 0.5
    f_param = 0.0
    c_param = 0.0
    stagnation_threshold = 3
    F = np.array([objective(x.copy()) for x in X])

    best_idx = int(np.argmin(F))
    Pbest = X[best_idx].copy()
    best_f = F[best_idx]

    # stagnation counters per agent (count of consecutive iterations without improvement)
    stagnation = np.zeros(pop_size, dtype=int)

    history = np.zeros(max_iter)

    # helper: K update function (paper gives a decaying/exponential-like form; use the pseudocode form)
    def compute_K(t):
        # Use formula from paper's pseudocode line: K = (1 + 2 * t^2 / (Tmax)^3 + f)
        # It's a direct transcription; if you prefer a different schedule (e.g. exponential decay),
        # we can change it. Keep f_param available.
        return 1.0 + 2.0 * (t ** 2) / (max_iter ** 3) + f_param

    # main loop
    ct = time.time()
    for t in range(max_iter):
        K = compute_K(t)

        for i in range(pop_size):
            r_choice = rng.random()  # choose exploration vs exploitation (Algorithm 1 uses r < 0.5)
            if r_choice < r_prob:
                # --- Phase 1: Exploration (Eq.4 & Eq.5) ---
                # W = r1 * (P(t) + 2*K)
                # P(t+1) = P(t) + W * (2*K + r2)
                # r1 in [0,2], r2 in [0,1]
                r1 = rng.random() * 2.0
                r2 = rng.random() * 1.0
                W = r1 * (X[i] + 2.0 * K)
                cand = X[i] + W * (2.0 * K + r2)

                # If stagnated for threshold iterations, apply Gaussian mutation variant (Eq.6)
                if stagnation[i] >= stagnation_threshold:
                    # Gaussian(muP, sigma) + r1 * ((P(t) + 2K) / W)
                    # Implement Gaussian around current position; sigma proportional to range
                    muP = X[i]
                    sigma = 0.1 * (ub - lb)  # moderate sigma
                    gauss = rng.normal(loc=muP, scale=np.maximum(sigma, 1e-9))
                    # avoid division by zero: if W has near-zero components, add tiny eps
                    eps = 1e-9
                    denom = np.where(np.abs(W) < eps, eps, W)
                    cand = gauss + r1 * ((X[i] + 2.0 * K) / denom)

                cand = _ensure_bounds(cand, lb, ub)
                f_cand = objective(cand)

                # Accept if better (minimization)
                if f_cand[i] < F[i]:
                    X[i] = cand[i]
                    F[i] = f_cand[i]
                    stagnation[i] = 0
                else:
                    stagnation[i] += 1

            else:
                # --- Phase 2: Exploitation (Eq.7 & Eq.8) ---
                # W = r3 * (K * Pbest + r3 * P(t))
                # P(t+1) = P(t) + K * W
                # r3 in [0,2]
                r3 = rng.random() * 2.0
                W = r3 * (K * Pbest + r3 * X[i])
                cand = X[i] + K * W

                # If stagnated, apply mutation (Eq.9): (r1 + K) * sin(F / (C * theta))
                if stagnation[i] >= stagnation_threshold:
                    # paper: F and C random in [-5,5], theta not fully defined — choose small angle scaling
                    F_rv = rng.uniform(-5.0, 5.0)
                    C_rv = rng.uniform(-5.0, 5.0)
                    # theta: use vector of small angles derived from (X range) to avoid div by 0
                    theta = (X[i] - lb) / (ub - lb + 1e-9) * np.pi  # in [0, pi]
                    sin_arg = F_rv / (C_rv * (theta + 1e-9))
                    # safe sin (broadcast)
                    cand = (rng.random() * 2.0 + K) * np.sin(sin_arg)

                cand = _ensure_bounds(cand, lb, ub)
                f_cand = objective(cand)

                if f_cand[i] < F[i]:
                    X[i] = cand[i]
                    F[i] = f_cand[i]
                    stagnation[i] = 0
                else:
                    stagnation[i] += 1

        # After all agents updated, update global best
        idx = int(np.argmin(F))
        if F[idx] < best_f:
            best_f = F[idx]
            Pbest = X[idx].copy()

        history[t - 1] = best_f
    ct = time.time() - ct
    return best_f, history, Pbest, ct
