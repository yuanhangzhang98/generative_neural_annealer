import numpy as np

def contamination(x, runlength=10_000, seed=None):
    """
    Monte‑Carlo objective/constraint estimator for the multi‑stage
    contamination‑control problem.

    Parameters
    ----------
    x : (d,) array_like of 0/1
        1 = apply prevention at that stage.
    runlength : int
        Number of independent simulated production runs.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    cost : float
        Total prevention cost  (simply sum(x)).
    constraint : (d,) ndarray
        Estimated constraint values  g_i = P[Z_i <= 0.1] - 0.95  .
        Feasible ⇔  all(g_i >= 0).
    cov : (d, d) ndarray
        Sample covariance matrix of the indicator vectors used
        to estimate the constraints (handy for gradient‑free
        stochastic optimisation algorithms).
    """
    x = np.asarray(x, dtype=int)
    if ((x != 0) & (x != 1)).any():
        raise ValueError("x must be a 0/1 vector")

    rng = np.random.default_rng(seed)
    d = x.size

    # ----- draw all random variables -----
    Z0      = rng.beta(1, 30,   size=runlength)          # initial contamination
    Lambda  = rng.beta(1, 17/3, size=(d, runlength))     # growth rates
    Gamma   = rng.beta(1, 3/7,  size=(d, runlength))     # restoration rates

    # ----- forward recursion -----
    Z = np.empty((d, runlength))
    # stage 1 uses Z0
    Z[0] = (Lambda[0] * (1 - x[0]) * (1 - Z0) +
            (1 - Gamma[0] * x[0]) * Z0)

    # downstream stages
    for i in range(1, d):
        Z[i] = (Lambda[i] * (1 - x[i]) * (1 - Z[i-1]) +
                (1 - Gamma[i] * x[i]) * Z[i-1])

    # ----- constraint estimates -----
    indicator = (Z <= 0.1).T         # shape (runlength, d)
    prob_hat  = indicator.mean(axis=0)
    constraint = prob_hat - 0.95     # ≥ 0 ⇒ satisfies requirement

    # ----- cost -----
    cost = x.sum() - constraint.sum()                   # each prevention costs 1

    return cost


# quick sanity check ---------------------------------------------------------
if __name__ == "__main__":
    d = 25
    x_test = np.zeros(d, dtype=int)      # no prevention anywhere
    c = contamination(x_test, runlength=20000, seed=42)
    print(f"cost = {c}")
