import numpy as np


def generate_3regular_3xorsat(n_vars: int,
                               return_solution: bool = False):
    """
    Generate one random 3‑regular 3‑XORSAT instance with a planted solution.

    Parameters
    ----------
    n_vars : int
        Number of Boolean variables (and therefore clauses).
    rng : numpy.random.Generator, optional
        Source of randomness (defaults to ``np.random.default_rng()``).
    return_solution : bool, default False
        If True, also return the planted satisfying assignment.

    Returns
    -------
    clauses : (n_vars, 3) ndarray of int
        Each row lists three *1‑based* variable indices; rows are all distinct
        and each column entry appears exactly three times overall.
    rhs : (n_vars,) ndarray of int
        Parity bit (0 or 1) for every clause.
    solution : (n_vars,) ndarray of int, optional
        The planted assignment that satisfies every XOR (only if
        ``return_solution`` is True).
    """
    if n_vars < 1:
        raise ValueError("n_vars must be positive")
    rng = np.random.default_rng()

    # --- 1. build a simple 3‑regular 3‑uniform hyper‑graph ------------------
    stubs = np.repeat(np.arange(n_vars), 3)          # 3 “half‑edges” per var
    for _ in range(1000):                            # few retries almost always suffice
        triples = rng.permutation(stubs).reshape(n_vars, 3)

        # no repeated variable inside a clause
        ok_intra = np.all(np.diff(np.sort(triples, axis=1), axis=1) != 0, axis=1)

        # no two clauses identical (after sorting their literals)
        sorted_triples = np.sort(triples, axis=1)
        _, unique_idx = np.unique(sorted_triples, axis=0, return_index=True)
        ok_inter = len(unique_idx) == n_vars

        if ok_intra.all() and ok_inter:
            break
    else:
        raise RuntimeError("Failed to create a simple 3‑regular instance")

    # --- 2. plant a random solution ----------------------------------------
    sol = rng.integers(0, 2, size=n_vars)            # random {0,1} assignment

    # parity of each triple under the solution becomes the RHS bit
    rhs = sol[triples].sum(axis=1) & 1               # “& 1” gives mod‑2

    # convert to 1‑based indices for conventional XORSAT notation
    # clauses = triples + 1
    clauses = triples

    if return_solution:
        return clauses, rhs, sol
    return clauses, rhs


if __name__ == "__main__":
    # quick demo
    n = 4
    clauses, rhs, sol = generate_3regular_3xorsat(n, return_solution=True)

    print(f"# 3‑regular 3‑XORSAT instance with {n} variables\n")
    for a, b, c, r in zip(*clauses.T, rhs):
        print(f"{a} {b} {c} {r}")
    print("\n# planted solution:")
    print("".join(map(str, sol)))