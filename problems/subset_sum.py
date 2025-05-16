import numpy as np


def generate_subset_sum_instance(n: int,
                                 return_solution: bool = False):
    """
    Create a random subset‑sum instance at (near‑)critical density d ≈ 1.

    Parameters
    ----------
    n : int
        Number of variables in the instance.
    return_solution : bool, optional
        If True, also return the binary witness vector x that satisfies a·x = t.
    rng : numpy.random.Generator, optional
        Supply your own PRNG; falls back to NumPy’s default if omitted.

    Returns
    -------
    a : numpy.ndarray  (shape=(n,), dtype=object)
        The list of positive integers (≈ n bits each).
    t : int
        Target sum.
    x : numpy.ndarray, optional (shape=(n,), dtype=np.int8)
        Witness subset (only if return_solution is True).
    """
    rng = np.random.default_rng()

    # Numbers need about n bits so that n / log2(max(a)) ≈ 1
    high = 1 << n            # 2**n  (exclusive upper bound)

    while True:
        # Draw n distinct n‑bit integers
        a = rng.choice(high - 1, size=n, replace=False) + 1  # [1, 2**n‑1]
        a = a.astype(np.int64)

        # Pick a random (non‑empty, non‑full) subset as the planted solution
        x = rng.integers(0, 2, size=n, dtype=np.int8)
        if x.any() and not x.all():
            t = int((a * x).sum())
            break  # with overwhelming probability the instance now has density ≈ 1

    if return_solution:
        return a, t, x
    return a, t


if __name__ == '__main__':
    # Example usage
    n = 10
    a, t, x = generate_subset_sum_instance(n, return_solution=True)
    print("a:", a)
    print("t:", t)
    print("x:", x)