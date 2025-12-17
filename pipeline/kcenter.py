import numpy as np


# Default Seed = 0
def kcenter_greedy(points, k, seed=0):
    points = np.asarray(points)
    n = points.shape[0]

    if k <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))

    selected = np.empty(k, dtype=int)
    selected[0] = first

    diff = points - points[first]
    dist2 = np.sum(diff * diff, axis=1)

    for i in range(1, k):
        idx = int(np.argmax(dist2))
        selected[i] = idx

        diff = points - points[idx]
        new_dist2 = np.sum(diff * diff, axis=1)
        dist2 = np.minimum(dist2, new_dist2)

    return selected
