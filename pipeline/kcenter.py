import numpy as np


# Class-wise K-Center Greedy
# Default Seed = 0
def kcenter_greedy(points, bits, k, seed=0):
    points = np.asarray(points)
    bits = np.asarray(bits)
    classes = np.unique(bits)

    n = points.shape[0]
    if k <= 0:
        return np.array([], dtype=int)

    n_classes = len(classes)
    base = k // n_classes
    rem = k % n_classes

    selected_all = []

    for i, c in enumerate(classes):
        mask = (bits == c)
        idx_class = np.where(mask)[0]
        pts_class = points[idx_class]
        n_c = pts_class.shape[0]

        k_c = base + (1 if i < rem else 0)
        if k_c <= 0 or n_c == 0:
            continue
        if k_c >= n_c:
            selected_all.append(idx_class)
            continue

        rng = np.random.default_rng(seed + i)
        first = int(rng.integers(0, n_c))

        selected = np.empty(k_c, dtype=int)
        selected[0] = first

        diff = pts_class - pts_class[first]
        dist2 = np.sum(diff * diff, axis=1)

        for j in range(1, k_c):
            idx = int(np.argmax(dist2))
            selected[j] = idx

            diff = pts_class - pts_class[idx]
            new_dist2 = np.sum(diff * diff, axis=1)
            dist2 = np.minimum(dist2, new_dist2)

        selected_all.append(idx_class[selected])

    return np.concatenate(selected_all).astype(int) if selected_all else np.array([], dtype=int)
