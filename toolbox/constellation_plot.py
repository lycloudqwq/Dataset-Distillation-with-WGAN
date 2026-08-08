import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

QAM16_COLORS = plt.get_cmap("tab20")(np.linspace(0, 1, 16))


def plot_constellation(
        points,
        bits,
        title="16QAM Constellation",
):
    points = np.asarray(points)
    bits = np.asarray(bits, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=QAM16_COLORS[bits],
        s=6,
        edgecolors="none",
        zorder=2,
    )

    ax.set_title(title)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.6)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    fig.tight_layout()
    return fig, ax


def plot_constellation_with_rbf_svm_boundary(
        points,
        bits,
        model,
        title="16QAM Constellation with RBF-SVM Boundary",
        grid_points=600,
        padding=0.5,
        region_alpha=0.12,
):
    points = np.asarray(points)
    fig, ax = plot_constellation(points, bits, title)

    x_min = points[:, 0].min() - padding
    x_max = points[:, 0].max() + padding
    y_min = points[:, 1].min() - padding
    y_max = points[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points),
        np.linspace(y_min, y_max, grid_points),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    predicted_bits = model.predict(grid).reshape(xx.shape)

    region_levels = np.arange(-0.5, 16.5, 1)
    region_cmap = ListedColormap(QAM16_COLORS)
    region_norm = BoundaryNorm(region_levels, region_cmap.N)
    ax.pcolormesh(
        xx,
        yy,
        predicted_bits,
        cmap=region_cmap,
        norm=region_norm,
        shading="nearest",
        alpha=region_alpha,
        zorder=0,
    )

    for bit in np.unique(predicted_bits):
        class_region = (predicted_bits == bit).astype(float)
        ax.contour(
            xx,
            yy,
            class_region,
            levels=[0.5],
            colors="black",
            linewidths=1.2,
            zorder=3,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return fig, ax
