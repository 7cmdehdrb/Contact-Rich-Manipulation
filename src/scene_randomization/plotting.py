"""Matplotlib is imported only for visualization, never in the generation path."""

from __future__ import annotations

import numpy as np

from .generator import SceneBatch, zone_masks


def draw_scene(ax, batch: SceneBatch, index: int, resolution: float = 0.5) -> None:
    from matplotlib.patches import Circle, Polygon, Rectangle
    c = batch.config
    p, goal, obs, path = batch.target[index], batch.goal[index], batch.obstacles[index], batch.approach[index]
    d = batch.direction[index]
    x0, x1, y0, y1 = c.board
    lo, hi = c.object_bounds
    nx, ny = max(2, int((hi[0] - lo[0]) / resolution)), max(2, int((hi[1] - lo[1]) / resolution))
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], nx), np.linspace(lo[1], hi[1], ny))
    points = np.stack((xx.ravel(), yy.ravel()), -1)[None]
    red, gray, _ = zone_masks(points, p[None], d[None], batch.distance[index:index + 1], c)
    rgba = np.ones((ny * nx, 4))
    rgba[:, :3] = [0.96, 0.97, 0.98]
    rgba[gray[0], :3] = [0.78, 0.80, 0.84]
    rgba[red[0], :3] = [1.0, 0.79, 0.77]
    ax.imshow(rgba.reshape(ny, nx, 4), origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]),
              interpolation="nearest")
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, edgecolor="#9a9da3", linestyle=":", linewidth=0.8))
    ax.plot([x0, x0, x1, x1], [y0, y1, y1, y0], color="#243042", lw=2)
    ax.plot([x0, x1], [y0, y0], "--", color="#43816c", lw=1)
    perp = np.array([-d[1], d[0]])
    rear = [p + xi * d + eta * perp for xi, eta in
            [(-c.rear_length, -c.rear_width / 2), (0, -c.rear_width / 2),
             (0, c.rear_width / 2), (-c.rear_length, c.rear_width / 2)]]
    ax.add_patch(Polygon(rear, fill=False, edgecolor="#666c78", linestyle=":", linewidth=0.8))
    ax.plot(path[:, 0], path[:, 1], color="#18845e", lw=1.5, zorder=4)
    ax.add_patch(Circle(path[-1], c.hand_radius, color="#18845e", zorder=5))
    for k, point in enumerate(obs):
        color = "#e08b36" if k < c.gray_obstacles else "#437fa5"
        ax.add_patch(Circle(point, c.radius, facecolor=color, edgecolor="#283442", lw=0.8, zorder=5))
        if k < c.rear_obstacles:
            ax.text(*point, "B", color="white", ha="center", va="center", fontsize=8, zorder=6)
    ax.add_patch(Circle(p, c.radius, facecolor="#754eaa", edgecolor="#3e245e", zorder=5))
    ax.add_patch(Circle(goal, c.radius, fill=False, edgecolor="#754eaa", linestyle="--", lw=1.5, zorder=5))
    ax.annotate("", xy=goal, xytext=p, arrowprops=dict(arrowstyle="->", color="#633f8f", lw=1.5), zorder=6)
    ax.set(xlim=(x0 - 2, x1 + 2), ylim=(y0 - 4, y1 + 2), aspect="equal")
    ax.set_title(f"#{index}  {np.degrees(batch.angle[index]):.1f} deg / {batch.distance[index]:.1f} cm"
                 f"  (attempts: {batch.attempts[index]})", fontsize=9)
    ax.tick_params(labelsize=7)


def save_gallery(batch: SceneBatch, path, count: int = 12) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    count = min(count, len(batch))
    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows + 1.0), squeeze=False)
    # Sorted indices illustrate the angle range without changing the samples.
    indices = np.argsort(batch.angle)[np.linspace(0, len(batch) - 1, count).astype(int)]
    for ax, i in zip(axes.flat, indices):
        draw_scene(ax, batch, int(i))
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(color="#ffcac4", label="RED: forbidden obstacle centers"),
                        Patch(color="#c7ccd6", label="GRAY: Tool/Wrist neighbors"),
                        Patch(color="#e08b36", label="GRAY obstacle (B: rear)"),
                        Patch(color="#437fa5", label="WHITE obstacle"),
                        Patch(color="#754eaa", label="Target / dashed goal"),
                        Line2D([0], [0], color="#18845e", label="Certified finger approach")],
               loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Push scenes | RED extends toward -y for Tool/Wrist | cm | open front at y=0", fontsize=14)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95), h_pad=2.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_distribution(batch: SceneBatch, path) -> None:
    import matplotlib.pyplot as plt
    c = batch.config
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(np.degrees(batch.angle), bins=np.linspace(0, 360, c.angle_bins + 1), color="#754eaa")
    axes[0].axhline(len(batch) / c.angle_bins, color="#d15a47", ls="--")
    axes[0].set(xlabel="Push angle (degrees)", ylabel="Scene count", title="Direction coverage")
    axes[1].hist(batch.distance, bins=np.linspace(c.distance_min, c.distance_max, c.distance_bins + 1), color="#437fa5")
    axes[1].axhline(len(batch) / c.distance_bins, color="#d15a47", ls="--")
    axes[1].set(xlabel="Push distance (cm)", title="Distance coverage")
    counts, _, _ = np.histogram2d(np.degrees(batch.angle), batch.distance,
                                bins=(c.angle_bins, c.distance_bins),
                                range=((0, 360), (c.distance_min, c.distance_max)))
    im = axes[2].imshow(counts.T, origin="lower", aspect="auto", cmap="viridis", vmin=0,
                       vmax=max(1, float(counts.max())),
                       extent=(0, 360, c.distance_min, c.distance_max))
    axes[2].set(xlabel="Angle (degrees)", ylabel="Distance (cm)",
                title=f"Joint coverage ({int(counts.min())}..{int(counts.max())} per bin)")
    fig.colorbar(im, ax=axes[2], label="Scene count")
    fig.suptitle(f"{len(batch):,} scenes | generation retries retain angle and distance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_zone_explanation(path) -> None:
    """Compare old capsule zones with fixed world -y Tool/Wrist occupancy."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch
    from .generator import Config
    c = Config()
    p = np.array([[-12., 22.]])
    d = np.array([[1., 0.]])
    distance = np.array([18.])
    goal = p[0] + distance[0] * d[0]
    lo, hi = c.object_bounds
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 365), np.linspace(lo[1], hi[1], 125))
    points = np.stack((xx.ravel(), yy.ravel()), -1)[None]
    red, gray, _ = zone_masks(points, p, d, distance, c)
    rel = points[0] - p[0]
    old_distance = np.hypot(rel[:, 0] - np.clip(rel[:, 0], 0, distance[0]), rel[:, 1])
    old_red = old_distance < 2 * c.radius + c.red_margin
    rear = (rel[:, 0] >= -c.rear_length) & (rel[:, 0] <= 0) & (np.abs(rel[:, 1]) <= c.rear_width / 2)
    old_gray = ((old_distance < c.gray_radius) | rear) & ~old_red
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    for ax, red_mask, gray_mask, title in zip(
            axes, (old_red, red[0]), (old_gray, gray[0]),
            ("Previous capsule zones", "Tool/Wrist zones extending toward -y")):
        rgba = np.ones((len(rel), 4))
        rgba[:, :3] = [0.96, 0.97, 0.98]
        rgba[gray_mask, :3] = [0.78, 0.80, 0.84]
        rgba[red_mask, :3] = [1.0, 0.79, 0.77]
        ax.imshow(rgba.reshape(xx.shape + (4,)), origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]))
        ax.add_patch(Circle(p[0], c.radius, color="#754eaa"))
        ax.add_patch(Circle(goal, c.radius, fill=False, linestyle="--", edgecolor="#754eaa"))
        ax.annotate("", xy=goal, xytext=p[0], arrowprops=dict(arrowstyle="->", color="#633f8f"))
        ax.add_patch(Circle((0, 8), c.radius, facecolor="#437fa5", edgecolor="#283442"))
        ax.annotate("Obstacle below the push path", xy=(0, 8), xytext=(10, 3), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#283442"))
        ax.plot([-50, -50, 50, 50], [0, 40, 40, 0], color="#243042", lw=2)
        ax.plot([-50, 50], [0, 0], "--", color="#43816c")
        ax.set(title=title, xlim=(-52, 52), ylim=(-4, 42), aspect="equal", xlabel="x (cm)")
    axes[0].set_ylabel("y (cm)")
    axes[1].plot([0], [8], marker="x", color="#a52424", markersize=14, markeredgewidth=3)
    fig.legend(handles=[Patch(color="#ffcac4", label="RED: forbidden obstacle centers"),
                        Patch(color="#c7ccd6", label="GRAY: Tool/Wrist neighbors")],
               loc="lower center", ncol=2)
    fig.suptitle("The same x footprint stays clear from the capsule's upper boundary toward the front")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)
