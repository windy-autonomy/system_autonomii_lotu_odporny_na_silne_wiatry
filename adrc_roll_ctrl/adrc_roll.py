#!/usr/bin/env python3
"""Create a Matplotlib GIF visualization of roll-angle ADRC control."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def disturbance_at(t: float) -> float:
    """External roll acceleration disturbance in rad/s^2."""
    if 1.2 <= t < 2.0:
        return math.radians(65.0)
    if 3.2 <= t < 3.8:
        return math.radians(-40.0)
    return 0.0


def simulate(total_time: float, dt: float) -> dict[str, list[float]]:
    """Simulate a roll plant controlled to a zero roll-angle reference."""
    # Plant: phi_ddot = b0 * delta_a + d(t) - damping * phi_dot
    b0 = 9.0
    damping = 1.25
    max_aileron = math.radians(25.0)

    # LADRC tuning. wc controls closed-loop speed, wo observer bandwidth.
    wc = 5.0
    wo = 24.0
    kp = wc * wc
    kd = 2.0 * wc
    beta1 = 3.0 * wo
    beta2 = 3.0 * wo * wo
    beta3 = wo * wo * wo

    phi = 0.0
    p = 0.0
    z1 = 0.0  # estimated roll angle
    z2 = 0.0  # estimated roll rate
    z3 = 0.0  # estimated total disturbance
    delta = 0.0

    data = {
        "t": [],
        "phi": [],
        "p": [],
        "delta": [],
        "disturbance": [],
        "z3": [],
    }

    steps = int(total_time / dt) + 1
    for i in range(steps):
        t = i * dt
        ref = 0.0
        d = disturbance_at(t)

        u0 = kp * (ref - z1) + kd * (0.0 - z2)
        delta_cmd = clamp((u0 - z3) / b0, -max_aileron, max_aileron)

        # A small actuator lag keeps the aileron motion readable.
        delta += (delta_cmd - delta) * min(1.0, dt / 0.06)

        phi_ddot = b0 * delta + d - damping * p
        phi += p * dt
        p += phi_ddot * dt

        # Extended state observer.
        e = z1 - phi
        z1 += (z2 - beta1 * e) * dt
        z2 += (z3 + b0 * delta - beta2 * e) * dt
        z3 += (-beta3 * e) * dt

        data["t"].append(t)
        data["phi"].append(phi)
        data["p"].append(p)
        data["delta"].append(delta)
        data["disturbance"].append(d)
        data["z3"].append(z3)

    return data


def rotate_points(points: list[tuple[float, float]], angle: float) -> list[tuple[float, float]]:
    ca = math.cos(angle)
    sa = math.sin(angle)
    return [(x * ca - y * sa, x * sa + y * ca) for x, y in points]


def make_aircraft_polygons(roll: float, aileron: float) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[tuple[float, float]],
]:
    wing = [(-3.2, -0.13), (3.2, -0.13), (3.35, 0.13), (-3.35, 0.13)]
    fuselage = [(-0.28, -0.52), (0.28, -0.52), (0.42, 0.28), (0.0, 0.62), (-0.42, 0.28)]

    left_aileron = [(-3.05, 0.16), (-1.85, 0.16), (-1.85, 0.16 - 0.95 * math.sin(aileron)), (-3.05, 0.16 - 0.95 * math.sin(aileron))]
    right_aileron = [(1.85, 0.16), (3.05, 0.16), (3.05, 0.16 + 0.95 * math.sin(aileron)), (1.85, 0.16 + 0.95 * math.sin(aileron))]

    return (
        rotate_points(wing, roll),
        rotate_points(left_aileron, roll),
        rotate_points(right_aileron, roll),
        rotate_points(fuselage, roll),
    )


def render_gif(data: dict[str, list[float]], output: Path, fps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import FancyArrowPatch, Polygon

    t = data["t"]
    roll_deg = [math.degrees(v) for v in data["phi"]]
    delta_deg = [math.degrees(v) for v in data["delta"]]
    disturbance_deg = [math.degrees(v) for v in data["disturbance"]]

    fig = plt.figure(figsize=(10.5, 6.2), dpi=110)
    fig.patch.set_facecolor("#f7f9fa")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.28)
    ax_plane = fig.add_subplot(grid[0])
    ax_plot = fig.add_subplot(grid[1])

    ax_plane.set_title("ADRC roll control to 0 deg reference", loc="left", fontsize=16, fontweight="bold")
    ax_plane.text(-4.2, 2.0, "front view: roll angle, differential ailerons, and disturbance rejection", fontsize=10, color="#33404c")
    ax_plane.set_xlim(-4.4, 4.4)
    ax_plane.set_ylim(-2.2, 2.2)
    ax_plane.set_aspect("equal")
    ax_plane.axis("off")
    ax_plane.axhline(0, color="#b7c1c8", linestyle=(0, (5, 5)), linewidth=1.3)

    wing_patch = Polygon([[0, 0]], closed=True, facecolor="#8fa7b7", edgecolor="#17212b", linewidth=2.0)
    left_aileron_patch = Polygon([[0, 0]], closed=True, facecolor="#d84a3a", edgecolor="#7d1f17", linewidth=1.5)
    right_aileron_patch = Polygon([[0, 0]], closed=True, facecolor="#d84a3a", edgecolor="#7d1f17", linewidth=1.5)
    fuselage_patch = Polygon([[0, 0]], closed=True, facecolor="#f2f4f3", edgecolor="#17212b", linewidth=2.0)
    for patch in (wing_patch, left_aileron_patch, right_aileron_patch, fuselage_patch):
        ax_plane.add_patch(patch)

    reference_line, = ax_plane.plot([-1.0, 1.0], [-1.55, -1.55], color="#17212b", linewidth=1.5)
    ax_plane.text(-0.95, -1.86, "reference: 0 deg roll", fontsize=10, color="#17212b")
    info_text = ax_plane.text(2.15, 1.35, "", fontsize=11, family="monospace", color="#17212b", va="top")
    gust_arrow = FancyArrowPatch((0, 0), (0, 0), connectionstyle="arc3,rad=0.35", arrowstyle="-|>", mutation_scale=18, color="#e09b2d", linewidth=2.8)
    ax_plane.add_patch(gust_arrow)
    gust_label = ax_plane.text(-0.65, 1.65, "gust torque", fontsize=10, color="#9a5d00", alpha=0.0)

    ax_plot.set_title("Time history", loc="left", fontsize=11, fontweight="bold")
    ax_plot.set_xlim(0, t[-1])
    ax_plot.set_ylim(-90, 90)
    ax_plot.axhline(0, color="#aab3ba", linewidth=1)
    ax_plot.grid(True, color="#e2e7eb", linewidth=0.8)
    ax_plot.set_xlabel("time [s]")
    ax_plot.set_ylabel("deg / deg/s^2")
    roll_line, = ax_plot.plot([], [], color="#1976a3", linewidth=2.4, label="roll angle")
    delta_line, = ax_plot.plot([], [], color="#d84a3a", linewidth=2.0, label="aileron deflection")
    disturbance_line, = ax_plot.plot([], [], color="#e09b2d", linewidth=1.8, label="disturbance")
    time_cursor = ax_plot.axvline(0, color="#30363d", linewidth=1)
    ax_plot.legend(loc="upper right", ncols=3, frameon=False, fontsize=9)

    step = max(1, round((1 / fps) / (t[1] - t[0])))
    frame_indices = list(range(0, len(t), step))

    def update(frame_number: int):
        i = frame_indices[frame_number]
        wing, left_aileron, right_aileron, fuselage = make_aircraft_polygons(data["phi"][i], data["delta"][i])
        wing_patch.set_xy(wing)
        left_aileron_patch.set_xy(left_aileron)
        right_aileron_patch.set_xy(right_aileron)
        fuselage_patch.set_xy(fuselage)

        info_text.set_text(
            f"t = {t[i]:4.2f} s\n"
            f"roll = {roll_deg[i]:6.2f} deg\n"
            f"aileron = {delta_deg[i]:6.2f} deg\n"
            f"disturbance = {disturbance_deg[i]:6.1f} deg/s^2"
        )

        if abs(disturbance_deg[i]) > 1.0:
            sign = 1 if disturbance_deg[i] > 0 else -1
            gust_arrow.set_positions((-0.8 * sign, 1.45), (0.8 * sign, 0.45))
            gust_arrow.set_alpha(1.0)
            gust_label.set_alpha(1.0)
            gust_label.set_position((-1.15 * sign, 1.68))
        else:
            gust_arrow.set_alpha(0.0)
            gust_label.set_alpha(0.0)

        roll_line.set_data(t[: i + 1], roll_deg[: i + 1])
        delta_line.set_data(t[: i + 1], delta_deg[: i + 1])
        disturbance_line.set_data(t[: i + 1], disturbance_deg[: i + 1])
        time_cursor.set_xdata([t[i], t[i]])
        return (
            wing_patch,
            left_aileron_patch,
            right_aileron_patch,
            fuselage_patch,
            reference_line,
            info_text,
            gust_arrow,
            gust_label,
            roll_line,
            delta_line,
            disturbance_line,
            time_cursor,
        )

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / fps, blit=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Matplotlib ADRC roll-control GIF.")
    parser.add_argument("--output", default="adrc_roll_control.gif", help="output GIF path")
    parser.add_argument("--duration", type=float, default=5.0, help="simulation duration in seconds")
    parser.add_argument("--fps", type=int, default=24, help="animation frames per second")
    parser.add_argument("--dt", type=float, default=0.002, help="simulation step in seconds")
    args = parser.parse_args()

    data = simulate(args.duration, args.dt)
    render_gif(data, Path(args.output), args.fps)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
