"""
MCTS Square Grid — Intersection
=================================
20x20 grid. gy=0 is the TOP of the screen, gy=19 is the BOTTOM.

Road layout (drive-on-right)
-----------------------------
  E-W road : rows (gy) 8–11
    Eastbound lanes : gy = 10, 11   (right side going east)
    Westbound lanes : gy =  8,  9   (right side going west)
  N-S road : cols (gx) 8–11
    Northbound lanes : gx =  8,  9  (right side going north = west cols)
    Southbound lanes : gx = 10, 11  (right side going south = east cols)
  Intersection box  : gx 8–11, gy 8–11
  Four corner blocks : off-road, unreachable

Agents
------
    Car     : West → East, eastbound lane (gy=10 or 11).
             State = (gx, gy, speed)
             gx    : 0–19
             gy    : 10 or 11  (lane change allowed)
             speed : 0–3 blocks advanced per timestep
             Actions = all combos of:
                         acceleration : -2, -1, 0, +1, +2  (clamped to [0, 3] speed)
             lane change  : -1, 0, +1  (gy ± 1, clamped to eastbound lanes)
                         → 15 actions total (5 accel × 3 lane)

    Scooter : South → West (fixed path, unknown to car).
             Starts gy=15 (near bottom), drives north in northbound lane (gx=9),
             turns left (West) at intersection, exits westward (gy=11).
                         Transitions are deterministic along SCOOTER_PATH.

MCTS
----
  Car knows the scooter's CURRENT position (visible) but not its future path.
  The cone { (gx,gy): (probability, depth) } encodes the predicted future.

Robustness test
---------------
    The default run is deterministic.
    A separate robustness mode samples scooter deviations from the kernel.
"""

import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import defaultdict, deque
from enum import IntEnum

class Direction(IntEnum):
    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

# ---------------------------------------------------------------------------
# Tunable parameters (all in one place)
# ---------------------------------------------------------------------------

# --- Grid and road layout ---
GRID_W, GRID_H = 20, 20
ROAD_HALF      = 2
CENTER         = 10
COL_LO, COL_HI = CENTER - ROAD_HALF, CENTER + ROAD_HALF   # 8,12 — N-S road cols
ROW_LO, ROW_HI = CENTER - ROAD_HALF, CENTER + ROAD_HALF   # 8,12 — E-W road rows
EASTBOUND_LANES = [10, 11]   # gy values for eastbound car lanes

# --- MCTS and planning ---
MCTS_ITERATIONS = 1000   # Number of MCTS rollouts per step
EXPLORATION     = 1.7    # UCB exploration constant
ROLLOUT_DEPTH   = 10     # Planning horizon (steps)
ROLLOUT_GAMMA   = 0.9    # Discount factor for future rewards

# --- Collision and cone ---
CONE_DEPTH              = 3     # How far to predict scooter cone
BPA_DEPTH               = 5     # BPA backtrack steps
COLLISION_APPROX_BUFFER = 4     # Cells/time for cone pruning
BPA_EPSILON             = 0.01  # BPA/collision truncation threshold

# --- Rewards and penalties (TUNED for yielding behavior, iteration 4) ---
GOAL_REWARD          = 100.0 # Strong reward for reaching the goal
TIMESTEP_PENALTY     = 1.0  # Penalty per timestep
HAZARD_SCALES        = [10.0, 50.0, 100.0]   # No-prune hazard-only scenarios
LANE_PENALTY         = 5.0   # Penalty for lane change
INTERSECTION_PENALTY = 2.0   # Low penalty for being in intersection
JERK_PENALTY         = 5.0   # Penalty per unit of |ds|-1 (aggressive accel)

# --- Car setup ---
CAR_START  = (0, 10, 3)    # (gx=0, gy=10 inner eastbound lane, speed=3)
GOAL_GX    = GRID_W - 1    # 19
SPEED_MIN, SPEED_MAX = 0, 3

# --- Scooter setup ---
SCOOTER_START = (Direction.NORTH, 11, GRID_H - 5)  # dir=N, gx=11, gy=15
SCOOTER_PATH  = ([0] * 7  # turn time = count of leading straight steps
               + [1]       # left turn: N->W
               + [0] * 13)  # forward west, exits top-left

# --- Car actions ---
CAR_ACTIONS  = [(ds, dl) for ds in (-2, -1, 0, 1, 2) for dl in (-1, 0, 1)]

# --- Scooter transition kernel ---
SCOOTER_TRANSITION_KERNEL = {
    (0, -1): 0.125,
    (-1, 0): 0.125,
    (0, 0): 0.5,
    (1, 0): 0.125,
    (0, 1): 0.125,
}

# --- Robustness study ---
ROBUSTNESS_RUN_COUNT = 30
ROBUSTNESS_ANIMATED_RUNS = 4 # Only animates the BPA pruning runs
ROBUSTNESS_HAZARD_SCALE = 50.0
STOCHASTIC_PERCENTAGE = 0.25

# ---------------------------------------------------------------------------
# Map parameters (do not tune below here)
# ---------------------------------------------------------------------------

def is_road(gx: int, gy: int) -> bool:
    on_ew = ROW_LO <= gy < ROW_HI
    on_ns = COL_LO <= gx < COL_HI
    return on_ew or on_ns

def in_bounds(gx: int, gy: int) -> bool:
    return 0 <= gx < GRID_W and 0 <= gy < GRID_H

def is_valid(gx: int, gy: int) -> bool:
    return in_bounds(gx, gy) and is_road(gx, gy)

# ---------------------------------------------------------------------------
# Directions  0=E 1=N 2=W 3=S
# gy=0 is TOP so North = gy decreases, South = gy increases
# ---------------------------------------------------------------------------

DIR_DELTA = {
    Direction.EAST: (1, 0),
    Direction.NORTH: (0, -1),
    Direction.WEST: (-1, 0),
    Direction.SOUTH: (0, 1),
}

def turn_left(d):
    return Direction((Direction(d).value + 1) % 4)


def turn_right(d):
    return Direction((Direction(d).value - 1) % 4)


def _sample_kernel_offset(rng: random.Random, kernel: dict) -> tuple[int, int]:
    total = sum(kernel.values())
    draw = rng.random() * total
    cumulative = 0.0
    last_offset = (0, 0)
    for offset, weight in kernel.items():
        last_offset = offset
        cumulative += weight
        if draw <= cumulative:
            return offset
    return last_offset


def _rotate_local(dx: int, dy: int, facing: Direction) -> tuple[int, int]:
    if facing == Direction.EAST:
        return dx, dy
    if facing == Direction.NORTH:
        return dy, -dx
    if facing == Direction.WEST:
        return -dx, -dy
    if facing == Direction.SOUTH:
        return -dy, dx
    return dx, dy


def _path_cells(start_gx: int, start_gy: int, end_gx: int, end_gy: int) -> set[tuple[int, int]]:
    """Approximate within-step occupied cells for an agent transition."""
    cells: set[tuple[int, int]] = {(start_gx, start_gy)}

    if end_gx != start_gx:
        step_x = 1 if end_gx > start_gx else -1
        for x in range(start_gx + step_x, end_gx + step_x, step_x):
            cells.add((x, start_gy))

    if end_gy != start_gy:
        step_y = 1 if end_gy > start_gy else -1
        for y in range(start_gy + step_y, end_gy + step_y, step_y):
            cells.add((end_gx, y))

    return cells


def _first_goal_step(car_traj: list) -> int | None:
    """Return first index where the car reaches goal, or None."""
    return next((i for i, st in enumerate(car_traj) if is_terminal(st)), None)


def _scenario_outcome(car_traj: list, scooter_traj: list) -> dict:
    """Classify scenario outcome for post-goal overlay color."""
    goal_step = _first_goal_step(car_traj)

    eval_end = goal_step if goal_step is not None else (len(car_traj) - 1)
    lane_change = any(
        car_traj[i][1] != car_traj[i - 1][1]
        for i in range(1, eval_end + 1)
    )

    stopped_far_from_intersection = any(
        (st[2] == 0)
        and (st[1] in EASTBOUND_LANES)
        and not (COL_LO - 2 <= st[0] < COL_HI + 2)
        for st in car_traj[:eval_end + 1]
    )

    collision_or_cross = False
    cut_in_front_close = False
    cut_in_front = False
    horizon = min(eval_end, len(scooter_traj) - 1)
    for i in range(1, horizon + 1):
        c0 = car_traj[i - 1]
        c1 = car_traj[i]
        s0 = scooter_traj[i - 1]
        s1 = scooter_traj[i]

        car_cells = _path_cells(c0[0], c0[1], c1[0], c1[1])
        sc_cells = _path_cells(s0[1], s0[2], s1[1], s1[2])

        if car_cells & sc_cells:
            collision_or_cross = True
            break

        # Red if car cuts in front of scooter by only 1-2 blocks.
        # "In front" is measured along scooter's heading direction.
        fdx, fdy = DIR_DELTA[s1[0]]
        for (cgx, cgy) in car_cells:
            # Restrict to the intersection neighborhood to avoid distant false positives.
            if not (COL_LO - 1 <= cgx <= COL_HI and ROW_LO - 1 <= cgy <= ROW_HI):
                continue
            for (sgx, sgy) in sc_cells:
                rel_x = cgx - sgx
                rel_y = cgy - sgy
                forward_gap = rel_x * fdx + rel_y * fdy
                lateral_gap = abs(rel_x * fdy - rel_y * fdx)
                # Only treat this as a cut-in-front case when the scooter is
                # crossing the car's eastbound heading at right angles.
                headings_perpendicular = (fdx == 0)
                if headings_perpendicular and 0 == forward_gap and lateral_gap <= 1:
                    cut_in_front_close = True
                    break
                elif headings_perpendicular and 1 <= forward_gap and lateral_gap <= 1:
                    cut_in_front = True
            if cut_in_front_close:
                break
        if cut_in_front_close:
            break

    if collision_or_cross or cut_in_front_close:
        color = "#d32f2f"  # red
    elif stopped_far_from_intersection or lane_change or cut_in_front:
        color = "#f9a825"  # yellow
    else:
        color = "#2e7d32"  # green

    return {
        "goal_step": goal_step,
        "color": color,
    }

def animate_four_way(
    scenarios: list,
    interval_ms: int = 500,
    save_gif: bool = True,
):
    """Show four simulations in a 2x2 layout, or three with one empty panel."""
    if len(scenarios) not in (3, 4):
        raise ValueError("animate_four_way expects 3 or 4 scenarios")

    n_frames = min(len(s["car"]) for s in scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=DARK_BG)
    axes = axes.flatten()

    artists = []
    for ax, s in zip(axes, scenarios):
        ax.set_facecolor(DARK_BG)
        _draw_static_map(ax)
        outcome = _scenario_outcome(s["car"], s["scooter"])

        cone_patches = {}
        for gx in range(GRID_W):
            for gy in range(GRID_H):
                if is_road(gx, gy):
                    p = plt.Rectangle((gx, gy), 1, 1, facecolor=CONE_COL, edgecolor="none", alpha=0.0, zorder=3)
                    ax.add_patch(p)
                    cone_patches[(gx, gy)] = p

        car_rect = plt.Rectangle((0, 0), 1.0, 1.0, facecolor=CAR_COL, edgecolor="white", linewidth=1.2, zorder=6)
        ax.add_patch(car_rect)
        scooter_dot, = ax.plot([], [], "o", color=SCOOTER_COL, markersize=12, markeredgecolor="white", markeredgewidth=0.8, zorder=7)
        step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white", fontsize=8, va="top", ha="left", fontfamily="monospace", zorder=8)

        # Post-goal outcome overlay.
        overlay_rect = mpatches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor=outcome["color"],
            alpha=0.5,
            edgecolor="none",
            zorder=9,
            visible=False,
        )
        ax.add_patch(overlay_rect)
        overlay_text = ax.text(
            0.5,
            0.5,
            "",
            transform=ax.transAxes,
            color="white",
            fontsize=44,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
            visible=False,
        )

        ax.set_title(s["title"], color="white")
        artists.append((s, outcome, cone_patches, car_rect, scooter_dot, step_text, overlay_rect, overlay_text))

    for ax in axes[len(scenarios):]:
        ax.axis("off")

    def _update(frame):
        t = frame % n_frames
        updated = []
        for s, outcome, cone_patches, car_rect, scooter_dot, step_text, overlay_rect, overlay_text in artists:
            car = s["car"][t]
            scooter = s["scooter"][t]
            cone = s["cones"][t]

            for p in cone_patches.values():
                p.set_alpha(0.0)
            if cone:
                max_p = max(v[0] for v in cone.values()) or 1.0
                for (cgx, cgy), (prob, _) in cone.items():
                    if (cgx, cgy) in cone_patches:
                        cone_patches[(cgx, cgy)].set_alpha(min(0.1 + 0.55 * prob / max_p, 0.72))

            gx, gy, speed = car
            car_rect.set_xy((gx, gy))
            _, sx, sy = scooter
            scooter_dot.set_data([sx + 0.5], [sy + 0.5])
            step_text.set_text(
                f"step {t:>2d} / {len(s['car']) - 1}\n"
                f"car   gx={gx:>2d}  gy={gy}  spd={speed}\n"
                f"scoot gx={sx:>2d}  gy={sy}"
            )

            goal_step = outcome["goal_step"]
            if goal_step is not None and t >= goal_step:
                overlay_rect.set_visible(True)
                overlay_text.set_visible(True)
                overlay_text.set_text(str(goal_step))
            else:
                overlay_rect.set_visible(False)
                overlay_text.set_visible(False)

            updated.extend(list(cone_patches.values()))
            updated.extend([car_rect, scooter_dot, step_text, overlay_rect, overlay_text])
        return updated

    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_gif:
        path = (
            "mcts_intersection_robustness.gif"
            if any(s["title"].startswith("ROBUSTNESS") for s in scenarios)
            else "mcts_pruning_comparison.gif"
        )
        anim.save(path, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        print(f"Saved {path}")

    plt.tight_layout()
    plt.show()


def plot_robustness_histogram(scenarios: list, save_png: bool = True):
    """Plot steps-to-goal counts by outcome color using grouped bars."""
    if not scenarios:
        return

    red_steps = []
    yellow_steps = []
    green_steps = []

    for s in scenarios:
        outcome = _scenario_outcome(s["car"], s["scooter"])
        goal_step = outcome["goal_step"]
        steps = goal_step if goal_step is not None else (len(s["car"]) - 1)

        if outcome["color"] == "#d32f2f":
            red_steps.append(steps)
        elif outcome["color"] == "#f9a825":
            yellow_steps.append(steps)
        else:
            green_steps.append(steps)

    all_steps = red_steps + yellow_steps + green_steps
    if not all_steps:
        return

    total_runs = len(all_steps)
    min_step = min(all_steps)
    max_step = max(all_steps)
    step_values = list(range(min_step, max_step + 1))

    red_counts = [red_steps.count(step) for step in step_values]
    yellow_counts = [yellow_steps.count(step) for step in step_values]
    green_counts = [green_steps.count(step) for step in step_values]

    red_pct = (100.0 * len(red_steps) / total_runs)
    yellow_pct = (100.0 * len(yellow_steps) / total_runs)
    green_pct = (100.0 * len(green_steps) / total_runs)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    bar_w = 0.22
    x = step_values
    red_x = [v - bar_w for v in x]
    yellow_x = x
    green_x = [v + bar_w for v in x]

    ax.bar(red_x, red_counts, width=bar_w, color="#d32f2f", edgecolor=GRID_LINE,
           label=f"Red ({red_pct:.1f}%)")
    ax.bar(yellow_x, yellow_counts, width=bar_w, color="#f9a825", edgecolor=GRID_LINE,
           label=f"Yellow ({yellow_pct:.1f}%)")
    ax.bar(green_x, green_counts, width=bar_w, color="#2e7d32", edgecolor=GRID_LINE,
           label=f"Green ({green_pct:.1f}%)")

    ax.set_title("Robustness: Steps To Goal (30 Runs)", color="white")
    ax.set_xlabel("Car steps to goal", color="white")
    ax.set_ylabel("Run count", color="white")
    ax.set_xticks(step_values)
    ax.set_xticklabels([str(v) for v in step_values], color="white")

    max_count = max(red_counts + yellow_counts + green_counts)
    if max_count <= 1:
        y_ticks = [0, 1]
    else:
        y_step = 2
        y_top = max_count if (max_count % y_step == 0) else (max_count + (y_step - (max_count % y_step)))
        y_ticks = list(range(0, y_top + y_step, y_step))
    ax.set_yticks(y_ticks)
    ax.set_ylim(0, y_ticks[-1])

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.grid(True, axis="y", color="#444", alpha=0.3)
    ax.legend(facecolor=DARK_BG, edgecolor="white", labelcolor="white")

    # Per-color percentage summary in-plot for quick readability.
    ax.text(
        0.99,
        0.98,
        f"Red: {red_pct:.1f}%   Yellow: {yellow_pct:.1f}%   Green: {green_pct:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="#161b22", edgecolor=GRID_LINE, alpha=0.8, boxstyle="round,pad=0.3"),
    )

    plt.tight_layout()
    if save_png:
        path = "robustness_steps_histogram.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    plt.show()


def _steps_by_outcome_color(scenarios: list) -> tuple[list, list, list]:
    red_steps = []
    yellow_steps = []
    green_steps = []

    for s in scenarios:
        outcome = _scenario_outcome(s["car"], s["scooter"])
        goal_step = outcome["goal_step"]
        steps = goal_step if goal_step is not None else (len(s["car"]) - 1)

        if outcome["color"] == "#d32f2f":
            red_steps.append(steps)
        elif outcome["color"] == "#f9a825":
            yellow_steps.append(steps)
        else:
            green_steps.append(steps)

    return red_steps, yellow_steps, green_steps


def _style_hist_axis(ax, step_values: list, red_counts: list, yellow_counts: list, green_counts: list):
    ax.set_xticks(step_values)
    ax.set_xticklabels([str(v) for v in step_values], color="white")

    max_count = max(red_counts + yellow_counts + green_counts) if step_values else 0
    if max_count <= 1:
        y_ticks = [0, 1]
    else:
        y_step = 2
        y_top = max_count if (max_count % y_step == 0) else (max_count + (y_step - (max_count % y_step)))
        y_ticks = list(range(0, y_top + y_step, y_step))

    ax.set_yticks(y_ticks)
    ax.set_ylim(0, y_ticks[-1])
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.grid(True, axis="y", color="#444", alpha=0.3)


def _plot_grouped_outcome_bars(ax, scenarios: list, title: str, step_values: list):
    red_steps, yellow_steps, green_steps = _steps_by_outcome_color(scenarios)
    total_runs = len(red_steps) + len(yellow_steps) + len(green_steps)

    red_counts = [red_steps.count(step) for step in step_values]
    yellow_counts = [yellow_steps.count(step) for step in step_values]
    green_counts = [green_steps.count(step) for step in step_values]

    red_pct = (100.0 * len(red_steps) / total_runs) if total_runs else 0.0
    yellow_pct = (100.0 * len(yellow_steps) / total_runs) if total_runs else 0.0
    green_pct = (100.0 * len(green_steps) / total_runs) if total_runs else 0.0

    bar_w = 0.22
    red_x = [v - bar_w for v in step_values]
    yellow_x = step_values
    green_x = [v + bar_w for v in step_values]

    ax.bar(red_x, red_counts, width=bar_w, color="#d32f2f", edgecolor=GRID_LINE,
           label=f"Red ({red_pct:.1f}%)")
    ax.bar(yellow_x, yellow_counts, width=bar_w, color="#f9a825", edgecolor=GRID_LINE,
           label=f"Yellow ({yellow_pct:.1f}%)")
    ax.bar(green_x, green_counts, width=bar_w, color="#2e7d32", edgecolor=GRID_LINE,
           label=f"Green ({green_pct:.1f}%)")

    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color="white")
    ax.set_ylabel("Run count", color="white")
    _style_hist_axis(ax, step_values, red_counts, yellow_counts, green_counts)
    ax.legend(facecolor=DARK_BG, edgecolor="white", labelcolor="white", loc="upper right")


def plot_robustness_comparison_histograms(
    bpa_scenarios: list,
    hazard_scenarios: list,
    hazard_scale: float,
    save_png: bool = True,
):
    """Plot BPA-pruned and hazard-model robustness histograms in one figure."""
    if not bpa_scenarios or not hazard_scenarios:
        return

    bpa_all = sum(_steps_by_outcome_color(bpa_scenarios), [])
    haz_all = sum(_steps_by_outcome_color(hazard_scenarios), [])
    all_steps = bpa_all + haz_all
    if not all_steps:
        return

    min_step = min(all_steps)
    max_step = max(all_steps)
    step_values = list(range(min_step, max_step + 1))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
        facecolor=DARK_BG,
    )

    _plot_grouped_outcome_bars(
        ax_top,
        bpa_scenarios,
        f"BPA-Pruned Robustness: Steps To Goal ({len(bpa_scenarios)} Runs)",
        step_values,
    )
    _plot_grouped_outcome_bars(
        ax_bottom,
        hazard_scenarios,
        f"Hazard Model Robustness (penalty={hazard_scale}): Steps To Goal ({len(hazard_scenarios)} Runs)",
        step_values,
    )

    ax_bottom.set_xlabel("Car steps to goal", color="white")

    plt.tight_layout()
    if save_png:
        path = "robustness_steps_histogram_comparison.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    plt.show()

# ---------------------------------------------------------------------------
# Car movement
# ---------------------------------------------------------------------------

def car_apply_action(state: tuple, action: tuple) -> tuple:
    """
    Apply (delta_speed, delta_lane) to (gx, gy, speed).
    Car always faces East; advances gx by new_speed each timestep.
    """
    gx, gy, speed = state
    ds, dl = action

    new_speed = max(SPEED_MIN, min(SPEED_MAX, speed + ds))
    # If the car is stopped, keep lane fixed for this step.
    if speed == 0 and dl != 0:
        new_gy = gy
    else:
        new_gy = max(min(EASTBOUND_LANES), min(max(EASTBOUND_LANES), gy + dl))
    new_gx    = min(gx + new_speed, GRID_W - 1)

    # Revert lane change if it lands off-road mid-advance
    if new_speed > 0 and not is_valid(new_gx, new_gy):
        new_gy = gy

    return (new_gx, new_gy, new_speed)

# ---------------------------------------------------------------------------
# Scooter movement
# ---------------------------------------------------------------------------

def _sc_apply(state: tuple, action: int) -> tuple:
    direction, gx, gy = state
    direction = Direction(direction)
    if   action == 0: new_dir = direction
    elif action == 1: new_dir = turn_left(direction)
    elif action == 2: new_dir = turn_right(direction)
    dx, dy = DIR_DELTA[new_dir]
    ngx = gx + (dx if action == 0 else 0)
    ngy = gy + (dy if action == 0 else 0)
    if not is_valid(ngx, ngy):
        ngx, ngy = gx, gy
    return (new_dir, ngx, ngy)

def scooter_sample(state: tuple, intended: int) -> tuple:
    """Deterministic scooter transition."""
    return _sc_apply(state, intended)


def scooter_sample_stochastic(state: tuple, intended: int, rng: random.Random) -> tuple:
    """Sample straight-step scooter deviation from the transition kernel."""
    if intended != 0:
        return _sc_apply(state, intended)

    direction, gx, gy = state
    direction = Direction(direction)
    dx, dy = DIR_DELTA[direction]
    ahead_x, ahead_y = gx + dx, gy + dy

    if rng.random() < STOCHASTIC_PERCENTAGE:
        offset_x, offset_y = _sample_kernel_offset(rng, SCOOTER_TRANSITION_KERNEL)
    else:
        offset_x = 0; offset_y = 0
    dev_x, dev_y = _rotate_local(offset_x, offset_y, direction)
    ngx, ngy = ahead_x + dev_x, ahead_y + dev_y
    if not is_valid(ngx, ngy):
        ngx, ngy = gx, gy
    return (direction, ngx, ngy)

# ---------------------------------------------------------------------------
# Scooter cone
# ---------------------------------------------------------------------------

def convolve_kernels(k1: dict, k2: dict) -> dict:
    """
    k1, k2: {(ix, iy): weight}, centred at (0,0).
    Returns {(ix, iy): weight} for (k1 * k2).
    """
    out = defaultdict(float)
    for (x1, y1), w1 in k1.items():
        for (x2, y2), w2 in k2.items():
            out[(x1 + x2, y1 + y2)] += w1 * w2
    return dict(out)

def build_depth_kernels(max_depth: int) -> list[dict]:
    """
    Returns kernels[d] = dict for depth d (1..max_depth), with
    kernels[d] = K convolved with itself d times.
    """
    # Base 3x3 kernel centred at (0,0)
    base = SCOOTER_TRANSITION_KERNEL
    kernels = [None]  # index 0 unused
    current = base
    for d in range(1, max_depth + 1):
        if d > 1:
            current = convolve_kernels(current, base)
        kernels.append(current)
    kernels.append(current)
    return kernels

def build_scooter_cone(
    scooter_state: tuple,
    scooter_path: list,
    max_depth=CONE_DEPTH,
    min_prob: float = 0.01,
) -> dict:
    """
    Returns {(gx, gy): (prob, depth)}.
    At depth d, uses K convolved with itself d times (support (2d+1)x(2d+1)).
    """
    depth_kernels = build_depth_kernels(max_depth)
    cone: dict = {}
    state = scooter_state

    for d in range(1, max_depth + 1):
        facing, gx, gy = state
        facing = Direction(facing)

        # One step ahead of scooter at this depth
        adx, ady = DIR_DELTA[facing]
        ahead_x, ahead_y = gx + adx, gy + ady

        kernel_d = depth_kernels[d]
        # Collect world cells for this depth
        depth_probs = {}

        for (ix, iy), w in kernel_d.items():
            if w <= 0.0:
                continue
            # (ix, iy) is local offset around ahead cell, before rotation
            wdx, wdy = _rotate_local(ix, iy, facing)
            nx, ny = ahead_x + wdx, ahead_y + wdy
            if not is_valid(nx, ny):
                continue
            depth_probs[(nx, ny)] = depth_probs.get((nx, ny), 0.0) + w

        if not depth_probs:
            # advance scooter and continue
            if d < len(scooter_path):
                intended = scooter_path[d - 1]
                state = _sc_apply(state, intended)
            continue

        # Renormalize this depth slice
        total_mass = sum(depth_probs.values())
        for (nx, ny), w in depth_probs.items():
            prob = w / total_mass
            if prob < min_prob:
                continue
            cell = (nx, ny)
            if cell not in cone:
                cone[cell] = (prob, d)
            elif cone[cell][1] > d:  # earlier depth wins
                cone[cell] = (prob, d)
            elif cone[cell][1] == d:
                cone[cell] = (max(cone[cell][0], prob), d)

        # advance scooter for next depth
        if d < len(scooter_path):
            intended = scooter_path[d - 1]
            state = _sc_apply(state, intended)

    return cone


def get_children(state: tuple, cone: dict = None, tree_depth: int = 0) -> list:
    """
    Return valid (delta_speed, delta_lane) actions from car state.
    """
    gx, gy, speed = state
    candidates = []

    for ds, dl in CAR_ACTIONS:
        new_speed = max(SPEED_MIN, min(SPEED_MAX, speed + ds))
        new_gy    = gy + dl

        if new_gy not in EASTBOUND_LANES:
            continue

        if new_speed > SPEED_MAX or new_speed < SPEED_MIN:
            continue

        # If the car is currently stopped, lane changes are not allowed.
        if speed == 0 and dl != 0:
            continue

        new_gx = min(gx + new_speed, GRID_W - 1)

        if not is_valid(new_gx, new_gy):
            continue

        candidates.append((ds, dl))

    # CONE PRUNING: filter candidates, including intermediate states
    if cone:
        safe_actions = []
        for action in candidates:  # candidates definitely exists here
            next_state = car_apply_action(state, action)
            ngx, ngy, _ = next_state
            
            is_hazardous = False
            
            # Check all intermediate gx values when accelerating/decelerating
            # (moving from gx to ngx along the current lane gy)
            # Also check adjacent lanes since scooter might be crossing
            for intermediate_gx in range(gx + 1, ngx + 1):
                # Check current lane and adjacent lanes
                for check_gy in [gy - 1, gy, gy + 1]:
                    if check_gy < 0 or check_gy >= GRID_H:
                        continue
                    if (intermediate_gx, check_gy) in cone:
                        cone_prob, cone_depth = cone[(intermediate_gx, check_gy)]
                        # Check current tree_depth and future tree_depths within buffer
                        for check_depth in range(tree_depth, tree_depth + COLLISION_APPROX_BUFFER + 1):
                            if abs(check_depth - cone_depth) <= COLLISION_APPROX_BUFFER and cone_prob > BPA_EPSILON:
                                is_hazardous = True
                                break
                        if is_hazardous:
                            break
                if is_hazardous:
                    break
            
            # Check all intermediate gy values when changing lanes
            # (moving from gy to ngy along the final gx ngx)
            # Also check adjacent gx positions
            if not is_hazardous:
                start_gy = min(gy, ngy)
                end_gy = max(gy, ngy)
                for intermediate_gy in range(start_gy, end_gy + 1):
                    # Check current gx and adjacent gx
                    for check_gx in [ngx - 1, ngx, ngx + 1]:
                        if check_gx < 0 or check_gx >= GRID_W:
                            continue
                        if (check_gx, intermediate_gy) in cone:
                            cone_prob, cone_depth = cone[(check_gx, intermediate_gy)]
                            # Check current tree_depth and future tree_depths within buffer
                            for check_depth in range(tree_depth, tree_depth + COLLISION_APPROX_BUFFER + 1):
                                if abs(check_depth - cone_depth) <= COLLISION_APPROX_BUFFER and cone_prob > BPA_EPSILON:
                                    is_hazardous = True
                                    break
                            if is_hazardous:
                                break
                    if is_hazardous:
                        break
            
            if not is_hazardous:
                safe_actions.append(action)
        
        # Use safe_actions if any, else fallback to original candidates
        candidates = safe_actions if safe_actions else candidates
    
    return candidates if candidates else [(0, 0)]

# ---------------------------------------------------------------------------
# BPA  (Backtracking Process Algorithm)
# ---------------------------------------------------------------------------

def _car_predecessors(target_state: tuple) -> list:
    """
    Invert car_apply_action: return all (prev_state, action) pairs that
    reach target_state in one deterministic step. Used by BPA backtracking.
    """
    tgx, tgy, tspeed = target_state
    result = []
    for prev_speed in range(SPEED_MIN, SPEED_MAX + 1):
        for prev_gy in EASTBOUND_LANES:
            for action in CAR_ACTIONS:
                ds, dl = action
                new_speed = max(SPEED_MIN, min(SPEED_MAX, prev_speed + ds))
                if new_speed != tspeed:
                    continue
                new_gy = max(min(EASTBOUND_LANES),
                             min(max(EASTBOUND_LANES), prev_gy + dl))
                if new_gy != tgy:
                    continue
                prev_gx = tgx - new_speed
                if not (0 <= prev_gx < GRID_W):
                    continue
                if new_speed == prev_speed and ds != 0:
                    continue
                if new_speed == 0 and dl != 0:
                    continue
                result.append(((prev_gx, prev_gy, prev_speed), action))
    return result


def build_bpa_hazard_map(cone: dict) -> dict:
    """
    Backtracking Process Algorithm (Algorithm 1).

    Top event: car in any cone cell at matching scooter depth.
    Backtracks BPA_DEPTH steps. Each predecessor probability =
    cone_prob / n_valid_actions (uniform MCTS prior). Paths below
    BPA_EPSILON are truncated per Algorithm 1.

    Returns { (gx, gy, speed, tree_depth) : hazard_probability in [0,1] }
    """
    hazard_map: dict = {}

    def _record(state, depth, prob):
        key = (*state, depth)
        hazard_map[key] = max(hazard_map.get(key, 0.0), prob)

    queue = deque()
    for (cgx, cgy), (cone_prob, cone_depth) in cone.items():
        if cone_prob < BPA_EPSILON:
            continue
        for spd in range(SPEED_MIN, SPEED_MAX + 1):
            top = (cgx, cgy, spd)
            _record(top, cone_depth, cone_prob)
            queue.append((top, cone_depth, cone_prob))

    for _ in range(BPA_DEPTH):
        next_q = deque()
        visited: set = set()
        while queue:
            state, depth, prob = queue.popleft()
            if depth <= 0:
                continue
            prev_depth = depth - 1
            for prev_state, action in _car_predecessors(state):
                n = max(len(get_children(prev_state)), 1)
                tp = prob / n
                if tp < BPA_EPSILON:
                    continue
                _record(prev_state, prev_depth, tp)
                key = (*prev_state, prev_depth)
                if key not in visited:
                    visited.add(key)
                    next_q.append((prev_state, prev_depth, tp))
        queue = next_q

    for k in hazard_map:
        hazard_map[k] = min(hazard_map[k], 1.0)
    return hazard_map


def is_terminal(state: tuple) -> bool:
    gx, gy, speed = state
    return gx >= GOAL_GX


def is_scooter_terminal(state: tuple) -> bool:
    """Scooter is considered done once it reaches the west map edge heading west."""
    direction, gx, gy = state
    return Direction(direction) == Direction.WEST and gx == 0


def reached_planning_horizon(depth: int, start_depth: int) -> bool:
    """Relative horizon so each MCTS call gets the full rollout budget."""
    return (depth - start_depth) >= ROLLOUT_DEPTH


# Module-level hazard map — updated each simulation step
_hazard_map: dict = {}
_active_hazard_scale: float = 10.0


def reward(state: tuple, next_state: tuple, depth: int = 0, cone: dict=None) -> float:
    """
    +1.0 at goal (GOAL_REWARD).
    -1.0 per timestep (TIMESTEP_PENALTY cost for waiting).
    All other penalties are disabled (0).
    """
    gx,  gy,  spd    = state
    ngx, ngy, nspeed = next_state
    
    if ngx >= GOAL_GX:
        return GOAL_REWARD
    lane_penalty = -LANE_PENALTY if ngy != gy else 0.0
    in_intersection = COL_LO <= ngx < COL_HI and ROW_LO <= ngy < ROW_HI
    intersection_penalty = -INTERSECTION_PENALTY if in_intersection else 0.0
    ds = abs(nspeed - spd)
    jerk_penalty = -JERK_PENALTY * max(0, ds - 1)

    return (
        -TIMESTEP_PENALTY
        + lane_penalty
        + intersection_penalty
        + jerk_penalty
    )


def reward_with_hazard(state: tuple, next_state: tuple, depth: int = 0, cone: dict=None) -> float:
    """
    Reward with hazard terms for risk-aware planning without hard pruning.
    """
    gx,  gy,  spd    = state
    ngx, ngy, nspeed = next_state
    if ngx >= GOAL_GX:
        return GOAL_REWARD

    lane_penalty = -LANE_PENALTY if ngy != gy else 0.0
    in_intersection = COL_LO <= ngx < COL_HI and ROW_LO <= ngy < ROW_HI
    intersection_penalty = -INTERSECTION_PENALTY if in_intersection else 0.0
    ds = abs(nspeed - spd)
    jerk_penalty = -JERK_PENALTY * max(0, ds - 1)

    hazard_prob = _hazard_map.get((ngx, ngy, nspeed, depth), 0.0)
    hazard_pen = -_active_hazard_scale * hazard_prob

    scooter_cutoff_penalty = 0.0
    if cone:
        for (cx, cy), (cprob, cdepth) in cone.items():
            dist = abs(cx - ngx) + abs(cy - ngy)
            time_diff = abs(cdepth - depth)
            if dist <= COLLISION_APPROX_BUFFER and time_diff <= COLLISION_APPROX_BUFFER:
                risk = cprob * (3.0 - dist) * (1.0 / (1 + time_diff))
                scooter_cutoff_penalty -= _active_hazard_scale * risk

    return (
        -TIMESTEP_PENALTY
        + lane_penalty
        + intersection_penalty
        + hazard_pen
        + scooter_cutoff_penalty
        + jerk_penalty
    )

# Module-level: reward_no_bpa for use in animate
def reward_no_bpa(state: tuple, next_state: tuple, depth: int = 0, cone: dict=None) -> float:
    """
    Same as reward(), but ignores hazard penalties (hazard_pen and scooter_cutoff_penalty).
    """
    gx,  gy,  spd    = state
    ngx, ngy, nspeed = next_state
    if ngx >= GOAL_GX:
        return GOAL_REWARD
    lane_penalty = -LANE_PENALTY if ngy != gy else 0.0
    in_intersection = COL_LO <= ngx < COL_HI and ROW_LO <= ngy < ROW_HI
    intersection_penalty = -INTERSECTION_PENALTY if in_intersection else 0.0
    ds           = abs(nspeed - spd)
    jerk_penalty = -JERK_PENALTY * max(0, ds - 1)
    hazard_prob = _hazard_map.get((ngx, ngy, nspeed, depth), 0.0)
    hazard_pen = -_active_hazard_scale * hazard_prob

    scooter_cutoff_penalty = 0.0
    if cone:
        for (cx, cy), (cprob, cdepth) in cone.items():
            dist = abs(cx - ngx) + abs(cy - ngy)
            time_diff = abs(cdepth - depth)
            if dist <= COLLISION_APPROX_BUFFER and time_diff <= COLLISION_APPROX_BUFFER:
                risk = cprob * (3.0 - dist) * (1.0 / (1 + time_diff))
                scooter_cutoff_penalty -= _active_hazard_scale * risk

    return (
        -TIMESTEP_PENALTY
        + lane_penalty
        + intersection_penalty
        + hazard_pen
        + jerk_penalty
    )

def rollout(state: tuple, prune_cone: dict, reward_cone: dict, start_depth: int, reward_fn) -> float:
    current  = state
    total    = 0.0
    discount = 1.0
    rollout_depth = start_depth
    for _ in range(ROLLOUT_DEPTH):
        if is_terminal(current):
            break
        # Pass cone and incrementing depth to match tree
        actions = get_children(current, cone=prune_cone, tree_depth=rollout_depth)
        if not actions:  # safety
            break
        action = _rollout_policy(current, actions)
        next_state = car_apply_action(current, action)
        total += discount * reward_fn(current, next_state, rollout_depth, reward_cone)
        discount *= ROLLOUT_GAMMA
        current = next_state
        rollout_depth += 1  # advance rollout depth
    return total

def _rollout_policy(state: tuple, actions: list) -> tuple:
    """Always push forward: prefer accel with no lane change.
    Fall back to any forward action, then any action.
    Avoid changing lanes during rollout (avoids the -2 penalty).
    """
    # 1st choice: accelerate straight
    accel_straight = [(ds, dl) for ds, dl in actions if ds > 0 and dl == 0]
    if accel_straight:
        return accel_straight[0]
    # 2nd choice: coast straight
    coast_straight = [(ds, dl) for ds, dl in actions if ds == 0 and dl == 0]
    if coast_straight:
        return coast_straight[0]
    # Fallback
    return random.choice(actions)

# ---------------------------------------------------------------------------
# Node + MCTS
# ---------------------------------------------------------------------------

class Node:
    def __init__(self, state, parent=None, action=None, depth=0):
        self.state    = state
        self.parent   = parent
        self.action   = action
        self.children = []
        self.visits   = 0
        self.value    = 0.0
        self.depth    = depth

    @property
    def q(self):
        return self.value / self.visits if self.visits > 0 else float("-inf")

    def ucb1(self, c):
        if self.visits == 0:
            return float("inf")
        return self.q + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        return len(self.children) == len(get_children(self.state))

    def best_child(self, c):
        return max(self.children, key=lambda n: n.ucb1(c))


def _select(node, c, cone, root_depth):
    while (
        node.is_fully_expanded()
        and not is_terminal(node.state)
        and not reached_planning_horizon(node.depth, root_depth)
    ):
        node = node.best_child(c)
    return node

def _expand(node, cone, root_depth):
    if is_terminal(node.state) or reached_planning_horizon(node.depth, root_depth):
        return node
    tried   = {ch.action for ch in node.children}
    untried = [a for a in get_children(node.state, cone, node.depth) if a not in tried]
    if not untried:
        return node
    action    = untried[0]
    new_state = car_apply_action(node.state, action)
    child     = Node(new_state, parent=node, action=action, depth=node.depth + 1)
    node.children.append(child)
    return child

def _backprop(node, r):
    while node:
        node.visits += 1
        node.value  += r
        node = node.parent

def mcts(root_state: tuple, prune_cone: dict = None, reward_cone: dict = None, tree_depth: int = 0,
         num_iter: int = MCTS_ITERATIONS, c: float = EXPLORATION, reward_fn=reward) -> tuple:
    root = Node(root_state, depth=tree_depth)
    for i in range(num_iter):
        leaf  = _select(root, c, prune_cone, root.depth)
        child = _expand(leaf, prune_cone, root.depth)
        # Include immediate transition reward + discounted rollout reward
        transition_r = reward_fn(leaf.state, child.state, leaf.depth, reward_cone)
        rollout_r = rollout(child.state, prune_cone, reward_cone, child.depth, reward_fn)
        r = transition_r + ROLLOUT_GAMMA * rollout_r
        _backprop(child, r)
    
    if not root.children:
        return (0, 0)

    # Choose the action with the highest Q-value
    return max(root.children, key=lambda n: n.q).action

# ---------------------------------------------------------------------------
# Simulation  (records per-step cones for animation)
# ---------------------------------------------------------------------------

def simulate(
    max_steps: int = 40,
    prune: bool = True,
    hazard_scale: float = 10.0,
    stochastic_scooter: bool = False,
    rng_seed: int | None = None,
) -> tuple:
    """
    Returns
    -------
    car_traj     : list of (gx, gy, speed) per step
    scooter_traj : list of (dir, gx, gy)   per step
    cones        : list of cone dicts, one per step (same length as trajs)
    """
    global _active_hazard_scale
    _active_hazard_scale = hazard_scale
    rng = random.Random(rng_seed) if rng_seed is not None else random

    car_state     = CAR_START
    scooter_state = SCOOTER_START
    car_traj      = [car_state]
    scooter_traj  = [scooter_state]
    cones         = []
    car_goal_logged = False

    for step in range(max_steps):
        remaining = SCOOTER_PATH[step:] if step < len(SCOOTER_PATH) else [0]
        cone = build_scooter_cone(
            scooter_state, remaining
        )
        cones.append(cone)

        # BPA: build hazard map from current scooter cone each step
        global _hazard_map
        _hazard_map = build_bpa_hazard_map(cone)

        # Stop only when both vehicles are done so animation can show both outcomes.
        car_done = is_terminal(car_state)
        scooter_done = is_scooter_terminal(scooter_state)
        if car_done and not car_goal_logged:
            print(f"  Car reached goal at step {step}.")
            car_goal_logged = True
        if car_done and scooter_done:
            print(f"  Both vehicles reached goal at step {step}.")
            break

        if car_done:
            car_action = (0, 0)
            car_next = car_state
        else:
            if prune:
                car_action = mcts(
                    car_state,
                    prune_cone=cone,
                    reward_cone=cone,
                    tree_depth=step,
                    reward_fn=reward,
                )
            else:
                car_action = mcts(
                    car_state,
                    prune_cone=None,
                    reward_cone=cone,
                    tree_depth=step,
                    reward_fn=reward_with_hazard,
                )
            car_next = car_apply_action(car_state, car_action)

        if scooter_done:
            sc_next = scooter_state
        else:
            sc_intended = SCOOTER_PATH[step] if step < len(SCOOTER_PATH) else 0
            if stochastic_scooter:
                sc_next = scooter_sample_stochastic(scooter_state, sc_intended, rng)
            else:
                sc_next = scooter_sample(scooter_state, sc_intended)

        ds, dl = car_action
        print(f"  Step {step:3d} | "
              f"car ({car_state[0]:2d},{car_state[1]} spd={car_state[2]}) "
              f"accel={ds:+d} lane={dl:+d} "
              f"→ ({car_next[0]:2d},{car_next[1]} spd={car_next[2]}) | "
              f"scooter ({scooter_state[1]:2d},{scooter_state[2]:2d})"
              f"→ ({sc_next[1]:2d},{sc_next[2]:2d})")

        car_state     = car_next
        scooter_state = sc_next
        car_traj.append(car_state)
        scooter_traj.append(scooter_state)

    while len(cones) < len(car_traj):
        cones.append(cones[-1] if cones else {})

    return car_traj, scooter_traj, cones

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

DARK_BG     = "#0d1117"
ROAD_COL    = "#21262d"
OFFROAD_COL = "#010409"
GRID_LINE   = "#30363d"
LANE_LINE   = "#6e7681"
GOAL_COL    = "#3fb950"
CAR_COL     = "#e53935"    # red rectangle (car)
SCOOTER_COL = "#43a047"    # green dot (scooter)
CONE_COL    = "#ffb300"    # amber cone (predicted future scooter states)


def _draw_static_map(ax):
    """Draw road grid, lane markings, goal — called once before animation."""
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            fc = ROAD_COL if is_road(gx, gy) else OFFROAD_COL
            ax.add_patch(plt.Rectangle(
                (gx, gy), 1, 1,
                facecolor=fc, edgecolor=GRID_LINE, linewidth=0.3, zorder=1))

    # E-W centre dashes
    cy = (ROW_LO + ROW_HI) / 2
    for gx in range(GRID_W):
        if COL_LO <= gx < COL_HI:
            continue
        ax.plot([gx + 0.1, gx + 0.9], [cy, cy],
                color=LANE_LINE, lw=0.5, ls="--", zorder=2)

    # N-S centre dashes
    cx = (COL_LO + COL_HI) / 2
    for gy in range(GRID_H):
        if ROW_LO <= gy < ROW_HI:
            continue
        ax.plot([cx, cx], [gy + 0.1, gy + 0.9],
                color=LANE_LINE, lw=0.5, ls="--", zorder=2)

    # Goal
    ax.add_patch(plt.Rectangle(
        (GOAL_GX, min(EASTBOUND_LANES)), 1, len(EASTBOUND_LANES),
        facecolor=GOAL_COL, alpha=0.35, linewidth=0, zorder=2))
    ax.text(GOAL_GX + 0.5, min(EASTBOUND_LANES) + 1.0, "GOAL",
            ha="center", va="center",
            color=GOAL_COL, fontsize=6, fontweight="bold", zorder=3)

    # Compass
    ax.annotate("", xy=(18.5, 0.4), xytext=(18.5, 1.3),
                arrowprops=dict(arrowstyle="->", color=LANE_LINE, lw=0.8))
    ax.text(18.5, 0.15, "N", color=LANE_LINE, fontsize=6,
            ha="center", va="center")

    ax.set_xlim(0, GRID_W)
    ax.set_ylim(GRID_H, 0)   # gy=0 at top
    ax.set_aspect("equal")
    ax.tick_params(colors=LANE_LINE, labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_LINE)


def animate(car_traj: list, scooter_traj: list, cones: list,
            interval_ms: int = 500, save_gif: bool = True):
    """
    Looping frame-by-frame animation.

    Each frame shows
    ----------------
    • Amber cone cells   — opacity ∝ collision probability at this timestep
    • Red rectangle      — car (width scales with speed)
    • Green circle       — scooter
    • HUD                — step, car speed, positions
    """

    n_frames = len(car_traj)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8), facecolor=DARK_BG, gridspec_kw={'width_ratios': [2, 1]})
    ax.set_facecolor(DARK_BG)
    ax2.set_facecolor(DARK_BG)
    _draw_static_map(ax)

    # Pre-create one Rectangle per road cell for the cone overlay.
    # We toggle alpha each frame rather than adding/removing patches.
    cone_patches = {}
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            if is_road(gx, gy):
                p = plt.Rectangle(
                    (gx, gy), 1, 1,
                    facecolor=CONE_COL, edgecolor="none",
                    alpha=0.0, zorder=3)
                ax.add_patch(p)
                cone_patches[(gx, gy)] = p

    # Car — red horizontal rectangle, centred on the cell
    car_rect = plt.Rectangle(
        (0, 0), 1.0, 1.0,
        facecolor=CAR_COL, edgecolor="white",
        linewidth=1.2, zorder=6)
    ax.add_patch(car_rect)

    # Scooter — green filled circle
    scooter_dot, = ax.plot([], [], "o",
                           color=SCOOTER_COL, markersize=12,
                           markeredgecolor="white", markeredgewidth=0.8,
                           zorder=7)

    # HUD text
    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        color="white", fontsize=9, va="top", ha="left",
        fontfamily="monospace", zorder=8)

    # Legend
    car_handle  = mpatches.Patch(color=CAR_COL,     label="Car (MCTS)")
    sc_handle   = mlines.Line2D([], [], color=SCOOTER_COL, marker="o",
                                linestyle="none", markersize=9, label="Scooter")
    cone_handle = mpatches.Patch(color=CONE_COL, alpha=0.55, label="Scooter cone")
    ax.legend(handles=[car_handle, sc_handle, cone_handle],
              loc="upper left", facecolor="#161b22",
              labelcolor="white", edgecolor=GRID_LINE, fontsize=8)

    ax.set_title("MCTS Intersection — Car (W→E) vs Scooter (S→W)",
                 color="white", fontsize=11, pad=8)

    def _update(frame):
        t = frame % n_frames   # loop

        car     = car_traj[t]
        scooter = scooter_traj[t]
        cone    = cones[t]

        # -- Cone: clear all, then shade active cells ------------------------
        for p in cone_patches.values():
            p.set_alpha(0.0)
        if cone:
            max_p = max(v[0] for v in cone.values()) or 1.0
            for (cgx, cgy), (prob, _) in cone.items():
                if (cgx, cgy) in cone_patches:
                    cone_patches[(cgx, cgy)].set_alpha(
                        min(0.1 + 0.55 * prob / max_p, 0.72))

        # -- Car rectangle: width grows with speed ---------------------------
        gx, gy, speed = car
        car_rect.set_xy((gx, gy))

        # -- Scooter dot -----------------------------------------------------
        _, sx, sy = scooter
        scooter_dot.set_data([sx + 0.5], [sy + 0.5])

        # -- HUD -------------------------------------------------------------
        step_text.set_text(
            f"step {t:>2d} / {n_frames - 1}\n"
            f"car   gx={gx:>2d}  gy={gy}  spd={speed}\n"
            f"scoот gx={sx:>2d}  gy={sy}")

        # -- Action reward comparison (bar chart) ----------------------------
        ax2.clear()
        ax2.set_facecolor(DARK_BG)
        ax2.set_title("Action Rewards: With vs. Without BPA", color="white")
        ax2.set_ylabel("Immediate Reward", color="white")
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.spines['left'].set_color('white')

        # Compute available actions and rewards with BPA
        actions_bpa = get_children(car, cone=cone, tree_depth=t)
        rewards_bpa = []
        for action in actions_bpa:
            next_state = car_apply_action(car, action)
            r = reward(car, next_state, t, cone)
            rewards_bpa.append((action, r))

        # Compute available actions and rewards without BPA (cone=None, reward_no_bpa)
        actions_no_bpa = get_children(car, cone=None, tree_depth=t)
        rewards_no_bpa = []
        for action in actions_no_bpa:
            next_state = car_apply_action(car, action)
            r = reward_no_bpa(car, next_state, t, cone)
            rewards_no_bpa.append((action, r))

        # Union of all actions for x-axis
        all_actions = sorted(set([a for a, _ in rewards_bpa] + [a for a, _ in rewards_no_bpa]))
        x = range(len(all_actions))
        bpa_vals = [dict(rewards_bpa).get(a, float('nan')) for a in all_actions]
        no_bpa_vals = [dict(rewards_no_bpa).get(a, float('nan')) for a in all_actions]

        bar_width = 0.4
        ax2.bar([i - bar_width/2 for i in x], bpa_vals, width=bar_width, label='With BPA', color='#e53935', alpha=0.7)
        ax2.bar([i + bar_width/2 for i in x], no_bpa_vals, width=bar_width, label='No BPA', color='#43a047', alpha=0.7)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([str(a) for a in all_actions], rotation=45, ha='right', color='white', fontsize=8)
        ax2.legend(facecolor=DARK_BG, edgecolor='white', fontsize=8)
        ax2.grid(True, axis='y', color='#444', alpha=0.3)

        return list(cone_patches.values()) + [car_rect, scooter_dot, step_text]


    # Blitting does not work well with multiple axes/subplots (esp. bar charts)
    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,  # Disable blitting for subplot compatibility
        repeat=True,
    )

    if save_gif:
        path = "mcts_intersection.gif"
        anim.save(path, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        print(f"Saved {path}")

    plt.tight_layout()
    plt.show()
    return anim


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== MCTS Intersection: Car (W→E) vs Scooter (S→W) ===")
    print(f"Grid      : {GRID_W}×{GRID_H}  (gy=0=top, gy=19=bottom)")
    print(f"E-W road  : rows {ROW_LO}–{ROW_HI-1}  |  N-S road: cols {COL_LO}–{COL_HI-1}")
    print(f"Car start : gx={CAR_START[0]}, gy={CAR_START[1]}, speed={CAR_START[2]}"
        f"  →  goal gx={GOAL_GX}")
    print(f"Scooter   : dir=N, gx={SCOOTER_START[1]}, gy={SCOOTER_START[2]} (bottom)"
        f"  →  turns W at intersection\n")

    scenarios = []

    print("Simulating BPA-pruned baseline (no hazard penalty in reward)...")
    car_traj, scooter_traj, cones = simulate(max_steps=40, prune=True, hazard_scale=0.0)
    scenarios.append({
        "title": "BPA PRUNED (NO HAZARD PENALTY)",
        "car": car_traj,
        "scooter": scooter_traj,
        "cones": cones,
    })

    for hs in HAZARD_SCALES:
        print(f"Simulating no-prune hazard-only (hazard scale={hs})...")
        car_traj, scooter_traj, cones = simulate(max_steps=40, prune=False, hazard_scale=hs)
        scenarios.append({
            "title": f"NO PRUNE (HAZARD={hs})",
            "car": car_traj,
            "scooter": scooter_traj,
            "cones": cones,
        })

    for s in scenarios:
        final = s["car"][-1]
        goal_step = _first_goal_step(s["car"])
        car_steps = goal_step if goal_step is not None else (len(s["car"]) - 1)
        print(f"\n{s['title']}: Car steps    : {car_steps}")
        print(f"{s['title']}: Car final    : gx={final[0]}, gy={final[1]}, speed={final[2]}")
        print(f"{s['title']}: Goal reached : {final[0] >= GOAL_GX}")

    animate_four_way(scenarios, interval_ms=500, save_gif=True)

    print("\nRunning scooter robustness test with stochastic deviations (BPA-pruned)...")
    robustness_scenarios = []
    robustness_seeds = [111 * i for i in range(1, ROBUSTNESS_RUN_COUNT + 1)]
    for run_idx, seed in enumerate(robustness_seeds, start=1):
        print(f"Simulating robustness run {run_idx} (seed={seed})...")
        car_traj, scooter_traj, cones = simulate(
            max_steps=40,
            prune=True,
            hazard_scale=0.0,
            stochastic_scooter=True,
            rng_seed=seed,
        )
        robustness_scenarios.append({
            "title": f"ROBUSTNESS RUN {run_idx} (seed={seed})",
            "car": car_traj,
            "scooter": scooter_traj,
            "cones": cones,
        })

    for s in robustness_scenarios:
        final = s["car"][-1]
        goal_step = _first_goal_step(s["car"])
        car_steps = goal_step if goal_step is not None else (len(s["car"]) - 1)
        print(f"\n{s['title']}: Car steps    : {car_steps}")
        print(f"{s['title']}: Car final    : gx={final[0]}, gy={final[1]}, speed={final[2]}")
        print(f"{s['title']}: Goal reached : {final[0] >= GOAL_GX}")

    print(f"\nRunning hazard-model robustness test (penalty={ROBUSTNESS_HAZARD_SCALE})...")
    hazard_robustness_scenarios = []
    for run_idx, seed in enumerate(robustness_seeds, start=1):
        print(f"Simulating hazard robustness run {run_idx} (seed={seed})...")
        car_traj, scooter_traj, cones = simulate(
            max_steps=40,
            prune=False,
            hazard_scale=ROBUSTNESS_HAZARD_SCALE,
            stochastic_scooter=True,
            rng_seed=seed,
        )
        hazard_robustness_scenarios.append({
            "title": f"HAZARD {ROBUSTNESS_HAZARD_SCALE} RUN {run_idx} (seed={seed})",
            "car": car_traj,
            "scooter": scooter_traj,
            "cones": cones,
        })

    for s in hazard_robustness_scenarios:
        final = s["car"][-1]
        goal_step = _first_goal_step(s["car"])
        car_steps = goal_step if goal_step is not None else (len(s["car"]) - 1)
        print(f"\n{s['title']}: Car steps    : {car_steps}")
        print(f"{s['title']}: Car final    : gx={final[0]}, gy={final[1]}, speed={final[2]}")
        print(f"{s['title']}: Goal reached : {final[0] >= GOAL_GX}")

    animate_four_way(
        robustness_scenarios[:ROBUSTNESS_ANIMATED_RUNS],
        interval_ms=500,
        save_gif=True,
    )
    plot_robustness_comparison_histograms(
        robustness_scenarios,
        hazard_robustness_scenarios,
        hazard_scale=ROBUSTNESS_HAZARD_SCALE,
        save_png=True,
    )