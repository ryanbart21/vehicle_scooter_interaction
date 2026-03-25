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
  Car    : West → East, eastbound lane (gy=10 or 11).
           State = (gx, gy, speed)
             gx    : 0–19
             gy    : 10 or 11  (lane change allowed)
             speed : 0–3 blocks advanced per timestep
           Actions = all combos of:
             acceleration : -1, 0, +1  (clamped to [0, 3])
             lane change  : -1, 0, +1  (gy ± 1, clamped to eastbound lanes)
           → 9 actions total (3 accel × 3 lane)

  Scooter : South → West (fixed path, unknown to car).
             Starts gy=19 (bottom), drives north in northbound lane (gx=9),
             turns left (West) at intersection, exits westward (gy=11).
             Stochastic transitions: may veer or fail to turn.

MCTS
----
  Car knows the scooter's CURRENT position (visible) but not its future path.
  The cone { (gx,gy): (probability, depth) } encodes the predicted future.
  Cone pruning goes in get_children() — implement your threshold logic there.
"""

import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from collections import deque
from typing import Optional


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
ROLLOUT_DEPTH   = 50     # Planning horizon (steps)
ROLLOUT_GAMMA   = 0.9    # Discount factor for future rewards

# --- Collision and cone ---
CONE_DEPTH              = 3     # How far to predict scooter cone
BPA_DEPTH               = 3     # BPA backtrack steps
COLLISION_APPROX_BUFFER = 4     # Cells/time for cone pruning
BPA_EPSILON             = 0.01  # BPA/collision truncation threshold

# --- Rewards and penalties (TUNED for yielding behavior, iteration 4) ---
# Hazard penalty disabled to test if car will proceed after yielding.
GOAL_REWARD          = 100.0 # Strong reward for reaching the goal
TIMESTEP_PENALTY     = 1.0  # Penalty per timestep
HAZARD_SCALE         = 10.0   # Hazard penalty disabled
LANE_PENALTY         = 10.0   # Penalty for lane change
INTERSECTION_PENALTY = 1.0   # Low penalty for being in intersection
JERK_PENALTY         = 1.0   # Penalty per unit of |ds|-1 (aggressive accel)

# --- Car setup ---
CAR_START  = (0, 10, 3)    # (gx=0, gy=10 inner eastbound lane, speed=2)
GOAL_GX    = GRID_W - 1    # 19
SPEED_MIN, SPEED_MAX = 0, 3

# --- Scooter setup ---
SCOOTER_START = (1, 11, GRID_H - 5)  # dir=N, gx=10, gy=16 (3 blocks higher)
SCOOTER_PATH  = ([0] * 6  # 10 steps north: gy 16->6 (westbound inner row)
               + [1]        # deterministic left turn: N->W
               + [0] * 13)  # forward west, exits top-left

# --- Car actions ---
CAR_ACTIONS  = [(ds, dl) for ds in (-2, -1, 0, 1, 2) for dl in (-1, 0, 1)]

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

DIR_DELTA = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}

def turn_left(d):  return (d + 1) % 4
def turn_right(d): return (d - 1) % 4

def animate_side_by_side(
    car_traj1, scooter_traj1, cones1,
    car_traj2, scooter_traj2, cones2,
    interval_ms: int = 500, save_gif: bool = True):
    """
    Show two simulations side by side:
    - Left: BPA pruning (current logic)
    - Right: No pruning (cone=None in get_children/mcts), but hazard penalties still apply
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    n_frames = min(len(car_traj1), len(car_traj2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor=DARK_BG)
    ax1.set_facecolor(DARK_BG)
    ax2.set_facecolor(DARK_BG)
    _draw_static_map(ax1)
    _draw_static_map(ax2)

    # --- Left (BPA pruning) ---
    cone_patches1 = {}
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            if is_road(gx, gy):
                p = plt.Rectangle((gx, gy), 1, 1, facecolor=CONE_COL, edgecolor="none", alpha=0.0, zorder=3)
                ax1.add_patch(p)
                cone_patches1[(gx, gy)] = p
    car_rect1 = plt.Rectangle((0, 0), 1.0, 1.0, facecolor=CAR_COL, edgecolor="white", linewidth=1.2, zorder=6)
    ax1.add_patch(car_rect1)
    scooter_dot1, = ax1.plot([], [], "o", color=SCOOTER_COL, markersize=12, markeredgecolor="white", markeredgewidth=0.8, zorder=7)
    step_text1 = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, color="white", fontsize=9, va="top", ha="left", fontfamily="monospace", zorder=8)

    # --- Right (No pruning) ---
    cone_patches2 = {}
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            if is_road(gx, gy):
                p = plt.Rectangle((gx, gy), 1, 1, facecolor=CONE_COL, edgecolor="none", alpha=0.0, zorder=3)
                ax2.add_patch(p)
                cone_patches2[(gx, gy)] = p
    car_rect2 = plt.Rectangle((0, 0), 1.0, 1.0, facecolor=CAR_COL, edgecolor="white", linewidth=1.2, zorder=6)
    ax2.add_patch(car_rect2)
    scooter_dot2, = ax2.plot([], [], "o", color=SCOOTER_COL, markersize=12, markeredgecolor="white", markeredgewidth=0.8, zorder=7)
    step_text2 = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, color="white", fontsize=9, va="top", ha="left", fontfamily="monospace", zorder=8)

    ax1.set_title("BPA PRUNING", color="white")
    ax2.set_title("NO PRUNING (hazard penalty only)", color="white")

    def _update(frame):
        t = frame % n_frames
        # --- Left (BPA pruning) ---
        car1 = car_traj1[t]
        scooter1 = scooter_traj1[t]
        cone1 = cones1[t]
        for p in cone_patches1.values():
            p.set_alpha(0.0)
        if cone1:
            max_p = max(v[0] for v in cone1.values()) or 1.0
            for (cgx, cgy), (prob, _) in cone1.items():
                if (cgx, cgy) in cone_patches1:
                    cone_patches1[(cgx, cgy)].set_alpha(min(0.1 + 0.55 * prob / max_p, 0.72))
        gx1, gy1, speed1 = car1
        car_rect1.set_xy((gx1, gy1))
        _, sx1, sy1 = scooter1
        scooter_dot1.set_data([sx1 + 0.5], [sy1 + 0.5])
        step_text1.set_text(f"step {t:>2d} / {len(car_traj1) - 1}\ncar   gx={gx1:>2d}  gy={gy1}  spd={speed1}\nscoot gx={sx1:>2d}  gy={sy1}")

        # --- Right (No pruning) ---
        car2 = car_traj2[t]
        scooter2 = scooter_traj2[t]
        cone2 = cones2[t]
        for p in cone_patches2.values():
            p.set_alpha(0.0)
        if cone2:
            max_p = max(v[0] for v in cone2.values()) or 1.0
            for (cgx, cgy), (prob, _) in cone2.items():
                if (cgx, cgy) in cone_patches2:
                    cone_patches2[(cgx, cgy)].set_alpha(min(0.1 + 0.55 * prob / max_p, 0.72))
        gx2, gy2, speed2 = car2
        car_rect2.set_xy((gx2, gy2))
        _, sx2, sy2 = scooter2
        scooter_dot2.set_data([sx2 + 0.5], [sy2 + 0.5])
        step_text2.set_text(f"step {t:>2d} / {len(car_traj2) - 1}\ncar   gx={gx2:>2d}  gy={gy2}  spd={speed2}\nscoot gx={sx2:>2d}  gy={sy2}")

        return (
            list(cone_patches1.values()) + [car_rect1, scooter_dot1, step_text1] +
            list(cone_patches2.values()) + [car_rect2, scooter_dot2, step_text2]
        )

    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_gif:
        path = "mcts_intersection.gif"
        anim.save(path, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        print(f"Saved {path}")

    plt.tight_layout()
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
    new_gy    = max(min(EASTBOUND_LANES), min(max(EASTBOUND_LANES), gy + dl))
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
    """Deterministic: scooter always executes SCOOTER_PATH exactly."""
    return _sc_apply(state, intended)

# ---------------------------------------------------------------------------
# Scooter cone
# ---------------------------------------------------------------------------

from collections import defaultdict

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
    base = {
        (0, -1): 0.125,
        (-1, 0): 0.125,
        (0, 0):  0.5,
        (1, 0):  0.125,
        (0, 1):  0.125,
    }
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

    def _rotate_local(dx: int, dy: int, facing: int) -> tuple[int, int]:
        if facing == 0:   # up
            return dx, dy
        elif facing == 1: # right
            return dy, -dx
        elif facing == 2: # down
            return -dx, -dy
        elif facing == 3: # left
            return -dy, dx
        return dx, dy

    for d in range(1, max_depth + 1):
        facing, gx, gy = state

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

        # Renormalise this depth slice
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

# ---------------------------------------------------------------------------
# ★  HOOK 1 — get_children  (add your cone pruning here)
# ---------------------------------------------------------------------------

def get_children(state: tuple, cone: dict = None, tree_depth: int = 0) -> list:
    """
    Return valid (delta_speed, delta_lane) actions from car state.
    """
    gx, gy, speed = state
    candidates = []

    # Existing candidate filtering (unchanged)
    for ds, dl in CAR_ACTIONS:
        new_speed = max(SPEED_MIN, min(SPEED_MAX, speed + ds))
        new_gy    = gy + dl

        if new_gy not in EASTBOUND_LANES:
            continue

        if new_speed > SPEED_MAX or new_speed < SPEED_MIN:
            continue

        # Allow acceleration from speed 0, even with lane change
        if speed == 0 and new_speed == 0 and dl != 0:
            continue

        new_gx = min(gx + new_speed, GRID_W - 1)

        if not is_valid(new_gx, new_gy):
            continue

        candidates.append((ds, dl))

    # Debug: print available actions if car is stuck at (16, 10, 0)
    # if gx == 16 and gy == 10 and speed == 0:
        # print(f"[DEBUG] get_children at (16,10,0): candidates = {candidates}")

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
    from collections import deque
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


# ---------------------------------------------------------------------------
# ★  HOOK 2 — Terminal
# ---------------------------------------------------------------------------

def is_terminal(state: tuple) -> bool:
    gx, gy, speed = state
    return gx >= GOAL_GX


def reached_planning_horizon(depth: int, start_depth: int) -> bool:
    """Relative horizon so each MCTS call gets the full rollout budget."""
    return (depth - start_depth) >= ROLLOUT_DEPTH

# ---------------------------------------------------------------------------
# ★  HOOK 3 — Reward
# ---------------------------------------------------------------------------

# Module-level hazard map — updated each simulation step
_hazard_map: dict = {}


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
    hazard_pen = -HAZARD_SCALE * hazard_prob

    scooter_cutoff_penalty = 0.0
    if cone:
        for (cx, cy), (cprob, cdepth) in cone.items():
            dist = abs(cx - ngx) + abs(cy - ngy)
            time_diff = abs(cdepth - depth)
            if dist <= COLLISION_APPROX_BUFFER and time_diff <= COLLISION_APPROX_BUFFER:
                risk = cprob * (3.0 - dist) * (1.0 / (1 + time_diff))
                scooter_cutoff_penalty -= HAZARD_SCALE * risk

    return (
        -TIMESTEP_PENALTY
        + lane_penalty
        + intersection_penalty
        + hazard_pen
        + jerk_penalty
    )

# ---------------------------------------------------------------------------
# ★  HOOK 4 — Rollout
# ---------------------------------------------------------------------------

def rollout(state: tuple, cone: dict, start_depth: int) -> float:
    current  = state
    total    = 0.0
    discount = 1.0
    rollout_depth = start_depth
    for _ in range(ROLLOUT_DEPTH):
        if is_terminal(current):
            break
        # PASS cone + incrementing depth to match tree!
        actions = get_children(current, cone=cone, tree_depth=rollout_depth)
        if not actions:  # safety
            break
        action = _rollout_policy(current, actions)
        next_state = car_apply_action(current, action)
        total += discount * reward(current, next_state, rollout_depth, cone)
        discount *= ROLLOUT_GAMMA
        current = next_state
        rollout_depth += 1  # advance rollout depth
    return total

def _rollout_policy(state: tuple, actions: list) -> tuple:
    """Always push forward: prefer accel with no lane change.
    Fall back to any forward action, then any action.
    Never change lanes during rollout (avoids the -2 penalty).
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

def mcts(root_state: tuple, cone: dict = None, tree_depth: int = 0,
         num_iter: int = MCTS_ITERATIONS, c: float = EXPLORATION) -> tuple:
    root = Node(root_state, depth=tree_depth)
    for i in range(num_iter):
        leaf  = _select(root, c, cone, root.depth)
        child = _expand(leaf, cone, root.depth)
        # Include immediate transition reward + discounted rollout reward
        transition_r = reward(leaf.state, child.state, leaf.depth, cone)
        rollout_r = rollout(child.state, cone, child.depth)
        r = transition_r + ROLLOUT_GAMMA * rollout_r
        _backprop(child, r)
    
    if not root.children:
        return (0, 0)

    # Choose the action with the highest Q-value
    return max(root.children, key=lambda n: n.q).action

# ---------------------------------------------------------------------------
# Simulation  (records per-step cones for animation)
# ---------------------------------------------------------------------------

def simulate(max_steps: int = 40, prune: bool = True) -> tuple:
    """
    Returns
    -------
    car_traj     : list of (gx, gy, speed) per step
    scooter_traj : list of (dir, gx, gy)   per step
    cones        : list of cone dicts, one per step (same length as trajs)
    """
    car_state     = CAR_START
    scooter_state = SCOOTER_START
    car_traj      = [car_state]
    scooter_traj  = [scooter_state]
    cones         = []

    for step in range(max_steps):
        remaining = SCOOTER_PATH[step:] if step < len(SCOOTER_PATH) else [0]
        cone = build_scooter_cone(
            scooter_state, remaining
        )
        cones.append(cone)

        # BPA: build hazard map from current scooter cone each step
        global _hazard_map
        _hazard_map = build_bpa_hazard_map(cone)

        # print(f"[DEBUG] simulate: step={step}, car_state={car_state}")
        if is_terminal(car_state):
            print(f"  Goal reached at step {step}!")
            break

        car_action  = mcts(car_state, cone=cone, tree_depth=step)
        car_next    = car_apply_action(car_state, car_action)

        sc_intended = SCOOTER_PATH[step] if step < len(SCOOTER_PATH) else 0
        sc_next     = scooter_sample(scooter_state, sc_intended)

        ds, dl = car_action
        print(f"  Step {step:3d} | "
              f"car ({car_state[0]:2d},{car_state[1]} spd={car_state[2]}) "
              f"accel={ds:+d} lane={dl:+d} "
              f"→ ({car_next[0]:2d},{car_next[1]} spd={car_next[2]}) | "
              f"scooter ({scooter_state[1]:2d},{scooter_state[2]:2d})"
              f"→ ({sc_next[1]:2d},{sc_next[2]:2d})")

        car_state     = car_next
        scooter_state = sc_next
        # print(f"[DEBUG] after action: step={step}, car_state={car_state}, car_action={car_action}")
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
CAR_COL     = "#e53935"    # red rectangle
SCOOTER_COL = "#43a047"    # green dot
CONE_COL    = "#ffb300"    # amber cone


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
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.lines as mlines

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
        w = max(0.6, min(speed + 0.8, 3.6))
        h = 0.55
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

    print("Simulating with BPA pruning (left)...")
    car_traj1, scooter_traj1, cones1 = simulate(max_steps=40, prune=True)
    print("Simulating without pruning (right)...")
    car_traj2, scooter_traj2, cones2 = simulate(max_steps=40, prune=False)

    final1 = car_traj1[-1]
    final2 = car_traj2[-1]
    print(f"\nLEFT (BPA pruning): Car steps    : {len(car_traj1)}")
    print(f"LEFT (BPA pruning): Car final    : gx={final1[0]}, gy={final1[1]}, speed={final1[2]}")
    print(f"LEFT (BPA pruning): Goal reached : {final1[0] >= GOAL_GX}")
    print(f"\nRIGHT (No pruning): Car steps    : {len(car_traj2)}")
    print(f"RIGHT (No pruning): Car final    : gx={final2[0]}, gy={final2[1]}, speed={final2[2]}")
    print(f"RIGHT (No pruning): Goal reached : {final2[0] >= GOAL_GX}")

    animate_side_by_side(car_traj1, scooter_traj1, cones1, car_traj2, scooter_traj2, cones2, interval_ms=500, save_gif=True)