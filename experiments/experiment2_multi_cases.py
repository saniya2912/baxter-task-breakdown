from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = REPO_ROOT / "scenes" / "poc_tower.xml"
OUT_DIR = REPO_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CAMERA_NAME = "cam_main"

BLOCK_HALF_Z = 0.015
BLOCK_H = 2 * BLOCK_HALF_Z
TABLE_TOP_Z = 0.38 + 0.02
TOWER_XY = np.array([0.0, 0.0], dtype=float)

BLOCKS = ["block_red", "block_green", "block_yellow"]

# Threshold to decide if a block is part of the tower (meters)
TOWER_RADIUS = 0.06


@dataclass
class Step:
    idx: int
    block_name: str
    target_xyz: np.ndarray


def _freejoint_qposadr(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[bid]
    return int(model.jnt_qposadr[jadr])


def teleport_free_body(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, xyz: np.ndarray) -> None:
    qadr = _freejoint_qposadr(model, body_name)
    data.qpos[qadr:qadr+3] = xyz
    data.qpos[qadr+3:qadr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def step_settle(model: mujoco.MjModel, data: mujoco.MjData, n: int = 60) -> None:
    for _ in range(n):
        mujoco.mj_step(model, data)


def render_rgb(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, w=640, h=480) -> np.ndarray:
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    renderer.close()
    return img


def save_png(path: Path, img: np.ndarray) -> None:
    import imageio.v2 as imageio
    imageio.imwrite(path, img)


def make_goal_plan(goal_colors_bottom_to_top: list[str]) -> list[Step]:
    steps: list[Step] = []
    for i, c in enumerate(goal_colors_bottom_to_top, start=1):
        z = TABLE_TOP_Z + BLOCK_HALF_Z + (i - 1) * BLOCK_H
        xyz = np.array([TOWER_XY[0], TOWER_XY[1], z], dtype=float)
        steps.append(Step(idx=i, block_name=f"block_{c}", target_xyz=xyz))
    return steps


def get_block_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def classify_tower_blocks(model: mujoco.MjModel, data: mujoco.MjData, blocks: list[str]) -> tuple[list[str], list[str]]:
    """Return (in_tower, off_tower) by XY distance to tower center."""
    in_tower, off = [], []
    for bn in blocks:
        p = get_block_pos(model, data, bn)
        d = float(np.linalg.norm(p[:2] - TOWER_XY))
        if d <= TOWER_RADIUS:
            in_tower.append(bn)
        else:
            off.append(bn)
    return in_tower, off


def tower_order_bottom_to_top(model: mujoco.MjModel, data: mujoco.MjData, tower_blocks: list[str]) -> list[str]:
    z_bn = []
    for bn in tower_blocks:
        z_bn.append((float(get_block_pos(model, data, bn)[2]), bn))
    z_bn.sort(key=lambda t: t[0])
    return [bn for _, bn in z_bn]


def dump_state(model: mujoco.MjModel, data: mujoco.MjData, blocks: list[str]) -> dict:
    out = {"blocks": []}
    for bn in blocks:
        p = get_block_pos(model, data, bn)
        out["blocks"].append({"name": bn, "xyz": [round(float(x), 4) for x in p.tolist()]})
    return out


def diagnose_and_correct(goal_order: list[str], tower_order: list[str], off_tower: list[str]) -> dict:
    """
    Diagnose using tower-only blocks. off_tower tells us what's not stacked.
    Handles:
      - missing_block
      - swap_or_order_error
      - wrong_block
      - wrong_block_and_missing (substitution)
    """
    result = {
        "ok": tower_order == goal_order,
        "goal_order": goal_order,
        "tower_order": tower_order,
        "off_tower": off_tower,
        "error_type": None,
        "message": "",
        "correction_steps": [],
        "next_instruction": ""
    }

    # If perfect, done.
    if result["ok"]:
        result["error_type"] = "none"
        result["message"] = "Tower matches goal."
        result["next_instruction"] = "Done."
        return result

    # 1) First, localize the earliest mismatch in the prefix we can see
    prefix_len = min(len(goal_order), len(tower_order))
    mismatch_idx = None
    for i in range(prefix_len):
        if goal_order[i] != tower_order[i]:
            mismatch_idx = i
            break

    # 2) If we have a mismatch in the visible prefix AND the expected block is off-tower,
    #    this is a substitution (wrong block placed while expected is missing from tower).
    if mismatch_idx is not None:
        expected = goal_order[mismatch_idx]
        found = tower_order[mismatch_idx]

        if expected in off_tower:
            result["error_type"] = "wrong_block_and_missing"
            result["message"] = (
                f"Substitution error at level {mismatch_idx+1}: expected {expected} but found {found}. "
                f"{expected} is off the tower."
            )
            result["correction_steps"] = [
                f"Remove {found} from level {mismatch_idx+1}.",
                f"Place {expected} at level {mismatch_idx+1}."
            ]
            # If the found block is still part of the goal, suggest where it should go next.
            if found in goal_order:
                correct_level = goal_order.index(found) + 1
                result["correction_steps"].append(f"Then place {found} at level {correct_level}.")
            result["next_instruction"] = (
                f"Remove {found.replace('block_', '')}, then place {expected.replace('block_', '')} "
                f"at level {mismatch_idx+1}."
            )
            return result

        # Otherwise mismatch but expected isn't off_tower -> treat as generic wrong block/order (below)

    # 3) Handle missing blocks (tower shorter than goal) when there was no substitution mismatch
    if len(tower_order) < len(goal_order):
        missing = [b for b in goal_order if b not in tower_order]
        result["error_type"] = "missing_block"
        if missing:
            m = missing[0]
            lvl = goal_order.index(m) + 1
            result["message"] = f"Missing block in tower: {m} is not stacked."
            result["correction_steps"] = [f"Add {m} to the tower at level {lvl}."]
            result["next_instruction"] = f"Place {m.replace('block_', '')} next."
        else:
            result["message"] = "Tower is incomplete (missing block)."
            result["correction_steps"] = ["Add the missing block to complete the tower."]
            result["next_instruction"] = "Add the missing block."
        return result

    # 4) If tower length matches but order differs, classify swap/order if possible
    if mismatch_idx is None:
        # Same length but mismatch_idx not found is unusual; treat as unknown
        result["error_type"] = "unknown"
        result["message"] = "Mismatch but could not localize."
        result["next_instruction"] = "Please rebuild the tower to match the goal."
        return result

    expected = goal_order[mismatch_idx]
    found = tower_order[mismatch_idx]

    # Swap/order error: expected exists elsewhere in tower
    if expected in tower_order:
        exp_pos = tower_order.index(expected)
        if exp_pos != mismatch_idx:
            result["error_type"] = "swap_or_order_error"
            result["message"] = (
                f"Order error: level {mismatch_idx+1} should be {expected} but is {found}. "
                f"{expected} is currently at level {exp_pos+1}."
            )
            result["correction_steps"] = [
                f"Move {expected} to level {mismatch_idx+1}.",
                f"Move {found} to its correct level."
            ]
            result["next_instruction"] = f"Place {expected.replace('block_', '')} at level {mismatch_idx+1}."
            return result

    # Fallback wrong block
    result["error_type"] = "wrong_block"
    result["message"] = f"Wrong block at level {mismatch_idx+1}: expected {expected}, found {found}."
    result["correction_steps"] = [
        f"Remove {found} from level {mismatch_idx+1}.",
        f"Place {expected} at level {mismatch_idx+1}."
    ]
    result["next_instruction"] = f"Place {expected.replace('block_', '')} at level {mismatch_idx+1}."
    return result



def build_goal(model, data, plan):
    scatter = {
        "block_red":    np.array([-0.20, -0.20, 0.65]),
        "block_green":  np.array([-0.25, -0.20, 0.65]),
        "block_yellow": np.array([-0.30, -0.20, 0.65]),
    }
    for bn, xyz in scatter.items():
        teleport_free_body(model, data, bn, xyz)
    step_settle(model, data, 40)

    for s in plan:
        teleport_free_body(model, data, s.block_name, s.target_xyz)
        step_settle(model, data, 40)


def run_case(case_name: str, inject_fn):
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    goal_colors = ["red", "green", "yellow"]
    goal_order = [f"block_{c}" for c in goal_colors]
    plan = make_goal_plan(goal_colors)

    build_goal(model, data, plan)

    # Save goal evidence
    img_goal = render_rgb(model, data, CAMERA_NAME)
    save_png(OUT_DIR / f"exp2v2_{case_name}_goal.png", img_goal)
    (OUT_DIR / f"exp2v2_{case_name}_goal_state.json").write_text(
        json.dumps(dump_state(model, data, BLOCKS), indent=2)
    )

    # Inject error
    inject_fn(model, data, plan)

    img_err = render_rgb(model, data, CAMERA_NAME)
    save_png(OUT_DIR / f"exp2v2_{case_name}_error.png", img_err)
    (OUT_DIR / f"exp2v2_{case_name}_error_state.json").write_text(
        json.dumps(dump_state(model, data, BLOCKS), indent=2)
    )

    # Observe using tower membership
    tower_blocks, off = classify_tower_blocks(model, data, BLOCKS)
    tower_order = tower_order_bottom_to_top(model, data, tower_blocks)
    diagnosis = diagnose_and_correct(goal_order, tower_order, off)
    diagnosis["case"] = case_name
    diagnosis["tower_blocks"] = tower_blocks

    (OUT_DIR / f"exp2v2_{case_name}_diagnosis.json").write_text(json.dumps(diagnosis, indent=2))

    print(f"\n=== CASE: {case_name} ===")
    print("Tower blocks:", tower_blocks)
    print("Off tower   :", off)
    print("Tower order :", tower_order)
    print(json.dumps(diagnosis, indent=2))


def inject_swap(model, data, plan):
    pos_green = plan[1].target_xyz.copy()
    pos_yellow = plan[2].target_xyz.copy()
    teleport_free_body(model, data, "block_green", pos_yellow)
    teleport_free_body(model, data, "block_yellow", pos_green)
    step_settle(model, data, 80)


def inject_missing_top(model, data, plan):
    # Place top block on the side (not in tower region)
    side_xyz = np.array([0.22, -0.18, TABLE_TOP_Z + BLOCK_HALF_Z], dtype=float)
    teleport_free_body(model, data, "block_yellow", side_xyz)
    step_settle(model, data, 80)


def inject_wrong_middle(model, data, plan):
    """
    Make tower: red correct, yellow occupies middle (wrong), and green is off tower.
    That creates a substitution error at level 2.
    """
    # Move green off-tower
    green_side = np.array([0.22, 0.18, TABLE_TOP_Z + BLOCK_HALF_Z], dtype=float)
    teleport_free_body(model, data, "block_green", green_side)

    # Put yellow at level 2 (middle)
    pos_level2 = plan[1].target_xyz.copy()
    teleport_free_body(model, data, "block_yellow", pos_level2)

    step_settle(model, data, 80)


def main():
    run_case("swap_top_two", inject_swap)
    run_case("missing_top", inject_missing_top)
    run_case("wrong_middle_substitution", inject_wrong_middle)

    print("\nSaved outputs as exp2v2_<case>_*.png/json in outputs/.")


if __name__ == "__main__":
    main()
