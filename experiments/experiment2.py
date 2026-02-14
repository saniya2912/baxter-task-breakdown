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

# Block size: in XML half-height=0.015 => full height=0.03
BLOCK_HALF_Z = 0.015
BLOCK_H = 2 * BLOCK_HALF_Z

# Table top: table body z=0.38, top geom half-thickness=0.02
TABLE_TOP_Z = 0.38 + 0.02

# Tower XY location (center of table)
TOWER_XY = np.array([0.0, 0.0], dtype=float)

BLOCKS = ["block_red", "block_green", "block_yellow"]


@dataclass
class Step:
    idx: int
    block_name: str
    target_xyz: np.ndarray


def _freejoint_qposadr(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body not found: {body_name}")
    jadr = model.body_jntadr[bid]
    if jadr < 0:
        raise ValueError(f"Body {body_name} has no joint (expected freejoint).")
    return int(model.jnt_qposadr[jadr])  # freejoint => qpos[qadr:qadr+7]


def teleport_free_body(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, xyz: np.ndarray) -> None:
    qadr = _freejoint_qposadr(model, body_name)
    data.qpos[qadr:qadr+3] = xyz
    data.qpos[qadr+3:qadr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # identity quat
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


def tower_order_bottom_to_top(model: mujoco.MjModel, data: mujoco.MjData, blocks: list[str]) -> list[str]:
    z_bn = []
    for bn in blocks:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        z_bn.append((float(data.xpos[bid][2]), bn))
    z_bn.sort(key=lambda t: t[0])
    return [bn for _, bn in z_bn]


def dump_state(model: mujoco.MjModel, data: mujoco.MjData, blocks: list[str]) -> dict:
    out = {"blocks": []}
    for bn in blocks:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        pos = data.xpos[bid].copy().tolist()
        out["blocks"].append({"name": bn, "xyz": [round(x, 4) for x in pos]})
    return out


def diagnose_and_correct(goal_order: list[str], current_order: list[str]) -> dict:
    """
    Structured diagnosis + minimal correction plan.
    """
    result = {
        "ok": current_order == goal_order,
        "goal_order": goal_order,
        "current_order": current_order,
        "error_type": None,
        "message": "",
        "correction_steps": [],
        "next_instruction": ""
    }

    if result["ok"]:
        result["error_type"] = "none"
        result["message"] = "Tower matches goal."
        result["next_instruction"] = "Done."
        return result

    # Find first mismatch
    mismatch_idx = None
    for i in range(min(len(goal_order), len(current_order))):
        if goal_order[i] != current_order[i]:
            mismatch_idx = i
            break

    if mismatch_idx is None:
        result["error_type"] = "unknown"
        result["message"] = "Tower mismatch but could not localize."
        result["next_instruction"] = "Please rebuild the tower to match the goal."
        return result

    expected = goal_order[mismatch_idx]
    found = current_order[mismatch_idx]

    # Detect swap/order error
    if expected in current_order:
        exp_pos = current_order.index(expected)
        if exp_pos != mismatch_idx:
            # likely a swap between mismatch_idx and exp_pos
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


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    # Goal definition (bottom->top)
    goal_colors = ["red", "green", "yellow"]
    goal_order = [f"block_{c}" for c in goal_colors]
    plan = make_goal_plan(goal_colors)

    print("\n=== GOAL ===")
    print("Goal tower (bottom->top):", goal_order)

    print("\n=== BUILD GOAL (teleport place) ===")
    # Start with scattered blocks
    scatter = {
        "block_red":    np.array([-0.20, -0.20, 0.65]),
        "block_green":  np.array([-0.25, -0.20, 0.65]),
        "block_yellow": np.array([-0.30, -0.20, 0.65]),
    }
    for bn, xyz in scatter.items():
        teleport_free_body(model, data, bn, xyz)
    step_settle(model, data, 40)

    # Build correct goal
    for s in plan:
        teleport_free_body(model, data, s.block_name, s.target_xyz)
        step_settle(model, data, 40)
        print(f"Placed: {s.block_name}")

    # Save goal-built evidence
    img_goal = render_rgb(model, data, CAMERA_NAME)
    save_png(OUT_DIR / "exp2_goal_built.png", img_goal)
    (OUT_DIR / "exp2_goal_state.json").write_text(json.dumps(dump_state(model, data, BLOCKS), indent=2))
    print(f"\nSaved: {OUT_DIR/'exp2_goal_built.png'}")
    print(f"Saved: {OUT_DIR/'exp2_goal_state.json'}")

    # Inject a human-like error: swap top two blocks (green <-> yellow)
    print("\n=== INJECT ERROR: swap green and yellow ===")
    pos_green = plan[1].target_xyz.copy()   # level 2
    pos_yellow = plan[2].target_xyz.copy()  # level 3
    teleport_free_body(model, data, "block_green", pos_yellow)
    teleport_free_body(model, data, "block_yellow", pos_green)
    step_settle(model, data, 80)

    # Save error evidence
    img_err = render_rgb(model, data, CAMERA_NAME)
    save_png(OUT_DIR / "exp2_with_error.png", img_err)
    (OUT_DIR / "exp2_error_state.json").write_text(json.dumps(dump_state(model, data, BLOCKS), indent=2))
    print(f"Saved: {OUT_DIR/'exp2_with_error.png'}")
    print(f"Saved: {OUT_DIR/'exp2_error_state.json'}")

    # Observe state + diagnose
    current_order = tower_order_bottom_to_top(model, data, BLOCKS)
    diagnosis = diagnose_and_correct(goal_order, current_order)

    print("\n=== OBSERVED CURRENT ORDER ===")
    print(current_order)

    print("\n=== DIAGNOSIS ===")
    print(json.dumps(diagnosis, indent=2))

    (OUT_DIR / "exp2_diagnosis.json").write_text(json.dumps(diagnosis, indent=2))
    print(f"\nSaved: {OUT_DIR/'exp2_diagnosis.json'}")


if __name__ == "__main__":
    main()
