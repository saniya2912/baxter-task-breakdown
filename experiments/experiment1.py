from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

import mujoco
from mujoco import viewer


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = REPO_ROOT / "scenes" / "poc_tower.xml"
OUT_DIR = REPO_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

CAMERA_NAME = "cam_main"


# Block size: in XML half-height=0.015 => full height=0.03
BLOCK_HALF_Z = 0.015
BLOCK_H = 2 * BLOCK_HALF_Z

# Table top height: table body z=0.38, top geom half-thickness=0.02
TABLE_TOP_Z = 0.38 + 0.02

# Tower XY location (center of table)
TOWER_XY = np.array([0.0, 0.0], dtype=float)


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
    qadr = model.jnt_qposadr[jadr]  # freejoint => qpos[ qadr : qadr+7 ]
    return int(qadr)


def teleport_free_body(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, xyz: np.ndarray) -> None:
    """Teleport a free body by writing qpos (pos + unit quaternion)."""
    qadr = _freejoint_qposadr(model, body_name)
    data.qpos[qadr:qadr+3] = xyz
    data.qpos[qadr+3:qadr+7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def render_rgb(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, w=640, h=480) -> np.ndarray:
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    renderer.close()
    return img


def save_png(path: Path, img: np.ndarray) -> None:
    import imageio.v2 as imageio
    imageio.imwrite(path, img)


def make_plan(goal_colors_bottom_to_top: list[str]) -> list[Step]:
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


def main():
    if not SCENE_XML.exists():
        raise FileNotFoundError(f"Missing: {SCENE_XML}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    blocks = ["block_red", "block_green", "block_yellow"]

    # Goal (bottom->top)
    goal_colors = ["red", "green", "yellow"]
    goal_order = [f"block_{c}" for c in goal_colors]
    plan = make_plan(goal_colors)

    print("\n=== GOAL ===")
    print("Goal tower (bottom->top):", goal_order)

    print("\n=== GENERATED PLAN ===")
    for s in plan:
        print(f"Step {s.idx}: place {s.block_name} at {np.round(s.target_xyz, 3).tolist()}")

    # Scatter blocks (initial)
    scatter = {
        "block_red":    np.array([-0.20, -0.20, 0.65]),
        "block_green":  np.array([-0.25, -0.20, 0.65]),
        "block_yellow": np.array([-0.30, -0.20, 0.65]),
    }
    for bn, xyz in scatter.items():
        teleport_free_body(model, data, bn, xyz)

    # Save initial image/state
    img0 = render_rgb(model, data, CAMERA_NAME)
    save_png(OUT_DIR / "poc_initial.png", img0)
    print(f"Saved: {OUT_DIR/'poc_initial.png'}")



    (OUT_DIR / "poc_initial_state.json").write_text(json.dumps(dump_state(model, data, blocks), indent=2))
    print(f"\nSaved: {OUT_DIR/'poc_initial.png'}")
    print(f"Saved: {OUT_DIR/'poc_initial_state.json'}")

    # Execute plan by teleport placing blocks to target
    print("\n=== EXECUTION (teleport place) ===")
    for s in plan:
        teleport_free_body(model, data, s.block_name, s.target_xyz)
        for _ in range(50):
            mujoco.mj_step(model, data)
        print(f"Executed step {s.idx}: {s.block_name}")

    # Save final image/state
    img1 = render_rgb(model, data, CAMERA_NAME)
    save_png(OUT_DIR / "poc_after_plan.png", img1)
    print(f"Saved: {OUT_DIR/'poc_after_plan.png'}")


    (OUT_DIR / "poc_after_state.json").write_text(json.dumps(dump_state(model, data, blocks), indent=2))
    print(f"\nSaved: {OUT_DIR/'poc_after_plan.png'}")
    print(f"Saved: {OUT_DIR/'poc_after_state.json'}")

    # Verify
    current_order = tower_order_bottom_to_top(model, data, blocks)
    print("\n=== VERIFICATION ===")
    print("Current order (bottom->top):", current_order)
    if current_order == goal_order:
        print("✅ SUCCESS: Tower matches goal.")
    else:
        print("❌ FAIL: Tower does not match goal.")
        for i in range(min(len(goal_order), len(current_order))):
            if goal_order[i] != current_order[i]:
                print(f"First mismatch at level {i+1}: expected {goal_order[i]} got {current_order[i]}")
                break

    # Optional viewer
    if os.environ.get("NO_VIEWER", "0") != "1":
        print("\nLaunching viewer (close window to exit).")
        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                mujoco.mj_step(model, data)
                v.sync()


if __name__ == "__main__":
    main()
