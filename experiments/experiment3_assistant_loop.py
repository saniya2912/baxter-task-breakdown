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
TOWER_RADIUS = 0.06

# Side placement spots for "removed" blocks (far from tower center)
SIDE_SPOTS = {
    "block_red":    np.array([0.25, -0.22, TABLE_TOP_Z + BLOCK_HALF_Z], dtype=float),
    "block_green":  np.array([0.25,  0.22, TABLE_TOP_Z + BLOCK_HALF_Z], dtype=float),
    "block_yellow": np.array([-0.25, -0.22, TABLE_TOP_Z + BLOCK_HALF_Z], dtype=float),
}


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
    data.qpos[qadr:qadr + 3] = xyz
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
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


def dump_state(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    out = {"blocks": []}
    for bn in BLOCKS:
        p = get_block_pos(model, data, bn)
        out["blocks"].append({"name": bn, "xyz": [round(float(x), 4) for x in p.tolist()]})
    return out


def diagnose_and_correct(goal_order: list[str], tower_order: list[str], off_tower: list[str]) -> dict:
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

    if result["ok"]:
        result["error_type"] = "none"
        result["message"] = "Tower matches goal."
        result["next_instruction"] = "Done."
        return result

    prefix_len = min(len(goal_order), len(tower_order))
    mismatch_idx = None
    for i in range(prefix_len):
        if goal_order[i] != tower_order[i]:
            mismatch_idx = i
            break

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
            if found in goal_order:
                correct_level = goal_order.index(found) + 1
                result["correction_steps"].append(f"Then place {found} at level {correct_level}.")
            result["next_instruction"] = (
                f"Remove {found.replace('block_', '')}, then place {expected.replace('block_', '')} "
                f"at level {mismatch_idx+1}."
            )
            return result

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

    if mismatch_idx is None:
        result["error_type"] = "unknown"
        result["message"] = "Mismatch but could not localize."
        result["next_instruction"] = "Please rebuild the tower to match the goal."
        return result

    expected = goal_order[mismatch_idx]
    found = tower_order[mismatch_idx]

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
        step_settle(model, data, 60)


# --- Error injections ---
def inject_swap(model, data, plan):
    pos_green = plan[1].target_xyz.copy()
    pos_yellow = plan[2].target_xyz.copy()
    teleport_free_body(model, data, "block_green", pos_yellow)
    teleport_free_body(model, data, "block_yellow", pos_green)
    step_settle(model, data, 120)


def inject_missing_top(model, data, plan):
    teleport_free_body(model, data, "block_yellow", SIDE_SPOTS["block_yellow"].copy())
    step_settle(model, data, 120)


def inject_wrong_middle(model, data, plan):
    teleport_free_body(model, data, "block_green", SIDE_SPOTS["block_green"].copy())
    teleport_free_body(model, data, "block_yellow", plan[1].target_xyz.copy())
    step_settle(model, data, 120)


# --- Robust correction: clear -> rebuild ---
def apply_correction(model, data, plan, diag: dict):
    goal_order = diag["goal_order"]

    def move_to_side(block_name: str):
        teleport_free_body(model, data, block_name, SIDE_SPOTS[block_name].copy())
        step_settle(model, data, 60)

    def place_at_level(block_name: str, level: int):
        xyz = plan[level - 1].target_xyz.copy()
        teleport_free_body(model, data, block_name, xyz)
        step_settle(model, data, 120)

    for b in BLOCKS:
        move_to_side(b)

    for lvl, b in enumerate(goal_order, start=1):
        place_at_level(b, lvl)

    step_settle(model, data, 180)


def run_loop(case_name: str, inject_fn, max_turns: int = 3) -> dict:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    goal_colors = ["red", "green", "yellow"]
    goal_order = [f"block_{c}" for c in goal_colors]
    plan = make_goal_plan(goal_colors)

    build_goal(model, data, plan)
    inject_fn(model, data, plan)

    history_path = OUT_DIR / f"exp3_{case_name}_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    final_diag = None
    solved_turn = None

    for t in range(max_turns + 1):
        tower_blocks, off = classify_tower_blocks(model, data, BLOCKS)
        tower_order = tower_order_bottom_to_top(model, data, tower_blocks)

        diag = diagnose_and_correct(goal_order, tower_order, off)
        final_diag = diag

        img = render_rgb(model, data, CAMERA_NAME)
        save_png(OUT_DIR / f"exp3_{case_name}_t{t:02d}.png", img)

        record = {
            "turn": t,
            "case": case_name,
            "tower_blocks": tower_blocks,
            "off_tower": off,
            "state": dump_state(model, data),
            "diagnosis": diag,
        }
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"\n--- {case_name} | TURN {t} ---")
        print("tower_order:", tower_order)
        print("error_type :", diag["error_type"])
        print("next       :", diag["next_instruction"])

        if diag["ok"]:
            solved_turn = t
            print("✅ solved")
            break

        apply_correction(model, data, plan, diag)

    summary = {
        "case": case_name,
        "solved": bool(final_diag and final_diag.get("ok", False)),
        "solved_turn": solved_turn,
        "final_error_type": None if final_diag is None else final_diag.get("error_type"),
        "final_message": None if final_diag is None else final_diag.get("message"),
        "history_file": str(history_path),
        "frames_glob": f"outputs/exp3_{case_name}_tXX.png"
    }
    (OUT_DIR / f"exp3_{case_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    summaries = []
    summaries.append(run_loop("swap_top_two", inject_swap, max_turns=3))
    summaries.append(run_loop("missing_top", inject_missing_top, max_turns=3))
    summaries.append(run_loop("wrong_middle_substitution", inject_wrong_middle, max_turns=3))

    (OUT_DIR / "exp3_all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print("\nSaved: outputs/exp3_<case>_summary.json and outputs/exp3_all_summaries.json")


if __name__ == "__main__":
    main()
