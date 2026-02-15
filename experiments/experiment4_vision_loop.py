from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mujoco
import cv2

# ---------------- Paths ----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = REPO_ROOT / "scenes" / "poc_tower.xml"

CAMERA_NAME = "cam_main"

# ---------------- Geometry ----------------
BLOCK_HALF_Z = 0.015
BLOCK_H = 2 * BLOCK_HALF_Z
TABLE_TOP_Z = 0.38 + 0.02

# ---------------- Learned Vision State ----------------
LEARNED_COLORS = None   # auto-filled on first frame


# ---------------- MuJoCo Helpers ----------------
def _freejoint_qposadr(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[bid]
    return int(model.jnt_qposadr[jadr])


def teleport_free_body(model, data, body_name, xyz):
    qadr = _freejoint_qposadr(model, body_name)
    data.qpos[qadr:qadr+3] = xyz
    data.qpos[qadr+3:qadr+7] = np.array([1, 0, 0, 0])
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def step_settle(model, data, n=60):
    for _ in range(n):
        mujoco.mj_step(model, data)


def render_rgb(model, data, camera_name, w=640, h=480):
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    renderer.close()
    return img


# ---------------- Vision: Auto-Calibrate Colors ----------------
def learn_block_colors(rgb):
    """
    Learn colors ONLY near tower workspace to avoid table dominance.
    """

    h, w = rgb.shape[:2]

    # ---- Crop to central workspace (where tower exists) ----
    cx, cy = w // 2, h // 2
    crop = rgb[cy-120:cy+120, cx-120:cx+120]

    small = cv2.resize(crop, (120, 120))
    pixels = small.reshape(-1, 3).astype(np.float32)

    # Convert to HSV to remove grey/white table pixels
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).reshape(-1,3)

    # Keep only saturated pixels (blocks are colorful, table is not)
    mask = hsv[:,1] > 60
    pixels = pixels[mask]

    if len(pixels) < 50:
        raise RuntimeError("Not enough colored pixels detected.")

    # ---- Cluster remaining colorful pixels ----
    K = 3
    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1),
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    print("🎨 Learned BLOCK colors:", centers)
    return centers



def detect_blocks(rgb):
    global LEARNED_COLORS

    if LEARNED_COLORS is None:
        LEARNED_COLORS = learn_block_colors(rgb)

    detections = {}

    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (w//2, h//2))

    for i, ref in enumerate(LEARNED_COLORS):

        # Distance from learned color
        dist = np.linalg.norm(small.astype(float) - ref[None,None,:], axis=2)

        mask = (dist < 45).astype(np.uint8) * 255

        # Clean segmentation
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
        mask = cv2.dilate(mask, None)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)

        if cv2.contourArea(c) < 150:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"]/M["m00"]) * 2
        cy = int(M["m01"]/M["m00"]) * 2

        detections[f"block_{i}"] = np.array([cx, cy])

    return detections



def order_by_height(detections):
    """
    Sort blocks bottom → top using image Y coordinate.
    """
    if not detections:
        return []

    return sorted(detections.keys(), key=lambda k: detections[k][1], reverse=True)


# ---------------- Symbolic Diagnosis ----------------
def diagnose(goal_len, tower_order):
    if len(tower_order) == goal_len:
        return {"ok": True, "error_type": "none", "next_instruction": "Done."}

    return {
        "ok": False,
        "error_type": "missing_block",
        "next_instruction": "Continue stacking blocks."
    }


# ---------------- Goal + Error Injection ----------------
def build_goal(model, data):
    heights = [TABLE_TOP_Z + BLOCK_HALF_Z + i*BLOCK_H for i in range(3)]
    names = ["block_red", "block_green", "block_yellow"]

    for name, z in zip(names, heights):
        teleport_free_body(model, data, name, np.array([0,0,z]))
        step_settle(model, data, 40)


def inject_swap(model, data):
    """
    Simulated human mistake.
    """
    teleport_free_body(model, data, "block_green", np.array([0,0,TABLE_TOP_Z + BLOCK_HALF_Z + 2*BLOCK_H]))
    teleport_free_body(model, data, "block_yellow", np.array([0,0,TABLE_TOP_Z + BLOCK_HALF_Z + 1*BLOCK_H]))
    step_settle(model, data, 80)


def apply_correction(model, data):
    """
    For PoC: rebuild correctly (simulates human following advice).
    """
    build_goal(model, data)


# ---------------- Assistant Loop ----------------
def run():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)

    build_goal(model, data)
    inject_swap(model, data)

    GOAL_LEN = 3

    for t in range(3):
        rgb = render_rgb(model, data, CAMERA_NAME)

        detections = detect_blocks(rgb)
        tower_order = order_by_height(detections)

        diag = diagnose(GOAL_LEN, tower_order)

        print(f"\n--- TURN {t} ---")
        print("detections :", detections)
        print("tower_order:", tower_order)
        print("error_type :", diag["error_type"])
        print("next       :", diag["next_instruction"])

        if diag["ok"]:
            print("✅ solved")
            break

        apply_correction(model, data)


if __name__ == "__main__":
    run()
