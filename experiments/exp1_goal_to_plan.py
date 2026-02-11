from pathlib import Path
import mujoco
import numpy as np

# --- Robust path handling (works regardless of where you run python from) ---
ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = str(ROOT / "scenes" / "exp1.xml")

# --- Grid definition (matches exp1_goal.xml) ---
TABLE_Z = 0.75                 # table top
GRID_ORIGIN_XY = (-0.15, -0.15)  # build_origin x,y in XML

# Cell size: choose something easy; for now one cell = 5cm
CELL_SIZE = 0.05

# Block height: in XML size z half-size = 0.015 -> height = 0.03
BLOCK_HALF_HEIGHT = 0.015
LAYER_HEIGHT = 2 * BLOCK_HALF_HEIGHT  # 0.03m


def parse_color_from_name(body_name: str) -> str:
    # expects names like block_red_01
    parts = body_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def round_half_up(x: float) -> int:
    """Deterministic rounding: 2.5 -> 3 (avoids NumPy bankers rounding)."""
    return int(np.floor(x + 0.5))


def pose_to_grid(xyz: np.ndarray) -> tuple[int, int, int]:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    x0, y0 = GRID_ORIGIN_XY

    cx = round_half_up((x - x0) / CELL_SIZE)
    cy = round_half_up((y - y0) / CELL_SIZE)

    # Convert z (center height) -> layer index starting at 1:
    # layer 1 center is TABLE_Z + BLOCK_HALF_HEIGHT
    z_rel = z - (TABLE_Z + BLOCK_HALF_HEIGHT)
    layer = 1 + round_half_up(z_rel / LAYER_HEIGHT)

    return cx, cy, layer


def extract_goal_graph(model: mujoco.MjModel, data: mujoco.MjData):
    graph = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name:
            continue
        if name.startswith("block_"):
            xyz = data.xpos[i].copy()  # world position of body i
            color = parse_color_from_name(name)
            cx, cy, layer = pose_to_grid(xyz)
            graph.append({
                "id": name,
                "color": color,
                "xyz": xyz,
                "cell": (cx, cy),
                "layer": layer,
            })
    return graph


def plan_from_graph(graph):
    # Sort: bottom to top (layer asc), then left->right (cx asc), then front->back (cy asc)
    sorted_blocks = sorted(graph, key=lambda b: (b["layer"], b["cell"][0], b["cell"][1], b["id"]))

    plan = []
    for step_idx, b in enumerate(sorted_blocks, start=1):
        plan.append({
            "step": step_idx,
            "place": {
                "color": b["color"],
                "cell": b["cell"],
                "layer": b["layer"],
            },
            "block_id": b["id"],
        })
    return plan


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    mujoco.mj_step(model, data)

    goal_graph = extract_goal_graph(model, data)
    print("\n--- Goal graph (GT) ---")
    for b in sorted(goal_graph, key=lambda x: x["id"]):
        print(f"{b['id']}: color={b['color']}, cell={b['cell']}, layer={b['layer']}, xyz={np.round(b['xyz'], 3)}")

    plan = plan_from_graph(goal_graph)
    print("\n--- Generated plan ---")
    for p in plan:
        pl = p["place"]
        print(f"Step {p['step']}: place {pl['color']} at cell={pl['cell']} layer={pl['layer']} (from {p['block_id']})")

    # Optional render check (confirms camera exists + render works)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="top")
    img = renderer.render()
    print("\nRendered goal image array shape:", img.shape)


if __name__ == "__main__":
    main()
