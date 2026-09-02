r"""G2：程序化基元场景生成与渲染（Blender headless 运行）。

用法（仓库根目录下）：
    C:\Users\ASUS\blender\blender.exe -b -P scripts/scene_generator.py

环境变量（批量生成时使用）：
    SYNTH_SEED  随机种子
    SYNTH_IDX   场景编号，如 003
"""
import json
import os
import random
from pathlib import Path

import bpy

SEED = int(os.environ.get("SYNTH_SEED", "0"))
IDX = os.environ.get("SYNTH_IDX", "000")
BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "outputs" / "synth"

COLORS = {
    "red": (0.8, 0.05, 0.05, 1),
    "blue": (0.05, 0.1, 0.8, 1),
    "green": (0.05, 0.6, 0.1, 1),
    "yellow": (0.9, 0.8, 0.05, 1),
    "purple": (0.5, 0.1, 0.7, 1),
    "cyan": (0.05, 0.7, 0.7, 1),
}
SHAPES = ["cube", "sphere", "cylinder"]
SLOTS = [(-1.2, 0.6), (1.1, 0.9), (-0.4, -0.8), (0.9, -1.1)]
SIZES = [0.6, 0.8, 1.0]


def clean():
    """清空默认场景。"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def add_obj(rng, shape, color, loc, name):
    """添加一个基元并返回其场景图记录。"""
    s = rng.choice(SIZES)
    if shape == "cube":
        bpy.ops.mesh.primitive_cube_add(size=s, location=(loc[0], loc[1], s / 2))
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=s / 2, location=(loc[0], loc[1], s / 2))
    else:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=s / 2, depth=s, location=(loc[0], loc[1], s / 2))
    obj = bpy.context.active_object
    obj.name = name
    obj.color = COLORS[color]
    return {"name": name, "shape": shape, "color": color,
            "location": [loc[0], loc[1], s / 2], "size": s}


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean()

    # 地面
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    bpy.context.active_object.color = (0.85, 0.85, 0.85, 1)

    # 物体（槽位+抖动防重叠，形状/颜色/尺寸随机）
    graph = []
    for i, slot in enumerate(SLOTS):
        jx = slot[0] + rng.uniform(-0.25, 0.25)
        jy = slot[1] + rng.uniform(-0.25, 0.25)
        graph.append(add_obj(rng, rng.choice(SHAPES),
                             rng.choice(list(COLORS)), (jx, jy), "obj%d" % i))

    # 相机 + 朝向约束
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (0, -6, 3)
    empty = bpy.data.objects.new("target", None)
    bpy.context.collection.objects.link(empty)
    empty.location = (0, 0, 0.4)
    con = cam.constraints.new("TRACK_TO")
    con.target = empty
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam

    # Workbench 渲染设置
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.color_type = "OBJECT"
    sc.render.resolution_x = 512
    sc.render.resolution_y = 512
    sc.render.filepath = str(OUT_DIR / ("scene_%s_view0.png" % IDX))
    bpy.ops.render.render(write_still=True)

    (OUT_DIR / ("scene_%s.json" % IDX)).write_text(
        json.dumps({"seed": SEED, "camera": [0, -6, 3], "objects": graph},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] rendered scene_%s_view0.png" % IDX)


if __name__ == "__main__":
    main()