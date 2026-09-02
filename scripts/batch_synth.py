r"""批量生成 100 个随机场景及对应 QA 数据集。

用法（仓库根目录下）：
    python scripts/batch_synth.py
"""
import os
import subprocess
import sys
from pathlib import Path

NUM_SCENES = 100
BLENDER = r"C:\Users\ASUS\blender\blender.exe"
ROOT = Path(__file__).resolve().parents[1]


def main():
    for i in range(NUM_SCENES):
        print("[%d/%d] rendering scene_%03d" % (i + 1, NUM_SCENES, i))
        env = dict(os.environ, SYNTH_SEED=str(i), SYNTH_IDX="%03d" % i)
        subprocess.run([BLENDER, "-b", "-P", "scripts/scene_generator.py"],
                       env=env, check=True, cwd=ROOT)
    print("generating QA ...")
    subprocess.run([sys.executable, "-m", "spatialforge.synth.qa_generator"],
                   check=True, cwd=ROOT)
    print("[ok] batch generation complete")


if __name__ == "__main__":
    main()