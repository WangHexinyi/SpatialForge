r"""G2：从场景图自动生成带真值的空间 QA（统一 schema）。

用法（仓库根目录下）：
    python -m spatialforge.synth.qa_generator
"""
import json
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spatialforge.schema import Sample, dump_samples

SYNTH = Path("outputs/synth")
OUT = Path("data/processed/synth_test.jsonl")


def ref(o):
    """自然语言指称：颜色+形状。"""
    return "the %s %s" % (o["color"], o["shape"])


def main():
    samples = []
    for scene_file in sorted(SYNTH.glob("scene_*.json")):
        g = json.loads(scene_file.read_text(encoding="utf-8"))
        objs = g["objects"]
        sid = scene_file.stem
        img = ("outputs/synth/%s_view0.png" % sid)

        # 仅使用指称唯一的物体，杜绝歧义问题
        cnt = Counter(ref(o) for o in objs)
        ok = [o for o in objs if cnt[ref(o)] == 1]

        def dist(o):
            return (o["location"][0] ** 2 + (o["location"][1] + 6) ** 2) ** 0.5

        qi = 0
        for a, b in permutations(ok, 2):
            # 相机固定于 (0,-6,3) 朝 +Y 看：x 小=左，y 小=前，dist 小=近
            for q, gt, dim in [
                ("Is %s to the left of %s? Answer with a single word: yes or no."
                 % (ref(a), ref(b)),
                 "yes" if a["location"][0] < b["location"][0] else "no",
                 "left_right"),
                ("Is %s in front of %s? Answer with a single word: yes or no."
                 % (ref(a), ref(b)),
                 "yes" if a["location"][1] < b["location"][1] else "no",
                 "front_back"),
                ("Is %s closer to the camera than %s? "
                 "Answer with a single word: yes or no." % (ref(a), ref(b)),
                 "yes" if dist(a) < dist(b) else "no",
                 "near_far"),
            ]:
                samples.append(Sample(
                    id="%s_q%d" % (sid, qi), source="synth", images=[img],
                    question=q, options=None, gt=gt, dims=[dim],
                    meta={"scene": sid}))
                qi += 1
        for shape, n in Counter(o["shape"] for o in objs).items():
            for cand in (n, n + 1):
                samples.append(Sample(
                    id="%s_q%d" % (sid, qi), source="synth", images=[img],
                    question="Are there exactly %d %ss in the scene? "
                             "Answer with a single word: yes or no."
                             % (cand, shape),
                    options=None,
                    gt="yes" if cand == n else "no",
                    dims=["count"], meta={"scene": sid}))
                qi += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dump_samples(samples, OUT)
    print("qa_total=%d -> %s" % (len(samples), OUT))
    for s in samples[:6]:
        print(" ", s.gt, "|", s.question)


if __name__ == "__main__":
    main()