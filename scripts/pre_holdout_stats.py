r"""从训练前结果缓存计算 holdout(场景>=80) 子集指标。"""
import json
from collections import defaultdict
from pathlib import Path

RES = Path("outputs/synth_results.jsonl")

st = defaultdict(lambda: [0, 0])
for line in open(RES, encoding="utf-8"):
    r = json.loads(line)
    if int(r["id"].split("_")[1]) < 80:
        continue
    ok = int(r["pred"] == r["gt"])
    for d in r["dims"] + ["overall"]:
        st[d][1] += 1
        st[d][0] += ok
for k, (c, t) in sorted(st.items()):
    print(f"{k}: {c}/{t} = {100 * c / t:.1f}")