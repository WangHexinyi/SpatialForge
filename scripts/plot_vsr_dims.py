"""G3-lite：读取评测缓存，生成分维度准确率柱状图。"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("outputs/vsr_results.jsonl")
OUT = Path("outputs/vsr_dim_accuracy.png")


def main() -> None:
    stats = defaultdict(lambda: [0, 0])
    for line in open(RESULTS, encoding="utf-8"):
        r = json.loads(line)
        if r["pred"] is None:
            continue
        ok = int(r["pred"] == r["gt"])
        for d in r["dims"] + ["overall"]:
            stats[d][1] += 1
            stats[d][0] += ok

    overall = 100 * stats["overall"][0] / stats["overall"][1]
    keys = sorted((k for k in stats if k != "overall"),
                  key=lambda k: stats[k][0] / stats[k][1])
    keys += ["overall"]
    accs = [100 * stats[k][0] / stats[k][1] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d9534f" if a < 70 else "#f0ad4e" if a < 80 else "#5cb85c"
              for a in accs[:-1]] + ["#337ab7"]
    bars = ax.bar(keys, accs, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Qwen2.5-VL-3B on VSR test (overall={overall:.1f})")
    ax.axhline(overall, ls="--", c="gray", lw=0.8)
    for b, a, k in zip(bars, accs, keys):
        ax.text(b.get_x() + b.get_width() / 2, a + 1,
                f"{a:.1f}\n(n={stats[k][1]})", ha="center", fontsize=8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()