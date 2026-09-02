"""VSR 适配器：将官方 random/test split 转换为 SpatialForge 统一 schema。"""
import json
from pathlib import Path

from spatialforge.schema import Sample, dump_samples

RAW_TEST = Path("data/raw/vsr/data/splits/random/test.jsonl")
OUT = Path("data/processed/vsr_test.jsonl")

REL2DIM = {
    "left of": "left_right", "right of": "left_right",
    "at the left side of": "left_right", "at the right side of": "left_right",
    "at the side of": "left_right",
    "in front of": "front_back", "behind": "front_back",
    "ahead of": "front_back", "at the back of": "front_back",
    "above": "vertical", "below": "vertical", "on top of": "vertical",
    "under": "vertical", "beneath": "vertical", "over": "vertical",
    "down from": "vertical",
    "near": "near_far", "close to": "near_far", "next to": "near_far",
    "beside": "near_far", "adjacent to": "near_far", "alongside": "near_far",
    "by": "near_far", "far away from": "near_far", "far from": "near_far",
    "away from": "near_far", "beyond": "near_far",
    "in": "topology", "inside": "topology", "contains": "topology",
    "within": "topology", "enclosed by": "topology", "outside": "topology",
    "out of": "topology", "into": "topology", "surrounding": "topology",
    "around": "topology", "in the middle of": "topology",
    "at the edge of": "topology", "through": "topology",
    "on": "contact", "touching": "contact", "attached to": "contact",
    "connected to": "contact", "against": "contact", "attached": "contact",
    "detached from": "contact", "off": "contact",
    "facing": "orientation", "facing away from": "orientation",
    "toward": "orientation", "opposite to": "orientation",
    "parallel to": "orientation", "perpendicular to": "orientation",
    "across from": "orientation", "across": "orientation", "along": "orientation",
    "part of": "part_whole", "has as a part": "part_whole",
    "consists of": "part_whole",
}


def convert() -> None:
    """执行转换；未映射关系告警并归入 other。"""
    samples, unmapped = [], {}
    for i, line in enumerate(open(RAW_TEST, encoding="utf-8")):
        r = json.loads(line)
        dim = REL2DIM.get(r["relation"])
        if dim is None:
            unmapped[r["relation"]] = unmapped.get(r["relation"], 0) + 1
            dim = "other"
        samples.append(Sample(
            id=f"vsr_test_{i}", source="vsr",
            images=[f"data/vsr_images/{r['image']}"],
            question=f"Statement: {r['caption']} Is this statement true? "
                     "Answer with a single word: yes or no.",
            options=None,
            gt="yes" if r["label"] == 1 else "no",
            dims=[dim],
            meta={"image_link": r["image_link"], "relation": r["relation"]},
        ))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dump_samples(samples, OUT)
    print(f"converted={len(samples)} -> {OUT}")
    if unmapped:
        print(f"[warn] unmapped (->other): {unmapped}")


if __name__ == "__main__":
    convert()