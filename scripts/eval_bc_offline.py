"""Phase E -- offline embodied-BC evaluation: pretrained base vs BC checkpoint.

Computes, on the held-out BC validation manifest: overall action accuracy,
per-action accuracy, invalid-output rate and Done precision/false-Done tendency
for a greedy single-token action decision.

    python scripts/eval_bc_offline.py \
        --val-manifest outputs/embodied_bc/dataset/bc_val.jsonl \
        --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
        [--adapter outputs/embodied_bc/train/run1/adapter] \
        --out outputs/embodied_bc/eval/offline1 --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.action_parser import parse_action_output  # noqa: E402
from spatialforge.embodied.bc_dataset import BC_ACTION_VOCAB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="outputs/embodied_bc/eval/offline")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="bc-vs-base")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from spatialforge.experiment.training import (
        format_chat_prompt,
        load_and_preprocess_image,
        load_training_records,
        run_inference_greedy,
    )

    base_dir = Path(args.val_manifest).parent
    records = load_training_records(args.val_manifest, base_dir=base_dir)
    if args.limit and args.limit < len(records):
        import random

        rng = random.Random(args.seed)
        by_house: dict = {}
        for rec in records:
            by_house.setdefault(rec.scene_id, []).append(rec)
        per = max(1, args.limit // max(1, len(by_house)))
        chosen = []
        for house in sorted(by_house):
            chosen.extend(rng.sample(by_house[house], min(per, len(by_house[house]))))
        chosen.sort(key=lambda r: r.sample_id)
        records = chosen[: args.limit]
    print(f"[eval] {len(records)} val records", flush=True)

    processor = AutoProcessor.from_pretrained(str(args.model_path))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    conf = {"action": 0, "invalid": 0}
    done_tp = done_pred = done_true = 0
    per_action = {a: {"tp": 0, "n": 0, "pred": 0} for a in BC_ACTION_VOCAB}
    t0 = time.time()
    for i, rec in enumerate(records):
        img = load_and_preprocess_image(rec.image_path)
        raw = run_inference_greedy(
            model, processor, img, rec.question, max_new_tokens=10, device="cuda"
        )
        parsed = parse_action_output(raw)
        pred = parsed["action"]
        gold = str(rec.answer)
        correct = pred == gold
        if pred is None:
            conf["invalid"] += 1
        else:
            conf["action"] += 1
            per_action[gold]["n"] += 1
            if pred == gold:
                per_action[gold]["tp"] += 1
            per_action[pred]["pred"] += 1
        if gold == "Done":
            done_true += 1
        if pred == "Done":
            done_pred += 1
            if gold == "Done":
                done_tp += 1
        rows.append({
            "id": rec.sample_id, "gold": gold, "pred": pred,
            "raw": raw, "correct": correct, "parse_reason": parsed["reason"],
        })
        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(records)} elapsed {time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0

    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    per_action_stats = {}
    for a in BC_ACTION_VOCAB:
        info = per_action[a]
        per_action_stats[a] = {
            "gold_count": info["n"],
            "gold_pct": round(info["n"] * 100.0 / max(n, 1), 2),
            "accuracy": round(info["tp"] * 100.0 / max(info["n"], 1), 2) if info["n"] else None,
            "pred_count": info["pred"],
            "precision": round(info["tp"] * 100.0 / max(info["pred"], 1), 2) if info["pred"] else None,
        }
    report = {
        "tag": args.tag,
        "model_path": str(args.model_path),
        "adapter": str(args.adapter) if args.adapter else None,
        "samples": n,
        "wall_sec": round(wall, 1),
        "samples_per_sec": round(n / max(wall, 0.001), 2),
        "overall_action_accuracy": round(correct * 100.0 / max(n, 1), 2),
        "invalid_output_rate": round(conf["invalid"] * 100.0 / max(n, 1), 2),
        "per_action": per_action_stats,
        "done": {
            "gold_done": done_true,
            "pred_done": done_pred,
            "done_precision": round(done_tp * 100.0 / max(done_pred, 1), 2) if done_pred else None,
            "done_recall": round(done_tp * 100.0 / max(done_true, 1), 2) if done_true else None,
            "false_done_rate": round((done_pred - done_tp) * 100.0 / max(n, 1), 2),
        },
    }
    with open(out_dir / f"eval_{args.tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / f"rows_{args.tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("[eval] " + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
