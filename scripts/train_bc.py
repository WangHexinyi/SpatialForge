"""Phase D -- Qwen2.5-VL-3B BC fine-tune driver (reuses the formal harness).

Loads the BC manifests produced by scripts/run_bc_dataset.py manifest and runs
the *historical formal training loop* (same LoRA r=8/alpha=16 semantics, same
effective-batch profiles, same vision-feature cache contract) against the new
embodied-BC objective:

    pi(a_t | first-person RGB_t, goal, recent permitted actions)
        -> exactly one legal AgentAction

Run with the torch interpreter (miniconda base), NVIDIA GPU:
    python scripts/train_bc.py --train-manifest outputs/embodied_bc/dataset/bc_train.jsonl \
        --val-manifest outputs/embodied_bc/dataset/bc_val.jsonl \
        --out outputs/embodied_bc/train/run1 [--profile max_performance] \
        [--epochs 1] [--lr 1e-4] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--out", default="outputs/embodied_bc/train/run1")
    ap.add_argument("--profile", default="max_performance")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--trainability", default="A", choices=["A", "B", "C"],
                    help="A: LM LoRA (vision frozen); B: LM LoRA + projector/merger; "
                         "C: broader multimodal LoRA")
    ap.add_argument("--microbatch", type=int, default=None,
                    help="override profile microbatch (grad accum auto-derived)")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="enable gradient checkpointing (required for 8B on 32GB)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-microbatches", type=int, default=None)
    ap.add_argument("--cache-dir", default="outputs/cache/bc_vision_features")
    ap.add_argument("--max-new-tokens-eval", type=int, default=12)
    args = ap.parse_args()

    import torch
    from peft import get_peft_model

    from spatialforge.experiment.training import (
        collect_trainability_targets,
        create_lora_config,
        get_execution_profile,
        load_training_records,
        run_formal_training_loop,
    )
    from spatialforge.models.vl_adapter import load_model, load_processor, resolve_spec

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.train_manifest).parent

    records = load_training_records(args.train_manifest, base_dir=base_dir)
    print(f"[train] loaded {len(records)} BC train records", flush=True)
    val_records = []
    if args.val_manifest:
        val_records = load_training_records(args.val_manifest, base_dir=base_dir)
        print(f"[train] loaded {len(val_records)} BC val records", flush=True)

    # manifest provenance copies
    for src, dst in ((args.train_manifest, run_dir / "train_manifest.jsonl"),
                     (args.val_manifest, run_dir / "val_manifest.jsonl")):
        if src and os.path.exists(src):
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())

    t0 = time.time()
    spec = resolve_spec(str(args.model_path), attn_implementation=args.attn)
    processor = load_processor(str(args.model_path))
    model = load_model(spec)
    targets = collect_trainability_targets(model, args.trainability)
    print(f"[train] family={spec.family} trainability={args.trainability} "
          f"lora_targets={len(targets)}", flush=True)
    lora_cfg = create_lora_config(target_modules=targets)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    config = get_execution_profile(args.profile)
    config = replace(
        config,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_checkpointing=bool(args.gradient_checkpointing or config.gradient_checkpointing),
    )
    if args.microbatch:
        mb = int(args.microbatch)
        config = replace(
            config,
            per_device_train_batch_size=mb,
            gradient_accumulation_steps=max(1, config.effective_batch_size // mb),
        )
    summary = run_formal_training_loop(
        model,
        processor,
        records,
        output_dir=run_dir,
        config=config,
        max_microbatches=args.max_microbatches,
        device="cuda",
        seed=args.seed,
        group="BC",
        use_optimized_pipeline=True,
        enable_telemetry=True,
        heartbeat_interval=25,
        cache_dir=args.cache_dir,
    )
    summary["wall_time_sec"] = round(time.time() - t0, 2)
    summary["model"] = str(args.model_path)
    summary["model_spec"] = spec.to_dict()
    summary["trainability_profile"] = args.trainability
    summary["lora_target_count"] = len(targets)
    summary["train_samples"] = len(records)
    summary["val_samples"] = len(val_records)
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[train] TRAINING_SUMMARY " + json.dumps(summary, default=str))
    print(f"[train] adapter saved at {run_dir / 'adapter'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
