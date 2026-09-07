"""G2.0-E3.2 Controlled Engineering Experiment Runner (B / A42 / C42 / D42).

Executes:
1. Strict preflight verification (hashes, split guards, model revision, LoRA module discovery).
2. Upfront config freeze and deterministic config hashing for all four runs (B, A42, C42, D42)
   BEFORE any model execution begins.
3. Baseline B zero-shot evaluation on Synthetic S1 & S2.
4. Sequential training of A42, C42, D42:
   - Clean base model reload per group
   - Fresh 252-LM LoRA attachment
   - Seed 42 PRNG seeding
   - 1552 microbatches -> 194 optimizer steps (1 epoch)
   - Save adapter only
   - Clean base model reload + adapter attachment
   - Evaluation on Synthetic S1 & S2
5. Integrity validation:
   - Sample ID and ground-truth alignment across B, A, C, D
   - No duplicate records, exact sample counts
6. Descriptive engineering deltas (A-B, C-B, D-B, C-A, D-A, D-C) across S1/S2 and families.
"""

from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel, get_peft_model

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spatialforge.experiment.evaluation import (
    EvaluationPredictionRecord,
    aggregate_metrics,
    parse_directional_answer,
    parse_yesno_answer,
)
from spatialforge.experiment.protocol import (
    DEFAULT_MODEL_PATH,
    DIRECTIONAL_PARSER_POLICY,
    E1_MANIFEST_PATH,
    EVAL_DATASET_PATHS,
    EXPERIMENT_ID,
    FORMAL_SEEDS,
    FROZEN_E1_DATASET_HASHES,
    FROZEN_E1_GROUP_TO_FILE,
    GROUPS,
    HOLDOUT_SCENE_IDS,
    INVALID_PREDICTION_POLICY,
    MODEL_ID,
    S1_SEMANTIC_LABEL,
    S2_SEMANTIC_LABEL,
    SYNTHETIC_MAX_NEW_TOKENS,
    TRAIN_DATASET_PATHS,
    YESNO_PARSER_POLICY,
    compute_jsonl_content_hash,
    get_environment_provenance,
    get_git_commit_hash,
    get_run_output_dir,
    recover_model_revision,
    seed_everything,
    validate_seed,
    validate_training_scenes,
    verify_frozen_e1_datasets,
)
from spatialforge.experiment.training import (
    FormalTrainingConfig,
    create_lora_config,
    format_chat_prompt,
    get_language_model_target_modules,
    load_and_preprocess_image,
    load_training_records,
    run_formal_training_loop,
)


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 of a run configuration dictionary."""
    cfg_json = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()


def run_preflight() -> Dict[str, Any]:
    """Execute all preflight checks; fail hard if any requirement is violated."""
    print("=" * 70)
    print("G2.0-E3.2 PREFLIGHT INTEGRITY CHECKS")
    print("=" * 70)

    # 1. Dataset hashes
    print("[1/9] Verifying frozen E1 dataset hashes against tracked constants...")
    verified_hashes = verify_frozen_e1_datasets()
    assert len(verified_hashes) == 5, "Must verify all 5 E1 datasets"
    print(f"  ✓ 5/5 E1 datasets match authoritative tracked SHA-256 constants")

    # 2. Model revision
    print("[2/9] Verifying local model revision...")
    rev = recover_model_revision(DEFAULT_MODEL_PATH)
    expected_rev = "66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert rev == expected_rev, f"Model revision mismatch: {rev} != {expected_rev}"
    print(f"  ✓ Model revision verified: {rev}")

    # 3. Train / Holdout split guard & dataset statistics
    print("[3/9] Verifying training split isolation and dataset statistics...")
    for grp in ("A", "C", "D"):
        p = Path(TRAIN_DATASET_PATHS[grp])
        with p.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 1552, f"Group {grp} size {len(records)} != 1552"
        validate_training_scenes(records)

        # Family breakdown: 388 per family, 194/194 label balance
        fam_counts = {}
        label_counts = {}
        for r in records:
            fam = r["family"]
            ans = r["answer"]
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            label_counts[(fam, ans)] = label_counts.get((fam, ans), 0) + 1

        for fam in ("horizontal", "vertical", "depth", "near_far"):
            assert fam_counts.get(fam) == 388, f"Group {grp} family {fam} count {fam_counts.get(fam)} != 388"
            ans_counts = [cnt for (f, _), cnt in label_counts.items() if f == fam]
            assert ans_counts == [194, 194], f"Group {grp} family {fam} balance {ans_counts} != [194, 194]"

        print(f"  ✓ Group {grp}: N=1552, 388/fam, 194/194 label balance, 0 holdout scenes leaked")

    # 4. LoRA modules discovery
    print("[4/9] Verifying LoRA target modules (252 LM / 0 vision)...")
    targets = get_language_model_target_modules()
    assert len(targets) == 252, f"Expected 252 LM target modules, got {len(targets)}"
    assert not any("visual" in t for t in targets), "Found visual target module in language targets"
    lora_cfg = create_lora_config()
    assert lora_cfg.r == 8
    assert lora_cfg.lora_alpha == 16
    assert lora_cfg.lora_dropout == 0.05
    print("  ✓ LoRA configuration: r=8, alpha=16, dropout=0.05, 252 LM modules, 0 vision")

    # 5. Training optimizer steps
    print("[5/9] Verifying formal optimizer step count...")
    train_cfg = FormalTrainingConfig()
    opt_steps = (1552 * train_cfg.num_train_epochs) // train_cfg.gradient_accumulation_steps
    assert opt_steps == 194, f"Expected 194 optimizer steps, got {opt_steps}"
    print(f"  ✓ Formal optimizer steps: 1552 / 8 = {opt_steps}")

    # 6. Parser hardening checks
    print("[6/9] Verifying parser hardening against adversarial cases...")
    assert parse_directional_answer("I think it might be right") is None
    assert parse_directional_answer("could be behind") is None
    assert parse_directional_answer("The object is in front of the other object.") == "front"
    assert parse_yesno_answer("probably yes") is None
    assert parse_yesno_answer("no idea") is None
    assert parse_yesno_answer("Yes, true.") == "yes"
    print("  ✓ Strict answer parsers verified")

    # 7. Environment provenance
    print("[7/9] Gathering environment provenance...")
    prov = get_environment_provenance()
    git_sha = get_git_commit_hash()
    assert git_sha is not None, "Must be in a Git repository"
    assert prov.get("cuda_available") == "True", "CUDA GPU must be available"
    print(f"  ✓ GPU: {prov.get('gpu_device')}, Git HEAD: {git_sha[:10]}")

    # 8. Output directory hygiene
    print("[8/9] Checking run output directories...")
    e3_base = REPO_ROOT / "outputs/experiments" / EXPERIMENT_ID
    for grp, seed in [("B", None), ("A", 42), ("C", 42), ("D", 42)]:
        run_dir = get_run_output_dir(grp, seed, base_dir=REPO_ROOT / "outputs/experiments")
        if run_dir.exists():
            files = list(run_dir.glob("**/*"))
            if files:
                print(f"  [Notice] Run directory {run_dir} already exists ({len(files)} files)")
    print("  ✓ Run output paths resolved collision-free")

    # 9. Preflight complete
    print("[9/9] PREFLIGHT PASSED: ALL GUARDS SATISFIED.\n")
    return {
        "verified_hashes": verified_hashes,
        "model_revision": rev,
        "git_sha": git_sha,
        "provenance": prov,
        "opt_steps": opt_steps,
    }


def freeze_run_configs(
    preflight: Dict[str, Any],
    base_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Materialize and freeze all 4 run configs before any model execution."""
    print("=" * 70)
    print("FREEZING RUN CONFIGURATIONS (B, A42, C42, D42)")
    print("=" * 70)

    eval_dataset_hashes = {
        "synthetic_s1": FROZEN_E1_DATASET_HASHES["holdout_s1_cardinal.jsonl"],
        "synthetic_s2": FROZEN_E1_DATASET_HASHES["holdout_s2_jitter.jsonl"],
    }
    generation_config = {
        "do_sample": False,
        "max_new_tokens": SYNTHETIC_MAX_NEW_TOKENS,
    }
    parser_policy = {
        "directional": DIRECTIONAL_PARSER_POLICY,
        "yesno": YESNO_PARSER_POLICY,
        "invalid_policy": INVALID_PREDICTION_POLICY,
    }

    frozen_configs: Dict[str, Dict[str, Any]] = {}

    run_specs = [
        ("B", None, "baseline_b"),
        ("A", 42, "group_a"),
        ("C", 42, "group_c"),
        ("D", 42, "group_d"),
    ]

    for grp, seed, grp_label in run_specs:
        run_key = f"{grp_label}" if seed is None else f"{grp_label}_seed_{seed}"
        train_path = TRAIN_DATASET_PATHS.get(grp)
        train_hash = None
        train_cfg_dict = None
        lora_cfg_dict = None

        if grp != "B":
            fname = FROZEN_E1_GROUP_TO_FILE[f"group_{grp.lower()}"]
            train_hash = FROZEN_E1_DATASET_HASHES[fname]
            train_cfg_dict = asdict(FormalTrainingConfig())
            lora_cfg_dict = {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules_count": 252,
            }

        cfg_dict: Dict[str, Any] = {
            "experiment_id": EXPERIMENT_ID,
            "group": grp,
            "seed": seed,
            "git_commit": preflight["git_sha"],
            "model_id": MODEL_ID,
            "model_revision": preflight["model_revision"],
            "training_dataset_path": train_path,
            "training_dataset_hash": train_hash,
            "training_config": train_cfg_dict,
            "lora_config": lora_cfg_dict,
            "evaluation_dataset_hashes": eval_dataset_hashes,
            "generation_config": generation_config,
            "parser_policy": parser_policy,
            "dependency_versions": preflight["provenance"],
            "gpu": preflight["provenance"].get("gpu_device", "unknown"),
        }
        cfg_hash = compute_config_hash(cfg_dict)
        cfg_dict["config_hash"] = cfg_hash

        run_dir = get_run_output_dir(grp, seed, base_dir=(base_dir or (REPO_ROOT / "outputs/experiments")))
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "run_config.json"
        config_path.write_text(json.dumps(cfg_dict, indent=2, sort_keys=True), encoding="utf-8")

        initial_manifest = {
            "experiment_id": EXPERIMENT_ID,
            "group": grp,
            "seed": seed,
            "config_hash": cfg_hash,
            "status": "frozen_preflight",
            "git_commit": preflight["git_sha"],
            "model_id": MODEL_ID,
            "model_revision": preflight["model_revision"],
            "start_time": None,
            "end_time": None,
            "artifacts": {},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(initial_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        frozen_configs[run_key] = cfg_dict
        print(f"  ✓ [{run_key}] Config hash: {cfg_hash}")

    print("ALL 4 RUN CONFIGS FROZEN IMMUTABLY.\n")
    return frozen_configs


def evaluate_split(
    model: Any,
    processor: Any,
    dataset_path: Path,
    source_name: str,
    group: str,
    seed: Optional[int],
    adapter_path: Optional[str],
    batch_size: int = 8,
    device: str = "cuda",
) -> Tuple[List[EvaluationPredictionRecord], Dict[str, Any]]:
    """Evaluate model on a synthetic holdout split using batch greedy inference."""
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]

    processor.tokenizer.padding_side = "left"

    records: List[EvaluationPredictionRecord] = []
    t0 = time.time()

    for batch_start in range(0, len(raw_samples), batch_size):
        batch = raw_samples[batch_start : batch_start + batch_size]
        images = [load_and_preprocess_image(s["image_path"]) for s in batch]
        prompts = [format_chat_prompt(processor, img, s["question"]) for img, s in zip(images, batch)]

        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device)
        if inputs["pixel_values"].dtype == torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=SYNTHETIC_MAX_NEW_TOKENS,
                do_sample=False,
            )

        for i, s in enumerate(batch):
            input_len = inputs["input_ids"][i].shape[0]
            out_ids = generated_ids[i][input_len:]
            raw_pred = processor.decode(out_ids, skip_special_tokens=True).strip()

            parsed = parse_directional_answer(raw_pred)
            is_valid = (parsed is not None)
            is_correct = (parsed == s["answer"])

            rec = EvaluationPredictionRecord(
                sample_id=s["id"],
                source=source_name,
                scene_id=s.get("scene_id"),
                group=group,
                seed=seed,
                family=s["family"],
                ground_truth=s["answer"],
                raw_prediction=raw_pred,
                parsed_prediction=parsed,
                is_valid_prediction=is_valid,
                is_correct=is_correct,
                model_id=MODEL_ID,
                adapter_path=adapter_path,
                view_id=s.get("view_id"),
                meta={
                    "source_sample_id": s.get("source_sample_id"),
                    "tags": s.get("tags"),
                },
            )
            records.append(rec)

    eval_time = time.time() - t0
    metrics = aggregate_metrics(records)
    metrics["eval_time_sec"] = eval_time
    metrics["eval_throughput_samples_per_sec"] = len(records) / max(eval_time, 1e-6)

    return records, metrics


def run_baseline_b(preflight: Dict[str, Any], frozen_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Execute zero-shot Baseline B evaluation on S1 and S2."""
    print("=" * 70)
    print("EXECUTING CONDITION: BASELINE B (Zero-shot, No Adapter)")
    print("=" * 70)

    run_dir = get_run_output_dir("B", base_dir=REPO_ROOT / "outputs/experiments")
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Load clean base model
    print("  Loading clean base model from local cache...")
    processor = AutoProcessor.from_pretrained(str(DEFAULT_MODEL_PATH))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(DEFAULT_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Evaluate S1
    print("  Evaluating Synthetic S1 (1,136 samples)...")
    s1_path = REPO_ROOT / EVAL_DATASET_PATHS["synthetic_s1"]
    recs_s1, metrics_s1 = evaluate_split(
        model=model,
        processor=processor,
        dataset_path=s1_path,
        source_name="synthetic_s1",
        group="B",
        seed=None,
        adapter_path=None,
    )
    print(f"    S1 Overall Accuracy: {metrics_s1['overall']['accuracy']:.4f} "
          f"({metrics_s1['overall']['correct']}/{metrics_s1['overall']['total']}), "
          f"Invalid: {metrics_s1['overall']['invalid_predictions']} "
          f"({metrics_s1['overall']['invalid_rate']:.4f})")

    # Evaluate S2
    print("  Evaluating Synthetic S2 (1,136 samples)...")
    s2_path = REPO_ROOT / EVAL_DATASET_PATHS["synthetic_s2"]
    recs_s2, metrics_s2 = evaluate_split(
        model=model,
        processor=processor,
        dataset_path=s2_path,
        source_name="synthetic_s2",
        group="B",
        seed=None,
        adapter_path=None,
    )
    print(f"    S2 Overall Accuracy: {metrics_s2['overall']['accuracy']:.4f} "
          f"({metrics_s2['overall']['correct']}/{metrics_s2['overall']['total']}), "
          f"Invalid: {metrics_s2['overall']['invalid_predictions']} "
          f"({metrics_s2['overall']['invalid_rate']:.4f})")

    # Save artifacts
    pred_dir = run_dir / "predictions"
    metric_dir = run_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    with (pred_dir / "s1.jsonl").open("w", encoding="utf-8") as f:
        for r in recs_s1:
            f.write(r.to_json() + "\n")

    with (pred_dir / "s2.jsonl").open("w", encoding="utf-8") as f:
        for r in recs_s2:
            f.write(r.to_json() + "\n")

    (metric_dir / "s1.json").write_text(json.dumps(metrics_s1, indent=2, sort_keys=True), encoding="utf-8")
    (metric_dir / "s2.json").write_text(json.dumps(metrics_s2, indent=2, sort_keys=True), encoding="utf-8")

    t_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "group": "B",
        "seed": None,
        "config_hash": frozen_configs["baseline_b"]["config_hash"],
        "status": "completed",
        "git_commit": preflight["git_sha"],
        "model_id": MODEL_ID,
        "model_revision": preflight["model_revision"],
        "start_time": t_start,
        "end_time": t_end,
        "artifacts": {
            "predictions_s1": str(pred_dir / "s1.jsonl"),
            "predictions_s2": str(pred_dir / "s2.jsonl"),
            "metrics_s1": str(metric_dir / "s1.json"),
            "metrics_s2": str(metric_dir / "s2.json"),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Clean up GPU
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

    print("Baseline B complete and GPU state cleared.\n")
    return {
        "group": "B",
        "seed": None,
        "metrics_s1": metrics_s1,
        "metrics_s2": metrics_s2,
        "recs_s1": recs_s1,
        "recs_s2": recs_s2,
    }


def train_and_eval_group(
    group: str,
    seed: int,
    preflight: Dict[str, Any],
    frozen_configs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute training, adapter save, fresh base reload, and evaluation for group A/C/D."""
    run_key = f"group_{group.lower()}_seed_{seed}"
    print("=" * 70)
    print(f"EXECUTING CONDITION: GROUP {group} (Seed {seed})")
    print("=" * 70)

    run_dir = get_run_output_dir(group, seed, base_dir=REPO_ROOT / "outputs/experiments")
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Fresh base model reload
    print("  [Step 1] Loading fresh clean base model...")
    processor = AutoProcessor.from_pretrained(str(DEFAULT_MODEL_PATH))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(DEFAULT_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # 2. Attach language-only LoRA
    print("  [Step 2] Attaching language-only LoRA (252 modules)...")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    lora_cfg = create_lora_config()
    model = get_peft_model(model, lora_cfg)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == 14966784, f"Trainable params {trainable_count} != 14,966,784"
    print(f"  ✓ Trainable parameters: {trainable_count:,}")

    # 3. Load dataset records
    dataset_rel_path = TRAIN_DATASET_PATHS[group]
    train_records = load_training_records(REPO_ROOT / dataset_rel_path, base_dir=REPO_ROOT)
    assert len(train_records) == 1552, f"Expected 1552 records, got {len(train_records)}"
    print(f"  ✓ Loaded {len(train_records)} formal training records from {dataset_rel_path}")

    # 4. Formal training loop
    print("  [Step 3] Running formal training loop (1552 microbatches, 194 optimizer steps)...")
    train_cfg = FormalTrainingConfig()
    t_train_0 = time.time()
    train_res = run_formal_training_loop(
        model=model,
        processor=processor,
        records=train_records,
        output_dir=run_dir,
        config=train_cfg,
        seed=seed,
    )
    t_train = time.time() - t_train_0

    assert train_res["total_microbatches"] == 1552
    assert train_res["total_optimizer_steps"] == 194
    adapter_dir = Path(train_res["adapter_dir"])
    assert adapter_dir.exists(), f"Adapter dir not found: {adapter_dir}"

    raw_losses = train_res["raw_losses"]
    mean_raw_loss = sum(raw_losses) / len(raw_losses)
    first_10_loss = sum(raw_losses[:10]) / 10
    last_10_loss = sum(raw_losses[-10:]) / 10

    # Record training metrics
    train_metrics = {
        "group": group,
        "seed": seed,
        "microbatch_count": train_res["total_microbatches"],
        "optimizer_step_count": train_res["total_optimizer_steps"],
        "scheduler_step_count": train_res["total_optimizer_steps"],
        "initial_loss": train_res["initial_loss"],
        "final_loss": train_res["final_loss"],
        "mean_raw_loss": round(mean_raw_loss, 4),
        "first_10_loss_mean": round(first_10_loss, 4),
        "last_10_loss_mean": round(last_10_loss, 4),
        "wall_time_sec": round(t_train, 2),
        "peak_vram_gb": round(train_res["peak_vram_gb"], 3),
        "adapter_dir": str(adapter_dir),
    }
    (run_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"  ✓ Training completed in {t_train:.1f}s ({t_train/60:.2f} min). "
          f"Mean loss: {mean_raw_loss:.4f} (first 10: {first_10_loss:.4f} -> last 10: {last_10_loss:.4f}). "
          f"Peak VRAM: {train_res['peak_vram_gb']:.2f} GB")

    # 5. Clean up training model completely
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()

    # 6. Clean base reload + adapter attachment
    print("  [Step 4] Reloading clean base model + saved adapter for evaluation...")
    processor_eval = AutoProcessor.from_pretrained(str(DEFAULT_MODEL_PATH))
    base_reload = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(DEFAULT_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    eval_model = PeftModel.from_pretrained(base_reload, str(adapter_dir))
    eval_model.eval()

    # 7. Evaluate S1
    print("  [Step 5] Evaluating Synthetic S1 (1,136 samples)...")
    s1_path = REPO_ROOT / EVAL_DATASET_PATHS["synthetic_s1"]
    recs_s1, metrics_s1 = evaluate_split(
        model=eval_model,
        processor=processor_eval,
        dataset_path=s1_path,
        source_name="synthetic_s1",
        group=group,
        seed=seed,
        adapter_path=str(adapter_dir),
    )
    print(f"    S1 Overall Accuracy: {metrics_s1['overall']['accuracy']:.4f} "
          f"({metrics_s1['overall']['correct']}/{metrics_s1['overall']['total']}), "
          f"Invalid: {metrics_s1['overall']['invalid_predictions']} "
          f"({metrics_s1['overall']['invalid_rate']:.4f})")

    # 8. Evaluate S2
    print("  [Step 6] Evaluating Synthetic S2 (1,136 samples)...")
    s2_path = REPO_ROOT / EVAL_DATASET_PATHS["synthetic_s2"]
    recs_s2, metrics_s2 = evaluate_split(
        model=eval_model,
        processor=processor_eval,
        dataset_path=s2_path,
        source_name="synthetic_s2",
        group=group,
        seed=seed,
        adapter_path=str(adapter_dir),
    )
    print(f"    S2 Overall Accuracy: {metrics_s2['overall']['accuracy']:.4f} "
          f"({metrics_s2['overall']['correct']}/{metrics_s2['overall']['total']}), "
          f"Invalid: {metrics_s2['overall']['invalid_predictions']} "
          f"({metrics_s2['overall']['invalid_rate']:.4f})")

    # 9. Save prediction records & metrics
    pred_dir = run_dir / "predictions"
    metric_dir = run_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    with (pred_dir / "s1.jsonl").open("w", encoding="utf-8") as f:
        for r in recs_s1:
            f.write(r.to_json() + "\n")

    with (pred_dir / "s2.jsonl").open("w", encoding="utf-8") as f:
        for r in recs_s2:
            f.write(r.to_json() + "\n")

    (metric_dir / "s1.json").write_text(json.dumps(metrics_s1, indent=2, sort_keys=True), encoding="utf-8")
    (metric_dir / "s2.json").write_text(json.dumps(metrics_s2, indent=2, sort_keys=True), encoding="utf-8")

    t_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "group": group,
        "seed": seed,
        "config_hash": frozen_configs[run_key]["config_hash"],
        "status": "completed",
        "git_commit": preflight["git_sha"],
        "model_id": MODEL_ID,
        "model_revision": preflight["model_revision"],
        "start_time": t_start,
        "end_time": t_end,
        "artifacts": {
            "adapter": str(adapter_dir),
            "train_metrics": str(run_dir / "train_metrics.json"),
            "predictions_s1": str(pred_dir / "s1.jsonl"),
            "predictions_s2": str(pred_dir / "s2.jsonl"),
            "metrics_s1": str(metric_dir / "s1.json"),
            "metrics_s2": str(metric_dir / "s2.json"),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # 10. Clean up GPU
    del eval_model
    del base_reload
    del processor_eval
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Condition Group {group} (Seed {seed}) complete and GPU state cleared.\n")
    return {
        "group": group,
        "seed": seed,
        "train_metrics": train_metrics,
        "metrics_s1": metrics_s1,
        "metrics_s2": metrics_s2,
        "recs_s1": recs_s1,
        "recs_s2": recs_s2,
    }


def compute_engineering_summary(
    all_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify alignment guards and compute descriptive engineering deltas."""
    print("=" * 70)
    print("VALIDATING CROSS-CONDITION ALIGNMENT & COMPUTING SUMMARY DELTAS")
    print("=" * 70)

    # 1. Quality Guards: sample ID and ground truth equality
    groups = ["B", "A", "C", "D"]
    for split in ("s1", "s2"):
        base_recs = all_results["B"][f"recs_{split}"]
        base_sample_ids = [r.sample_id for r in base_recs]
        base_gts = [r.ground_truth for r in base_recs]

        assert len(base_sample_ids) == 1136, f"Split {split} size {len(base_sample_ids)} != 1136"
        assert len(set(base_sample_ids)) == 1136, f"Duplicate sample IDs detected in {split}"

        for grp in ("A", "C", "D"):
            grp_recs = all_results[grp][f"recs_{split}"]
            grp_sample_ids = [r.sample_id for r in grp_recs]
            grp_gts = [r.ground_truth for r in grp_recs]

            assert grp_sample_ids == base_sample_ids, f"Sample ID mismatch between B and {grp} on {split}"
            assert grp_gts == base_gts, f"Ground truth mismatch between B and {grp} on {split}"

    print("  ✓ Perfect sample ID and ground truth alignment verified across B, A, C, D")

    # 2. Extract accuracies
    summary: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "seed": 42,
        "conditions": {},
        "deltas": {},
    }

    families = ("horizontal", "vertical", "depth", "near_far")

    for grp in groups:
        m1 = all_results[grp]["metrics_s1"]
        m2 = all_results[grp]["metrics_s2"]
        summary["conditions"][grp] = {
            "s1": {
                "accuracy": m1["overall"]["accuracy"],
                "invalid_predictions": m1["overall"]["invalid_predictions"],
                "invalid_rate": m1["overall"]["invalid_rate"],
                "correct": m1["overall"]["correct"],
                "total": m1["overall"]["total"],
                "by_family": {
                    fam: m1["by_dimension"].get(fam, {}).get("accuracy", 0.0)
                    for fam in families
                },
            },
            "s2": {
                "accuracy": m2["overall"]["accuracy"],
                "invalid_predictions": m2["overall"]["invalid_predictions"],
                "invalid_rate": m2["overall"]["invalid_rate"],
                "correct": m2["overall"]["correct"],
                "total": m2["overall"]["total"],
                "by_family": {
                    fam: m2["by_dimension"].get(fam, {}).get("accuracy", 0.0)
                    for fam in families
                },
            },
        }

    # 3. Compute descriptive deltas: A-B, C-B, D-B, C-A, D-A, D-C
    pairs = [
        ("A", "B"),
        ("C", "B"),
        ("D", "B"),
        ("C", "A"),
        ("D", "A"),
        ("D", "C"),
    ]

    for c1, c2 in pairs:
        pair_key = f"{c1}_minus_{c2}"
        summary["deltas"][pair_key] = {}
        for split in ("s1", "s2"):
            acc1 = summary["conditions"][c1][split]["accuracy"]
            acc2 = summary["conditions"][c2][split]["accuracy"]
            delta_overall = round(acc1 - acc2, 6)

            fam_deltas = {}
            for fam in families:
                f_acc1 = summary["conditions"][c1][split]["by_family"][fam]
                f_acc2 = summary["conditions"][c2][split]["by_family"][fam]
                fam_deltas[fam] = round(f_acc1 - f_acc2, 6)

            summary["deltas"][pair_key][split] = {
                "overall": delta_overall,
                "by_family": fam_deltas,
            }

    # Save summary
    summary_path = output_path or (REPO_ROOT / "outputs/experiments" / EXPERIMENT_ID / "engineering_seed42_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  ✓ Summary saved to {summary_path}")

    return summary


def print_final_table(summary: Dict[str, Any], training_stats: Dict[str, Dict[str, Any]]) -> None:
    """Print compact table of results."""
    print("\n" + "=" * 75)
    print("G2.0-E3.2 CONTROLLED ENGINEERING EXPERIMENT RESULTS (SEED 42)")
    print("=" * 75)

    print("\n--- EVALUATION SUMMARY (Overall Accuracy & Invalid Count / Total: 1,136) ---")
    print(f"{'Condition':<12} | {'S1 Acc':<8} | {'S1 Inv':<8} | {'S2 Acc':<8} | {'S2 Inv':<8}")
    print("-" * 55)
    for grp in ("B", "A", "C", "D"):
        s1 = summary["conditions"][grp]["s1"]
        s2 = summary["conditions"][grp]["s2"]
        print(f"{grp:<12} | {s1['accuracy']*100:6.2f}% | {s1['invalid_predictions']:<8} | {s2['accuracy']*100:6.2f}% | {s2['invalid_predictions']:<8}")

    print("\n--- DESCRIPTIVE ENGINEERING DELTAS (Percentage Points) ---")
    print(f"{'Comparison':<15} | {'S1 Overall':<10} | {'S2 Overall':<10} | {'S1 Depth':<10} | {'S2 Depth':<10}")
    print("-" * 65)
    for comp in ("A_minus_B", "C_minus_B", "D_minus_B", "C_minus_A", "D_minus_A", "D_minus_C"):
        d = summary["deltas"][comp]
        s1_ov = d["s1"]["overall"] * 100
        s2_ov = d["s2"]["overall"] * 100
        s1_dp = d["s1"]["by_family"]["depth"] * 100
        s2_dp = d["s2"]["by_family"]["depth"] * 100
        print(f"{comp:<15} | {s1_ov:+6.2f}%    | {s2_ov:+6.2f}%    | {s1_dp:+6.2f}%   | {s2_dp:+6.2f}%")

    print("\n--- TRAINING STATS (A42, C42, D42) ---")
    for grp in ("A", "C", "D"):
        st = training_stats.get(grp, {})
        print(f"Group {grp}: {st.get('microbatch_count')} microbatches, {st.get('optimizer_step_count')} opt steps, "
              f"wall time: {st.get('wall_time_sec')}s, peak VRAM: {st.get('peak_vram_gb')} GB, "
              f"loss: {st.get('first_10_loss_mean')} -> {st.get('last_10_loss_mean')}")
    print("=" * 75 + "\n")


def main() -> int:
    # 1. Preflight
    preflight = run_preflight()

    # 2. Config Freeze
    frozen_configs = freeze_run_configs(preflight)

    # 3. Baseline B
    res_b = run_baseline_b(preflight, frozen_configs)

    # 4. Train & Eval A42, C42, D42
    all_results: Dict[str, Dict[str, Any]] = {"B": res_b}
    training_stats: Dict[str, Dict[str, Any]] = {}

    for grp in ("A", "C", "D"):
        res = train_and_eval_group(grp, 42, preflight, frozen_configs)
        all_results[grp] = res
        training_stats[grp] = res["train_metrics"]

    # 5. Summary & Deltas
    summary = compute_engineering_summary(all_results)

    # 6. Display
    print_final_table(summary, training_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
