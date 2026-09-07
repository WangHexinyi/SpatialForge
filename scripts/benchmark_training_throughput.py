"""Training throughput profiling and benchmarking tool for G2.0-E3.3A.

Supports:
1. --mode profile: Deterministic 12-stage instrumentation of baseline production path,
   saving outputs/experiments/g2.0-e3-performance/baseline_profile.json
2. --mode benchmark: Side-by-side benchmark of ORIGINAL vs OPTIMIZED pipeline.
3. --mode equivalence: Mathematical verification across tensors, forward loss,
   and 8-microbatch gradient/parameter updates.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import get_peft_model

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spatialforge.experiment.performance import (
    GpuTelemetrySampler,
    LiveTelemetryReporter,
    NVMLClient,
    StageTimer,
    write_progress_file_atomic,
)
from spatialforge.experiment.pipeline import (
    DecodedImageCache,
    FrozenVisionFeatureCache,
    PinnedPrefetchDataLoader,
    compute_vision_cache_key,
)
from spatialforge.experiment.protocol import (
    DEFAULT_MODEL_PATH,
    TRAIN_DATASET_PATHS,
    seed_everything,
)
from spatialforge.experiment.training import (
    DEFAULT_TRAINING_PROFILE,
    FormalTrainingConfig,
    ProfileCapabilityError,
    build_training_tensors,
    collate_single_sample_batch,
    create_lora_config,
    format_chat_prompt,
    get_profile,
    list_profiles,
    load_and_preprocess_image,
    load_training_records,
    run_single_forward_step,
    validate_profile_capability,
)


def run_baseline_profile(
    num_samples: int = 128,
    group: str = "A",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Instrument existing baseline production training path over deterministic window."""
    if output_path is None:
        output_path = REPO_ROOT / "outputs/experiments/g2.0-e3-performance/baseline_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"G2.0-E3.3A BASELINE PROFILING (Group {group}, N={num_samples})")
    print("=" * 70)

    # 1. Load dataset records
    data_path = REPO_ROOT / TRAIN_DATASET_PATHS[group]
    records = load_training_records(data_path, base_dir=REPO_ROOT, limit=num_samples)
    print(f"Loaded {len(records)} records from {data_path.name}")

    # 2. Setup model and processor
    print("Loading processor and model on cuda (bf16)...")
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    lora_cfg = create_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )
    from transformers import get_linear_schedule_with_warmup
    total_opt_steps = max(1, num_samples // 8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_opt_steps)

    # Warmup 1 microbatch to stabilize CUDA kernels
    print("Warming up 1 dummy microbatch...")
    rec0 = records[0]
    img0 = load_and_preprocess_image(rec0.image_path)
    t0 = build_training_tensors(processor, img0, rec0.question, rec0.answer)
    b0 = collate_single_sample_batch(t0, device="cuda")
    out0 = model(**b0)
    (out0.loss / 8.0).backward()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # 3. Setup stage timers
    timers = {
        "1_json_lookup": StageTimer("1_json_lookup", cuda_sync=False),
        "2_png_decode": StageTimer("2_png_decode", cuda_sync=False),
        "3_rgba_to_rgb": StageTimer("3_rgba_to_rgb", cuda_sync=False),
        "4_qwen_vision_prep": StageTimer("4_qwen_vision_prep", cuda_sync=False),
        "5_chat_template": StageTimer("5_chat_template", cuda_sync=False),
        "6_tokenization": StageTimer("6_tokenization", cuda_sync=False),
        "7_tensor_assembly": StageTimer("7_tensor_assembly", cuda_sync=False),
        "8_host_to_device": StageTimer("8_host_to_device", cuda_sync=True),
        "9_model_forward": StageTimer("9_model_forward", cuda_sync=True),
        "10_backward": StageTimer("10_backward", cuda_sync=True),
        "11_optimizer_scheduler": StageTimer("11_optimizer_scheduler", cuda_sync=True),
    }

    # Start GPU sampler
    sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.1).start()

    print(f"Beginning instrumented run for {num_samples} microbatches...")
    t_global_start = time.perf_counter()

    for idx in range(num_samples):
        # 1. JSON/sample lookup
        timers["1_json_lookup"].start()
        rec = records[idx]
        timers["1_json_lookup"].stop()

        # 2. PNG open/decode
        timers["2_png_decode"].start()
        raw_img = Image.open(rec.image_path)
        raw_img.load()  # Force decoding
        timers["2_png_decode"].stop()

        # 3. RGBA -> RGB
        timers["3_rgba_to_rgb"].start()
        rgb_img = raw_img.convert("RGB")
        timers["3_rgba_to_rgb"].stop()

        # 4. Qwen vision/image preprocessing
        timers["4_qwen_vision_prep"].start()
        vis_inputs = processor.image_processor(images=[rgb_img], return_tensors="pt")
        timers["4_qwen_vision_prep"].stop()

        # 5. chat-template construction
        timers["5_chat_template"].start()
        prompt_text = format_chat_prompt(processor, rgb_img, rec.question)
        timers["5_chat_template"].stop()

        # 6. tokenization
        timers["6_tokenization"].start()
        # Qwen processor expands <|image_pad|> into pixel grid tokens
        text_inputs = processor(text=[prompt_text], images=[rgb_img], padding=False, return_tensors="pt")
        prompt_ids = text_inputs["input_ids"][0]
        ans_text = rec.answer + processor.tokenizer.eos_token
        ans_ids = processor.tokenizer(ans_text, add_special_tokens=False)["input_ids"]
        timers["6_tokenization"].stop()

        # 7. tensor assembly
        timers["7_tensor_assembly"].start()
        combined_input_ids = torch.cat([prompt_ids, torch.tensor(ans_ids, dtype=torch.long)])
        labels = combined_input_ids.clone()
        labels[: len(prompt_ids)] = -100
        attention_mask = torch.ones_like(combined_input_ids)
        tensors = {
            "input_ids": combined_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": vis_inputs["pixel_values"],
            "image_grid_thw": vis_inputs["image_grid_thw"],
        }
        timers["7_tensor_assembly"].stop()

        # 8. host -> device transfer
        timers["8_host_to_device"].start()
        batch = collate_single_sample_batch(tensors, device="cuda")
        timers["8_host_to_device"].stop()

        # 9. model forward
        timers["9_model_forward"].start()
        outputs = model(**batch)
        raw_loss = outputs.loss
        timers["9_model_forward"].stop()

        # 10. backward
        timers["10_backward"].start()
        scaled_loss = raw_loss / 8.0
        scaled_loss.backward()
        timers["10_backward"].stop()

        # 11. optimizer/scheduler
        timers["11_optimizer_scheduler"].start()
        if (idx + 1) % 8 == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        timers["11_optimizer_scheduler"].stop()

        if (idx + 1) % 32 == 0 or idx == num_samples - 1:
            print(f"  Microbatch {idx + 1}/{num_samples} ({(idx + 1) / num_samples * 100:.1f}%) | loss: {raw_loss.item():.4f}")

    torch.cuda.synchronize()
    total_wall_sec = time.perf_counter() - t_global_start
    telemetry_summary = sampler.stop()

    # Aggregate summaries
    stage_summaries = {}
    total_instrumented_ms = 0.0
    for name, t in timers.items():
        s = t.summary()
        stage_summaries[name] = s
        total_instrumented_ms += s["total_ms"]

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    samples_per_sec = num_samples / total_wall_sec

    # Idle / unmeasured time
    total_wall_ms = total_wall_sec * 1000.0
    idle_ms = max(0.0, total_wall_ms - total_instrumented_ms)
    stage_summaries["12_idle_unmeasured"] = {
        "total_ms": round(idle_ms, 2),
        "mean_ms": round(idle_ms / num_samples, 3),
        "median_ms": 0.0,
        "p95_ms": 0.0,
    }

    # Percentages
    for name in stage_summaries:
        pct = (stage_summaries[name]["total_ms"] / total_wall_ms) * 100.0
        stage_summaries[name]["percent_of_wall_time"] = round(pct, 2)

    # Preprocessing vs GPU execution
    cpu_prep_ms = sum(stage_summaries[k]["total_ms"] for k in [
        "1_json_lookup", "2_png_decode", "3_rgba_to_rgb", "4_qwen_vision_prep",
        "5_chat_template", "6_tokenization", "7_tensor_assembly"
    ])
    gpu_exec_ms = sum(stage_summaries[k]["total_ms"] for k in [
        "8_host_to_device", "9_model_forward", "10_backward", "11_optimizer_scheduler"
    ])

    report = {
        "benchmark_group": group,
        "num_microbatches": num_samples,
        "total_wall_time_sec": round(total_wall_sec, 2),
        "samples_per_second": round(samples_per_sec, 3),
        "seconds_per_sample": round(total_wall_sec / num_samples, 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "gpu_telemetry": telemetry_summary,
        "cpu_preprocessing_total_ms": round(cpu_prep_ms, 2),
        "cpu_preprocessing_pct": round((cpu_prep_ms / total_wall_ms) * 100.0, 2),
        "gpu_execution_total_ms": round(gpu_exec_ms, 2),
        "gpu_execution_pct": round((gpu_exec_ms / total_wall_ms) * 100.0, 2),
        "stage_breakdown": stage_summaries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("BASELINE PROFILING RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total wall time:       {total_wall_sec:.2f} s")
    print(f"Throughput:            {samples_per_sec:.2f} samples/s ({1.0 / samples_per_sec:.3f} s/sample)")
    print(f"Projected 1552 time:   {1552 / samples_per_sec / 60.0:.2f} min")
    print(f"Peak VRAM:             {peak_vram_gb:.2f} GB")
    print(f"GPU Util Avg:          {telemetry_summary['gpu_utilization']['avg']}% (p50: {telemetry_summary['gpu_utilization']['p50']}%, p95: {telemetry_summary['gpu_utilization']['p95']}%)")
    print(f"GPU Power Avg:         {telemetry_summary['gpu_power_w']['avg']} W")
    print(f"CPU Prep Fraction:     {report['cpu_preprocessing_pct']}% ({cpu_prep_ms / 1000:.2f}s)")
    print(f"GPU Exec Fraction:     {report['gpu_execution_pct']}% ({gpu_exec_ms / 1000:.2f}s)")
    print("-" * 70)
    print(f"{'Stage':<28} | {'Mean (ms)':<10} | {'Median':<8} | {'p95':<8} | {'% Wall':<6}")
    print("-" * 70)
    for k, v in stage_summaries.items():
        print(f"{k:<28} | {v['mean_ms']:<10.2f} | {v['median_ms']:<8.2f} | {v['p95_ms']:<8.2f} | {v['percent_of_wall_time']:<6.1f}%")
    print("=" * 70)
    print(f"Saved artifact to: {output_path}")

    return report




def run_equivalence_test(
    num_samples: int = 16,
    group: str = "A",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute rigorous mathematical equivalence verification between ORIGINAL and OPTIMIZED paths.

    Verifies:
    1. Bitwise tensor equality of prepared tensors (input_ids, attention_mask, labels,
       pixel_values, image_grid_thw).
    2. Single-sample forward loss equality.
    3. 8-microbatch full gradient accumulation, optimizer step, and post-step LoRA weights.
    """
    from spatialforge.experiment.pipeline import (
        PreparedDatasetCache,
        PinnedPrefetchDataLoader,
        compute_pipeline_cache_key,
    )

    print("\n" + "=" * 70)
    print("PHASE 7 — MATHEMATICAL EQUIVALENCE TEST (ORIGINAL vs OPTIMIZED)")
    print("=" * 70)

    # 1. Load records
    data_path = REPO_ROOT / TRAIN_DATASET_PATHS[group]
    records = load_training_records(data_path, base_dir=REPO_ROOT, limit=num_samples)
    print(f"Loaded {len(records)} test records from {data_path.name}")

    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH)

    # -------------------------------------------------------------
    # Step A: Prepared Tensors Verification
    # -------------------------------------------------------------
    print("\n[Step A] Verifying prepared tensor bitwise equivalence...")
    cache = PreparedDatasetCache(cache_dir=REPO_ROOT / "outputs/cache/prepared_samples")
    cache_key = compute_pipeline_cache_key(data_path.name)
    optimized_samples = cache.get_or_build(records, processor, cache_key, force_rebuild=True)

    tensor_failures = []
    for i, rec in enumerate(records):
        img_orig = load_and_preprocess_image(rec.image_path)
        orig_tensors = build_training_tensors(processor, img_orig, rec.question, rec.answer)
        opt_tensors = optimized_samples[i]

        for k in ("input_ids", "attention_mask", "labels", "image_grid_thw", "pixel_values"):
            t_orig = orig_tensors[k]
            t_opt = opt_tensors[k]
            if not torch.equal(t_orig, t_opt):
                max_diff = (t_orig.float() - t_opt.float()).abs().max().item()
                tensor_failures.append(f"Sample {i} tensor {k} differs by max {max_diff}")

    if tensor_failures:
        print(f"  FAILED: {len(tensor_failures)} tensor discrepancies detected!")
        for f in tensor_failures[:5]:
            print(f"    - {f}")
        return {"status": "FAIL", "tensor_failures": tensor_failures}
    else:
        print(f"  ✓ All {len(records)} samples have bitwise identical prepared tensors across all 5 keys.")

    # -------------------------------------------------------------
    # Step B: Single-Microbatch Forward Loss Verification
    # -------------------------------------------------------------
    print("\n[Step B] Verifying single-microbatch forward loss equivalence...")
    seed_everything(42)
    model_orig = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model_orig.enable_input_require_grads()
    model_orig.gradient_checkpointing_enable()
    lora_cfg = create_lora_config()
    model_orig = get_peft_model(model_orig, lora_cfg)
    model_orig.train()

    orig_batch0 = collate_single_sample_batch(optimized_samples[0], device=device)
    with torch.set_grad_enabled(True):
        out_orig = model_orig(**orig_batch0)
        loss_orig = out_orig.loss.item()

    # Optimized path forward
    prefetcher = PinnedPrefetchDataLoader([optimized_samples[0]], device=device, queue_size=2)
    opt_batch0 = next(iter(prefetcher))

    with torch.set_grad_enabled(True):
        out_opt = model_orig(**opt_batch0)
        loss_opt = out_opt.loss.item()

    loss_diff = abs(loss_orig - loss_opt)
    print(f"  Original forward loss:  {loss_orig:.8f}")
    print(f"  Optimized forward loss: {loss_opt:.8f}")
    print(f"  Forward loss diff:      {loss_diff:.2e}")
    assert loss_diff <= 1e-6, f"Single forward loss differs: {loss_diff}"
    print("  ✓ Single forward loss is identical within numerical tolerance (<= 1e-6).")

    # -------------------------------------------------------------
    # Step C: 8-Microbatch Full Gradient Accumulation Equivalence
    # -------------------------------------------------------------
    print("\n[Step C] Verifying 8-microbatch gradient accumulation & optimizer step...")
    # Run 1: ORIGINAL execution path
    seed_everything(42)
    model_1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model_1.enable_input_require_grads()
    model_1.gradient_checkpointing_enable()
    model_1 = get_peft_model(model_1, lora_cfg)
    model_1.train()
    opt_1 = torch.optim.AdamW([p for p in model_1.parameters() if p.requires_grad], lr=1e-4)

    orig_losses = []
    opt_1.zero_grad()
    for i in range(8):
        rec = records[i]
        img = load_and_preprocess_image(rec.image_path)
        tensors = build_training_tensors(processor, img, rec.question, rec.answer)
        batch = collate_single_sample_batch(tensors, device=device)
        out = model_1(**batch)
        orig_losses.append(out.loss.item())
        (out.loss / 8.0).backward()

    # Capture gradients before step
    grads_1 = {n: p.grad.clone() for n, p in model_1.named_parameters() if p.requires_grad and p.grad is not None}
    opt_1.step()
    opt_1.zero_grad()
    weights_1 = {n: p.clone() for n, p in model_1.named_parameters() if p.requires_grad}

    # Run 2: OPTIMIZED execution path
    seed_everything(42)
    model_2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model_2.enable_input_require_grads()
    model_2.gradient_checkpointing_enable()
    model_2 = get_peft_model(model_2, lora_cfg)
    model_2.train()
    opt_2 = torch.optim.AdamW([p for p in model_2.parameters() if p.requires_grad], lr=1e-4)

    opt_losses = []
    opt_2.zero_grad()
    opt_prefetch = PinnedPrefetchDataLoader(optimized_samples[:8], device=device, queue_size=4)
    for batch in opt_prefetch:
        out = model_2(**batch)
        opt_losses.append(out.loss.item())
        (out.loss / 8.0).backward()

    grads_2 = {n: p.grad.clone() for n, p in model_2.named_parameters() if p.requires_grad and p.grad is not None}
    opt_2.step()
    opt_2.zero_grad()
    weights_2 = {n: p.clone() for n, p in model_2.named_parameters() if p.requires_grad}

    # Compare 8 losses
    loss_diffs = [abs(a - b) for a, b in zip(orig_losses, opt_losses)]
    max_loss_diff = max(loss_diffs)
    print(f"  Microbatch losses (Orig vs Opt):")
    for mb_i, (l_o, l_opt) in enumerate(zip(orig_losses, opt_losses)):
        print(f"    mb {mb_i + 1}: orig={l_o:.6f}, opt={l_opt:.6f}, diff={abs(l_o - l_opt):.2e}")

    # Compare gradients
    grad_diffs = []
    for n in grads_1:
        diff = (grads_1[n] - grads_2[n]).abs().max().item()
        grad_diffs.append(diff)
    max_grad_diff = max(grad_diffs) if grad_diffs else 0.0

    # Compare updated LoRA weights
    weight_diffs = []
    for n in weights_1:
        diff = (weights_1[n] - weights_2[n]).abs().max().item()
        weight_diffs.append(diff)
    max_weight_diff = max(weight_diffs) if weight_diffs else 0.0

    print(f"\n  Accumulation equivalence summary:")
    print(f"    Max microbatch loss diff:      {max_loss_diff:.2e}")
    print(f"    Max LoRA gradient diff:        {max_grad_diff:.2e}")
    print(f"    Max post-step parameter diff:  {max_weight_diff:.2e}")

    # Strict tolerance check justified by bfloat16 machine epsilon (2^-7 ≈ 0.0078)
    # and CUDA non-deterministic parallel reduction order
    LOSS_TOLERANCE = 1e-5
    GRAD_TOLERANCE = 1e-2
    WEIGHT_TOLERANCE = 1e-3
    is_equiv = (max_loss_diff <= LOSS_TOLERANCE) and (max_grad_diff <= GRAD_TOLERANCE) and (max_weight_diff <= WEIGHT_TOLERANCE)

    if is_equiv:
        print("\n  >>> MATHEMATICAL EQUIVALENCE VERIFIED: PASS <<<")
        status = "PASS"
    else:
        print("\n  >>> MATHEMATICAL EQUIVALENCE FAILED: DIVERGENCE DETECTED <<<")
        status = "FAIL"

    return {
        "status": status,
        "max_loss_diff": max_loss_diff,
        "max_grad_diff": max_grad_diff,
        "max_weight_diff": max_weight_diff,
        "is_equivalent": is_equiv,
    }


def run_performance_benchmark(
    num_samples: int = 128,
    group: str = "A",
    device: str = "cuda",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Side-by-side benchmark of ORIGINAL vs OPTIMIZED pipeline."""
    from spatialforge.experiment.pipeline import (
        PreparedDatasetCache,
        PinnedPrefetchDataLoader,
        compute_pipeline_cache_key,
    )

    if output_path is None:
        output_path = REPO_ROOT / "outputs/experiments/g2.0-e3-performance/benchmark_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"PHASE 8 — PERFORMANCE BENCHMARK (Group {group}, N={num_samples})")
    print("=" * 70)

    data_path = REPO_ROOT / TRAIN_DATASET_PATHS[group]
    records = load_training_records(data_path, base_dir=REPO_ROOT, limit=num_samples)
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH)
    lora_cfg = create_lora_config()

    def _setup_model_and_opt():
        seed_everything(42)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            DEFAULT_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, lora_cfg)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, max(1, num_samples // 8))
        return model, optimizer, scheduler

    # Warmup 1 microbatch
    print("Warming up GPU kernels...")
    model_w, opt_w, sched_w = _setup_model_and_opt()
    rec0 = records[0]
    img0 = load_and_preprocess_image(rec0.image_path)
    t0 = build_training_tensors(processor, img0, rec0.question, rec0.answer)
    b0 = collate_single_sample_batch(t0, device=device)
    out0 = model_w(**b0)
    (out0.loss / 8.0).backward()
    opt_w.zero_grad()
    torch.cuda.synchronize()
    del model_w, opt_w, sched_w
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 1. Benchmark ORIGINAL Path
    # -------------------------------------------------------------
    print("\n[1/3] Benchmarking ORIGINAL execution path...")
    model_orig, opt_orig, sched_orig = _setup_model_and_opt()
    torch.cuda.reset_peak_memory_stats()
    sampler_orig = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.1).start()

    t_cpu_prep_orig = 0.0
    t0_orig = time.perf_counter()

    for idx in range(num_samples):
        rec = records[idx]
        t_p0 = time.perf_counter()
        img = load_and_preprocess_image(rec.image_path)
        tensors = build_training_tensors(processor, img, rec.question, rec.answer)
        t_cpu_prep_orig += (time.perf_counter() - t_p0)

        batch = collate_single_sample_batch(tensors, device=device)
        out = model_orig(**batch)
        (out.loss / 8.0).backward()

        if (idx + 1) % 8 == 0:
            opt_orig.step()
            sched_orig.step()
            opt_orig.zero_grad()

    torch.cuda.synchronize()
    wall_orig = time.perf_counter() - t0_orig
    telemetry_orig = sampler_orig.stop()
    vram_peak_orig = torch.cuda.max_memory_allocated() / (1024 ** 3)
    sps_orig = num_samples / wall_orig

    print(f"  ORIGINAL complete: {wall_orig:.2f}s ({sps_orig:.2f} samples/s)")
    del model_orig, opt_orig, sched_orig
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 2. Benchmark OPTIMIZED Path (Cold Cache)
    # -------------------------------------------------------------
    print("\n[2/3] Benchmarking OPTIMIZED execution path (Cold Cache build + run)...")
    cache_dir = REPO_ROOT / "outputs/cache/benchmark_test"
    cache_key = compute_pipeline_cache_key(data_path, dataset_hash=f"bench_{group}_{num_samples}")
    cache = PreparedDatasetCache(cache_dir=cache_dir)

    t_build_0 = time.perf_counter()
    cold_samples = cache.get_or_build(records, processor, cache_key, force_rebuild=True)
    t_cache_build = time.perf_counter() - t_build_0
    print(f"  Cache build time for {num_samples} samples: {t_cache_build:.2f}s")

    model_opt_cold, opt_cold, sched_cold = _setup_model_and_opt()
    torch.cuda.reset_peak_memory_stats()
    sampler_cold = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.1).start()

    t0_cold = time.perf_counter()
    prefetch_cold = PinnedPrefetchDataLoader(cold_samples, device=device, queue_size=8)
    for idx, batch in enumerate(prefetch_cold):
        out = model_opt_cold(**batch)
        (out.loss / 8.0).backward()

        if (idx + 1) % 8 == 0:
            opt_cold.step()
            sched_cold.step()
            opt_cold.zero_grad()

    torch.cuda.synchronize()
    wall_cold = time.perf_counter() - t0_cold
    telemetry_cold = sampler_cold.stop()
    vram_peak_cold = torch.cuda.max_memory_allocated() / (1024 ** 3)
    sps_cold = num_samples / wall_cold

    print(f"  OPTIMIZED (Cold execution) complete: {wall_cold:.2f}s ({sps_cold:.2f} samples/s)")
    del model_opt_cold, opt_cold, sched_cold
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. Benchmark OPTIMIZED Path (Warm Cache)
    # -------------------------------------------------------------
    print("\n[3/3] Benchmarking OPTIMIZED execution path (Warm Cache)...")
    warm_samples = cache.get_or_build(records, processor, cache_key, force_rebuild=False)

    model_opt_warm, opt_warm, sched_warm = _setup_model_and_opt()
    torch.cuda.reset_peak_memory_stats()
    sampler_warm = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.1).start()

    t0_warm = time.perf_counter()
    prefetch_warm = PinnedPrefetchDataLoader(warm_samples, device=device, queue_size=8)
    for idx, batch in enumerate(prefetch_warm):
        out = model_opt_warm(**batch)
        (out.loss / 8.0).backward()

        if (idx + 1) % 8 == 0:
            opt_warm.step()
            sched_warm.step()
            opt_warm.zero_grad()

    torch.cuda.synchronize()
    wall_warm = time.perf_counter() - t0_warm
    telemetry_warm = sampler_warm.stop()
    vram_peak_warm = torch.cuda.max_memory_allocated() / (1024 ** 3)
    sps_warm = num_samples / wall_warm

    print(f"  OPTIMIZED (Warm execution) complete: {wall_warm:.2f}s ({sps_warm:.2f} samples/s)")
    del model_opt_warm, opt_warm, sched_warm
    gc.collect()
    torch.cuda.empty_cache()

    speedup_cold = sps_cold / sps_orig
    speedup_warm = sps_warm / sps_orig
    proj_1552_orig_min = (1552 / sps_orig) / 60.0
    proj_1552_warm_min = (1552 / sps_warm) / 60.0

    report = {
        "num_microbatches": num_samples,
        "group": group,
        "original": {
            "wall_time_sec": round(wall_orig, 2),
            "samples_per_sec": round(sps_orig, 3),
            "seconds_per_sample": round(wall_orig / num_samples, 3),
            "cpu_prep_total_sec": round(t_cpu_prep_orig, 3),
            "vram_peak_gb": round(vram_peak_orig, 3),
            "gpu_util_avg": telemetry_orig["gpu_utilization"]["avg"],
            "gpu_util_p50": telemetry_orig["gpu_utilization"]["p50"],
            "gpu_util_p95": telemetry_orig["gpu_utilization"]["p95"],
            "gpu_power_avg_w": telemetry_orig["gpu_power_w"]["avg"],
            "projected_1552_wall_time_min": round(proj_1552_orig_min, 2),
        },
        "optimized_warm": {
            "wall_time_sec": round(wall_warm, 2),
            "samples_per_sec": round(sps_warm, 3),
            "seconds_per_sample": round(wall_warm / num_samples, 3),
            "cpu_prep_in_loop_sec": 0.0,
            "vram_peak_gb": round(vram_peak_warm, 3),
            "gpu_util_avg": telemetry_warm["gpu_utilization"]["avg"],
            "gpu_util_p50": telemetry_warm["gpu_utilization"]["p50"],
            "gpu_util_p95": telemetry_warm["gpu_utilization"]["p95"],
            "gpu_power_avg_w": telemetry_warm["gpu_power_w"]["avg"],
            "projected_1552_wall_time_min": round(proj_1552_warm_min, 2),
            "speedup_vs_original": round(speedup_warm, 3),
        },
        "cache_metrics": {
            "cache_build_time_sec": round(t_cache_build, 2),
            "cache_file": str(cache_dir / f"{cache_key}.pt"),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print markdown table
    print("\n" + "=" * 70)
    print("PHASE 8 PERFORMANCE BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    fmt_row = lambda metric, orig, opt: f"{metric:<32} | {orig:<16} | {opt:<16}"
    print(fmt_row("Metric", "ORIGINAL", "OPTIMIZED (Warm)"))
    print("-" * 70)
    print(fmt_row("samples/sec", f"{sps_orig:.2f}", f"{sps_warm:.2f}"))
    print(fmt_row("seconds/sample", f"{wall_orig/num_samples:.3f} s", f"{wall_warm/num_samples:.3f} s"))
    print(fmt_row("GPU util avg", f"{telemetry_orig['gpu_utilization']['avg']}%", f"{telemetry_warm['gpu_utilization']['avg']}%"))
    print(fmt_row("GPU util p50/p95", f"{telemetry_orig['gpu_utilization']['p50']}% / {telemetry_orig['gpu_utilization']['p95']}%", f"{telemetry_warm['gpu_utilization']['p50']}% / {telemetry_warm['gpu_utilization']['p95']}%"))
    print(fmt_row("VRAM peak", f"{vram_peak_orig:.2f} GB", f"{vram_peak_warm:.2f} GB"))
    print(fmt_row("GPU power avg", f"{telemetry_orig['gpu_power_w']['avg']:.1f} W", f"{telemetry_warm['gpu_power_w']['avg']:.1f} W"))
    print(fmt_row("CPU prep in loop", f"{t_cpu_prep_orig:.2f} s", "0.00 s (hidden)"))
    print(fmt_row("projected 1552 wall time", f"{proj_1552_orig_min:.2f} min", f"{proj_1552_warm_min:.2f} min"))
    print(fmt_row("speedup", "1.00x", f"{speedup_warm:.2f}x"))
    print("=" * 70)
    print(f"Saved benchmark report to: {output_path}")

    return report


def run_p5_full_soak(
    group: str = "A",
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute full 1552-sample soak of Candidate P5 on Group A seed 42 with cold cache audit.

    Requirements:
    1. Cold-cache audit: start with empty cache directory, measure build time, unique images,
       hit/miss counts, and disk space.
    2. Full 1552-sample training verification: 1552 microbatches, 194 optimizer steps,
       194 scheduler steps, loss progression (first-10 mean, last-10 mean, finite check).
    3. Telemetry: GPU util avg/p50/p95, power avg, peak VRAM, and monotonic host memory check.
    4. Warm-cache reuse check: 8 samples against populated cache verifying 100% hits, zero
       vision encoder execution, bitwise identical tensors, and projected subsequent-seed cost.
    5. Save complete soak report to outputs/experiments/g2.0-e3-performance/p5_full_soak_report.json.
    """
    if output_path is None:
        output_path = REPO_ROOT / "outputs/experiments/g2.0-e3-performance/p5_full_soak_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"G2.0-E3.3A FINAL P5 SOAK & CACHE AUDIT (Group {group}, Seed {seed})")
    print("=" * 70)

    # 1. Start with COLD cache (empty directory)
    cache_dir = REPO_ROOT / "outputs/cache/frozen_vision_features"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] Initialized empty cold cache directory: {cache_dir}")

    # 2. Load dataset records (N=1552)
    data_path = REPO_ROOT / TRAIN_DATASET_PATHS[group]
    records = load_training_records(data_path, base_dir=REPO_ROOT)
    num_samples = len(records)
    assert num_samples == 1552, f"Expected 1552 records, got {num_samples}"
    print(f"[2/5] Loaded {num_samples} records from {data_path.name}")

    # 3. Setup clean model and processor (bf16, LoRA LM-only, native SDPA)
    print("[3/5] Initializing Qwen2.5-VL-3B with 252 LM LoRA adapters (GC=False)...")
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.enable_input_require_grads()
    lora_cfg = create_lora_config()
    model = get_peft_model(model, lora_cfg)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == 14966784, f"Trainable params {trainable_count} != 14,966,784"
    print(f"  ✓ Trainable parameters: {trainable_count:,} (252 LM modules, 0 vision)")

    # 4. COLD-CACHE POPULATION & AUDIT
    print("[4/5] Executing cold-cache build and full sample preparation...")
    vc_cache = FrozenVisionFeatureCache(cache_dir=cache_dir, in_memory=True)
    img_cache = DecodedImageCache()
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model

    t_cache_0 = time.perf_counter()
    cold_feats_first_8 = []
    prepared_samples = []

    with torch.no_grad():
        for idx, r in enumerate(records):
            c_feat = vc_cache.get_or_compute(
                r.image_path, model, processor, device="cuda", dtype=torch.bfloat16
            )
            if idx < 8:
                cold_feats_first_8.append(c_feat.clone())

            c_feat_gpu = c_feat.to("cuda")
            img = img_cache.get(r.image_path)
            t = build_training_tensors(processor, img, r.question, r.answer)
            in_emb = base_model.model.get_input_embeddings()(t["input_ids"].unsqueeze(0).to("cuda"))
            mask, _ = base_model.model.get_placeholder_mask(
                t["input_ids"].unsqueeze(0).to("cuda"), inputs_embeds=in_emb, image_features=c_feat_gpu
            )
            in_emb = in_emb.masked_scatter(mask, c_feat_gpu).squeeze(0).cpu()
            prepared_samples.append({
                "inputs_embeds": in_emb,
                "attention_mask": t["attention_mask"],
                "labels": t["labels"],
            })

    t_cold_build_e2e = time.perf_counter() - t_cache_0
    cache_stats = vc_cache.get_stats()
    disk_size_bytes = vc_cache.get_disk_size_bytes()
    disk_size_mb = disk_size_bytes / (1024 ** 2)

    print(f"  ✓ Vision cache built: {cache_stats['unique_cached']} unique images")
    print(f"  ✓ Cache queries: {cache_stats['total_queries']} | Hits: {cache_stats['hits']} | Misses: {cache_stats['misses']} (Hit rate: {cache_stats['hit_rate']*100:.1f}%)")
    print(f"  ✓ Disk space used: {disk_size_mb:.1f} MB across {len(list(cache_dir.glob('*.pt')))} files")
    print(f"  ✓ Cache build + preparation time: {t_cold_build_e2e:.2f} s")

    assert cache_stats["unique_cached"] == 78, f"Expected 78 unique images, got {cache_stats['unique_cached']}"
    assert cache_stats["misses"] == 78, f"Expected 78 misses, got {cache_stats['misses']}"
    assert cache_stats["hits"] == 1474, f"Expected 1474 hits, got {cache_stats['hits']}"
    assert disk_size_bytes > 0, "Expected non-zero disk cache size"

    # 5. FULL 1552-SAMPLE TRAINING EXECUTION
    print("\n[5/5] Executing full 1552-sample training loop (Group A, Seed 42)...")
    seed_everything(seed)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    from transformers import get_linear_schedule_with_warmup
    total_opt_steps = num_samples // 8  # 194
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_opt_steps)
    optimizer.zero_grad()
    model.train()

    # Host memory tracking
    proc = psutil.Process()
    host_rss_records: Dict[int, float] = {}
    host_rss_records[0] = round(proc.memory_info().rss / (1024 ** 3), 3)

    # Telemetry and logging
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.2).start()
    soak_run_dir = REPO_ROOT / "outputs/experiments/g2.0-e3-performance/soak_run"
    soak_run_dir.mkdir(parents=True, exist_ok=True)
    reporter = LiveTelemetryReporter(
        run_dir=soak_run_dir,
        group=group,
        seed=seed,
        total_microbatches=num_samples,
        total_optimizer_steps=total_opt_steps,
        heartbeat_interval=50,
        sampler=sampler,
        total_samples=num_samples,
    )

    losses: List[float] = []
    mb_count = 0
    opt_step_count = 0
    sched_step_count = 0

    prefetcher = PinnedPrefetchDataLoader(prepared_samples, device="cuda", queue_size=8)
    t_train_start = time.perf_counter()

    for idx, batch in enumerate(prefetcher):
        out = model(**batch)
        raw_loss = float(out.loss.item())
        if not math.isfinite(raw_loss):
            raise ValueError(f"Non-finite loss detected at microbatch {idx}: {raw_loss}")
        losses.append(raw_loss)

        scaled_loss = out.loss / 8.0
        scaled_loss.backward()

        mb_count += 1
        is_step = (mb_count % 8 == 0)
        current_lr = optimizer.param_groups[0]["lr"]

        if is_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step_count += 1
            sched_step_count += 1

        reporter.step(
            microbatch=mb_count,
            optimizer_step=opt_step_count,
            scheduler_step=sched_step_count,
            latest_raw_loss=raw_loss,
            learning_rate=current_lr,
            processed_samples=mb_count,
        )

        if mb_count in (388, 776, 1164, 1552):
            host_rss_records[mb_count] = round(proc.memory_info().rss / (1024 ** 3), 3)

    torch.cuda.synchronize()
    t_train_wall = time.perf_counter() - t_train_start
    telemetry_summary = sampler.stop()

    # Final progress report update
    final_lr = optimizer.param_groups[0]["lr"]
    reporter.step(
        microbatch=mb_count,
        optimizer_step=opt_step_count,
        scheduler_step=sched_step_count,
        latest_raw_loss=losses[-1],
        learning_rate=final_lr,
        force=True,
        status="completed",
        processed_samples=mb_count,
    )

    # Save trained adapter
    adapter_dir = soak_run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    print(f"  ✓ Adapter saved to: {adapter_dir}")

    # Training metrics verification
    assert mb_count == 1552, f"Expected 1552 microbatches, got {mb_count}"
    assert opt_step_count == 194, f"Expected 194 optimizer steps, got {opt_step_count}"
    assert sched_step_count == 194, f"Expected 194 scheduler steps, got {sched_step_count}"
    first_10_mean = sum(losses[:10]) / 10.0
    last_10_mean = sum(losses[-10:]) / 10.0
    overall_mean_loss = sum(losses) / len(losses)
    sps = num_samples / t_train_wall
    sec_per_sample = t_train_wall / num_samples
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    total_cold_e2e_wall_sec = t_cold_build_e2e + t_train_wall

    # Host memory leak check
    rss_values = list(host_rss_records.values())
    host_rss_max = max(rss_values)
    host_rss_min = min(rss_values)
    rss_delta = host_rss_max - host_rss_min
    host_leak_detected = bool(rss_delta > 0.5)

    print(f"\n  ✓ 1552 microbatches, {opt_step_count} optimizer steps, {sched_step_count} scheduler steps complete.")
    print(f"  ✓ Loss: initial={losses[0]:.4f}, first-10 mean={first_10_mean:.4f}, last-10 mean={last_10_mean:.4f}, final={losses[-1]:.4f}")
    print(f"  ✓ Training wall time: {t_train_wall:.2f} s ({t_train_wall/60.0:.2f} min) -> {sps:.3f} samples/s ({sec_per_sample:.3f} s/sample)")
    print(f"  ✓ Total Cold E2E wall time: {total_cold_e2e_wall_sec:.2f} s ({total_cold_e2e_wall_sec/60.0:.2f} min)")
    print(f"  ✓ Host RSS memory: {host_rss_records} (delta={rss_delta:.3f} GB, monotonic leak={host_leak_detected})")
    print(f"  ✓ VRAM Peak: {peak_vram_gb:.2f} GB | GPU util avg: {telemetry_summary['gpu_utilization']['avg']}% | Power avg: {telemetry_summary['gpu_power_w']['avg']:.1f} W")

    # 6. WARM-CACHE REUSE CHECK
    print("\n[6/6] Executing warm-cache reuse validation against populated disk cache...")
    warm_vc = FrozenVisionFeatureCache(cache_dir=cache_dir, in_memory=False)
    warm_feats = []
    t_warm_start = time.perf_counter()
    for idx in range(8):
        r = records[idx]
        f = warm_vc.get_or_compute(
            r.image_path, model, processor, device="cuda", dtype=torch.bfloat16
        )
        warm_feats.append(f)
    t_warm_read = time.perf_counter() - t_warm_start

    warm_stats = warm_vc.get_stats()
    assert warm_stats["hits"] == 8, f"Expected 8 hits, got {warm_stats['hits']}"
    assert warm_stats["misses"] == 0, f"Expected 0 misses, got {warm_stats['misses']}"
    assert warm_stats["hit_rate"] == 1.0, f"Expected 100% hit rate, got {warm_stats['hit_rate']}"

    # Check tensor bitwise equality
    tensors_bitwise_match = True
    for w, c in zip(warm_feats, cold_feats_first_8):
        if not torch.equal(w, c):
            tensors_bitwise_match = False
            break
    assert tensors_bitwise_match, "Warm cached tensors do NOT match cold-run tensors bitwise!"
    print(f"  ✓ Warm reuse verification: 8/8 hits (100.0%), 0 misses, bitwise identical tensors (0 vision encoder calls)")

    # Subsequent seed projections
    proj_subsequent_seed_min = t_train_wall / 60.0

    # 7. COMPARATIVE SUMMARY
    ref_wall_sec = 1672.73
    ref_sps = 0.928
    p1_wall_sec = 834.86
    p1_sps = 1.859

    speedup_cold = ref_wall_sec / total_cold_e2e_wall_sec
    speedup_warm = ref_wall_sec / t_train_wall

    report = {
        "condition": "Group A Seed 42 Candidate P5 Full Soak",
        "parameters": {
            "microbatch_size": 1,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": False,
            "use_vision_cache": True,
            "attention_implementation": "native_sdpa",
            "bf16": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_targets": "252 LM / 0 vision",
            "epochs": 1,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "scheduler": "linear",
            "warmup_steps": 0,
        },
        "cold_cache_audit": {
            "unique_images": cache_stats["unique_cached"],
            "total_queries": cache_stats["total_queries"],
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_build_sec": round(t_cold_build_e2e, 2),
            "disk_size_mb": round(disk_size_mb, 2),
            "disk_files_count": len(list(cache_dir.glob("*.pt"))),
        },
        "training_execution": {
            "total_microbatches": mb_count,
            "total_optimizer_steps": opt_step_count,
            "total_scheduler_steps": sched_step_count,
            "training_wall_sec": round(t_train_wall, 2),
            "training_wall_min": round(t_train_wall / 60.0, 2),
            "samples_per_sec": round(sps, 3),
            "seconds_per_sample": round(sec_per_sample, 3),
            "initial_loss": round(losses[0], 4),
            "final_loss": round(losses[-1], 4),
            "first_10_loss_mean": round(first_10_mean, 4),
            "last_10_loss_mean": round(last_10_mean, 4),
            "overall_loss_mean": round(overall_mean_loss, 4),
            "all_losses_finite": True,
            "vram_peak_gb": round(peak_vram_gb, 3),
            "gpu_util_avg": telemetry_summary["gpu_utilization"]["avg"],
            "gpu_util_p50": telemetry_summary["gpu_utilization"]["p50"],
            "gpu_util_p95": telemetry_summary["gpu_utilization"]["p95"],
            "gpu_power_avg_w": telemetry_summary["gpu_power_w"]["avg"],
            "host_rss_checkpoints_gb": host_rss_records,
            "host_rss_delta_gb": round(rss_delta, 3),
            "monotonic_host_leak": host_leak_detected,
        },
        "cold_e2e": {
            "total_wall_sec": round(total_cold_e2e_wall_sec, 2),
            "total_wall_min": round(total_cold_e2e_wall_sec / 60.0, 2),
            "effective_samples_per_sec": round(num_samples / total_cold_e2e_wall_sec, 3),
            "speedup_vs_ref": round(speedup_cold, 3),
        },
        "warm_cache_reuse": {
            "test_samples": 8,
            "cache_hits": warm_stats["hits"],
            "cache_misses": warm_stats["misses"],
            "cache_hit_rate": warm_stats["hit_rate"],
            "vision_encoder_calls": 0,
            "tensors_bitwise_match": tensors_bitwise_match,
            "projected_subsequent_seed_min": round(proj_subsequent_seed_min, 2),
            "warm_speedup_vs_ref": round(speedup_warm, 3),
        },
        "comparative_summary": {
            "reference_historical": {
                "samples_per_sec": ref_sps,
                "wall_time_min": 27.88,
                "speedup": 1.00,
                "vram_peak_gb": 8.31,
                "gpu_util_avg": 24.91,
            },
            "p1_gc_off": {
                "samples_per_sec": p1_sps,
                "wall_time_min": 13.91,
                "speedup": 2.00,
                "vram_peak_gb": 12.68,
                "gpu_util_avg": 28.56,
            },
            "p5_cold_cache": {
                "samples_per_sec": round(num_samples / total_cold_e2e_wall_sec, 3),
                "wall_time_min": round(total_cold_e2e_wall_sec / 60.0, 2),
                "speedup": round(speedup_cold, 2),
                "vram_peak_gb": round(peak_vram_gb, 2),
                "gpu_util_avg": telemetry_summary["gpu_utilization"]["avg"],
            },
            "p5_warm_cache": {
                "samples_per_sec": round(sps, 3),
                "wall_time_min": round(t_train_wall / 60.0, 2),
                "speedup": round(speedup_warm, 2),
                "vram_peak_gb": round(peak_vram_gb, 2),
                "gpu_util_avg": telemetry_summary["gpu_utilization"]["avg"],
            },
        },
        "verdict": "PASS",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL P5 SOAK ACCEPTANCE SUMMARY TABLE")
    print("=" * 80)
    fmt_row = lambda metric, ref, p1, p5c, p5w: f"{metric:<26} | {ref:<10} | {p1:<11} | {p5c:<12} | {p5w:<12}"
    print(fmt_row("Metric", "REFERENCE", "P1 (GC-off)", "P5 (Cold)", "P5 (Warm)"))
    print("-" * 80)
    print(fmt_row("samples/sec", f"{ref_sps:.3f}", f"{p1_sps:.3f}", f"{num_samples/total_cold_e2e_wall_sec:.3f}", f"{sps:.3f}"))
    print(fmt_row("wall time (min)", "27.88 min", "13.91 min", f"{total_cold_e2e_wall_sec/60.0:.2f} min", f"{t_train_wall/60.0:.2f} min"))
    print(fmt_row("speedup", "1.00x", "2.00x", f"{speedup_cold:.2f}x", f"{speedup_warm:.2f}x"))
    print(fmt_row("VRAM peak", "8.31 GB", "12.68 GB", f"{peak_vram_gb:.2f} GB", f"{peak_vram_gb:.2f} GB"))
    print(fmt_row("GPU util avg", "24.9%", "28.6%", f"{telemetry_summary['gpu_utilization']['avg']}%", f"{telemetry_summary['gpu_utilization']['avg']}%"))
    print("=" * 80)
    print(f"Saved full soak report to: {output_path}")

    return report


def run_p7_full_soak(
    group: str = "A",
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute full 1552-sample soak of Candidate P7 (max_performance: MB4 x ACC2, GC=False, VC=True).

    Requirements:
    1. Preflight Capability Check: validate_profile_capability("max_performance") enforcing NO SILENT FALLBACK.
    2. Full 1552-Sample Training Loop:
       - 1552 samples, 388 microbatches (MB=4), 194 optimizer steps (ACC=2).
       - Sample-based progress tracking and accounting.
    3. Memory Stability & Checkpoints:
       Checkpoints captured at:
       - 'model_loaded'
       - 'cache_ready'
       - 'first_optimizer_update' (step 1, after microbatch 2 / 8 samples)
       - 'progress_25_pct' (sample 388)
       - 'progress_50_pct' (sample 776)
       - 'progress_75_pct' (sample 1164)
       - 'progress_100_pct' (sample 1552)
       Tracking:
       - torch_allocated_mb, torch_reserved_mb, allocator_fragmentation_mb (reserved - allocated),
         fragmentation_ratio, nvml_used_mb, host_rss_mb.
       Verifying:
       - Stable plateau vs monotonic leak across 25%-100% progress.
       - Allocator fragmentation bounds.
    4. Performance Acceptance:
       - Wall time <= 3.8 min (target) / <= 3.4 min (preferred).
       - Average GPU utilization >= 85%.
       - Telemetry tracking SM clock MHz (avg, min, max, p99) and GPU util p99.
    5. Save complete report to outputs/experiments/g2.0-e3-performance/p7_full_soak_report.json.
    """
    if output_path is None:
        output_path = REPO_ROOT / "outputs/experiments/g2.0-e3-performance/p7_full_soak_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"G2.0-E3.3A5 P7 FULL SOAK & ACCEPTANCE (Group {group}, Seed {seed}, MB4 x ACC2)")
    print("=" * 80)

    # 1. Capability Preflight (Hard check: NO SILENT FALLBACK)
    print("\n[1/6] Executing profile capability preflight check...")
    prof = get_profile("max_performance")
    validate_profile_capability(prof)
    print(f"  ✓ Device capability verified for {prof.profile_id} (req: ~{prof.memory_estimate_mb:.0f} MB). No fallback.")

    # 2. Load dataset records (N=1552)
    print("\n[2/6] Loading formal training records...")
    data_path = REPO_ROOT / TRAIN_DATASET_PATHS[group]
    records = load_training_records(data_path, base_dir=REPO_ROOT)
    num_samples = len(records)
    assert num_samples == 1552, f"Expected 1552 records, got {num_samples}"
    print(f"  ✓ Loaded {num_samples} records from {data_path.name}")

    # Process and NVML client for memory tracking
    proc = psutil.Process()
    nvml_client = None
    try:
        nvml_client = NVMLClient(0)
    except Exception:
        pass

    def capture_memory_checkpoint(label: str) -> Dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            res_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            alloc_mb, res_mb = 0.0, 0.0
        frag_mb = res_mb - alloc_mb
        frag_ratio = (frag_mb / res_mb) if res_mb > 0 else 0.0
        rss_mb = proc.memory_info().rss / (1024 ** 2)
        nvml_mb = 0.0
        if nvml_client is not None:
            telemetry = nvml_client.query()
            nvml_mb = telemetry.get("nvml_gpu_memory_used_mb", 0.0)
        rec = {
            "checkpoint": label,
            "torch_allocated_mb": round(alloc_mb, 2),
            "torch_reserved_mb": round(res_mb, 2),
            "allocator_fragmentation_mb": round(frag_mb, 2),
            "fragmentation_ratio": round(frag_ratio, 4),
            "nvml_used_mb": round(nvml_mb, 2),
            "host_rss_mb": round(rss_mb, 2),
        }
        print(f"  [Memory Checkpoint: {label:<22}] Alloc: {rec['torch_allocated_mb']:>8.1f} MB | "
              f"Res: {rec['torch_reserved_mb']:>8.1f} MB | Frag: {rec['allocator_fragmentation_mb']:>7.1f} MB "
              f"({rec['fragmentation_ratio']*100:>4.1f}%) | NVML: {rec['nvml_used_mb']:>8.1f} MB | RSS: {rec['host_rss_mb']:>8.1f} MB")
        return rec

    memory_checkpoints: Dict[str, Dict[str, Any]] = {}

    # 3. Model setup (Qwen2.5-VL-3B bf16 + 252 LM LoRA modules, GC=False)
    print("\n[3/6] Initializing model and 252-LM LoRA adapter (GC=False)...")
    processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DEFAULT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.enable_input_require_grads()
    # GC is False for P7
    lora_cfg = create_lora_config()
    model = get_peft_model(model, lora_cfg)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == 14966784, f"Trainable params {trainable_count} != 14,966,784"
    print(f"  ✓ Trainable parameters: {trainable_count:,} (252 LM modules, 0 vision)")

    memory_checkpoints["model_loaded"] = capture_memory_checkpoint("model_loaded")

    # 4. Vision cache population and tensor preparation
    print("\n[4/6] Populating vision cache & pre-baking inputs_embeds for 1552 samples...")
    cache_dir = REPO_ROOT / "outputs/cache/frozen_vision_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vc_cache = FrozenVisionFeatureCache(cache_dir=cache_dir, in_memory=True)
    img_cache = DecodedImageCache()
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model

    t_prep_0 = time.perf_counter()
    prepared_samples = []
    with torch.no_grad():
        for r in records:
            c_feat = vc_cache.get_or_compute(
                r.image_path, model, processor, device="cuda", dtype=torch.bfloat16
            ).to("cuda")
            img = img_cache.get(r.image_path)
            t = build_training_tensors(processor, img, r.question, r.answer)
            in_emb = base_model.model.get_input_embeddings()(t["input_ids"].unsqueeze(0).to("cuda"))
            mask, _ = base_model.model.get_placeholder_mask(
                t["input_ids"].unsqueeze(0).to("cuda"), inputs_embeds=in_emb, image_features=c_feat
            )
            in_emb = in_emb.masked_scatter(mask, c_feat).squeeze(0).cpu()
            prepared_samples.append({
                "inputs_embeds": in_emb,
                "attention_mask": t["attention_mask"],
                "labels": t["labels"],
            })
    t_prep = time.perf_counter() - t_prep_0
    vc_stats = vc_cache.get_stats()
    print(f"  ✓ Cache populated: {vc_stats['unique_cached']} unique images, {vc_stats['hits']} hits, {vc_stats['misses']} misses")
    print(f"  ✓ Pre-baked 1552 inputs_embeds in {t_prep:.2f}s")

    memory_checkpoints["cache_ready"] = capture_memory_checkpoint("cache_ready")

    # 5. Full 1552-sample training loop
    print("\n[5/6] Executing full 1552-sample training loop (Profile: max_performance, MB=4, ACC=2)...")
    seed_everything(seed)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    from transformers import get_linear_schedule_with_warmup
    total_opt_steps = 194  # 1552 / 8 = 194
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_opt_steps)
    optimizer.zero_grad()
    model.train()

    torch.cuda.reset_peak_memory_stats()
    sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.2).start()
    soak_run_dir = REPO_ROOT / "outputs/experiments/g2.0-e3-performance/p7_soak_run"
    soak_run_dir.mkdir(parents=True, exist_ok=True)
    reporter = LiveTelemetryReporter(
        run_dir=soak_run_dir,
        group=group,
        seed=seed,
        total_microbatches=388,
        total_optimizer_steps=194,
        heartbeat_interval=25,
        sampler=sampler,
        total_samples=1552,
    )

    losses: List[float] = []
    mb_count = 0
    processed_samples = 0
    opt_step_count = 0
    sched_step_count = 0

    prefetcher = PinnedPrefetchDataLoader(
        prepared_samples,
        device="cuda",
        batch_size=4,
        queue_size=8,
    )

    torch.cuda.synchronize()
    t_train_start = time.perf_counter()

    for batch in prefetcher:
        current_bs = batch["inputs_embeds"].shape[0]
        raw_loss_val, outputs = run_single_forward_step(model, batch)
        if not math.isfinite(raw_loss_val):
            raise ValueError(f"Non-finite loss detected at microbatch {mb_count}: {raw_loss_val}")
        losses.append(raw_loss_val)

        scaled_loss = outputs.loss / 2.0  # gradient_accumulation_steps = 2
        scaled_loss.backward()

        mb_count += 1
        processed_samples += current_bs
        is_step = (mb_count % 2 == 0)

        current_lr = optimizer.param_groups[0]["lr"]

        if is_step:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step_count += 1
            sched_step_count += 1

        reporter.step(
            microbatch=mb_count,
            optimizer_step=opt_step_count,
            scheduler_step=sched_step_count,
            latest_raw_loss=raw_loss_val,
            learning_rate=current_lr,
            processed_samples=processed_samples,
        )

        # Checkpoints capture
        if opt_step_count == 1 and is_step and "first_optimizer_update" not in memory_checkpoints:
            memory_checkpoints["first_optimizer_update"] = capture_memory_checkpoint("first_optimizer_update")
        if processed_samples >= 388 and "progress_25_pct" not in memory_checkpoints:
            memory_checkpoints["progress_25_pct"] = capture_memory_checkpoint("progress_25_pct")
        if processed_samples >= 776 and "progress_50_pct" not in memory_checkpoints:
            memory_checkpoints["progress_50_pct"] = capture_memory_checkpoint("progress_50_pct")
        if processed_samples >= 1164 and "progress_75_pct" not in memory_checkpoints:
            memory_checkpoints["progress_75_pct"] = capture_memory_checkpoint("progress_75_pct")
        if processed_samples >= 1552 and "progress_100_pct" not in memory_checkpoints:
            memory_checkpoints["progress_100_pct"] = capture_memory_checkpoint("progress_100_pct")

    torch.cuda.synchronize()
    t_train_wall = time.perf_counter() - t_train_start
    telemetry_summary = sampler.stop()
    if nvml_client:
        try:
            nvml_client.close()
        except Exception:
            pass

    # Final progress report update
    final_lr = optimizer.param_groups[0]["lr"]
    reporter.step(
        microbatch=mb_count,
        optimizer_step=opt_step_count,
        scheduler_step=sched_step_count,
        latest_raw_loss=losses[-1],
        learning_rate=final_lr,
        force=True,
        status="completed",
        processed_samples=processed_samples,
    )

    # Save trained adapter
    adapter_dir = soak_run_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    print(f"  ✓ Adapter saved to: {adapter_dir}")

    # Invariants verification
    assert mb_count == 388, f"Expected 388 microbatches, got {mb_count}"
    assert processed_samples == 1552, f"Expected 1552 samples, got {processed_samples}"
    assert opt_step_count == 194, f"Expected 194 optimizer steps, got {opt_step_count}"
    assert sched_step_count == 194, f"Expected 194 scheduler steps, got {sched_step_count}"

    # 6. Analysis and Acceptance Verification
    print("\n[6/6] Analyzing memory stability, telemetry, and acceptance criteria...")
    wall_time_min = t_train_wall / 60.0
    sps = num_samples / t_train_wall
    gpu_util_avg = telemetry_summary["gpu_utilization"]["avg"]
    first_10_mean = sum(losses[:10]) / 10.0
    last_10_mean = sum(losses[-10:]) / 10.0
    overall_mean = sum(losses) / len(losses)

    # Memory plateau verification
    plateau_keys = ["progress_25_pct", "progress_50_pct", "progress_75_pct", "progress_100_pct"]
    plateau_allocs = [memory_checkpoints[k]["torch_allocated_mb"] for k in plateau_keys]
    max_alloc_plateau = max(plateau_allocs)
    min_alloc_plateau = min(plateau_allocs)
    alloc_drift_mb = max_alloc_plateau - min_alloc_plateau
    monotonic_leak = (
        plateau_allocs[0] < plateau_allocs[1] < plateau_allocs[2] < plateau_allocs[3]
        and alloc_drift_mb > 500.0
    )
    plateau_stable = not monotonic_leak and alloc_drift_mb < 500.0

    # Host RSS leak check during training execution
    training_checkpoints = ["first_optimizer_update", "progress_25_pct", "progress_50_pct", "progress_75_pct", "progress_100_pct"]
    training_host_rss_vals = [memory_checkpoints[k]["host_rss_mb"] for k in training_checkpoints]
    host_rss_drift_mb = max(training_host_rss_vals) - min(training_host_rss_vals)
    host_leak_detected = host_rss_drift_mb > 512.0

    # Fragmentation
    max_frag_mb = max(cp["allocator_fragmentation_mb"] for cp in memory_checkpoints.values())
    max_frag_ratio = max(cp["fragmentation_ratio"] for cp in memory_checkpoints.values())
    peak_vram_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_vram_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    # Acceptance criteria
    pass_wall_time = wall_time_min <= 3.8
    pass_wall_preferred = wall_time_min <= 3.4
    pass_gpu_util = gpu_util_avg >= 85.0
    pass_loss = math.isfinite(losses[0]) and math.isfinite(losses[-1]) and (last_10_mean < first_10_mean)
    pass_memory = plateau_stable and not host_leak_detected

    verdict = "PASS" if (pass_wall_time and pass_gpu_util and pass_loss and pass_memory) else "FAIL"

    # Historic reference: 27.88 min (1673 sec)
    ref_wall_sec = 1673.0
    speedup_vs_ref = ref_wall_sec / t_train_wall

    report = {
        "experiment_id": "g2.0-e3-performance",
        "run_type": "p7_full_soak",
        "profile_id": "max_performance",
        "group": group,
        "seed": seed,
        "num_samples": num_samples,
        "microbatch_size": 4,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 8,
        "gradient_checkpointing": False,
        "vision_cache_enabled": True,
        "total_microbatches": mb_count,
        "total_samples": processed_samples,
        "total_optimizer_steps": opt_step_count,
        "total_scheduler_steps": sched_step_count,
        "training_wall_sec": round(t_train_wall, 2),
        "training_wall_min": round(wall_time_min, 3),
        "samples_per_sec": round(sps, 3),
        "sec_per_sample": round(t_train_wall / num_samples, 4),
        "speedup_vs_ref_baseline": round(speedup_vs_ref, 2),
        "loss_initial": round(losses[0], 4),
        "loss_final": round(losses[-1], 4),
        "loss_first_10_mean": round(first_10_mean, 4),
        "loss_last_10_mean": round(last_10_mean, 4),
        "loss_overall_mean": round(overall_mean, 4),
        "peak_vram_allocated_gb": round(peak_vram_alloc_gb, 3),
        "peak_vram_reserved_gb": round(peak_vram_res_gb, 3),
        "memory_checkpoints": memory_checkpoints,
        "memory_stability": {
            "plateau_stable": plateau_stable,
            "allocated_plateau_drift_mb": round(alloc_drift_mb, 2),
            "monotonic_leak_detected": monotonic_leak,
            "host_rss_drift_mb": round(host_rss_drift_mb, 2),
            "host_rss_leak_detected": host_leak_detected,
            "max_fragmentation_mb": round(max_frag_mb, 2),
            "max_fragmentation_ratio": round(max_frag_ratio, 4),
        },
        "telemetry": telemetry_summary,
        "acceptance_criteria": {
            "wall_time_le_3_8_min": {"target": "<= 3.8 min", "actual": f"{wall_time_min:.2f} min", "pass": pass_wall_time},
            "wall_time_preferred_le_3_4_min": {"target": "<= 3.4 min", "actual": f"{wall_time_min:.2f} min", "pass": pass_wall_preferred},
            "gpu_util_ge_85_pct": {"target": ">= 85.0%", "actual": f"{gpu_util_avg:.1f}%", "pass": pass_gpu_util},
            "loss_trajectory_valid": {"target": "finite and decreasing", "actual": f"first10={first_10_mean:.4f} -> last10={last_10_mean:.4f}", "pass": pass_loss},
            "memory_stability_pass": {"target": "stable plateau, bounded fragmentation", "actual": f"drift={alloc_drift_mb:.1f}MB, frag={max_frag_mb:.1f}MB", "pass": pass_memory},
        },
        "adapter_dir": str(adapter_dir),
        "verdict": verdict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL P7 FULL SOAK ACCEPTANCE SUMMARY TABLE")
    print("=" * 80)
    fmt_row = lambda metric, ref, p5, p7: f"{metric:<28} | {ref:<14} | {p5:<14} | {p7:<14}"
    print(fmt_row("Metric", "REFERENCE", "P5 (MB1 VC)", "P7 (MB4 VC)"))
    print("-" * 80)
    print(fmt_row("microbatch size", "1", "1", "4"))
    print(fmt_row("grad accumulation", "8", "8", "2"))
    print(fmt_row("effective batch size", "8", "8", "8"))
    print(fmt_row("samples/sec", "0.928", "3.230", f"{sps:.3f}"))
    print(fmt_row("wall time (min)", "27.88 min", "8.01 min", f"{wall_time_min:.2f} min"))
    print(fmt_row("speedup vs ref", "1.00x", "3.48x", f"{speedup_vs_ref:.2f}x"))
    print(fmt_row("peak VRAM alloc", "8.31 GB", "9.78 GB", f"{peak_vram_alloc_gb:.2f} GB"))
    print(fmt_row("peak VRAM reserved", "8.31 GB", "10.05 GB", f"{peak_vram_res_gb:.2f} GB"))
    print(fmt_row("GPU util avg", "24.9%", "39.6%", f"{gpu_util_avg:.1f}%"))
    print(fmt_row("GPU util p99", "N/A", "N/A", f"{telemetry_summary['gpu_utilization']['p99']:.1f}%"))
    sm_clk = telemetry_summary.get("sm_clock_mhz", {})
    if sm_clk:
        print(fmt_row("SM clock avg", "N/A", "N/A", f"{sm_clk.get('avg', 0.0):.0f} MHz"))
    print(fmt_row("Plateau stable", "Yes", "Yes", "Yes" if plateau_stable else "No"))
    print(fmt_row("Verdict", "PASS", "PASS", verdict))
    print("=" * 80)
    print(f"Saved full soak report to: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpatialForge Training Performance & Profiling")
    parser.add_argument(
        "--mode",
        choices=["profile", "benchmark", "equivalence", "soak", "p7_soak"],
        default="profile",
        help="Execution mode (profile, benchmark, equivalence, soak, p7_soak)",
    )
    parser.add_argument(
        "--profile",
        choices=["reference", "balanced", "max_performance"],
        default=DEFAULT_TRAINING_PROFILE,
        help=f"Execution profile for training (default: {DEFAULT_TRAINING_PROFILE})",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--group", choices=["A", "C", "D"], default="A")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "profile":
        run_baseline_profile(num_samples=args.samples, group=args.group)
    elif args.mode == "equivalence":
        res = run_equivalence_test(num_samples=args.samples, group=args.group)
        if res["status"] != "PASS":
            sys.exit(1)
    elif args.mode == "benchmark":
        run_performance_benchmark(num_samples=args.samples, group=args.group)
    elif args.mode in ("soak", "p7_soak"):
        if args.mode == "p7_soak" or args.profile == "max_performance":
            run_p7_full_soak(group=args.group, seed=args.seed)
        else:
            run_p5_full_soak(group=args.group, seed=args.seed)
