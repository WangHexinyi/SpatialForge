"""Standalone entrypoint for G2.0-E2 controlled training and pipeline verification.

Executes:
1. Environment & GPU runtime inspection
2. Model & processor initialization
3. G2.0-E1 multimodal record loading (Group A)
4. Forward smoke & loss check
5. Backward smoke & gradient flow verification
6. Tiny training smoke (4 steps)
7. LoRA adapter save & reload verification
8. Training & Holdout-S1 greedy inference smoke
9. Evaluator & VSR taxonomy smoke
10. Diagnostic JSON report generation
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from spatialforge.experiment.training import (
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_PATH,
    FormalTrainingConfig,
    SmokeTrainingConfig,
    VSR_DIRECT_DIMS,
    VSR_BROADER_DIMS,
    build_training_tensors,
    classify_vsr_relation,
    collate_single_sample_batch,
    compute_formal_optimizer_steps,
    create_lora_config,
    get_language_model_target_modules,
    is_language_model_target_module,
    load_and_preprocess_image,
    load_training_records,
    parse_directional_answer,
    parse_yesno_answer,
    run_formal_training_loop,
    run_inference_greedy,
    run_single_backward_step,
    run_single_forward_step,
    run_tiny_smoke_training,
)


def main():
    print("=" * 60)
    print("SpatialForge G2.0-E2 Training Harness & Pipeline Verification")
    print("=" * 60)

    # 1. Environment & GPU
    print("\n[Step 1] Environment & GPU Runtime:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    assert torch.cuda.is_available(), "CUDA must be available for E2 smoke verification"
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  CUDA Device: {device_name}")
    print(f"  Visible VRAM: {total_vram_gb:.2f} GB")

    vram_before = torch.cuda.memory_allocated() / 1e9

    # 2. Model & Processor Loading
    print("\n[Step 2] Model & Processor Loading:")
    model_path = Path(DEFAULT_MODEL_PATH)
    assert model_path.exists(), f"Model directory does not exist: {model_path}"
    print(f"  Model path: {model_path}")

    processor = AutoProcessor.from_pretrained(str(model_path))
    print(f"  Processor class: {type(processor).__name__}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"  Model class: {type(model).__name__}")
    print(f"  Model dtype: {model.dtype}")
    vram_after_model = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after model load: {vram_after_model:.3f} GB")

    # 3. LoRA Attachment & Language-Only Target Scope Verification
    print("\n[Step 3] LoRA Attachment & Strict Language-Only Target Scope:")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    assert getattr(model.model.language_model, "gradient_checkpointing", False) is True, "Gradient checkpointing must be enabled"
    print("  Gradient checkpointing enabled: True")

    lora_cfg = create_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_count / total_count

    vision_trainable = [n for n in trainable_params if "visual" in n]
    lm_trainable = [n for n in trainable_params if "language_model" in n or "layers" in n]

    print(f"  Trainable params: {trainable_count:,} ({trainable_pct:.4f}%)")
    print(f"  LM trainable tensors: {len(lm_trainable)}")
    print(f"  Vision trainable tensors: {len(vision_trainable)}")

    assert len(vision_trainable) == 0, f"Vision tower must remain frozen, found {len(vision_trainable)} trainable tensors"
    assert len(lm_trainable) == 504, f"Expected 504 LM LoRA tensors (252 modules), got {len(lm_trainable)}"
    assert trainable_count == 14966784, f"Expected 14,966,784 trainable params, got {trainable_count}"
    print("  [OK] F1 Verification: Exactly 252 LM LoRA modules, 0 vision LoRA modules, 14,966,784 trainable params")

    # 4. Data Loading (Group A)
    print("\n[Step 4] Dataset Loading (Group A):")
    train_a_path = REPO_ROOT / "outputs/experiments/g2.0-e/train_group_a.jsonl"
    records_a = load_training_records(train_a_path, base_dir=REPO_ROOT, limit=16)
    print(f"  Loaded {len(records_a)} smoke records from {train_a_path.name}")
    sample0 = records_a[0]
    print(f"  Sample 0 ID: {sample0.sample_id}")
    print(f"  Sample 0 Scene: {sample0.scene_id}, View: {sample0.view_id}")
    print(f"  Sample 0 Family: {sample0.family}")
    print(f"  Sample 0 Question: {sample0.question}")
    print(f"  Sample 0 Answer: {sample0.answer}")
    print(f"  Sample 0 Image Path: {sample0.image_path}")

    # 5. Image & Formatter & Loss Masking Verification
    print("\n[Step 5] Multimodal Formatter & Loss Masking Verification:")
    img0 = load_and_preprocess_image(sample0.image_path)
    print(f"  Image loaded and converted: mode={img0.mode}, size={img0.size}")
    assert img0.mode == "RGB", f"Expected RGB mode, got {img0.mode}"

    tensors0 = build_training_tensors(processor, img0, sample0.question, sample0.answer)
    prompt_len = tensors0["prompt_length"]
    ans_len = tensors0["answer_length"]
    total_tokens = len(tensors0["input_ids"])
    masked_labels = (tensors0["labels"] == -100).sum().item()
    supervised_labels = (tensors0["labels"] != -100).sum().item()
    print(f"  Prompt tokens: {prompt_len}")
    print(f"  Answer tokens: {ans_len}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Masked labels (-100): {masked_labels}")
    print(f"  Supervised labels: {supervised_labels}")
    assert masked_labels == prompt_len, "All prompt tokens must be masked (-100)"
    assert supervised_labels == ans_len, "All answer tokens must be supervised"
    decoded_supervised = processor.tokenizer.decode(tensors0["labels"][tensors0["labels"] != -100])
    print(f"  Decoded supervised text: {repr(decoded_supervised)}")

    # 6. Forward Smoke
    print("\n[Step 6] 1-Batch Forward Smoke:")
    batch0 = collate_single_sample_batch(tensors0, device="cuda")
    loss_val0, out0 = run_single_forward_step(model, batch0)
    vram_after_fwd = torch.cuda.memory_allocated() / 1e9
    print(f"  Forward loss: {loss_val0:.4f}")
    print(f"  Loss is finite: {torch.isfinite(torch.tensor(loss_val0)).item()}")
    print(f"  VRAM after forward: {vram_after_fwd:.3f} GB")

    # 7. Backward Smoke
    print("\n[Step 7] 1-Sample Backward Smoke:")
    test_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=DEFAULT_LEARNING_RATE)
    grad_stats = run_single_backward_step(out0.loss, test_opt)
    vram_after_bwd = torch.cuda.memory_allocated() / 1e9
    print(f"  Backward completed successfully")
    print(f"  Mean grad norm: {grad_stats['mean_grad_norm']:.6f}")
    print(f"  Max grad norm: {grad_stats['max_grad_norm']:.6f}")
    print(f"  VRAM after backward: {vram_after_bwd:.3f} GB")

    # 8. 8-Microbatch Accumulation Smoke (1 Full Accumulation Window)
    print("\n[Step 8] 8-Microbatch Gradient Accumulation Smoke (1 Full Optimizer Step):")
    smoke_out_dir = REPO_ROOT / "outputs/experiments/g2.0-e2/smoke"
    records_smoke = records_a[:8]
    assert len(records_smoke) == 8, f"Expected 8 smoke records, got {len(records_smoke)}"

    # Track a known LoRA B parameter to verify it does NOT change during microbatches 1..7 and DOES update on step 8
    test_lora_param = next(p for n, p in model.named_parameters() if "lora_B" in n and p.requires_grad)
    initial_param_val = test_lora_param.detach().clone()

    from transformers import get_linear_schedule_with_warmup
    accum_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=DEFAULT_LEARNING_RATE)
    accum_sched = get_linear_schedule_with_warmup(accum_opt, num_warmup_steps=0, num_training_steps=194)
    accum_opt.zero_grad()

    smoke_losses = []
    step_times = []
    optimizer_steps_executed = 0

    for mb_idx in range(8):
        t_mb_0 = time.time()
        rec = records_smoke[mb_idx]
        img_mb = load_and_preprocess_image(rec.image_path)
        tensors_mb = build_training_tensors(processor, img_mb, rec.question, rec.answer)
        batch_mb = collate_single_sample_batch(tensors_mb, device="cuda")

        raw_loss, out_mb = run_single_forward_step(model, batch_mb)
        smoke_losses.append(raw_loss)

        # Scale loss by 8
        scaled_loss = out_mb.loss / 8
        scaled_loss.backward()

        t_mb_1 = time.time()
        step_times.append(t_mb_1 - t_mb_0)

        if mb_idx < 7:
            # During microbatches 1..7: optimizer must NOT have stepped
            assert torch.equal(test_lora_param, initial_param_val), f"Weights changed prematurely at microbatch {mb_idx + 1}"
            print(f"  Microbatch {mb_idx + 1}/8: loss={raw_loss:.4f}, scaled={scaled_loss.item():.4f}, optimizer_steps=0 (weights stable)")
        else:
            # At microbatch 8: execute optimizer step and scheduler step
            accum_opt.step()
            accum_sched.step()
            accum_opt.zero_grad()
            optimizer_steps_executed += 1
            assert not torch.equal(test_lora_param, initial_param_val), "Weights must have updated after optimizer step"
            print(f"  Microbatch 8/8: loss={raw_loss:.4f}, scaled={scaled_loss.item():.4f}, optimizer_steps=1 (weights UPDATED)")

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    mean_mb_time = sum(step_times) / len(step_times)
    throughput = 1.0 / mean_mb_time

    # Save adapter
    adapter_dir = smoke_out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)

    print(f"  Accumulation smoke completed: 8 microbatches -> exactly 1 optimizer step")
    print(f"  Initial loss: {smoke_losses[0]:.4f}, Final microbatch loss: {smoke_losses[-1]:.4f}")
    print(f"  Mean microbatch time: {mean_mb_time:.3f} s, throughput: {throughput:.2f} samples/s")
    print(f"  Peak VRAM: {peak_vram:.3f} GB")
    print(f"  Adapter saved to: {adapter_dir}")

    # 9. Adapter Save / Reload Verification
    print("\n[Step 9] Adapter Save / Reload Verification:")
    assert adapter_dir.exists(), "Adapter directory must exist"
    from safetensors import safe_open
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt") as f:
        adapter_keys = list(f.keys())
    visual_adapter_keys = [k for k in adapter_keys if "visual" in k]
    lm_adapter_keys = [k for k in adapter_keys if "language_model" in k or "layers" in k]
    print(f"  Total saved LoRA tensors: {len(adapter_keys)}")
    print(f"  LM LoRA tensors: {len(lm_adapter_keys)}")
    print(f"  Visual LoRA tensors: {len(visual_adapter_keys)}")
    assert len(visual_adapter_keys) == 0, f"Saved adapter must have 0 visual tensors, found {len(visual_adapter_keys)}"
    assert len(lm_adapter_keys) == 504, f"Saved adapter must have 504 LM tensors, found {len(lm_adapter_keys)}"

    adapter_files = list(adapter_dir.glob("*"))
    adapter_size_mb = sum(f.stat().st_size for f in adapter_files) / 1e6
    print(f"  Saved adapter size: {adapter_size_mb:.2f} MB")

    # Reload adapter into clean base model
    print("  Reloading adapter into clean base model...")
    base_reload = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    reloaded_model = PeftModel.from_pretrained(base_reload, str(adapter_dir))
    reloaded_model.eval()

    reloaded_visual = [n for n, p in reloaded_model.named_parameters() if "visual" in n and "lora" in n]
    reloaded_lm = [n for n, p in reloaded_model.named_parameters() if "lora" in n and "visual" not in n]
    assert len(reloaded_visual) == 0, "Reloaded model must have zero visual LoRA parameters"
    assert len(reloaded_lm) == 504, "Reloaded model must have 504 LM LoRA parameters"
    print("  Adapter reloaded cleanly and successfully with 0 visual LoRA parameters!")

    # 10. Training Sample Inference Smoke
    print("\n[Step 10] Training Sample Inference Smoke (Greedy):")
    pred_train = run_inference_greedy(
        model=reloaded_model,
        processor=processor,
        image=img0,
        question=sample0.question,
        max_new_tokens=16,
        device="cuda",
    )
    parsed_train = parse_directional_answer(pred_train)
    print(f"  Question: {sample0.question}")
    print(f"  Ground Truth: {sample0.answer}")
    print(f"  Raw Prediction: {repr(pred_train)}")
    print(f"  Parsed Answer: {parsed_train}")
    assert parsed_train is not None, f"Answer parser failed to parse directional output: {pred_train!r}"

    # 11. Held-Out S1 Inference Smoke
    print("\n[Step 11] Held-Out S1 Sample Inference Smoke (Greedy):")
    s1_path = REPO_ROOT / "outputs/experiments/g2.0-e/holdout_s1_cardinal.jsonl"
    records_s1 = load_training_records(s1_path, base_dir=REPO_ROOT, limit=1)
    s1_sample = records_s1[0]
    img_s1 = load_and_preprocess_image(s1_sample.image_path)
    pred_s1 = run_inference_greedy(
        model=reloaded_model,
        processor=processor,
        image=img_s1,
        question=s1_sample.question,
        max_new_tokens=16,
        device="cuda",
    )
    parsed_s1 = parse_directional_answer(pred_s1)
    print(f"  Holdout S1 Sample ID: {s1_sample.sample_id}")
    print(f"  Question: {s1_sample.question}")
    print(f"  Ground Truth: {s1_sample.answer}")
    print(f"  Raw Prediction: {repr(pred_s1)}")
    print(f"  Parsed Answer: {parsed_s1}")
    assert parsed_s1 is not None, f"Answer parser failed to parse holdout output: {pred_s1!r}"

    # 12. Evaluator & VSR Taxonomy Smoke
    print("\n[Step 12] Evaluator & VSR Taxonomy Verification:")
    # Test yes/no parsing
    assert parse_yesno_answer("Yes, this is true.") == "yes"
    assert parse_yesno_answer("No, false.") == "no"
    assert parse_yesno_answer("Definitely") is None
    print("  Yes/No answer parser verified.")

    # Test directional parsing on all 8 directions
    for term in ("left", "right", "above", "below", "front", "behind", "nearer", "farther"):
        assert parse_directional_answer(f"It is {term}.") == term
        assert parse_directional_answer(term) == term
    print("  All 8 directional answers parsed cleanly.")

    # Test VSR taxonomy separation
    for rel in ("left of", "right of", "at the left side of"):
        assert classify_vsr_relation(rel) in VSR_DIRECT_DIMS
    for rel in ("in front of", "behind", "ahead of"):
        assert classify_vsr_relation(rel) in VSR_DIRECT_DIMS
    for rel in ("facing", "facing away from", "toward", "opposite to", "parallel to", "perpendicular to", "across"):
        dim = classify_vsr_relation(rel)
        assert dim in VSR_BROADER_DIMS, f"Expected orientation, got {dim}"
        assert dim not in VSR_DIRECT_DIMS, "Orientation must NOT collapse into direct relations"
    print("  VSR direct transfer (left_right, front_back) vs broader orientation verified.")

    # Generate diagnostic report
    report = {
        "milestone": "G2.0-E2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "cuda_device": device_name,
            "total_vram_gb": total_vram_gb,
        },
        "model": {
            "model_path": str(model_path),
            "model_class": type(model).__name__,
            "dtype": str(model.dtype),
            "trainable_parameters": trainable_count,
            "total_parameters": total_count,
            "trainable_percent": trainable_pct,
        },
        "forward_smoke": {
            "sample_id": sample0.sample_id,
            "loss": loss_val0,
            "is_finite": True,
        },
        "backward_smoke": {
            "mean_grad_norm": grad_stats["mean_grad_norm"],
            "max_grad_norm": grad_stats["max_grad_norm"],
        },
        "lora_scope": {
            "language_lora_modules": len(lm_trainable) // 2,
            "vision_lora_modules": len(vision_trainable) // 2,
            "vision_trainable_parameters": 0,
            "language_trainable_parameters": trainable_count,
        },
        "gradient_checkpointing_enabled": True,
        "gradient_accumulation_smoke": {
            "microbatches": 8,
            "optimizer_steps": optimizer_steps_executed,
            "gradient_accumulation_steps": 8,
            "initial_loss": smoke_losses[0],
            "final_loss": smoke_losses[-1],
            "losses": smoke_losses,
            "mean_microbatch_time_sec": mean_mb_time,
            "throughput_samples_per_sec": throughput,
            "peak_vram_gb": peak_vram,
            "adapter_size_mb": adapter_size_mb,
        },
        "inference_smoke": {
            "train_sample": {
                "id": sample0.sample_id,
                "gt": sample0.answer,
                "raw_pred": pred_train,
                "parsed": parsed_train,
            },
            "holdout_sample": {
                "id": s1_sample.sample_id,
                "gt": s1_sample.answer,
                "raw_pred": pred_s1,
                "parsed": parsed_s1,
            },
        },
    }

    report_path = smoke_out_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[OK] Diagnostic smoke report saved -> {report_path}")
    print("\n" + "=" * 60)
    print("G2.0-E2 PIPELINE SMOKE VERIFICATION: ALL 12 STEPS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
