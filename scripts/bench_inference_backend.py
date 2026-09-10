"""Inference backend benchmark: Transformers vs vLLM (same model/prompt).

Runs the same real BC question on the same frame with either backend and
records single-request latency, batched throughput, VRAM and output
correctness (strict-parsed action). Run with the matching interpreter:

    python scripts/bench_inference_backend.py --backend transformers ...
    /root/autodl-tmp/.venv-vllm/bin/python scripts/bench_inference_backend.py --backend vllm ...
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
from spatialforge.embodied.bc_dataset import build_bc_question  # noqa: E402


def _find_image() -> str:
    for root in ("outputs/embodied_bc/dataset_v2", "outputs/embodied_bc/dataset"):
        candidates = sorted(Path(root).glob("**/frames/frame_0000.png"))
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError("no frame_0000.png found")


def _percentiles(vals):
    s = sorted(vals)
    n = len(s)
    return {
        "mean": round(sum(s) / max(n, 1), 1),
        "p50": round(s[n // 2], 1) if s else None,
        "p95": round(s[min(n - 1, int(0.95 * (n - 1)))], 1) if s else None,
    }


def bench_transformers(args):
    import torch
    from spatialforge.experiment.training import load_and_preprocess_image
    from spatialforge.models.vl_adapter import (
        format_prompt, infer_greedy, load_model, load_processor, resolve_spec,
    )

    spec = resolve_spec(args.model_path, adapter=args.adapter)
    processor = load_processor(args.model_path)
    model = load_model(spec)
    model.eval()
    image = load_and_preprocess_image(args.image)
    question = args.question

    latencies = []
    raws = []
    for _ in range(args.runs):
        t0 = time.time()
        raw = infer_greedy(model, processor, image, question, max_new_tokens=12)
        latencies.append((time.time() - t0) * 1000.0)
        raws.append(raw)

    # batched throughput (dynamic-batch server equivalent)
    batch_latencies = []
    batch_actions = []
    if args.batch > 1:
        prompts = [format_prompt(processor, image, question) for _ in range(args.batch)]
        images = [image] * args.batch
        for _ in range(max(1, args.runs // 2)):
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            if inputs["pixel_values"].dtype == torch.float32:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            lens = inputs["attention_mask"].sum(dim=1).tolist()
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=12, do_sample=False)
            batch_latencies.append((time.time() - t0) * 1000.0)
            batch_actions = [
                parse_action_output(processor.decode(gen[i][int(lens[i]):],
                                                     skip_special_tokens=True).strip())
                for i in range(args.batch)
            ]
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {
        "backend": "transformers",
        "model": args.model_path,
        "adapter": args.adapter,
        "runs": args.runs,
        "single_latency_ms": _percentiles(latencies),
        "raws": raws,
        "valid_rate": round(sum(1 for r in raws if parse_action_output(r)["action"])
                            * 100.0 / max(len(raws), 1), 2),
        "batch": args.batch,
        "batch_latency_ms": _percentiles(batch_latencies) if batch_latencies else None,
        "batch_items_per_sec": round(
            args.batch / (sum(batch_latencies) / max(len(batch_latencies), 1) / 1000.0), 3
        ) if batch_latencies else None,
        "batch_actions": [a["action"] for a in batch_actions],
        "peak_vram_gb": round(peak, 2),
    }


def bench_vllm(args):
    from PIL import Image
    from vllm import LLM, SamplingParams

    image = Image.open(args.image).convert("RGB")
    question = args.question
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 1},
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    load_s = time.time() - t0
    sampling = SamplingParams(max_tokens=12, temperature=0.0)

    def make_inputs(n):
        return [[{
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": question},
            ],
        }] for _ in range(n)]

    # warmup + single
    llm.chat(make_inputs(1), sampling)
    latencies = []
    raws = []
    for _ in range(args.runs):
        t0 = time.time()
        outs = llm.chat(make_inputs(1), sampling)
        latencies.append((time.time() - t0) * 1000.0)
        raws.append(outs[0].outputs[0].text.strip())

    batch_latencies = []
    batch_actions = []
    if args.batch > 1:
        for _ in range(max(1, args.runs // 2)):
            t0 = time.time()
            outs = llm.chat(make_inputs(args.batch), sampling)
            batch_latencies.append((time.time() - t0) * 1000.0)
            batch_actions = [parse_action_output(o.outputs[0].text.strip()) for o in outs]
    return {
        "backend": "vllm",
        "model": args.model_path,
        "load_seconds": round(load_s, 1),
        "runs": args.runs,
        "single_latency_ms": _percentiles(latencies),
        "raws": raws,
        "valid_rate": round(sum(1 for r in raws if parse_action_output(r)["action"])
                            * 100.0 / max(len(raws), 1), 2),
        "batch": args.batch,
        "batch_latency_ms": _percentiles(batch_latencies) if batch_latencies else None,
        "batch_items_per_sec": round(
            args.batch / (sum(batch_latencies) / max(len(batch_latencies), 1) / 1000.0), 3
        ) if batch_latencies else None,
        "batch_actions": [a["action"] for a in batch_actions],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    ap.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--question", default=None)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", default="outputs/diagnostics/inference_backend")
    args = ap.parse_args()

    args.image = args.image or _find_image()
    args.question = args.question or build_bc_question(
        "Find a mug.", "mug", 0.0, [], max_steps=100, current_step=0,
    )
    if args.backend == "transformers":
        report = bench_transformers(args)
    else:
        report = bench_vllm(args)
    report["image"] = os.path.abspath(args.image)
    report["question"] = args.question
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"inference_{args.backend}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print("[inference_bench] " + json.dumps(report, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
