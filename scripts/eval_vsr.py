r"""通用 VLM 空间评测（旗标参数化版）。

用法：
    python scripts/eval_vsr.py --test <jsonl> --out <jsonl> [--adapter <dir>] [--model <dir>]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from spatialforge.schema import load_samples


def parse_yesno(text: str):
    """宽松解析 yes/no；无法判定返回 None。"""
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    samples = load_samples(Path(args.test))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print("[info] adapter loaded:", args.adapter)
    processor = AutoProcessor.from_pretrained(args.model)

    stats = defaultdict(lambda: [0, 0])
    n_missing = n_unparsed = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fout:
        for s in tqdm(samples):
            img = Path(s.images[0])
            if not img.exists():
                n_missing += 1
                continue
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(img.resolve())},
                {"type": "text", "text": s.question}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, padding=True,
                               return_tensors="pt").to("cuda")
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=8)
            pred = parse_yesno(processor.decode(
                ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            if pred is None:
                n_unparsed += 1
            correct = int(pred == s.gt)
            for d in s.dims + ["overall"]:
                stats[d][1] += 1
                stats[d][0] += correct
            fout.write(json.dumps({"id": s.id, "gt": s.gt, "pred": pred,
                                   "dims": s.dims,
                                   "relation": s.meta.get("relation", "")},
                                  ensure_ascii=False) + "\n")

    print(f"missing={n_missing} unparsed={n_unparsed}")
    for k, (c, t) in sorted(stats.items()):
        print(f"{k}: {c}/{t} = {100 * c / t:.1f}")


if __name__ == "__main__":
    main()