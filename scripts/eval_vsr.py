"""G1 评测：Qwen2.5-VL-3B 全量跑 VSR test，输出总分与分维度准确率。"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使包导入成立

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from spatialforge.schema import load_samples

MODEL_DIR = Path("models/Qwen2.5-VL-3B-Instruct")
TEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/vsr_test.jsonl")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/vsr_results.jsonl")


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


def main() -> None:
    samples = load_samples(TEST)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="auto")
    if len(sys.argv) > 3:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, sys.argv[3])
        print("[info] adapter loaded:", sys.argv[3])
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    stats = defaultdict(lambda: [0, 0])
    n_missing = n_unparsed = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT, "w", encoding="utf-8") as fout:
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