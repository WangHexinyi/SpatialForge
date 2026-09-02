"""G0 冒烟测试：加载 Qwen2.5-VL-3B-Instruct 回答一个空间问题。"""
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_DIR = Path("models/Qwen2.5-VL-3B-Instruct")


def make_test_image() -> str:
    """生成左红右蓝测试图，返回绝对路径。"""
    Path("data").mkdir(exist_ok=True)
    img = Image.new("RGB", (512, 256), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([64, 96, 160, 192], fill="red")
    d.rectangle([352, 96, 448, 192], fill="blue")
    out = Path("data/smoke_test.png").resolve()
    img.save(out)
    return str(out)


def main() -> None:
    assert MODEL_DIR.exists(), f"未找到模型目录：{MODEL_DIR}"
    assert torch.cuda.is_available(), "CUDA 不可用"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": make_test_image()},
        {"type": "text", "text": "红色方块是否在蓝色方块的左边？只回答一个字：是或否。"},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=8)
    answer = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"模型回答：{answer.strip()}")
    print(f"显存占用：{torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()