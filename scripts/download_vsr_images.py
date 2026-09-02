"""下载 VSR test 集所需 COCO 图像（并发、跳过已存在、失败可重跑）。"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROCESSED = Path("data/processed/vsr_test.jsonl")
OUT_DIR = Path("data/vsr_images")


def fetch(item: tuple) -> str:
    """下载单张图像；已存在则跳过。"""
    name, link = item
    out = OUT_DIR / name
    if out.exists():
        return "skip"
    try:
        r = requests.get(link, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        return "ok"
    except Exception as e:  # 单张失败不中断整体
        print(f"[fail] {name}: {e}")
        return "fail"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = {}
    for line in open(PROCESSED, encoding="utf-8"):
        s = json.loads(line)
        items[Path(s["images"][0]).name] = s["meta"]["image_link"]
    print(f"unique images: {len(items)}")
    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(fetch, items.items()))
    print(f"done: ok={results.count('ok')} skip={results.count('skip')} "
          f"fail={results.count('fail')}")


if __name__ == "__main__":
    main()