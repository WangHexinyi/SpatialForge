"""SpatialForge 统一样本 schema：适配器与评测/模型层之间的唯一中间表示。"""
import json
from dataclasses import dataclass, field
from typing import List, Optional

VALID_DIMS = {"left_right", "front_back", "vertical", "near_far", "topology",
              "contact", "orientation", "part_whole", "count", "between",
              "occlusion", "other"}


@dataclass
class Sample:
    """一条统一的空间推理样本。"""
    id: str                      # 全局唯一，建议 f"{source}_{原始id}"
    source: str                  # 来源基准名，如 "vsr"
    images: List[str]            # 图片路径/URL 列表（单图即长度为 1）
    question: str
    options: Optional[List[str]] # 选择题选项；开放题为 None
    gt: str                      # 标准答案
    dims: List[str] = field(default_factory=list)  # 空间维度标签（多标签）
    meta: dict = field(default_factory=dict)   # 存放来源基准的原始字段

    def __post_init__(self) -> None:
        assert self.images, "images 不得为空"
        bad = set(self.dims) - VALID_DIMS
        assert not bad, f"非法维度标签：{bad}"


def dump_samples(samples: List[Sample], path: str) -> None:
    """序列化至 JSONL。"""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")


def load_samples(path: str) -> List[Sample]:
    """从 JSONL 反序列化。"""
    with open(path, encoding="utf-8") as f:
        return [Sample(**json.loads(line)) for line in f]