r"""G4：按场景划分合成集为 train(000-079) / holdout(080-099)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatialforge.schema import Sample, load_samples, dump_samples

SRC = Path("data/processed/synth_test.jsonl")
TRAIN = Path("data/processed/synth_train.jsonl")
HOLD = Path("data/processed/synth_holdout.jsonl")


def scene_of(s: Sample) -> int:
    """从 id（scene_080_q12）解析场景号。"""
    return int(s.id.split("_")[1])


def main():
    samples = load_samples(SRC)
    train = [s for s in samples if scene_of(s) < 80]
    hold = [s for s in samples if scene_of(s) >= 80]
    dump_samples(train, TRAIN)
    dump_samples(hold, HOLD)
    print("train=%d hold=%d" % (len(train), len(hold)))


if __name__ == "__main__":
    main()