"""GPU-backed ProcTHOR runtime benchmark.

Loads a real house, drives a few genuine environment actions and, while doing
so, samples nvidia-smi to prove the rendering workload is on the NVIDIA GPU.

Usage (in the Python 3.10 venv, with the NVIDIA X display running, e.g. :0):
    python scripts/bench_gpu_thor.py --house <house.json> --x-display :0 \
        [--width 256 --height 256 --steps 6]

Prints renderer identification (glxinfo on the chosen display) and a
nvidia-smi snapshot before/during/after. This is a smoke benchmark, not a
full training framework.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

sys.path.insert(0, ".")

from spatialforge.embodied.backends.procthor import ProcTHORBackend  # noqa: E402
from spatialforge.embodied.contracts import AgentActionType  # noqa: E402
from spatialforge.embodied.environment import EmbodiedEnvironment  # noqa: E402


def glx_renderer(display: str) -> str:
    try:
        out = subprocess.run(
            ["glxinfo", "-B"], env={**__import__("os").environ, "DISPLAY": display},
            capture_output=True, text=True, timeout=20,
        ).stdout
        for line in out.splitlines():
            if "OpenGL renderer string" in line:
                return line.split(":", 1)[1].strip()
    except Exception as e:  # noqa: BLE001
        return f"glxinfo error: {e}"
    return "unknown"


def nvidia_smi_snapshot(label: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"{label}: nvidia-smi error {e}"
    return f"{label}: gpu_util%,mem_MiB,pwr_W -> {out}"


class SMIProbe(threading.Thread):
    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._halt = False

    def run(self):
        while not self._halt:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=15,
                ).stdout.strip()
                self.samples.append(out)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def stop(self):
        self._halt = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", required=True)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    print("renderer:", glx_renderer(args.x_display))
    print(nvidia_smi_snapshot("before"))

    probe = SMIProbe(0.4)
    probe.start()

    t0 = time.time()
    backend = ProcTHORBackend(
        house=args.house, width=args.width, height=args.height,
        quality="Low", x_display=args.x_display,
    )
    info = backend.load_house()
    load_s = time.time() - t0
    print(f"house_load: {load_s:.1f}s  reachable={len(backend.reachable_positions())} "
          f"objects={info.object_count}")

    env = EmbodiedEnvironment(backend, episode_max_steps=args.steps + 2)
    tr = env.start_episode(info.house_id, "Mug", max_steps=args.steps + 2)
    rgb0 = tr.observation.rgb
    t1 = time.time()
    step_times = []
    actions = [AgentActionType.MOVE_AHEAD, AgentActionType.ROTATE_RIGHT,
               AgentActionType.MOVE_AHEAD, AgentActionType.ROTATE_LEFT,
               AgentActionType.LOOK_DOWN, AgentActionType.MOVE_AHEAD]
    for i, a in enumerate(actions[:args.steps]):
        s = time.time()
        tr = env.step(a)
        step_times.append(time.time() - s)
        print(f"  step{i+1}: {a.value} success={tr.action.success} "
              f"rgb_changed={not (rgb0 == tr.observation.rgb).all()}")
    t2 = time.time()

    # Done verifier
    s = time.time()
    fin = env.step(AgentActionType.DONE)
    step_times.append(time.time() - s)
    print(f"  Done verifier success={env.episode.success} reason={env.episode.success_reason}")

    probe.stop()
    probe.join(timeout=5)
    print(nvidia_smi_snapshot("after"))

    n = len(step_times)
    mean_ms = sum(step_times) / n * 1000
    print(f"\nepisode_wall: {t2 - t1:.2f}s  env_steps={n}  "
          f"mean_step={mean_ms:.0f}ms  steps_per_sec={n / max(t2 - t1, 1e-6):.2f}")
    peak_util = max((int(s.split(",")[0].strip()) for s in probe.samples if s), default=0)
    print(f"peak_gpu_util_during_rollout={peak_util}%  smi_samples={len(probe.samples)}")
    print(f"house_load={load_s:.2f}s")
    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
