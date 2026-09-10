"""Researcher-side orchestration for embodied BC rollout runs.

Parent-side logic shared by the concurrency autotune sweep and the dataset
generation run: spawns one ``scripts/embodied_bc_worker.py`` subprocess per slot
(venv python), partitions a fixed deterministic job list round-robin, collects
per-job JSONL results, monitors heartbeats, respawns crashed shards, and samples
GPU (NVML) + CPU (psutil) telemetry. Never imports ai2thor.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

DEFAULT_VENV_PYTHON = "/root/autodl-tmp/.venv-procthor/bin/python"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "embodied_bc_worker.py")


def partition_jobs(jobs: List[Dict[str, Any]], num_slots: int) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_slots)]
    for i, job in enumerate(jobs):
        shards[i % num_slots].append(job)
    return shards


def load_result_rows(results_file: str) -> List[Dict[str, Any]]:
    rows = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_all_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    by_job: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(results_dir):
        return by_job
    for name in sorted(os.listdir(results_dir)):
        if not name.startswith("results-"):
            continue
        for row in load_result_rows(os.path.join(results_dir, name)):
            by_job[row["job_id"]] = row
    return by_job


def _row_quantiles(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    s = sorted(vals)
    n = len(s)
    def q(p):
        return s[min(n - 1, int(p / 100.0 * (n - 1)))]
    return {
        "count": n,
        "mean_ms": round(sum(s) / n, 1),
        "p50_ms": round(q(50), 1),
        "p95_ms": round(q(95), 1),
        "max_ms": round(q(100), 1),
    }


def aggregate_results(rows: List[Dict[str, Any]], wall_s: float) -> Dict[str, Any]:
    """Aggregate per-job rows into the standard throughput summary."""
    total_steps = sum(int(r.get("steps", 0)) for r in rows)
    successes = [r for r in rows if r.get("success")]
    success_steps = sum(int(r.get("steps", 0)) for r in successes)
    episodes = len(rows)
    crashed = [r for r in rows if r.get("crashed")]
    step_times: List[float] = []
    for r in rows:
        step_times.extend([float(x) for x in (r.get("env_step_times") or [])])
    wall = max(wall_s, 0.001)
    outcomes: Dict[str, int] = {}
    for r in rows:
        code = str(r.get("outcome_code") or r.get("status") or "?")
        outcomes[code] = outcomes.get(code, 0) + 1
    return {
        "episodes": episodes,
        "successful_episodes": len(successes),
        "total_env_steps": total_steps,
        "wall_sec": round(wall, 1),
        "aggregate_steps_per_sec": round(total_steps / wall, 3),
        "episodes_per_min": round(episodes / wall * 60.0, 2),
        "successful_episodes_per_min": round(len(successes) / wall * 60.0, 2),
        "env_step_latency_ms": _row_quantiles(step_times),
        "mean_episode_wall_s": round(
            sum(float(r.get("elapsed_s", 0)) for r in rows) / max(len(rows), 1), 2),
        "crashed_jobs": len(crashed),
        "outcome_codes": outcomes,
    }


class RunTelemetry:
    """Background sampler: NVML GPU stats + psutil CPU/RAM + worker process info."""

    def __init__(self):
        try:
            from spatialforge.experiment.performance import GpuTelemetrySampler
            self.gpu_sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.3).start()
        except Exception:
            self.gpu_sampler = None
        import psutil
        self.psutil = psutil
        self.cpu_util_samples: List[float] = []
        self.ram_samples: List[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.cpu_util_samples.append(self.psutil.cpu_percent(interval=None))
            self.ram_samples.append(self.psutil.virtual_memory().percent)
            self._stop.wait(0.5)

    def _summary(self, arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"avg": 0.0, "p95": 0.0, "max": 0.0}
        s = sorted(arr)
        n = len(s)
        return {
            "avg": round(sum(s) / n, 1),
            "p95": round(s[min(n - 1, int(0.95 * (n - 1)))], 1),
            "max": round(s[-1], 1),
        }

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        self._t.join(timeout=2)
        gpu = self.gpu_sampler.stop() if self.gpu_sampler else None
        return {
            "gpu": gpu,
            "cpu": {"overall": self._summary(self.cpu_util_samples)},
            "ram": self._summary(self.ram_samples),
        }


def run_shard(
    python: str,
    slot: int,
    shard_jobs: List[Dict[str, Any]],
    results_dir: str,
    hb_dir: str,
    persist: bool,
    out_root: Optional[str],
    persist_root: Optional[str],
    width: int,
    x_display: str,
    require_nvidia: bool,
    tag: str,
    env: Optional[Dict[str, str]] = None,
    worker_script: str = WORKER_SCRIPT,
    extra_args: Optional[List[str]] = None,
) -> int:
    """Run one shard's jobs in a fresh worker subprocess. Returns returncode."""
    jobs_file = os.path.join(hb_dir, f"slot-{slot:02d}-jobs.json")
    with open(jobs_file, "w", encoding="utf-8") as f:
        json.dump({"jobs": shard_jobs}, f)
    cmd = [
        python, worker_script,
        "--jobs", jobs_file, "--slot", str(slot),
        "--results", os.path.join(results_dir, f"results-{slot:02d}.jsonl"),
        "--hb-dir", hb_dir,
        "--width", str(width), "--x-display", x_display,
    ]
    if worker_script.endswith("embodied_bc_worker.py"):
        cmd.append("--tag")
        cmd.append(tag)
    if require_nvidia:
        cmd.append("--require-nvidia")
    if persist:
        cmd.append("--persist")
        cmd.extend(["--out-root", persist_root or out_root])
    cmd.extend(extra_args or [])
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    err_log = open(os.path.join(hb_dir, f"slot-{slot:02d}.stderr.log"), "w")
    proc = subprocess.Popen(cmd, env=run_env, stdout=subprocess.DEVNULL,
                            stderr=err_log, start_new_session=True)
    proc._sf_err = err_log
    return proc


def run_worker_pool(
    jobs: List[Dict[str, Any]],
    num_workers: int,
    out_dir: str,
    python: str = DEFAULT_VENV_PYTHON,
    persist: bool = False,
    persist_root: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    x_display: str = ":0",
    require_nvidia: bool = True,
    tag: str = "run",
    job_timeout_s: float = 600.0,
    max_respawns_per_slot: int = 2,
    env: Optional[Dict[str, str]] = None,
    worker_script: str = WORKER_SCRIPT,
    extra_worker_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full job pool over ``num_workers`` slots with crash respawn.

    Returns aggregated results + telemetry summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    results_dir = os.path.join(out_dir, "results")
    hb_dir = os.path.join(out_dir, "heartbeats")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(hb_dir, exist_ok=True)

    tele = RunTelemetry()
    t0 = time.time()

    # partition once; on respawn we filter out already-finished job ids
    shards = partition_jobs(jobs, num_workers)

    done: Dict[str, Dict[str, Any]] = {}
    processes: Dict[int, subprocess.Popen] = {}
    respawns: Dict[int, int] = {}
    last_hb_ts: Dict[int, float] = {}
    # monotonic staleness tracking (immune to NTP wall-clock jumps)
    hb_last_seen_mono: Dict[int, float] = {}
    hb_last_mtime: Dict[int, float] = {}
    lock = threading.Lock()

    def remaining(shard: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [j for j in shard if j["job_id"] not in done]

    def spawn_all() -> None:
        for slot in range(num_workers):
            shard = remaining(shards[slot])
            if not shard:
                continue
            p = run_shard(python, slot, shard, results_dir, hb_dir, persist,
                          out_dir, persist_root, width, x_display,
                          require_nvidia, tag, env, worker_script,
                          extra_worker_args)
            processes[slot] = p
            last_hb_ts[slot] = time.time()
            hb_last_seen_mono[slot] = time.monotonic()
            hb_last_mtime[slot] = 0.0

    spawn_all()

    while processes:
        time.sleep(2.0)
        # harvest results
        rows = load_all_results(results_dir)
        with lock:
            for row in rows.values():
                done[row["job_id"]] = row
        finished = True
        for slot, proc in list(processes.items()):
            rc = proc.poll()
            if rc is None:
                finished = False
                # per-job watchdog: a worker stuck on one job (heartbeat stale
                # beyond job_timeout_s) is killed and its shard respawned.
                # Staleness is measured with time.monotonic() so NTP wall-clock
                # jumps can neither fake liveness nor cause spurious kills.
                hb_file = os.path.join(hb_dir, f"slot-{slot:02d}.json")
                if os.path.exists(hb_file):
                    try:
                        with open(hb_file) as f:
                            hb = json.load(f)
                        mtime = os.path.getmtime(hb_file)
                        if hb.get("state") == "running":
                            if mtime > hb_last_mtime.get(slot, 0.0):
                                hb_last_mtime[slot] = mtime
                                hb_last_seen_mono[slot] = time.monotonic()
                            if (time.monotonic() - hb_last_seen_mono.get(slot, time.monotonic())
                                    > job_timeout_s):
                                _log(f"[pool] slot {slot} job {hb.get('job_id')} exceeded "
                                     f"{int(job_timeout_s)}s watchdog; killing slot")
                                try:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                                # fall through to respawn handling below
                    except Exception:
                        pass
                if proc.poll() is None:
                    continue
            processes.pop(slot, None)
            # worker exited (or was watchdog-killed); respawn unfinished jobs
            shard = remaining(shards[slot])
            if not shard:
                continue
            if respawns.get(slot, 0) >= max_respawns_per_slot:
                # leave the unfinished jobs recorded as missing (aggregated later)
                continue
            respawns[slot] = respawns.get(slot, 0) + 1
            _log(f"[pool] slot {slot} exited rc={rc}; respawn {respawns[slot]} "
                 f"({len(shard)} jobs left)")
            p = run_shard(python, slot, shard, results_dir, hb_dir, persist,
                          out_dir, persist_root, width, x_display,
                          require_nvidia, tag, env, worker_script,
                          extra_worker_args)
            processes[slot] = p
            last_hb_ts[slot] = time.time()
        if finished and not processes:
            break

    # graceful: kill any stragglers (their results already harvested)
    for proc in processes.values():
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    wall = time.time() - t0
    tele_agg = tele.stop()
    rows = list(done.values())
    # mark jobs that never completed (respawn budget exhausted / stragglers)
    missing = [j["job_id"] for j in jobs if j["job_id"] not in done]
    agg = aggregate_results(rows, wall)
    return {
        "num_workers": num_workers,
        "jobs_total": len(jobs),
        "jobs_completed": len(rows),
        "jobs_missing": missing,
        "aggregate": agg,
        "telemetry": tele_agg,
        "respawns": respawns,
        "out_dir": out_dir,
    }


def _log(msg: str) -> None:
    print(f"[run] {msg}", flush=True)
