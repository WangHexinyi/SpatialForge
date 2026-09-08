"""GPU telemetry, live progress reporting, and profiling utilities.

Provides lightweight, non-intrusive monitoring of GPU utilization, memory, and power
via direct NVML ctypes calls without subprocess overhead.
Also provides atomic live progress reporting (heartbeat + progress.json) and
fine-grained stage profiling instrumentation.
"""

from dataclasses import asdict, dataclass
import ctypes
from ctypes import byref, c_int, c_uint, c_ulonglong, Structure
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# =====================================================================
# NVML Ctypes Wrapper (Subprocess-free, low-overhead GPU metrics)
# =====================================================================

class _NVMLUtilization(Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class _NVMLMemory(Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


class NVMLClient:
    """Singleton-style low-overhead NVML client using libnvidia-ml.so."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.available = False
        self._handle = None
        self._lib = None
        self._lock = threading.Lock()

        try:
            for soname in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
                try:
                    self._lib = ctypes.CDLL(soname)
                    break
                except OSError:
                    continue

            if self._lib is not None:
                ret = self._lib.nvmlInit_v2()
                if ret == 0:
                    handle = ctypes.c_void_p()
                    ret_dev = self._lib.nvmlDeviceGetHandleByIndex_v2(device_index, byref(handle))
                    if ret_dev == 0:
                        self._handle = handle
                        self.available = True
        except Exception:
            self.available = False

    def query(self) -> Dict[str, float]:
        """Query instantaneous GPU statistics. Thread-safe and fast (<50us)."""
        if not self.available or self._lib is None or self._handle is None:
            return {
                "gpu_utilization_pct": 0.0,
                "gpu_memory_used_mb": 0.0,
                "gpu_memory_total_mb": 0.0,
                "nvml_gpu_memory_used_mb": 0.0,
                "nvml_gpu_memory_total_mb": 0.0,
                "gpu_power_w": 0.0,
                "sm_clock_mhz": 0.0,
            }

        with self._lock:
            try:
                util = _NVMLUtilization()
                mem = _NVMLMemory()
                power = c_uint()
                clock = c_uint()
                sm_clock = 0.0

                self._lib.nvmlDeviceGetUtilizationRates(self._handle, byref(util))
                self._lib.nvmlDeviceGetMemoryInfo(self._handle, byref(mem))
                self._lib.nvmlDeviceGetPowerUsage(self._handle, byref(power))
                if hasattr(self._lib, "nvmlDeviceGetClockInfo"):
                    ret_clk = self._lib.nvmlDeviceGetClockInfo(self._handle, 1, byref(clock))
                    if ret_clk == 0:
                        sm_clock = float(clock.value)

                return {
                    "gpu_utilization_pct": float(util.gpu),
                    "gpu_memory_used_mb": float(mem.used / (1024 * 1024)),
                    "gpu_memory_total_mb": float(mem.total / (1024 * 1024)),
                    "nvml_gpu_memory_used_mb": float(mem.used / (1024 * 1024)),
                    "nvml_gpu_memory_total_mb": float(mem.total / (1024 * 1024)),
                    "gpu_power_w": float(power.value / 1000.0),
                    "sm_clock_mhz": sm_clock,
                }
            except Exception:
                return {
                    "gpu_utilization_pct": 0.0,
                    "gpu_memory_used_mb": 0.0,
                    "gpu_memory_total_mb": 0.0,
                    "nvml_gpu_memory_used_mb": 0.0,
                    "nvml_gpu_memory_total_mb": 0.0,
                    "gpu_power_w": 0.0,
                    "sm_clock_mhz": 0.0,
                }

    def close(self) -> None:
        """Shutdown NVML if initialized."""
        if self.available and self._lib is not None:
            with self._lock:
                try:
                    self._lib.nvmlShutdown()
                except Exception:
                    pass
                self.available = False


class GpuTelemetrySampler:
    """Background sampling thread collecting GPU utilization, memory, and power stats."""

    def __init__(self, device_index: int = 0, sample_interval_sec: float = 0.2):
        self.device_index = device_index
        self.sample_interval_sec = sample_interval_sec
        self.nvml = NVMLClient(device_index)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self.util_samples: List[float] = []
        self.power_samples: List[float] = []
        self.mem_samples: List[float] = []
        self.clock_samples: List[float] = []
        self.latest_stats: Dict[str, float] = self.nvml.query()

    def start(self) -> "GpuTelemetrySampler":
        """Start the background sampler thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="GpuTelemetrySampler")
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.is_set():
            stats = self.nvml.query()
            with self._lock:
                self.latest_stats = stats
                self.util_samples.append(stats["gpu_utilization_pct"])
                self.power_samples.append(stats["gpu_power_w"])
                self.mem_samples.append(stats["gpu_memory_used_mb"])
                self.clock_samples.append(stats.get("sm_clock_mhz", 0.0))
            self._stop_event.wait(self.sample_interval_sec)

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return aggregated summary."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.get_summary()

    def get_latest(self) -> Dict[str, float]:
        """Get latest sampled stats."""
        with self._lock:
            return dict(self.latest_stats)

    def get_summary(self) -> Dict[str, Any]:
        """Compute statistical summary of recorded telemetry samples."""
        with self._lock:
            utils = list(self.util_samples)
            powers = list(self.power_samples)
            mems = list(self.mem_samples)
            clocks = list(self.clock_samples)

        def _stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
            s = sorted(arr)
            n = len(s)
            p50_idx = int(0.50 * (n - 1))
            p95_idx = int(0.95 * (n - 1))
            p99_idx = int(0.99 * (n - 1))
            return {
                "avg": round(sum(s) / n, 2),
                "p50": round(s[p50_idx], 2),
                "p95": round(s[p95_idx], 2),
                "p99": round(s[p99_idx], 2),
                "min": round(s[0], 2),
                "max": round(s[-1], 2),
            }

        return {
            "sample_count": len(utils),
            "gpu_utilization": _stats(utils),
            "gpu_power_w": _stats(powers),
            "gpu_memory_mb": _stats(mems),
            "nvml_gpu_memory_mb": _stats(mems),
            "sm_clock_mhz": _stats(clocks),
        }

    def close(self) -> None:
        self.stop()
        self.nvml.close()


# =====================================================================
# Atomic Live Telemetry / Heartbeat Reporter
# =====================================================================

def write_progress_file_atomic(file_path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Atomically write JSON data to file using temp file + flush + fsync + replace."""
    dest_path = Path(file_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.parent / f".{dest_path.name}.tmp.{os.getpid()}"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # fsync may not be supported on some virtual filesystems

    os.replace(temp_path, dest_path)


class LiveTelemetryReporter:
    """Thread-safe, non-intrusive live progress telemetry reporter.

    Emits human-readable heartbeat console output every `heartbeat_interval` microbatches
    and atomically updates `progress.json` in the active run output directory.
    Guaranteed NOT to affect RNG state or synchronize CUDA needlessly.
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        group: str,
        seed: Optional[int],
        total_microbatches: int,
        total_optimizer_steps: int,
        heartbeat_interval: int = 50,
        sampler: Optional[GpuTelemetrySampler] = None,
        tag: str = "TRAIN",
        enabled: bool = True,
        total_samples: Optional[int] = None,
    ):
        self.run_dir = Path(run_dir)
        self.group = group
        self.seed = seed
        self.total_microbatches = total_microbatches
        self.total_samples = total_samples if total_samples is not None else total_microbatches
        self.total_optimizer_steps = total_optimizer_steps
        self.heartbeat_interval = max(1, heartbeat_interval)
        self.sampler = sampler
        self.tag = tag
        self.enabled = enabled

        self.progress_json_path = self.run_dir / "progress.json"
        self.t_start = time.time()
        self.last_report_time = self.t_start
        self.last_report_mb = 0

    def compute_progress_metrics(
        self,
        microbatch: int,
        optimizer_step: int,
        scheduler_step: int,
        latest_raw_loss: float,
        learning_rate: float,
        status: str = "running",
        processed_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute deterministic progress and rate statistics without side effects."""
        current_samples = processed_samples if processed_samples is not None else microbatch
        elapsed = max(0.001, time.time() - self.t_start)
        samples_per_sec = current_samples / elapsed
        remaining_samples = max(0, self.total_samples - current_samples)
        eta_sec = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0.0
        progress_pct = (current_samples / max(self.total_samples, 1)) * 100.0

        gpu_stats = self.sampler.get_latest() if self.sampler is not None else {}
        gpu_util = gpu_stats.get("gpu_utilization_pct", 0.0)
        nvml_mem_used = gpu_stats.get("nvml_gpu_memory_used_mb", gpu_stats.get("gpu_memory_used_mb", 0.0))
        nvml_mem_total = gpu_stats.get("nvml_gpu_memory_total_mb", gpu_stats.get("gpu_memory_total_mb", 0.0))
        gpu_power = gpu_stats.get("gpu_power_w", 0.0)

        # PyTorch memory tracking if CUDA is available
        torch_allocated = 0.0
        torch_reserved = 0.0
        torch_peak_allocated = 0.0
        torch_peak_reserved = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                torch_allocated = float(torch.cuda.memory_allocated() / (1024 * 1024))
                torch_reserved = float(torch.cuda.memory_reserved() / (1024 * 1024))
                torch_peak_allocated = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                torch_peak_reserved = float(torch.cuda.max_memory_reserved() / (1024 * 1024))
        except Exception:
            pass

        now_iso = datetime.now(timezone.utc).isoformat()

        return {
            "status": status,
            "group": self.group,
            "seed": self.seed,
            "microbatch": microbatch,
            "total_microbatches": self.total_microbatches,
            "processed_samples": current_samples,
            "total_samples": self.total_samples,
            "optimizer_step": optimizer_step,
            "total_optimizer_steps": self.total_optimizer_steps,
            "scheduler_step": scheduler_step,
            "progress_pct": round(progress_pct, 2),
            "latest_raw_loss": round(float(latest_raw_loss), 4),
            "learning_rate": learning_rate,
            "samples_per_sec": round(samples_per_sec, 2),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta_sec, 1),
            "gpu_utilization_pct": round(gpu_util, 1),
            "gpu_memory_used_mb": round(nvml_mem_used if nvml_mem_used > 0.0 else torch_allocated, 1),
            "gpu_memory_peak_mb": round(torch_peak_allocated, 1),
            "nvml_gpu_memory_used_mb": round(nvml_mem_used, 1),
            "nvml_gpu_memory_total_mb": round(nvml_mem_total, 1),
            "torch_memory_allocated_mb": round(torch_allocated, 1),
            "torch_memory_reserved_mb": round(torch_reserved, 1),
            "torch_peak_allocated_mb": round(torch_peak_allocated, 1),
            "torch_peak_reserved_mb": round(torch_peak_reserved, 1),
            "gpu_power_w": round(gpu_power, 1),
            "updated_at": now_iso,
        }

    def emit_heartbeat(
        self,
        metrics: Dict[str, Any],
    ) -> None:
        """Emit one concise console status line."""
        grp_seed = f"{self.group}{self.seed if self.seed is not None else ''}"
        msg = (
            f"[{grp_seed} {self.tag}] "
            f"sample {metrics['processed_samples']}/{metrics['total_samples']} "
            f"mb {metrics['microbatch']}/{metrics['total_microbatches']} "
            f"optimizer {metrics['optimizer_step']}/{metrics['total_optimizer_steps']} "
            f"scheduler {metrics['scheduler_step']}/{metrics['total_optimizer_steps']} "
            f"progress {metrics['progress_pct']:.1f}% "
            f"raw_loss {metrics['latest_raw_loss']:.4f} "
            f"lr {metrics['learning_rate']:.2e} "
            f"samples_per_sec {metrics['samples_per_sec']:.2f} "
            f"elapsed {metrics['elapsed_seconds']:.1f}s "
            f"ETA {metrics['eta_seconds']:.1f}s "
            f"GPU util {metrics['gpu_utilization_pct']:.0f}% "
            f"NVML {metrics['nvml_gpu_memory_used_mb']:.0f}MB / Torch alloc {metrics['torch_memory_allocated_mb']:.0f}MB (peak {metrics['torch_peak_allocated_mb']:.0f}MB) "
            f"power {metrics['gpu_power_w']:.1f}W"
        )
        print(msg, flush=True)

    def step(
        self,
        microbatch: int,
        optimizer_step: int,
        scheduler_step: int,
        latest_raw_loss: float,
        learning_rate: float,
        force: bool = False,
        status: str = "running",
        processed_samples: Optional[int] = None,
    ) -> None:
        """Call after each microbatch; emits heartbeat and writes progress.json on interval."""
        if not self.enabled:
            return

        is_interval = (microbatch % self.heartbeat_interval == 0)
        is_final = (microbatch >= self.total_microbatches)

        if force or is_interval or is_final:
            metrics = self.compute_progress_metrics(
                microbatch=microbatch,
                optimizer_step=optimizer_step,
                scheduler_step=scheduler_step,
                latest_raw_loss=latest_raw_loss,
                learning_rate=learning_rate,
                status=status,
                processed_samples=processed_samples,
            )
            self.emit_heartbeat(metrics)
            write_progress_file_atomic(self.progress_json_path, metrics)
            self.last_report_mb = microbatch
            self.last_report_time = time.time()


# =====================================================================
# Profiling Instrumentation (Phase 1 Detailed Stage Latency Breakdown)
# =====================================================================

@dataclass
class StageTimer:
    """High-resolution stage latency recorder with optional CUDA synchronization."""

    name: str
    cuda_sync: bool = False
    durations: List[float] = None

    def __post_init__(self):
        if self.durations is None:
            self.durations = []
        self._t0 = 0.0

    def start(self) -> None:
        if self.cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self.cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        dt = time.perf_counter() - self._t0
        self.durations.append(dt)
        return dt

    def summary(self) -> Dict[str, float]:
        if not self.durations:
            return {"total_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
        s = sorted(self.durations)
        n = len(s)
        total_ms = sum(s) * 1000.0
        mean_ms = (sum(s) / n) * 1000.0
        med_ms = s[int(0.50 * (n - 1))] * 1000.0
        p95_ms = s[int(0.95 * (n - 1))] * 1000.0
        return {
            "total_ms": round(total_ms, 2),
            "mean_ms": round(mean_ms, 3),
            "median_ms": round(med_ms, 3),
            "p95_ms": round(p95_ms, 3),
        }
