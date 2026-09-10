import time
import unittest

from spatialforge.experiment.performance import GpuTelemetrySampler


class _FakeNvml:
    def __init__(self):
        self.util = 10.0

    def query(self):
        self.util += 5.0
        return {
            "gpu_utilization_pct": self.util,
            "gpu_power_w": 100.0,
            "gpu_memory_used_mb": 1000.0,
            "sm_clock_mhz": 1500.0,
        }

    def close(self):
        pass


class TelemetryPhaseTests(unittest.TestCase):
    def test_phase_markers_split_hot_loop_from_prep(self):
        sampler = GpuTelemetrySampler(device_index=0, sample_interval_sec=0.01)
        sampler.nvml = _FakeNvml()
        sampler.start()
        sampler.set_phase("vision_cache_prep")
        time.sleep(0.08)
        sampler.set_phase("train_active")
        time.sleep(0.08)
        summary = sampler.stop()
        phases = summary["phases"]
        self.assertIn("vision_cache_prep", phases)
        self.assertIn("train_active", phases)
        prep = phases["vision_cache_prep"]["gpu_utilization"]["avg"]
        train = phases["train_active"]["gpu_utilization"]["avg"]
        self.assertGreater(train, prep)
        self.assertGreater(phases["train_active"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
