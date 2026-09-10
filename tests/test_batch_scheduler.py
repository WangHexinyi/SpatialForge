import threading
import time
import unittest

import importlib.util
from pathlib import Path

# scripts/ is not a package; load the module by path (importing it would require torch)
_spec = importlib.util.spec_from_file_location(
    "model_action_server_mod",
    str(Path(__file__).resolve().parents[1] / "scripts" / "model_action_server.py"),
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


class BatchSchedulerTests(unittest.TestCase):
    def test_results_match_requests(self):
        seen = []

        def runner(batch):
            seen.append(len(batch))
            return [{"id": it.question} for it in batch]

        sched = mod.BatchScheduler(runner, max_batch=4, window_ms=15.0).start()
        try:
            results = [None] * 8
            threads = []
            for i in range(8):
                def worker(i=i):
                    results[i] = sched.submit(object(), f"q{i}")
                t = threading.Thread(target=worker, daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            self.assertEqual([r["id"] for r in results], [f"q{i}" for i in range(8)])
            self.assertTrue(seen)
            self.assertLessEqual(max(seen), 4)
        finally:
            sched.stop()

    def test_single_request_batch(self):
        sched = mod.BatchScheduler(lambda b: [{"ok": True}], max_batch=8, window_ms=1.0).start()
        try:
            self.assertEqual(sched.submit(object(), "q")["ok"], True)
            self.assertEqual(sched.items_served, 1)
        finally:
            sched.stop()

    def test_runner_error_propagates_as_error_result(self):
        def bad(batch):
            raise RuntimeError("boom")

        sched = mod.BatchScheduler(bad, max_batch=2, window_ms=1.0).start()
        try:
            res = sched.submit(object(), "q")
            self.assertIn("error", res)
        finally:
            sched.stop()


if __name__ == "__main__":
    unittest.main()
