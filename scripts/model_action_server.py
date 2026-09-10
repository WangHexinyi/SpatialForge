"""Torch-side model action server for real closed-loop rollouts.

Loads an external VLM (base or base + LoRA adapter) once and serves greedy
action decisions over localhost HTTP. A dynamic batch scheduler lets several
rollout workers share one GPU efficiently: concurrent requests are collected
into one batched generate call (bounded by ``--max-batch`` and
``--batch-window-ms``) instead of running serially.

Endpoints:
    GET  /health          -> model spec, batch stats
    POST /action          -> {"image_base64", "question"} -> one decision
    POST /action_batch    -> {"items":[{...}, ...]} -> N decisions

Run with the torch interpreter:
    python scripts/model_action_server.py --port 8765 \
        --model-path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
        [--adapter ...] [--max-batch 8] [--batch-window-ms 8]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialforge.embodied.action_parser import parse_action_output  # noqa: E402


def _decode_image(b64: str):
    from PIL import Image

    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class _BatchItem:
    __slots__ = ("image", "question", "event", "result")

    def __init__(self, image, question):
        self.image = image
        self.question = question
        self.event = threading.Event()
        self.result = None

    def set_result(self, result):
        self.result = result
        self.event.set()


class BatchScheduler:
    """Collects concurrent action requests into bounded batched inference."""

    def __init__(self, runner, max_batch: int = 8, window_ms: float = 8.0,
                 max_wait_ms: float = 250.0):
        self.runner = runner
        self.max_batch = max(1, int(max_batch))
        self.window_s = max(0.0, window_ms / 1000.0)
        self.max_wait_s = max(self.window_s, max_wait_ms / 1000.0)
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="BatchScheduler")
        self.batches_served = 0
        self.items_served = 0
        self.batch_sizes = []
        self._stop = threading.Event()

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def submit(self, image, question) -> dict:
        item = _BatchItem(image, question)
        self._queue.put(item)
        if not item.event.wait(timeout=self.max_wait_s + 600.0):
            return {"error": "batch scheduler timeout"}
        return item.result

    def _next_batch(self):
        first = self._queue.get()
        batch = [first]
        deadline = time.perf_counter() + self.window_s
        while len(batch) < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self):
        while not self._stop.is_set():
            try:
                batch = self._next_batch()
            except Exception:
                continue
            try:
                results = self.runner(batch)
            except Exception as exc:  # noqa: BLE001
                results = [{"error": str(exc)} for _ in batch]
            for item, result in zip(batch, results):
                item.set_result(result)
            self.batches_served += 1
            self.items_served += len(batch)
            self.batch_sizes.append(len(batch))


class ActionServerHandler(BaseHTTPRequestHandler):
    server_version = "SpatialForgeModelAction/0.2"

    def log_message(self, fmt, *args):  # quiet
        return

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            self._json(200, SERVER.health())
        elif self.path.startswith("/stats"):
            self._json(200, SERVER.stats())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return
        if self.path.startswith("/action_batch"):
            try:
                items = payload["items"]
                images = [_decode_image(it["image_base64"]) for it in items]
                questions = [str(it["question"]) for it in items]
            except Exception as exc:
                self._json(400, {"error": f"bad request: {exc}"})
                return
            results = SERVER.decide_many(images, questions)
            self._json(200, {"results": results, "batch_size": len(items)})
            return
        if self.path.startswith("/action"):
            try:
                image = _decode_image(payload["image_base64"])
                question = str(payload["question"])
            except Exception as exc:
                self._json(400, {"error": f"bad request: {exc}"})
                return
            self._json(200, SERVER.decide(image, question))
            return
        self._json(404, {"error": "not found"})


class ModelActionServer:
    def __init__(self, model_path: str, adapter: str, max_batch: int = 8,
                 batch_window_ms: float = 8.0, attn: str = "sdpa",
                 info_file: str = None):
        from spatialforge.models.vl_adapter import load_model, load_processor, resolve_spec

        self.spec = resolve_spec(model_path, adapter=adapter, attn_implementation=attn)
        self.processor = load_processor(model_path)
        self.model = load_model(self.spec)
        self.model.eval()
        self.model_label = f"{self.spec.model_id or model_path}" + (
            f" + {adapter}" if adapter else " (base)")
        self.info_file = info_file
        self.calls = 0
        self.total_latency_ms = 0.0
        self._lock = threading.Lock()
        self.scheduler = BatchScheduler(
            self._run_batch, max_batch=max_batch, window_ms=batch_window_ms,
        ).start()
        if info_file:
            try:
                with open(info_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "model": {
                            "model_id": self.spec.model_id or model_path,
                            "model_path": model_path,
                            "family": self.spec.family,
                            "adapter": adapter,
                            "precision": self.spec.dtype,
                            "backend": "transformers",
                            "attn_implementation": self.spec.attn_implementation,
                            "max_batch": max_batch,
                            "batch_window_ms": batch_window_ms,
                        }
                    }, f, indent=2)
            except Exception:
                pass

    def _run_batch(self, batch) -> list:
        import torch

        from spatialforge.models.vl_adapter import format_prompt

        t0 = time.time()
        images = [it.image for it in batch]
        prompts = [format_prompt(self.processor, it.image, it.question) for it in batch]
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=12, do_sample=False,
            )
        out = []
        for i, item in enumerate(batch):
            n_prompt = int(prompt_lens[i])
            text = self.processor.decode(generated[i][n_prompt:], skip_special_tokens=True)
            raw = text.strip()
            out.append({
                "raw": raw,
                "parsed": parse_action_output(raw),
                "latency_ms": round((time.time() - t0) * 1000.0, 1),
            })
        with self._lock:
            self.calls += len(batch)
            self.total_latency_ms += (time.time() - t0) * 1000.0
        return out

    def decide(self, image, question: str) -> dict:
        t0 = time.time()
        result = self.scheduler.submit(image, question)
        result = dict(result or {})
        result["latency_ms"] = round((time.time() - t0) * 1000.0, 1)
        with self._lock:
            result["server_calls"] = self.calls
        return result

    def decide_many(self, images, questions) -> list:
        # one request carrying N observations: submit concurrently via threads
        out = [None] * len(images)
        threads = []

        def worker(i):
            out[i] = self.decide(images[i], questions[i])

        for i in range(len(images)):
            t = threading.Thread(target=worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return out

    def health(self) -> dict:
        return {
            "ok": True,
            "model": self.model_label,
            "spec": self.spec.to_dict(),
            "batch": {
                "max_batch": self.scheduler.max_batch,
                "window_ms": self.scheduler.window_s * 1000.0,
                "batches_served": self.scheduler.batches_served,
                "items_served": self.scheduler.items_served,
                "recent_batch_sizes": self.scheduler.batch_sizes[-50:],
            },
        }

    def stats(self) -> dict:
        with self._lock:
            calls = self.calls
            total = self.total_latency_ms
        sizes = self.scheduler.batch_sizes
        return {
            "calls": calls,
            "mean_batch_latency_ms": round(total / max(self.scheduler.batches_served, 1), 2),
            "batches_served": self.scheduler.batches_served,
            "items_served": self.scheduler.items_served,
            "mean_batch_size": round(sum(sizes) / max(len(sizes), 1), 2),
            "recent_batch_sizes": sizes[-50:],
        }


SERVER = None


def main() -> int:
    global SERVER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-path",
                    default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--batch-window-ms", type=float, default=8.0)
    ap.add_argument("--info-file", default=None)
    ap.add_argument("--ready-file", default=None,
                    help="touch this file once the model is loaded and serving")
    args = ap.parse_args()

    print("[model_server] loading model ...", flush=True)
    SERVER = ModelActionServer(
        args.model_path, args.adapter, max_batch=args.max_batch,
        batch_window_ms=args.batch_window_ms, attn=args.attn,
        info_file=args.info_file,
    )
    srv = ThreadingHTTPServer((args.host, args.port), ActionServerHandler)
    srv.daemon_threads = True
    if args.ready_file:
        with open(args.ready_file, "w") as f:
            f.write("ready")
    print(f"[model_server] ready on http://{args.host}:{args.port} "
          f"model={SERVER.model_label} max_batch={args.max_batch}", flush=True)
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
