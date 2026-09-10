"""Torch-side model action server for real closed-loop rollouts.

Loads the Qwen2.5-VL-3B base (or base + BC LoRA adapter) once, serves greedy
action decisions over localhost HTTP using only stdlib ``http.server``. One
decision = one image + BC prompt -> raw text + strict parsed action. Inference
is serialized behind a lock so several rollout workers may share one server.

Run with the torch interpreter (miniconda base):
    python scripts/model_action_server.py --port 8765 \
        [--model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct] \
        [--adapter outputs/embodied_bc/train/run1/adapter]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, ".")

from spatialforge.embodied.action_parser import parse_action_output  # noqa: E402


class ActionServerHandler(BaseHTTPRequestHandler):
    server_version = "SpatialForgeModelAction/0.1"

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
            self._json(200, {"ok": True, "model": SERVER.model_label})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/action"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            img_b64 = payload["image_base64"]
            question = str(payload["question"])
            image = _decode_image(img_b64)
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return
        result = SERVER.decide(image, question)
        self._json(200, result)


def _decode_image(b64: str):
    from PIL import Image

    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class ModelActionServer:
    def __init__(self, model_path: str, adapter: str):
        self.model_label = f"qwen2.5-vl-3b + {adapter}" if adapter else "qwen2.5-vl-3b (base)"
        self._lock = threading.Lock()
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from spatialforge.experiment.training import (
            format_chat_prompt,
            run_inference_greedy,
        )

        self.processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model
        self._infer = run_inference_greedy
        self._format = format_chat_prompt
        self.calls = 0
        self.total_latency_ms = 0.0

    def decide(self, image, question: str) -> dict:
        t0 = time.time()
        with self._lock:  # serialized GPU inference
            raw = self._infer(
                self.model, self.processor, image, question,
                max_new_tokens=12, device="cuda",
            )
        parsed = parse_action_output(raw)
        with self._lock:
            self.calls += 1
            self.total_latency_ms += (time.time() - t0) * 1000.0
        return {
            "raw": raw,
            "parsed": parsed,
            "latency_ms": round((time.time() - t0) * 1000.0, 1),
            "server_calls": self.calls,
        }


SERVER = None


def main() -> int:
    global SERVER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-path",
                    default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--ready-file", default=None,
                    help="touch this file once the model is loaded and serving")
    args = ap.parse_args()

    print("[model_server] loading model ...", flush=True)
    SERVER = ModelActionServer(args.model_path, args.adapter)
    srv = ThreadingHTTPServer((args.host, args.port), ActionServerHandler)
    srv.daemon_threads = True
    if args.ready_file:
        with open(args.ready_file, "w") as f:
            f.write("ready")
    print(f"[model_server] ready on http://{args.host}:{args.port} "
          f"model={SERVER.model_label}", flush=True)
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
