"""Embodied Inspector HTTP server (new mainline UI).

Serves the bilingual Inspector frontend and a small API to drive a live
object-search episode through the embodied runtime. This is a fresh server
kept separate from the legacy ``server.py`` so the old God-View/trajectory
surface is not destabilized.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from spatialforge.i18n import get_dictionary, supported_languages
from spatialforge.inspector.embodied_session import EmbodiedSession

WEB_DIR = Path(__file__).resolve().parent / "embodied_web"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODES_ROOT = _REPO_ROOT / "outputs" / "episodes"


def _query(url: str) -> Dict[str, str]:
    q = parse_qs(urlparse(url).query)
    return {k: v[0] for k, v in q.items()}


class _Handler(BaseHTTPRequestHandler):
    session: Optional[EmbodiedSession] = None

    # ---- helpers ---------------------------------------------------
    def _send_json(self, payload, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, name: str) -> None:
        root = WEB_DIR
        target = (root / name).resolve()
        if not str(target).startswith(str(root.resolve())) or not target.is_file():
            self.send_error(404)
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), ctype)

    def _read_json_body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # ---- routing ---------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send_static("index.html")
            elif path == "/styles.css":
                self._send_static("styles.css")
            elif path == "/app.js":
                self._send_static("app.js")
            elif path == "/api/backends":
                self._send_json(_Handler.session.available_backends())
            elif path == "/api/i18n":
                self._send_json(
                    {
                        "dictionary": get_dictionary(),
                        "languages": supported_languages(),
                        "default": "bilingual",
                    }
                )
            elif path == "/api/state":
                self._send_json(_Handler.session.state())
            elif path == "/api/frame":
                png = _Handler.session.observation_image_png()
                if png is None:
                    self._send_json({"error": "no frame"}, 404)
                else:
                    self._send_bytes(png, "image/png")
            elif path == "/api/teacher":
                self._send_json(_Handler.session.teacher_path())
            elif path == "/api/episodes/list":
                self._send_json({"episodes": _Handler.session.list_episodes()})
            elif path == "/api/episodes/state":
                idx = _query(self.path).get("step", 0)
                self._send_json(_Handler.session.episode_state(idx))
            elif path == "/api/episodes/frame":
                q = _query(self.path)
                if q.get("frame"):
                    png = _Handler.session.episode_frame_by_name(str(q.get("frame")))
                else:
                    png = _Handler.session.episode_frame_png(int(q.get("step", 0)))
                if png is None:
                    self._send_json({"error": "no episode frame"}, 404)
                else:
                    self._send_bytes(png, "image/png")
            else:
                self._send_json({"error": "not found", "path": path}, 404)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_json_body()
        try:
            if path == "/api/house/load":
                self._ensure_session(body.get("backend"), body.get("seed", 0))
                info = _Handler.session.load_house(
                    body.get("house_id", ""), seed=body.get("seed", 0)
                )
                self._send_json({"house": info, "backend": _Handler.session.backend_name})
            elif path == "/api/episode/start":
                self._ensure_session(body.get("backend"), body.get("seed", 0))
                if not _Handler.session.house():
                    _Handler.session.load_house(
                        body.get("house_id", ""), seed=body.get("seed", 0)
                    )
                fwd = {
                    k: body[k]
                    for k in ("instruction", "start", "max_steps", "seed")
                    if body.get(k) is not None
                }
                state = _Handler.session.start(
                    body.get("target_category", "mug"), **fwd
                )
                self._send_json(state)
            elif path == "/api/action/step":
                state = _Handler.session.step(body.get("action_type", ""))
                self._send_json(state)
            elif path == "/api/episodes/load":
                state = _Handler.session.load_episode(body.get("episode_id", ""))
                self._send_json(state)
            else:
                self._send_json({"error": "not found", "path": path}, 404)
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": str(e)}, 500)

    def _ensure_session(self, backend: Optional[str], seed: int) -> None:
        backend = backend or "deterministic"
        if (
            _Handler.session is None
            or _Handler.session.backend_name != backend
            or (seed is not None and seed != getattr(_Handler.session, "_seed", None))
        ):
            _Handler.session = EmbodiedSession(backend, seed=int(seed or 0))

    def log_message(self, fmt, *args):  # noqa: A003
        pass


def create_embodied_server(
    host: str = "127.0.0.1",
    port: int = 8100,
    episodes_root: Optional[str] = None,
):
    _Handler.session = EmbodiedSession("deterministic", seed=0)
    if episodes_root is None:
        episodes_root = os.environ.get("SF_EPISODES_ROOT") or str(DEFAULT_EPISODES_ROOT)
    _Handler.session.set_episodes_root(episodes_root)
    server = ThreadingHTTPServer((host, port), _Handler)
    return server


if __name__ == "__main__":  # pragma: no cover
    import sys

    host = "127.0.0.1"
    port = 8100
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    print(f"SpatialForge Embodied Inspector on http://{host}:{port}")
    create_embodied_server(host, port).serve_forever()
