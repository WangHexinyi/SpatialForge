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
VENDOR_DIR = Path(__file__).resolve().parent / "web" / "vendor"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODES_ROOT = _REPO_ROOT / "outputs" / "episodes"


def _query(url: str) -> Dict[str, str]:
    q = parse_qs(urlparse(url).query)
    return {k: v[0] for k, v in q.items()}


def _camera_from_query(q: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Build an explicit ThirdPartyCamera pose from browser query params.

    The browser owns the camera state (position/rotation/target/distance/mode)
    and sends the resolved pose; the server only applies it. Presets (follow /
    overview) are used when ``cam_mode`` is absent or the pose is incomplete.
    """

    def f(key):
        try:
            return float(q.get(key))
        except (TypeError, ValueError):
            return None

    mode = q.get("cam_mode")
    if not mode:
        return None
    px, py, pz = f("px"), f("py"), f("pz")
    rx, ry, rz = f("rx"), f("ry"), f("rz")
    if None in (px, py, pz) or None in (rx, ry, rz):
        return None
    return {
        "mode": mode,
        "position": {"x": px, "y": py, "z": pz},
        "rotation": {"x": rx, "y": ry, "z": rz},
        "fieldOfView": f("fov") if f("fov") is not None else 70.0,
        "target": {"x": f("tx") or 0.0, "y": f("ty") or 0.0, "z": f("tz") or 0.0},
        "distance": f("dist"),
    }


class _Handler(BaseHTTPRequestHandler):
    session: Optional[EmbodiedSession] = None
    # Resolved once at server creation; re-applied to every new session so that
    # switching backend (manual ProcTHOR <-> replay) never loses the episode
    # roots.
    episodes_roots: list = []

    # ---- helpers ---------------------------------------------------
    def _send_json(self, payload, code: int = 200, head: bool = False) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client aborted; not an application error

    def _send_bytes(self, data, ctype: str, head: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client aborted; not an application error

    def _send_static(self, name: str, head: bool = False) -> None:
        root = WEB_DIR
        target = (root / name).resolve()
        if not str(target).startswith(str(root.resolve())) or not target.is_file():
            self.send_error(404)
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".mjs": "application/javascript; charset=utf-8",
            ".map": "application/json; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), ctype, head=head)

    def _send_vendor(self, name: str, head: bool = False) -> None:
        root = VENDOR_DIR
        target = (root / name).resolve()
        if not str(target).startswith(str(root.resolve())) or not target.is_file():
            self.send_error(404)
            return
        ctype = {
            ".js": "application/javascript; charset=utf-8",
            ".mjs": "application/javascript; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), ctype, head=head)

    def _route_static(self, path: str, head: bool = False) -> bool:
        """Serve the frontend module graph. Returns True when the path matched.

        The browser loads these as ES modules, so every import target (app.js,
        god3d.js, scene2d.js, vendor/*) MUST be routed here; a 404 on any of them
        aborts the whole module graph and leaves the UI non-functional.
        """
        if path in ("/", "/index.html"):
            self._send_static("index.html", head=head)
        elif path == "/styles.css":
            self._send_static("styles.css", head=head)
        elif path == "/app.js":
            self._send_static("app.js", head=head)
        elif path == "/god3d.js":
            self._send_static("god3d.js", head=head)
        elif path == "/wall_geometry.js":
            self._send_static("wall_geometry.js", head=head)
        elif path == "/world_space.js":
            self._send_static("world_space.js", head=head)
        elif path == "/scene2d.js":
            self._send_static("scene2d.js", head=head)
        elif path.startswith("/vendor/"):
            self._send_vendor(path[len("/vendor/"):], head=head)
        else:
            return False
        return True

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
    def do_HEAD(self):  # noqa: N802
        """HEAD for static frontend assets (so ``curl -I`` / link checks work).

        API/render endpoints are intentionally excluded: a HEAD must never
        trigger a real Unity render or an episode load as a side effect.
        """
        path = urlparse(self.path).path
        try:
            if self._route_static(path, head=True):
                return
            self.send_error(405, "HEAD is only supported for static assets")
        except Exception:  # noqa: BLE001
            self.send_error(500)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            if self._route_static(path):
                pass
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
            elif path == "/api/scene3d":
                self._send_json(_Handler.session.live_scene3d())
            elif path == "/api/episodes/scene3d":
                ep = _query(self.path).get("episode_id") or None
                self._send_json(_Handler.session.episode_scene3d(ep))
            elif path == "/api/unity/status":
                self._send_json(_Handler.session.unity_godview_status())
            elif path == "/api/houses":
                self._send_json({"houses": _Handler.session.list_houses()})
            elif path == "/api/unity/prewarm":
                ep = _query(self.path).get("episode_id") or None
                try:
                    self._send_json(_Handler.session.prewarm_unity(ep))
                except Exception as e:  # noqa: BLE001
                    self._send_json({"prewarming": False, "error": str(e)}, 503)
            elif path == "/api/episodes/unity_godview":
                q = _query(self.path)
                try:
                    png = _Handler.session.unity_godview_png(
                        episode_id=q.get("episode_id") or None,
                        step=q.get("step", 0),
                        view=q.get("view", "follow"),
                        camera=_camera_from_query(q),
                        quality=q.get("q", "full"),
                        width=q.get("w"),
                        height=q.get("h"),
                    )
                except Exception as e:  # noqa: BLE001
                    self._send_json({"error": str(e)}, 503)
                    return
                if png is None:
                    self._send_json({"error": "no unity godview frame"}, 404)
                else:
                    self._send_bytes(png, "image/jpeg")
            elif path == "/api/unity/live_godview":
                q = _query(self.path)
                try:
                    png = _Handler.session.live_unity_png(
                        view=q.get("view", "follow"),
                        camera=_camera_from_query(q),
                    )
                except Exception as e:  # noqa: BLE001
                    self._send_json({"error": str(e)}, 503)
                    return
                if png is None:
                    self._send_json({"error": "no live unity godview frame"}, 404)
                else:
                    self._send_bytes(png, "image/jpeg")
            elif path == "/api/episodes/semantics":
                ep = _query(self.path).get("episode_id")
                if not ep:
                    self._send_json({"error": "episode_id required"}, 400)
                else:
                    self._send_json(_Handler.session.episode_semantics(ep))
            elif path == "/api/telemetry":
                self._send_json(_Handler.session.telemetry())
            elif path == "/api/model":
                tele = _Handler.session.telemetry()
                self._send_json(tele.get("model") or {"model": None})
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
        except (BrokenPipeError, ConnectionResetError):
            return  # client aborted mid-response; not an application error
        except Exception as e:  # noqa: BLE001
            try:
                self._send_json({"error": str(e)}, 500)
            except (BrokenPipeError, ConnectionResetError):
                pass

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
            elif path == "/api/telemetry":
                self._send_json(_Handler.session.ingest_telemetry(body))
            else:
                self._send_json({"error": "not found", "path": path}, 404)
        except (BrokenPipeError, ConnectionResetError):
            return  # client aborted mid-response; not an application error
        except Exception as e:  # noqa: BLE001
            try:
                self._send_json({"error": str(e)}, 500)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _ensure_session(self, backend: Optional[str], seed: int) -> None:
        backend = backend or "deterministic"
        if (
            _Handler.session is None
            or _Handler.session.backend_name != backend
            or (seed is not None and seed != getattr(_Handler.session, "_seed", None))
        ):
            _Handler.session = EmbodiedSession(backend, seed=int(seed or 0))
            if _Handler.episodes_roots:
                _Handler.session.set_episodes_roots(_Handler.episodes_roots)

    def log_message(self, fmt, *args):  # noqa: A003
        pass


def _resolve_episodes_roots(episodes_root: Optional[str]) -> list:
    if episodes_root is not None:
        return [episodes_root]
    env_multi = os.environ.get("SF_EPISODES_ROOTS")
    if env_multi:
        return [p for p in env_multi.split(os.pathsep) if p]
    env_single = os.environ.get("SF_EPISODES_ROOT")
    if env_single:
        return [env_single]
    # default: aggregate known post-fix + legacy research roots so the curated
    # ordering can promote a post-fix valid episode above legacy garbage.
    candidates = [
        _REPO_ROOT / "outputs" / "episodes_postfix",
        _REPO_ROOT / "outputs" / "episodes",
        _REPO_ROOT / "outputs" / "embodied_bc" / "model_rollout",
    ]
    existing = [str(p) for p in candidates if p.is_dir()]
    return existing or [str(DEFAULT_EPISODES_ROOT)]


def create_embodied_server(
    host: str = "127.0.0.1",
    port: int = 8100,
    episodes_root: Optional[str] = None,
):
    _Handler.episodes_roots = _resolve_episodes_roots(episodes_root)
    _Handler.session = EmbodiedSession("deterministic", seed=0)
    _Handler.session.set_episodes_roots(_Handler.episodes_roots)
    server = ThreadingHTTPServer((host, port), _Handler)
    return server


if __name__ == "__main__":  # pragma: no cover
    import atexit
    import signal
    import sys

    host = "127.0.0.1"
    port = 8100
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    server = create_embodied_server(host, port)

    def _shutdown(*_args):
        # Close every Unity engine this process owns; otherwise a killed server
        # leaves orphaned thor-Linux64 processes holding GPU memory.
        try:
            if _Handler.session is not None:
                _Handler.session.close_god_cameras()
                backend = getattr(_Handler.session, "backend", None)
                if backend is not None:
                    backend.close()
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    print(f"SpatialForge Embodied Inspector on http://{host}:{port}")
    print("  episodes roots: " + os.pathsep.join(str(p) for p in _Handler.session.episodes_roots))
    try:
        from spatialforge.inspector.god_camera import _HAS_PROCTHOR

        if _HAS_PROCTHOR:
            print("  Unity God View: available (real AI2-THOR third-party camera)")
        else:
            print(
                "  Unity God View: UNAVAILABLE (ai2thor not importable here).\n"
                "    Run this server with the ProcTHOR venv python, e.g.\n"
                "    /root/autodl-tmp/.venv-procthor/bin/python -m "
                "spatialforge.inspector.embodied_server 127.0.0.1 8101"
            )
    except Exception:
        pass
    server.serve_forever()
