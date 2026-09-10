"""Real Unity God View camera for the Inspector (AI2-THOR AddThirdPartyCamera).

This is the *primary* researcher God View: the actual rendered ProcTHOR/Unity
scene, viewed from an external (third-party) camera. It is **not** model input,
and it is not a Three.js semantic proxy. The Three.js semantic view is demoted
to a secondary diagnostic overlay.

The renderer owns a live AI2-THOR engine for one house and renders frames on
demand through ``AddThirdPartyCamera`` / ``UpdateThirdPartyCamera``. When the
engine / display is unavailable it reports a loud reason and returns ``None``;
it never fabricates a scene.

This module imports ``ai2thor`` lazily and is safe to import on a CPU-only box.
"""

from __future__ import annotations

import io
import math
import threading
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - environment dependent
    import ai2thor  # noqa: F401

    _HAS_PROCTHOR = True
except Exception:  # pragma: no cover
    _HAS_PROCTHOR = False

from spatialforge.embodied.rendering import current_renderer, resolve_render_quality

#: Neutral background for the researcher God View third-party camera.
#:
#: The AI2-THOR ``Procedural`` scene renders an aerial city photograph through
#: Unity's skybox, and ``AddThirdPartyCamera`` defaults to Skybox clear flags,
#: so the city appears behind/around the house. Passing ``skyboxColor`` switches
#: only that third-party camera to a solid background (AI2-THOR parses it as an
#: HTML color). It never touches the first-person camera, the scene skybox, or
#: scene lighting, so model observations stay bit-identical.
GOD_VIEW_SKYBOX_COLOR = "#101418"

#: Measured on the RTX 4080 SUPER (640x480 baseline): Unity ``UpdateThirdParty
#: Camera`` is a synchronous engine render at ~99 ms / 384x288, ~107 ms /
#: 512x384, ~127 ms / 640x480 and ~202 ms / 1280x960. JPEG encode is ~50x faster
#: than PNG (1-11 ms vs 58-608 ms), so the God View always ships JPEG.
#:
#: Interactive drag uses a real low-latency preview; the settle frame uses the
#: browser's viewport size (clamped) for a sharp steady state. ``ChangeResolution``
#: switches the owned engine at runtime (house/objects/camera persist).
GOD_VIEW_PREVIEW_SIZE = (512, 384)
GOD_VIEW_FULL_SIZE = (1280, 960)
GOD_VIEW_MAX_SIZE = (1600, 1200)
GOD_VIEW_PREVIEW_JPEG_QUALITY = 80
GOD_VIEW_FULL_JPEG_QUALITY = 90
GOD_VIEW_LIVE_JPEG_QUALITY = 85


class GodCameraUnavailable(RuntimeError):
    pass


class GodCameraRenderer:
    """Render the real Unity scene through a third-party (God) camera."""

    def __init__(
        self,
        house_path: str,
        *,
        width: int = 640,
        height: int = 480,
        x_display: Optional[str] = None,
        quality: Optional[str] = None,
        require_nvidia: bool = False,
        backend: Any = None,
    ):
        self.house_path = str(house_path)
        self.width = int(width)
        self.height = int(height)
        self.x_display = x_display
        self.quality = resolve_render_quality(quality)
        self.require_nvidia = bool(require_nvidia)
        # When a live backend/controller is supplied we render through it (one
        # Unity engine, no restart) instead of creating a second one.
        self._provided_backend = backend
        self._owns_backend = backend is None
        self._backend = None
        self._added = False
        self._lock = threading.RLock()
        self._bounds: Optional[Dict[str, Any]] = None
        self._last_agent: Optional[Dict[str, Any]] = None
        self._render_size: Optional[tuple] = None

    # ------------------------------------------------------------------
    def unavailable_reason(self) -> Optional[str]:
        if not _HAS_PROCTHOR:
            return "ai2thor is not importable in this interpreter"
        if not Path(self.house_path).is_file():
            return f"house json not found: {self.house_path}"
        return None

    def renderer_name(self) -> str:
        return current_renderer(self.x_display)

    # ------------------------------------------------------------------
    def _compute_bounds(self) -> Dict[str, Any]:
        rooms = (self._backend._house_data or {}).get("rooms", [])
        xs, zs = [], []
        for r in rooms:
            for p in r.get("floorPolygon", []):
                xs.append(float(p.get("x", 0.0)))
                zs.append(float(p.get("z", 0.0)))
        if not xs:
            return {"center": [0.0, 0.0], "span": [4.0, 4.0]}
        return {
            "center": [(min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2],
            "span": [max(xs) - min(xs), max(zs) - min(zs)],
        }

    def _ensure(self) -> None:
        if self._backend is not None:
            return
        if self._provided_backend is not None:
            self._backend = self._provided_backend
            self._bounds = self._compute_bounds()
            self._render_size = (
                int(getattr(self._backend, "width", self.width) or self.width),
                int(getattr(self._backend, "height", self.height) or self.height),
            )
            return
        reason = self.unavailable_reason()
        if reason:
            raise GodCameraUnavailable(reason)
        from spatialforge.embodied.backends.procthor import ProcTHORBackend

        backend = ProcTHORBackend(
            house=self.house_path,
            width=self.width,
            height=self.height,
            quality=self.quality,
            x_display=self.x_display,
            require_nvidia=self.require_nvidia,
        )
        backend.load_house()
        self._backend = backend
        self._bounds = self._compute_bounds()
        self._render_size = (self.width, self.height)

    def _ensure_resolution(self, width: int, height: int) -> None:
        """Switch the *owned* engine's render resolution at runtime.

        ``ChangeResolution`` keeps the house/objects/third-party camera intact
        and only reallocates the render targets, so preview vs settle frames do
        not need a second engine. A shared backend (live mode) is never resized:
        its resolution is the model's observation resolution.
        """
        if not self._owns_backend or self._backend is None:
            return
        target = (int(width), int(height))
        if self._render_size == target:
            return
        try:
            self._backend.controller.step(
                dict(action="ChangeResolution", x=target[0], y=target[1])
            )
            self._render_size = target
        except Exception:  # noqa: BLE001 - keep the current resolution on failure
            pass

    def close(self) -> None:
        with self._lock:
            if self._backend is not None and self._owns_backend:
                try:
                    self._backend.close()
                except Exception:  # noqa: BLE001
                    pass
            self._backend = None
            self._added = False

    # ------------------------------------------------------------------
    def _camera_for(
        self, view: str, agent_pose: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        view = (view or "follow").lower()
        if view in ("overview", "top", "house") or not agent_pose:
            cx, cz = self._bounds["center"]
            span = max(self._bounds["span"])
            return {
                "position": {"x": cx, "y": max(6.0, span * 0.6), "z": cz - 0.1},
                "rotation": {"x": 60.0, "y": 0.0, "z": 0.0},
                "fieldOfView": 60.0,
            }
        pos = agent_pose.get("position") or [0.0, 0.9, 0.0]
        yaw = float(agent_pose.get("rotation_yaw_deg") or 0.0)
        rad = math.radians(yaw)
        d, h = 2.2, 2.0
        return {
            "position": {
                "x": float(pos[0]) - math.sin(rad) * d,
                "y": float(pos[1]) + h,
                "z": float(pos[2]) - math.cos(rad) * d,
            },
            "rotation": {"x": 32.0, "y": yaw, "z": 0.0},
            "fieldOfView": 70.0,
        }

    @staticmethod
    def _pose_key(pose: Optional[Dict[str, Any]]):
        if not pose or not pose.get("position"):
            return None
        p = pose["position"]
        return (
            round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4),
            round(float(pose.get("rotation_yaw_deg") or 0.0), 3),
            round(float(pose.get("camera_horizon_deg") or 0.0), 3),
        )

    @staticmethod
    def _explicit_camera(camera: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate an explicit free-camera pose sent by the browser."""
        if not camera:
            return None
        pos = camera.get("position")
        rot = camera.get("rotation")
        if not pos or not rot:
            return None

        def coord(v, key, idx):
            if isinstance(v, dict):
                return float(v.get(key, 0.0))
            return float(v[idx])

        try:
            return {
                "position": {
                    "x": coord(pos, "x", 0),
                    "y": coord(pos, "y", 1),
                    "z": coord(pos, "z", 2),
                },
                "rotation": {
                    "x": coord(rot, "x", 0),
                    "y": coord(rot, "y", 1),
                    "z": coord(rot, "z", 2),
                },
                "fieldOfView": float(camera.get("fieldOfView", camera.get("fov", 70.0)) or 70.0),
            }
        except (TypeError, ValueError, KeyError, IndexError):
            return None

    def render(
        self,
        *,
        agent_pose: Optional[Dict[str, Any]] = None,
        view: str = "follow",
        camera: Optional[Dict[str, Any]] = None,
        teleport: bool = True,
        resolution: Optional[tuple] = None,
        image_format: str = "jpeg",
        jpeg_quality: int = 85,
    ) -> bytes:
        """Render one real Unity frame; raises GodCameraUnavailable on failure.

        ``camera`` (when valid) is an explicit ThirdPartyCamera pose produced by
        browser orbit/pan/zoom; it overrides the Follow/Overview preset so the
        browser interaction is the real source of camera truth.

        ``resolution`` switches the owned engine's render size (preview vs
        settle). It is ignored for a shared/live backend so model observations
        keep their native resolution. ``image_format`` defaults to JPEG because
        PNG encode dominates the frame budget (measured 50x slower).
        """
        with self._lock:
            self._ensure()
            if resolution:
                self._ensure_resolution(int(resolution[0]), int(resolution[1]))
            explicit = self._explicit_camera(camera)
            cam = explicit or self._camera_for(view, agent_pose)
            if teleport and agent_pose and agent_pose.get("position"):
                key = self._pose_key(agent_pose)
                if key != self._last_agent:
                    self._backend.set_agent_pose(
                        agent_pose["position"],
                        rotation_yaw_deg=float(agent_pose.get("rotation_yaw_deg") or 0.0),
                        horizon_deg=float(agent_pose.get("camera_horizon_deg") or 0.0),
                        render=False,
                    )
                    self._last_agent = key
            params = {
                "position": cam["position"],
                "rotation": cam["rotation"],
                "fieldOfView": cam["fieldOfView"],
                # researcher-only: solid neutral background instead of the
                # Procedural scene's city skybox (third-party camera only).
                "skyboxColor": GOD_VIEW_SKYBOX_COLOR,
            }
            if not self._added:
                ev = self._backend.controller.step(
                    dict(action="AddThirdPartyCamera", **params)
                )
                self._added = True
            else:
                ev = self._backend.controller.step(
                    dict(action="UpdateThirdPartyCamera", thirdPartyCameraId=0, **params)
                )
            md = ev.metadata
            if not md.get("lastActionSuccess", False):
                raise GodCameraUnavailable(
                    f"God camera action failed: {md.get('errorMessage')}"
                )
            try:
                frame = ev.third_party_camera_frames[0]
            except Exception as e:  # noqa: BLE001
                raise GodCameraUnavailable(f"no third-party camera frame: {e}") from e
            from PIL import Image

            buf = io.BytesIO()
            image = Image.fromarray(frame)
            if str(image_format).lower() in ("jpg", "jpeg"):
                image.save(buf, format="JPEG", quality=int(jpeg_quality))
            else:
                image.save(buf, format="PNG")
            return buf.getvalue()
