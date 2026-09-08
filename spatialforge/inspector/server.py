"""Lightweight standard-library HTTP server for SpatialForge God View Inspector.

Features:
- Pure Python standard library (`http.server`); zero third-party web framework dependencies.
- Read-only endpoints for scenes, viewpoints, samples, snapshots, runs, profiles, and telemetry.
- Safe static file serving for web UI assets with directory traversal protection.
- Safe image serving for rendered observations with sandboxed path resolution.
- Live telemetry integration via NVMLClient and progress.json reader.
- Authoritative Training Profile exposition and capability preflight (NO SILENT FALLBACK).
"""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from spatialforge.experiment.training import (
    FIRST_CLASS_PROFILES,
    PROFILE_BALANCED,
    PROFILE_MAX_PERFORMANCE,
    PROFILE_REFERENCE,
    ProfileCapabilityError,
    get_profile,
    validate_profile_capability,
)
from spatialforge.inspector.artifacts import ArtifactRepository, resolve_safe_path
from spatialforge.inspector.projection import run_blender_projection_probe
from spatialforge.inspector.snapshot import build_inspector_snapshot
from spatialforge.environment.trajectory import (
    TrajectoryConfig,
    compute_adaptive_framing,
    generate_trajectory,
    parse_waypoint_text,
    TRAJECTORY_FAMILY_SURVEY_ORBIT,
    DISTANCE_MODE_ADAPTIVE,
)


class InspectorRequestHandler(SimpleHTTPRequestHandler):
    """Handler routing API calls and static workbench UI files."""

    repo: ArtifactRepository
    web_dir: Path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.web_dir), **kwargs)

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(raw)

    def _send_error_json(self, message: str, status: int = 400, error_type: str = "BadRequest") -> None:
        self._send_json({"error": error_type, "message": message, "status": status}, status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Route APIs
        if path.startswith("/api/"):
            self.handle_api(path, query)
            return

        # Static file serving from web directory
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else ""
            try:
                data = json.loads(post_body) if post_body else {}
            except json.JSONDecodeError:
                data = {}
            query = {k: [v] for k, v in data.items()}
            self.handle_api(path, query)
            return
        self._send_error_json("POST not supported for static files", status=405)

    def handle_api(self, path: str, query: Dict[str, list]) -> None:
        try:
            if path == "/api/scenes":
                scenes = self.repo.list_scenes()
                self._send_json({"scenes": scenes})

            elif path.startswith("/api/scene/"):
                scene_id = path[len("/api/scene/") :].strip()
                if not scene_id:
                    self._send_error_json("Missing scene_id")
                    return
                try:
                    scene_state = self.repo.load_scene(scene_id)
                    views = self.repo.list_views_for_scene(scene_id)
                    objects = [
                        {
                            "object_index": i,
                            "name": obj.name,
                            "shape": obj.shape,
                            "color": obj.color,
                            "size": obj.size,
                            "location": list(obj.location),
                        }
                        for i, obj in enumerate(scene_state.objects)
                    ]
                    challenge_meta = self.repo.load_challenge_metadata(scene_id)
                    self._send_json({
                        "scene_id": scene_id,
                        "seed": scene_state.seed,
                        "views": views,
                        "objects": objects,
                        "challenge_metadata": challenge_meta,
                    })
                except FileNotFoundError:
                    self._send_error_json(f"Scene {scene_id!r} not found", status=404, error_type="NotFound")

            elif path == "/api/samples":
                scene_id = query.get("scene_id", [None])[0]
                view_id = query.get("view_id", [None])[0]
                samples = self.repo.list_samples(scene_id=scene_id, view_id=view_id)
                self._send_json({
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "count": len(samples),
                    "samples": samples,
                })

            elif path == "/api/snapshot":
                scene_id = query.get("scene_id", [None])[0]
                view_id = query.get("view_id", [None])[0]
                sample_id = query.get("sample_id", [None])[0]
                run_id = query.get("run_id", [None])[0]
                dist_str = query.get("distance", [None])[0]
                camera_dist = float(dist_str) if dist_str and dist_str != "none" else None

                if not scene_id or not view_id:
                    self._send_error_json("Missing required query parameters: scene_id and view_id")
                    return

                try:
                    snapshot = build_inspector_snapshot(
                        repo=self.repo,
                        scene_id=scene_id,
                        view_id=view_id,
                        sample_id=sample_id,
                        run_id=run_id,
                        camera_distance=camera_dist,
                    )
                    self._send_json(snapshot.to_dict())
                except FileNotFoundError as e:
                    self._send_error_json(str(e), status=404, error_type="NotFound")
                except KeyError as e:
                    self._send_error_json(str(e), status=404, error_type="KeyError")

            elif path == "/api/view_sets":
                scene_id = query.get("scene_id", ["scene_000"])[0]
                view_sets = self.repo.list_view_sets(scene_id)
                self._send_json(view_sets)

            elif path == "/api/runs":
                runs = self.repo.list_runs()
                self._send_json({"runs": runs})

            elif path == "/api/profiles":
                profiles = []
                for pid in (PROFILE_REFERENCE, PROFILE_BALANCED, PROFILE_MAX_PERFORMANCE):
                    prof = FIRST_CLASS_PROFILES[pid]
                    profiles.append({
                        "profile_id": prof.profile_id,
                        "display_name": prof.display_name,
                        "microbatch_size": prof.microbatch_size,
                        "gradient_accumulation_steps": prof.gradient_accumulation_steps,
                        "effective_batch_size": prof.effective_batch_size,
                        "gradient_checkpointing": prof.gradient_checkpointing,
                        "use_vision_cache": prof.use_vision_cache,
                        "memory_estimate_mb": prof.memory_estimate_mb,
                        "intended_use": prof.intended_use,
                        "description": prof.description,
                    })
                self._send_json({
                    "profiles": profiles,
                    "default_profile_id": PROFILE_MAX_PERFORMANCE,
                })

            elif path == "/api/profiles/validate":
                profile_id = query.get("profile", [None])[0]
                if not profile_id:
                    self._send_error_json("Missing profile query parameter")
                    return

                mem_override = query.get("available_memory_mb", [None])[0]
                mem_val = float(mem_override) if mem_override is not None else None

                try:
                    prof = get_profile(profile_id)
                    validate_profile_capability(prof, available_memory_mb=mem_val)
                    self._send_json({
                        "status": "compatible",
                        "requested_profile": prof.profile_id,
                        "display_name": prof.display_name,
                        "required_memory_estimate_mb": prof.memory_estimate_mb,
                        "message": f"Profile {prof.display_name} capability preflight passed.",
                    })
                except ProfileCapabilityError as e:
                    self._send_json({
                        "status": "incompatible",
                        "error": "ProfileCapabilityError",
                        "requested_profile": e.requested_profile,
                        "required_memory_estimate_mb": e.required_memory_estimate,
                        "available_memory_mb": e.available_memory,
                        "recommended_profile": e.recommended_profile,
                        "message": str(e),
                    })
                except ValueError as e:
                    self._send_error_json(str(e), status=400, error_type="InvalidProfile")

            elif path == "/api/telemetry":
                run_id = query.get("run_id", [None])[0]
                rt = self.repo.load_runtime_telemetry(run_id=run_id)
                self._send_json(rt.to_dict())

            elif path == "/api/image":
                raw_path = query.get("path", [None])[0]
                if not raw_path:
                    self._send_error_json("Missing path parameter")
                    return

                try:
                    img_file = resolve_safe_path(self.repo.repo_root, raw_path)
                    if not img_file.exists() or not img_file.is_file():
                        self._send_error_json(f"Image not found: {raw_path!r}", status=404, error_type="NotFound")
                        return

                    suffix = img_file.suffix.lower()
                    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
                        self._send_error_json("Disallowed image extension", status=403, error_type="Forbidden")
                        return

                    mime = "image/png" if suffix == ".png" else "image/jpeg"
                    data = img_file.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "public, max-age=86400")
                    self.end_headers()
                    self.wfile.write(data)
                except PermissionError as e:
                    self._send_error_json(str(e), status=403, error_type="PermissionDenied")
                except Exception as e:
                    self._send_error_json(str(e), status=500, error_type="InternalError")

            elif path == "/api/probe":
                res = run_blender_projection_probe()
                self._send_json(res)

            elif path == "/api/trajectory":
                scene_id = query.get("scene_id", [None])[0]
                if not scene_id:
                    self._send_error_json("Missing scene_id query parameter")
                    return
                try:
                    scene_state = self.repo.load_scene(scene_id)
                except FileNotFoundError:
                    self._send_error_json(f"Scene {scene_id!r} not found", status=404, error_type="NotFound")
                    return

                framing_mode = query.get("framing_mode", ["balanced"])[0]
                traj_family = query.get("trajectory_family", [None])[0]
                dist_mode = query.get("distance_mode", ["adaptive"])[0]
                interp_mode = query.get("interpolation_mode", ["catmull_rom"])[0]

                try:
                    frame_count = int(query.get("frame_count", [60])[0])
                except (ValueError, TypeError):
                    frame_count = 60
                try:
                    seed = int(query.get("seed", [0])[0])
                except (ValueError, TypeError):
                    seed = 0

                def _get_float(key: str, default: Optional[float]) -> Optional[float]:
                    val = query.get(key, [None])[0]
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            pass
                    return default

                azimuth_cycles = _get_float("azimuth_cycles", 1.0) or 1.0
                elevation_base = _get_float("elevation_base_deg", 25.0) or 25.0
                elevation_amp = _get_float("elevation_amplitude_deg", 15.0) or 15.0
                elevation_cycles = _get_float("elevation_cycles", 2.5) or 2.5
                radius_amp = _get_float("radius_amplitude_frac", 0.15) or 0.15
                radius_cycles = _get_float("radius_cycles", 1.5) or 1.5
                target_bias_frac = _get_float("target_bias_fraction", 0.10) or 0.10
                manual_dist = _get_float("manual_distance", None)
                dist_scale = _get_float("distance_scale", 1.0) or 1.0
                speed_m_s = _get_float("speed_m_s", None)
                if speed_m_s is None:
                    speed_m_s = _get_float("speed_m_per_s", 1.0) or 1.0

                waypoints = None
                waypoint_targets = None
                wp_text = query.get("waypoints_text", [None])[0]
                if wp_text:
                    try:
                        wp_pts, wp_tgts = parse_waypoint_text(wp_text)
                        waypoints = tuple(wp_pts)
                        if wp_tgts:
                            waypoint_targets = tuple(wp_tgts)
                    except ValueError as e:
                        self._send_error_json(f"Invalid waypoints: {e}", status=400)
                        return

                wp_json = query.get("waypoints_json", [None])[0] or query.get("waypoints", [None])[0]
                if wp_json and not waypoints:
                    try:
                        raw_list = json.loads(wp_json) if isinstance(wp_json, str) else wp_json
                        parsed_pts = []
                        parsed_tgts = []
                        has_any_target = False
                        for p in raw_list:
                            if isinstance(p, dict):
                                if "x" in p and "y" in p and "z" in p:
                                    parsed_pts.append((float(p["x"]), float(p["y"]), float(p["z"])))
                                    if "tx" in p and "ty" in p and "tz" in p:
                                        has_any_target = True
                                        parsed_tgts.append((float(p["tx"]), float(p["ty"]), float(p["tz"])))
                                elif "position" in p:
                                    pos = p["position"]
                                    parsed_pts.append((float(pos[0]), float(pos[1]), float(pos[2])))
                                    if "target" in p and p["target"]:
                                        has_any_target = True
                                        tgt = p["target"]
                                        parsed_tgts.append((float(tgt[0]), float(tgt[1]), float(tgt[2])))
                                else:
                                    raise ValueError(f"Unrecognized waypoint dict: {p}")
                            elif isinstance(p, (list, tuple)):
                                if len(p) >= 6:
                                    parsed_pts.append((float(p[0]), float(p[1]), float(p[2])))
                                    has_any_target = True
                                    parsed_tgts.append((float(p[3]), float(p[4]), float(p[5])))
                                elif len(p) >= 3:
                                    parsed_pts.append((float(p[0]), float(p[1]), float(p[2])))
                                else:
                                    raise ValueError(f"Waypoint list has fewer than 3 coordinates: {p}")
                            else:
                                raise ValueError(f"Unsupported waypoint item type: {type(p)}")
                        waypoints = tuple(parsed_pts)
                        if has_any_target and len(parsed_tgts) == len(parsed_pts):
                            waypoint_targets = tuple(parsed_tgts)
                    except Exception as e:
                        self._send_error_json(f"Invalid waypoints_json: {e}", status=400)
                        return

                cfg_kwargs: Dict[str, Any] = {
                    "frame_count": frame_count,
                    "seed": seed,
                    "framing_mode": framing_mode,
                    "azimuth_cycles": azimuth_cycles,
                    "elevation_base_deg": elevation_base,
                    "elevation_amplitude_deg": elevation_amp,
                    "elevation_cycles": elevation_cycles,
                    "radius_amplitude_frac": radius_amp,
                    "radius_cycles": radius_cycles,
                    "target_bias_fraction": target_bias_frac,
                    "distance_mode": dist_mode,
                    "manual_distance": manual_dist,
                    "distance_scale": dist_scale,
                    "speed_m_per_s": speed_m_s,
                    "interpolation_mode": interp_mode,
                }
                if traj_family:
                    cfg_kwargs["trajectory_family"] = traj_family
                if waypoints:
                    cfg_kwargs["waypoints"] = waypoints
                if waypoint_targets:
                    cfg_kwargs["waypoint_targets"] = waypoint_targets

                cfg = TrajectoryConfig(**cfg_kwargs)
                traj = generate_trajectory(scene_state, cfg)
                self._send_json(traj.to_dict())

            elif path == "/api/framing":
                scene_id = query.get("scene_id", [None])[0]
                if not scene_id:
                    self._send_error_json("Missing scene_id query parameter")
                    return
                try:
                    scene_state = self.repo.load_scene(scene_id)
                except FileNotFoundError:
                    self._send_error_json(f"Scene {scene_id!r} not found", status=404, error_type="NotFound")
                    return

                framing_mode = query.get("framing_mode", ["balanced"])[0]
                framing_res = compute_adaptive_framing(scene_state, framing_mode=framing_mode)
                self._send_json(framing_res.to_dict())

            elif path in ("/api/challenge", "/api/challenge_summary"):
                scene_id = query.get("scene_id", [None])[0]
                if not scene_id:
                    self._send_error_json("Missing scene_id query parameter")
                    return
                try:
                    challenge_meta = self.repo.load_challenge_metadata(scene_id)
                    self._send_json(challenge_meta)
                except FileNotFoundError:
                    self._send_error_json(f"Scene {scene_id!r} not found", status=404, error_type="NotFound")
                except Exception as e:
                    self._send_error_json(str(e), status=500, error_type="InternalError")

            else:
                self._send_error_json(f"Unknown API endpoint {path}", status=404, error_type="NotFound")

        except Exception as e:
            self._send_error_json(f"Internal server error: {e}", status=500, error_type="InternalError")


def create_inspector_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    repo_root: Optional[Path] = None,
) -> ThreadingHTTPServer:
    """Instantiate a configured ThreadingHTTPServer instance."""
    repo = ArtifactRepository(repo_root=repo_root)
    web_dir = Path(__file__).resolve().parent / "web"

    class BoundHandler(InspectorRequestHandler):
        pass

    BoundHandler.repo = repo
    BoundHandler.web_dir = web_dir

    server = ThreadingHTTPServer((host, port), BoundHandler)
    return server
