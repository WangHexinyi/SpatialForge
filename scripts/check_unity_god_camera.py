"""Validate the real AI2-THOR Unity inspection (God) camera.

Renders the loaded house through AI2-THOR's ``AddThirdPartyCamera`` so the
researcher gets a true rendered scene view (not model input). This is the
backing capability for the God View "UNITY CAMERA" mode; wiring it into the
inspector requires the server to own a live engine (next sub-gate).

    python scripts/check_unity_god_camera.py \
        --house /root/autodl-tmp/val_house_0.json --x-display :0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="/root/autodl-tmp/val_house_0.json")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--x-display", default=":0")
    ap.add_argument("--out", default="outputs/diagnostics/unity_god_camera")
    args = ap.parse_args()

    from spatialforge.embodied.backends.procthor import ProcTHORBackend

    backend = ProcTHORBackend(
        house=args.house, width=args.width, height=args.height,
        x_display=args.x_display, require_nvidia=True,
    )
    try:
        backend.load_house()
        md = backend._event.metadata
        rooms = backend._house_data.get("rooms", [])
        bounds = None
        xs, zs = [], []
        for r in rooms:
            for p in r.get("floorPolygon", []):
                xs.append(p["x"])
                zs.append(p["z"])
        if xs:
            cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
            bounds = {"center": [cx, cz], "span": [max(xs) - min(xs), max(zs) - min(zs)]}
        else:
            cx, cz = 0.0, 0.0

        ev = backend.controller.step(dict(
            action="AddThirdPartyCamera",
            rotation={"x": 60.0, "y": 0.0, "z": 0.0},
            position={"x": cx, "y": 6.0, "z": cz - 0.1},
            fieldOfView=60.0,
        ))
        md2 = ev.metadata
        frames = md2.get("thirdPartyCameras") or []
        frame = None
        try:
            frame = ev.third_party_camera_frames[0]
        except Exception:
            frame = None
        report = {
            "schema": "unity_god_camera_check.v1",
            "house": os.path.abspath(args.house),
            "success": bool(md2.get("lastActionSuccess", False)),
            "error": md2.get("errorMessage"),
            "third_party_cameras": len(frames),
            "bounds": bounds,
            "resolution": [args.width, args.height],
        }
        if frame is not None:
            arr = np.asarray(frame)
            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, "unity_god_camera.png")
            Image.fromarray(arr.astype(np.uint8)).save(path)
            report["frame_shape"] = list(arr.shape)
            report["mean_luminance"] = round(float(arr.mean()), 2)
            report["png"] = os.path.abspath(path)
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "unity_god_camera_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("[unity_god_camera] " + json.dumps(report, indent=2))
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main())
