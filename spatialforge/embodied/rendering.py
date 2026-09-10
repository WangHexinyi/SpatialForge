"""Renderer detection / GPU guard for the embodied runtime.

AI2-THOR's Linux64 build renders through GLX on an X display. If that display
is a software X server (Xvfb) the effective GL is Mesa llvmpipe, which is the
large-scale rollout performance blocker. This module detects the renderer that
AI2-THOR would actually use on a given display and lets callers *require* an
NVIDIA renderer so that GPU init failures surface loudly instead of silently
falling back to llvmpipe.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional


#: Sensor-fidelity default render quality. The exposure diagnostic
#: (outputs/diagnostics/exposure + render_quality) shows AI2-THOR ``Low``
#: blows out highlights (mean frac>=250 up to 0.38), while ``Medium`` is the
#: lowest quality whose dynamic range is visually normal (mean frac>=250
#: ~0.01, max ~0.03). Overridable for explicit experiments via
#: ``SF_RENDER_QUALITY``; never silently downgraded.
DEFAULT_RENDER_QUALITY = "Medium"
VALID_RENDER_QUALITIES = ("Low", "Medium", "High", "Very High", "Ultra")


def resolve_render_quality(quality: Optional[str] = None) -> str:
    q = quality or os.environ.get("SF_RENDER_QUALITY") or DEFAULT_RENDER_QUALITY
    if q not in VALID_RENDER_QUALITIES:
        raise ValueError(
            f"unknown AI2-THOR quality {q!r}; valid: {VALID_RENDER_QUALITIES}"
        )
    return q


class RendererNotAvailableError(RuntimeError):
    pass


def _glxinfo_renderer(display: Optional[str]) -> Optional[str]:
    env = dict(os.environ)
    if display:
        env["DISPLAY"] = display
    try:
        out = subprocess.run(
            ["glxinfo", "-B"], env=env, capture_output=True, text=True, timeout=25,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if "OpenGL renderer string" in line:
            return line.split(":", 1)[1].strip()
    return None


def current_renderer(display: Optional[str] = None) -> str:
    """Return the OpenGL renderer name the display would serve (or 'unknown')."""
    return _glxinfo_renderer(display) or "unknown"


def is_nvidia(renderer: Optional[str]) -> bool:
    return bool(renderer and "NVIDIA" in renderer)


def resolve_display(display: Optional[str]) -> Optional[str]:
    if display:
        return display
    return os.environ.get("DISPLAY")


def ensure_nvidia_renderer(display: Optional[str] = None) -> str:
    """Raise if the resolved display does not serve an NVIDIA OpenGL renderer.

    Used to guarantee GPU-backed AI2-THOR rendering with **no silent fallback**
    to llvmpipe. Returns the renderer name on success.
    """
    disp = resolve_display(display)
    renderer = current_renderer(disp)
    if not is_nvidia(renderer):
        raise RendererNotAvailableError(
            "GPU rendering required but NOT available.\n"
            f"  display={disp or '(unset)'}\n"
            f"  renderer={renderer}\n"
            "This is NOT llvmpipe software fallback; GPU init failed. Start an "
            "NVIDIA-backed X display first, e.g.\n"
            "    python scripts/start_nvidia_xorg.py start --display :0\n"
            "then pass x_display=:0 / --x-display :0."
        )
    return renderer
