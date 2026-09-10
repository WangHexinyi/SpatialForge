#!/usr/bin/env python3
"""Start a reproducible NVIDIA-backed virtual X display for AI2-THOR GPU
rendering, and fail loudly (no silent llvmpipe fallback).

Why not stock ``ai2thor-xorg``? On hosts that expose many PCI GPUs only some of
which are actually mapped/visible, ai2thor-xorg enumerates every NVIDIA device
from ``lspci`` and Xorg then fails on inaccessible ones. Here we target the one
GPU that ``nvidia-smi`` reports, so it is deterministic and container-friendly.

Usage:
    python scripts/start_nvidia_xorg.py start [--display :0] [--width 2048 --height 1024]
    python scripts/start_nvidia_xorg.py stop [--display :0]
    python scripts/start_nvidia_xorg.py status [--display :0]

start returns exit 0 only after glxinfo on the display reports an NVIDIA
renderer. If GPU init fails it exits non-zero with the Xorg log path.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

CONF_DIR = Path("/tmp/ai2thor-gpu")
LOG_DIR = Path("/var/log")


def _nvidia_visible_bus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id,name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError("nvidia-smi did not report a visible NVIDIA GPU")
    line = out.stdout.strip().splitlines()[0]
    # nvidia-smi may print a 4- or 8-digit PCI domain (e.g. 00000000:4C:00.0)
    m = re.match(r"([0-9a-fA-F]+):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-9])", line)
    if not m:
        raise RuntimeError(f"cannot parse nvidia-smi bus id: {line}")
    _, b, d, f = m.groups()
    bus_id = f"PCI:{int(b, 16)}:{int(d, 16)}:{int(f)}"
    gpu_name = line.split(",", 1)[1].strip() if "," in line else "NVIDIA GPU"
    return bus_id, gpu_name


def write_conf(display, width, height):
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    bus_id, gpu_name = _nvidia_visible_bus()
    conf = f"""# NVIDIA-backed virtual display for AI2-THOR GPU rendering (auto-generated)
# GPU: {gpu_name}  bus: {bus_id}
Section "ServerLayout"
    Identifier "Layout0"
    Screen "Screen0"
EndSection
Section "Module"
    Load "glx"
EndSection
Section "Device"
    Identifier "Device0"
    Driver "nvidia"
    VendorName "NVIDIA Corporation"
    BusID "{bus_id}"
    Option "AllowEmptyInitialConfiguration" "True"
    Option "Interactive" "False"
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "Device0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual {width} {height}
    EndSubSection
EndSection
"""
    p = CONF_DIR / f"xorg-{display}.conf"
    p.write_text(conf)
    return p, gpu_name


def glx_renderer(display):
    out = subprocess.run(
        ["glxinfo", "-B"], env={**os.environ, "DISPLAY": display},
        capture_output=True, text=True, timeout=25,
    )
    for line in out.stdout.splitlines():
        if "OpenGL renderer string" in line:
            return line.split(":", 1)[1].strip()
    return None


def _is_nvidia_running(display):
    r = glx_renderer(display)
    return bool(r and ("NVIDIA" in r))


def start(display, width, height):
    disp_num = display.split(":")[-1].split(".")[0]
    conf, gpu_name = write_conf(display, width, height)
    log = LOG_DIR / f"ai2thor-xorg.{disp_num}.log"
    err = LOG_DIR / f"ai2thor-xorg-error.{disp_num}.log"
    cmd = (
        f"Xorg -quiet -noreset +extension GLX +extension RANDR +extension RENDER "
        f"-logfile {log} -config {conf} :{disp_num}"
    )
    with open(err, "w") as e:
        proc = subprocess.Popen(
            shlex_split(cmd), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=e,
        )
    time.sleep(0.3)
    if proc.poll() is not None:
        print(f"FAIL: Xorg exited early (rc={proc.returncode}); log={log} err={err}")
        print(Path(err).read_text()[-2000:] if Path(err).exists() else "")
        sys.exit(1)
    # wait until NVIDIA GLX is served (or timeout)
    for _ in range(40):
        if _is_nvidia_running(display):
            print(f"OK: NVIDIA GPU display {display} ready "
                  f"(GPU {gpu_name}, renderer {glx_renderer(display)})")
            return
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    print(f"FAIL: {display} did not come up as NVIDIA within timeout. log={log}")
    if Path(log).exists():
        print(Path(log).read_text()[-3000:])
    sys.exit(1)


def stop(display):
    disp_num = display.split(":")[-1].split(".")[0]
    pid = _find_xorg_pid(disp_num)
    if pid:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            if not _is_alive(pid):
                print(f"stopped Xorg :{disp_num}")
                return
            time.sleep(0.3)
        print(f"warning: Xorg :{disp_num} did not stop cleanly")
    else:
        print(f"no Xorg running on :{disp_num}")


def _find_xorg_pid(disp_num):
    out = subprocess.run(["pgrep", "-f", rf"Xorg.*:{disp_num}\b"], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    return pids[0] if pids else None


def _is_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def status(display):
    r = glx_renderer(display) if _display_exists(display) else None
    print(f"{display}: {r or 'not responding / no GLX'}")
    return 0 if (r and "NVIDIA" in r) else 1


def _display_exists(display):
    try:
        return subprocess.run(
            ["xdpyinfo", "-display", display], capture_output=True, text=True,
            timeout=10,
        ).returncode == 0
    except Exception:
        return False


def shlex_split(cmd):
    import shlex
    return shlex.split(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["start", "stop", "status"])
    ap.add_argument("--display", default=":0")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()
    if args.command == "start":
        start(args.display, args.width, args.height)
    elif args.command == "stop":
        stop(args.display)
    else:
        sys.exit(status(args.display))


if __name__ == "__main__":
    main()
