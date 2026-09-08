#!/usr/bin/env python3
"""Launcher script for SpatialForge God View Research Inspector.

Usage:
    python scripts/run_god_view.py --port 8000
    python scripts/run_god_view.py --host 0.0.0.0 --port 8080

For remote AutoDL usage via SSH port forwarding:
    ssh -L 8000:127.0.0.1:8000 root@<autodl-host> -p <port>
    Then open http://localhost:8000 in your local browser.
"""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spatialforge.inspector.server import create_inspector_server


def main():
    parser = argparse.ArgumentParser(description="Run SpatialForge God View Research Inspector.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    print("=" * 70)
    print(" SPATIALFORGE 3D GOD VIEW / RESEARCH INSPECTOR (G1 VERTICAL SLICE)")
    print("=" * 70)
    print(f" * Server address: http://{args.host}:{args.port}/")
    print(f" * Local access:   http://localhost:{args.port}/")
    print(f" * SSH Tunnel:     ssh -L {args.port}:127.0.0.1:{args.port} root@<host>")
    print(" * Scientific boundary: PRIVILEGED ENVIRONMENT TRUTH IS NEVER MODEL INPUT.")
    print("=" * 70)
    print("Serving read-only research workbench. Press Ctrl+C to stop.")

    server = create_inspector_server(host=args.host, port=args.port, repo_root=_REPO_ROOT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down God View Inspector server...")
    finally:
        server.server_close()
        print("Server stopped cleanly.")


if __name__ == "__main__":
    main()
