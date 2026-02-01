#!/usr/bin/env python3
"""
Streamlit UI Launcher
=====================

Starts the Streamlit web interface.

Usage:
    python run_ui.py [--port PORT]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Start the Lung Nodule MAS UI")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run the UI on (default: 8501)"
    )
    
    args = parser.parse_args()
    
    # Get the path to app.py
    project_root = Path(__file__).parent
    app_path = project_root / "ui" / "app.py"
    
    print("=" * 60)
    print("🫁 Lung Nodule Multi-Agent System - Web Interface")
    print("=" * 60)
    print(f"Starting Streamlit at http://localhost:{args.port}")
    print()
    print("⚠️  Make sure the API server is running:")
    print("    python run_api.py")
    print()
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print()
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])


if __name__ == "__main__":
    main()
