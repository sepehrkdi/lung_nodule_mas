#!/usr/bin/env python3
"""
Combined Launcher
=================

Starts both the FastAPI backend and Streamlit frontend.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
from multiprocessing import Process


def run_api():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


def run_ui():
    """Run the Streamlit server."""
    project_root = Path(__file__).parent
    app_path = project_root / "ui" / "app.py"
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])


def main():
    print("=" * 60)
    print("🫁 Lung Nodule Multi-Agent System")
    print("=" * 60)
    print()
    print("Starting services...")
    print()
    print("  📡 API Server:  http://localhost:8000")
    print("  📡 API Docs:    http://localhost:8000/docs")
    print("  🌐 Web UI:      http://localhost:8501")
    print()
    print("=" * 60)
    print("Press Ctrl+C to stop all services")
    print("=" * 60)
    print()
    
    # Start processes
    api_process = Process(target=run_api, name="API Server")
    ui_process = Process(target=run_ui, name="Streamlit UI")
    
    processes = [api_process, ui_process]
    
    def cleanup(signum=None, frame=None):
        """Clean up processes on exit."""
        print("\n\nShutting down services...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        print("All services stopped.")
        sys.exit(0)
    
    # Handle signals
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Start API first
        api_process.start()
        print("✅ API Server starting...")
        
        # Wait a moment for API to initialize
        time.sleep(2)
        
        # Start UI
        ui_process.start()
        print("✅ Streamlit UI starting...")
        
        # Wait for processes
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"Error: {e}")
        cleanup()


if __name__ == "__main__":
    main()
