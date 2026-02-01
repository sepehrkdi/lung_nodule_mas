#!/usr/bin/env python3
"""
API Server Launcher
===================

Starts the FastAPI backend server.

Usage:
    python run_api.py [--port PORT] [--host HOST]
"""

import argparse
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description="Start the Lung Nodule MAS API server")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--reload", "-r",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🫁 Lung Nodule Multi-Agent System - API Server")
    print("=" * 60)
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API Documentation: http://localhost:{args.port}/docs")
    print(f"Health Check: http://localhost:{args.port}/health")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print()
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
