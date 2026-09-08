"""
SpaceNetra Enterprise Production CLI Orchestrator.

Unified command-line interface for satellite change detection, multi-temporal series analysis,
semantic vector retrieval, model training, event extraction, and server deployment.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SpaceNetra Real-Time Satellite Change Detection & Geospatial AI Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 1. Train Subcommand
    train_parser = subparsers.add_parser("train", help="Train Change Detection Deep Learning Models")
    train_parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to training config YAML")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    # 2. Predict Subcommand
    predict_parser = subparsers.add_parser("predict", help="Execute Sentinel-2 Satellite Change Inference")
    predict_parser.add_argument("--t1", type=str, required=True, help="Path to T1 satellite image GeoTIFF")
    predict_parser.add_argument("--t2", type=str, required=True, help="Path to T2 satellite image GeoTIFF")
    predict_parser.add_argument("--output", type=str, default="outputs/change_map.png", help="Output change map PNG/TIFF path")
    predict_parser.add_argument("--arch", type=str, default="changeformer", choices=["siamese_unet", "changeformer"], help="Model architecture")

    # 3. Analyze Subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Execute Multi-Temporal Trend Series Analysis")
    analyze_parser.add_argument("--images", nargs="+", required=True, help="List of satellite GeoTIFF image paths in temporal order")
    analyze_parser.add_argument("--output", type=str, default="outputs/temporal_trend.png", help="Output trend chart PNG path")
    analyze_parser.add_argument("--metric", type=str, default="ndvi", choices=["ndvi", "ndwi"], help="Spectral index metric")

    # 4. Search Subcommand
    search_parser = subparsers.add_parser("search", help="Execute RemoteCLIP Semantic Vector Search")
    search_parser.add_argument("--query", type=str, required=True, help="Text query or image path to search")
    search_parser.add_argument("--index", type=str, default="outputs/vector_index.faiss", help="Path to FAISS vector index file")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of top matches to retrieve")

    # 5. Events Subcommand
    events_parser = subparsers.add_parser("events", help="Extract Spatial Change Events & Export GeoJSON")
    events_parser.add_argument("--mask", type=str, required=True, help="Path to binary change mask image")
    events_parser.add_argument("--output", type=str, default="outputs/change_events.geojson", help="Output GeoJSON file path")

    # 6. Serve Subcommand
    serve_parser = subparsers.add_parser("serve", help="Start SpaceNetra Web App & REST API Server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # 7. Verify Subcommand
    subparsers.add_parser("verify", help="Run System Verification & Docker Container Checks")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        cmd = [sys.executable, "train.py", "--config", args.config, "--epochs", str(args.epochs)]
        subprocess.run(cmd, check=True)

    elif args.command == "predict":
        cmd = [sys.executable, "predict_sentinel.py", "--t1", args.t1, "--t2", args.t2, "--output", args.output, "--arch", args.arch]
        subprocess.run(cmd, check=True)

    elif args.command == "analyze":
        cmd = [sys.executable, "analyze_series.py", "--images"] + args.images + ["--output", args.output, "--metric", args.metric]
        subprocess.run(cmd, check=True)

    elif args.command == "search":
        cmd = [sys.executable, "search_scenes.py", "--query", args.query, "--index", args.index, "--top_k", str(args.top_k)]
        subprocess.run(cmd, check=True)

    elif args.command == "events":
        cmd = [sys.executable, "build_events.py", "--mask", args.mask, "--output", args.output]
        subprocess.run(cmd, check=True)

    elif args.command == "serve":
        import uvicorn
        print(f"🚀 Starting SpaceNetra Server on {args.host}:{args.port}...")
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command == "verify":
        cmd = [sys.executable, "scripts/verify_docker.py"]
        res = subprocess.run(cmd)
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
