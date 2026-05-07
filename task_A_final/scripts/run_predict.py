from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import env_version_tag, load_text_config, relativize, resolve_run_dir, snapshot_config, utc_now_iso, write_metadata
from taska.recovery import predict_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task A final prediction pipeline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--debug-gap-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    config = load_text_config(args.config)
    config["_config_path"] = str(args.config.resolve())

    stats = predict_records(config=config, input_path=args.input, output_path=args.out, debug_out_path=args.debug_gap_csv)
    run_dir = resolve_run_dir(args.out)
    stem = args.out.stem
    if stem.endswith("_8"):
        snapshot_name = "config_8.yaml"
    elif stem.endswith("_16"):
        snapshot_name = "config_16.yaml"
    else:
        snapshot_name = f"{stem}.config.yaml"
    config_snapshot = snapshot_config(args.config, run_dir, suffix=snapshot_name)
    metadata = {
        "run_name": run_dir.name,
        "timestamp": utc_now_iso(),
        "version_tag": env_version_tag(),
        "script": "scripts/run_predict.py",
        "predict_mode": str(config.get("strategy", {}).get("name", "pchip_only")),
        "analysis_mode": None,
        "input_path": relativize(args.input),
        "output_path": relativize(args.out),
        "debug_gap_csv": relativize(args.debug_gap_csv) if args.debug_gap_csv is not None else None,
        "config_path": relativize(args.config),
        "config_snapshot_path": relativize(config_snapshot),
        "runtime_sec": time.time() - start,
        "used_wrapper": bool(stats.get("predict_impl") == "legacy_wrapper"),
        "map_path": config.get("map", {}).get("osm_path", config.get("legacy_wrapper", {}).get("map_path")),
        "cache_path": config.get("map", {}).get("cache_path", config.get("legacy_wrapper", {}).get("cache_path")),
        "strategy_summary": config.get("strategy", {}),
        "stats": stats,
    }
    metadata_path = write_metadata(run_dir, metadata)
    print(f"Prediction saved: {args.out}")
    print(f"Metadata saved: {metadata_path}")


if __name__ == "__main__":
    main()
