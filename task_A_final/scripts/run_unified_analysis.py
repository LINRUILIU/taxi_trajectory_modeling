from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import env_version_tag, list_outputs, relativize, save_json, utc_now_iso
from taska.unified_analysis import run_unified_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task A unified analysis with case gallery")
    parser.add_argument("--input-8", required=True, type=Path)
    parser.add_argument("--input-16", required=True, type=Path)
    parser.add_argument("--pred-8", required=True, type=Path)
    parser.add_argument("--pred-16", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--run-name", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    result = run_unified_analysis(
        input8_path=args.input_8,
        input16_path=args.input_16,
        pred8_path=args.pred_8,
        pred16_path=args.pred_16,
        gt_path=args.gt,
        out_dir=args.out_dir,
        run_name=args.run_name,
        map_path=args.map,
    )
    metadata = {
        "run_name": args.run_name,
        "timestamp": utc_now_iso(),
        "version_tag": env_version_tag(),
        "script": "scripts/run_unified_analysis.py",
        "analysis_mode": "unified_single_version",
        "input_paths": {
            "input_8": relativize(args.input_8),
            "input_16": relativize(args.input_16),
            "pred_8": relativize(args.pred_8),
            "pred_16": relativize(args.pred_16),
            "gt": relativize(args.gt),
            "map": relativize(args.map) if args.map is not None else None,
        },
        "runtime_sec": time.time() - start,
        "generated_outputs": list_outputs(result["output_files"]),
    }
    save_json(args.out_dir / "unified_analysis_metadata.json", metadata)
    print(f"Unified analysis saved: {args.out_dir}")


if __name__ == "__main__":
    main()
