from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.analyze import analyze_predictions
from taska.io_utils import env_version_tag, list_outputs, relativize, save_json, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task A final analysis pipeline")
    parser.add_argument("--input-8", required=True, type=Path)
    parser.add_argument("--input-16", required=True, type=Path)
    parser.add_argument("--pred-8", required=True, type=Path)
    parser.add_argument("--pred-16", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    result = analyze_predictions(
        input8_path=args.input_8,
        input16_path=args.input_16,
        pred8_path=args.pred_8,
        pred16_path=args.pred_16,
        gt_path=args.gt,
        out_dir=args.out_dir,
    )
    metadata = {
        "run_name": args.out_dir.parent.name if args.out_dir.parent != args.out_dir else args.out_dir.name,
        "timestamp": utc_now_iso(),
        "version_tag": env_version_tag(),
        "script": "scripts/run_analyze.py",
        "predict_mode": None,
        "analysis_mode": "gap_level_official_x_shape",
        "input_paths": {
            "input_8": relativize(args.input_8),
            "input_16": relativize(args.input_16),
            "pred_8": relativize(args.pred_8),
            "pred_16": relativize(args.pred_16),
            "gt": relativize(args.gt),
        },
        "runtime_sec": time.time() - start,
        "generated_outputs": list_outputs(result["output_files"]),
        "summary": result["quadrant_summary"],
    }
    save_json(args.out_dir / "analysis_metadata.json", metadata)
    print(f"Analysis saved: {args.out_dir}")


if __name__ == "__main__":
    main()
