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
from taska.longitudinal_analysis import run_longitudinal_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run longitudinal Task A comparison across multiple versions")
    parser.add_argument("--versions", nargs="+", required=True, help="Version specs in form name=pred8,pred16")
    parser.add_argument("--input-8", required=True, type=Path)
    parser.add_argument("--input-16", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    result = run_longitudinal_analysis(
        version_specs=args.versions,
        input8_path=args.input_8,
        input16_path=args.input_16,
        gt_path=args.gt,
        out_dir=args.out_dir,
        map_path=args.map,
    )
    metadata = {
        "timestamp": utc_now_iso(),
        "version_tag": env_version_tag(),
        "script": "scripts/run_longitudinal_analysis.py",
        "analysis_mode": "longitudinal_multi_version",
        "input_paths": {
            "input_8": relativize(args.input_8),
            "input_16": relativize(args.input_16),
            "gt": relativize(args.gt),
            "map": relativize(args.map) if args.map is not None else None,
            "versions": args.versions,
        },
        "runtime_sec": time.time() - start,
        "generated_outputs": list_outputs(result["output_files"]),
    }
    save_json(args.out_dir / "longitudinal_metadata.json", metadata)
    print(f"Longitudinal analysis saved: {args.out_dir}")


if __name__ == "__main__":
    main()
