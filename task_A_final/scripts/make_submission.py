from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.io_utils import copy_file, save_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final submission files")
    parser.add_argument("--pred-8", required=True, type=Path)
    parser.add_argument("--pred-16", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out8 = args.out_dir / "pred_test_8.pkl"
    out16 = args.out_dir / "pred_test_16.pkl"
    copy_file(args.pred_8, out8)
    copy_file(args.pred_16, out16)
    save_text(
        args.out_dir / "submission_manifest.md",
        "\n".join(
            [
                "# Submission Manifest",
                "",
                f"- pred_test_8.pkl <- {args.pred_8}",
                f"- pred_test_16.pkl <- {args.pred_16}",
            ]
        )
        + "\n",
    )
    print(f"Submission files prepared under: {args.out_dir}")


if __name__ == "__main__":
    main()
