from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from taska.analyze import run_static_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task A static EDA")
    parser.add_argument("--input-8", required=True, type=Path)
    parser.add_argument("--input-16", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_static_eda(args.input_8, args.input_16, args.out_dir)
    print(f"Static EDA saved: {args.out_dir}")


if __name__ == "__main__":
    main()
