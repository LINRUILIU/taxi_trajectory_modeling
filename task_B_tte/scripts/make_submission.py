from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_submission(pred_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pred_test.pkl"
    shutil.copy2(pred_path, out_path)
    save_text(
        out_dir / "submission_manifest.md",
        "\n".join(
            [
                "# Submission Manifest",
                "",
                f"- pred_test.pkl <- {pred_path}",
            ]
        )
        + "\n",
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Task B submission file")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = prepare_submission(args.pred, args.out_dir)
    print(f"Submission file prepared: {out_path}")


if __name__ == "__main__":
    main()
