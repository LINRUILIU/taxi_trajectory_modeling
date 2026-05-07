from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "task_B_tte"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def relativize(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def build_predict_command(
    runner: str,
    model_in: Path,
    input_path: Path,
    output_path: Path,
    osm: Path | None,
    map_cache: Path | None,
    map_force_rebuild: bool,
) -> list[str]:
    if runner == "phase4":
        cmd = [
            sys.executable,
            str(TASK_DIR / "phase4_residual_ensemble.py"),
            "predict",
            "--model-in",
            str(model_in),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    elif runner == "baseline":
        cmd = [
            sys.executable,
            str(TASK_DIR / "baseline_tte.py"),
            "predict",
            "--model-in",
            str(model_in),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        if osm is not None:
            cmd.extend(["--osm", str(osm)])
        if map_cache is not None:
            cmd.extend(["--map-cache", str(map_cache)])
        if map_force_rebuild:
            cmd.append("--map-force-rebuild")
    else:
        raise ValueError(f"Unknown runner: {runner}")
    return cmd


def run_command(cmd: list[str], cwd: Path) -> float:
    start = time.time()
    subprocess.run(cmd, cwd=str(cwd), check=True)
    return time.time() - start


def check_pair(input_path: Path, pred_path: Path) -> list[str]:
    sys.path.insert(0, str(TASK_DIR / "scripts"))
    from smoke_check import check_pair as smoke_check_pair

    return smoke_check_pair(input_path=input_path, pred_path=pred_path, label=pred_path.parent.name)


def prepare_submission(pred_path: Path, out_dir: Path) -> Path:
    sys.path.insert(0, str(TASK_DIR / "scripts"))
    from make_submission import prepare_submission as prepare

    return prepare(pred_path=pred_path, out_dir=out_dir)


def run_bundle(
    *,
    label: str,
    runner: str,
    model_in: Path,
    input_path: Path,
    output_dir: Path,
    bundle_dir: Path,
    osm: Path | None,
    map_cache: Path | None,
    map_force_rebuild: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "pred_test.pkl"

    cmd = build_predict_command(
        runner=runner,
        model_in=model_in,
        input_path=input_path,
        output_path=pred_path,
        osm=osm,
        map_cache=map_cache,
        map_force_rebuild=map_force_rebuild,
    )
    runtime_sec = run_command(cmd, cwd=ROOT)

    issues = check_pair(input_path=input_path, pred_path=pred_path)
    if issues:
        raise RuntimeError("Smoke check failed:\n- " + "\n- ".join(issues))

    submission_path = prepare_submission(pred_path=pred_path, out_dir=bundle_dir)

    metadata = {
        "label": label,
        "timestamp": utc_now_iso(),
        "runner": runner,
        "input_path": relativize(input_path),
        "model_path": relativize(model_in),
        "prediction_path": relativize(pred_path),
        "bundle_path": relativize(submission_path),
        "bundle_dir": relativize(bundle_dir),
        "runtime_sec": runtime_sec,
        "command": cmd,
        "map": {
            "osm": relativize(osm) if osm is not None else None,
            "map_cache": relativize(map_cache) if map_cache is not None else None,
            "force_rebuild": bool(map_force_rebuild),
        },
        "smoke_check": "passed",
    }
    save_json(output_dir / "metadata.json", metadata)
    save_json(output_dir / "onsite_config.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Task B onsite workflow with primary and fallback bundles")
    parser.add_argument("--input", required=True, type=Path, help="Path to classroom test_input.pkl")

    parser.add_argument("--primary-runner", choices=["phase4", "baseline"], default="phase4")
    parser.add_argument("--primary-model", type=Path, default=Path("task_B_tte/model_phase4_residual_ensemble.pkl"))
    parser.add_argument("--primary-output-dir", type=Path, default=Path("task_B_tte/submissions/onsite_primary"))
    parser.add_argument("--primary-bundle-dir", type=Path, default=Path("task_B_tte/submissions/onsite_primary_bundle"))

    parser.add_argument("--fallback-runner", choices=["phase4", "baseline"], default="baseline")
    parser.add_argument("--fallback-model", type=Path, default=Path("task_B_tte/model_baseline_hgb.pkl"))
    parser.add_argument("--fallback-output-dir", type=Path, default=Path("task_B_tte/submissions/onsite_fallback"))
    parser.add_argument("--fallback-bundle-dir", type=Path, default=Path("task_B_tte/submissions/onsite_fallback_bundle"))
    parser.add_argument("--skip-fallback", action="store_true")

    parser.add_argument("--osm", type=Path, default=None)
    parser.add_argument("--map-cache", type=Path, default=Path("task_B_tte/map_segments_cache.pkl"))
    parser.add_argument("--map-force-rebuild", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    summary: dict[str, dict] = {}
    summary["primary"] = run_bundle(
        label="primary",
        runner=args.primary_runner,
        model_in=args.primary_model,
        input_path=args.input,
        output_dir=args.primary_output_dir,
        bundle_dir=args.primary_bundle_dir,
        osm=args.osm,
        map_cache=args.map_cache,
        map_force_rebuild=args.map_force_rebuild,
    )

    if not args.skip_fallback:
        summary["fallback"] = run_bundle(
            label="fallback",
            runner=args.fallback_runner,
            model_in=args.fallback_model,
            input_path=args.input,
            output_dir=args.fallback_output_dir,
            bundle_dir=args.fallback_bundle_dir,
            osm=args.osm,
            map_cache=args.map_cache,
            map_force_rebuild=args.map_force_rebuild,
        )

    summary_path = TASK_DIR / "submissions" / "onsite_run_summary.json"
    save_json(summary_path, summary)
    print(f"Onsite workflow finished. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
