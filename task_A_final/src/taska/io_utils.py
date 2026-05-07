from __future__ import annotations

import json
import os
import pickle
import shutil
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def project_root() -> Path:
    return repo_root() / "task_A_final"


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_text_config(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must deserialize to a dict")
    return data


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pathify(path_str: str | Path | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root() / p


def relativize(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except Exception:
        return str(Path(path).resolve())


def ensure_prediction_records(records: Sequence[Dict[str, Any]]) -> None:
    if not isinstance(records, Sequence):
        raise TypeError("Prediction object must be a sequence")
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise TypeError(f"Prediction item #{idx} must be a dict")
        if "traj_id" not in item or "coords" not in item:
            raise ValueError(f"Prediction item #{idx} missing traj_id/coords")


def resolve_run_dir(out_path: Path) -> Path:
    return out_path.parent


def snapshot_config(config_path: Path, run_dir: Path, suffix: str = "config.snapshot.yaml") -> Path:
    dst = run_dir / suffix
    copy_file(config_path, dst)
    return dst


def write_metadata(run_dir: Path, metadata: Dict[str, Any]) -> Path:
    path = run_dir / "metadata.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if isinstance(existing, dict):
            merged = dict(existing)
            runs = list(existing.get("prediction_runs", [])) if isinstance(existing.get("prediction_runs"), list) else []
            runs.append(metadata)
            merged["prediction_runs"] = runs
            merged["last_updated"] = utc_now_iso()
            save_json(path, merged)
            return path
    save_json(path, metadata)
    return path


def list_outputs(paths: Iterable[Path]) -> list[str]:
    return [relativize(p) for p in paths if p.exists()]


def env_version_tag() -> str:
    return os.environ.get("TASKA_VERSION_TAG", "manual")
