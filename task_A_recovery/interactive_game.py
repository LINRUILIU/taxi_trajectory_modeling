from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analyze_recovery import load_or_build_road_segments
from game_core import (
    CaseData,
    CasePoolResult,
    apply_gap_fill,
    build_case_data,
    build_case_pool,
    build_id_map,
    case_completion_status,
    evaluate_case,
    fill_gap_with_stroke,
    init_player_prediction,
    load_pickle,
    save_pickle,
)
from game_plotting import filter_road_segments_bbox, save_case_overlay_png


DEFAULT_INPUT = {
    "8": Path("task_A_recovery/val_input_8.pkl"),
    "16": Path("task_A_recovery/val_input_16.pkl"),
}

DEFAULT_PRED23 = {
    "8": Path("task_A_recovery/pred_hmm_val_8_b23_e5_gapaware.pkl"),
    "16": Path("task_A_recovery/pred_hmm_val_16_b23_e5_gapaware.pkl"),
}

DEFAULT_PRED28 = {
    "8": Path("task_A_recovery/pred_hmm_val_8_b28_turncurve.pkl"),
    "16": Path("task_A_recovery/pred_hmm_val_16_b28_turncurve.pkl"),
}


def normalize_dataset(dataset: str) -> str:
    ds = dataset.strip()
    if ds in {"8", "1/8"}:
        return "8"
    if ds in {"16", "1/16"}:
        return "16"
    raise ValueError(f"unsupported dataset: {dataset}")


class ViewTransform:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        draw_rect: Tuple[int, int, int, int],
    ):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.rect_x, self.rect_y, self.rect_w, self.rect_h = draw_rect

        wx = max(1e-12, self.x_max - self.x_min)
        wy = max(1e-12, self.y_max - self.y_min)
        sx = self.rect_w / wx
        sy = self.rect_h / wy
        self.scale = min(sx, sy)

        draw_w = wx * self.scale
        draw_h = wy * self.scale
        self.offset_x = self.rect_x + 0.5 * (self.rect_w - draw_w)
        self.offset_y = self.rect_y + 0.5 * (self.rect_h - draw_h)

    def world_to_screen(self, lon: float, lat: float) -> Tuple[int, int]:
        x = self.offset_x + (float(lon) - self.x_min) * self.scale
        y = self.offset_y + (self.y_max - float(lat)) * self.scale
        return int(round(x)), int(round(y))

    def screen_to_world(self, x: int, y: int) -> Tuple[float, float]:
        lon = self.x_min + (float(x) - self.offset_x) / max(1e-12, self.scale)
        lat = self.y_max - (float(y) - self.offset_y) / max(1e-12, self.scale)
        return float(lon), float(lat)

    def contains_screen_point(self, x: int, y: int) -> bool:
        return (
            self.rect_x <= x < self.rect_x + self.rect_w
            and self.rect_y <= y < self.rect_y + self.rect_h
        )


class CaseEditor:
    def __init__(self, case: CaseData):
        self.case = case
        self.player_pred = init_player_prediction(case.input_coords, case.mask)
        self.current_gap_idx = 0
        self.gap_fills: Dict[int, np.ndarray] = {}
        self.gap_vertices: Dict[int, List[Tuple[float, float]]] = {}
        self.gap_started: Dict[int, bool] = {}
        self.gap_completed: Dict[int, bool] = {}
        self.gap_history: Dict[int, List[Dict[str, object]]] = {}
        self.submitted = False

    def current_gap(self):
        if not self.case.gaps:
            return None
        idx = int(np.clip(self.current_gap_idx, 0, len(self.case.gaps) - 1))
        self.current_gap_idx = idx
        return self.case.gaps[idx]

    def set_gap(self, gap_idx: int) -> None:
        if not self.case.gaps:
            self.current_gap_idx = 0
            return
        self.current_gap_idx = int(np.clip(gap_idx, 0, len(self.case.gaps) - 1))

    def next_gap(self) -> None:
        self.set_gap(self.current_gap_idx + 1)

    def prev_gap(self) -> None:
        self.set_gap(self.current_gap_idx - 1)

    def completed_gap_count(self) -> int:
        return int(sum(1 for v in self.gap_completed.values() if v))

    def _snapshot_current_gap_state(self) -> Dict[str, object]:
        fill = self.gap_fills.get(self.current_gap_idx)
        if fill is not None:
            fill = fill.copy()

        vertices = list(self.gap_vertices.get(self.current_gap_idx, []))
        started = bool(self.gap_started.get(self.current_gap_idx, False))
        completed = bool(self.gap_completed.get(self.current_gap_idx, False))

        return {
            "fill": fill,
            "vertices": vertices,
            "started": started,
            "completed": completed,
        }

    def _push_current_gap_history(self) -> None:
        self.gap_history.setdefault(self.current_gap_idx, []).append(self._snapshot_current_gap_state())

    def _restore_current_gap_state(self, snapshot: Dict[str, object]) -> None:
        gap = self.current_gap()
        if gap is None:
            return

        fill = snapshot.get("fill")
        if fill is None:
            self.player_pred[gap.missing_indices] = np.nan
            self.gap_fills.pop(self.current_gap_idx, None)
        else:
            fill_arr = np.asarray(fill, dtype=np.float64)
            self.gap_fills[self.current_gap_idx] = fill_arr
            apply_gap_fill(self.player_pred, gap=gap, points=fill_arr)

        vertices = snapshot.get("vertices", [])
        self.gap_vertices[self.current_gap_idx] = [(float(x), float(y)) for x, y in vertices]
        self.gap_started[self.current_gap_idx] = bool(snapshot.get("started", False))
        self.gap_completed[self.current_gap_idx] = bool(snapshot.get("completed", False))
        self.submitted = False

    def current_gap_points(self) -> Optional[np.ndarray]:
        pts = self.gap_fills.get(self.current_gap_idx)
        if pts is None:
            return None
        return pts

    def current_gap_vertices(self) -> List[Tuple[float, float]]:
        return list(self.gap_vertices.get(self.current_gap_idx, []))

    def current_gap_started(self) -> bool:
        return bool(self.gap_started.get(self.current_gap_idx, False))

    def current_gap_completed(self) -> bool:
        return bool(self.gap_completed.get(self.current_gap_idx, False))

    def set_current_gap_started(self, started: bool, record_history: bool = True) -> None:
        if record_history:
            self._push_current_gap_history()
        self.gap_started[self.current_gap_idx] = bool(started)
        if not started:
            self.gap_completed[self.current_gap_idx] = False
        self.submitted = False

    def set_current_gap_completed(self, completed: bool, record_history: bool = True) -> None:
        if record_history:
            self._push_current_gap_history()
        self.gap_completed[self.current_gap_idx] = bool(completed)
        if completed:
            self.gap_started[self.current_gap_idx] = True
        self.submitted = False

    def _prepare_vertices_for_fill(
        self,
        vertices: List[Tuple[float, float]],
        start_anchor: np.ndarray,
        end_anchor: np.ndarray,
    ) -> np.ndarray:
        if not vertices:
            return np.empty((0, 2), dtype=np.float64)

        arr = np.asarray(vertices, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.empty((0, 2), dtype=np.float64)

        # If user explicitly clicked known anchors, keep them as UI nodes but exclude
        # anchor-like endpoints from missing-point fill calculation.
        if arr.shape[0] > 0:
            d0 = float(np.hypot(arr[0, 0] - float(start_anchor[0]), arr[0, 1] - float(start_anchor[1])))
            if d0 <= 1e-8:
                arr = arr[1:]

        if arr.shape[0] > 0:
            d1 = float(np.hypot(arr[-1, 0] - float(end_anchor[0]), arr[-1, 1] - float(end_anchor[1])))
            if d1 <= 1e-8:
                arr = arr[:-1]

        return arr

    def set_current_gap_vertices(self, vertices: Sequence[Tuple[float, float]], record_history: bool = True) -> None:
        gap = self.current_gap()
        if gap is None:
            return

        if record_history:
            self._push_current_gap_history()

        vtx = [(float(x), float(y)) for x, y in vertices]
        self.gap_vertices[self.current_gap_idx] = vtx

        if len(vtx) == 0:
            self.player_pred[gap.missing_indices] = np.nan
            self.gap_fills.pop(self.current_gap_idx, None)
        else:
            start_anchor = self.case.input_coords[gap.start_idx]
            end_anchor = self.case.input_coords[gap.end_idx]
            stroke_fill = self._prepare_vertices_for_fill(vtx, start_anchor=start_anchor, end_anchor=end_anchor)
            if stroke_fill.shape[0] == 0:
                self.player_pred[gap.missing_indices] = np.nan
                self.gap_fills.pop(self.current_gap_idx, None)
            else:
                filled = fill_gap_with_stroke(
                    start_anchor=start_anchor,
                    end_anchor=end_anchor,
                    stroke_points=stroke_fill,
                    missing_count=gap.missing_count,
                )
                self.gap_fills[self.current_gap_idx] = filled
                apply_gap_fill(self.player_pred, gap=gap, points=filled)

        self.gap_completed[self.current_gap_idx] = False
        self.submitted = False

    def append_current_gap_vertex(self, point: Tuple[float, float], record_history: bool = True) -> None:
        vtx = self.current_gap_vertices()
        vtx.append((float(point[0]), float(point[1])))
        self.set_current_gap_vertices(vtx, record_history=record_history)

    def remove_current_gap_vertex(self, index: int, record_history: bool = True) -> bool:
        vtx = self.current_gap_vertices()
        if index < 0 or index >= len(vtx):
            return False
        vtx.pop(index)
        self.set_current_gap_vertices(vtx, record_history=record_history)
        return True

    def apply_stroke(self, stroke_points: Optional[np.ndarray]) -> bool:
        gap = self.current_gap()
        if gap is None:
            return False

        start_anchor = self.case.input_coords[gap.start_idx]
        end_anchor = self.case.input_coords[gap.end_idx]

        filled = fill_gap_with_stroke(
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            stroke_points=stroke_points,
            missing_count=gap.missing_count,
        )

        self._push_current_gap_history()

        self.gap_fills[self.current_gap_idx] = filled
        if stroke_points is None:
            self.gap_vertices[self.current_gap_idx] = []
        else:
            arr = np.asarray(stroke_points, dtype=np.float64)
            finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
            arr = arr[finite]
            self.gap_vertices[self.current_gap_idx] = [(float(x), float(y)) for x, y in arr]
        self.gap_completed[self.current_gap_idx] = False
        self.submitted = False
        apply_gap_fill(self.player_pred, gap=gap, points=filled)
        return True

    def erase_current_gap_by_mask(self, erase_mask: np.ndarray, record_history: bool = True) -> bool:
        gap = self.current_gap()
        if gap is None:
            return False

        current = self.gap_fills.get(self.current_gap_idx)
        if current is None:
            return False

        erase_mask = np.asarray(erase_mask, dtype=bool)
        if erase_mask.shape != (gap.missing_count,):
            return False
        if not np.any(erase_mask):
            return False

        if record_history:
            self._push_current_gap_history()

        updated = current.copy()
        updated[erase_mask] = np.nan
        self.gap_fills[self.current_gap_idx] = updated
        self.submitted = False
        apply_gap_fill(self.player_pred, gap=gap, points=updated)
        return True

    def undo_current_gap(self) -> bool:
        gap = self.current_gap()
        if gap is None:
            return False

        hist = self.gap_history.get(self.current_gap_idx, [])
        if hist:
            restored = hist.pop()
            self._restore_current_gap_state(restored)
            return True

        if self.current_gap_idx in self.gap_fills:
            self.reset_current_gap(clear_history=False)
            return True

        return False

    def reset_current_gap(self, clear_history: bool = True) -> None:
        gap = self.current_gap()
        if gap is None:
            return

        self.player_pred[gap.missing_indices] = np.nan
        self.gap_fills.pop(self.current_gap_idx, None)
        self.gap_vertices[self.current_gap_idx] = []
        self.gap_started[self.current_gap_idx] = False
        self.gap_completed[self.current_gap_idx] = False
        self.submitted = False
        if clear_history:
            self.gap_history[self.current_gap_idx] = []

    def clear_current_gap_with_history(self) -> None:
        self._push_current_gap_history()
        self.reset_current_gap(clear_history=False)

    def all_gaps_completed(self) -> bool:
        if not self.case.gaps:
            return True
        for i in range(len(self.case.gaps)):
            if not bool(self.gap_completed.get(i, False)):
                return False
        return True

    def replay_case(self) -> None:
        self.player_pred = init_player_prediction(self.case.input_coords, self.case.mask)
        self.current_gap_idx = 0
        self.gap_fills.clear()
        self.gap_vertices.clear()
        self.gap_started.clear()
        self.gap_completed.clear()
        self.gap_history.clear()
        self.submitted = False

    def completion(self) -> Dict[str, int]:
        return case_completion_status(case=self.case, player_pred_coords=self.player_pred)

    def is_complete(self) -> bool:
        c = self.completion()
        return c["unfilled_missing"] == 0


class UIState:
    def __init__(self):
        self.show_map = True
        self.show_help = True
        self.show_gt = False
        self.show_b23 = False
        self.show_b28 = False


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task A interactive trajectory game: freely draw missing segments and export comparison against baselines."
    )
    parser.add_argument("--dataset", choices=["8", "16", "1/8", "1/16"], default="16")

    parser.add_argument("--input", type=Path, default=None, help="Input pkl for selected dataset")
    parser.add_argument("--gt", type=Path, default=Path("task_A_recovery/val_gt.pkl"))
    parser.add_argument("--pred23", type=Path, default=None, help="Prediction pkl for baseline23_e5")
    parser.add_argument("--pred28", type=Path, default=None, help="Prediction pkl for baseline28")

    parser.add_argument("--map", type=Path, default=Path("map"), help="OSM XML path")
    parser.add_argument(
        "--map-road-cache",
        type=Path,
        default=Path("task_A_recovery/map_roads_overlay_cache.pkl"),
        help="Road segments cache for visualization",
    )
    parser.add_argument("--map-force-rebuild", action="store_true")
    parser.add_argument("--disable-map", action="store_true")
    parser.add_argument("--map-max-segments", type=int, default=3500)
    parser.add_argument("--map-max-segments-plot", type=int, default=5000)

    parser.add_argument("--traj-id", type=int, default=-1, help="Play one specific traj_id; -1 means use case pool")
    parser.add_argument("--case-pool-size", type=int, default=30)
    parser.add_argument("--case-pool-json", type=Path, default=None, help="Optional prebuilt case pool json")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-dir", type=Path, default=Path("task_A_recovery/game_outputs"))
    parser.add_argument("--session-name", type=str, default="")

    parser.add_argument("--screen-width", type=int, default=1600)
    parser.add_argument("--screen-height", type=int, default=920)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--snap-radius-px", type=int, default=14, help="Known-point snapping radius in pixels")
    parser.add_argument("--round-label", type=str, default="", help="Optional round label shown in UI panel")
    parser.add_argument("--progress-offset", type=int, default=0, help="Global progress offset for this round")
    parser.add_argument("--progress-total", type=int, default=0, help="Global total case count across rounds")

    parser.add_argument("--no-ui", action="store_true", help="Only build case pool and session metadata")
    return parser.parse_args()


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    ds = normalize_dataset(args.dataset)
    args.dataset = ds

    if args.input is None:
        args.input = DEFAULT_INPUT[ds]
    if args.pred23 is None:
        args.pred23 = DEFAULT_PRED23[ds]
    if args.pred28 is None:
        args.pred28 = DEFAULT_PRED28[ds]

    return args


def _finite_rows(arr: np.ndarray) -> np.ndarray:
    c = np.asarray(arr, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    ok = np.isfinite(c[:, 0]) & np.isfinite(c[:, 1])
    return c[ok]


def compute_play_bbox(
    case: CaseData,
    player_pred_coords: np.ndarray,
    node_vertices: Optional[Sequence[Tuple[float, float]]] = None,
    pad_ratio: float = 0.08,
) -> Tuple[float, float, float, float]:
    chunks = [
        _finite_rows(case.input_coords[case.mask]),
        _finite_rows(player_pred_coords),
    ]

    if node_vertices:
        chunks.append(_finite_rows(np.asarray(node_vertices, dtype=np.float64)))

    valid = [x for x in chunks if x.size > 0]
    if not valid:
        return 0.0, 1.0, 0.0, 1.0

    merged = np.vstack(valid)
    x_min = float(np.min(merged[:, 0]))
    x_max = float(np.max(merged[:, 0]))
    y_min = float(np.min(merged[:, 1]))
    y_max = float(np.max(merged[:, 1]))

    dx = max(1e-6, x_max - x_min)
    dy = max(1e-6, y_max - y_min)
    x_pad = dx * pad_ratio
    y_pad = dy * pad_ratio

    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _iter_finite_segments(coords: np.ndarray) -> List[np.ndarray]:
    c = np.asarray(coords, dtype=np.float64)
    finite = np.isfinite(c[:, 0]) & np.isfinite(c[:, 1])
    segments: List[np.ndarray] = []

    start = None
    for i, ok in enumerate(finite):
        if ok and start is None:
            start = i
        if start is not None and (not ok):
            if i - start >= 2:
                segments.append(c[start:i])
            start = None

    if start is not None and c.shape[0] - start >= 2:
        segments.append(c[start:])

    return segments


def _draw_polyline(surface, pygame, transform: ViewTransform, coords: np.ndarray, color, width: int) -> None:
    for seg in _iter_finite_segments(coords):
        pts = [transform.world_to_screen(p[0], p[1]) for p in seg]
        if len(pts) >= 2:
            pygame.draw.lines(surface, color, False, pts, width)


def _draw_text(surface, font, text: str, x: int, y: int, color) -> None:
    img = font.render(text, True, color)
    surface.blit(img, (x, y))


def _create_ui_font(pygame, size: int):
    candidates = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]

    for name in candidates:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)

    return pygame.font.SysFont(None, size)


def save_case_artifacts(
    editor: CaseEditor,
    case: CaseData,
    case_dir: Path,
    road_segments: Optional[Dict[str, np.ndarray]],
    map_max_segments_plot: int,
) -> Dict:
    case_dir.mkdir(parents=True, exist_ok=True)

    pred_record = [{"traj_id": int(case.traj_id), "coords": editor.player_pred.astype(np.float32)}]
    save_pickle(case_dir / "player_pred.pkl", pred_record)

    metrics = evaluate_case(case=case, player_pred_coords=editor.player_pred)
    completion = editor.completion()
    metrics["completion"] = completion
    metrics["is_complete"] = bool(completion["unfilled_missing"] == 0)

    with (case_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_case_overlay_png(
        case=case,
        player_pred_coords=editor.player_pred,
        out_path=case_dir / "case_overlay.png",
        road_segments=road_segments,
        map_max_segments=map_max_segments_plot,
        metrics=metrics,
    )

    return metrics


def save_session_summary(
    session_dir: Path,
    case_ids: Sequence[int],
    editors: Dict[int, CaseEditor],
    saved_metrics: Dict[int, Dict],
    current_case_index: int,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_case_ids = sorted(saved_metrics.keys())

    pred_records = []
    for tid in case_ids:
        if tid not in editors:
            continue
        editor = editors[tid]
        if not editor.submitted:
            continue
        pred_records.append(
            {
                "traj_id": int(tid),
                "coords": editor.player_pred.astype(np.float32),
            }
        )

    if pred_records:
        save_pickle(session_dir / "player_predictions_all_loaded_cases.pkl", pred_records)

    saved_pred_records = []
    for tid in saved_case_ids:
        case_pred_path = session_dir / "cases" / str(tid) / "player_pred.pkl"
        loaded = False

        if case_pred_path.exists():
            try:
                data = load_pickle(case_pred_path)
                if isinstance(data, list) and data:
                    rec0 = data[0]
                    if isinstance(rec0, dict) and ("coords" in rec0):
                        saved_pred_records.append(
                            {
                                "traj_id": int(tid),
                                "coords": np.asarray(rec0["coords"], dtype=np.float32),
                            }
                        )
                        loaded = True
            except Exception:
                loaded = False

        if loaded:
            continue

        if tid not in editors:
            continue

        editor = editors[tid]
        saved_pred_records.append(
            {
                "traj_id": int(tid),
                "coords": editor.player_pred.astype(np.float32),
            }
        )

    if saved_pred_records:
        save_pickle(session_dir / "player_predictions_saved_cases.pkl", saved_pred_records)

    player_mae = []
    b23_mae = []
    b28_mae = []
    for tid in saved_case_ids:
        m = saved_metrics[tid]
        p = m.get("player", {}).get("mae", math.nan)
        b23 = m.get("baseline23_e5", {}).get("mae", math.nan)
        b28 = m.get("baseline28_turncurve", {}).get("mae", math.nan)
        if np.isfinite(p):
            player_mae.append(float(p))
        if np.isfinite(b23):
            b23_mae.append(float(b23))
        if np.isfinite(b28):
            b28_mae.append(float(b28))

    summary = {
        "saved_case_count": int(len(saved_case_ids)),
        "saved_case_ids": [int(x) for x in saved_case_ids],
        "player_mae_mean": float(np.mean(player_mae)) if player_mae else math.nan,
        "baseline23_mae_mean": float(np.mean(b23_mae)) if b23_mae else math.nan,
        "baseline28_mae_mean": float(np.mean(b28_mae)) if b28_mae else math.nan,
        "current_case_index": int(current_case_index),
        "total_cases": int(len(case_ids)),
    }

    with (session_dir / "session_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    state = {
        "current_case_index": int(current_case_index),
        "saved_case_ids": [int(x) for x in saved_case_ids],
    }
    with (session_dir / "session_state.json").open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def build_session_dir(out_dir: Path, session_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if session_name.strip():
        name = session_name.strip()
    else:
        name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = out_dir / name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def load_case_pool_from_json(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "case_ids" in obj:
        raw = obj["case_ids"]
    elif isinstance(obj, list):
        raw = obj
    else:
        raise ValueError(f"invalid case pool json format: {path}")

    ids = [int(x) for x in raw]
    uniq = []
    seen = set()
    for tid in ids:
        if tid not in seen:
            uniq.append(tid)
            seen.add(tid)
    return uniq


def build_case_pool_and_write(
    case_pool_result: CasePoolResult,
    session_dir: Path,
    dataset: str,
    seed: int,
) -> None:
    out = {
        "dataset": dataset,
        "seed": int(seed),
        "count": int(len(case_pool_result.case_ids)),
        "case_ids": [int(x) for x in case_pool_result.case_ids],
        "metrics_table": case_pool_result.metrics_table,
    }
    with (session_dir / "case_pool.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def load_data(args: argparse.Namespace):
    ensure_exists(args.input, "input")
    ensure_exists(args.gt, "gt")
    ensure_exists(args.pred23, "pred23")
    ensure_exists(args.pred28, "pred28")

    input_records = load_pickle(args.input)
    gt_records = load_pickle(args.gt)
    pred23_records = load_pickle(args.pred23)
    pred28_records = load_pickle(args.pred28)

    input_by_id = build_id_map(input_records)
    gt_by_id = build_id_map(gt_records)
    pred23_by_id = build_id_map(pred23_records)
    pred28_by_id = build_id_map(pred28_records)

    return input_records, gt_records, pred23_records, pred28_records, input_by_id, gt_by_id, pred23_by_id, pred28_by_id


def prepare_case_ids(
    args: argparse.Namespace,
    input_records,
    gt_records,
    pred23_records,
    pred28_records,
) -> CasePoolResult:
    if args.traj_id >= 0:
        metrics_table = [
            {
                "traj_id": float(args.traj_id),
                "mae_b23": math.nan,
                "mae_b28": math.nan,
                "mae_combo": math.nan,
                "missing_count": math.nan,
            }
        ]
        return CasePoolResult(case_ids=[int(args.traj_id)], metrics_table=metrics_table)

    if args.case_pool_json is not None:
        ids = load_case_pool_from_json(args.case_pool_json)
        rows = [{"traj_id": float(x), "mae_b23": math.nan, "mae_b28": math.nan, "mae_combo": math.nan, "missing_count": math.nan} for x in ids]
        return CasePoolResult(case_ids=ids, metrics_table=rows)

    return build_case_pool(
        input_records=input_records,
        gt_records=gt_records,
        pred23_records=pred23_records,
        pred28_records=pred28_records,
        target_count=args.case_pool_size,
        seed=args.seed,
    )


def run_no_ui(
    args: argparse.Namespace,
    session_dir: Path,
    case_pool_result: CasePoolResult,
    input_by_id: Dict[int, Dict],
) -> None:
    valid_ids = [tid for tid in case_pool_result.case_ids if tid in input_by_id]
    out = {
        "dataset": args.dataset,
        "requested_case_count": int(len(case_pool_result.case_ids)),
        "valid_case_count": int(len(valid_ids)),
        "valid_case_ids": [int(x) for x in valid_ids],
    }
    with (session_dir / "no_ui_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("No-UI mode finished.")
    print(f"Session directory: {session_dir}")
    print(f"Valid cases: {len(valid_ids)}")


def run_ui(
    args: argparse.Namespace,
    session_dir: Path,
    case_ids: List[int],
    input_by_id: Dict[int, Dict],
    gt_by_id: Dict[int, Dict],
    pred23_by_id: Dict[int, Dict],
    pred28_by_id: Dict[int, Dict],
    road_segments: Optional[Dict[str, np.ndarray]],
) -> None:
    try:
        import pygame
    except ImportError as e:
        raise ImportError(
            "pygame is not installed. Install with pip install pygame, then rerun interactive_game.py"
        ) from e

    valid_case_ids = [tid for tid in case_ids if tid in input_by_id and tid in gt_by_id and tid in pred23_by_id and tid in pred28_by_id]
    if not valid_case_ids:
        raise RuntimeError("No valid case ids after id intersection checks.")

    case_data_by_id: Dict[int, CaseData] = {}
    for tid in valid_case_ids:
        case_data_by_id[tid] = build_case_data(
            traj_id=tid,
            input_by_id=input_by_id,
            gt_by_id=gt_by_id,
            pred23_by_id=pred23_by_id,
            pred28_by_id=pred28_by_id,
        )

    pygame.init()
    caption = f"TaskA Interactive Game | dataset={args.dataset}"
    if args.round_label.strip():
        caption = f"TaskA Interactive Game | {args.round_label.strip()} | dataset={args.dataset}"
    pygame.display.set_caption(caption)
    screen = pygame.display.set_mode((args.screen_width, args.screen_height))
    clock = pygame.time.Clock()

    font = _create_ui_font(pygame, args.font_size)
    small_font = _create_ui_font(pygame, max(12, args.font_size - 2))

    ui = UIState()

    panel_w = 430
    draw_rect = (0, 0, args.screen_width - panel_w, args.screen_height)

    editors: Dict[int, CaseEditor] = {}
    saved_metrics: Dict[int, Dict] = {}

    case_idx = 0
    line_vertices: List[Tuple[float, float]] = []
    snap_effect_deadline_ms: Dict[int, int] = {}
    snap_effect_duration_ms = 220
    active_case_tid: Optional[int] = None
    active_gap_idx = -1
    status_msg = ""
    status_deadline_ms = 0
    is_left_drawing = False
    last_draw_screen: Optional[Tuple[int, int]] = None
    draw_min_step_px = 6.0

    def current_tid() -> int:
        return int(valid_case_ids[case_idx])

    def current_case() -> CaseData:
        return case_data_by_id[current_tid()]

    def current_editor() -> CaseEditor:
        tid = current_tid()
        if tid not in editors:
            editors[tid] = CaseEditor(current_case())
        return editors[tid]

    def save_current_case() -> None:
        tid = current_tid()
        case = current_case()
        editor = current_editor()
        case_dir = session_dir / "cases" / str(tid)
        metrics = save_case_artifacts(
            editor=editor,
            case=case,
            case_dir=case_dir,
            road_segments=road_segments,
            map_max_segments_plot=args.map_max_segments_plot,
        )
        saved_metrics[tid] = metrics
        save_session_summary(
            session_dir=session_dir,
            case_ids=valid_case_ids,
            editors=editors,
            saved_metrics=saved_metrics,
            current_case_index=case_idx,
        )

    def load_vertices_from_editor(cur_editor: CaseEditor) -> List[Tuple[float, float]]:
        return cur_editor.current_gap_vertices()

    def build_known_screen_points(cur_case: CaseData, cur_transform: ViewTransform) -> List[Tuple[int, int, float, float, int, int]]:
        points: List[Tuple[int, int, float, float, int, int]] = []
        for order, idx in enumerate(cur_case.known_indices):
            lon = float(cur_case.input_coords[idx, 0])
            lat = float(cur_case.input_coords[idx, 1])
            sx, sy = cur_transform.world_to_screen(lon, lat)
            points.append((int(order), int(idx), lon, lat, int(sx), int(sy)))
        return points

    def set_status(msg: str, duration_ms: int = 1400) -> None:
        nonlocal status_msg, status_deadline_ms
        status_msg = msg
        status_deadline_ms = int(pygame.time.get_ticks() + duration_ms)

    def ensure_current_gap_start_state(cur_editor: CaseEditor) -> None:
        cur_gap = cur_editor.current_gap()
        if cur_gap is None:
            return
        if cur_editor.current_gap_idx <= 0:
            return
        if cur_editor.current_gap_started():
            return
        prev_done = bool(cur_editor.gap_completed.get(cur_editor.current_gap_idx - 1, False))
        if prev_done:
            cur_editor.set_current_gap_started(True, record_history=False)

    def try_complete_current_gap(cur_editor: CaseEditor) -> bool:
        cur_gap = cur_editor.current_gap()
        if cur_gap is None:
            return True

        if cur_editor.current_gap_completed():
            return True

        vertices = cur_editor.current_gap_vertices()
        if len(vertices) <= 0:
            return False

        if not cur_editor.current_gap_started():
            cur_editor.set_current_gap_started(True, record_history=False)

        cur_editor.set_current_gap_completed(True, record_history=False)
        return True

    def try_complete_all_gaps(cur_editor: CaseEditor) -> bool:
        if not cur_editor.case.gaps:
            return True

        prev_idx = int(cur_editor.current_gap_idx)
        ok = True
        for gi in range(len(cur_editor.case.gaps)):
            cur_editor.set_gap(gi)
            if not try_complete_current_gap(cur_editor):
                ok = False
                break

        cur_editor.set_gap(prev_idx)
        return ok

    def append_vertex_from_screen(
        cur_editor: CaseEditor,
        mx: int,
        my: int,
        cur_transform: ViewTransform,
        show_status: bool = True,
    ) -> bool:
        nonlocal line_vertices

        cur_gap = cur_editor.current_gap()
        if cur_gap is None:
            return False

        if cur_editor.current_gap_completed():
            return False

        lon, lat = cur_transform.screen_to_world(mx, my)

        cur_vertices = cur_editor.current_gap_vertices()
        if cur_vertices:
            if math.hypot(cur_vertices[-1][0] - lon, cur_vertices[-1][1] - lat) <= 1e-10:
                return False

        if not cur_editor.current_gap_started():
            cur_editor.set_current_gap_started(True, record_history=False)

        cur_editor.append_current_gap_vertex((lon, lat), record_history=True)
        line_vertices = load_vertices_from_editor(cur_editor)

        if show_status:
            set_status("已添加节点；画完按→进入下一段", duration_ms=900)
        return True

    def remove_nearest_vertex(mx: int, my: int, cur_transform: ViewTransform) -> int:
        nonlocal line_vertices
        if not line_vertices:
            return -1

        best_idx = -1
        best_d2 = float("inf")
        for i, (lon, lat) in enumerate(line_vertices):
            sx, sy = cur_transform.world_to_screen(lon, lat)
            d2 = float(sx - mx) * float(sx - mx) + float(sy - my) * float(sy - my)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        return best_idx

    def current_gap_ready_to_advance(cur_editor: CaseEditor) -> bool:
        return try_complete_current_gap(cur_editor)

    running = True

    while running:
        case = current_case()
        editor = current_editor()
        ensure_current_gap_start_state(editor)
        gap = editor.current_gap()

        if active_case_tid != current_tid() or active_gap_idx != editor.current_gap_idx:
            line_vertices = load_vertices_from_editor(editor)
            active_case_tid = current_tid()
            active_gap_idx = editor.current_gap_idx

        x_min, x_max, y_min, y_max = compute_play_bbox(
            case=case,
            player_pred_coords=editor.player_pred,
            node_vertices=line_vertices,
        )

        transform = ViewTransform(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, draw_rect=draw_rect)
        known_screen_points = build_known_screen_points(case, transform)
        target_known_order = None

        road_subset = None
        if ui.show_map:
            road_subset = filter_road_segments_bbox(
                road_segments=road_segments,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                max_segments=args.map_max_segments,
                seed=args.seed + int(case.traj_id),
            )

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break

            if ev.type == pygame.KEYDOWN:
                if ev.key in {pygame.K_ESCAPE, pygame.K_q}:
                    running = False
                    break
                elif ev.key == pygame.K_h:
                    ui.show_help = not ui.show_help
                elif ev.key == pygame.K_m:
                    ui.show_map = not ui.show_map
                elif ev.key == pygame.K_g:
                    ui.show_gt = not ui.show_gt
                elif ev.key == pygame.K_1:
                    ui.show_b23 = not ui.show_b23
                elif ev.key == pygame.K_2:
                    ui.show_b28 = not ui.show_b28
                elif ev.key in {pygame.K_RIGHT, pygame.K_PERIOD}:
                    if current_gap_ready_to_advance(editor):
                        editor.next_gap()
                        ensure_current_gap_start_state(editor)
                        line_vertices = load_vertices_from_editor(editor)
                    else:
                        set_status("当前段还没有绘制节点，不能跳到下一段")
                elif ev.key in {pygame.K_LEFT, pygame.K_COMMA}:
                    editor.prev_gap()
                    ensure_current_gap_start_state(editor)
                    line_vertices = load_vertices_from_editor(editor)
                elif ev.key == pygame.K_u:
                    editor.undo_current_gap()
                    line_vertices = load_vertices_from_editor(editor)
                elif ev.key == pygame.K_z and (ev.mod & pygame.KMOD_CTRL):
                    editor.undo_current_gap()
                    line_vertices = load_vertices_from_editor(editor)
                elif ev.key == pygame.K_r:
                    editor.clear_current_gap_with_history()
                    line_vertices = []
                elif ev.key == pygame.K_BACKSPACE:
                    if line_vertices:
                        editor.remove_current_gap_vertex(len(line_vertices) - 1, record_history=True)
                        line_vertices = load_vertices_from_editor(editor)
                elif ev.key == pygame.K_RETURN:
                    if try_complete_all_gaps(editor):
                        editor.submitted = True
                        set_status("已提交，按N进入下一case")
                    else:
                        set_status("还有未绘制的gap，至少给每段画一个节点后再提交")
                elif ev.key == pygame.K_s:
                    if editor.submitted:
                        save_current_case()
                        set_status("当前case已保存")
                    else:
                        set_status("未提交，S不会保存")
                elif ev.key == pygame.K_n:
                    if case_idx < len(valid_case_ids) - 1:
                        if editor.submitted:
                            save_current_case()
                            set_status("已保存并进入下一case")
                        else:
                            set_status("未提交，已跳过当前case（不记录轨迹）")
                        case_idx += 1
                        line_vertices = []
                        active_case_tid = None
                        active_gap_idx = -1
                    else:
                        set_status("已经是最后一个case")
                elif ev.key == pygame.K_p:
                    if case_idx > 0:
                        case_idx -= 1
                    line_vertices = []
                    active_case_tid = None
                    active_gap_idx = -1
                elif ev.key == pygame.K_t:
                    editor.replay_case()
                    ensure_current_gap_start_state(editor)
                    line_vertices = load_vertices_from_editor(editor)
                    active_gap_idx = editor.current_gap_idx
                    set_status("已重玩当前case，新提交将覆盖旧提交")

            if ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                if transform.contains_screen_point(mx, my):
                    if ev.button == 1:
                        cur_gap = editor.current_gap()
                        if cur_gap is None:
                            continue
                        if editor.current_gap_completed():
                            set_status("当前段已完成，请切换下一段或回到前一段修改")
                            continue
                        appended = append_vertex_from_screen(
                            cur_editor=editor,
                            mx=mx,
                            my=my,
                            cur_transform=transform,
                            show_status=True,
                        )
                        if appended:
                            is_left_drawing = True
                            last_draw_screen = (mx, my)

                    elif ev.button == 3:
                        remove_idx = remove_nearest_vertex(mx=mx, my=my, cur_transform=transform)
                        if remove_idx >= 0:
                            editor.remove_current_gap_vertex(remove_idx, record_history=True)
                            line_vertices = load_vertices_from_editor(editor)
                            set_status("已删除节点并重连")
                        else:
                            set_status("当前段没有可删除节点")

            if ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    is_left_drawing = False
                    last_draw_screen = None

            if ev.type == pygame.MOUSEMOTION:
                if is_left_drawing and ev.buttons[0]:
                    mx, my = ev.pos
                    if not transform.contains_screen_point(mx, my):
                        continue

                    if last_draw_screen is not None:
                        step = math.hypot(float(mx - last_draw_screen[0]), float(my - last_draw_screen[1]))
                        if step < draw_min_step_px:
                            continue

                    appended = append_vertex_from_screen(
                        cur_editor=editor,
                        mx=mx,
                        my=my,
                        cur_transform=transform,
                        show_status=False,
                    )
                    if appended:
                        last_draw_screen = (mx, my)

        # Refresh state after interaction updates.
        gap = editor.current_gap()
        target_known_order = None

        screen.fill((244, 245, 247))

        map_surface = pygame.Surface((draw_rect[2], draw_rect[3]))
        map_surface.fill((250, 250, 251))

        if road_subset is not None:
            for seg in road_subset:
                x1, y1 = transform.world_to_screen(float(seg[0]), float(seg[1]))
                x2, y2 = transform.world_to_screen(float(seg[2]), float(seg[3]))
                pygame.draw.line(map_surface, (190, 193, 198), (x1, y1), (x2, y2), 1)

        if ui.show_gt:
            _draw_polyline(map_surface, pygame, transform, case.gt_coords, (49, 93, 172), 2)
        if ui.show_b23:
            _draw_polyline(map_surface, pygame, transform, case.pred23_coords, (235, 138, 20), 2)
        if ui.show_b28:
            _draw_polyline(map_surface, pygame, transform, case.pred28_coords, (44, 158, 73), 2)

        # Draw user-created node chains only (single visual style, no extra recovery-colored line).
        for gi, g in enumerate(case.gaps):
            started = bool(editor.gap_started.get(gi, False))
            if not started:
                continue

            vertices = editor.gap_vertices.get(gi, [])
            chain_world = [(float(case.input_coords[g.start_idx, 0]), float(case.input_coords[g.start_idx, 1]))]
            chain_world.extend(vertices)
            if bool(editor.gap_completed.get(gi, False)):
                chain_world.append((float(case.input_coords[g.end_idx, 0]), float(case.input_coords[g.end_idx, 1])))

            chain_screen = [transform.world_to_screen(p[0], p[1]) for p in chain_world]
            if len(chain_screen) >= 2:
                pygame.draw.lines(map_surface, (0, 109, 119), False, chain_screen, 2)

        now_ms = int(pygame.time.get_ticks())
        for i, idx, lon, lat, x, y in known_screen_points:
            pygame.draw.circle(map_surface, (20, 20, 20), (x, y), 3)
            lbl = small_font.render(str(int(i)), True, (33, 33, 33))
            map_surface.blit(lbl, (x + 3, y - 9))

        for gap_idx, gap in enumerate(case.gaps):
            p0 = case.input_coords[gap.start_idx]
            p1 = case.input_coords[gap.end_idx]
            mx = 0.5 * (float(p0[0]) + float(p1[0]))
            my = 0.5 * (float(p0[1]) + float(p1[1]))
            sx, sy = transform.world_to_screen(mx, my)
            label = f"dt={int(round(gap.delta_t_sec))}s m={gap.missing_count}"
            color = (110, 110, 110)
            if gap_idx == editor.current_gap_idx:
                color = (44, 91, 144)
            timg = small_font.render(label, True, color)
            map_surface.blit(timg, (sx + 3, sy + 3))

        if len(line_vertices) >= 1:
            pts = [transform.world_to_screen(p[0], p[1]) for p in line_vertices]
            for i, p in enumerate(pts):
                pygame.draw.circle(map_surface, (0, 109, 119), p, 4)
                idx_lbl = small_font.render(str(i + 1), True, (0, 109, 119))
                map_surface.blit(idx_lbl, (p[0] + 4, p[1] + 2))

        screen.blit(map_surface, (draw_rect[0], draw_rect[1]))

        panel_x = draw_rect[2]
        pygame.draw.rect(screen, (236, 238, 241), (panel_x, 0, panel_w, args.screen_height))
        pygame.draw.line(screen, (210, 213, 218), (panel_x, 0), (panel_x, args.screen_height), 2)

        c = editor.completion()
        lines = [
            f"dataset: {args.dataset}",
            f"round: {args.round_label.strip() if args.round_label.strip() else '-'}",
            f"case: {case_idx + 1}/{len(valid_case_ids)}",
            f"traj_id: {case.traj_id}",
            f"gaps: {editor.completed_gap_count()}/{len(case.gaps)}",
            f"missing filled: {c['filled_missing']}/{c['total_missing']}",
            f"submitted: {'yes' if editor.submitted else 'no'}",
            "tool: left click/drag draw | right delete nearest",
            f"map: {'on' if ui.show_map else 'off'}",
            f"GT: {'on' if ui.show_gt else 'off'}",
            f"B23: {'on' if ui.show_b23 else 'off'}",
            f"B28: {'on' if ui.show_b28 else 'off'}",
        ]

        if args.progress_total > 0:
            global_idx = int(args.progress_offset) + case_idx + 1
            lines.append(f"global progress: {global_idx}/{int(args.progress_total)}")

        if gap is not None:
            lines += [
                "",
                f"current gap: {editor.current_gap_idx + 1}/{len(case.gaps)}",
                f"idx: {gap.start_idx}->{gap.end_idx}",
                f"missing: {gap.missing_count}",
                f"nodes: {len(line_vertices)}",
                f"dt: {int(round(gap.delta_t_sec))} s",
                f"known order: {gap.known_start_order}->{gap.known_end_order}",
            ]

            if editor.current_gap_completed():
                lines.append("phase: current gap completed")
            elif not editor.current_gap_started():
                lines.append("phase: draw current gap freely, then press Right")
            else:
                lines.append("phase: keep drawing, press Right to finish this gap")

        lines += [
            "",
            "mouse controls:",
            "left click/drag: draw node chain",
            "right click: delete nearest node",
            "nodes are connected in order by straight lines",
            "Backspace: remove last node",
            "",
            "keyboard:",
            "Left/Right: prev/next gap (Right auto-finishes current gap)",
            "U or Ctrl+Z: undo current gap",
            "R: reset current gap",
            "Enter: auto-finish drawn gaps then submit",
            "G: toggle GT",
            "1: toggle B23",
            "2: toggle B28",
            "M: toggle map",
            "S: save (submitted only)",
            "N: next case (unsubmitted will be skipped)",
            "T: replay current case",
            "P: prev case",
            "H: toggle help",
            "Q or Esc: quit",
        ]

        if status_msg and now_ms <= status_deadline_ms:
            lines += ["", f"status: {status_msg}"]

        y = 18
        for t in lines:
            if (not ui.show_help) and y > 290:
                break
            _draw_text(screen, font if y < 180 else small_font, t, panel_x + 14, y, (34, 34, 34))
            y += 22 if y < 180 else 20

        pygame.display.flip()
        clock.tick(max(1, args.fps))

    save_session_summary(
        session_dir=session_dir,
        case_ids=valid_case_ids,
        editors=editors,
        saved_metrics=saved_metrics,
        current_case_index=case_idx,
    )
    pygame.quit()

    print("Interactive session closed.")
    print(f"Session directory: {session_dir}")


def main() -> None:
    args = parse_args()
    args = resolve_default_paths(args)

    ensure_exists(args.input, "input")
    ensure_exists(args.gt, "gt")
    ensure_exists(args.pred23, "pred23")
    ensure_exists(args.pred28, "pred28")

    if not args.disable_map:
        ensure_exists(args.map, "map")

    session_dir = build_session_dir(out_dir=args.out_dir, session_name=args.session_name)

    (
        input_records,
        gt_records,
        pred23_records,
        pred28_records,
        input_by_id,
        gt_by_id,
        pred23_by_id,
        pred28_by_id,
    ) = load_data(args)

    case_pool_result = prepare_case_ids(
        args=args,
        input_records=input_records,
        gt_records=gt_records,
        pred23_records=pred23_records,
        pred28_records=pred28_records,
    )

    build_case_pool_and_write(
        case_pool_result=case_pool_result,
        session_dir=session_dir,
        dataset=args.dataset,
        seed=args.seed,
    )

    if args.no_ui:
        run_no_ui(
            args=args,
            session_dir=session_dir,
            case_pool_result=case_pool_result,
            input_by_id=input_by_id,
        )
        return

    road_segments = None
    if not args.disable_map:
        road_segments = load_or_build_road_segments(
            osm_path=args.map,
            cache_path=args.map_road_cache,
            force_rebuild=args.map_force_rebuild,
        )

    run_ui(
        args=args,
        session_dir=session_dir,
        case_ids=case_pool_result.case_ids,
        input_by_id=input_by_id,
        gt_by_id=gt_by_id,
        pred23_by_id=pred23_by_id,
        pred28_by_id=pred28_by_id,
        road_segments=road_segments,
    )


if __name__ == "__main__":
    main()
