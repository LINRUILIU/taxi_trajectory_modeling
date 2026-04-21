import argparse
import json
import math
import pickle
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


DRIVABLE_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
}


def _parse_way_for_overlay(elem: ET.Element) -> Tuple[Optional[str], List[int]]:
    refs: List[int] = []
    highway: Optional[str] = None
    for child in elem:
        if child.tag == "nd":
            ref = child.attrib.get("ref")
            if ref is not None:
                refs.append(int(ref))
        elif child.tag == "tag":
            if child.attrib.get("k") == "highway":
                highway = child.attrib.get("v")
    return highway, refs


def load_road_segments_from_osm(osm_path: Path) -> Dict[str, np.ndarray]:
    print(f"[overlay] pass1 parse drivable ways: {osm_path}")
    needed_nodes = set()
    ways: List[List[int]] = []

    for _, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag == "way":
            highway, refs = _parse_way_for_overlay(elem)
            if highway in DRIVABLE_HIGHWAYS and len(refs) >= 2:
                ways.append(refs)
                needed_nodes.update(refs)
            elem.clear()
        elif elem.tag in {"node", "bounds"}:
            elem.clear()

    print(f"[overlay] pass1 done: ways={len(ways)}, needed_nodes={len(needed_nodes)}")

    print("[overlay] pass2 parse nodes")
    node_coords: Dict[int, Tuple[float, float]] = {}
    for _, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag == "node":
            node_id = int(elem.attrib["id"])
            if node_id in needed_nodes:
                lat = float(elem.attrib["lat"])
                lon = float(elem.attrib["lon"])
                node_coords[node_id] = (lon, lat)
            elem.clear()
        elif elem.tag == "way":
            elem.clear()

    lon1, lat1, lon2, lat2 = [], [], [], []
    for refs in ways:
        for a, b in zip(refs[:-1], refs[1:]):
            pa = node_coords.get(a)
            pb = node_coords.get(b)
            if pa is None or pb is None:
                continue
            lon1.append(pa[0])
            lat1.append(pa[1])
            lon2.append(pb[0])
            lat2.append(pb[1])

    if not lon1:
        return {
            "lon1": np.empty(0, dtype=np.float64),
            "lat1": np.empty(0, dtype=np.float64),
            "lon2": np.empty(0, dtype=np.float64),
            "lat2": np.empty(0, dtype=np.float64),
            "min_lon": np.empty(0, dtype=np.float64),
            "max_lon": np.empty(0, dtype=np.float64),
            "min_lat": np.empty(0, dtype=np.float64),
            "max_lat": np.empty(0, dtype=np.float64),
        }

    a_lon1 = np.asarray(lon1, dtype=np.float64)
    a_lat1 = np.asarray(lat1, dtype=np.float64)
    a_lon2 = np.asarray(lon2, dtype=np.float64)
    a_lat2 = np.asarray(lat2, dtype=np.float64)

    segments = {
        "lon1": a_lon1,
        "lat1": a_lat1,
        "lon2": a_lon2,
        "lat2": a_lat2,
        "min_lon": np.minimum(a_lon1, a_lon2),
        "max_lon": np.maximum(a_lon1, a_lon2),
        "min_lat": np.minimum(a_lat1, a_lat2),
        "max_lat": np.maximum(a_lat1, a_lat2),
    }

    print(f"[overlay] built segments: {a_lon1.size}")
    return segments


def load_or_build_road_segments(osm_path: Path, cache_path: Optional[Path], force_rebuild: bool) -> Dict[str, np.ndarray]:
    if cache_path is not None and cache_path.exists() and not force_rebuild:
        print(f"[overlay] load cache: {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    segments = load_road_segments_from_osm(osm_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(segments, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[overlay] cache saved: {cache_path}")
    return segments


def build_road_segment_grid_index(road_segments: Dict[str, np.ndarray], cell_size_deg: float) -> Dict:
    if road_segments["lon1"].size == 0:
        return {
            "grid": {},
            "min_lon": 0.0,
            "min_lat": 0.0,
            "cell_size_deg": cell_size_deg,
        }

    min_lon = float(np.min(road_segments["min_lon"]))
    min_lat = float(np.min(road_segments["min_lat"]))
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for i in range(road_segments["lon1"].size):
        gx0 = int((float(road_segments["min_lon"][i]) - min_lon) / cell_size_deg)
        gx1 = int((float(road_segments["max_lon"][i]) - min_lon) / cell_size_deg)
        gy0 = int((float(road_segments["min_lat"][i]) - min_lat) / cell_size_deg)
        gy1 = int((float(road_segments["max_lat"][i]) - min_lat) / cell_size_deg)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid[(gx, gy)].append(i)

    return {
        "grid": dict(grid),
        "min_lon": min_lon,
        "min_lat": min_lat,
        "cell_size_deg": cell_size_deg,
    }


def point_to_segments_distance_m(
    lon: float,
    lat: float,
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    meter_per_lon = 111320.0 * cos_lat
    meter_per_lat = 111320.0

    x1 = (lon1 - lon) * meter_per_lon
    y1 = (lat1 - lat) * meter_per_lat
    x2 = (lon2 - lon) * meter_per_lon
    y2 = (lat2 - lat) * meter_per_lat

    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy

    t = np.zeros_like(denom, dtype=np.float64)
    valid = denom > 1e-12
    t[valid] = -(x1[valid] * dx[valid] + y1[valid] * dy[valid]) / denom[valid]
    t = np.clip(t, 0.0, 1.0)

    px = x1 + t * dx
    py = y1 + t * dy
    return np.sqrt(px * px + py * py)


def gather_candidate_segment_indices(
    lon: float,
    lat: float,
    road_segments: Dict[str, np.ndarray],
    seg_index: Dict,
    radius_m: float,
) -> np.ndarray:
    cell_size = seg_index["cell_size_deg"]
    gx = int((lon - seg_index["min_lon"]) / cell_size)
    gy = int((lat - seg_index["min_lat"]) / cell_size)

    deg_lat = radius_m / 111320.0
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    deg_lon = radius_m / (111320.0 * cos_lat)
    ring = max(1, int(math.ceil(max(deg_lat, deg_lon) / cell_size)))

    candidate_idxs: List[int] = []
    grid = seg_index["grid"]
    for dx in range(-ring, ring + 1):
        for dy in range(-ring, ring + 1):
            key = (gx + dx, gy + dy)
            if key in grid:
                candidate_idxs.extend(grid[key])

    if not candidate_idxs:
        return np.empty(0, dtype=np.int64)

    cand = np.unique(np.asarray(candidate_idxs, dtype=np.int64))

    lon_pad = deg_lon
    lat_pad = deg_lat
    mask = (
        (road_segments["max_lon"][cand] >= lon - lon_pad)
        & (road_segments["min_lon"][cand] <= lon + lon_pad)
        & (road_segments["max_lat"][cand] >= lat - lat_pad)
        & (road_segments["min_lat"][cand] <= lat + lat_pad)
    )
    return cand[mask]


def nearest_road_distance_m(
    lon: float,
    lat: float,
    road_segments: Dict[str, np.ndarray],
    seg_index: Dict,
    search_radius_m: float,
) -> float:
    best = float("inf")
    for scale in (1.0, 2.0, 4.0):
        radius = search_radius_m * scale
        cand = gather_candidate_segment_indices(
            lon=lon,
            lat=lat,
            road_segments=road_segments,
            seg_index=seg_index,
            radius_m=radius,
        )
        if cand.size == 0:
            continue

        d = point_to_segments_distance_m(
            lon=lon,
            lat=lat,
            lon1=road_segments["lon1"][cand],
            lat1=road_segments["lat1"][cand],
            lon2=road_segments["lon2"][cand],
            lat2=road_segments["lat2"][cand],
        )
        if d.size == 0:
            continue
        best = min(best, float(np.min(d)))
        if best <= search_radius_m:
            break

    return best


def sample_missing_pred_points(
    input_records: Iterable[Dict],
    pred_records: Iterable[Dict],
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, int]:
    if max_points <= 0:
        return np.empty((0, 2), dtype=np.float64), 0

    pred_by_id = {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in pred_records}
    rng = np.random.default_rng(seed)
    reservoir: List[Tuple[float, float]] = []
    seen = 0

    for item in input_records:
        traj_id = int(item["traj_id"])
        pred = pred_by_id.get(traj_id)
        if pred is None:
            continue

        mask = np.asarray(item["mask"], dtype=bool)
        missing_idx = np.where(~mask)[0]
        for idx in missing_idx:
            p = pred[int(idx)]
            seen += 1
            if len(reservoir) < max_points:
                reservoir.append((float(p[0]), float(p[1])))
            else:
                j = int(rng.integers(0, seen))
                if j < max_points:
                    reservoir[j] = (float(p[0]), float(p[1]))

    if not reservoir:
        return np.empty((0, 2), dtype=np.float64), seen
    return np.asarray(reservoir, dtype=np.float64), seen


def compute_topology_violation_metrics(
    input_records: Iterable[Dict],
    pred_records: Iterable[Dict],
    road_segments: Dict[str, np.ndarray],
    seg_index: Dict,
    violation_threshold_m: float,
    max_eval_points: int,
    seed: int,
) -> Dict[str, float]:
    sample_pts, total_missing_points = sample_missing_pred_points(
        input_records=input_records,
        pred_records=pred_records,
        max_points=max_eval_points,
        seed=seed,
    )
    if sample_pts.size == 0:
        return {
            "topology_violation_rate": math.nan,
            "topology_violation_count": 0,
            "topology_eval_points": 0,
            "topology_total_missing_points": int(total_missing_points),
            "topology_nearest_road_mean_m": math.nan,
            "topology_nearest_road_p95_m": math.nan,
            "topology_violation_threshold_m": float(violation_threshold_m),
        }

    dists = np.empty(sample_pts.shape[0], dtype=np.float64)
    for i in range(sample_pts.shape[0]):
        lon = float(sample_pts[i, 0])
        lat = float(sample_pts[i, 1])
        dists[i] = nearest_road_distance_m(
            lon=lon,
            lat=lat,
            road_segments=road_segments,
            seg_index=seg_index,
            search_radius_m=max(violation_threshold_m * 1.2, 30.0),
        )

    violations = (~np.isfinite(dists)) | (dists > violation_threshold_m)
    violation_count = int(np.sum(violations))
    finite_dists = dists[np.isfinite(dists)]
    mean_dist = float(np.mean(finite_dists)) if finite_dists.size > 0 else math.nan
    p95_dist = float(np.percentile(finite_dists, 95)) if finite_dists.size > 0 else math.nan

    return {
        "topology_violation_rate": float(violation_count / max(1, dists.size)),
        "topology_violation_count": violation_count,
        "topology_eval_points": int(dists.size),
        "topology_total_missing_points": int(total_missing_points),
        "topology_nearest_road_mean_m": mean_dist,
        "topology_nearest_road_p95_m": p95_dist,
        "topology_violation_threshold_m": float(violation_threshold_m),
    }


def draw_road_overlay(
    ax,
    road_segments: Dict[str, np.ndarray],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    color: str,
    alpha: float,
    linewidth: float,
    max_segments: int,
    label: Optional[str] = None,
) -> bool:
    if road_segments["lon1"].size == 0:
        return False

    dx = x_max - x_min
    dy = y_max - y_min
    pad_x = max(0.0008, dx * 0.15)
    pad_y = max(0.0008, dy * 0.15)
    bx0, bx1 = x_min - pad_x, x_max + pad_x
    by0, by1 = y_min - pad_y, y_max + pad_y

    mask = (
        (road_segments["max_lon"] >= bx0)
        & (road_segments["min_lon"] <= bx1)
        & (road_segments["max_lat"] >= by0)
        & (road_segments["min_lat"] <= by1)
    )
    idx = np.where(mask)[0]
    if idx.size == 0:
        return False

    if idx.size > max_segments:
        pick = np.linspace(0, idx.size - 1, max_segments, dtype=np.int64)
        idx = idx[pick]

    segs = np.stack(
        [
            np.stack([road_segments["lon1"][idx], road_segments["lat1"][idx]], axis=1),
            np.stack([road_segments["lon2"][idx], road_segments["lat2"][idx]], axis=1),
        ],
        axis=1,
    )
    lc = LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha, zorder=1)
    if label is not None:
        lc.set_label(label)
    ax.add_collection(lc)
    return True


def haversine_meters(lon1, lat1, lon2, lat2):
    r = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def point_distance_m(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(haversine_meters(p1[0], p1[1], p2[0], p2[1]))


def angle_degrees(prev_p: np.ndarray, cur_p: np.ndarray, next_p: np.ndarray) -> float:
    v1 = cur_p - prev_p
    v2 = next_p - cur_p
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return math.nan
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def get_missing_gap_sizes(mask: np.ndarray) -> np.ndarray:
    gap_sizes = np.zeros(len(mask), dtype=np.int32)
    i = 0
    while i < len(mask):
        if mask[i]:
            i += 1
            continue
        start = i
        while i < len(mask) and not mask[i]:
            i += 1
        end = i - 1
        size = end - start + 1
        gap_sizes[start : end + 1] = size
    return gap_sizes


def global_metrics(errors: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(errors.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p75": float(np.percentile(errors, 75)),
        "p95": float(np.percentile(errors, 95)),
    }


def bucket_gap(dataset_name: str, gap_size: float) -> str:
    g = int(gap_size)
    if dataset_name == "1/8":
        if g <= 2:
            return "1-2"
        if g <= 4:
            return "3-4"
        if g <= 7:
            return "5-7"
        return "8+"
    if g <= 4:
        return "1-4"
    if g <= 8:
        return "5-8"
    if g <= 15:
        return "9-15"
    return "16+"


def bucket_turn(angle: float) -> str:
    if not np.isfinite(angle):
        return "invalid"
    if angle < 30.0:
        return "straight(<30)"
    if angle <= 90.0:
        return "turn(30-90)"
    return "sharp(>90)"


def bucket_position(pos_ratio: float) -> str:
    if pos_ratio < 0.3:
        return "start(0-30%)"
    if pos_ratio < 0.7:
        return "middle(30-70%)"
    return "end(70-100%)"


def bucket_traj_length(length: float) -> str:
    n = int(length)
    if n <= 100:
        return "50-100"
    if n <= 150:
        return "101-150"
    if n <= 240:
        return "151-240"
    return "241+"


def build_speed_bucket_fn(rows: List[Dict]) -> Tuple[Callable[[float], str], Tuple[float, float]]:
    values = np.array([r["speed_var"] for r in rows if np.isfinite(r["speed_var"])], dtype=np.float64)
    if values.size < 100:
        q1, q2 = 1.0, 3.0
    else:
        q1, q2 = np.quantile(values, [0.33, 0.66])
        if q2 - q1 < 1e-9:
            q1, q2 = float(np.min(values)), float(np.max(values))
            if q2 - q1 < 1e-9:
                q1, q2 = 1.0, 3.0

    def _bucket(v: float) -> str:
        if not np.isfinite(v):
            return "invalid"
        if v <= q1:
            return "low"
        if v <= q2:
            return "mid"
        return "high"

    return _bucket, (float(q1), float(q2))


def summarize_by_bucket(rows: List[Dict], dataset_name: str, dimension: str, bucket_fn: Callable[[Dict], str]) -> List[Dict]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        bucket = bucket_fn(row)
        grouped[bucket].append(float(row["error_m"]))

    out = []
    for bucket, vals in grouped.items():
        arr = np.asarray(vals, dtype=np.float64)
        out.append(
            {
                "dataset": dataset_name,
                "dimension": dimension,
                "bucket": bucket,
                "count": int(arr.size),
                "mae": float(np.mean(np.abs(arr))),
                "rmse": float(np.sqrt(np.mean(arr**2))),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }
        )
    out.sort(key=lambda x: (x["dimension"], x["bucket"]))
    return out


def collect_point_rows(
    dataset_name: str,
    input_records: Iterable[Dict],
    pred_records: Iterable[Dict],
    gt_records: Iterable[Dict],
) -> Tuple[List[Dict], np.ndarray, Dict[int, float]]:
    pred_by_id = {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in pred_records}
    gt_by_id = {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in gt_records}

    rows: List[Dict] = []
    all_errors = []
    traj_mae: Dict[int, float] = {}

    for item in input_records:
        traj_id = int(item["traj_id"])
        timestamps = np.asarray(item["timestamps"], dtype=np.int64)
        input_coords = np.asarray(item["coords"], dtype=np.float64)
        mask = np.asarray(item["mask"], dtype=bool)
        missing = ~mask
        if not np.any(missing):
            traj_mae[traj_id] = 0.0
            continue

        pred = pred_by_id[traj_id]
        gt = gt_by_id[traj_id]
        gap_sizes = get_missing_gap_sizes(mask)

        miss_idx = np.where(missing)[0]
        traj_errors = []
        for idx in miss_idx:
            err = point_distance_m(pred[idx], gt[idx])
            traj_errors.append(err)
            all_errors.append(err)

            if 0 < idx < len(gt) - 1:
                turn = angle_degrees(gt[idx - 1], gt[idx], gt[idx + 1])
                dt1 = max(1, int(timestamps[idx] - timestamps[idx - 1]))
                dt2 = max(1, int(timestamps[idx + 1] - timestamps[idx]))
                v1 = point_distance_m(gt[idx - 1], gt[idx]) / dt1
                v2 = point_distance_m(gt[idx], gt[idx + 1]) / dt2
                speed_var = abs(v2 - v1)
            else:
                turn = math.nan
                speed_var = math.nan

            rows.append(
                {
                    "dataset": dataset_name,
                    "traj_id": traj_id,
                    "idx": int(idx),
                    "error_m": float(err),
                    "gap_size": int(gap_sizes[idx]),
                    "turn_angle": float(turn),
                    "position_ratio": float(idx / max(1, len(gt) - 1)),
                    "traj_length": int(len(gt)),
                    "speed_var": float(speed_var),
                }
            )

        traj_mae[traj_id] = float(np.mean(traj_errors))

    return rows, np.asarray(all_errors, dtype=np.float64), traj_mae


def plot_length_distribution(input8: List[Dict], input16: List[Dict], out_path: Path) -> None:
    l8 = np.array([len(item["coords"]) for item in input8], dtype=np.int32)
    l16 = np.array([len(item["coords"]) for item in input16], dtype=np.int32)

    bins = np.arange(50, 246, 5)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    # Draw filled histograms with lower alpha, then add step outlines to prevent visual masking.
    ax.hist(l8, bins=bins, alpha=0.28, label="1/8", color="#1f77b4", histtype="stepfilled", zorder=1)
    ax.hist(l16, bins=bins, alpha=0.28, label="1/16", color="#ff7f0e", histtype="stepfilled", zorder=1)
    ax.hist(l8, bins=bins, color="#1f77b4", histtype="step", linewidth=1.35, zorder=3)
    ax.hist(l16, bins=bins, color="#ff7f0e", histtype="step", linewidth=1.35, zorder=3)
    ax.set_title("Trajectory Length Distribution")
    ax.set_xlabel("Points per trajectory")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def sample_array(arr: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if arr.size <= max_n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=max_n, replace=False)
    return arr[idx]


def plot_interval_distribution(input8: List[Dict], input16: List[Dict], out_path: Path, seed: int) -> None:
    dt8 = []
    dt16 = []
    for item in input8:
        ts = np.asarray(item["timestamps"], dtype=np.int64)
        dt8.append(np.diff(ts))
    for item in input16:
        ts = np.asarray(item["timestamps"], dtype=np.int64)
        dt16.append(np.diff(ts))

    a8 = sample_array(np.concatenate(dt8).astype(np.float64), 300000, seed)
    a16 = sample_array(np.concatenate(dt16).astype(np.float64), 300000, seed + 1)

    bins = np.arange(0, 61, 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(a8, bins=bins, alpha=0.55, label="1/8", color="#1f77b4")
    ax.hist(a16, bins=bins, alpha=0.55, label="1/16", color="#ff7f0e")
    ax.set_title("Timestamp Interval Distribution")
    ax.set_xlabel("Delta time (seconds)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gap_distribution(rows8: List[Dict], rows16: List[Dict], out_path: Path) -> None:
    g8 = np.array([r["gap_size"] for r in rows8], dtype=np.int32)
    g16 = np.array([r["gap_size"] for r in rows16], dtype=np.int32)
    max_gap = int(max(g8.max(initial=1), g16.max(initial=1)))
    bins = np.arange(1, max_gap + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(g8, bins=bins, alpha=0.55, label="1/8", color="#1f77b4")
    ax.hist(g16, bins=bins, alpha=0.55, label="1/16", color="#ff7f0e")
    ax.set_title("Missing Gap Size Distribution")
    ax.set_xlabel("Gap size (consecutive missing points)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_hist(errors8: np.ndarray, errors16: np.ndarray, out_path: Path) -> None:
    cap = float(np.percentile(np.concatenate([errors8, errors16]), 99))
    bins = np.linspace(0.0, max(10.0, cap), 120)
    e8 = errors8[errors8 <= cap]
    e16 = errors16[errors16 <= cap]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(e8, bins=bins, alpha=0.55, label="1/8", color="#1f77b4")
    ax.hist(e16, bins=bins, alpha=0.55, label="1/16", color="#ff7f0e")
    ax.set_title("Point Error Distribution (up to P99)")
    ax.set_xlabel("Haversine error (meters)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_box(errors8: np.ndarray, errors16: np.ndarray, out_path: Path, seed: int) -> None:
    e8 = sample_array(errors8, 100000, seed)
    e16 = sample_array(errors16, 100000, seed + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot([e8, e16], tick_labels=["1/8", "1/16"], showfliers=False)
    ax.set_title("Error Boxplot (outliers hidden)")
    ax.set_ylabel("Haversine error (meters)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_error_vs_gap(rows: List[Dict], dataset_name: str, out_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = len(rows)
    take = min(60000, n)
    idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)

    gap = np.array([rows[i]["gap_size"] for i in idx], dtype=np.float64)
    err = np.array([rows[i]["error_m"] for i in idx], dtype=np.float64)
    angle = np.array([rows[i]["turn_angle"] for i in idx], dtype=np.float64)
    angle = np.where(np.isfinite(angle), np.clip(angle, 0.0, 180.0), 0.0)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sc = ax.scatter(gap, err, c=angle, cmap="viridis", s=6, alpha=0.22, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Turn angle (deg)")
    ax.set_title(f"Error vs Gap Size ({dataset_name})")
    ax.set_xlabel("Gap size")
    ax.set_ylabel("Haversine error (meters)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gt_overlay(gt_records: List[Dict], out_path: Path, seed: int, max_traj: int = 900) -> None:
    rng = random.Random(seed)
    selected = gt_records if len(gt_records) <= max_traj else rng.sample(gt_records, max_traj)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for item in selected:
        c = np.asarray(item["coords"], dtype=np.float64)
        ax.plot(c[:, 0], c[:, 1], color="#2f4f4f", alpha=0.05, linewidth=0.5)
    ax.set_title("Sampled Ground-Truth Trajectory Overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pick_representative_ids(traj_mae: Dict[int, float], k: int = 6) -> List[int]:
    if not traj_mae:
        return []
    pairs = sorted(traj_mae.items(), key=lambda x: x[1])
    n = len(pairs)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    chosen = []
    for q in quantiles[:k]:
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        chosen.append(pairs[idx][0])
    # Deduplicate while keeping order.
    uniq = []
    seen = set()
    for tid in chosen:
        if tid not in seen:
            seen.add(tid)
            uniq.append(tid)
    if len(uniq) < k:
        for tid, _ in reversed(pairs):
            if tid not in seen:
                seen.add(tid)
                uniq.append(tid)
            if len(uniq) >= k:
                break
    return uniq[:k]


def plot_case_figure(
    dataset_name: str,
    chosen_ids: List[int],
    input_records: Dict[int, Dict],
    pred_by_id: Dict[int, np.ndarray],
    gt_by_id: Dict[int, np.ndarray],
    traj_mae: Dict[int, float],
    out_path: Path,
    road_segments: Optional[Dict[str, np.ndarray]] = None,
    overlay_map_on_cases: bool = False,
    map_color: str = "#9fa6ad",
    map_alpha: float = 0.30,
    map_linewidth: float = 0.55,
    map_max_segments_per_case: int = 5000,
) -> None:
    if not chosen_ids:
        return

    cols = 3
    rows = int(math.ceil(len(chosen_ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    road_label_added = False

    for i, traj_id in enumerate(chosen_ids):
        ax = axes[i // cols, i % cols]
        inp = input_records[traj_id]
        mask = np.asarray(inp["mask"], dtype=bool)
        inp_coords = np.asarray(inp["coords"], dtype=np.float64)
        pred = pred_by_id[traj_id]
        gt = gt_by_id[traj_id]

        if overlay_map_on_cases and road_segments is not None:
            x_all = np.concatenate([gt[:, 0], pred[:, 0], inp_coords[mask, 0]])
            y_all = np.concatenate([gt[:, 1], pred[:, 1], inp_coords[mask, 1]])
            added = draw_road_overlay(
                ax=ax,
                road_segments=road_segments,
                x_min=float(np.min(x_all)),
                x_max=float(np.max(x_all)),
                y_min=float(np.min(y_all)),
                y_max=float(np.max(y_all)),
                color=map_color,
                alpha=map_alpha,
                linewidth=map_linewidth,
                max_segments=map_max_segments_per_case,
                label="Road" if not road_label_added else None,
            )
            if added:
                road_label_added = True

        ax.plot(gt[:, 0], gt[:, 1], color="#1f77b4", linewidth=1.2, label="GT", zorder=3)
        ax.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=1.1, label="Pred", zorder=4)
        ax.scatter(inp_coords[mask, 0], inp_coords[mask, 1], s=8, color="black", label="Known", zorder=5)
        ax.set_title(f"traj {traj_id} | MAE {traj_mae[traj_id]:.1f}m")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(alpha=0.2)

    total_axes = rows * cols
    for j in range(len(chosen_ids), total_axes):
        axes[j // cols, j % cols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3)
    fig.suptitle(f"Representative Trajectory Cases ({dataset_name})", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = ["dataset", "dimension", "bucket", "count", "mae", "rmse", "p75", "p95"]
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            vals = [str(row[k]) for k in keys]
            f.write(",".join(vals) + "\n")


def get_bucket_mae(summary_rows: List[Dict], dataset: str, dimension: str, bucket: str) -> float:
    for row in summary_rows:
        if row["dataset"] == dataset and row["dimension"] == dimension and row["bucket"] == bucket:
            return float(row["mae"])
    return math.nan


def make_decision_summary(
    out_path: Path,
    metrics8: Dict[str, float],
    metrics16: Dict[str, float],
    summary_rows: List[Dict],
    speed_q8: Tuple[float, float],
    speed_q16: Tuple[float, float],
) -> None:
    sharp8 = get_bucket_mae(summary_rows, "1/8", "turn_angle", "sharp(>90)")
    straight8 = get_bucket_mae(summary_rows, "1/8", "turn_angle", "straight(<30)")
    sharp16 = get_bucket_mae(summary_rows, "1/16", "turn_angle", "sharp(>90)")
    straight16 = get_bucket_mae(summary_rows, "1/16", "turn_angle", "straight(<30)")

    curve_trigger_8 = np.isfinite(sharp8) and np.isfinite(straight8) and sharp8 >= 1.5 * max(1e-9, straight8)
    curve_trigger_16 = np.isfinite(sharp16) and np.isfinite(straight16) and sharp16 >= 1.5 * max(1e-9, straight16)

    short8 = get_bucket_mae(summary_rows, "1/8", "gap_size", "1-2")
    long8 = get_bucket_mae(summary_rows, "1/8", "gap_size", "5-7")
    short16 = get_bucket_mae(summary_rows, "1/16", "gap_size", "1-4")
    long16 = get_bucket_mae(summary_rows, "1/16", "gap_size", "9-15")

    gap_trigger_8 = np.isfinite(short8) and np.isfinite(long8) and long8 >= 1.3 * max(1e-9, short8)
    gap_trigger_16 = np.isfinite(short16) and np.isfinite(long16) and long16 >= 1.3 * max(1e-9, short16)

    lines = []
    lines.append("# Task A Visualization and Error Analysis Summary")
    lines.append("")
    lines.append("## Global Metrics")
    lines.append(f"- 1/8: MAE={metrics8['mae']:.4f} m, RMSE={metrics8['rmse']:.4f} m, P95={metrics8['p95']:.4f} m")
    lines.append(f"- 1/16: MAE={metrics16['mae']:.4f} m, RMSE={metrics16['rmse']:.4f} m, P95={metrics16['p95']:.4f} m")
    topo8 = metrics8.get("topology_violation_rate", math.nan)
    topo16 = metrics16.get("topology_violation_rate", math.nan)
    if np.isfinite(topo8) and np.isfinite(topo16):
        thr8 = metrics8.get("topology_violation_threshold_m", math.nan)
        thr16 = metrics16.get("topology_violation_threshold_m", math.nan)
        lines.append(
            f"- 1/8 topology_violation_rate={topo8:.4f} (thr={thr8:.1f}m, n={int(metrics8.get('topology_eval_points', 0))})"
        )
        lines.append(
            f"- 1/16 topology_violation_rate={topo16:.4f} (thr={thr16:.1f}m, n={int(metrics16.get('topology_eval_points', 0))})"
        )
    lines.append("")
    lines.append("## Speed Bucket Thresholds")
    lines.append(f"- 1/8 speed_var quantiles: q33={speed_q8[0]:.4f}, q66={speed_q8[1]:.4f} (m/s)")
    lines.append(f"- 1/16 speed_var quantiles: q33={speed_q16[0]:.4f}, q66={speed_q16[1]:.4f} (m/s)")
    lines.append("")
    lines.append("## Trigger Checks")
    lines.append(f"- Curvature trigger 1/8 (sharp >= 1.5x straight): {curve_trigger_8}")
    lines.append(f"- Curvature trigger 1/16 (sharp >= 1.5x straight): {curve_trigger_16}")
    lines.append(f"- Long-gap trigger 1/8 (5-7 >= 1.3x 1-2): {gap_trigger_8}")
    lines.append(f"- Long-gap trigger 1/16 (9-15 >= 1.3x 1-4): {gap_trigger_16}")
    lines.append("")
    lines.append("## Suggested Next Step")

    if curve_trigger_8 or curve_trigger_16:
        lines.append("- Priority 1: add spline or smoothing baseline, then evaluate map-aware correction for turning segments.")
    else:
        lines.append("- Priority 1: add spline baseline first; curvature is not yet a dominant failure mode.")

    if gap_trigger_8 or gap_trigger_16:
        lines.append("- Priority 2: focus on long-gap reconstruction; if spline is still weak, evaluate route-constrained methods.")
    else:
        lines.append("- Priority 2: long-gap growth is mild; prioritize robustness and outlier handling.")

    lines.append("- Keep linear baseline as diagnostic lower bound for all future comparisons.")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task A visualization and error analysis for baseline recovery.")
    parser.add_argument("--input8", type=Path, default=Path("task_A_recovery/val_input_8.pkl"))
    parser.add_argument("--input16", type=Path, default=Path("task_A_recovery/val_input_16.pkl"))
    parser.add_argument("--pred8", type=Path, default=Path("task_A_recovery/pred_linear_val_8.pkl"))
    parser.add_argument("--pred16", type=Path, default=Path("task_A_recovery/pred_linear_val_16.pkl"))
    parser.add_argument("--gt", type=Path, default=Path("task_A_recovery/val_gt.pkl"))
    parser.add_argument("--map", type=Path, default=None, help="Optional OSM XML path used for road overlay on case plots")
    parser.add_argument("--overlay-map-on-cases", action="store_true", help="Overlay road segments on case comparison figures")
    parser.add_argument(
        "--map-road-cache",
        type=Path,
        default=Path("task_A_recovery/map_roads_overlay_cache.pkl"),
        help="Cache file for parsed road segments used in case overlay",
    )
    parser.add_argument("--map-force-rebuild", action="store_true", help="Force rebuilding road overlay cache")
    parser.add_argument("--map-max-segments-per-case", type=int, default=5000, help="Max road segments drawn per case subplot")
    parser.add_argument("--map-alpha", type=float, default=0.30, help="Road overlay alpha on case plots")
    parser.add_argument("--map-linewidth", type=float, default=0.55, help="Road overlay line width on case plots")
    parser.add_argument("--map-color", type=str, default="#9fa6ad", help="Road overlay color on case plots")
    parser.add_argument(
        "--disable-topology-metric",
        action="store_true",
        help="Disable topology violation metric even when --map is provided",
    )
    parser.add_argument(
        "--topology-violation-threshold-m",
        type=float,
        default=35.0,
        help="Distance threshold for topology violation count",
    )
    parser.add_argument(
        "--topology-eval-max-points",
        type=int,
        default=30000,
        help="Max sampled missing points per dataset for topology metric",
    )
    parser.add_argument(
        "--topology-cell-size-deg",
        type=float,
        default=0.002,
        help="Cell size for topology metric road-segment index",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("task_A_recovery/analysis_outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    input8 = load_pickle(args.input8)
    input16 = load_pickle(args.input16)
    pred8 = load_pickle(args.pred8)
    pred16 = load_pickle(args.pred16)
    gt = load_pickle(args.gt)

    rows8, errors8, traj_mae8 = collect_point_rows("1/8", input8, pred8, gt)
    rows16, errors16, traj_mae16 = collect_point_rows("1/16", input16, pred16, gt)

    metrics8 = global_metrics(errors8)
    metrics16 = global_metrics(errors16)

    speed_bucket8, speed_q8 = build_speed_bucket_fn(rows8)
    speed_bucket16, speed_q16 = build_speed_bucket_fn(rows16)

    summary_rows = []
    summary_rows += summarize_by_bucket(rows8, "1/8", "gap_size", lambda r: bucket_gap("1/8", r["gap_size"]))
    summary_rows += summarize_by_bucket(rows8, "1/8", "turn_angle", lambda r: bucket_turn(r["turn_angle"]))
    summary_rows += summarize_by_bucket(rows8, "1/8", "position", lambda r: bucket_position(r["position_ratio"]))
    summary_rows += summarize_by_bucket(rows8, "1/8", "traj_length", lambda r: bucket_traj_length(r["traj_length"]))
    summary_rows += summarize_by_bucket(rows8, "1/8", "speed_var", lambda r: speed_bucket8(r["speed_var"]))

    summary_rows += summarize_by_bucket(rows16, "1/16", "gap_size", lambda r: bucket_gap("1/16", r["gap_size"]))
    summary_rows += summarize_by_bucket(rows16, "1/16", "turn_angle", lambda r: bucket_turn(r["turn_angle"]))
    summary_rows += summarize_by_bucket(rows16, "1/16", "position", lambda r: bucket_position(r["position_ratio"]))
    summary_rows += summarize_by_bucket(rows16, "1/16", "traj_length", lambda r: bucket_traj_length(r["traj_length"]))
    summary_rows += summarize_by_bucket(rows16, "1/16", "speed_var", lambda r: speed_bucket16(r["speed_var"]))

    write_csv(args.out_dir / "bucket_summary.csv", summary_rows)

    need_topology_metric = (not args.disable_topology_metric) and (args.map is not None)
    road_segments = None
    if args.overlay_map_on_cases and args.map is None:
        raise ValueError("--overlay-map-on-cases requires --map <osm_path>")

    if args.map is not None and (args.overlay_map_on_cases or need_topology_metric):
        road_segments = load_or_build_road_segments(
            osm_path=args.map,
            cache_path=args.map_road_cache,
            force_rebuild=args.map_force_rebuild,
        )

    if need_topology_metric and road_segments is not None:
        seg_index = build_road_segment_grid_index(
            road_segments=road_segments,
            cell_size_deg=args.topology_cell_size_deg,
        )
        topo8 = compute_topology_violation_metrics(
            input_records=input8,
            pred_records=pred8,
            road_segments=road_segments,
            seg_index=seg_index,
            violation_threshold_m=args.topology_violation_threshold_m,
            max_eval_points=args.topology_eval_max_points,
            seed=args.seed,
        )
        topo16 = compute_topology_violation_metrics(
            input_records=input16,
            pred_records=pred16,
            road_segments=road_segments,
            seg_index=seg_index,
            violation_threshold_m=args.topology_violation_threshold_m,
            max_eval_points=args.topology_eval_max_points,
            seed=args.seed + 1,
        )
        metrics8.update(topo8)
        metrics16.update(topo16)

    with (args.out_dir / "global_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"1/8": metrics8, "1/16": metrics16}, f, ensure_ascii=False, indent=2)

    plot_length_distribution(input8, input16, args.out_dir / "length_distribution.png")
    plot_interval_distribution(input8, input16, args.out_dir / "interval_distribution.png", args.seed)
    plot_gap_distribution(rows8, rows16, args.out_dir / "missing_gap_distribution.png")
    plot_error_hist(errors8, errors16, args.out_dir / "error_histogram.png")
    plot_error_box(errors8, errors16, args.out_dir / "error_boxplot.png", args.seed)
    plot_error_vs_gap(rows8, "1/8", args.out_dir / "error_vs_gap_8.png", args.seed)
    plot_error_vs_gap(rows16, "1/16", args.out_dir / "error_vs_gap_16.png", args.seed + 1)
    plot_gt_overlay(gt, args.out_dir / "gt_overlay.png", args.seed)

    input8_by_id = {int(item["traj_id"]): item for item in input8}
    input16_by_id = {int(item["traj_id"]): item for item in input16}
    pred8_by_id = {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in pred8}
    pred16_by_id = {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in pred16}
    gt_by_id = {int(item["traj_id"]): np.asarray(item["coords"], dtype=np.float64) for item in gt}

    chosen8 = pick_representative_ids(traj_mae8, k=6)
    chosen16 = pick_representative_ids(traj_mae16, k=6)
    plot_case_figure(
        "1/8",
        chosen8,
        input8_by_id,
        pred8_by_id,
        gt_by_id,
        traj_mae8,
        args.out_dir / "case_comparison_8.png",
        road_segments=road_segments,
        overlay_map_on_cases=args.overlay_map_on_cases,
        map_color=args.map_color,
        map_alpha=args.map_alpha,
        map_linewidth=args.map_linewidth,
        map_max_segments_per_case=args.map_max_segments_per_case,
    )
    plot_case_figure(
        "1/16",
        chosen16,
        input16_by_id,
        pred16_by_id,
        gt_by_id,
        traj_mae16,
        args.out_dir / "case_comparison_16.png",
        road_segments=road_segments,
        overlay_map_on_cases=args.overlay_map_on_cases,
        map_color=args.map_color,
        map_alpha=args.map_alpha,
        map_linewidth=args.map_linewidth,
        map_max_segments_per_case=args.map_max_segments_per_case,
    )

    make_decision_summary(
        out_path=args.out_dir / "decision_summary.md",
        metrics8=metrics8,
        metrics16=metrics16,
        summary_rows=summary_rows,
        speed_q8=speed_q8,
        speed_q16=speed_q16,
    )

    print("Analysis finished.")
    print(f"Output directory: {args.out_dir}")
    print("Generated files:")
    print("- bucket_summary.csv")
    print("- global_metrics.json")
    print("- decision_summary.md")
    print("- length_distribution.png")
    print("- interval_distribution.png")
    print("- missing_gap_distribution.png")
    print("- error_histogram.png")
    print("- error_boxplot.png")
    print("- error_vs_gap_8.png")
    print("- error_vs_gap_16.png")
    print("- gt_overlay.png")
    print("- case_comparison_8.png")
    print("- case_comparison_16.png")
    if np.isfinite(metrics8.get("topology_violation_rate", math.nan)):
        print(
            "- topology_violation_rate(1/8): "
            f"{metrics8['topology_violation_rate']:.4f} "
            f"(thr={metrics8['topology_violation_threshold_m']:.1f}m, n={metrics8['topology_eval_points']})"
        )
    if np.isfinite(metrics16.get("topology_violation_rate", math.nan)):
        print(
            "- topology_violation_rate(1/16): "
            f"{metrics16['topology_violation_rate']:.4f} "
            f"(thr={metrics16['topology_violation_threshold_m']:.1f}m, n={metrics16['topology_eval_points']})"
        )


if __name__ == "__main__":
    main()
