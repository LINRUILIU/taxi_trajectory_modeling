import math
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


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


MAP_FEATURE_NAMES = [
    "map_snap_mean_m",
    "map_snap_p90_m",
    "map_snap_max_m",
    "map_snap_ratio_20m",
    "map_snap_ratio_50m",
    "map_candidate_density_mean",
    "map_candidate_density_p90",
    "map_heading_diff_mean",
    "map_heading_diff_p90",
    "map_no_candidate_ratio",
]


def haversine_meters(lon1, lat1, lon2, lat2):
    r = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r * c


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1r = math.radians(lon1)
    lat1r = math.radians(lat1)
    lon2r = math.radians(lon2)
    lat2r = math.radians(lat2)

    dlon = lon2r - lon1r
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float("nan")
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _parse_way_for_segments(elem: ET.Element) -> Tuple[Optional[str], List[int]]:
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
    print(f"[mapfeat] pass1 parse drivable ways: {osm_path}")
    needed_nodes = set()
    ways: List[List[int]] = []

    for _, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag == "way":
            highway, refs = _parse_way_for_segments(elem)
            if highway in DRIVABLE_HIGHWAYS and len(refs) >= 2:
                ways.append(refs)
                needed_nodes.update(refs)
            elem.clear()
        elif elem.tag in {"node", "bounds"}:
            elem.clear()

    print(f"[mapfeat] pass1 done: ways={len(ways)}, needed_nodes={len(needed_nodes)}")

    print("[mapfeat] pass2 parse nodes")
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

    print(f"[mapfeat] built segments: {a_lon1.size}")
    return segments


def load_or_build_road_segments(osm_path: Path, cache_path: Optional[Path], force_rebuild: bool) -> Dict[str, np.ndarray]:
    if cache_path is not None and cache_path.exists() and not force_rebuild:
        print(f"[mapfeat] load cache: {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    segments = load_road_segments_from_osm(osm_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(segments, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[mapfeat] cache saved: {cache_path}")
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


def _segment_bearing_deg(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)

    dlon = lon2r - lon1r
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    out = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0

    near_zero = (np.abs(x) < 1e-12) & (np.abs(y) < 1e-12)
    out = out.astype(np.float64)
    out[near_zero] = np.nan
    return out


def build_map_runtime(
    osm_path: Path,
    cache_path: Optional[Path],
    force_rebuild: bool,
    cell_size_deg: float,
) -> Dict:
    road_segments = load_or_build_road_segments(osm_path=osm_path, cache_path=cache_path, force_rebuild=force_rebuild)
    seg_index = build_road_segment_grid_index(road_segments=road_segments, cell_size_deg=cell_size_deg)
    seg_bearing = _segment_bearing_deg(
        road_segments["lon1"],
        road_segments["lat1"],
        road_segments["lon2"],
        road_segments["lat2"],
    )
    return {
        "road_segments": road_segments,
        "seg_index": seg_index,
        "seg_bearing_deg": seg_bearing,
    }


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

    cand = np.asarray(candidate_idxs, dtype=np.int64)
    cand = np.unique(cand)

    lon_pad = deg_lon
    lat_pad = deg_lat
    bbox_mask = (
        (road_segments["max_lon"][cand] >= lon - lon_pad)
        & (road_segments["min_lon"][cand] <= lon + lon_pad)
        & (road_segments["max_lat"][cand] >= lat - lat_pad)
        & (road_segments["min_lat"][cand] <= lat + lat_pad)
    )
    cand = cand[bbox_mask]
    return cand


def _safe_stat(arr: np.ndarray, fn, default: float) -> float:
    if arr.size == 0:
        return float(default)
    v = float(fn(arr))
    if not np.isfinite(v):
        return float(default)
    return v


def extract_map_features_for_coords(
    coords: np.ndarray,
    map_runtime: Optional[Dict],
    query_radius_m: float,
    max_points: int,
    near_threshold_1_m: float,
    near_threshold_2_m: float,
) -> np.ndarray:
    if map_runtime is None:
        return np.zeros(len(MAP_FEATURE_NAMES), dtype=np.float32)

    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros(len(MAP_FEATURE_NAMES), dtype=np.float32)

    n = pts.shape[0]
    if n <= max_points:
        sample_idx = np.arange(n, dtype=np.int64)
    else:
        sample_idx = np.unique(np.linspace(0, n - 1, num=max_points, dtype=np.int64))

    road_segments = map_runtime["road_segments"]
    seg_index = map_runtime["seg_index"]
    seg_bearing = map_runtime["seg_bearing_deg"]

    nearest_dists: List[float] = []
    local_density: List[float] = []
    heading_diffs: List[float] = []
    no_cand = 0

    for i in sample_idx:
        lon = float(pts[i, 0])
        lat = float(pts[i, 1])
        cand = gather_candidate_segment_indices(
            lon=lon,
            lat=lat,
            road_segments=road_segments,
            seg_index=seg_index,
            radius_m=query_radius_m,
        )

        if cand.size == 0:
            no_cand += 1
            nearest_dists.append(float(query_radius_m * 2.0))
            local_density.append(0.0)
            continue

        dists = point_to_segments_distance_m(
            lon=lon,
            lat=lat,
            lon1=road_segments["lon1"][cand],
            lat1=road_segments["lat1"][cand],
            lon2=road_segments["lon2"][cand],
            lat2=road_segments["lat2"][cand],
        )

        best_local = int(np.argmin(dists))
        best_dist = float(dists[best_local])
        nearest_dists.append(best_dist)
        local_density.append(float(np.sum(dists <= query_radius_m)))

        if 0 < i < (n - 1):
            heading = bearing_deg(
                lon1=float(pts[i - 1, 0]),
                lat1=float(pts[i - 1, 1]),
                lon2=float(pts[i + 1, 0]),
                lat2=float(pts[i + 1, 1]),
            )
            seg_idx = int(cand[best_local])
            seg_h = float(seg_bearing[seg_idx])
            if np.isfinite(heading) and np.isfinite(seg_h):
                heading_diffs.append(float(angle_diff_deg(heading, seg_h)))

    d_arr = np.asarray(nearest_dists, dtype=np.float64)
    den_arr = np.asarray(local_density, dtype=np.float64)
    hd_arr = np.asarray(heading_diffs, dtype=np.float64)

    near1 = min(float(near_threshold_1_m), float(near_threshold_2_m))
    near2 = max(float(near_threshold_1_m), float(near_threshold_2_m))

    feat = np.array(
        [
            _safe_stat(d_arr, np.mean, default=float(query_radius_m * 2.0)),
            _safe_stat(d_arr, lambda x: np.percentile(x, 90), default=float(query_radius_m * 2.0)),
            _safe_stat(d_arr, np.max, default=float(query_radius_m * 2.0)),
            float(np.mean(d_arr <= near1)) if d_arr.size else 0.0,
            float(np.mean(d_arr <= near2)) if d_arr.size else 0.0,
            _safe_stat(den_arr, np.mean, default=0.0),
            _safe_stat(den_arr, lambda x: np.percentile(x, 90), default=0.0),
            _safe_stat(hd_arr, np.mean, default=90.0),
            _safe_stat(hd_arr, lambda x: np.percentile(x, 90), default=90.0),
            float(no_cand / max(1, sample_idx.size)),
        ],
        dtype=np.float64,
    )

    feat[~np.isfinite(feat)] = 0.0
    return feat.astype(np.float32)
