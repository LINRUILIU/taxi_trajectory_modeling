import argparse
import heapq
import math
import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

ROAD_CLASS_PENALTY = {
    "motorway": 1.00,
    "motorway_link": 1.04,
    "trunk": 1.02,
    "trunk_link": 1.06,
    "primary": 1.08,
    "primary_link": 1.12,
    "secondary": 1.15,
    "secondary_link": 1.20,
    "tertiary": 1.24,
    "tertiary_link": 1.28,
    "unclassified": 1.35,
    "residential": 1.42,
    "living_street": 1.52,
    "service": 1.65,
    "road": 1.32,
}


_WORKER_GRAPH: Optional[Dict] = None
_WORKER_PREDICT_KWARGS: Optional[Dict] = None


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f)


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
    b = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return b


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _safe_interpolate_1d(x_all: np.ndarray, x_known: np.ndarray, y_known: np.ndarray) -> np.ndarray:
    if len(x_known) == 0:
        return np.full_like(x_all, np.nan, dtype=np.float64)
    if len(x_known) == 1:
        return np.full_like(x_all, y_known[0], dtype=np.float64)
    return np.interp(x_all, x_known, y_known)


def _pchip_endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / max(1e-12, (h0 + h1))
    if m * d0 <= 0.0:
        return 0.0
    if d0 * d1 < 0.0 and abs(m) > abs(3.0 * d0):
        return 3.0 * d0
    return m


def _safe_pchip_interpolate_1d(x_all: np.ndarray, x_known: np.ndarray, y_known: np.ndarray) -> np.ndarray:
    n = len(x_known)
    if n == 0:
        return np.full_like(x_all, np.nan, dtype=np.float64)
    if n == 1:
        return np.full_like(x_all, y_known[0], dtype=np.float64)
    if n == 2:
        return np.interp(x_all, x_known, y_known)

    h = np.diff(x_known).astype(np.float64)
    if np.any(h <= 0.0):
        return np.interp(x_all, x_known, y_known)

    delta = np.diff(y_known).astype(np.float64) / h
    m = np.zeros(n, dtype=np.float64)

    for i in range(1, n - 1):
        d_im1 = delta[i - 1]
        d_i = delta[i]
        if d_im1 == 0.0 or d_i == 0.0 or d_im1 * d_i < 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            den = (w1 / d_im1 + w2 / d_i)
            if abs(den) < 1e-12:
                m[i] = 0.0
            else:
                m[i] = (w1 + w2) / den

    m[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    xq = np.asarray(x_all, dtype=np.float64)
    yq = np.empty_like(xq, dtype=np.float64)

    left_mask = xq <= x_known[0]
    right_mask = xq >= x_known[-1]
    yq[left_mask] = y_known[0]
    yq[right_mask] = y_known[-1]

    mid_mask = ~(left_mask | right_mask)
    if np.any(mid_mask):
        xm = xq[mid_mask]
        idx = np.searchsorted(x_known, xm, side="right") - 1
        idx = np.clip(idx, 0, n - 2)

        x0 = x_known[idx]
        x1 = x_known[idx + 1]
        y0 = y_known[idx]
        y1 = y_known[idx + 1]
        m0 = m[idx]
        m1 = m[idx + 1]
        hi = x1 - x0
        t = (xm - x0) / hi

        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        ym = h00 * y0 + h10 * hi * m0 + h01 * y1 + h11 * hi * m1
        yq[mid_mask] = ym

    return yq


def smooth_segment_points(points: np.ndarray, window: int, sigma: float) -> np.ndarray:
    if window <= 0 or points.shape[0] < 3:
        return points

    radius = int(max(1, window))
    sigma_val = float(max(1e-3, sigma))

    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_val) ** 2)
    kernel = kernel / np.sum(kernel)

    out = points.copy().astype(np.float64)
    n = points.shape[0]
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)

        k_lo = radius - (i - lo)
        k_hi = radius + (hi - i)
        w = kernel[k_lo:k_hi]
        w = w / max(1e-12, np.sum(w))
        out[i] = np.sum(points[lo:hi] * w[:, None], axis=0)

    return out.astype(points.dtype)


def polyline_max_turn_deg(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0

    max_turn = 0.0
    for i in range(1, points.shape[0] - 1):
        b_in = bearing_deg(
            float(points[i - 1, 0]),
            float(points[i - 1, 1]),
            float(points[i, 0]),
            float(points[i, 1]),
        )
        b_out = bearing_deg(
            float(points[i, 0]),
            float(points[i, 1]),
            float(points[i + 1, 0]),
            float(points[i + 1, 1]),
        )
        if np.isfinite(b_in) and np.isfinite(b_out):
            max_turn = max(max_turn, float(angle_diff_deg(b_out, b_in)))

    return max_turn


def limit_polyline_turn_deg(points: np.ndarray, max_turn_deg: float, num_iters: int) -> np.ndarray:
    if points.shape[0] < 3:
        return points

    turn_cap = float(np.clip(max_turn_deg, 0.0, 179.0))
    if turn_cap >= 179.0:
        return points

    out = points.copy().astype(np.float64)
    iters = max(1, int(num_iters))
    for _ in range(iters):
        changed = False
        for i in range(1, out.shape[0] - 1):
            b_in = bearing_deg(
                float(out[i - 1, 0]),
                float(out[i - 1, 1]),
                float(out[i, 0]),
                float(out[i, 1]),
            )
            b_out = bearing_deg(
                float(out[i, 0]),
                float(out[i, 1]),
                float(out[i + 1, 0]),
                float(out[i + 1, 1]),
            )
            if not (np.isfinite(b_in) and np.isfinite(b_out)):
                continue

            turn = float(angle_diff_deg(b_out, b_in))
            if turn <= turn_cap:
                continue

            out[i] = 0.5 * (out[i - 1] + out[i + 1])
            changed = True

        if not changed:
            break

    return out.astype(points.dtype)


def linear_interpolate_traj(timestamps: np.ndarray, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x_all = timestamps.astype(np.float64)
    known_idx = np.where(mask)[0]
    x_known = x_all[known_idx]

    lon_known = coords[known_idx, 0].astype(np.float64)
    lat_known = coords[known_idx, 1].astype(np.float64)

    lon_full = _safe_interpolate_1d(x_all, x_known, lon_known)
    lat_full = _safe_interpolate_1d(x_all, x_known, lat_known)
    pred = np.stack([lon_full, lat_full], axis=1)

    pred[mask] = coords[mask]
    return pred.astype(np.float32)


def pchip_interpolate_traj(timestamps: np.ndarray, coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x_all = timestamps.astype(np.float64)
    known_idx = np.where(mask)[0]
    x_known = x_all[known_idx]

    lon_known = coords[known_idx, 0].astype(np.float64)
    lat_known = coords[known_idx, 1].astype(np.float64)

    lon_full = _safe_pchip_interpolate_1d(x_all, x_known, lon_known)
    lat_full = _safe_pchip_interpolate_1d(x_all, x_known, lat_known)
    pred = np.stack([lon_full, lat_full], axis=1)

    pred[mask] = coords[mask]
    return pred.astype(np.float32)


def interpolate_traj(timestamps: np.ndarray, coords: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "pchip":
        return pchip_interpolate_traj(timestamps=timestamps, coords=coords, mask=mask)
    return linear_interpolate_traj(timestamps=timestamps, coords=coords, mask=mask)


def parse_oneway_value(v: str) -> str:
    vv = (v or "").strip().lower()
    if vv in {"yes", "1", "true"}:
        return "forward"
    if vv in {"-1", "reverse"}:
        return "reverse"
    return "both"


def is_drivable_highway(v: Optional[str]) -> bool:
    return v in DRIVABLE_HIGHWAYS


def get_road_class_penalty(highway: Optional[str]) -> float:
    if highway is None:
        return 1.40
    return float(ROAD_CLASS_PENALTY.get(highway, 1.40))


def _parse_way_element(elem: ET.Element) -> Tuple[Optional[str], str, List[int]]:
    refs: List[int] = []
    highway: Optional[str] = None
    oneway = "both"

    for child in elem:
        if child.tag == "nd":
            ref = child.attrib.get("ref")
            if ref is not None:
                refs.append(int(ref))
        elif child.tag == "tag":
            k = child.attrib.get("k")
            v = child.attrib.get("v", "")
            if k == "highway":
                highway = v
            elif k == "oneway":
                oneway = parse_oneway_value(v)

    return highway, oneway, refs


def build_road_graph_from_osm(osm_path: Path) -> Dict:
    print(f"[map] pass1 parse ways: {osm_path}")
    needed_node_ids = set()
    ways: List[Tuple[List[int], str, float]] = []
    bounds = None

    for _, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag == "bounds":
            bounds = {
                "min_lat": float(elem.attrib.get("minlat", "nan")),
                "min_lon": float(elem.attrib.get("minlon", "nan")),
                "max_lat": float(elem.attrib.get("maxlat", "nan")),
                "max_lon": float(elem.attrib.get("maxlon", "nan")),
            }
            elem.clear()
        elif elem.tag == "way":
            highway, oneway, refs = _parse_way_element(elem)
            if is_drivable_highway(highway) and len(refs) >= 2:
                ways.append((refs, oneway, get_road_class_penalty(highway)))
                needed_node_ids.update(refs)
            elem.clear()
        elif elem.tag == "node":
            elem.clear()

    print(f"[map] pass1 done: drivable_ways={len(ways)}, needed_nodes={len(needed_node_ids)}")
    if len(ways) == 0:
        raise ValueError("No drivable ways found in OSM file.")

    print("[map] pass2 parse nodes")
    node_coords: Dict[int, Tuple[float, float]] = {}
    for _, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag == "node":
            node_id = int(elem.attrib["id"])
            if node_id in needed_node_ids:
                lat = float(elem.attrib["lat"])
                lon = float(elem.attrib["lon"])
                node_coords[node_id] = (lon, lat)
            elem.clear()
        elif elem.tag == "way":
            elem.clear()

    print(f"[map] pass2 done: kept_nodes={len(node_coords)}")

    adjacency: Dict[int, List[Tuple[int, float, float, float]]] = defaultdict(list)
    edge_count = 0

    for refs, oneway, road_penalty in ways:
        for a, b in zip(refs[:-1], refs[1:]):
            pa = node_coords.get(a)
            pb = node_coords.get(b)
            if pa is None or pb is None:
                continue

            dist = float(haversine_meters(pa[0], pa[1], pb[0], pb[1]))
            if dist <= 0.0:
                continue

            br_ab = bearing_deg(pa[0], pa[1], pb[0], pb[1])
            br_ba = bearing_deg(pb[0], pb[1], pa[0], pa[1])

            if oneway in {"both", "forward"}:
                adjacency[a].append((b, dist, br_ab, road_penalty))
                edge_count += 1
            if oneway in {"both", "reverse"}:
                adjacency[b].append((a, dist, br_ba, road_penalty))
                edge_count += 1

    graph_nodes = set(adjacency.keys())
    for src, nbrs in adjacency.items():
        for dst, _, _, _ in nbrs:
            graph_nodes.add(dst)

    node_ids = np.array(sorted(graph_nodes), dtype=np.int64)
    lons = np.empty(len(node_ids), dtype=np.float64)
    lats = np.empty(len(node_ids), dtype=np.float64)

    for i, nid in enumerate(node_ids):
        lon, lat = node_coords[nid]
        lons[i] = lon
        lats[i] = lat

    print(f"[map] graph done: nodes={len(node_ids)}, directed_edges={edge_count}")

    return {
        "bounds": bounds,
        "node_ids": node_ids,
        "lons": lons,
        "lats": lats,
        "adjacency": dict(adjacency),
    }


def build_grid_index(lons: np.ndarray, lats: np.ndarray, cell_size_deg: float) -> Dict:
    min_lon = float(lons.min())
    min_lat = float(lats.min())
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for idx, (lon, lat) in enumerate(zip(lons, lats)):
        gx = int((lon - min_lon) / cell_size_deg)
        gy = int((lat - min_lat) / cell_size_deg)
        grid[(gx, gy)].append(idx)

    return {
        "grid": dict(grid),
        "min_lon": min_lon,
        "min_lat": min_lat,
        "cell_size_deg": cell_size_deg,
    }


def prepare_graph_runtime(graph_data: Dict, cell_size_deg: float) -> Dict:
    node_ids = np.asarray(graph_data["node_ids"], dtype=np.int64)
    lons = np.asarray(graph_data["lons"], dtype=np.float64)
    lats = np.asarray(graph_data["lats"], dtype=np.float64)

    if len(node_ids) == 0:
        raise ValueError("Empty graph nodes.")

    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}
    grid_info = build_grid_index(lons, lats, cell_size_deg=cell_size_deg)

    # Backward compatibility:
    # - very old cache stores (dst, dist)
    # - old cache stores (dst, dist, bearing)
    # - new cache stores (dst, dist, bearing, road_penalty)
    adjacency_norm: Dict[int, List[Tuple[int, float, float, float]]] = defaultdict(list)
    out_bearings: Dict[int, List[float]] = defaultdict(list)
    has_edge_class_penalty = False
    for src, nbrs in graph_data["adjacency"].items():
        src_idx = id_to_idx.get(int(src))
        if src_idx is None:
            continue
        src_lon = lons[src_idx]
        src_lat = lats[src_idx]

        for item in nbrs:
            if len(item) == 4:
                dst, dist, br, cls_pen = item
                br_val = float(br)
                cls_pen_val = max(1.0, float(cls_pen))
                has_edge_class_penalty = True
            elif len(item) == 3:
                dst, dist, br = item
                br_val = float(br)
                cls_pen_val = 1.0
            elif len(item) == 2:
                dst, dist = item
                dst_idx = id_to_idx.get(int(dst))
                if dst_idx is None:
                    continue
                br_val = bearing_deg(src_lon, src_lat, lons[dst_idx], lats[dst_idx])
                cls_pen_val = 1.0
            else:
                continue

            adjacency_norm[int(src)].append((int(dst), float(dist), float(br_val), float(cls_pen_val)))
            if np.isfinite(br_val):
                out_bearings[int(src)].append(float(br_val))

    return {
        "bounds": graph_data.get("bounds"),
        "node_ids": node_ids,
        "lons": lons,
        "lats": lats,
        "adjacency": dict(adjacency_norm),
        "id_to_idx": id_to_idx,
        "grid": grid_info["grid"],
        "grid_min_lon": grid_info["min_lon"],
        "grid_min_lat": grid_info["min_lat"],
        "cell_size_deg": grid_info["cell_size_deg"],
        "out_bearings": dict(out_bearings),
        "has_edge_class_penalty": has_edge_class_penalty,
    }


def _graph_data_has_edge_class_penalty(graph_data: Dict) -> bool:
    adj = graph_data.get("adjacency", {})
    for nbrs in adj.values():
        for item in nbrs:
            return len(item) >= 4
    return False


def load_or_build_graph(osm_path: Path, cache_path: Optional[Path], cell_size_deg: float, force_rebuild: bool) -> Dict:
    graph_data = None

    if cache_path is not None and cache_path.exists() and not force_rebuild:
        print(f"[map] load cache: {cache_path}")
        with cache_path.open("rb") as f:
            graph_data = pickle.load(f)
        if not _graph_data_has_edge_class_penalty(graph_data):
            print("[map] cache format lacks road class penalties, rebuilding graph cache.")
            graph_data = None

    if graph_data is None:
        graph_data = build_road_graph_from_osm(osm_path)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[map] cache saved: {cache_path}")

    return prepare_graph_runtime(graph_data, cell_size_deg=cell_size_deg)


def gather_candidate_indices(lon: float, lat: float, graph: Dict, radius_m: float) -> np.ndarray:
    cell_size = graph["cell_size_deg"]
    min_lon = graph["grid_min_lon"]
    min_lat = graph["grid_min_lat"]

    gx = int((lon - min_lon) / cell_size)
    gy = int((lat - min_lat) / cell_size)

    deg_lat = radius_m / 111320.0
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    deg_lon = radius_m / (111320.0 * cos_lat)
    ring = int(math.ceil(max(deg_lat, deg_lon) / cell_size))

    candidate_idxs: List[int] = []
    grid = graph["grid"]
    for dx in range(-ring, ring + 1):
        for dy in range(-ring, ring + 1):
            key = (gx + dx, gy + dy)
            if key in grid:
                candidate_idxs.extend(grid[key])

    if not candidate_idxs:
        return np.empty(0, dtype=np.int64)
    return np.asarray(candidate_idxs, dtype=np.int64)


def nearest_k_candidates(
    lon: float,
    lat: float,
    heading: float,
    graph: Dict,
    radius_m: float,
    k: int,
    sigma_d: float,
    heading_weight: float,
) -> List[Dict]:
    cand_idx = gather_candidate_indices(lon, lat, graph, radius_m=radius_m)
    if cand_idx.size == 0:
        return []

    cand_lons = graph["lons"][cand_idx]
    cand_lats = graph["lats"][cand_idx]
    dists = haversine_meters(cand_lons, cand_lats, lon, lat)

    valid = dists <= radius_m
    if not np.any(valid):
        return []

    cand_idx = cand_idx[valid]
    dists = dists[valid]

    rows = []
    has_heading = np.isfinite(heading)
    for idx, snap_d in zip(cand_idx, dists):
        node_id = int(graph["node_ids"][idx])

        if has_heading:
            bearings = graph["out_bearings"].get(node_id, [])
            if bearings:
                hd = min(angle_diff_deg(heading, b) for b in bearings)
            else:
                hd = 180.0
        else:
            hd = 0.0

        emit = float(snap_d / max(1e-6, sigma_d) + heading_weight * (hd / 180.0))
        rows.append(
            {
                "node_id": node_id,
                "snap_dist": float(snap_d),
                "heading_diff": float(hd),
                "emit": emit,
            }
        )

    rows.sort(key=lambda x: x["emit"])
    return rows[:k]


def nearest_node_in_radius(lon: float, lat: float, graph: Dict, radius_m: float) -> Tuple[Optional[int], float]:
    cand_idx = gather_candidate_indices(lon, lat, graph, radius_m=radius_m)
    if cand_idx.size == 0:
        return None, float("inf")

    cand_lons = graph["lons"][cand_idx]
    cand_lats = graph["lats"][cand_idx]
    dists = haversine_meters(cand_lons, cand_lats, lon, lat)
    best_i = int(np.argmin(dists))
    best_d = float(dists[best_i])
    if best_d > radius_m:
        return None, best_d

    node_id = int(graph["node_ids"][cand_idx[best_i]])
    return node_id, best_d


def snap_points_to_road_nodes(points: np.ndarray, graph: Dict, max_dist_m: float) -> np.ndarray:
    if max_dist_m <= 0.0 or points.size == 0:
        return points

    snapped = points.copy()
    for i in range(points.shape[0]):
        lon = float(points[i, 0])
        lat = float(points[i, 1])
        node_id, _ = nearest_node_in_radius(lon=lon, lat=lat, graph=graph, radius_m=max_dist_m)
        if node_id is None:
            continue
        node_idx = graph["id_to_idx"].get(node_id)
        if node_idx is None:
            continue
        snapped[i, 0] = float(graph["lons"][node_idx])
        snapped[i, 1] = float(graph["lats"][node_idx])
    return snapped


def _edge_cost_with_preference(edge_dist: float, road_penalty: float, road_class_weight: float) -> float:
    class_weight = max(0.0, float(road_class_weight))
    class_penalty = max(1.0, float(road_penalty))
    return float(edge_dist) * (1.0 + class_weight * (class_penalty - 1.0))


def _turn_penalty_cost(prev_bearing: float, next_bearing: float, turn_penalty_m: float, turn_angle_threshold_deg: float) -> float:
    if turn_penalty_m <= 0.0 or (not np.isfinite(prev_bearing)) or (not np.isfinite(next_bearing)):
        return 0.0

    thr = max(0.0, min(179.0, float(turn_angle_threshold_deg)))
    turn_angle = angle_diff_deg(prev_bearing, next_bearing)
    if turn_angle <= thr:
        return 0.0

    norm = (turn_angle - thr) / max(1e-6, 180.0 - thr)
    return float(turn_penalty_m) * norm


def _reconstruct_path_from_state_chain(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    end_state: Tuple[int, int],
) -> List[int]:
    states = [end_state]
    cur = end_state
    while cur in came_from:
        cur = came_from[cur]
        states.append(cur)
    states.reverse()
    return [state[0] for state in states]


def _astar_route(
    start_node: int,
    end_node: int,
    graph: Dict,
    max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
    return_path: bool,
) -> Tuple[Optional[List[int]], float]:
    if start_node == end_node:
        return [start_node] if return_path else None, 0.0

    adjacency = graph["adjacency"]
    if start_node not in adjacency:
        return None, float("inf")

    id_to_idx = graph["id_to_idx"]
    goal_idx = id_to_idx.get(end_node)
    if goal_idx is None:
        return None, float("inf")

    goal_lon = graph["lons"][goal_idx]
    goal_lat = graph["lats"][goal_idx]

    def heuristic(node_id: int) -> float:
        idx = id_to_idx.get(node_id)
        if idx is None:
            return float("inf")
        lon = graph["lons"][idx]
        lat = graph["lats"][idx]
        return float(haversine_meters(lon, lat, goal_lon, goal_lat))

    start_state = (int(start_node), -1)
    g_pref: Dict[Tuple[int, int], float] = {start_state: 0.0}
    g_geo: Dict[Tuple[int, int], float] = {start_state: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    open_heap: List[Tuple[float, float, int, int, float]] = []
    heapq.heappush(open_heap, (heuristic(start_node), 0.0, int(start_node), -1, float("nan")))

    expansions = 0
    while open_heap:
        _, pref_cur, cur, prev, in_bearing = heapq.heappop(open_heap)
        state = (cur, prev)
        if pref_cur > g_pref.get(state, float("inf")) + 1e-9:
            continue

        if cur == end_node:
            if return_path:
                return _reconstruct_path_from_state_chain(came_from, state), float(g_geo[state])
            return None, float(g_geo[state])

        expansions += 1
        if expansions > max_expansions:
            return None, float("inf")

        for nxt, edge_dist, edge_bearing, road_penalty in adjacency.get(cur, []):
            pref_next = (
                pref_cur
                + _edge_cost_with_preference(
                    edge_dist=edge_dist,
                    road_penalty=road_penalty,
                    road_class_weight=astar_road_class_weight,
                )
                + _turn_penalty_cost(
                    prev_bearing=in_bearing,
                    next_bearing=edge_bearing,
                    turn_penalty_m=astar_turn_penalty_m,
                    turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
                )
            )

            next_state = (int(nxt), cur)
            if pref_next + 1e-9 < g_pref.get(next_state, float("inf")):
                g_pref[next_state] = pref_next
                g_geo[next_state] = g_geo[state] + float(edge_dist)
                came_from[next_state] = state
                heapq.heappush(
                    open_heap,
                    (
                        pref_next + heuristic(int(nxt)),
                        pref_next,
                        int(nxt),
                        cur,
                        float(edge_bearing),
                    ),
                )

    return None, float("inf")


def astar_shortest_distance(
    start_node: int,
    end_node: int,
    graph: Dict,
    max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
) -> float:
    _, route_geo = _astar_route(
        start_node=start_node,
        end_node=end_node,
        graph=graph,
        max_expansions=max_expansions,
        astar_road_class_weight=astar_road_class_weight,
        astar_turn_penalty_m=astar_turn_penalty_m,
        astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
        return_path=False,
    )
    return route_geo


def astar_shortest_path(
    start_node: int,
    end_node: int,
    graph: Dict,
    max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
) -> Optional[List[int]]:
    path, _ = _astar_route(
        start_node=start_node,
        end_node=end_node,
        graph=graph,
        max_expansions=max_expansions,
        astar_road_class_weight=astar_road_class_weight,
        astar_turn_penalty_m=astar_turn_penalty_m,
        astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
        return_path=True,
    )
    return path


def sample_polyline_by_ratios(polyline: np.ndarray, ratios: np.ndarray) -> np.ndarray:
    if polyline.shape[0] < 2:
        return np.repeat(polyline[0:1], repeats=len(ratios), axis=0)

    seg_lens = haversine_meters(
        polyline[:-1, 0],
        polyline[:-1, 1],
        polyline[1:, 0],
        polyline[1:, 1],
    ).astype(np.float64)

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(polyline[0:1], repeats=len(ratios), axis=0)

    t = np.clip(ratios, 0.0, 1.0) * total
    out_lon = np.interp(t, cum, polyline[:, 0])
    out_lat = np.interp(t, cum, polyline[:, 1])
    return np.stack([out_lon, out_lat], axis=1)


def sample_polyline_speed_aware_by_ratios(
    polyline: np.ndarray,
    ratios: np.ndarray,
    turn_slow_angle_deg: float,
    min_speed_factor: float,
) -> np.ndarray:
    if polyline.shape[0] < 2:
        return np.repeat(polyline[0:1], repeats=len(ratios), axis=0)

    seg_lens = haversine_meters(
        polyline[:-1, 0],
        polyline[:-1, 1],
        polyline[1:, 0],
        polyline[1:, 1],
    ).astype(np.float64)

    if seg_lens.size == 0:
        return np.repeat(polyline[0:1], repeats=len(ratios), axis=0)

    total_dist = float(np.sum(seg_lens))
    if total_dist < 1e-6:
        return np.repeat(polyline[0:1], repeats=len(ratios), axis=0)

    n_nodes = polyline.shape[0]
    turn_at_node = np.full(n_nodes, np.nan, dtype=np.float64)

    if n_nodes >= 3:
        for i in range(1, n_nodes - 1):
            b_in = bearing_deg(
                float(polyline[i - 1, 0]),
                float(polyline[i - 1, 1]),
                float(polyline[i, 0]),
                float(polyline[i, 1]),
            )
            b_out = bearing_deg(
                float(polyline[i, 0]),
                float(polyline[i, 1]),
                float(polyline[i + 1, 0]),
                float(polyline[i + 1, 1]),
            )
            if np.isfinite(b_in) and np.isfinite(b_out):
                turn_at_node[i] = angle_diff_deg(b_out, b_in)

    seg_turn = np.zeros(seg_lens.size, dtype=np.float64)
    for i in range(seg_lens.size):
        t0 = float(turn_at_node[i]) if np.isfinite(turn_at_node[i]) else 0.0
        t1 = float(turn_at_node[i + 1]) if np.isfinite(turn_at_node[i + 1]) else 0.0
        seg_turn[i] = max(t0, t1)

    thr = float(np.clip(turn_slow_angle_deg, 0.0, 179.0))
    min_sf = float(np.clip(min_speed_factor, 0.05, 1.0))

    if thr >= 179.0:
        speed_factor = np.ones_like(seg_turn, dtype=np.float64)
    else:
        norm = np.clip((seg_turn - thr) / max(1e-6, 180.0 - thr), 0.0, 1.0)
        speed_factor = 1.0 - norm * (1.0 - min_sf)
        speed_factor = np.clip(speed_factor, min_sf, 1.0)

    seg_time = seg_lens / np.maximum(speed_factor, 1e-6)
    cum_time = np.concatenate([[0.0], np.cumsum(seg_time)])
    total_time = float(cum_time[-1])
    if total_time < 1e-6:
        return sample_polyline_by_ratios(polyline=polyline, ratios=ratios)

    t = np.clip(ratios, 0.0, 1.0) * total_time
    out_lon = np.interp(t, cum_time, polyline[:, 0])
    out_lat = np.interp(t, cum_time, polyline[:, 1])
    return np.stack([out_lon, out_lat], axis=1)


def infer_anchor_heading(anchor_pos: int, known_idx: np.ndarray, coords: np.ndarray) -> float:
    cur_idx = int(known_idx[anchor_pos])
    if anchor_pos == 0:
        if len(known_idx) == 1:
            return float("nan")
        nxt_idx = int(known_idx[anchor_pos + 1])
        return bearing_deg(float(coords[cur_idx, 0]), float(coords[cur_idx, 1]), float(coords[nxt_idx, 0]), float(coords[nxt_idx, 1]))
    if anchor_pos == len(known_idx) - 1:
        prev_idx = int(known_idx[anchor_pos - 1])
        return bearing_deg(float(coords[prev_idx, 0]), float(coords[prev_idx, 1]), float(coords[cur_idx, 0]), float(coords[cur_idx, 1]))

    prev_idx = int(known_idx[anchor_pos - 1])
    nxt_idx = int(known_idx[anchor_pos + 1])
    return bearing_deg(float(coords[prev_idx, 0]), float(coords[prev_idx, 1]), float(coords[nxt_idx, 0]), float(coords[nxt_idx, 1]))


def get_route_dist(
    u: int,
    v: int,
    graph: Dict,
    dist_cache: Dict[Tuple[int, int], float],
    astar_max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
) -> float:
    key = (u, v)
    if key in dist_cache:
        return dist_cache[key]

    d = astar_shortest_distance(
        start_node=u,
        end_node=v,
        graph=graph,
        max_expansions=astar_max_expansions,
        astar_road_class_weight=astar_road_class_weight,
        astar_turn_penalty_m=astar_turn_penalty_m,
        astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
    )
    dist_cache[key] = d
    return d


def get_route_path(
    u: int,
    v: int,
    graph: Dict,
    path_cache: Dict[Tuple[int, int], Optional[List[int]]],
    astar_max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
) -> Optional[List[int]]:
    key = (u, v)
    if key in path_cache:
        return path_cache[key]

    p = astar_shortest_path(
        start_node=u,
        end_node=v,
        graph=graph,
        max_expansions=astar_max_expansions,
        astar_road_class_weight=astar_road_class_weight,
        astar_turn_penalty_m=astar_turn_penalty_m,
        astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
    )
    path_cache[key] = p
    return p


def viterbi_decode_candidates(
    anchor_coords: np.ndarray,
    anchor_times: np.ndarray,
    anchor_candidates: List[List[Dict]],
    graph: Dict,
    dist_cache: Dict[Tuple[int, int], float],
    astar_max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
    trans_weight_detour: float,
    trans_weight_speed: float,
    trans_weight_turn: float,
    no_path_penalty: float,
) -> Optional[List[int]]:
    n = len(anchor_candidates)
    if n == 0:
        return None
    for cands in anchor_candidates:
        if len(cands) == 0:
            return None

    dp_prev = np.array([c["emit"] for c in anchor_candidates[0]], dtype=np.float64)
    backptr: List[np.ndarray] = []

    for i in range(1, n):
        cands_prev = anchor_candidates[i - 1]
        cands_cur = anchor_candidates[i]

        prev_lon, prev_lat = anchor_coords[i - 1]
        cur_lon, cur_lat = anchor_coords[i]
        straight = float(haversine_meters(prev_lon, prev_lat, cur_lon, cur_lat))
        dt = float(max(1, int(anchor_times[i] - anchor_times[i - 1])))
        v_ref = max(straight / dt, 0.1)
        anchor_heading = bearing_deg(float(prev_lon), float(prev_lat), float(cur_lon), float(cur_lat))

        dp_cur = np.full(len(cands_cur), np.inf, dtype=np.float64)
        bp_cur = np.full(len(cands_cur), -1, dtype=np.int64)

        for j, cj in enumerate(cands_cur):
            best_val = np.inf
            best_k = -1
            node_j = int(cj["node_id"])

            for k, ck in enumerate(cands_prev):
                node_k = int(ck["node_id"])
                route_dist = get_route_dist(
                    u=node_k,
                    v=node_j,
                    graph=graph,
                    dist_cache=dist_cache,
                    astar_max_expansions=astar_max_expansions,
                    astar_road_class_weight=astar_road_class_weight,
                    astar_turn_penalty_m=astar_turn_penalty_m,
                    astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
                )

                if not np.isfinite(route_dist):
                    trans = no_path_penalty
                else:
                    route_dist = max(route_dist, straight)
                    detour_ratio = route_dist / max(straight, 1e-6)
                    v_route = route_dist / dt
                    speed_dev = abs(v_route - v_ref) / max(v_ref, 0.5)

                    turn_dev = 0.0
                    if np.isfinite(anchor_heading) and trans_weight_turn > 0.0:
                        idx_k = graph["id_to_idx"].get(node_k)
                        idx_j = graph["id_to_idx"].get(node_j)
                        if idx_k is not None and idx_j is not None:
                            route_heading = bearing_deg(
                                float(graph["lons"][idx_k]),
                                float(graph["lats"][idx_k]),
                                float(graph["lons"][idx_j]),
                                float(graph["lats"][idx_j]),
                            )
                            if np.isfinite(route_heading):
                                turn_dev = angle_diff_deg(route_heading, anchor_heading) / 180.0

                    trans = (
                        trans_weight_detour * (detour_ratio - 1.0)
                        + trans_weight_speed * speed_dev
                        + trans_weight_turn * turn_dev
                    )

                val = dp_prev[k] + cj["emit"] + trans
                if val < best_val:
                    best_val = val
                    best_k = k

            dp_cur[j] = best_val
            bp_cur[j] = best_k

        backptr.append(bp_cur)
        dp_prev = dp_cur

    last = int(np.argmin(dp_prev))
    states = [last]
    for i in range(n - 2, -1, -1):
        bp = backptr[i]
        prev = int(bp[states[-1]])
        if prev < 0:
            return None
        states.append(prev)
    states.reverse()
    return states


def _gap_position_for_fusion(
    gap: int,
    gap_alpha_small_threshold: int,
    gap_alpha_large_threshold: int,
) -> float:
    if gap_alpha_small_threshold >= 0 and gap_alpha_large_threshold >= 0 and gap_alpha_large_threshold > gap_alpha_small_threshold:
        v = (float(gap) - float(gap_alpha_small_threshold)) / float(gap_alpha_large_threshold - gap_alpha_small_threshold)
        return float(np.clip(v, 0.0, 1.0))

    if gap_alpha_large_threshold > 0:
        return float(np.clip(float(gap) / float(gap_alpha_large_threshold), 0.0, 1.0))

    if gap_alpha_small_threshold >= 0:
        return 0.0 if gap <= gap_alpha_small_threshold else 1.0

    return 0.5


def _resolve_fusion_controls(
    conf: float,
    gap: int,
    alpha_lo: float,
    alpha_hi: float,
    pure_thr: float,
    resample_snap_radius_m: float,
    fusion_mode: str,
    fusion_conf_power: float,
    fusion_gap_bias: float,
    fusion_snap_conf_threshold: float,
    gap_alpha_small_threshold: int,
    gap_alpha_large_threshold: int,
) -> Tuple[float, float, float]:
    conf_for_rules = float(np.clip(conf, 0.0, 1.0))

    if fusion_mode == "continuous":
        gap_pos = _gap_position_for_fusion(
            gap=gap,
            gap_alpha_small_threshold=gap_alpha_small_threshold,
            gap_alpha_large_threshold=gap_alpha_large_threshold,
        )
        gap_centered = 2.0 * gap_pos - 1.0
        conf_for_rules = float(np.clip(conf_for_rules + float(fusion_gap_bias) * gap_centered, 0.0, 1.0))

        conf_curve = conf_for_rules ** max(1e-6, float(fusion_conf_power))
        alpha = float(np.clip(alpha_lo + (alpha_hi - alpha_lo) * conf_curve, alpha_lo, alpha_hi))
        if conf_for_rules >= pure_thr:
            alpha = 1.0

        if resample_snap_radius_m > 0.0 and fusion_snap_conf_threshold > 0.0:
            snap_scale = float(
                np.clip(
                    (float(fusion_snap_conf_threshold) - conf_for_rules) / max(1e-6, float(fusion_snap_conf_threshold)),
                    0.0,
                    1.0,
                )
            )
            snap_radius_eff = float(resample_snap_radius_m) * snap_scale
        else:
            snap_radius_eff = float(resample_snap_radius_m)

        return alpha, snap_radius_eff, conf_for_rules

    # Legacy policy: keep existing behavior for alpha and fixed snap radius.
    if conf_for_rules >= pure_thr:
        alpha = 1.0
    else:
        alpha = float(np.clip(alpha_lo + (alpha_hi - alpha_lo) * conf_for_rules, alpha_lo, alpha_hi))

    return alpha, float(resample_snap_radius_m), conf_for_rules


def recover_traj_with_hmm_map(
    timestamps: np.ndarray,
    coords: np.ndarray,
    mask: np.ndarray,
    graph: Dict,
    k_candidates: int,
    candidate_radius_m: float,
    candidate_radius_fallback_m: float,
    emission_sigma_dist: float,
    emission_heading_weight: float,
    min_gap_map: int,
    astar_max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
    trans_weight_detour: float,
    trans_weight_speed: float,
    trans_weight_turn: float,
    no_path_penalty: float,
    max_snap_dist_m: float,
    max_detour_ratio: float,
    min_speed_mps: float,
    max_speed_mps: float,
    resample_snap_radius_m: float,
    alpha_min: float,
    alpha_max: float,
    pure_map_conf_threshold: float,
    gap_alpha_small_threshold: int,
    gap_alpha_large_threshold: int,
    small_gap_alpha_max: float,
    large_gap_alpha_min: float,
    large_gap_pure_map_conf_threshold: float,
    low_conf_threshold: float,
    low_conf_alpha_cap: float,
    confidence_snap_sigma: float,
    confidence_detour_sigma: float,
    confidence_speed_sigma: float,
    interpolate_mode: str,
    low_conf_smooth_threshold: float,
    low_conf_smooth_window: int,
    low_conf_smooth_sigma: float,
    low_conf_smooth_strength: float,
    path_sampling_mode: str,
    sampling_turn_slow_angle_deg: float,
    sampling_min_speed_factor: float,
    fusion_mode: str,
    fusion_conf_power: float,
    fusion_gap_bias: float,
    fusion_snap_conf_threshold: float,
    post_smooth_map_bind_mode: str,
    post_smooth_map_bind_conf_threshold: float,
    post_smooth_map_bind_radius_m: float,
    post_smooth_map_bind_min_gap: int,
    sharp_turn_align_threshold_deg: float,
    sharp_turn_heading_tolerance_deg: float,
    sharp_turn_alpha_min: float,
    sharp_turn_conf_threshold: float,
    high_speed_var_threshold: float,
    high_speed_max_turn_deg: float,
    high_speed_curvature_iters: int,
    high_speed_curvature_strength: float,
    dist_cache: Dict[Tuple[int, int], float],
    path_cache: Dict[Tuple[int, int], Optional[List[int]]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    pred_base = interpolate_traj(timestamps=timestamps, coords=coords, mask=mask, mode=interpolate_mode).astype(np.float64)
    pred = pred_base.copy()

    known_idx = np.where(mask)[0]
    if len(known_idx) < 2:
        return pred.astype(np.float32), {
            "segments_total": 0,
            "segments_try_map": 0,
            "segments_map_success": 0,
            "fallback_no_candidates": 0,
            "fallback_no_path": 0,
            "fallback_low_confidence": 0,
            "fallback_snap_too_far": 0,
            "fallback_detour_too_large": 0,
            "fallback_speed_too_low": 0,
            "fallback_speed_too_high": 0,
        }

    anchor_coords = np.stack([coords[known_idx, 0], coords[known_idx, 1]], axis=1).astype(np.float64)
    anchor_times = timestamps[known_idx].astype(np.int64)

    anchor_candidates: List[List[Dict]] = []
    for ai in range(len(known_idx)):
        heading = infer_anchor_heading(anchor_pos=ai, known_idx=known_idx, coords=coords)
        lon, lat = float(anchor_coords[ai, 0]), float(anchor_coords[ai, 1])

        cands = nearest_k_candidates(
            lon=lon,
            lat=lat,
            heading=heading,
            graph=graph,
            radius_m=candidate_radius_m,
            k=k_candidates,
            sigma_d=emission_sigma_dist,
            heading_weight=emission_heading_weight,
        )

        if not cands and candidate_radius_fallback_m > candidate_radius_m:
            cands = nearest_k_candidates(
                lon=lon,
                lat=lat,
                heading=heading,
                graph=graph,
                radius_m=candidate_radius_fallback_m,
                k=max(1, k_candidates),
                sigma_d=emission_sigma_dist,
                heading_weight=emission_heading_weight,
            )

        anchor_candidates.append(cands)

    state_ids = viterbi_decode_candidates(
        anchor_coords=anchor_coords,
        anchor_times=anchor_times,
        anchor_candidates=anchor_candidates,
        graph=graph,
        dist_cache=dist_cache,
        astar_max_expansions=astar_max_expansions,
        astar_road_class_weight=astar_road_class_weight,
        astar_turn_penalty_m=astar_turn_penalty_m,
        astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
        trans_weight_detour=trans_weight_detour,
        trans_weight_speed=trans_weight_speed,
        trans_weight_turn=trans_weight_turn,
        no_path_penalty=no_path_penalty,
    )

    stats = {
        "segments_total": 0,
        "segments_try_map": 0,
        "segments_map_success": 0,
        "fallback_no_candidates": 0,
        "fallback_no_path": 0,
        "fallback_low_confidence": 0,
        "fallback_snap_too_far": 0,
        "fallback_detour_too_large": 0,
        "fallback_speed_too_low": 0,
        "fallback_speed_too_high": 0,
    }

    if state_ids is None:
        stats["fallback_no_candidates"] += len(known_idx) - 1
        pred[mask] = coords[mask]
        return pred.astype(np.float32), stats

    chosen = [anchor_candidates[i][state_ids[i]] for i in range(len(state_ids))]

    for seg_i, (a, b) in enumerate(zip(known_idx[:-1], known_idx[1:])):
        gap = int(b - a - 1)
        if gap <= 0:
            continue

        stats["segments_total"] += 1
        if gap < min_gap_map:
            continue

        stats["segments_try_map"] += 1

        ca = chosen[seg_i]
        cb = chosen[seg_i + 1]

        node_a = int(ca["node_id"])
        node_b = int(cb["node_id"])

        path_nodes = get_route_path(
            u=node_a,
            v=node_b,
            graph=graph,
            path_cache=path_cache,
            astar_max_expansions=astar_max_expansions,
            astar_road_class_weight=astar_road_class_weight,
            astar_turn_penalty_m=astar_turn_penalty_m,
            astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
        )
        if not path_nodes or len(path_nodes) < 2:
            stats["fallback_no_path"] += 1
            continue

        path_coords = []
        for nid in path_nodes:
            idx = graph["id_to_idx"].get(nid)
            if idx is None:
                continue
            path_coords.append([graph["lons"][idx], graph["lats"][idx]])

        if len(path_coords) < 2:
            stats["fallback_no_path"] += 1
            continue

        lon_a, lat_a = float(coords[a, 0]), float(coords[a, 1])
        lon_b, lat_b = float(coords[b, 0]), float(coords[b, 1])

        idx_a = graph["id_to_idx"].get(node_a)
        idx_b = graph["id_to_idx"].get(node_b)
        snap_lon_a = float(graph["lons"][idx_a]) if idx_a is not None else lon_a
        snap_lat_a = float(graph["lats"][idx_a]) if idx_a is not None else lat_a
        snap_lon_b = float(graph["lons"][idx_b]) if idx_b is not None else lon_b
        snap_lat_b = float(graph["lats"][idx_b]) if idx_b is not None else lat_b

        straight = float(haversine_meters(lon_a, lat_a, lon_b, lat_b))

        route_dist = float(
            np.sum(
                haversine_meters(
                    np.asarray(path_coords[:-1])[:, 0],
                    np.asarray(path_coords[:-1])[:, 1],
                    np.asarray(path_coords[1:])[:, 0],
                    np.asarray(path_coords[1:])[:, 1],
                )
            )
        )
        route_dist = max(route_dist, straight)

        dt = float(max(1, int(timestamps[b] - timestamps[a])))
        speed = route_dist / dt
        detour_ratio = route_dist / max(straight, 1e-6)

        snap_bad = ca["snap_dist"] > max_snap_dist_m or cb["snap_dist"] > max_snap_dist_m
        detour_bad = detour_ratio > max_detour_ratio
        speed_low_bad = speed < min_speed_mps
        speed_high_bad = speed > max_speed_mps
        if snap_bad or detour_bad or speed_low_bad or speed_high_bad:
            stats["fallback_low_confidence"] += 1
            if snap_bad:
                stats["fallback_snap_too_far"] += 1
            elif detour_bad:
                stats["fallback_detour_too_large"] += 1
            elif speed_low_bad:
                stats["fallback_speed_too_low"] += 1
            else:
                stats["fallback_speed_too_high"] += 1
            continue

        polyline = np.vstack(
            [
                np.array([[snap_lon_a, snap_lat_a]], dtype=np.float64),
                np.asarray(path_coords, dtype=np.float64),
                np.array([[snap_lon_b, snap_lat_b]], dtype=np.float64),
            ]
        )
        polyline_turn_max = polyline_max_turn_deg(polyline)

        dt2 = float(timestamps[b] - timestamps[a])
        if dt2 <= 0:
            ratios = (np.arange(a + 1, b) - a) / float(b - a)
        else:
            ratios = (timestamps[a + 1 : b] - timestamps[a]) / dt2

        if path_sampling_mode == "speed-aware":
            filled_map = sample_polyline_speed_aware_by_ratios(
                polyline=polyline,
                ratios=ratios,
                turn_slow_angle_deg=sampling_turn_slow_angle_deg,
                min_speed_factor=sampling_min_speed_factor,
            )
        else:
            filled_map = sample_polyline_by_ratios(polyline=polyline, ratios=ratios)

        snap_avg = 0.5 * (ca["snap_dist"] + cb["snap_dist"])
        speed_ref = max(straight / dt, 0.1)
        speed_dev = abs(speed - speed_ref) / max(speed_ref, 0.5)

        conf_snap = math.exp(-snap_avg / max(1e-6, confidence_snap_sigma))
        conf_detour = math.exp(-max(0.0, detour_ratio - 1.0) / max(1e-6, confidence_detour_sigma))
        conf_speed = math.exp(-speed_dev / max(1e-6, confidence_speed_sigma))
        conf = conf_snap * conf_detour * conf_speed

        alpha_lo = float(alpha_min)
        alpha_hi = float(alpha_max)
        pure_thr = float(pure_map_conf_threshold)

        if gap_alpha_small_threshold >= 0 and gap <= gap_alpha_small_threshold:
            alpha_hi = min(alpha_hi, small_gap_alpha_max)
            # Short gaps are usually stable with interpolation; avoid forced map lock.
            pure_thr = max(pure_thr, 1.1)

        if gap_alpha_large_threshold >= 0 and gap >= gap_alpha_large_threshold:
            alpha_lo = max(alpha_lo, large_gap_alpha_min)
            pure_thr = min(pure_thr, large_gap_pure_map_conf_threshold)

        alpha_lo = float(np.clip(alpha_lo, 0.0, 1.0))
        alpha_hi = float(np.clip(alpha_hi, 0.0, 1.0))
        if alpha_hi < alpha_lo:
            alpha_hi = alpha_lo

        alpha, snap_radius_eff, conf_for_rules = _resolve_fusion_controls(
            conf=conf,
            gap=gap,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            pure_thr=pure_thr,
            resample_snap_radius_m=resample_snap_radius_m,
            fusion_mode=fusion_mode,
            fusion_conf_power=fusion_conf_power,
            fusion_gap_bias=fusion_gap_bias,
            fusion_snap_conf_threshold=fusion_snap_conf_threshold,
            gap_alpha_small_threshold=gap_alpha_small_threshold,
            gap_alpha_large_threshold=gap_alpha_large_threshold,
        )

        if low_conf_threshold >= 0.0 and conf_for_rules < low_conf_threshold:
            alpha = min(alpha, float(np.clip(low_conf_alpha_cap, 0.0, 1.0)))

        if (
            sharp_turn_align_threshold_deg >= 0.0
            and polyline_turn_max >= sharp_turn_align_threshold_deg
            and conf_for_rules >= sharp_turn_conf_threshold
            and polyline.shape[0] >= 2
        ):
            obs_heading_a = infer_anchor_heading(anchor_pos=seg_i, known_idx=known_idx, coords=coords)
            obs_heading_b = infer_anchor_heading(anchor_pos=seg_i + 1, known_idx=known_idx, coords=coords)
            route_heading_a = bearing_deg(
                float(polyline[0, 0]),
                float(polyline[0, 1]),
                float(polyline[1, 0]),
                float(polyline[1, 1]),
            )
            route_heading_b = bearing_deg(
                float(polyline[-2, 0]),
                float(polyline[-2, 1]),
                float(polyline[-1, 0]),
                float(polyline[-1, 1]),
            )

            heading_tol = float(np.clip(sharp_turn_heading_tolerance_deg, 0.0, 180.0))
            start_ok = (
                not np.isfinite(obs_heading_a)
                or not np.isfinite(route_heading_a)
                or angle_diff_deg(obs_heading_a, route_heading_a) <= heading_tol
            )
            end_ok = (
                not np.isfinite(obs_heading_b)
                or not np.isfinite(route_heading_b)
                or angle_diff_deg(obs_heading_b, route_heading_b) <= heading_tol
            )

            if start_ok and end_ok:
                alpha = max(alpha, float(np.clip(sharp_turn_alpha_min, 0.0, 1.0)))

        # Keep sampled points close to drivable graph nodes to reduce wall-crossing artifacts.
        if snap_radius_eff > 0.0:
            filled_map = snap_points_to_road_nodes(
                points=filled_map,
                graph=graph,
                max_dist_m=snap_radius_eff,
            )

        base_seg = pred_base[a + 1 : b].astype(np.float64)
        filled = alpha * filled_map + (1.0 - alpha) * base_seg

        if (
            low_conf_smooth_threshold >= 0.0
            and conf_for_rules < low_conf_smooth_threshold
            and low_conf_smooth_window > 0
            and filled.shape[0] >= 3
        ):
            smoothed = smooth_segment_points(
                points=filled,
                window=low_conf_smooth_window,
                sigma=low_conf_smooth_sigma,
            ).astype(np.float64)
            smooth_w = float(np.clip(low_conf_smooth_strength, 0.0, 1.0))
            filled = (1.0 - smooth_w) * filled + smooth_w * smoothed

        if (
            high_speed_var_threshold >= 0.0
            and speed_dev >= high_speed_var_threshold
            and high_speed_max_turn_deg >= 0.0
            and filled.shape[0] >= 3
        ):
            limited = limit_polyline_turn_deg(
                points=filled,
                max_turn_deg=high_speed_max_turn_deg,
                num_iters=high_speed_curvature_iters,
            ).astype(np.float64)
            curv_w = float(np.clip(high_speed_curvature_strength, 0.0, 1.0))
            filled = (1.0 - curv_w) * filled + curv_w * limited

        if post_smooth_map_bind_mode != "off" and gap >= max(0, int(post_smooth_map_bind_min_gap)) and filled.shape[0] > 0:
            bind_allowed = post_smooth_map_bind_mode == "all"
            if post_smooth_map_bind_mode == "high-conf":
                bind_allowed = conf_for_rules >= post_smooth_map_bind_conf_threshold
            elif post_smooth_map_bind_mode == "low-conf":
                bind_allowed = conf_for_rules < post_smooth_map_bind_conf_threshold

            if bind_allowed:
                if post_smooth_map_bind_radius_m > 0.0:
                    bind_radius = float(post_smooth_map_bind_radius_m)
                else:
                    bind_radius = float(max(snap_radius_eff, resample_snap_radius_m))

                if bind_radius > 0.0:
                    filled = snap_points_to_road_nodes(
                        points=filled,
                        graph=graph,
                        max_dist_m=bind_radius,
                    )

        pred[a + 1 : b] = filled
        stats["segments_map_success"] += 1

    pred[mask] = coords[mask]
    return pred.astype(np.float32), stats


def evaluate_missing_only(pred_records: Iterable[Dict], input_records: Iterable[Dict], gt_records: Iterable[Dict]) -> Dict[str, float]:
    pred_by_id = {item["traj_id"]: np.asarray(item["coords"], dtype=np.float64) for item in pred_records}
    input_by_id = {item["traj_id"]: item for item in input_records}
    gt_by_id = {item["traj_id"]: np.asarray(item["coords"], dtype=np.float64) for item in gt_records}

    dists = []
    for traj_id, pred_coords in pred_by_id.items():
        input_item = input_by_id[traj_id]
        mask = np.asarray(input_item["mask"], dtype=bool)
        missing = ~mask
        if not np.any(missing):
            continue

        gt_coords = gt_by_id[traj_id]
        dist = haversine_meters(
            lon1=pred_coords[missing, 0],
            lat1=pred_coords[missing, 1],
            lon2=gt_coords[missing, 0],
            lat2=gt_coords[missing, 1],
        )
        dists.append(dist)

    if not dists:
        return {"mae": math.nan, "rmse": math.nan, "count": 0}

    all_dists = np.concatenate(dists)
    return {
        "mae": float(np.mean(np.abs(all_dists))),
        "rmse": float(np.sqrt(np.mean(all_dists**2))),
        "count": int(all_dists.size),
    }


def _empty_agg_stats() -> Dict[str, int]:
    return {
        "segments_total": 0,
        "segments_try_map": 0,
        "segments_map_success": 0,
        "fallback_no_candidates": 0,
        "fallback_no_path": 0,
        "fallback_low_confidence": 0,
        "fallback_snap_too_far": 0,
        "fallback_detour_too_large": 0,
        "fallback_speed_too_low": 0,
        "fallback_speed_too_high": 0,
    }


def _merge_agg_stats(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for k in dst:
        dst[k] += int(src.get(k, 0))


def _split_chunks(records: List[Dict], chunk_size: int) -> List[List[Dict]]:
    if chunk_size <= 0:
        chunk_size = max(1, len(records))
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def _build_predict_kwargs(
    k_candidates: int,
    candidate_radius_m: float,
    candidate_radius_fallback_m: float,
    emission_sigma_dist: float,
    emission_heading_weight: float,
    min_gap_map: int,
    astar_max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
    trans_weight_detour: float,
    trans_weight_speed: float,
    trans_weight_turn: float,
    no_path_penalty: float,
    max_snap_dist_m: float,
    max_detour_ratio: float,
    min_speed_mps: float,
    max_speed_mps: float,
    resample_snap_radius_m: float,
    alpha_min: float,
    alpha_max: float,
    pure_map_conf_threshold: float,
    gap_alpha_small_threshold: int,
    gap_alpha_large_threshold: int,
    small_gap_alpha_max: float,
    large_gap_alpha_min: float,
    large_gap_pure_map_conf_threshold: float,
    low_conf_threshold: float,
    low_conf_alpha_cap: float,
    confidence_snap_sigma: float,
    confidence_detour_sigma: float,
    confidence_speed_sigma: float,
    interpolate_mode: str,
    low_conf_smooth_threshold: float,
    low_conf_smooth_window: int,
    low_conf_smooth_sigma: float,
    low_conf_smooth_strength: float,
    path_sampling_mode: str,
    sampling_turn_slow_angle_deg: float,
    sampling_min_speed_factor: float,
    fusion_mode: str,
    fusion_conf_power: float,
    fusion_gap_bias: float,
    fusion_snap_conf_threshold: float,
    post_smooth_map_bind_mode: str,
    post_smooth_map_bind_conf_threshold: float,
    post_smooth_map_bind_radius_m: float,
    post_smooth_map_bind_min_gap: int,
    sharp_turn_align_threshold_deg: float,
    sharp_turn_heading_tolerance_deg: float,
    sharp_turn_alpha_min: float,
    sharp_turn_conf_threshold: float,
    high_speed_var_threshold: float,
    high_speed_max_turn_deg: float,
    high_speed_curvature_iters: int,
    high_speed_curvature_strength: float,
) -> Dict:
    return {
        "k_candidates": k_candidates,
        "candidate_radius_m": candidate_radius_m,
        "candidate_radius_fallback_m": candidate_radius_fallback_m,
        "emission_sigma_dist": emission_sigma_dist,
        "emission_heading_weight": emission_heading_weight,
        "min_gap_map": min_gap_map,
        "astar_max_expansions": astar_max_expansions,
        "astar_road_class_weight": astar_road_class_weight,
        "astar_turn_penalty_m": astar_turn_penalty_m,
        "astar_turn_angle_threshold_deg": astar_turn_angle_threshold_deg,
        "trans_weight_detour": trans_weight_detour,
        "trans_weight_speed": trans_weight_speed,
        "trans_weight_turn": trans_weight_turn,
        "no_path_penalty": no_path_penalty,
        "max_snap_dist_m": max_snap_dist_m,
        "max_detour_ratio": max_detour_ratio,
        "min_speed_mps": min_speed_mps,
        "max_speed_mps": max_speed_mps,
        "resample_snap_radius_m": resample_snap_radius_m,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "pure_map_conf_threshold": pure_map_conf_threshold,
        "gap_alpha_small_threshold": gap_alpha_small_threshold,
        "gap_alpha_large_threshold": gap_alpha_large_threshold,
        "small_gap_alpha_max": small_gap_alpha_max,
        "large_gap_alpha_min": large_gap_alpha_min,
        "large_gap_pure_map_conf_threshold": large_gap_pure_map_conf_threshold,
        "low_conf_threshold": low_conf_threshold,
        "low_conf_alpha_cap": low_conf_alpha_cap,
        "confidence_snap_sigma": confidence_snap_sigma,
        "confidence_detour_sigma": confidence_detour_sigma,
        "confidence_speed_sigma": confidence_speed_sigma,
        "interpolate_mode": interpolate_mode,
        "low_conf_smooth_threshold": low_conf_smooth_threshold,
        "low_conf_smooth_window": low_conf_smooth_window,
        "low_conf_smooth_sigma": low_conf_smooth_sigma,
        "low_conf_smooth_strength": low_conf_smooth_strength,
        "path_sampling_mode": path_sampling_mode,
        "sampling_turn_slow_angle_deg": sampling_turn_slow_angle_deg,
        "sampling_min_speed_factor": sampling_min_speed_factor,
        "fusion_mode": fusion_mode,
        "fusion_conf_power": fusion_conf_power,
        "fusion_gap_bias": fusion_gap_bias,
        "fusion_snap_conf_threshold": fusion_snap_conf_threshold,
        "post_smooth_map_bind_mode": post_smooth_map_bind_mode,
        "post_smooth_map_bind_conf_threshold": post_smooth_map_bind_conf_threshold,
        "post_smooth_map_bind_radius_m": post_smooth_map_bind_radius_m,
        "post_smooth_map_bind_min_gap": post_smooth_map_bind_min_gap,
        "sharp_turn_align_threshold_deg": sharp_turn_align_threshold_deg,
        "sharp_turn_heading_tolerance_deg": sharp_turn_heading_tolerance_deg,
        "sharp_turn_alpha_min": sharp_turn_alpha_min,
        "sharp_turn_conf_threshold": sharp_turn_conf_threshold,
        "high_speed_var_threshold": high_speed_var_threshold,
        "high_speed_max_turn_deg": high_speed_max_turn_deg,
        "high_speed_curvature_iters": high_speed_curvature_iters,
        "high_speed_curvature_strength": high_speed_curvature_strength,
    }


def build_predictions_serial(
    records: Iterable[Dict],
    graph: Dict,
    k_candidates: int,
    candidate_radius_m: float,
    candidate_radius_fallback_m: float,
    emission_sigma_dist: float,
    emission_heading_weight: float,
    min_gap_map: int,
    astar_max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
    trans_weight_detour: float,
    trans_weight_speed: float,
    trans_weight_turn: float,
    no_path_penalty: float,
    max_snap_dist_m: float,
    max_detour_ratio: float,
    min_speed_mps: float,
    max_speed_mps: float,
    resample_snap_radius_m: float,
    alpha_min: float,
    alpha_max: float,
    pure_map_conf_threshold: float,
    gap_alpha_small_threshold: int,
    gap_alpha_large_threshold: int,
    small_gap_alpha_max: float,
    large_gap_alpha_min: float,
    large_gap_pure_map_conf_threshold: float,
    low_conf_threshold: float,
    low_conf_alpha_cap: float,
    confidence_snap_sigma: float,
    confidence_detour_sigma: float,
    confidence_speed_sigma: float,
    interpolate_mode: str,
    low_conf_smooth_threshold: float,
    low_conf_smooth_window: int,
    low_conf_smooth_sigma: float,
    low_conf_smooth_strength: float,
    path_sampling_mode: str,
    sampling_turn_slow_angle_deg: float,
    sampling_min_speed_factor: float,
    fusion_mode: str,
    fusion_conf_power: float,
    fusion_gap_bias: float,
    fusion_snap_conf_threshold: float,
    post_smooth_map_bind_mode: str,
    post_smooth_map_bind_conf_threshold: float,
    post_smooth_map_bind_radius_m: float,
    post_smooth_map_bind_min_gap: int,
    sharp_turn_align_threshold_deg: float,
    sharp_turn_heading_tolerance_deg: float,
    sharp_turn_alpha_min: float,
    sharp_turn_conf_threshold: float,
    high_speed_var_threshold: float,
    high_speed_max_turn_deg: float,
    high_speed_curvature_iters: int,
    high_speed_curvature_strength: float,
    verbose_every: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    outputs: List[Dict] = []
    dist_cache: Dict[Tuple[int, int], float] = {}
    path_cache: Dict[Tuple[int, int], Optional[List[int]]] = {}

    agg = _empty_agg_stats()

    for i, rec in enumerate(records, start=1):
        traj_id = rec["traj_id"]
        timestamps = np.asarray(rec["timestamps"], dtype=np.int64)
        coords = np.asarray(rec["coords"], dtype=np.float64)
        mask = np.asarray(rec["mask"], dtype=bool)

        pred_coords, stats = recover_traj_with_hmm_map(
            timestamps=timestamps,
            coords=coords,
            mask=mask,
            graph=graph,
            k_candidates=k_candidates,
            candidate_radius_m=candidate_radius_m,
            candidate_radius_fallback_m=candidate_radius_fallback_m,
            emission_sigma_dist=emission_sigma_dist,
            emission_heading_weight=emission_heading_weight,
            min_gap_map=min_gap_map,
            astar_max_expansions=astar_max_expansions,
            astar_road_class_weight=astar_road_class_weight,
            astar_turn_penalty_m=astar_turn_penalty_m,
            astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
            trans_weight_detour=trans_weight_detour,
            trans_weight_speed=trans_weight_speed,
            trans_weight_turn=trans_weight_turn,
            no_path_penalty=no_path_penalty,
            max_snap_dist_m=max_snap_dist_m,
            max_detour_ratio=max_detour_ratio,
            min_speed_mps=min_speed_mps,
            max_speed_mps=max_speed_mps,
            resample_snap_radius_m=resample_snap_radius_m,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            pure_map_conf_threshold=pure_map_conf_threshold,
            gap_alpha_small_threshold=gap_alpha_small_threshold,
            gap_alpha_large_threshold=gap_alpha_large_threshold,
            small_gap_alpha_max=small_gap_alpha_max,
            large_gap_alpha_min=large_gap_alpha_min,
            large_gap_pure_map_conf_threshold=large_gap_pure_map_conf_threshold,
            low_conf_threshold=low_conf_threshold,
            low_conf_alpha_cap=low_conf_alpha_cap,
            confidence_snap_sigma=confidence_snap_sigma,
            confidence_detour_sigma=confidence_detour_sigma,
            confidence_speed_sigma=confidence_speed_sigma,
            interpolate_mode=interpolate_mode,
            low_conf_smooth_threshold=low_conf_smooth_threshold,
            low_conf_smooth_window=low_conf_smooth_window,
            low_conf_smooth_sigma=low_conf_smooth_sigma,
            low_conf_smooth_strength=low_conf_smooth_strength,
            path_sampling_mode=path_sampling_mode,
            sampling_turn_slow_angle_deg=sampling_turn_slow_angle_deg,
            sampling_min_speed_factor=sampling_min_speed_factor,
            fusion_mode=fusion_mode,
            fusion_conf_power=fusion_conf_power,
            fusion_gap_bias=fusion_gap_bias,
            fusion_snap_conf_threshold=fusion_snap_conf_threshold,
            post_smooth_map_bind_mode=post_smooth_map_bind_mode,
            post_smooth_map_bind_conf_threshold=post_smooth_map_bind_conf_threshold,
            post_smooth_map_bind_radius_m=post_smooth_map_bind_radius_m,
            post_smooth_map_bind_min_gap=post_smooth_map_bind_min_gap,
            sharp_turn_align_threshold_deg=sharp_turn_align_threshold_deg,
            sharp_turn_heading_tolerance_deg=sharp_turn_heading_tolerance_deg,
            sharp_turn_alpha_min=sharp_turn_alpha_min,
            sharp_turn_conf_threshold=sharp_turn_conf_threshold,
            high_speed_var_threshold=high_speed_var_threshold,
            high_speed_max_turn_deg=high_speed_max_turn_deg,
            high_speed_curvature_iters=high_speed_curvature_iters,
            high_speed_curvature_strength=high_speed_curvature_strength,
            dist_cache=dist_cache,
            path_cache=path_cache,
        )

        for k in agg:
            agg[k] += stats[k]

        outputs.append({"traj_id": traj_id, "coords": pred_coords})

        if verbose_every > 0 and i % verbose_every == 0:
            success_rate = 100.0 * agg["segments_map_success"] / max(1, agg["segments_try_map"])
            print(
                f"[pred] processed={i}, map_success_rate={success_rate:.2f}% "
                f"(try={agg['segments_try_map']}, success={agg['segments_map_success']})"
            )

    return outputs, agg


def _process_worker_init(
    osm_path: str,
    cache_path: Optional[str],
    cell_size_deg: float,
    predict_kwargs: Dict,
) -> None:
    global _WORKER_GRAPH, _WORKER_PREDICT_KWARGS
    cache = Path(cache_path) if cache_path else None
    _WORKER_GRAPH = load_or_build_graph(
        osm_path=Path(osm_path),
        cache_path=cache,
        cell_size_deg=cell_size_deg,
        force_rebuild=False,
    )
    _WORKER_PREDICT_KWARGS = predict_kwargs


def _process_worker_run(records_chunk: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    if _WORKER_GRAPH is None or _WORKER_PREDICT_KWARGS is None:
        raise RuntimeError("Worker is not initialized properly.")
    return build_predictions_serial(
        records=records_chunk,
        graph=_WORKER_GRAPH,
        verbose_every=0,
        **_WORKER_PREDICT_KWARGS,
    )


def build_predictions(
    records: Iterable[Dict],
    graph: Optional[Dict],
    k_candidates: int,
    candidate_radius_m: float,
    candidate_radius_fallback_m: float,
    emission_sigma_dist: float,
    emission_heading_weight: float,
    min_gap_map: int,
    astar_max_expansions: int,
    astar_road_class_weight: float,
    astar_turn_penalty_m: float,
    astar_turn_angle_threshold_deg: float,
    trans_weight_detour: float,
    trans_weight_speed: float,
    trans_weight_turn: float,
    no_path_penalty: float,
    max_snap_dist_m: float,
    max_detour_ratio: float,
    min_speed_mps: float,
    max_speed_mps: float,
    resample_snap_radius_m: float,
    alpha_min: float,
    alpha_max: float,
    pure_map_conf_threshold: float,
    gap_alpha_small_threshold: int,
    gap_alpha_large_threshold: int,
    small_gap_alpha_max: float,
    large_gap_alpha_min: float,
    large_gap_pure_map_conf_threshold: float,
    low_conf_threshold: float,
    low_conf_alpha_cap: float,
    confidence_snap_sigma: float,
    confidence_detour_sigma: float,
    confidence_speed_sigma: float,
    interpolate_mode: str,
    low_conf_smooth_threshold: float,
    low_conf_smooth_window: int,
    low_conf_smooth_sigma: float,
    low_conf_smooth_strength: float,
    path_sampling_mode: str,
    sampling_turn_slow_angle_deg: float,
    sampling_min_speed_factor: float,
    fusion_mode: str,
    fusion_conf_power: float,
    fusion_gap_bias: float,
    fusion_snap_conf_threshold: float,
    post_smooth_map_bind_mode: str,
    post_smooth_map_bind_conf_threshold: float,
    post_smooth_map_bind_radius_m: float,
    post_smooth_map_bind_min_gap: int,
    sharp_turn_align_threshold_deg: float,
    sharp_turn_heading_tolerance_deg: float,
    sharp_turn_alpha_min: float,
    sharp_turn_conf_threshold: float,
    high_speed_var_threshold: float,
    high_speed_max_turn_deg: float,
    high_speed_curvature_iters: int,
    high_speed_curvature_strength: float,
    verbose_every: int,
    executor_mode: str,
    num_workers: int,
    chunk_size: int,
    osm_path: Path,
    cache_path: Optional[Path],
    cell_size_deg: float,
    force_rebuild: bool,
) -> Tuple[List[Dict], Dict[str, int]]:
    records_list = list(records)
    if len(records_list) == 0:
        return [], _empty_agg_stats()

    worker_count = max(1, int(num_workers))
    predict_kwargs = _build_predict_kwargs(
        k_candidates=k_candidates,
        candidate_radius_m=candidate_radius_m,
        candidate_radius_fallback_m=candidate_radius_fallback_m,
        emission_sigma_dist=emission_sigma_dist,
        emission_heading_weight=emission_heading_weight,
        min_gap_map=min_gap_map,
        astar_max_expansions=astar_max_expansions,
        astar_road_class_weight=astar_road_class_weight,
        astar_turn_penalty_m=astar_turn_penalty_m,
        astar_turn_angle_threshold_deg=astar_turn_angle_threshold_deg,
        trans_weight_detour=trans_weight_detour,
        trans_weight_speed=trans_weight_speed,
        trans_weight_turn=trans_weight_turn,
        no_path_penalty=no_path_penalty,
        max_snap_dist_m=max_snap_dist_m,
        max_detour_ratio=max_detour_ratio,
        min_speed_mps=min_speed_mps,
        max_speed_mps=max_speed_mps,
        resample_snap_radius_m=resample_snap_radius_m,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        pure_map_conf_threshold=pure_map_conf_threshold,
        gap_alpha_small_threshold=gap_alpha_small_threshold,
        gap_alpha_large_threshold=gap_alpha_large_threshold,
        small_gap_alpha_max=small_gap_alpha_max,
        large_gap_alpha_min=large_gap_alpha_min,
        large_gap_pure_map_conf_threshold=large_gap_pure_map_conf_threshold,
        low_conf_threshold=low_conf_threshold,
        low_conf_alpha_cap=low_conf_alpha_cap,
        confidence_snap_sigma=confidence_snap_sigma,
        confidence_detour_sigma=confidence_detour_sigma,
        confidence_speed_sigma=confidence_speed_sigma,
        interpolate_mode=interpolate_mode,
        low_conf_smooth_threshold=low_conf_smooth_threshold,
        low_conf_smooth_window=low_conf_smooth_window,
        low_conf_smooth_sigma=low_conf_smooth_sigma,
        low_conf_smooth_strength=low_conf_smooth_strength,
        path_sampling_mode=path_sampling_mode,
        sampling_turn_slow_angle_deg=sampling_turn_slow_angle_deg,
        sampling_min_speed_factor=sampling_min_speed_factor,
        fusion_mode=fusion_mode,
        fusion_conf_power=fusion_conf_power,
        fusion_gap_bias=fusion_gap_bias,
        fusion_snap_conf_threshold=fusion_snap_conf_threshold,
        post_smooth_map_bind_mode=post_smooth_map_bind_mode,
        post_smooth_map_bind_conf_threshold=post_smooth_map_bind_conf_threshold,
        post_smooth_map_bind_radius_m=post_smooth_map_bind_radius_m,
        post_smooth_map_bind_min_gap=post_smooth_map_bind_min_gap,
        sharp_turn_align_threshold_deg=sharp_turn_align_threshold_deg,
        sharp_turn_heading_tolerance_deg=sharp_turn_heading_tolerance_deg,
        sharp_turn_alpha_min=sharp_turn_alpha_min,
        sharp_turn_conf_threshold=sharp_turn_conf_threshold,
        high_speed_var_threshold=high_speed_var_threshold,
        high_speed_max_turn_deg=high_speed_max_turn_deg,
        high_speed_curvature_iters=high_speed_curvature_iters,
        high_speed_curvature_strength=high_speed_curvature_strength,
    )

    if executor_mode == "serial" or worker_count == 1:
        if graph is None:
            raise ValueError("graph is required in serial mode")
        return build_predictions_serial(
            records=records_list,
            graph=graph,
            verbose_every=verbose_every,
            **predict_kwargs,
        )

    chunks = _split_chunks(records_list, chunk_size=max(1, chunk_size))
    outputs_by_chunk: List[Optional[List[Dict]]] = [None] * len(chunks)
    stats_by_chunk: List[Optional[Dict[str, int]]] = [None] * len(chunks)

    if executor_mode == "thread":
        if graph is None:
            raise ValueError("graph is required in thread mode")

        print(f"[pred-thread] workers={worker_count}, chunks={len(chunks)}, chunk_size={max(1, chunk_size)}")
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            fut_to_idx = {
                ex.submit(
                    build_predictions_serial,
                    records=chunk,
                    graph=graph,
                    verbose_every=0,
                    **predict_kwargs,
                ): idx
                for idx, chunk in enumerate(chunks)
            }

            done = 0
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                out_chunk, agg_chunk = fut.result()
                outputs_by_chunk[idx] = out_chunk
                stats_by_chunk[idx] = agg_chunk
                done += 1
                if verbose_every > 0 and (done % max(1, len(chunks) // 10) == 0 or done == len(chunks)):
                    print(f"[pred-thread] finished_chunks={done}/{len(chunks)}")

    elif executor_mode == "process":
        if cache_path is None:
            raise ValueError("process mode requires a cache path")

        # Build or refresh cache once in parent process to avoid concurrent cache writes.
        _ = load_or_build_graph(
            osm_path=osm_path,
            cache_path=cache_path,
            cell_size_deg=cell_size_deg,
            force_rebuild=force_rebuild,
        )

        print(f"[pred-process] workers={worker_count}, chunks={len(chunks)}, chunk_size={max(1, chunk_size)}")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_process_worker_init,
            initargs=(str(osm_path), str(cache_path), cell_size_deg, predict_kwargs),
        ) as ex:
            fut_to_idx = {ex.submit(_process_worker_run, chunk): idx for idx, chunk in enumerate(chunks)}

            done = 0
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                out_chunk, agg_chunk = fut.result()
                outputs_by_chunk[idx] = out_chunk
                stats_by_chunk[idx] = agg_chunk
                done += 1
                if verbose_every > 0 and (done % max(1, len(chunks) // 10) == 0 or done == len(chunks)):
                    print(f"[pred-process] finished_chunks={done}/{len(chunks)}")
    else:
        raise ValueError(f"Unknown executor mode: {executor_mode}")

    outputs: List[Dict] = []
    agg = _empty_agg_stats()
    for idx in range(len(chunks)):
        out_chunk = outputs_by_chunk[idx]
        agg_chunk = stats_by_chunk[idx]
        if out_chunk is not None:
            outputs.extend(out_chunk)
        if agg_chunk is not None:
            _merge_agg_stats(agg, agg_chunk)

    return outputs, agg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task A baseline2.3: HMM/Viterbi map-constrained recovery with topology-aware routing.")
    parser.add_argument("--input", required=True, type=Path, help="Path to val/test input .pkl")
    parser.add_argument("--output", required=True, type=Path, help="Path to output prediction .pkl")
    parser.add_argument("--map", type=Path, default=Path("map"), help="Path to OSM XML file")
    parser.add_argument("--cache", type=Path, default=Path("task_A_recovery/map_graph_cache.pkl"), help="Graph cache path")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild graph cache")

    parser.add_argument("--cell-size-deg", type=float, default=0.0015)
    parser.add_argument("--k-candidates", type=int, default=4)
    parser.add_argument("--candidate-radius-m", type=float, default=140.0)
    parser.add_argument("--candidate-radius-fallback-m", type=float, default=260.0)

    parser.add_argument("--emission-sigma-dist", type=float, default=55.0)
    parser.add_argument("--emission-heading-weight", type=float, default=0.7)

    parser.add_argument("--min-gap-map", type=int, default=4)
    parser.add_argument("--astar-max-expansions", type=int, default=7000)
    parser.add_argument("--astar-road-class-weight", type=float, default=1.00)
    parser.add_argument("--astar-turn-penalty-m", type=float, default=55.0)
    parser.add_argument("--astar-turn-angle-threshold-deg", type=float, default=55.0)
    parser.add_argument("--trans-weight-detour", type=float, default=3.0)
    parser.add_argument("--trans-weight-speed", type=float, default=1.5)
    parser.add_argument("--trans-weight-turn", type=float, default=0.50)
    parser.add_argument("--no-path-penalty", type=float, default=8.0)

    parser.add_argument("--max-snap-dist-m", type=float, default=90.0)
    parser.add_argument("--max-detour-ratio", type=float, default=2.80)
    parser.add_argument("--min-speed-mps", type=float, default=0.25)
    parser.add_argument("--max-speed-mps", type=float, default=48.0)
    parser.add_argument("--resample-snap-radius-m", type=float, default=45.0)

    parser.add_argument("--alpha-min", type=float, default=0.45)
    parser.add_argument("--alpha-max", type=float, default=0.90)
    parser.add_argument("--pure-map-conf-threshold", type=float, default=0.85)
    parser.add_argument("--gap-alpha-small-threshold", type=int, default=-1, help="Apply short-gap alpha cap when gap <= this value; -1 disables")
    parser.add_argument("--gap-alpha-large-threshold", type=int, default=-1, help="Apply long-gap alpha floor when gap >= this value; -1 disables")
    parser.add_argument("--small-gap-alpha-max", type=float, default=1.0)
    parser.add_argument("--large-gap-alpha-min", type=float, default=0.0)
    parser.add_argument("--large-gap-pure-map-conf-threshold", type=float, default=1.0)
    parser.add_argument("--low-conf-threshold", type=float, default=-1.0, help="Confidence threshold for low-confidence alpha cap; <0 disables")
    parser.add_argument("--low-conf-alpha-cap", type=float, default=1.0)
    parser.add_argument("--confidence-snap-sigma", type=float, default=55.0)
    parser.add_argument("--confidence-detour-sigma", type=float, default=1.60)
    parser.add_argument("--confidence-speed-sigma", type=float, default=1.2)
    parser.add_argument(
        "--interpolate-mode",
        choices=["linear", "pchip"],
        default="linear",
        help="Base interpolation mode before map fusion.",
    )
    parser.add_argument(
        "--low-conf-smooth-threshold",
        type=float,
        default=-1.0,
        help="Apply smoothing when fusion confidence is below this threshold; <0 disables.",
    )
    parser.add_argument(
        "--low-conf-smooth-window",
        type=int,
        default=0,
        help="Smoothing window radius for low-confidence segments (0 disables).",
    )
    parser.add_argument(
        "--low-conf-smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used in low-confidence smoothing.",
    )
    parser.add_argument(
        "--low-conf-smooth-strength",
        type=float,
        default=1.0,
        help="Blend weight of low-confidence smoothing in [0,1].",
    )
    parser.add_argument(
        "--path-sampling-mode",
        choices=["distance", "speed-aware"],
        default="distance",
        help="Map-path sampling policy for filling missing points.",
    )
    parser.add_argument(
        "--sampling-turn-slow-angle-deg",
        type=float,
        default=30.0,
        help="Turn-angle threshold for speed-aware sampling slowdown.",
    )
    parser.add_argument(
        "--sampling-min-speed-factor",
        type=float,
        default=0.60,
        help="Minimum segment speed factor in speed-aware sampling (0,1].",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["legacy", "continuous"],
        default="legacy",
        help="Fusion policy for blending map and interpolation results.",
    )
    parser.add_argument(
        "--fusion-conf-power",
        type=float,
        default=1.0,
        help="Confidence shaping exponent in continuous mode; <1 favors map blending at mid confidence.",
    )
    parser.add_argument(
        "--fusion-gap-bias",
        type=float,
        default=0.0,
        help="Continuous mode confidence bias by gap position (positive favors long gaps, negative favors short gaps).",
    )
    parser.add_argument(
        "--fusion-snap-conf-threshold",
        type=float,
        default=0.70,
        help="Continuous mode threshold where snap radius smoothly decays to zero.",
    )
    parser.add_argument(
        "--post-smooth-map-bind-mode",
        choices=["off", "all", "high-conf", "low-conf"],
        default="off",
        help="Bind segment points to road nodes after smoothing and curvature control.",
    )
    parser.add_argument(
        "--post-smooth-map-bind-conf-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold used by post-smooth map binding mode.",
    )
    parser.add_argument(
        "--post-smooth-map-bind-radius-m",
        type=float,
        default=-1.0,
        help="Binding radius for post-smooth map bind; <=0 uses max(snap_radius_eff, resample_snap_radius_m).",
    )
    parser.add_argument(
        "--post-smooth-map-bind-min-gap",
        type=int,
        default=0,
        help="Apply post-smooth map binding only when gap >= this value.",
    )
    parser.add_argument(
        "--sharp-turn-align-threshold-deg",
        type=float,
        default=-1.0,
        help="Enable sharp-turn heading alignment when route max turn >= threshold; <0 disables.",
    )
    parser.add_argument(
        "--sharp-turn-heading-tolerance-deg",
        type=float,
        default=60.0,
        help="Anchor-route heading tolerance used by sharp-turn alignment.",
    )
    parser.add_argument(
        "--sharp-turn-alpha-min",
        type=float,
        default=0.90,
        help="Minimum fusion alpha enforced for aligned sharp-turn segments.",
    )
    parser.add_argument(
        "--sharp-turn-conf-threshold",
        type=float,
        default=0.20,
        help="Minimum confidence to enable sharp-turn alignment.",
    )
    parser.add_argument(
        "--high-speed-var-threshold",
        type=float,
        default=-1.0,
        help="Enable curvature limiting when speed deviation >= threshold; <0 disables.",
    )
    parser.add_argument(
        "--high-speed-max-turn-deg",
        type=float,
        default=-1.0,
        help="Maximum local turn angle during high speed-variation curvature limiting; <0 disables.",
    )
    parser.add_argument(
        "--high-speed-curvature-iters",
        type=int,
        default=1,
        help="Iteration count for high speed-variation curvature limiting.",
    )
    parser.add_argument(
        "--high-speed-curvature-strength",
        type=float,
        default=1.0,
        help="Blend weight of curvature-limited result in [0,1].",
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose-every", type=int, default=500)
    parser.add_argument("--executor-mode", choices=["serial", "thread", "process"], default="serial")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker count for thread/process modes")
    parser.add_argument("--chunk-size", type=int, default=256, help="Trajectory chunk size for parallel execution")
    parser.add_argument("--gt", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_records = load_pickle(args.input)
    if args.limit > 0:
        input_records = input_records[: args.limit]

    graph: Optional[Dict] = None
    if args.executor_mode in {"serial", "thread"}:
        graph = load_or_build_graph(
            osm_path=args.map,
            cache_path=args.cache,
            cell_size_deg=args.cell_size_deg,
            force_rebuild=args.force_rebuild,
        )

    if args.num_workers <= 0:
        auto_workers = max(1, (os.cpu_count() or 1) - 1)
        args.num_workers = auto_workers
    print(
        "Executor settings | "
        f"mode={args.executor_mode} | workers={args.num_workers} | chunk_size={max(1, args.chunk_size)}"
    )

    pred_records, agg_stats = build_predictions(
        records=input_records,
        graph=graph,
        k_candidates=args.k_candidates,
        candidate_radius_m=args.candidate_radius_m,
        candidate_radius_fallback_m=args.candidate_radius_fallback_m,
        emission_sigma_dist=args.emission_sigma_dist,
        emission_heading_weight=args.emission_heading_weight,
        min_gap_map=args.min_gap_map,
        astar_max_expansions=args.astar_max_expansions,
        astar_road_class_weight=args.astar_road_class_weight,
        astar_turn_penalty_m=args.astar_turn_penalty_m,
        astar_turn_angle_threshold_deg=args.astar_turn_angle_threshold_deg,
        trans_weight_detour=args.trans_weight_detour,
        trans_weight_speed=args.trans_weight_speed,
        trans_weight_turn=args.trans_weight_turn,
        no_path_penalty=args.no_path_penalty,
        max_snap_dist_m=args.max_snap_dist_m,
        max_detour_ratio=args.max_detour_ratio,
        min_speed_mps=args.min_speed_mps,
        max_speed_mps=args.max_speed_mps,
        resample_snap_radius_m=args.resample_snap_radius_m,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        pure_map_conf_threshold=args.pure_map_conf_threshold,
        gap_alpha_small_threshold=args.gap_alpha_small_threshold,
        gap_alpha_large_threshold=args.gap_alpha_large_threshold,
        small_gap_alpha_max=args.small_gap_alpha_max,
        large_gap_alpha_min=args.large_gap_alpha_min,
        large_gap_pure_map_conf_threshold=args.large_gap_pure_map_conf_threshold,
        low_conf_threshold=args.low_conf_threshold,
        low_conf_alpha_cap=args.low_conf_alpha_cap,
        confidence_snap_sigma=args.confidence_snap_sigma,
        confidence_detour_sigma=args.confidence_detour_sigma,
        confidence_speed_sigma=args.confidence_speed_sigma,
        interpolate_mode=args.interpolate_mode,
        low_conf_smooth_threshold=args.low_conf_smooth_threshold,
        low_conf_smooth_window=args.low_conf_smooth_window,
        low_conf_smooth_sigma=args.low_conf_smooth_sigma,
        low_conf_smooth_strength=args.low_conf_smooth_strength,
        path_sampling_mode=args.path_sampling_mode,
        sampling_turn_slow_angle_deg=args.sampling_turn_slow_angle_deg,
        sampling_min_speed_factor=args.sampling_min_speed_factor,
        fusion_mode=args.fusion_mode,
        fusion_conf_power=args.fusion_conf_power,
        fusion_gap_bias=args.fusion_gap_bias,
        fusion_snap_conf_threshold=args.fusion_snap_conf_threshold,
        post_smooth_map_bind_mode=args.post_smooth_map_bind_mode,
        post_smooth_map_bind_conf_threshold=args.post_smooth_map_bind_conf_threshold,
        post_smooth_map_bind_radius_m=args.post_smooth_map_bind_radius_m,
        post_smooth_map_bind_min_gap=args.post_smooth_map_bind_min_gap,
        sharp_turn_align_threshold_deg=args.sharp_turn_align_threshold_deg,
        sharp_turn_heading_tolerance_deg=args.sharp_turn_heading_tolerance_deg,
        sharp_turn_alpha_min=args.sharp_turn_alpha_min,
        sharp_turn_conf_threshold=args.sharp_turn_conf_threshold,
        high_speed_var_threshold=args.high_speed_var_threshold,
        high_speed_max_turn_deg=args.high_speed_max_turn_deg,
        high_speed_curvature_iters=args.high_speed_curvature_iters,
        high_speed_curvature_strength=args.high_speed_curvature_strength,
        verbose_every=args.verbose_every,
        executor_mode=args.executor_mode,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        osm_path=args.map,
        cache_path=args.cache,
        cell_size_deg=args.cell_size_deg,
        force_rebuild=args.force_rebuild,
    )

    save_pickle(args.output, pred_records)
    print(f"Saved predictions: {args.output} (n={len(pred_records)})")

    map_try = agg_stats["segments_try_map"]
    map_ok = agg_stats["segments_map_success"]
    map_rate = 100.0 * map_ok / max(1, map_try)
    print(
        "Map segment stats | "
        f"total={agg_stats['segments_total']} | try_map={map_try} | "
        f"success={map_ok} ({map_rate:.2f}%) | "
        f"fallback_no_candidates={agg_stats['fallback_no_candidates']} | "
        f"fallback_no_path={agg_stats['fallback_no_path']} | "
        f"fallback_low_conf={agg_stats['fallback_low_confidence']} | "
        f"fallback_snap={agg_stats['fallback_snap_too_far']} | "
        f"fallback_detour={agg_stats['fallback_detour_too_large']} | "
        f"fallback_speed_low={agg_stats['fallback_speed_too_low']} | "
        f"fallback_speed_high={agg_stats['fallback_speed_too_high']}"
    )

    if args.gt is not None:
        gt_records = load_pickle(args.gt)
        metrics = evaluate_missing_only(pred_records, input_records, gt_records)
        print(
            "Missing-point evaluation | "
            f"count={metrics['count']} | "
            f"MAE={metrics['mae']:.4f} m | "
            f"RMSE={metrics['rmse']:.4f} m"
        )


if __name__ == "__main__":
    main()
