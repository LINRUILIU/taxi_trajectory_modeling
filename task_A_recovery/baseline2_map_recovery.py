import argparse
import heapq
import math
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
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


def _safe_interpolate_1d(x_all: np.ndarray, x_known: np.ndarray, y_known: np.ndarray) -> np.ndarray:
    if len(x_known) == 0:
        return np.full_like(x_all, np.nan, dtype=np.float64)
    if len(x_known) == 1:
        return np.full_like(x_all, y_known[0], dtype=np.float64)
    return np.interp(x_all, x_known, y_known)


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


def parse_oneway_value(v: str) -> str:
    vv = (v or "").strip().lower()
    if vv in {"yes", "1", "true"}:
        return "forward"
    if vv in {"-1", "reverse"}:
        return "reverse"
    return "both"


def is_drivable_highway(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v in DRIVABLE_HIGHWAYS


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
    ways: List[Tuple[List[int], str]] = []
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
                ways.append((refs, oneway))
                needed_node_ids.update(refs)
            elem.clear()
        elif elem.tag == "node":
            # Nodes are not used in pass1; clear here to keep memory stable.
            elem.clear()

    print(
        f"[map] pass1 done: drivable_ways={len(ways)}, "
        f"needed_nodes={len(needed_node_ids)}"
    )

    if len(ways) == 0:
        raise ValueError(
            "No drivable highways were parsed from OSM file. "
            "Check highway filtering rules or OSM file integrity."
        )

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
            # Ways are not used in pass2.
            elem.clear()

    print(f"[map] pass2 done: kept_nodes={len(node_coords)}")

    adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    edge_count = 0
    skipped_edges = 0

    for refs, oneway in ways:
        for a, b in zip(refs[:-1], refs[1:]):
            pa = node_coords.get(a)
            pb = node_coords.get(b)
            if pa is None or pb is None:
                skipped_edges += 1
                continue

            dist = float(haversine_meters(pa[0], pa[1], pb[0], pb[1]))
            if dist <= 0.0:
                continue

            if oneway in {"both", "forward"}:
                adjacency[a].append((b, dist))
                edge_count += 1
            if oneway in {"both", "reverse"}:
                adjacency[b].append((a, dist))
                edge_count += 1

    graph_nodes = set(adjacency.keys())
    for src, nbrs in adjacency.items():
        for dst, _ in nbrs:
            graph_nodes.add(dst)

    node_ids = np.array(sorted(graph_nodes), dtype=np.int64)
    lons = np.empty(len(node_ids), dtype=np.float64)
    lats = np.empty(len(node_ids), dtype=np.float64)

    for i, nid in enumerate(node_ids):
        lon, lat = node_coords[nid]
        lons[i] = lon
        lats[i] = lat

    print(
        f"[map] graph done: nodes={len(node_ids)}, directed_edges={edge_count}, "
        f"skipped_edges={skipped_edges}"
    )

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

    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}
    grid_info = build_grid_index(lons, lats, cell_size_deg=cell_size_deg)

    return {
        "bounds": graph_data.get("bounds"),
        "node_ids": node_ids,
        "lons": lons,
        "lats": lats,
        "adjacency": graph_data["adjacency"],
        "id_to_idx": id_to_idx,
        "grid": grid_info["grid"],
        "grid_min_lon": grid_info["min_lon"],
        "grid_min_lat": grid_info["min_lat"],
        "cell_size_deg": grid_info["cell_size_deg"],
    }


def load_or_build_graph(osm_path: Path, cache_path: Optional[Path], cell_size_deg: float, force_rebuild: bool) -> Dict:
    graph_data = None

    if cache_path is not None and cache_path.exists() and not force_rebuild:
        print(f"[map] load cache: {cache_path}")
        with cache_path.open("rb") as f:
            graph_data = pickle.load(f)

    if graph_data is None:
        graph_data = build_road_graph_from_osm(osm_path)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[map] cache saved: {cache_path}")

    runtime_graph = prepare_graph_runtime(graph_data, cell_size_deg=cell_size_deg)
    return runtime_graph


def nearest_node(
    lon: float,
    lat: float,
    graph: Dict,
    max_dist_m: float,
) -> Tuple[Optional[int], float]:
    cell_size = graph["cell_size_deg"]
    min_lon = graph["grid_min_lon"]
    min_lat = graph["grid_min_lat"]

    gx = int((lon - min_lon) / cell_size)
    gy = int((lat - min_lat) / cell_size)

    deg_lat = max_dist_m / 111320.0
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    deg_lon = max_dist_m / (111320.0 * cos_lat)
    ring = int(math.ceil(max(deg_lat, deg_lon) / cell_size))

    candidate_idxs: List[int] = []
    grid = graph["grid"]
    for dx in range(-ring, ring + 1):
        for dy in range(-ring, ring + 1):
            key = (gx + dx, gy + dy)
            if key in grid:
                candidate_idxs.extend(grid[key])

    if not candidate_idxs:
        return None, float("inf")

    cand = np.asarray(candidate_idxs, dtype=np.int64)
    cand_lons = graph["lons"][cand]
    cand_lats = graph["lats"][cand]

    dists = haversine_meters(cand_lons, cand_lats, lon, lat)
    best_i = int(np.argmin(dists))
    best_d = float(dists[best_i])
    if best_d > max_dist_m:
        return None, best_d

    node_id = int(graph["node_ids"][cand[best_i]])
    return node_id, best_d


def reconstruct_path(came_from: Dict[int, int], end_node: int) -> List[int]:
    path = [end_node]
    cur = end_node
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def astar_shortest_path(
    start_node: int,
    end_node: int,
    graph: Dict,
    max_expansions: int,
) -> Optional[List[int]]:
    if start_node == end_node:
        return [start_node]

    adjacency = graph["adjacency"]
    if start_node not in adjacency:
        return None

    id_to_idx = graph["id_to_idx"]
    if end_node not in id_to_idx:
        return None

    goal_idx = id_to_idx[end_node]
    goal_lon = graph["lons"][goal_idx]
    goal_lat = graph["lats"][goal_idx]

    def heuristic(node_id: int) -> float:
        idx = id_to_idx.get(node_id)
        if idx is None:
            return float("inf")
        lon = graph["lons"][idx]
        lat = graph["lats"][idx]
        return float(haversine_meters(lon, lat, goal_lon, goal_lat))

    open_heap: List[Tuple[float, float, int]] = []
    g_score: Dict[int, float] = {start_node: 0.0}
    came_from: Dict[int, int] = {}

    h0 = heuristic(start_node)
    heapq.heappush(open_heap, (h0, 0.0, start_node))

    expansions = 0
    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)
        best_known = g_score.get(cur, float("inf"))
        if g_cur > best_known + 1e-9:
            continue

        if cur == end_node:
            return reconstruct_path(came_from, end_node)

        expansions += 1
        if expansions > max_expansions:
            return None

        for nxt, w in adjacency.get(cur, []):
            g_next = g_cur + w
            if g_next + 1e-9 < g_score.get(nxt, float("inf")):
                g_score[nxt] = g_next
                came_from[nxt] = cur
                f_next = g_next + heuristic(nxt)
                heapq.heappush(open_heap, (f_next, g_next, nxt))

    return None


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


def recover_traj_with_map(
    timestamps: np.ndarray,
    coords: np.ndarray,
    mask: np.ndarray,
    graph: Dict,
    min_gap_map: int,
    max_match_dist_m: float,
    max_snap_dist_m: float,
    astar_max_expansions: int,
    max_detour_ratio: float,
    min_speed_mps: float,
    max_speed_mps: float,
    map_blend_alpha: float,
    path_cache: Dict[Tuple[int, int], Optional[List[int]]],
) -> Tuple[np.ndarray, Dict[str, int]]:
    pred = linear_interpolate_traj(timestamps=timestamps, coords=coords, mask=mask).astype(np.float64)

    known_idx = np.where(mask)[0]
    stats = {
        "segments_total": 0,
        "segments_try_map": 0,
        "segments_map_success": 0,
        "fallback_no_match": 0,
        "fallback_no_path": 0,
        "fallback_low_confidence": 0,
    }

    for a, b in zip(known_idx[:-1], known_idx[1:]):
        gap = int(b - a - 1)
        if gap <= 0:
            continue

        stats["segments_total"] += 1
        if gap < min_gap_map:
            continue

        stats["segments_try_map"] += 1

        lon_a, lat_a = float(coords[a, 0]), float(coords[a, 1])
        lon_b, lat_b = float(coords[b, 0]), float(coords[b, 1])

        start_node, start_snap_d = nearest_node(lon_a, lat_a, graph, max_dist_m=max_match_dist_m)
        end_node, end_snap_d = nearest_node(lon_b, lat_b, graph, max_dist_m=max_match_dist_m)

        if start_node is None or end_node is None:
            stats["fallback_no_match"] += 1
            continue

        key = (start_node, end_node)
        path_nodes = path_cache.get(key)
        if path_nodes is None and key not in path_cache:
            path_nodes = astar_shortest_path(
                start_node=start_node,
                end_node=end_node,
                graph=graph,
                max_expansions=astar_max_expansions,
            )
            path_cache[key] = path_nodes

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

        straight_dist = float(haversine_meters(lon_a, lat_a, lon_b, lat_b))
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
        route_dist = max(route_dist, straight_dist)

        dt = float(max(1, int(timestamps[b] - timestamps[a])))
        speed = route_dist / dt
        detour_ratio = route_dist / max(straight_dist, 1e-6)

        if (
            start_snap_d > max_snap_dist_m
            or end_snap_d > max_snap_dist_m
            or detour_ratio > max_detour_ratio
            or speed < min_speed_mps
            or speed > max_speed_mps
        ):
            stats["fallback_low_confidence"] += 1
            continue

        polyline = np.vstack(
            [
                np.array([[lon_a, lat_a]], dtype=np.float64),
                np.asarray(path_coords, dtype=np.float64),
                np.array([[lon_b, lat_b]], dtype=np.float64),
            ]
        )

        dt = float(timestamps[b] - timestamps[a])
        if dt <= 0:
            ratios = (np.arange(a + 1, b) - a) / float(b - a)
        else:
            ratios = (timestamps[a + 1 : b] - timestamps[a]) / dt

        filled_map = sample_polyline_by_ratios(polyline=polyline, ratios=ratios)
        if map_blend_alpha < 1.0:
            linear_seg = pred[a + 1 : b].astype(np.float64)
            filled = map_blend_alpha * filled_map + (1.0 - map_blend_alpha) * linear_seg
        else:
            filled = filled_map
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
    mae = float(np.mean(np.abs(all_dists)))
    rmse = float(np.sqrt(np.mean(all_dists**2)))
    return {"mae": mae, "rmse": rmse, "count": int(all_dists.size)}


def build_predictions(
    records: Iterable[Dict],
    graph: Dict,
    min_gap_map: int,
    max_match_dist_m: float,
    max_snap_dist_m: float,
    astar_max_expansions: int,
    max_detour_ratio: float,
    min_speed_mps: float,
    max_speed_mps: float,
    map_blend_alpha: float,
    verbose_every: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    outputs: List[Dict] = []
    path_cache: Dict[Tuple[int, int], Optional[List[int]]] = {}

    agg = {
        "segments_total": 0,
        "segments_try_map": 0,
        "segments_map_success": 0,
        "fallback_no_match": 0,
        "fallback_no_path": 0,
        "fallback_low_confidence": 0,
    }

    for i, rec in enumerate(records, start=1):
        traj_id = rec["traj_id"]
        timestamps = np.asarray(rec["timestamps"], dtype=np.int64)
        coords = np.asarray(rec["coords"], dtype=np.float64)
        mask = np.asarray(rec["mask"], dtype=bool)

        pred_coords, stats = recover_traj_with_map(
            timestamps=timestamps,
            coords=coords,
            mask=mask,
            graph=graph,
            min_gap_map=min_gap_map,
            max_match_dist_m=max_match_dist_m,
            max_snap_dist_m=max_snap_dist_m,
            astar_max_expansions=astar_max_expansions,
            max_detour_ratio=max_detour_ratio,
            min_speed_mps=min_speed_mps,
            max_speed_mps=max_speed_mps,
            map_blend_alpha=map_blend_alpha,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task A baseline2: map-constrained trajectory recovery.")
    parser.add_argument("--input", required=True, type=Path, help="Path to val/test input .pkl")
    parser.add_argument("--output", required=True, type=Path, help="Path to output prediction .pkl")
    parser.add_argument("--map", type=Path, default=Path("map"), help="Path to OSM XML file")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("task_A_recovery/map_graph_cache.pkl"),
        help="Graph cache path (.pkl)",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding map graph cache")
    parser.add_argument("--cell-size-deg", type=float, default=0.0015, help="Grid cell size in degrees")
    parser.add_argument("--min-gap-map", type=int, default=4, help="Use map route only when gap >= this threshold")
    parser.add_argument("--max-match-dist-m", type=float, default=220.0, help="Max distance for nearest-node match")
    parser.add_argument("--max-snap-dist-m", type=float, default=90.0, help="Max allowed snap distance for confident map usage")
    parser.add_argument("--astar-max-expansions", type=int, default=6000, help="Max expansions for A* search")
    parser.add_argument("--max-detour-ratio", type=float, default=2.0, help="Reject map route when path/straight exceeds this ratio")
    parser.add_argument("--min-speed-mps", type=float, default=0.3, help="Min plausible average speed on routed segment")
    parser.add_argument("--max-speed-mps", type=float, default=35.0, help="Max plausible average speed on routed segment")
    parser.add_argument("--map-blend-alpha", type=float, default=0.7, help="Blend weight for map result (1.0 means pure map)")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N trajectories (0 means all)")
    parser.add_argument("--verbose-every", type=int, default=500, help="Progress print interval")
    parser.add_argument("--gt", type=Path, default=None, help="Optional GT .pkl for validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_records = load_pickle(args.input)
    if args.limit > 0:
        input_records = input_records[: args.limit]

    graph = load_or_build_graph(
        osm_path=args.map,
        cache_path=args.cache,
        cell_size_deg=args.cell_size_deg,
        force_rebuild=args.force_rebuild,
    )

    pred_records, agg_stats = build_predictions(
        records=input_records,
        graph=graph,
        min_gap_map=args.min_gap_map,
        max_match_dist_m=args.max_match_dist_m,
        max_snap_dist_m=args.max_snap_dist_m,
        astar_max_expansions=args.astar_max_expansions,
        max_detour_ratio=args.max_detour_ratio,
        min_speed_mps=args.min_speed_mps,
        max_speed_mps=args.max_speed_mps,
        map_blend_alpha=args.map_blend_alpha,
        verbose_every=args.verbose_every,
    )

    save_pickle(args.output, pred_records)
    print(f"Saved predictions: {args.output} (n={len(pred_records)})")

    map_try = agg_stats["segments_try_map"]
    map_ok = agg_stats["segments_map_success"]
    map_rate = 100.0 * map_ok / max(1, map_try)
    print(
        "Map segment stats | "
        f"total={agg_stats['segments_total']} | "
        f"try_map={map_try} | success={map_ok} ({map_rate:.2f}%) | "
        f"fallback_no_match={agg_stats['fallback_no_match']} | "
        f"fallback_no_path={agg_stats['fallback_no_path']} | "
        f"fallback_low_conf={agg_stats['fallback_low_confidence']}"
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
