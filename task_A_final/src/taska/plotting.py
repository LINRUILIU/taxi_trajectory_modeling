from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_static_histogram(values_a: np.ndarray, values_b: np.ndarray, labels: tuple[str, str], title: str, xlabel: str, out_path: Path, bins) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(values_a, bins=bins, alpha=0.55, label=labels[0], color="#1f77b4")
    ax.hist(values_b, bins=bins, alpha=0.55, label=labels[1], color="#ff7f0e")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, out_path)


def plot_official_vs_shape_scatter(rows: List[Dict], out_path: Path, seed: int = 42, max_points: int = 20000) -> None:
    if not rows:
        return
    rng = np.random.default_rng(seed)
    take = min(max_points, len(rows))
    idx = rng.choice(len(rows), size=take, replace=False) if take < len(rows) else np.arange(len(rows))
    official = np.array([rows[i]["official_mae_m"] for i in idx], dtype=np.float64)
    shape = np.array([rows[i]["shape_symmetric_m"] for i in idx], dtype=np.float64)
    gap = np.array([rows[i]["gap_size"] for i in idx], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    sc = ax.scatter(official, shape, c=gap, cmap="viridis", s=10, alpha=0.28, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Gap size")
    ax.set_title("Gap-Level Official vs Shape")
    ax.set_xlabel("Official MAE (m)")
    ax.set_ylabel("Shape symmetric distance (m)")
    ax.grid(alpha=0.2)
    _save(fig, out_path)


def plot_gap_distribution(rows8: List[Dict], rows16: List[Dict], out_path: Path) -> None:
    g8 = np.array([r["gap_size"] for r in rows8], dtype=np.int32)
    g16 = np.array([r["gap_size"] for r in rows16], dtype=np.int32)
    max_gap = int(max(g8.max(initial=1), g16.max(initial=1)))
    bins = np.arange(1, max_gap + 2) - 0.5
    plot_static_histogram(
        g8,
        g16,
        ("1/8", "1/16"),
        "Missing Point Weighted Gap Distribution",
        "Gap size",
        out_path,
        bins=bins,
    )


def plot_length_distribution(lengths8: np.ndarray, lengths16: np.ndarray, out_path: Path) -> None:
    bins = np.arange(50, 246, 5)
    plot_static_histogram(lengths8, lengths16, ("1/8", "1/16"), "Trajectory Length Distribution", "Points per trajectory", out_path, bins=bins)


def plot_interval_distribution(dt8: np.ndarray, dt16: np.ndarray, out_path: Path) -> None:
    bins = np.arange(0, 61, 1)
    plot_static_histogram(dt8, dt16, ("1/8", "1/16"), "Timestamp Interval Distribution", "Delta time (seconds)", out_path, bins=bins)
