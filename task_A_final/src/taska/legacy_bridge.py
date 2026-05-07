from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_legacy_recovery_module() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    return _load_module(
        "legacy_baseline2_hmm_map_recovery",
        root / "task_A_recovery" / "baseline2_hmm_map_recovery.py",
    )


@lru_cache(maxsize=1)
def load_legacy_game_core_module() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    return _load_module(
        "legacy_game_core",
        root / "task_A_recovery" / "game_core.py",
    )


@lru_cache(maxsize=1)
def load_legacy_analyze_module() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    return _load_module(
        "legacy_analyze_recovery",
        root / "task_A_recovery" / "analyze_recovery.py",
    )
