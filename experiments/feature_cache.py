"""
Segment-id keyed access to cached frozen-model features (d1-3B probe, HuBERT-ECG probe).

The extraction scripts store one feature matrix per window set as <set>_<feat>.npy, with the row order in
<set>_ids.csv. v2 (revision of 2026-11-05) evaluates other window sets (988 BUT QDB windows, 998 CinC 2011 records),
most of which were already extracted under other set names. These helpers look rows up by segment id across every
cached set in a directory, so a window is never extracted twice and never matched by position.

A segment id that appears in two cached sets must have identical features in both (checked on load); otherwise the
lookup raises, because it would be ambiguous which extraction to use.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


_MEMO: Dict[tuple, Dict[str, np.ndarray]] = {}


def _index(cache_dir: Path, feat: Optional[str]) -> Dict[str, np.ndarray]:
    """feat=None reads <set>.npy (HuBERT-ECG cache layout); otherwise <set>_<feat>.npy (d1-3B probe layout).
    Memoized per process; the key includes every ids file with its modification time, so new sets are seen."""
    id_files = sorted(Path(cache_dir).glob("*_ids.csv"))
    key = (str(Path(cache_dir).resolve()), feat, tuple((str(f), f.stat().st_mtime_ns) for f in id_files))
    if key in _MEMO:
        return _MEMO[key]
    rows: Dict[str, np.ndarray] = {}
    for ids_path in id_files:
        name = ids_path.name[: -len("_ids.csv")]
        mat_path = ids_path.with_name(f"{name}.npy" if feat is None else f"{name}_{feat}.npy")
        if not mat_path.exists():
            continue
        ids = pd.read_csv(ids_path)["segment_id"].astype(str).tolist()
        mat = np.load(mat_path)
        if mat.shape[0] != len(ids):
            raise ValueError(f"{mat_path}: {mat.shape[0]} rows but {len(ids)} ids")
        for sid, vec in zip(ids, mat):
            if sid in rows and not np.allclose(rows[sid], vec, rtol=0, atol=1e-5):
                raise ValueError(f"{sid}: different cached {feat} features in two sets ({name} and an earlier one)")
            rows.setdefault(sid, vec)
    _MEMO[key] = rows
    return rows


def missing_ids(cache_dir: Path, feat: Optional[str], segment_ids: Iterable[str]) -> List[str]:
    """Segment ids (in input order) without cached features of this kind."""
    have = _index(cache_dir, feat)
    return [s for s in map(str, segment_ids) if s not in have]


def load_features(cache_dir: Path, feat: Optional[str], segment_ids: Iterable[str]) -> np.ndarray:
    """Feature matrix with one row per segment id, in the given order; raises if any id is not cached."""
    have = _index(cache_dir, feat)
    ids = list(map(str, segment_ids))
    miss = [s for s in ids if s not in have]
    if miss:
        raise KeyError(f"{len(miss)} of {len(ids)} segment ids have no cached '{feat}' features in {cache_dir} "
                       f"(first: {miss[:3]})")
    return np.stack([have[s] for s in ids])


def save_set(cache_dir: Path, name: str, segment_ids: List[str], features: Dict[Optional[str], np.ndarray]) -> None:
    """Write a new cached set; refuses to overwrite an existing one."""
    cache_dir = Path(cache_dir)
    ids_path = cache_dir / f"{name}_ids.csv"
    if ids_path.exists():
        raise FileExistsError(f"{ids_path} exists; choose a new set name")
    for feat, mat in features.items():
        if mat.shape[0] != len(segment_ids):
            raise ValueError(f"{feat}: {mat.shape[0]} rows for {len(segment_ids)} ids")
        np.save(cache_dir / (f"{name}.npy" if feat is None else f"{name}_{feat}.npy"), mat.astype(np.float32))
    pd.Series(segment_ids).to_csv(ids_path, index=False, header=["segment_id"])
