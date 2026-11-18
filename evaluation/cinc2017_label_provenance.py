"""
CinC 2017 label provenance (round V10, 2026-11-18; review finding PAPER-004).

Compares the label file read by experiments/cinc2017_extraction.py ($ECG_DATASETS_ROOT/cinc2017/REFERENCE.csv) and the
labels stored with the saved predictions against the label versions published with the challenge
(REFERENCE-v0.csv ... REFERENCE-v3.csv from https://physionet.org/files/challenge-2017/1.0.0/), record by record.
Reads files only; changes nothing.

Usage: ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/cinc2017_label_provenance.py <dir with REFERENCE-v*.csv>
Output: <results>/cinc2017_label_provenance.csv (one row per label file: SHA-256, records, noisy records, agreement with
        the file used, noisy records among the 8,512 analysed recordings).
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import LOCAL_DATASETS_DIR, RESULTS_DIR  # noqa: E402

USED = LOCAL_DATASETS_DIR / "cinc2017" / "REFERENCE.csv"


def read_labels(path: Path) -> pd.Series:
    d = pd.read_csv(path, header=None, names=["rec", "lab"], dtype=str)
    d["rec"] = d.rec.str.split("/").str[-1].str.strip()
    s = d.set_index("rec").lab.str.strip()
    if s.index.duplicated().any():
        raise ValueError(f"{path}: duplicated record names")
    return s


def main(version_dir: Path):
    used = read_labels(USED)
    w = pd.read_csv(RESULTS_DIR / "external_window_predictions.csv", low_memory=False,
                    usecols=["dataset", "record_name", "label"])
    saved = w[w.dataset == "CinC2017"].groupby("record_name").label.first()
    if len(saved) != 8512:
        raise ValueError(f"expected 8,512 analysed recordings, got {len(saved)}")
    if not ((used.loc[saved.index] == "~").astype(int).values == saved.values).all():
        raise ValueError("saved labels differ from the label file read by the extraction script")
    files = {"used by the extraction script": USED} | {p.name: p for p in sorted(version_dir.glob("REFERENCE-v*.csv"))}
    rows = []
    for name, p in files.items():
        s = read_labels(p)
        common = used.index.intersection(s.index)
        rows.append({"file": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "records": len(s),
                     "noisy_records": int((s == "~").sum()),
                     "records_agreeing_with_used_file": int((s.loc[common] == used.loc[common]).sum()),
                     "noisy_among_8512_analysed": int((s.reindex(saved.index) == "~").sum())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cinc2017_label_provenance.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
