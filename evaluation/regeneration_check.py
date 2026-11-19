"""
Regeneration check of stored results from saved predictions (round V16, 2026-11-19; review finding PAPER-003, with
PAPER-001 and PAPER-006).

Compares the outputs of a fresh run of
  evaluation/paired_comparisons.py       (all paired AUROC tests and their Holm adjustment),
  evaluation/bootstrap_p_audit.py        (BUT QDB resample counts and unrounded bootstrap p-values),
  evaluation/local_threshold_triviality.py (joint summaries of the local-threshold draws)
written to a copy of the results directory with the committed files in the original directory. None of these
scripts fits a model; they read saved predictions and draw-level results only.

Each file is reported as IDENTICAL (same SHA-256), NUMERICALLY EQUAL (same columns, rows and text; numbers equal
within an absolute tolerance of 1e-12) or DIFFERENT. The script exits with status 1 if any file differs or is missing.

Usage (the fresh run must already have been made with ECG_RESULTS_DIR=<copy>):
  python evaluation/regeneration_check.py <original results dir> <regenerated results dir> <log file>
"""

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ["paired_comparisons.csv", "butqdb_bootstrap_p_audit.csv", "butqdb_bootstrap_p_audit_all_pairs.csv",
         "local_threshold_triviality.csv"]
SCRIPTS = ["evaluation/paired_comparisons.py", "evaluation/bootstrap_p_audit.py",
           "evaluation/local_threshold_triviality.py"]
ATOL = 1e-12


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(a: Path, b: Path) -> str:
    if not a.exists() or not b.exists():
        return "MISSING"
    if sha256(a) == sha256(b):
        return "IDENTICAL"
    x, y = pd.read_csv(a, low_memory=False), pd.read_csv(b, low_memory=False)
    if list(x.columns) != list(y.columns) or x.shape != y.shape:
        return "DIFFERENT"
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            if not np.allclose(x[c].values.astype(float), y[c].values.astype(float), rtol=0, atol=ATOL, equal_nan=True):
                return "DIFFERENT"
        elif not x[c].astype(str).equals(y[c].astype(str)):
            return "DIFFERENT"
    return "NUMERICALLY EQUAL"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    orig, regen, log = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = [(f, compare(orig / f, regen / f), sha256(orig / f) if (orig / f).exists() else "-") for f in FILES]
    dirty = git("status", "--porcelain", "--", *SCRIPTS, "config.py")
    lines = [f"Regeneration check, {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
             f"Code: commit {git('rev-parse', 'HEAD')}; scripts {'MODIFIED: ' + dirty if dirty else 'unmodified'}",
             f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}",
             f"Scripts rerun (no model fitted): {', '.join(SCRIPTS)}",
             f"Committed outputs: {orig}", "",
             f"{'file':42s} {'result':18s} sha256 of committed file"]
    lines += [f"{f:42s} {r:18s} {h}" for f, r, h in rows]
    ok = all(r in ("IDENTICAL", "NUMERICALLY EQUAL") for _, r, _ in rows)
    lines += ["", "ALL MATCH" if ok else "MISMATCH"]
    log.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
