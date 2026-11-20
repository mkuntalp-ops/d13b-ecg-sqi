"""
Submission manifest (round V21, 2026-11-20; review finding PAPER-005): size and SHA-256 of every file that the submitted
version depends on, so that each table, figure and saved output can be matched to the exact bytes in the repository.

Covered: the saved outputs (<results>/ and <results>_quota_sensitivity/), the figures, the Scientific Reports manuscript
sources and generated files (manuscript/srep/), and the analysis code (evaluation/, experiments/, features/, models/,
config.py, requirements files). Files ignored by git (for example local-only notes) are excluded, and so is the manifest
itself. The commit that contains the manifest identifies the code version; the export binding files
(evaluation/bind_export_to_commit.py) tie the exported PDF and DOCX files to that commit.

Regeneration: ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/build_manuscript_tables.py, then
evaluation/assemble_q1_manuscript.py --variant srep and evaluation/verify_v2_prose.py --variant srep; the byte-for-byte
regeneration of the paired tests is evaluation/regeneration_check.py.

Output: <results>/submission_manifest.csv
Usage:  ECG_RESULTS_DIR=results/v2_2026-11 python evaluation/submission_manifest.py
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import RESULTS_DIR  # noqa: E402

OUT = RESULTS_DIR / "submission_manifest.csv"


def main():
    rel = RESULTS_DIR.resolve().relative_to(ROOT)
    paths = [str(rel), f"{rel}_quota_sensitivity", "manuscript/srep", "evaluation", "experiments", "features", "models",
             "config.py", "requirements.txt", "requirements-lock.txt"]
    paths = [p for p in paths if (ROOT / p).exists()]
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *paths], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.split("\n")
    out_rel = str(OUT.resolve().relative_to(ROOT))
    rows = []
    for f in sorted(x for x in files if x and x != out_rel and "/export/" not in x and "__pycache__" not in x):
        p = ROOT / f
        if p.is_file():
            rows.append({"path": f, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{len(rows)} files; [Saved] {OUT}")


if __name__ == "__main__":
    main()
