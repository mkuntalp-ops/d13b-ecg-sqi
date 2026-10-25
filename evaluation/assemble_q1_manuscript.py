"""
Assemble the Q1 main manuscript and supplement from their templates (Q1 revision, 2026-10-25).

Each "{{Table X}}" marker in manuscript/main_q1_template.md and manuscript/supplement_q1_template.md is replaced by the
generated table "## Table X" of results/manuscript_tables_2026-10-12.md; "{{CSV:heuristic_refit}}" is rendered from
results/heuristic_refit_summary.csv. No numbers are typed by hand into the tables. Unknown markers raise an error.
Run after evaluation/build_manuscript_tables.py. Outputs: manuscript/main_q1.md, manuscript/supplement_q1.md
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_manuscript_tables import GEN, table_blocks  # noqa: E402

PAIRS = [("main_q1_template.md", "main_q1.md"), ("supplement_q1_template.md", "supplement_q1.md")]


def heuristic_refit_table() -> str:
    s = pd.read_csv(ROOT / "results" / "heuristic_refit_summary.csv")
    rows = ["| Evaluation set | Hand-set AUROC | Refitted AUROC | Δ (refit − hand-set) [95% CI] | p | Test |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |"]
    for r in s.itertuples():
        p = "< 0.001" if r.p < 0.001 else f"{r.p:.3f}"
        rows.append(f"| {r.dataset} | {r.auroc_handset:.3f} | {r.auroc_refit:.3f} | {r.delta_refit_minus_handset:+.3f} "
                    f"[{r.ci_lo:+.3f} to {r.ci_hi:+.3f}] | {p} | {r.method} |")
    return "\n".join(rows)


def fill(text: str, blocks: dict) -> str:
    def sub(m):
        key = m.group(1)
        if key == "CSV:heuristic_refit":
            return heuristic_refit_table()
        if key not in blocks:
            raise KeyError(f"no generated table for marker {{{{{key}}}}}")
        return "\n".join(blocks[key])
    return re.sub(r"\{\{([^}]+)\}\}", sub, text)


def main():
    blocks = table_blocks(GEN.read_text(encoding="utf-8").splitlines())
    for src, dst in PAIRS:
        text = re.sub(r"<!--.*?-->\n?", "", (ROOT / "manuscript" / src).read_text(encoding="utf-8"), flags=re.S)
        text = fill(text, blocks)
        (ROOT / "manuscript" / dst).write_text(text, encoding="utf-8")
        print(f"[Saved] manuscript/{dst}")


if __name__ == "__main__":
    main()
