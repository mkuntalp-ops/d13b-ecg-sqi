"""
Assemble the Q1 main manuscript and supplement from their templates (Q1 revision, 2026-10-25).

Each "{{Table X}}" marker in manuscript/v2/*_template.md is replaced by the generated table "## Table X" of
<RESULTS_DIR>/manuscript_tables_2026-10-12.md; "{{CSV:heuristic_refit}}" is rendered from
<RESULTS_DIR>/heuristic_refit_summary.csv. No numbers are typed by hand into the tables. Unknown markers raise an error.
Citations are written as [@key] and numbered here from manuscript/v2/references_v2.md.
Run after evaluation/build_manuscript_tables.py, with ECG_RESULTS_DIR=results/v2_2026-11.
Outputs: manuscript/v2/main_v2.md, manuscript/v2/supplement_v2.md
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_manuscript_tables import GEN, RESULTS_DIR, table_blocks  # noqa: E402

# v2 templates and outputs live in manuscript/v2/; the v1 files (main_q1*.md, supplement_q1*.md) are left as submitted.
PAIRS = [("v2/main_v2_template.md", "v2/main_v2.md"), ("v2/supplement_v2_template.md", "v2/supplement_v2.md")]
REFERENCES = ROOT / "manuscript" / "v2" / "references_v2.md"


def number_citations(text: str) -> str:
    """Replace [@key; @key] by numbers in order of first appearance and fill {{References}}. Unknown or unused keys raise."""
    entries = dict(ln.split(": ", 1) for ln in REFERENCES.read_text(encoding="utf-8").splitlines() if ln.strip())
    order: list = []

    def compress(nums: list) -> str:
        nums, out, i = sorted(set(nums)), [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(str(nums[i]) if j == i else (f"{nums[i]}, {nums[j]}" if j == i + 1 else f"{nums[i]}–{nums[j]}"))
            i = j + 1
        return "[" + ", ".join(out) + "]"

    def sub(m):
        nums = []
        for key in re.findall(r"@([a-z0-9_]+)", m.group(1)):
            if key not in entries:
                raise KeyError(f"unknown reference key @{key}")
            if key not in order:
                order.append(key)
            nums.append(order.index(key) + 1)
        return compress(nums)

    text = re.sub(r"\[(@[a-z0-9_]+(?:; ?@[a-z0-9_]+)*)\]", sub, text)
    unused = set(entries) - set(order)
    if unused:
        raise ValueError(f"references never cited: {sorted(unused)}")
    refs = "\n\n".join(f"[{i + 1}] {entries[k]}" for i, k in enumerate(order))
    assert text.count("{{References}}") == 1
    return text.replace("{{References}}", refs)


def heuristic_refit_table() -> str:
    s = pd.read_csv(RESULTS_DIR / "heuristic_refit_summary.csv")
    rows = ["| Evaluation set | Hand-set heuristic AUROC | LR replacement AUROC | Δ (replacement − hand-set) [95% CI] | p | Test |",
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
        if key == "References":
            return m.group(0)  # filled by number_citations
        if key not in blocks:
            raise KeyError(f"no generated table for marker {{{{{key}}}}}")
        return "\n".join(blocks[key])
    return re.sub(r"\{\{([^}]+)\}\}", sub, text)


def main():
    blocks = table_blocks(GEN.read_text(encoding="utf-8").splitlines())
    for src, dst in PAIRS:
        text = re.sub(r"<!--.*?-->\n?", "", (ROOT / "manuscript" / src).read_text(encoding="utf-8"), flags=re.S)
        text = fill(text, blocks)
        if "{{References}}" in text:
            text = number_citations(text)
        (ROOT / "manuscript" / dst).write_text(text, encoding="utf-8")
        print(f"[Saved] manuscript/{dst}")


if __name__ == "__main__":
    main()
