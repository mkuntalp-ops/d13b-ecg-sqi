"""
Copy the generated numeric tables into the manuscript.

Each "## Table X" section of results/manuscript_tables_2026-10-12.md is a markdown table. The manuscript table with
the same header row is replaced row for row (header, separator and body). Tables whose header is not found are
reported and left alone; prose, captions and figures are never touched. Run after
evaluation/build_manuscript_tables.py, then check the prose numbers by hand.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEN = PROJECT_ROOT / "results" / "manuscript_tables_2026-10-12.md"
MAN = PROJECT_ROOT / "manuscript" / "final_manuscript.md"


def table_blocks(lines):
    blocks, i = {}, 0
    while i < len(lines):
        if lines[i].startswith("## Table "):
            title = lines[i][3:].strip()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("|"):
                j += 1
            k = j
            while k < len(lines) and lines[k].startswith("|"):
                k += 1
            if k > j:
                blocks[title] = lines[j:k]
            i = k
        else:
            i += 1
    return blocks


def main():
    gen = GEN.read_text(encoding="utf-8").splitlines()
    man = MAN.read_text(encoding="utf-8").splitlines()
    replaced, missing = [], []
    blocks = table_blocks(gen)
    # tables that share a header (Tables 1b and 2b) are matched in the order they appear in the manuscript
    order = sorted(blocks, key=lambda t: (t.split()[1].rstrip("bcdef"), t))
    seen = {}
    for title in order:
        rows = blocks[title]
        header = rows[0]
        hits = [i for i, l in enumerate(man) if l == header]
        n_same = sum(blocks[t][0] == header for t in blocks)
        if len(hits) != n_same:
            missing.append((title, len(hits)))
            continue
        i = hits[seen.get(header, 0)]
        seen[header] = seen.get(header, 0) + 1
        k = i
        while k < len(man) and man[k].startswith("|"):
            k += 1
        man[i:k] = rows
        replaced.append(title)
    MAN.write_text("\n".join(man) + "\n", encoding="utf-8")
    print("replaced:", ", ".join(replaced))
    for title, n in missing:
        print(f"not replaced: {title} (header found {n} times)")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
