"""
Post-commit binding record for one export round (V4, 2026-11-14; follow-up review on PAPER-009).

The export manifest (MANIFEST_vN.txt, written by export_manuscript.sh) is left unchanged. This script writes a separate
record, BINDING_vN_<commit>.txt, next to it. Read-only apart from that file. For each variant it records:
  - the full commit id;
  - for every source in the manifest, whether its SHA-256 equals that of the same path in the commit (git show);
  - whether the current DOCX/PDF files still have the SHA-256 recorded in the manifest;
  - the other production inputs: the figures referenced by the markdown (hash on disk and in the commit) and the CSS;
  - the export command, the tool versions recorded in the manifest, and the inputs that are not in the commit.

What a match shows: the sources recorded at export time are identical to the files in the commit, and the output
files are the ones hashed at export time. It does not prove by itself that re-running the export reproduces
byte-identical PDFs. Chrome output contains a creation date, and system fonts are taken from the export machine.

Usage: python evaluation/bind_export_to_commit.py <round> <commit>
"""

import hashlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_blob(commit: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main():
    rnd, commit = sys.argv[1], sys.argv[2]
    full = subprocess.run(["git", "-C", str(ROOT), "rev-parse", commit], capture_output=True, text=True,
                          check=True).stdout.strip()
    all_ok = True
    for variant in ("srep", "v2"):
        out_dir = ROOT / "manuscript" / variant / "export" / f"v{rnd}"
        manifest = (out_dir / f"MANIFEST_v{rnd}.txt").read_text(encoding="utf-8")
        lines = [f"Post-commit binding record, round v{rnd}, variant {variant}",
                 f"created: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
                 f"commit: {full}",
                 f"manifest: MANIFEST_v{rnd}.txt (unchanged; sha256 {sha(manifest.encode())})", "",
                 "Manifest header (state at export time):"]
        lines += ["  " + l for l in manifest.split("\n# SHA-256")[0].strip().splitlines()]
        lines += ["", "1. Sources recorded in the manifest vs. the same path in the commit"]
        for h, name in re.findall(r"^source  ([0-9a-f]{64})  (\S+)$", manifest, re.M):
            path = name if "/" in name else f"manuscript/{variant}/{name}"
            blob = git_blob(full, path)
            ok = blob is not None and sha(blob) == h
            all_ok &= ok
            lines.append(f"  {'MATCH   ' if ok else 'MISMATCH'} {h}  {path}")
        lines += ["", "2. Output files now vs. the hash recorded at export time"]
        for h, name in re.findall(r"^output  ([0-9a-f]{64})  (\S+)$", manifest, re.M):
            ok = sha((out_dir / name).read_bytes()) == h
            all_ok &= ok
            lines.append(f"  {'UNCHANGED' if ok else 'CHANGED  '} {h}  {name}")
        lines += ["", "3. Other production inputs (not hashed in the manifest)"]
        figs = set()
        for name in re.findall(r"^source  [0-9a-f]{64}  (\S+\.md)$", manifest, re.M):
            text = (ROOT / "manuscript" / variant / name).read_text(encoding="utf-8")
            figs |= set(re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text))
        for f in sorted(figs):
            p = (ROOT / "manuscript" / variant / f).resolve()
            rel = p.relative_to(ROOT).as_posix()
            blob = git_blob(full, rel)
            ok = blob is not None and sha(p.read_bytes()) == sha(blob)
            all_ok &= ok
            mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            lines.append(f"  figure  {'MATCH   ' if ok else 'MISMATCH'} {sha(p.read_bytes())}  {rel}  (modified {mtime})")
        lines.append("  The figures were not hashed at export time. The match above is for the files now on disk, and the")
        lines.append("  modification times (all before the export) are weaker evidence that they were the files used.")
        lines.append("  stylesheet  evaluation/pmea_print.css (hashed in section 1)")
        lines += ["", "4. Export command (evaluation/export_manuscript.sh, in the commit):",
                  f"  evaluation/export_manuscript.sh {rnd}",
                  "  per file: pandoc <name>.md --from <reader> -o <name>_vN.docx;",
                  "            pandoc <name>.md --from <reader> --standalone --embed-resources --css evaluation/pmea_print.css",
                  "              --metadata pagetitle=<name> -o <tmp>/<name>_vN.html;",
                  "            Chrome --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf=<name>_vN.pdf",
                  "  The <reader> value and the tool versions are in the manifest header above.",
                  "", "5. Inputs outside the commit",
                  "  - pandoc and Google Chrome binaries (versions above);",
                  "  - system fonts named by the stylesheet (Georgia, serif) and the default monospace font of the export",
                  "    machine (macOS);",
                  "  - no data files are read by the export; the markdown already contains every number and table.",
                  "", "Note: the markdown sources were generated from the saved results by build_manuscript_tables.py and",
                  "assemble_q1_manuscript.py, which are in the commit together with the results they read."]
        (out_dir / f"BINDING_v{rnd}_{full[:7]}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[binding] {out_dir / f'BINDING_v{rnd}_{full[:7]}.txt'}")
    print("ALL MATCH" if all_ok else "SOME MISMATCH")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
