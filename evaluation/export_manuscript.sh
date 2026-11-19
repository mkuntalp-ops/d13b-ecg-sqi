#!/usr/bin/env bash
# Export the assembled manuscripts to DOCX and PDF for one revision round, and write a SHA-256 manifest
# (V4, 2026-11-14; the follow-up review asked for the export commands to be recorded, not reconstructed).
#
# Usage: evaluation/export_manuscript.sh <round>          e.g.  evaluation/export_manuscript.sh 4
# Output: manuscript/<variant>/export/v<round>/<name>_v<round>.{docx,pdf} and
#         manuscript/<variant>/export/v<round>/MANIFEST_v<round>.txt (code version, tools, SHA-256 of sources and outputs)
# Run after assemble_q1_manuscript.py and verify_v2_prose.py. Refuses to overwrite an existing round folder,
# so earlier exports stay as the record of their round.
# Tools: pandoc (markdown -> DOCX, markdown -> standalone HTML with evaluation/pmea_print.css),
#        Google Chrome headless (HTML -> PDF). Override Chrome with $CHROME.
#        V14: evaluation/docx_layout.py adds page numbers (and line numbers in main_srep) to the srep DOCX files;
#        the print CSS adds page numbers to every PDF.
set -euo pipefail

ROUND="${1:?usage: export_manuscript.sh <round number>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
CSS="$ROOT/evaluation/pmea_print.css"
PYTHON="${PYTHON:-python3}"   # evaluation/docx_layout.py uses the standard library only
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

files_of() { case "$1" in
  srep) echo "main_srep supplement_srep tripod_ai_checklist_srep cover_letter_srep" ;;
  v2)   echo "main_v2 supplement_v2 tripod_ai_checklist_v2" ;; esac; }
reader_of() { case "$1" in
  srep) echo "markdown+pipe_tables+lists_without_preceding_blankline+superscript" ;;
  v2)   echo "markdown+pipe_tables+lists_without_preceding_blankline" ;; esac; }   # bash 3.2: no associative arrays

for variant in srep v2; do
  src="$ROOT/manuscript/$variant"
  out="$src/export/v$ROUND"
  if [ -e "$out" ]; then echo "refusing to overwrite existing $out" >&2; exit 1; fi
  mkdir -p "$out"
  manifest="$out/MANIFEST_v$ROUND.txt"
  {
    echo "Export manifest, round v$ROUND, variant $variant"
    echo "created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git HEAD: $(git -C "$ROOT" rev-parse HEAD)  branch: $(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
    # V7 correction: tracked changes and untracked files are reported separately and the git output is kept in the
    # manifest. The earlier single check also counted the untracked export directory and so always printed DIRTY.
    tracked="$(git -C "$ROOT" status --porcelain --untracked-files=no -- evaluation manuscript/$variant results tests)"
    untracked="$(git -C "$ROOT" ls-files --others --exclude-standard -- evaluation manuscript/$variant results tests |
                 { grep -v "^manuscript/$variant/export/" || true; })"
    if [ -z "$tracked" ]; then
      echo "tracked files (evaluation/, manuscript/$variant, results/, tests/): no uncommitted changes"
    else
      echo "tracked files (evaluation/, manuscript/$variant, results/, tests/): UNCOMMITTED CHANGES (git status --porcelain):"
      echo "$tracked" | sed 's/^/  /'
      echo "  the source hashes below identify the exact inputs; the commit that contains them is recorded separately"
    fi
    if [ -z "$untracked" ]; then
      echo "untracked files in the same paths (export directory excluded): none"
    else
      echo "untracked files in the same paths (export directory excluded):"
      echo "$untracked" | sed 's/^/  /'
    fi
    echo "pandoc: $(pandoc --version | head -1)"
    echo "chrome: $("$CHROME" --version)"
    echo "pandoc reader: $(reader_of $variant)"
    echo
    echo "# SHA-256  file"
  } > "$manifest"
  ( cd "$src" && shasum -a 256 $(for n in $(files_of $variant); do echo "$n.md"; done) ) | sed 's/^/source  /' >> "$manifest"
  ( cd "$ROOT" && shasum -a 256 evaluation/pmea_print.css evaluation/docx_layout.py ) | sed 's/^/source  /' >> "$manifest"
  # figures embedded by pandoc (V4 follow-up: hash them at export time too), paths relative to the repository root
  for n in $(files_of $variant); do { grep -oE '!\[[^]]*\]\([^) ]+\)' "$src/$n.md" || true; } | sed -E 's/.*\(([^) ]+)\)/\1/'; done | sort -u |
  while read -r fig; do
    dir="$(cd "$src/$(dirname "$fig")" && pwd)"
    ( cd "$ROOT" && shasum -a 256 "${dir#$ROOT/}/$(basename "$fig")" )
  done | sed 's/^/figure  /' >> "$manifest"
  for n in $(files_of $variant); do
    ( cd "$src"
      pandoc "$n.md" --from "$(reader_of $variant)" -o "$out/${n}_v$ROUND.docx"
      # V14: page numbers (and line numbers for the main text) in the Scientific Reports DOCX files
      case "$n" in
        main_srep) "$PYTHON" "$ROOT/evaluation/docx_layout.py" "$out/${n}_v$ROUND.docx" --line-numbers ;;
        supplement_srep|tripod_ai_checklist_srep) "$PYTHON" "$ROOT/evaluation/docx_layout.py" "$out/${n}_v$ROUND.docx" ;;
      esac
      pandoc "$n.md" --from "$(reader_of $variant)" --standalone --embed-resources --css "$CSS" \
        --metadata pagetitle="$n" -o "$TMP/${n}_v$ROUND.html" )
    "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
      --print-to-pdf="$out/${n}_v$ROUND.pdf" "file://$TMP/${n}_v$ROUND.html" 2>/dev/null
  done
  ( cd "$out" && shasum -a 256 *_v"$ROUND".docx *_v"$ROUND".pdf ) | sed 's/^/output  /' >> "$manifest"
  echo "[export] $out"
done
