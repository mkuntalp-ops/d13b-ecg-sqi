"""
Pack text files into one text file for reviewers whose environment does not accept archives (V6, 2026-11-14).

Each file is preceded by '===== FILE n/N: <path> =====' and its SHA-256 and followed by '===== END FILE n: <path> ====='.
Bytes are written unchanged (CRLF files keep their line endings), so every file can be cut out and its hash checked.

Usage: python evaluation/make_text_bundle.py <out.txt> <root_dir> "<header text>" <relative paths...>
"""

import hashlib
import sys
from pathlib import Path


def main():
    out, root, header, paths = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    head = [header, "Each file starts with '===== FILE n/N: <path> =====' and its SHA-256, so it can be cut out unchanged.",
            "", "CONTENTS"]
    body = []
    for i, p in enumerate(paths, 1):
        b = (root / p).read_bytes()
        h = hashlib.sha256(b).hexdigest()
        text = b.decode("utf-8")
        head.append(f"  {i:2d}. {p}  ({len(b):,} bytes, sha256 {h})")
        body += ["", f"===== FILE {i}/{len(paths)}: {p} =====", f"sha256: {h}", "",
                 text[:-1] if text.endswith("\n") else text, f"===== END FILE {i}: {p} ====="]
    out.write_bytes(("\n".join(head + body) + "\n").encode("utf-8"))
    print(f"[bundle] {out} ({out.stat().st_size:,} bytes, {len(paths)} files)")


if __name__ == "__main__":
    main()
