"""Check every DOI reference in a reference file against Crossref (title, first author, year, volume, pages, journal,
update/retraction notices). Usage: python crossref_check.py references.md out.csv"""
import csv
import json
import re
import sys
import time
import unicodedata
import urllib.request
from difflib import SequenceMatcher

UA = {"User-Agent": "reference-check/1.0 (scientific manuscript audit)"}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"[^a-z0-9 ]+", " ", s).split()


def get(url):
    for _ in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2)
    raise err


rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    if ":" not in line:
        continue
    key, ref = line.split(":", 1)
    ref = ref.strip()
    m = re.search(r"(?:doi\.org/|doi:)(10\.[^\s,;]+?)(?:[.)]?\s|[.)]?$)", ref)
    if not m:
        rows.append({"key": key, "doi": "", "status": "NO DOI (check manually)", "ref": ref})
        continue
    doi = m.group(1).rstrip(".")
    if doi.startswith("10.13026") or doi.startswith("10.22489"):
        pass
    try:
        msg = get("https://api.crossref.org/works/" + urllib.request.quote(doi))["message"]
    except Exception as e:  # noqa: BLE001
        # DataCite DOIs (PhysioNet) are not in Crossref
        try:
            d = get("https://api.datacite.org/dois/" + urllib.request.quote(doi))["data"]["attributes"]
            msg = {"title": [d["titles"][0]["title"]], "author": [{"family": c.get("familyName", c.get("name", ""))} for c in d["creators"]],
                   "issued": {"date-parts": [[d["publicationYear"]]]}, "container-title": [d.get("publisher", "")], "source": "DataCite"}
        except Exception as e2:  # noqa: BLE001
            rows.append({"key": key, "doi": doi, "status": f"DOI NOT RESOLVED ({e}; {e2})", "ref": ref})
            continue
    title = (msg.get("title") or [""])[0]
    # title in the reference: text after the author block and before the journal
    tsim = max(SequenceMatcher(None, " ".join(norm(title)), " ".join(norm(seg))).ratio() for seg in re.split(r"\. ", ref))
    fam = (msg.get("author") or [{"family": msg.get("publisher", "")}])[0].get("family", "") or (msg.get("author") or [{}])[0].get("name", "")
    author_ok = bool(fam) and norm(fam)[0] in norm(ref.split(".")[0] + " " + ref[:60])
    year = str((msg.get("issued") or msg.get("published") or {}).get("date-parts", [[None]])[0][0])
    year_ok = year in ref
    vol, page = msg.get("volume", ""), (msg.get("page", "") or msg.get("article-number", "")).split("-")[0]
    vol_ok = (not vol) or vol in ref
    page_ok = (not page) or page in ref
    upd = [f"{u.get('type')}:{u.get('DOI')}" for u in msg.get("updated-by", [])] if "updated-by" in msg else []
    rel = msg.get("relation", {})
    notices = upd + [f"{k}:{v[0].get('id')}" for k, v in rel.items() if k in ("is-retracted-by", "has-correction", "is-corrected-by")]
    flags = [n for n, ok in (("title", tsim >= 0.85), ("author", author_ok), ("year", year_ok), ("volume", vol_ok),
                             ("page", page_ok)) if not ok]
    rows.append({"key": key, "doi": doi, "status": "MATCH" if not flags else "CHECK: " + ",".join(flags),
                 "crossref_title": title, "crossref_first_author": fam, "crossref_year": year,
                 "crossref_journal": (msg.get("container-title") or [""])[0], "crossref_volume": vol, "crossref_page": page,
                 "title_similarity": round(tsim, 3), "type": msg.get("type", msg.get("source", "")),
                 "notices": "; ".join(notices), "ref": ref})
    time.sleep(0.3)

with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: ["key", "status", "doi"].index(k) if k in ("key", "status", "doi") else 9))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['key']:18} {r['status']:28} {r.get('notices', '')} | {r.get('crossref_title', '')[:70]}")
