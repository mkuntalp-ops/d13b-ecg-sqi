"""Add page numbers (centred footer) and, optionally, continuous line numbers to a pandoc-written DOCX.

Scientific Reports asks for page and line numbers to help editors and reviewers (submission guidelines, checked
2026-11-19); pandoc's default DOCX has neither. Called by evaluation/export_manuscript.sh before the outputs are hashed.

Usage: python evaluation/docx_layout.py <file.docx> [--line-numbers]
The file is rewritten in place. Text and tables are not touched: the script extends pandoc's final section-properties
element, which holds only footnote settings, or adds one (A4, 2.54 cm margins, footer reference and, with
--line-numbers, continuous line numbering),
one footer part with a PAGE field, its relationship and its content-type override.
"""
import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

FOOTER_ID = "rIdPageNumberFooter"
FOOTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
          '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
          '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
          '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
          '<w:r><w:t>1</w:t></w:r>'
          '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


def sect_pr(line_numbers: bool, footnote_pr: str = "") -> str:
    # child order follows CT_SectPr: footerReference, footnotePr, pgSz, pgMar, lnNumType
    ln = '<w:lnNumType w:countBy="1" w:restart="continuous"/>' if line_numbers else ""
    return (f'<w:sectPr><w:footerReference w:type="default" r:id="{FOOTER_ID}"/>{footnote_pr}'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" '
            f'w:gutter="0"/>{ln}</w:sectPr>')


def add_layout(path: Path, line_numbers: bool) -> None:
    with zipfile.ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = parts["word/document.xml"].decode("utf-8")
    if "word/footer_pagenumber.xml" in parts or FOOTER_ID in doc:
        raise ValueError(f"{path}: page-number footer already present")
    if doc.count("</w:body>") != 1 or 'xmlns:r="' not in doc:
        raise ValueError(f"{path}: unexpected document.xml structure")
    # pandoc writes one final sectPr holding only footnote properties; keep those and add the layout around them
    found = re.findall(r"<w:sectPr>(.*?)</w:sectPr>", doc, flags=re.S)
    if len(found) > 1 or doc.count("<w:sectPr") != len(found):
        raise ValueError(f"{path}: expected at most one plain sectPr")
    if found:
        inner = found[0]
        fn = re.findall(r"<w:footnotePr>.*?</w:footnotePr>", inner, flags=re.S)
        if len(fn) > 1 or re.sub(r"<w:footnotePr>.*?</w:footnotePr>", "", inner, flags=re.S).strip():
            raise ValueError(f"{path}: sectPr holds more than footnote properties; refusing to rewrite it")
        new = sect_pr(line_numbers, fn[0] if fn else "")
        doc = doc.replace(f"<w:sectPr>{inner}</w:sectPr>", new)
        if not doc.rstrip().endswith(new + "</w:body></w:document>") and not re.search(re.escape(new) + r"\s*</w:body>", doc):
            raise ValueError(f"{path}: sectPr is not the last element of the body")
    else:
        doc = doc.replace("</w:body>", sect_pr(line_numbers) + "</w:body>")
    parts["word/document.xml"] = doc.encode("utf-8")

    rels = parts["word/_rels/document.xml.rels"].decode("utf-8")
    if rels.count("</Relationships>") != 1 or FOOTER_ID in rels:
        raise ValueError(f"{path}: unexpected relationships part")
    rels = rels.replace("</Relationships>",
                        '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
                        f'Id="{FOOTER_ID}" Target="footer_pagenumber.xml" /></Relationships>')
    parts["word/_rels/document.xml.rels"] = rels.encode("utf-8")

    ct = parts["[Content_Types].xml"].decode("utf-8")
    if ct.count("</Types>") != 1:
        raise ValueError(f"{path}: unexpected content types part")
    ct = ct.replace("</Types>", '<Override PartName="/word/footer_pagenumber.xml" ContentType="application/'
                                'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml" /></Types>')
    parts["[Content_Types].xml"] = ct.encode("utf-8")
    parts["word/footer_pagenumber.xml"] = FOOTER.encode("utf-8")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=path.parent) as tmp:
        tmp_path = Path(tmp.name)
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml first, as Word expects
        for n in ["[Content_Types].xml"] + [n for n in parts if n != "[Content_Types].xml"]:
            z.writestr(n, parts[n])
    shutil.move(tmp_path, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("docx", type=Path)
    ap.add_argument("--line-numbers", action="store_true")
    a = ap.parse_args()
    add_layout(a.docx, a.line_numbers)
    print(f"[docx layout] {a.docx.name}: page numbers{' and line numbers' if a.line_numbers else ''}")


if __name__ == "__main__":
    main()
