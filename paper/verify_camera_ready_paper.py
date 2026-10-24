"""Structural and content checks for the final camera-ready paper."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import fitz
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OLD = Path(r"D:\Paper_KSE_before_edit.pdf")
PDF = ROOT / "output" / "paper" / "KSE2026_CameraReady_Revised_6Pages.pdf"
DOCX = ROOT / "output" / "paper" / "KSE2026_CameraReady_Revised_6Pages.docx"


def main() -> None:
    for path in (OLD, PDF, DOCX):
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing paper artifact: {path}")

    reader = PdfReader(PDF)
    if len(reader.pages) != 6:
        raise AssertionError(f"Expected exactly 6 KSE pages, found {len(reader.pages)}")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized = " ".join(text.split())

    required = [
        "Granger-Causal Return Spillovers",
        "1,959 training forecasts",
        "modified DM=0.476",
        "Jaccard=0.059",
        "VNI->GOLD",
        "VIX->VNI",
        "65% and a receiver in 35%",
        "20 April 2020",
        "2,383 effective development observations",
        "9.18% Extreme",
        "not significantly outperform LSTM",
        "This work was supported by Vietnam National University, Hanoi, under research project QG.26.68.",
        "NGC TEST PERFORMANCE ACROSS ALL TARGET RETURNS",
        "max_abs_repeat_diff=0.0",
        "0.0063",
    ]
    for phrase in required:
        if phrase not in normalized:
            raise AssertionError(f"Required camera-ready statement not found: {phrase}")

    forbidden = [
        "140-64-32-1",
        "1,979 training",
        "59.00",
        "5.399",
        "2,764 post-burn-in",
        "VNI is mainly a receiver",
        "Completing the causal wavelet rerun",
    ]
    for phrase in forbidden:
        if phrase in normalized:
            raise AssertionError(f"Legacy claim leaked into final paper: {phrase}")

    rendered = fitz.open(PDF)
    for page in rendered:
        width = float(page.rect.width)
        height = float(page.rect.height)
        if not (590 <= width <= 600 and 840 <= height <= 845):
            raise AssertionError(f"Unexpected page geometry: {width} x {height}")
        if not page.get_text("text").strip():
            raise AssertionError("Blank PDF page detected")

    font_names = set()
    for page in rendered:
        for font in page.get_fonts(full=True):
            base_font = font[3]
            font_type = font[2]
            font_names.add(base_font)
            if font_type == "Type3":
                raise AssertionError(f"Bitmap/Type3 font found: {base_font}")
    if not any("TimesNewRoman" in name for name in font_names):
        raise AssertionError(f"Times New Roman not found in embedded PDF fonts: {sorted(font_names)}")
    if not any("CambriaMath" in name for name in font_names):
        raise AssertionError(f"Cambria Math not found in embedded PDF fonts: {sorted(font_names)}")

    with ZipFile(DOCX) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    if "[[MATH:" in document_xml or "[[/MATH]]" in document_xml:
        raise AssertionError("Unconverted equation marker remains in DOCX")
    if document_xml.count("<m:oMath") < 6:
        raise AssertionError("Expected at least six native Word equations")
    if document_xml.count("<w:insideV") < 11:
        raise AssertionError("Not all paper tables have inside vertical borders")

    print(f"PASS: exactly {len(reader.pages)} A4 pages (KSE limit: 6)")
    print("PASS: required revised results are present")
    print("PASS: selected legacy numbers/claims are absent")
    print("PASS: native Word equations, embedded fonts, and vertical table borders")
    print(f"PASS: original paper preserved at {OLD}")


if __name__ == "__main__":
    main()
