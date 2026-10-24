"""Export a DOCX through installed Microsoft Word and rasterize every page."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
import win32com.client


MATH_PREFIX = "[[MATH:"
MATH_SUFFIX = "[[/MATH]]"


def build_professional_equations(document) -> int:
    """Convert marked UnicodeMath paragraphs into editable Word equations."""
    converted = 0
    # Materialize the paragraph list first because replacing ranges mutates it.
    paragraphs = [document.Paragraphs(i) for i in range(1, document.Paragraphs.Count + 1)]
    for paragraph in paragraphs:
        raw = paragraph.Range.Text.rstrip("\r\x07")
        if not raw.startswith(MATH_PREFIX) or MATH_SUFFIX not in raw:
            continue
        header, tail = raw.split("]]", 1)
        number = header[len(MATH_PREFIX) :]
        linear, _ = tail.split(MATH_SUFFIX, 1)
        start = paragraph.Range.Start
        paragraph.Range.Text = f"{linear}\t({number})\r"
        equation_range = document.Range(Start=start, End=start + len(linear))
        document.OMaths.Add(equation_range)
        equation_range.OMaths(1).BuildUp()
        converted += 1
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("render_dir", type=Path)
    args = parser.parse_args()

    docx = args.docx.resolve()
    pdf = args.pdf.resolve()
    render_dir = args.render_dir.resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(docx), ReadOnly=False, AddToRecentFiles=False)
        converted = build_professional_equations(document)
        if converted:
            document.Save()
        document.Repaginate()
        pages = int(document.ComputeStatistics(2))  # wdStatisticPages
        document.ExportAsFixedFormat(
            OutputFileName=str(pdf),
            ExportFormat=17,  # wdExportFormatPDF
            OpenAfterExport=False,
            OptimizeFor=0,
            Range=0,
            Item=0,
            IncludeDocProps=True,
            KeepIRM=True,
            CreateBookmarks=0,
            DocStructureTags=True,
            BitmapMissingFonts=False,
            UseISO19005_1=False,
        )
    finally:
        if document is not None:
            document.Close(SaveChanges=False)
        word.Quit()

    rendered = fitz.open(pdf)
    for i, page in enumerate(rendered):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        (render_dir / f"page-{i + 1}.png").write_bytes(pixmap.tobytes("png"))
    print(f"Professional equations converted: {converted}")
    print(f"Word pages: {pages}; PDF pages: {len(rendered)}")
    print(f"Saved {pdf}")
    print(f"Rendered {render_dir}")


if __name__ == "__main__":
    main()
