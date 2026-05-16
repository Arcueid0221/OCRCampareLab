#!/usr/bin/env python3
"""EasyOCR — Chinese + English text recognition.

Usage:
  python easy_ocr.py --input ../单词.pdf --output outputs/easy.json --dpi 150 --max-pages 3
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import fitz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EasyOCR — Chinese / English OCR via EasyOCR"
    )
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    parser.add_argument(
        "--max-pages", type=int, default=0,
        help="Max pages to process; 0 = all (default: 0)"
    )
    parser.add_argument(
        "--languages", nargs="+", default=["ch_sim", "en"],
        help="EasyOCR language codes (default: ch_sim en)"
    )
    parser.add_argument(
        "--canvas-size", type=int, default=1920,
        help="EasyOCR max canvas dimension (default: 1920)"
    )
    args = parser.parse_args()

    input_pdf = Path(args.input).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_pdf.is_file():
        print(f"Error: input file not found → {input_pdf}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Lazy import so --help is fast
    # ------------------------------------------------------------------
    try:
        import easyocr  # type: ignore[import-untyped]
    except ImportError as exc:
        print("Error: EasyOCR not installed. Run:  pip install easyocr", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Initialising EasyOCR (lang={args.languages}, canvas={args.canvas_size}) …")
    reader = easyocr.Reader(args.languages, gpu=False)  # CPU mode

    doc = fitz.open(str(input_pdf))
    total_pages = len(doc)
    max_pages = total_pages if args.max_pages <= 0 else min(args.max_pages, total_pages)

    t0 = time.time()
    page_results: list[dict] = []
    total_lines = 0
    total_conf_sum = 0.0

    max_digits = len(str(max_pages))

    for page_idx in range(max_pages):
        t_page = time.time()
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=args.dpi)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            pix.save(str(tmp_path))
            raw = reader.readtext(str(tmp_path), canvas_size=args.canvas_size)
        finally:
            tmp_path.unlink(missing_ok=True)

        # EasyOCR returns: [([[x1,y1],...,[x4,y4]], "text", confidence), ...]
        lines: list[dict] = []
        for bbox_pts, text, confidence in raw:
            x = float(bbox_pts[0][0])
            y = float(bbox_pts[0][1])
            w = float(bbox_pts[2][0]) - x
            h = float(bbox_pts[2][1]) - y

            lines.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": {"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)},
                "polygon": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in bbox_pts],
            })

        elapsed = time.time() - t_page
        lc = len(lines)
        avg_conf = round(sum(ln["confidence"] for ln in lines) / lc, 4) if lc else 0.0

        page_results.append({
            "page_number": page_idx + 1,
            "width": pix.width,
            "height": pix.height,
            "elapsed_seconds": round(elapsed, 2),
            "line_count": lc,
            "avg_confidence": avg_conf,
            "lines": lines,
        })

        total_lines += lc
        total_conf_sum += sum(ln["confidence"] for ln in lines)
        print(
            f"  Page {page_idx + 1:>{max_digits}d}/{max_pages}: "
            f"{lc:>4d} lines, avg conf {avg_conf:.3f}, {elapsed:.1f}s"
        )

    doc.close()
    total_elapsed = round(time.time() - t0, 2)

    result = {
        "engine": "easyocr",
        "input_pdf": str(input_pdf),
        "dpi": args.dpi,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "pages": max_pages,
            "total_lines": total_lines,
            "avg_confidence": round(total_conf_sum / total_lines, 4) if total_lines else 0.0,
            "total_elapsed_seconds": total_elapsed,
        },
        "pages": page_results,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"\nDone. {max_pages} page(s), {total_lines} lines, {total_elapsed}s")
    print(f"Output → {output_path}")


if __name__ == "__main__":
    main()
