#!/usr/bin/env python3
"""Compare OCR results from Apple Vision, PaddleOCR and EasyOCR.

Usage:
  python compare_confidence.py \\
    --apple outputs/apple.json \\
    --paddle outputs/paddle.json \\
    --easy outputs/easy.json \\
    --output outputs/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        print(f"Error: file not found → {p}", file=sys.stderr)
        sys.exit(1)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def _format_row(cols: list[str], widths: list[int]) -> str:
    return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cols, widths)) + " │"


def _format_sep(widths: list[int]) -> str:
    return "├─" + "─┼─".join("─" * w for w in widths) + "─┤"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Apple Vision / PaddleOCR / EasyOCR JSON outputs"
    )
    parser.add_argument("--apple", required=True, help="Apple Vision JSON")
    parser.add_argument("--paddle", required=True, help="PaddleOCR JSON")
    parser.add_argument("--easy", required=True, help="EasyOCR JSON")
    parser.add_argument("--output", required=True, help="Output comparison JSON")
    args = parser.parse_args()

    # Load ------------------------------------------------------------------
    apple = _load(args.apple)
    paddle = _load(args.paddle)
    easy = _load(args.easy)

    engines: dict[str, dict] = {
        "apple_vision": apple,
        "paddleocr": paddle,
        "easyocr": easy,
    }

    # ------------------------------------------------------------------
    # Overall summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  OCR Engine Comparison — Overall Summary")
    print("=" * 72)

    header = ["Engine", "Pages", "Total Lines", "Avg Confidence", "Total Time (s)"]
    widths = [16, 7, 12, 15, 16]
    print(_format_row(header, widths))
    print(_format_sep(widths))

    best_engine = ""
    best_conf = -1.0

    for name, data in engines.items():
        s = data.get("summary", {})
        pages = s.get("pages", 0)
        lines = s.get("total_lines", 0)
        conf = s.get("avg_confidence", 0)
        elapsed = s.get("total_elapsed_seconds", 0)
        row = [name, str(pages), str(lines), f"{conf:.4f}", f"{elapsed:.2f}"]
        print(_format_row(row, widths))
        if conf > best_conf:
            best_conf = conf
            best_engine = name

    print("-" * 72)
    print(f"  Highest avg confidence: {best_engine} ({best_conf:.4f})")
    print("  NOTE: confidence scores are NOT directly comparable across engines.")
    print()

    # ------------------------------------------------------------------
    # Per-page comparison table
    # ------------------------------------------------------------------
    max_pages = max(
        len(apple.get("pages", [])),
        len(paddle.get("pages", [])),
        len(easy.get("pages", [])),
    )

    if max_pages > 0:
        print("=" * 96)
        print("  Per-Page Comparison")
        print("=" * 96)

        ph = ["Page", "Engine", "Lines", "Avg Conf", "Time (s)"]
        pw = [7, 16, 8, 12, 10]
        print(_format_row(ph, pw))
        print(_format_sep(pw))

        for page_idx in range(max_pages):
            for eng_name, data in engines.items():
                pages = data.get("pages", [])
                if page_idx < len(pages):
                    p = pages[page_idx]
                    row = [
                        str(p.get("page_number", "?")),
                        eng_name,
                        str(p.get("line_count", 0)),
                        f"{p.get('avg_confidence', 0):.4f}",
                        f"{p.get('elapsed_seconds', 0):.2f}",
                    ]
                else:
                    row = [str(page_idx + 1), eng_name, "—", "—", "—"]
                print(_format_row(row, pw))
            if page_idx < max_pages - 1:
                print(_format_sep(pw))
        print()

    # ------------------------------------------------------------------
    # Build and write comparison JSON
    # ------------------------------------------------------------------
    comparison: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "apple": str(Path(args.apple).resolve()),
            "paddle": str(Path(args.paddle).resolve()),
            "easy": str(Path(args.easy).resolve()),
        },
        "engine_summaries": {
            name: data.get("summary", {}) for name, data in engines.items()
        },
        "winner": {
            "engine": best_engine,
            "avg_confidence": best_conf,
            "caveat": "Confidence metrics are engine-specific and not directly comparable. "
                       "Manual inspection of recognition quality is recommended.",
        },
        "per_page_comparison": [],
    }

    for page_idx in range(max_pages):
        page_entry: dict[str, Any] = {"page_number": page_idx + 1, "engines": {}}
        for eng_name, data in engines.items():
            pages = data.get("pages", [])
            if page_idx < len(pages):
                p = pages[page_idx]
                page_entry["engines"][eng_name] = {
                    "line_count": p.get("line_count", 0),
                    "avg_confidence": p.get("avg_confidence", 0),
                    "elapsed_seconds": p.get("elapsed_seconds", 0),
                }
            else:
                page_entry["engines"][eng_name] = None
        comparison["per_page_comparison"].append(page_entry)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)

    print(f"Comparison written → {output_path.resolve()}")


if __name__ == "__main__":
    main()
