#!/usr/bin/env python3
"""Detect yellow highlighter regions in a PDF rendered to page images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect yellow highlighter regions in PDF pages.")
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", required=True, help="Path to highlight JSON")
    parser.add_argument("--dpi", type=int, default=240, help="Render DPI (default: 240)")
    parser.add_argument("--max-pages", type=int, default=0, help="Max pages; 0 = all")
    parser.add_argument("--min-area", type=int, default=120, help="Minimum contour area in pixels")
    parser.add_argument("--h-min", type=int, default=18, help="HSV hue min for yellow")
    parser.add_argument("--h-max", type=int, default=45, help="HSV hue max for yellow")
    parser.add_argument("--s-min", type=int, default=45, help="HSV saturation min")
    parser.add_argument("--v-min", type=int, default=110, help="HSV value min")
    parser.add_argument("--morph-kernel", type=int, default=3, help="Morphology kernel size; 0 disables morphology")
    parser.add_argument("--close-iterations", type=int, default=1, help="Morphological close iterations")
    parser.add_argument("--open-iterations", type=int, default=1, help="Morphological open iterations")
    return parser.parse_args()


def pixmap_to_rgb(pix: fitz.Pixmap) -> np.ndarray:
    channels = pix.n
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, channels)
    if pix.alpha and channels > 1:
        arr = arr[:, :, : channels - 1]
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    return arr[:, :, :3]


def detect_regions(
    rgb: np.ndarray,
    *,
    min_area: int,
    h_min: int,
    h_max: int,
    s_min: int,
    v_min: int,
    morph_kernel: int,
    close_iterations: int,
    open_iterations: int,
) -> list[dict[str, Any]]:
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if morph_kernel > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        if close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        if open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        roi_mask = mask[y : y + h, x : x + w]
        fill_ratio = float((roi_mask > 0).mean())
        confidence = max(0.0, min(1.0, 0.45 + fill_ratio * 0.55))
        regions.append(
            {
                "bbox": {"x": round(float(x), 1), "y": round(float(y), 1), "w": round(float(w), 1), "h": round(float(h), 1)},
                "area": round(area, 1),
                "fill_ratio": round(fill_ratio, 4),
                "confidence": round(confidence, 4),
            }
        )

    regions.sort(key=lambda region: (region["bbox"]["y"], region["bbox"]["x"]))
    return regions


def main() -> None:
    args = parse_args()
    input_pdf = Path(args.input).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_pdf.is_file():
        print(f"Error: input file not found -> {input_pdf}", file=sys.stderr)
        sys.exit(1)

    started = time.time()
    pages: list[dict[str, Any]] = []
    total_regions = 0
    with fitz.open(str(input_pdf)) as document:
        max_pages = len(document) if args.max_pages <= 0 else min(args.max_pages, len(document))
        for page_index in range(max_pages):
            page_started = time.time()
            pix = document[page_index].get_pixmap(dpi=args.dpi, alpha=False)
            rgb = pixmap_to_rgb(pix)
            regions = detect_regions(
                rgb,
                min_area=args.min_area,
                h_min=args.h_min,
                h_max=args.h_max,
                s_min=args.s_min,
                v_min=args.v_min,
                morph_kernel=args.morph_kernel,
                close_iterations=args.close_iterations,
                open_iterations=args.open_iterations,
            )
            total_regions += len(regions)
            pages.append(
                {
                    "page_number": page_index + 1,
                    "width": pix.width,
                    "height": pix.height,
                    "elapsed_seconds": round(time.time() - page_started, 3),
                    "region_count": len(regions),
                    "regions": regions,
                }
            )
            print(f"  Page {page_index + 1}/{max_pages}: {len(regions)} yellow region(s)")

    result = {
        "engine": "yellow_hsv",
        "input_pdf": str(input_pdf),
        "dpi": args.dpi,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "min_area": args.min_area,
            "h_min": args.h_min,
            "h_max": args.h_max,
            "s_min": args.s_min,
            "v_min": args.v_min,
            "morph_kernel": args.morph_kernel,
            "close_iterations": args.close_iterations,
            "open_iterations": args.open_iterations,
        },
        "summary": {
            "pages": len(pages),
            "total_regions": total_regions,
            "total_elapsed_seconds": round(time.time() - started, 3),
        },
        "pages": pages,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"Output -> {output_path}")


if __name__ == "__main__":
    main()
