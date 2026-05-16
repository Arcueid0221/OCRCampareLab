#!/usr/bin/env python3
"""Apple Vision OCR via PyObjC, macOS only.

The script intentionally outputs raw OCR lines plus geometry. Notebook-specific
parsing belongs in postprocess_notes.py so this file stays reusable.

Usage:
  python apple_ocr.py --input ../单词.pdf --output outputs/apple.json --max-pages 3
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz


def _cg_image_from_path(image_path: str):
    """Create a CGImage from a file path using ImageIO."""
    import Quartz
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(image_path))
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    if source is None:
        raise OSError(f"Cannot create CGImageSource from {image_path}")
    cg = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg is None:
        raise OSError(f"Cannot decode image at {image_path}")
    return cg


def _image_size(cg_image) -> tuple[int, int]:
    import Quartz

    return Quartz.CGImageGetWidth(cg_image), Quartz.CGImageGetHeight(cg_image)


def _vision_level(name: str):
    import Vision

    if name == "fast":
        return Vision.VNRequestTextRecognitionLevelFast
    return Vision.VNRequestTextRecognitionLevelAccurate


def _candidate_to_dict(candidate: object) -> dict[str, Any]:
    text = str(candidate.string()).strip()
    confidence = float(candidate.confidence())
    return {
        "text": text,
        "confidence": round(confidence, 4),
    }


def _run_vision_ocr(
    cg_image: object,
    *,
    languages: list[str],
    recognition_level: str,
    language_correction: bool,
    min_confidence: float,
    source_pass: str,
) -> list[dict[str, Any]]:
    """Run VNRecognizeTextRequest and return normalized line dictionaries."""
    import Vision

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cg_image, None
    )

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLanguages_(languages)
    request.setRecognitionLevel_(_vision_level(recognition_level))
    request.setUsesLanguageCorrection_(language_correction)

    success = handler.performRequests_error_([request], None)
    if not success:
        raise RuntimeError("VNImageRequestHandler returned error")

    observations = request.results()
    if observations is None:
        return []

    width, height = _image_size(cg_image)
    lines: list[dict[str, Any]] = []

    for observation_index, obs in enumerate(observations):
        candidates: list[dict[str, Any]] = []
        try:
            top_candidates = obs.topCandidates_(3) or []
            candidates = [
                _candidate_to_dict(candidate)
                for candidate in top_candidates
                if str(candidate.string()).strip()
            ]
        except Exception:
            candidates = []

        if candidates:
            text = candidates[0]["text"]
            confidence = float(candidates[0]["confidence"])
        else:
            text = str(obs.text()).strip()
            confidence = float(obs.confidence())

        if not text:
            continue

        bbox_norm = obs.boundingBox()
        x = bbox_norm.origin.x * width
        y = (1.0 - bbox_norm.origin.y - bbox_norm.size.height) * height
        w = bbox_norm.size.width * width
        h = bbox_norm.size.height * height

        lines.append(
            {
                "index": observation_index,
                "text": text,
                "confidence": round(confidence, 4),
                "is_low_confidence": confidence < min_confidence,
                "source_pass": source_pass,
                "candidates": candidates,
                "bbox": {
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "w": round(w, 1),
                    "h": round(h, 1),
                },
                "polygon": [
                    [round(x, 1), round(y, 1)],
                    [round(x + w, 1), round(y, 1)],
                    [round(x + w, 1), round(y + h, 1)],
                    [round(x, 1), round(y + h, 1)],
                ],
            }
        )

    return lines


def _pixmap_stats(pix: fitz.Pixmap) -> dict[str, Any]:
    """Cheap pre-OCR image stats. These are hints, not page classifications."""
    try:
        import numpy as np
    except ImportError:
        return {"available": False}

    channels = pix.n
    if channels <= 0:
        return {"available": False}

    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    if arr.size == 0:
        return {"available": False}

    try:
        arr = arr.reshape(pix.height, pix.width, channels)
    except ValueError:
        return {"available": False}

    if pix.alpha and channels > 1:
        arr = arr[:, :, : channels - 1]

    if arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype("float32")
        gray = rgb.mean(axis=2)
    else:
        gray = arr[:, :, 0].astype("float32")

    dark_ratio = float((gray < 96).mean())
    light_ratio = float((gray > 232).mean())
    stddev = float(gray.std())
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        vertical_edges = np.abs(np.diff(gray, axis=0)).mean()
        horizontal_edges = np.abs(np.diff(gray, axis=1)).mean()
        edge_score = float((vertical_edges + horizontal_edges) / 2.0)
    else:
        edge_score = 0.0

    # A real blank white page and a flat colored cover both have little ink-like
    # contrast. OCR still gets the final say through no_ocr_lines.
    likely_blank_image = dark_ratio < 0.002 and edge_score < 1.5
    return {
        "available": True,
        "dark_ratio": round(dark_ratio, 6),
        "light_ratio": round(light_ratio, 6),
        "gray_stddev": round(stddev, 3),
        "edge_score": round(edge_score, 3),
        "likely_blank_image": bool(likely_blank_image),
    }


def _blank_status(lines: list[dict[str, Any]], stats: dict[str, Any]) -> tuple[bool, str | None]:
    if not lines:
        return True, "no_ocr_lines"
    if stats.get("likely_blank_image") and len(lines) <= 1:
        return True, "likely_blank_image"
    return False, None


def _page_avg_confidence(lines: list[dict[str, Any]]) -> float:
    if not lines:
        return 0.0
    return round(sum(float(line["confidence"]) for line in lines) / len(lines), 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apple Vision OCR, extract raw text lines from a PDF on macOS."
    )
    parser.add_argument("--input", required=True, help="Path to input PDF")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--dpi", type=int, default=240, help="Render DPI (default: 240)")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages to process; 0 = all (default: 0)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en-US"],
        help="Recognition languages, ordered by priority (default: en-US)",
    )
    parser.add_argument(
        "--recognition-level",
        choices=["accurate", "fast"],
        default="accurate",
        help="Apple Vision recognition level (default: accurate)",
    )
    parser.add_argument(
        "--language-correction",
        choices=["on", "off"],
        default="off",
        help="Apple language correction (default: off)",
    )
    parser.add_argument(
        "--supplemental-chinese-pass",
        choices=["on", "off"],
        default="off",
        help="Compatibility option. Keep off for English-only word extraction (default: off)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Mark lines below this confidence as low confidence; never filters them.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_pdf = Path(args.input).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_pdf.is_file():
        print(f"Error: input file not found -> {input_pdf}", file=sys.stderr)
        sys.exit(1)

    try:
        import Quartz  # noqa: F401
        import Vision  # noqa: F401
    except ImportError as exc:
        print("Error: this script requires macOS with PyObjC Vision/Quartz.", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        sys.exit(1)

    doc = fitz.open(str(input_pdf))
    total_pages = len(doc)
    max_pages = total_pages if args.max_pages <= 0 else min(args.max_pages, total_pages)
    max_digits = len(str(max_pages))

    t0 = time.time()
    page_results: list[dict[str, Any]] = []
    total_lines = 0
    total_conf_sum = 0.0
    blank_pages = 0

    for page_idx in range(max_pages):
        t_page = time.time()
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=args.dpi, alpha=False)
        image_stats = _pixmap_stats(pix)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            pix.save(str(tmp_path))
            cg_image = _cg_image_from_path(str(tmp_path))
            lines = _run_vision_ocr(
                cg_image,
                languages=args.languages,
                recognition_level=args.recognition_level,
                language_correction=args.language_correction == "on",
                min_confidence=args.min_confidence,
                source_pass="primary",
            )
            img_w, img_h = _image_size(cg_image)
        finally:
            tmp_path.unlink(missing_ok=True)

        elapsed = time.time() - t_page
        is_blank, blank_reason = _blank_status(lines, image_stats)
        if is_blank:
            blank_pages += 1

        line_count = len(lines)
        avg_confidence = _page_avg_confidence(lines)
        page_results.append(
            {
                "page_number": page_idx + 1,
                "width": img_w,
                "height": img_h,
                "elapsed_seconds": round(elapsed, 2),
                "line_count": line_count,
                "avg_confidence": avg_confidence,
                "is_blank": is_blank,
                "blank_reason": blank_reason,
                "image_stats": image_stats,
                "lines": lines,
            }
        )

        total_lines += line_count
        total_conf_sum += sum(float(line["confidence"]) for line in lines)
        marker = "blank" if is_blank else "text"
        print(
            f"  Page {page_idx + 1:>{max_digits}d}/{max_pages}: "
            f"{line_count:>4d} lines, avg conf {avg_confidence:.3f}, "
            f"{elapsed:.1f}s, {marker}"
        )

    doc.close()
    total_elapsed = round(time.time() - t0, 2)
    overall_confidence = round(total_conf_sum / total_lines, 4) if total_lines else 0.0

    result = {
        "engine": "apple_vision",
        "input_pdf": str(input_pdf),
        "dpi": args.dpi,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "languages": args.languages,
            "recognition_level": args.recognition_level,
            "language_correction": args.language_correction,
            "supplemental_chinese_pass": args.supplemental_chinese_pass,
            "min_confidence": args.min_confidence,
        },
        "summary": {
            "pages": max_pages,
            "blank_pages": blank_pages,
            "total_lines": total_lines,
            "avg_confidence": overall_confidence,
            "total_elapsed_seconds": total_elapsed,
        },
        "pages": page_results,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"\nDone. {max_pages} page(s), {total_lines} lines, {total_elapsed}s")
    print(f"Output -> {output_path}")


if __name__ == "__main__":
    main()
