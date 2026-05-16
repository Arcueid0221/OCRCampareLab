#!/usr/bin/env python3
"""Extract ordered English vocabulary records from Apple Vision OCR JSON.

The output intentionally contains only dates, page/column order, corrected
ordinals, English words, OCR geometry, highlight flags, and review metadata.
Definitions are handled later by dictionary_cache.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


ENGLISH_RE = re.compile(r"[A-Za-z]")
DATE_FULL_RE = re.compile(
    r"(?P<year>20\d{2})\s*[-/.]\s*(?P<month>\d{1,2})\s*[-/.]\s*(?P<day>\d{1,2})"
)
DATE_CN_RE = re.compile(
    r"(?:(?P<year>20\d{2})\s*\u5e74\s*)?"
    r"(?P<month>\d{1,2})\s*\u6708\s*(?P<day>\d{1,2})\s*\u65e5"
)
DATE_SHORT_RE = re.compile(r"(?<!\d)(?P<month>\d{1,2})\s*[./]\s*(?P<day>\d{1,2})(?!\d)")
ENTRY_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?P<ordinal_num>\d{1,3})\s*[\.\),:;_\-]?\s*"
    r"|(?P<ordinal_ocr>[A-Za-z0-9]{1,3})\s*[\.\),:;_\-]\s*"
    r")?"
    r"(?P<word>[A-Za-z][A-Za-z'\u2019\-]{1,})"
    r"\b"
)
WORD_STOPLIST = {
    "adj",
    "adv",
    "n",
    "v",
    "vi",
    "vt",
    "prep",
    "conj",
    "pron",
    "aux",
    "int",
}
OCR_DIGIT_MAP = str.maketrans(
    {
        "o": "0",
        "O": "0",
        "l": "1",
        "I": "1",
        "|": "1",
        "s": "5",
        "S": "5",
        "b": "6",
        "B": "6",
        "g": "9",
        "q": "9",
        "z": "2",
        "Z": "2",
    }
)


@dataclass(frozen=True)
class Line:
    index: int
    text: str
    confidence: float
    bbox: dict[str, float]
    candidates: list[dict[str, Any]]

    @property
    def x(self) -> float:
        return float(self.bbox.get("x", 0.0))

    @property
    def y(self) -> float:
        return float(self.bbox.get("y", 0.0))

    @property
    def w(self) -> float:
        return float(self.bbox.get("w", 0.0))

    @property
    def h(self) -> float:
        return float(self.bbox.get("h", 0.0))

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process Apple OCR JSON into ordered English vocabulary records."
    )
    parser.add_argument("--input", required=True, help="Path to apple_ocr.py JSON output")
    parser.add_argument("--output", required=True, help="Path to words JSON output")
    parser.add_argument("--highlight-json", help="Optional output from highlight_detect.py")
    parser.add_argument(
        "--default-year",
        type=int,
        default=datetime.now().year,
        help="Year used for month/day date formats (default: current year)",
    )
    parser.add_argument(
        "--allow-ambiguous-short-date",
        action="store_true",
        help="Treat dates such as 5.5 as valid; by default day < 10 stays ambiguous.",
    )
    parser.add_argument(
        "--column-split",
        type=float,
        default=0.5,
        help="Page-width ratio used to split left/right columns.",
    )
    parser.add_argument(
        "--min-word-confidence",
        type=float,
        default=0.35,
        help="Mark OCR word rows below this confidence as needs_review.",
    )
    parser.add_argument(
        "--word-zone-ratio",
        type=float,
        default=0.18,
        help="Only accept words whose x starts within this page-width ratio from the column start.",
    )
    parser.add_argument(
        "--include-missing-ordinal",
        action="store_true",
        help="Include English lines without raw OCR ordinal in the main word sequence.",
    )
    parser.add_argument(
        "--highlight-word-overlap",
        type=float,
        default=0.18,
        help="Highlight hit threshold: intersection / word bbox area.",
    )
    parser.add_argument(
        "--highlight-region-overlap",
        type=float,
        default=0.35,
        help="Highlight hit threshold: intersection / yellow region area.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        print(f"Error: file not found -> {path}", file=sys.stderr)
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def make_lines(page: dict[str, Any]) -> list[Line]:
    lines: list[Line] = []
    for fallback_index, raw in enumerate(page.get("lines") or []):
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        bbox = raw.get("bbox") or {}
        lines.append(
            Line(
                index=int(raw.get("index", fallback_index)),
                text=text,
                confidence=float(raw.get("confidence") or 0.0),
                bbox={
                    "x": float(bbox.get("x", 0.0)),
                    "y": float(bbox.get("y", 0.0)),
                    "w": float(bbox.get("w", 0.0)),
                    "h": float(bbox.get("h", 0.0)),
                },
                candidates=list(raw.get("candidates") or []),
            )
        )
    return lines


def normalize_text(text: str) -> str:
    return (
        text.replace("\u2019", "'")
        .replace("\uff0e", ".")
        .replace("\u3002", ".")
        .replace("\uff0c", ",")
        .replace("\u3001", ".")
        .replace("\uff1a", ":")
        .replace("\uff09", ")")
        .strip()
    )


def normalize_word(raw_word: str) -> str:
    word = raw_word.replace("\u2019", "'").strip().strip(".,:;_-()[]{}")
    return re.sub(r"[^A-Za-z'\-]", "", word).lower()


def valid_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def candidate_texts(line: Line) -> list[str]:
    texts = [line.text]
    for candidate in line.candidates:
        text = str(candidate.get("text") or "").strip()
        if text and text not in texts:
            texts.append(text)
    return texts


def find_page_date(
    page: dict[str, Any],
    lines: list[Line],
    *,
    default_year: int,
    allow_ambiguous_short_date: bool,
) -> tuple[str | None, set[int], list[str]]:
    height = float(page.get("height") or 1.0)
    top_limit = max(140.0, height * 0.16)
    top_lines = sorted([line for line in lines if line.y <= top_limit], key=lambda line: (line.y, line.x))
    review_reasons: list[str] = []

    for line in top_lines:
        for text in candidate_texts(line):
            normalized = normalize_text(text)

            match = DATE_FULL_RE.search(normalized)
            if match:
                iso = valid_date(int(match.group("year")), int(match.group("month")), int(match.group("day")))
                if iso:
                    return iso, {line.index}, review_reasons

            match = DATE_CN_RE.search(normalized)
            if match:
                iso = valid_date(int(match.group("year") or default_year), int(match.group("month")), int(match.group("day")))
                if iso:
                    return iso, {line.index}, review_reasons

            match = DATE_SHORT_RE.search(normalized)
            if match:
                month = int(match.group("month"))
                day = int(match.group("day"))
                if day < 10 and not allow_ambiguous_short_date:
                    review_reasons.append(f"ambiguous_short_date:{month}.{day}")
                    continue
                iso = valid_date(default_year, month, day)
                if iso:
                    return iso, {line.index}, review_reasons

    return None, set(), dedupe(review_reasons)


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def split_columns(page: dict[str, Any], lines: list[Line], column_split: float) -> dict[str, list[Line]]:
    width = float(page.get("width") or 1.0)
    split_x = width * column_split
    columns = {"left": [], "right": []}
    for line in lines:
        columns["left" if line.cx < split_x else "right"].append(line)
    for column_lines in columns.values():
        column_lines.sort(key=lambda line: (line.y, line.x))
    return columns


def parse_ordinal(raw: str | None) -> int | None:
    if not raw:
        return None
    normalized = raw.strip().translate(OCR_DIGIT_MAP)
    normalized = re.sub(r"[^0-9]", "", normalized)
    if not normalized:
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def ordinal_confidence(raw_ordinal: str | None, parsed: int | None, corrected: int) -> float:
    if raw_ordinal is None or parsed is None:
        return 0.55
    if parsed == corrected:
        return 1.0
    if abs(parsed - corrected) <= 2:
        return 0.75
    return 0.35


def choose_corrected_ordinal(parsed: int | None, expected_next: int) -> int:
    if parsed is None or parsed <= 0:
        return expected_next
    if parsed < expected_next:
        return expected_next
    if parsed - expected_next <= 3:
        return parsed
    return expected_next


def merge_bbox(lines: list[Line]) -> dict[str, float]:
    min_x = min(line.x for line in lines)
    min_y = min(line.y for line in lines)
    max_x = max(line.x + line.w for line in lines)
    max_y = max(line.y + line.h for line in lines)
    return {
        "x": round(min_x, 1),
        "y": round(min_y, 1),
        "w": round(max_x - min_x, 1),
        "h": round(max_y - min_y, 1),
    }


def parse_word_line(line: Line) -> tuple[str | None, str | None, str | None]:
    candidates = candidate_texts(line)
    for text in candidates:
        normalized = normalize_text(text)
        match = ENTRY_RE.match(normalized)
        if not match:
            continue
        raw_ordinal = match.group("ordinal_num") or match.group("ordinal_ocr")
        word = normalize_word(match.group("word"))
        if len(word) <= 2 or word in WORD_STOPLIST:
            continue
        if raw_ordinal is None and line.confidence < 0.85:
            continue
        return raw_ordinal, match.group("word"), word
    return None, None, None


def unassigned_word_record(
    line: Line,
    *,
    page_number: int,
    column: str,
    reason: str,
) -> dict[str, Any] | None:
    raw_ordinal, word_raw, word_normalized = parse_word_line(line)
    if word_raw is None or word_normalized is None:
        return None
    return {
        "page_number": page_number,
        "column": column,
        "raw_ordinal": raw_ordinal,
        "word_raw": word_raw,
        "word_normalized": word_normalized,
        "ocr_confidence": round(line.confidence, 4),
        "bbox": merge_bbox([line]),
        "source_text": line.text,
        "line_indices": [line.index],
        "reason": reason,
    }


def bbox_area(bbox: dict[str, float]) -> float:
    return max(0.0, float(bbox.get("w", 0.0))) * max(0.0, float(bbox.get("h", 0.0)))


def intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    ax1 = float(a.get("x", 0.0))
    ay1 = float(a.get("y", 0.0))
    ax2 = ax1 + float(a.get("w", 0.0))
    ay2 = ay1 + float(a.get("h", 0.0))
    bx1 = float(b.get("x", 0.0))
    by1 = float(b.get("y", 0.0))
    bx2 = bx1 + float(b.get("w", 0.0))
    by2 = by1 + float(b.get("h", 0.0))
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def load_highlights(path: str | None) -> dict[int, list[dict[str, Any]]]:
    if not path:
        return {}
    data = load_json(Path(path))
    highlights: dict[int, list[dict[str, Any]]] = {}
    for page in data.get("pages") or []:
        page_number = int(page.get("page_number") or 0)
        highlights[page_number] = list(page.get("regions") or [])
    return highlights


def highlight_for_word(
    bbox: dict[str, float],
    regions: list[dict[str, Any]],
    *,
    word_overlap_threshold: float,
    region_overlap_threshold: float,
) -> tuple[bool, float, list[int]]:
    word_area = bbox_area(bbox)
    if word_area <= 0:
        return False, 0.0, []

    best_confidence = 0.0
    matched_indices: list[int] = []
    for index, region in enumerate(regions):
        region_bbox = region.get("bbox") or {}
        region_area = max(float(region.get("area") or 0.0), bbox_area(region_bbox))
        inter = intersection_area(bbox, region_bbox)
        if inter <= 0:
            continue
        word_overlap = inter / word_area
        region_overlap = inter / region_area if region_area > 0 else 0.0
        if word_overlap >= word_overlap_threshold or region_overlap >= region_overlap_threshold:
            matched_indices.append(index)
            best_confidence = max(best_confidence, float(region.get("confidence") or 0.0), word_overlap, region_overlap)

    return bool(matched_indices), round(min(best_confidence, 1.0), 4), matched_indices


def extract_page_words(
    page: dict[str, Any],
    lines: list[Line],
    *,
    note_date: str | None,
    date_line_indices: set[int],
    start_sequence: int,
    column_split: float,
    min_word_confidence: float,
    word_zone_ratio: float,
    include_missing_ordinal: bool,
    page_regions: list[dict[str, Any]],
    highlight_word_overlap: float,
    highlight_region_overlap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    columns = split_columns(
        page,
        [line for line in lines if line.index not in date_line_indices],
        column_split,
    )
    page_number = int(page.get("page_number") or 0)
    words: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    sequence = start_sequence
    expected_ordinal = 1
    page_width = float(page.get("width") or 1.0)
    split_x = page_width * column_split
    word_zone_width = page_width * word_zone_ratio

    for column in ("left", "right"):
        column_start = 0.0 if column == "left" else split_x
        for line in columns[column]:
            if line.x - column_start > word_zone_width:
                continue
            raw_ordinal, word_raw, word_normalized = parse_word_line(line)
            if word_raw is None or word_normalized is None:
                continue
            if raw_ordinal is None and not include_missing_ordinal:
                record = unassigned_word_record(
                    line,
                    page_number=page_number,
                    column=column,
                    reason="missing_raw_ordinal_excluded_from_sequence",
                )
                if record:
                    unassigned.append(record)
                continue

            parsed_ordinal = parse_ordinal(raw_ordinal)
            corrected_ordinal = choose_corrected_ordinal(parsed_ordinal, expected_ordinal)
            ord_confidence = ordinal_confidence(raw_ordinal, parsed_ordinal, corrected_ordinal)
            bbox = merge_bbox([line])
            is_highlighted, highlight_confidence, highlight_region_indices = highlight_for_word(
                bbox,
                page_regions,
                word_overlap_threshold=highlight_word_overlap,
                region_overlap_threshold=highlight_region_overlap,
            )

            review_reasons: list[str] = []
            if raw_ordinal is None:
                review_reasons.append("missing_raw_ordinal")
            if parsed_ordinal is not None and abs(parsed_ordinal - corrected_ordinal) > 3:
                review_reasons.append("raw_ordinal_far_from_corrected")
            if line.confidence < min_word_confidence:
                review_reasons.append("low_word_confidence")
            if note_date is None:
                review_reasons.append("page_date_needs_review")

            words.append(
                {
                    "date": note_date,
                    "page_number": page_number,
                    "column": column,
                    "sequence": sequence,
                    "raw_ordinal": raw_ordinal,
                    "corrected_ordinal": corrected_ordinal,
                    "word_raw": word_raw,
                    "word_normalized": word_normalized,
                    "ocr_confidence": round(line.confidence, 4),
                    "ordinal_confidence": round(ord_confidence, 4),
                    "bbox": bbox,
                    "source_text": line.text,
                    "line_indices": [line.index],
                    "is_highlighted": is_highlighted,
                    "highlight_confidence": highlight_confidence,
                    "highlight_region_indices": highlight_region_indices,
                    "definition_status": "not_requested",
                    "needs_review": bool(review_reasons),
                    "review_reasons": review_reasons,
                }
            )
            sequence += 1
            expected_ordinal = corrected_ordinal + 1

    return words, unassigned


def get_day(days: list[dict[str, Any]], index: dict[str, dict[str, Any]], key: str, note_date: str | None) -> dict[str, Any]:
    if key in index:
        return index[key]
    day = {
        "date": note_date,
        "pages": [],
        "words": [],
        "needs_review": note_date is None,
        "review_reasons": ["page_has_no_date_and_no_previous_date"] if note_date is None else [],
    }
    index[key] = day
    days.append(day)
    return day


def append_page(day: dict[str, Any], page_number: int) -> None:
    if page_number not in day["pages"]:
        day["pages"].append(page_number)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = load_json(input_path)
    highlights = load_highlights(args.highlight_json)

    days: list[dict[str, Any]] = []
    day_index: dict[str, dict[str, Any]] = {}
    current_day_key: str | None = None
    blank_pages: list[int] = []
    page_summaries: list[dict[str, Any]] = []
    unassigned_words: list[dict[str, Any]] = []
    total_words = 0

    for page in raw.get("pages") or []:
        page_number = int(page.get("page_number") or 0)
        lines = make_lines(page)
        if page.get("is_blank") or not lines:
            blank_pages.append(page_number)
            page_summaries.append(
                {
                    "page_number": page_number,
                    "status": "blank",
                    "line_count": len(lines),
                    "word_count": 0,
                    "date": None,
                    "highlight_region_count": len(highlights.get(page_number, [])),
                    "review_reasons": [],
                }
            )
            continue

        detected_date, date_line_indices, date_review_reasons = find_page_date(
            page,
            lines,
            default_year=args.default_year,
            allow_ambiguous_short_date=args.allow_ambiguous_short_date,
        )
        if detected_date:
            current_day_key = detected_date
            day = get_day(days, day_index, detected_date, detected_date)
        elif current_day_key is not None:
            day = get_day(days, day_index, current_day_key, day_index[current_day_key]["date"])
        else:
            current_day_key = "__undated__"
            day = get_day(days, day_index, current_day_key, None)

        page_review_reasons = list(date_review_reasons)
        if detected_date is None and day["date"] is None:
            page_review_reasons.append("page_has_no_date_and_no_previous_date")
        elif detected_date is None:
            page_review_reasons.append("date_inherited_from_previous_page")

        append_page(day, page_number)
        words, page_unassigned = extract_page_words(
            page,
            lines,
            note_date=day["date"],
            date_line_indices=date_line_indices,
            start_sequence=len(day["words"]) + 1,
            column_split=args.column_split,
            min_word_confidence=args.min_word_confidence,
            word_zone_ratio=args.word_zone_ratio,
            include_missing_ordinal=args.include_missing_ordinal,
            page_regions=highlights.get(page_number, []),
            highlight_word_overlap=args.highlight_word_overlap,
            highlight_region_overlap=args.highlight_region_overlap,
        )
        day["words"].extend(words)
        unassigned_words.extend(page_unassigned)
        total_words += len(words)

        if page_review_reasons:
            day["needs_review"] = True
            day["review_reasons"] = dedupe(day["review_reasons"] + page_review_reasons)

        page_summaries.append(
            {
                "page_number": page_number,
                "status": "processed",
                "line_count": len(lines),
                "word_count": len(words),
                "unassigned_word_count": len(page_unassigned),
                "highlighted_words": sum(1 for word in words if word["is_highlighted"]),
                "highlight_region_count": len(highlights.get(page_number, [])),
                "date": day["date"],
                "date_detected_on_page": detected_date,
                "date_inherited": detected_date is None and day["date"] is not None,
                "needs_review": bool(page_review_reasons or any(word["needs_review"] for word in words)),
                "review_reasons": dedupe(page_review_reasons),
            }
        )

    total_needs_review = 0
    total_highlighted = 0
    for day in days:
        day["pages"] = sorted(day["pages"])
        total_needs_review += sum(1 for word in day["words"] if word["needs_review"])
        total_highlighted += sum(1 for word in day["words"] if word["is_highlighted"])
        day["needs_review"] = bool(day["needs_review"] or any(word["needs_review"] for word in day["words"]))
        day["review_reasons"] = dedupe(day["review_reasons"])

    result = {
        "source_engine": raw.get("engine"),
        "input_json": str(input_path.resolve()),
        "highlight_json": str(Path(args.highlight_json).resolve()) if args.highlight_json else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "default_year": args.default_year,
            "allow_ambiguous_short_date": args.allow_ambiguous_short_date,
            "column_split": args.column_split,
            "min_word_confidence": args.min_word_confidence,
            "word_zone_ratio": args.word_zone_ratio,
            "include_missing_ordinal": args.include_missing_ordinal,
            "highlight_word_overlap": args.highlight_word_overlap,
            "highlight_region_overlap": args.highlight_region_overlap,
        },
        "summary": {
            "days": len(days),
            "pages": len(page_summaries),
            "blank_pages": len(blank_pages),
            "blank_page_numbers": blank_pages,
            "words": total_words,
            "highlighted_words": total_highlighted,
            "unassigned_words": len(unassigned_words),
            "needs_review": total_needs_review,
        },
        "page_summaries": page_summaries,
        "unassigned_words": unassigned_words,
        "days": days,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(
        "Done. "
        f"{len(days)} day(s), {total_words} word(s), "
        f"{total_highlighted} highlighted, {total_needs_review} need review."
    )
    print(f"Output -> {output_path}")


if __name__ == "__main__":
    main()
