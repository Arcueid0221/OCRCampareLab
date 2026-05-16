#!/usr/bin/env python3
"""Compare dictionary-based correction strategies for OCR vocabulary output.

The script reads postprocess_notes.py output, keeps the original OCR fields, and
adds correction results from local, network, and hybrid strategies. It is meant
to sit between OCR post-processing and later dictionary/API enrichment.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"^[a-z][a-z'\-]{2,}$")
ENTRY_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?:\d{1,3}|[A-Za-z0-9]{1,3})\s*[\.\),:;_\-]\s*"
    r")?"
    r"(?P<word>[A-Za-z][A-Za-z'\u2019\-]{1,})"
    r"\b"
)
WORD_STOPLIST = {
    "adj",
    "adv",
    "aux",
    "conj",
    "int",
    "prep",
    "pron",
    "vi",
    "vt",
}
CHECK_SCHEMA = """
CREATE TABLE IF NOT EXISTS dictionary_word_checks (
  word TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  word_exists INTEGER,
  status TEXT NOT NULL,
  http_status INTEGER,
  error_message TEXT,
  raw_json TEXT,
  checked_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Candidate:
    word: str
    score: float
    edit_distance: int
    sequence_ratio: float
    zipf: float
    source: str
    network_exists: bool | None = None

    def to_json(self) -> dict[str, Any]:
        result = {
            "word": self.word,
            "score": round(self.score, 4),
            "edit_distance": self.edit_distance,
            "sequence_ratio": round(self.sequence_ratio, 4),
            "zipf": round(self.zipf, 4),
            "source": self.source,
        }
        if self.network_exists is not None:
            result["network_exists"] = self.network_exists
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add local/network/hybrid dictionary correction results to OCR words JSON."
    )
    parser.add_argument("--input", required=True, help="Path to apple_words_shortdate.json")
    parser.add_argument("--output", required=True, help="Path to corrected comparison JSON")
    parser.add_argument(
        "--mode",
        choices=["compare", "local", "network", "hybrid"],
        default="compare",
        help="Correction mode. compare stores all three strategies and selects hybrid.",
    )
    parser.add_argument(
        "--db",
        default="outputs/correction_cache.sqlite",
        help="SQLite cache for network dictionary checks.",
    )
    parser.add_argument(
        "--custom-wordlist",
        default="wordlists/custom_words.txt",
        help="Optional one-word-per-line local vocabulary supplement.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=80000,
        help="Number of wordfreq English words to load for local correction.",
    )
    parser.add_argument(
        "--local-candidate-limit",
        type=int,
        default=8,
        help="Max local fuzzy candidates kept for each word.",
    )
    parser.add_argument(
        "--network-candidate-limit",
        type=int,
        default=5,
        help="Max candidate words checked through dictionaryapi.dev.",
    )
    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=0.9,
        help="Minimum confidence for auto_corrected.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.78,
        help="Minimum confidence for candidate_review.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.05,
        help="Minimum score margin over second candidate for auto correction.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.08,
        help="Delay between uncached network requests.",
    )
    parser.add_argument(
        "--offline-network",
        action="store_true",
        help="Use cache only; never call dictionaryapi.dev.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_word(raw_word: str) -> str:
    word = raw_word.replace("\u2019", "'").strip().strip(".,:;_-()[]{}")
    return re.sub(r"[^A-Za-z'\-]", "", word).lower()


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


def normalize_candidate_text(text: str) -> str | None:
    match = ENTRY_RE.match(normalize_text(text))
    if not match:
        return None
    word = normalize_word(match.group("word"))
    if len(word) <= 2 or word in WORD_STOPLIST:
        return None
    return word


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        print(f"Error: file not found -> {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def edit_distance(a: str, b: str, *, max_distance: int | None = None) -> int | None:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if max_distance is not None and abs(len(a) - len(b)) > max_distance:
        return None

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        row_min = current[0]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if max_distance is not None and row_min > max_distance:
            return None
        previous = current
    distance = previous[-1]
    if max_distance is not None and distance > max_distance:
        return None
    return distance


def similarity_score(raw_word: str, candidate: str, *, zipf: float, source: str) -> Candidate | None:
    max_len = max(len(raw_word), len(candidate))
    if max_len <= 0:
        return None
    max_dist = max(2, round(max_len * 0.34))
    distance = edit_distance(raw_word, candidate, max_distance=max_dist)
    if distance is None:
        return None

    edit_similarity = 1.0 - distance / max_len
    sequence_ratio = SequenceMatcher(None, raw_word, candidate).ratio()
    prefix_bonus = 0.0
    if raw_word[:1] == candidate[:1]:
        prefix_bonus += 0.025
    if raw_word[:2] == candidate[:2]:
        prefix_bonus += 0.025
    if raw_word[-1:] == candidate[-1:]:
        prefix_bonus += 0.015
    frequency_bonus = min(max(zipf - 2.0, 0.0) / 40.0, 0.08)
    source_bonus = 0.02 if source == "ocr_candidate" else 0.0
    score = (0.58 * edit_similarity) + (0.34 * sequence_ratio) + prefix_bonus + frequency_bonus + source_bonus
    if raw_word[:1] != candidate[:1] and distance > 1:
        score -= 0.035
    return Candidate(
        word=candidate,
        score=max(0.0, min(score, 1.0)),
        edit_distance=distance,
        sequence_ratio=sequence_ratio,
        zipf=zipf,
        source=source,
    )


class LocalLexicon:
    def __init__(self, *, top_n: int, custom_wordlist: Path) -> None:
        try:
            from wordfreq import top_n_list, zipf_frequency
        except ImportError:
            print("Error: wordfreq is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
            sys.exit(1)

        self._zipf_frequency = zipf_frequency
        self.words: list[str] = []
        self.word_set: set[str] = set()
        self.custom_words: set[str] = set()
        self.length_buckets: dict[int, list[str]] = {}

        for raw in top_n_list("en", top_n):
            word = normalize_word(str(raw))
            if self._accept_word(word):
                self._add_word(word)

        if custom_wordlist.is_file():
            for raw_line in custom_wordlist.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                word = normalize_word(line.split(",")[0])
                if self._accept_word(word):
                    self.custom_words.add(word)
                    self._add_word(word)

    @staticmethod
    def _accept_word(word: str) -> bool:
        return bool(WORD_RE.fullmatch(word)) and word not in WORD_STOPLIST

    def _add_word(self, word: str) -> None:
        if word not in self.word_set:
            self.word_set.add(word)
            self.words.append(word)
            self.length_buckets.setdefault(len(word), []).append(word)

    def zipf(self, word: str) -> float:
        if word in self.custom_words:
            return max(5.5, float(self._zipf_frequency(word, "en")))
        return float(self._zipf_frequency(word, "en"))

    def is_valid(self, word: str) -> bool:
        return word in self.word_set or word in self.custom_words or self.zipf(word) >= 1.8

    def candidates(self, word: str, *, limit: int) -> list[Candidate]:
        if not word or len(word) <= 2:
            return []
        length_delta = max(2, round(len(word) * 0.25))
        candidates: list[Candidate] = []
        for candidate_length in range(len(word) - length_delta, len(word) + length_delta + 1):
            for candidate_word in self.length_buckets.get(candidate_length, []):
                if candidate_word == word:
                    continue
                candidate = similarity_score(
                    word,
                    candidate_word,
                    zipf=self.zipf(candidate_word),
                    source="wordfreq",
                )
                if candidate and candidate.score >= 0.72:
                    candidates.append(candidate)
        candidates.sort(key=lambda item: (item.score, -item.edit_distance, item.zipf), reverse=True)
        return candidates[:limit]


class NetworkChecker:
    def __init__(self, db_path: Path, *, offline: bool, request_delay: float) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(CHECK_SCHEMA)
        self.conn.commit()
        self.offline = offline
        self.request_delay = request_delay
        self.cache_hits = 0
        self.requests = 0
        self.errors = 0
        self._memo: dict[str, dict[str, Any]] = {}

    def check(self, word: str) -> dict[str, Any]:
        if word in self._memo:
            return self._memo[word]

        row = self.conn.execute(
            "SELECT * FROM dictionary_word_checks WHERE word = ?",
            (word,),
        ).fetchone()
        if row:
            self.cache_hits += 1
            result = self._row_to_result(row)
            self._memo[word] = result
            return result

        if self.offline:
            result = {
                "word": word,
                "exists": None,
                "status": "offline",
                "http_status": None,
                "error_message": "offline and cache miss",
                "cache_hit": False,
            }
            self._memo[word] = result
            return result

        if self.requests > 0 and self.request_delay > 0:
            time.sleep(self.request_delay)

        self.requests += 1
        result = self._fetch(word)
        self._save(result)
        self._memo[word] = result
        return result

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> dict[str, Any]:
        exists_value = row["word_exists"]
        return {
            "word": row["word"],
            "exists": None if exists_value is None else bool(exists_value),
            "status": row["status"],
            "http_status": row["http_status"],
            "error_message": row["error_message"],
            "cache_hit": True,
        }

    def _fetch(self, word: str) -> dict[str, Any]:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}"
        request = urllib.request.Request(url, headers={"User-Agent": "OCRCompareLab/1.0"})
        raw_text = ""
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                raw_text = response.read().decode("utf-8")
                data = json.loads(raw_text)
                exists = isinstance(data, list) and bool(data)
                return {
                    "word": word,
                    "exists": exists,
                    "status": "found" if exists else "not_found",
                    "http_status": response.status,
                    "error_message": None,
                    "raw_json": raw_text,
                    "cache_hit": False,
                }
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {
                    "word": word,
                    "exists": False,
                    "status": "not_found",
                    "http_status": exc.code,
                    "error_message": None,
                    "raw_json": None,
                    "cache_hit": False,
                }
            self.errors += 1
            return {
                "word": word,
                "exists": None,
                "status": "error",
                "http_status": exc.code,
                "error_message": f"HTTP {exc.code}",
                "raw_json": raw_text or None,
                "cache_hit": False,
            }
        except Exception as exc:
            self.errors += 1
            return {
                "word": word,
                "exists": None,
                "status": "error",
                "http_status": None,
                "error_message": str(exc),
                "raw_json": raw_text or None,
                "cache_hit": False,
            }

    def _save(self, result: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO dictionary_word_checks (
              word, provider, word_exists, status, http_status, error_message, raw_json, checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(word) DO UPDATE SET
              provider=excluded.provider,
              word_exists=excluded.word_exists,
              status=excluded.status,
              http_status=excluded.http_status,
              error_message=excluded.error_message,
              raw_json=excluded.raw_json,
              checked_at=excluded.checked_at
            """,
            (
                result["word"],
                "dictionaryapi",
                None if result.get("exists") is None else int(bool(result["exists"])),
                result.get("status", "error"),
                result.get("http_status"),
                result.get("error_message"),
                result.get("raw_json"),
                now_iso(),
            ),
        )
        self.conn.commit()


def correction_result(
    *,
    status: str,
    word: str,
    confidence: float,
    source: str,
    candidates: list[Candidate] | None = None,
    checks: list[dict[str, Any]] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "word": word,
        "confidence": round(max(0.0, min(confidence, 1.0)), 4),
        "source": source,
        "candidates": [candidate.to_json() for candidate in (candidates or [])],
        "checks": checks or [],
        "error_message": error_message,
    }


def classify_candidates(
    original: str,
    candidates: list[Candidate],
    *,
    source: str,
    auto_threshold: float,
    review_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    if not candidates:
        return correction_result(status="not_found", word=original, confidence=0.0, source=source)
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = best.score - second_score
    if best.score >= auto_threshold and (margin >= margin_threshold or best.edit_distance <= 1):
        return correction_result(
            status="auto_corrected",
            word=best.word,
            confidence=best.score,
            source=source,
            candidates=candidates,
        )
    if best.score >= review_threshold:
        return correction_result(
            status="candidate_review",
            word=best.word,
            confidence=best.score,
            source=source,
            candidates=candidates,
        )
    return correction_result(
        status="not_found",
        word=original,
        confidence=best.score,
        source=source,
        candidates=candidates,
    )


def local_correction(
    word: str,
    lexicon: LocalLexicon,
    *,
    candidate_limit: int,
    auto_threshold: float,
    review_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    if len(word) <= 2:
        return correction_result(status="skipped", word=word, confidence=0.0, source="local")
    if lexicon.is_valid(word):
        return correction_result(status="exact", word=word, confidence=0.96, source="local")
    candidates = lexicon.candidates(word, limit=candidate_limit)
    return classify_candidates(
        word,
        candidates,
        source="local",
        auto_threshold=auto_threshold,
        review_threshold=review_threshold,
        margin_threshold=margin_threshold,
    )


def ocr_candidate_words(record: dict[str, Any], ocr_index: dict[tuple[int, int], list[str]]) -> list[str]:
    page_number = int(record.get("page_number") or 0)
    candidates: list[str] = []
    for line_index in record.get("line_indices") or []:
        try:
            key = (page_number, int(line_index))
        except (TypeError, ValueError):
            continue
        candidates.extend(ocr_index.get(key, []))
    return dedupe(candidates)


def network_correction(
    word: str,
    candidates: list[str],
    checker: NetworkChecker,
    lexicon: LocalLexicon,
    *,
    candidate_limit: int,
    auto_threshold: float,
    review_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    if len(word) <= 2:
        return correction_result(status="skipped", word=word, confidence=0.0, source="network")

    checks: list[dict[str, Any]] = []
    original_check = checker.check(word)
    checks.append(public_check(original_check))
    if original_check.get("exists") is True:
        return correction_result(status="exact", word=word, confidence=0.97, source="network", checks=checks)

    valid_candidates: list[Candidate] = []
    for candidate_word in candidates[:candidate_limit]:
        if candidate_word == word or len(candidate_word) <= 2:
            continue
        check = checker.check(candidate_word)
        checks.append(public_check(check))
        if check.get("exists") is not True:
            continue
        candidate = similarity_score(
            word,
            candidate_word,
            zipf=lexicon.zipf(candidate_word),
            source="ocr_candidate",
        )
        if candidate:
            valid_candidates.append(
                Candidate(
                    word=candidate.word,
                    score=min(candidate.score + 0.04, 1.0),
                    edit_distance=candidate.edit_distance,
                    sequence_ratio=candidate.sequence_ratio,
                    zipf=candidate.zipf,
                    source="ocr_candidate",
                    network_exists=True,
                )
            )

    result = classify_candidates(
        word,
        valid_candidates,
        source="network",
        auto_threshold=auto_threshold,
        review_threshold=review_threshold,
        margin_threshold=margin_threshold,
    )
    result["checks"] = checks
    if original_check.get("status") in {"error", "offline"} and not valid_candidates:
        result["error_message"] = original_check.get("error_message")
    return result


def hybrid_correction(
    word: str,
    ocr_candidates: list[str],
    local_result: dict[str, Any],
    network_result: dict[str, Any],
    lexicon: LocalLexicon,
    checker: NetworkChecker,
    *,
    local_candidate_limit: int,
    network_candidate_limit: int,
    auto_threshold: float,
    review_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    if len(word) <= 2:
        return correction_result(status="skipped", word=word, confidence=0.0, source="hybrid")

    if network_result["status"] == "exact":
        return correction_result(
            status="exact",
            word=word,
            confidence=0.98,
            source="hybrid",
            checks=network_result.get("checks", []),
        )
    if local_result["status"] == "exact":
        return correction_result(
            status="exact",
            word=word,
            confidence=0.93,
            source="hybrid_local_exact",
            checks=network_result.get("checks", []),
        )

    local_candidates = lexicon.candidates(word, limit=local_candidate_limit)
    merged: dict[str, Candidate] = {candidate.word: candidate for candidate in local_candidates}
    for candidate_word in ocr_candidates:
        if candidate_word == word:
            continue
        candidate = similarity_score(
            word,
            candidate_word,
            zipf=lexicon.zipf(candidate_word),
            source="ocr_candidate",
        )
        if not candidate:
            continue
        existing = merged.get(candidate.word)
        if existing is None or candidate.score > existing.score:
            merged[candidate.word] = candidate

    checked_candidates: list[Candidate] = []
    checks = list(network_result.get("checks") or [])
    for candidate in sorted(merged.values(), key=lambda item: item.score, reverse=True)[:network_candidate_limit]:
        check = checker.check(candidate.word)
        checks.append(public_check(check))
        if check.get("exists") is True:
            checked_candidates.append(
                Candidate(
                    word=candidate.word,
                    score=min(candidate.score + 0.06, 1.0),
                    edit_distance=candidate.edit_distance,
                    sequence_ratio=candidate.sequence_ratio,
                    zipf=candidate.zipf,
                    source=f"{candidate.source}+network",
                    network_exists=True,
                )
            )
        elif check.get("exists") is None and candidate.score >= auto_threshold + 0.03:
            checked_candidates.append(
                Candidate(
                    word=candidate.word,
                    score=max(candidate.score - 0.05, 0.0),
                    edit_distance=candidate.edit_distance,
                    sequence_ratio=candidate.sequence_ratio,
                    zipf=candidate.zipf,
                    source=f"{candidate.source}+network_unavailable",
                    network_exists=None,
                )
            )

    result = classify_candidates(
        word,
        checked_candidates,
        source="hybrid",
        auto_threshold=auto_threshold,
        review_threshold=review_threshold,
        margin_threshold=margin_threshold,
    )
    result["checks"] = checks
    return result


def public_check(check: dict[str, Any]) -> dict[str, Any]:
    return {
        "word": check.get("word"),
        "exists": check.get("exists"),
        "status": check.get("status"),
        "http_status": check.get("http_status"),
        "cache_hit": check.get("cache_hit", False),
        "error_message": check.get("error_message"),
    }


def load_ocr_index(notes: dict[str, Any]) -> dict[tuple[int, int], list[str]]:
    input_json = notes.get("input_json")
    if not input_json:
        return {}
    path = Path(str(input_json))
    if not path.is_file():
        return {}
    raw = load_json(path)
    index: dict[tuple[int, int], list[str]] = {}
    for page in raw.get("pages") or []:
        page_number = int(page.get("page_number") or 0)
        for fallback_index, line in enumerate(page.get("lines") or []):
            line_index = int(line.get("index", fallback_index))
            words: list[str] = []
            source_text = str(line.get("text") or "")
            normalized_source = normalize_candidate_text(source_text)
            if normalized_source:
                words.append(normalized_source)
            for candidate in line.get("candidates") or []:
                normalized = normalize_candidate_text(str(candidate.get("text") or ""))
                if normalized:
                    words.append(normalized)
            index[(page_number, line_index)] = dedupe(words)
    return index


def selected_strategy(mode: str) -> str:
    return "hybrid" if mode == "compare" else mode


def apply_correction_to_record(
    record: dict[str, Any],
    *,
    lexicon: LocalLexicon,
    checker: NetworkChecker,
    ocr_index: dict[tuple[int, int], list[str]],
    args: argparse.Namespace,
    counters: dict[str, dict[str, int]],
) -> None:
    original = normalize_word(str(record.get("word_normalized") or record.get("word_raw") or ""))
    record["word_normalized_raw"] = original
    if not original:
        result = correction_result(status="skipped", word=original, confidence=0.0, source="hybrid")
        corrections = {"hybrid": result} if args.mode != "compare" else {"local": result, "network": result, "hybrid": result}
    else:
        ocr_words = ocr_candidate_words(record, ocr_index)
        local_result = local_correction(
            original,
            lexicon,
            candidate_limit=args.local_candidate_limit,
            auto_threshold=args.auto_threshold,
            review_threshold=args.review_threshold,
            margin_threshold=args.margin_threshold,
        )
        network_result = network_correction(
            original,
            ocr_words,
            checker,
            lexicon,
            candidate_limit=args.network_candidate_limit,
            auto_threshold=args.auto_threshold,
            review_threshold=args.review_threshold,
            margin_threshold=args.margin_threshold,
        )
        hybrid_result = hybrid_correction(
            original,
            ocr_words,
            local_result,
            network_result,
            lexicon,
            checker,
            local_candidate_limit=args.local_candidate_limit,
            network_candidate_limit=args.network_candidate_limit,
            auto_threshold=args.auto_threshold,
            review_threshold=args.review_threshold,
            margin_threshold=args.margin_threshold,
        )
        all_results = {
            "local": local_result,
            "network": network_result,
            "hybrid": hybrid_result,
        }
        corrections = all_results if args.mode == "compare" else {args.mode: all_results[args.mode]}

    selected = selected_strategy(args.mode)
    selected_result = corrections.get(selected) or corrections.get("hybrid") or next(iter(corrections.values()))
    record["correction_selected"] = selected
    record["corrections"] = corrections
    if selected_result["status"] in {"exact", "auto_corrected"}:
        record["word_normalized"] = selected_result["word"]
    else:
        record["word_normalized"] = original

    for strategy, result in corrections.items():
        status = str(result.get("status") or "unknown")
        counters.setdefault(strategy, {})
        counters[strategy][status] = counters[strategy].get(status, 0) + 1

    if selected_result["status"] == "candidate_review":
        record["needs_review"] = True
        record["review_reasons"] = dedupe(list(record.get("review_reasons") or []) + ["word_correction_needs_review"])
    elif selected_result["status"] == "not_found":
        record["needs_review"] = True
        record["review_reasons"] = dedupe(list(record.get("review_reasons") or []) + ["word_not_found_in_dictionary"])


def process_records(
    data: dict[str, Any],
    *,
    lexicon: LocalLexicon,
    checker: NetworkChecker,
    ocr_index: dict[tuple[int, int], list[str]],
    args: argparse.Namespace,
) -> tuple[int, int, dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    main_counters: dict[str, dict[str, int]] = {}
    unassigned_counters: dict[str, dict[str, int]] = {}
    main_count = 0
    unassigned_count = 0
    for day in data.get("days") or []:
        for record in day.get("words") or []:
            apply_correction_to_record(
                record,
                lexicon=lexicon,
                checker=checker,
                ocr_index=ocr_index,
                args=args,
                counters=main_counters,
            )
            main_count += 1
    for record in data.get("unassigned_words") or []:
        apply_correction_to_record(
            record,
            lexicon=lexicon,
            checker=checker,
            ocr_index=ocr_index,
            args=args,
            counters=unassigned_counters,
        )
        unassigned_count += 1
    return main_count, unassigned_count, main_counters, unassigned_counters


def build_summary(
    *,
    processed_words: int,
    unassigned_processed: int,
    counters: dict[str, dict[str, int]],
    unassigned_counters: dict[str, dict[str, int]],
    checker: NetworkChecker,
    lexicon: LocalLexicon,
    args: argparse.Namespace,
) -> dict[str, Any]:
    hybrid = counters.get("hybrid", {})
    local = counters.get("local", {})
    network = counters.get("network", {})
    return {
        "processed_words": processed_words,
        "unassigned_processed": unassigned_processed,
        "local_auto_corrected": local.get("auto_corrected", 0),
        "network_confirmed": network.get("exact", 0),
        "hybrid_auto_corrected": hybrid.get("auto_corrected", 0),
        "candidate_review": hybrid.get("candidate_review", 0),
        "not_found": hybrid.get("not_found", 0),
        "skipped": hybrid.get("skipped", 0),
        "strategy_counts": counters,
        "unassigned_strategy_counts": unassigned_counters,
        "local_words": len(lexicon.word_set),
        "custom_words": len(lexicon.custom_words),
        "network_provider": "dictionaryapi",
        "network_cache_db": str(Path(args.db).resolve()),
        "network_cache_hits": checker.cache_hits,
        "network_requests": checker.requests,
        "network_errors": checker.errors,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = load_json(input_path)
    data = copy.deepcopy(source)
    lexicon = LocalLexicon(top_n=args.top_n, custom_wordlist=Path(args.custom_wordlist))
    checker = NetworkChecker(Path(args.db), offline=args.offline_network, request_delay=args.request_delay)
    ocr_index = load_ocr_index(source)

    processed_words, unassigned_processed, counters, unassigned_counters = process_records(
        data,
        lexicon=lexicon,
        checker=checker,
        ocr_index=ocr_index,
        args=args,
    )

    data["correction_settings"] = {
        "mode": args.mode,
        "selected_strategy": selected_strategy(args.mode),
        "local_provider": "wordfreq",
        "local_top_n": args.top_n,
        "custom_wordlist": str(Path(args.custom_wordlist).resolve()),
        "network_provider": "dictionaryapi",
        "auto_threshold": args.auto_threshold,
        "review_threshold": args.review_threshold,
        "margin_threshold": args.margin_threshold,
        "local_candidate_limit": args.local_candidate_limit,
        "network_candidate_limit": args.network_candidate_limit,
        "offline_network": args.offline_network,
    }
    data["correction_summary"] = build_summary(
        processed_words=processed_words,
        unassigned_processed=unassigned_processed,
        counters=counters,
        unassigned_counters=unassigned_counters,
        checker=checker,
        lexicon=lexicon,
        args=args,
    )
    data["correction_created_at"] = now_iso()
    data["correction_input_json"] = str(input_path.resolve())

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Done. "
        f"{processed_words} main word(s), {unassigned_processed} unassigned word(s), "
        f"hybrid auto={data['correction_summary']['hybrid_auto_corrected']}, "
        f"review={data['correction_summary']['candidate_review']}, "
        f"not_found={data['correction_summary']['not_found']}, "
        f"network requests={checker.requests}, cache hits={checker.cache_hits}."
    )
    print(f"Output -> {output_path}")


if __name__ == "__main__":
    main()
