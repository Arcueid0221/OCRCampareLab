#!/usr/bin/env python3
"""Local dictionary cache with optional third-party API lookup.

Default provider uses dictionaryapi.dev for English definitions and phonetics.
Optional DeepSeek enrichment can fill Chinese meaning and exam-level examples
when DEEPSEEK_API_KEY is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS dictionary_entries (
  word TEXT PRIMARY KEY,
  pos TEXT,
  meaning_cn TEXT,
  meaning_en TEXT,
  phonetic_uk TEXT,
  phonetic_us TEXT,
  example_en TEXT,
  example_cn TEXT,
  source TEXT NOT NULL,
  provider TEXT NOT NULL,
  status TEXT NOT NULL,
  error_message TEXT,
  raw_json TEXT,
  updated_at TEXT NOT NULL
);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate/read local dictionary cache for OCR words.")
    parser.add_argument("--notes", required=True, help="Post-processed notes JSON from postprocess_notes.py")
    parser.add_argument("--db", default="outputs/dictionary_cache.sqlite", help="SQLite cache path")
    parser.add_argument("--output", required=True, help="Dictionary result JSON output")
    parser.add_argument("--provider", choices=["dictionaryapi"], default="dictionaryapi")
    parser.add_argument("--offline", action="store_true", help="Only read cache; never call external APIs")
    parser.add_argument("--deepseek", choices=["on", "off"], default="off", help="Optionally enrich CN fields/examples")
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--limit", type=int, default=0, help="Limit number of unique words; 0 = all")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def load_notes_words(path: Path) -> list[str]:
    if not path.is_file():
        print(f"Error: notes file not found -> {path}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    words: list[str] = []
    for day in data.get("days") or []:
        for word in day.get("words") or []:
            normalized = str(word.get("word_normalized") or "").strip().lower()
            if normalized and normalized not in words:
                words.append(normalized)
    return words


def row_to_entry(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "word": row["word"],
        "pos": row["pos"] or "",
        "meaning_cn": row["meaning_cn"] or "",
        "meaning_en": row["meaning_en"] or "",
        "phonetic_uk": row["phonetic_uk"] or "",
        "phonetic_us": row["phonetic_us"] or "",
        "example_en": row["example_en"] or "",
        "example_cn": row["example_cn"] or "",
        "source": row["source"],
        "provider": row["provider"],
        "status": row["status"],
        "error_message": row["error_message"],
        "updated_at": row["updated_at"],
        "cache_hit": True,
    }


def get_cached(conn: sqlite3.Connection, word: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM dictionary_entries WHERE word = ?", (word,)).fetchone()
    return row_to_entry(row) if row else None


def save_entry(conn: sqlite3.Connection, entry: dict[str, Any], raw_json: Any | None = None) -> None:
    conn.execute(
        """
        INSERT INTO dictionary_entries (
          word, pos, meaning_cn, meaning_en, phonetic_uk, phonetic_us,
          example_en, example_cn, source, provider, status, error_message,
          raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(word) DO UPDATE SET
          pos=excluded.pos,
          meaning_cn=excluded.meaning_cn,
          meaning_en=excluded.meaning_en,
          phonetic_uk=excluded.phonetic_uk,
          phonetic_us=excluded.phonetic_us,
          example_en=excluded.example_en,
          example_cn=excluded.example_cn,
          source=excluded.source,
          provider=excluded.provider,
          status=excluded.status,
          error_message=excluded.error_message,
          raw_json=excluded.raw_json,
          updated_at=excluded.updated_at
        """,
        (
            entry["word"],
            entry.get("pos", ""),
            entry.get("meaning_cn", ""),
            entry.get("meaning_en", ""),
            entry.get("phonetic_uk", ""),
            entry.get("phonetic_us", ""),
            entry.get("example_en", ""),
            entry.get("example_cn", ""),
            entry.get("source", "api"),
            entry.get("provider", "dictionaryapi"),
            entry.get("status", "fetched"),
            entry.get("error_message"),
            json.dumps(raw_json, ensure_ascii=False) if raw_json is not None else None,
            entry.get("updated_at", now_iso()),
        ),
    )
    conn.commit()


def http_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
    body = None
    final_headers = {"User-Agent": "OCRCompareLab/1.0"}
    if headers:
        final_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        final_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=final_headers, method=method)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_dictionaryapi(word: str) -> tuple[dict[str, Any], Any | None]:
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}"
    try:
        data = http_json(url)
    except urllib.error.HTTPError as exc:
        return failed_entry(word, "dictionaryapi", f"HTTP {exc.code}"), None
    except Exception as exc:
        return failed_entry(word, "dictionaryapi", str(exc)), None

    if not isinstance(data, list) or not data:
        return failed_entry(word, "dictionaryapi", "empty response"), data

    first = data[0]
    phonetic_uk = ""
    phonetic_us = ""
    for phonetic in first.get("phonetics") or []:
        text = str(phonetic.get("text") or "")
        audio = str(phonetic.get("audio") or "").lower()
        if not phonetic_uk and ("-uk" in audio or "uk" in audio):
            phonetic_uk = text
        if not phonetic_us and ("-us" in audio or "us" in audio):
            phonetic_us = text
        if text and not phonetic_uk:
            phonetic_uk = text
        if text and not phonetic_us:
            phonetic_us = text

    parts: list[str] = []
    definitions: list[str] = []
    example_en = ""
    for meaning in first.get("meanings") or []:
        pos = str(meaning.get("partOfSpeech") or "").strip()
        if pos and pos not in parts:
            parts.append(pos)
        for definition in meaning.get("definitions") or []:
            text = str(definition.get("definition") or "").strip()
            if text and text not in definitions:
                definitions.append(text)
            if not example_en:
                example_en = str(definition.get("example") or "").strip()

    entry = {
        "word": word,
        "pos": "; ".join(parts[:4]),
        "meaning_cn": "",
        "meaning_en": " | ".join(definitions[:4]),
        "phonetic_uk": phonetic_uk,
        "phonetic_us": phonetic_us,
        "example_en": example_en,
        "example_cn": "",
        "source": "api",
        "provider": "dictionaryapi",
        "status": "partial" if definitions else "failed",
        "error_message": None if definitions else "no definitions in response",
        "updated_at": now_iso(),
        "cache_hit": False,
    }
    return entry, data


def failed_entry(word: str, provider: str, error: str) -> dict[str, Any]:
    return {
        "word": word,
        "pos": "",
        "meaning_cn": "",
        "meaning_en": "",
        "phonetic_uk": "",
        "phonetic_us": "",
        "example_en": "",
        "example_cn": "",
        "source": "api",
        "provider": provider,
        "status": "failed",
        "error_message": error,
        "updated_at": now_iso(),
        "cache_hit": False,
    }


def maybe_enrich_deepseek(entry: dict[str, Any], *, model: str) -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return entry
    prompt = {
        "word": entry["word"],
        "meaning_en": entry.get("meaning_en", ""),
        "required_json": {
            "meaning_cn": "中文核心释义",
            "example_en": "考研难度英文例句",
            "example_cn": "例句中文解释",
        },
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. Do not include markdown."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 800,
    }
    try:
        data = http_json(
            "https://api.deepseek.com/chat/completions",
            method="POST",
            payload=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        content = data["choices"][0]["message"]["content"]
        enriched = json.loads(content)
    except Exception as exc:
        entry["error_message"] = f"{entry.get('error_message') or ''}; deepseek: {exc}".strip("; ")
        return entry

    for key in ("meaning_cn", "example_en", "example_cn"):
        value = str(enriched.get(key) or "").strip()
        if value:
            entry[key] = value
    if entry.get("meaning_cn") and entry.get("meaning_en"):
        entry["status"] = "fetched"
    entry["source"] = "api+deepseek"
    return entry


def main() -> None:
    args = parse_args()
    words = load_notes_words(Path(args.notes))
    if args.limit > 0:
        words = words[: args.limit]
    conn = connect(Path(args.db))

    definitions: dict[str, dict[str, Any]] = {}
    fetched = 0
    cache_hits = 0
    failed = 0

    for index, word in enumerate(words, start=1):
        cached = get_cached(conn, word)
        if cached:
            definitions[word] = cached
            cache_hits += 1
            print(f"[{index}/{len(words)}] cache {word}")
            continue

        if args.offline:
            entry = failed_entry(word, args.provider, "offline and cache miss")
            definitions[word] = entry
            failed += 1
            print(f"[{index}/{len(words)}] offline-miss {word}")
            continue

        print(f"[{index}/{len(words)}] fetch {word}")
        entry, raw_json = fetch_dictionaryapi(word)
        if args.deepseek == "on" and entry["status"] != "failed":
            entry = maybe_enrich_deepseek(entry, model=args.deepseek_model)
        save_entry(conn, entry, raw_json)
        definitions[word] = entry
        fetched += 1
        if entry["status"] == "failed":
            failed += 1
        time.sleep(0.12)

    result = {
        "created_at": now_iso(),
        "notes": str(Path(args.notes).resolve()),
        "db": str(Path(args.db).resolve()),
        "settings": {
            "provider": args.provider,
            "offline": args.offline,
            "deepseek": args.deepseek,
            "deepseek_model": args.deepseek_model if args.deepseek == "on" else None,
        },
        "summary": {
            "words": len(words),
            "cache_hits": cache_hits,
            "fetched": fetched,
            "failed": failed,
        },
        "definitions": definitions,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Output -> {output_path}")


if __name__ == "__main__":
    main()
