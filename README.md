# OCRCompareLab

Independent OCR comparison tool — runs three OCR engines (Apple Vision, PaddleOCR, EasyOCR) against the same PDF and compares confidence, line counts, speed, and output format.

## Quick Start

```bash
cd /Users/lwl/Desktop/考研/英语/OCRCompareLab

# 1. Create virtual environment & install dependencies
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Run the current main OCR path: Apple Vision, English-first
python apple_ocr.py --input ../单词.pdf --output outputs/apple_english.json --max-pages 3

# 3. Detect yellow highlights in the first 5 pages
python highlight_detect.py \
  --input ../单词.pdf \
  --output outputs/highlights_5.json \
  --max-pages 5 \
  --morph-kernel 3 \
  --close-iterations 1 \
  --open-iterations 1

# 4. Post-process Apple OCR notes into ordered English words
python postprocess_notes.py \
  --input outputs/apple_english.json \
  --highlight-json outputs/highlights_5.json \
  --output outputs/apple_words.json

# If you want 5.5 / 5.6 style dates to be parsed as dates:
python postprocess_notes.py \
  --input outputs/apple_english.json \
  --highlight-json outputs/highlights_5.json \
  --output outputs/apple_words_shortdate.json \
  --allow-ambiguous-short-date

# 5. Compare local/network/hybrid dictionary correction strategies
python correct_words.py \
  --input outputs/apple_words_shortdate.json \
  --output outputs/apple_words_corrected_compare.json \
  --mode compare

# Optional: run the older three-engine comparison path
python apple_ocr.py --input ../单词.pdf --output outputs/apple_compare.json --max-pages 3
python paddle_ocr.py --input ../单词.pdf --output outputs/paddle_compare.json --dpi 150 --max-pages 3
python easy_ocr.py --input ../单词.pdf --output outputs/easy_compare.json --dpi 150 --max-pages 3
python compare_confidence.py \
  --apple outputs/apple_compare.json \
  --paddle outputs/paddle_compare.json \
  --easy outputs/easy_compare.json \
  --output outputs/comparison.json
```

## Scripts

| Script | Description | Requirements |
|--------|-------------|-------------|
| `apple_ocr.py` | Apple Vision OCR via PyObjC | macOS only |
| `paddle_ocr.py` | PaddleOCR (Chinese model) | Cross-platform |
| `easy_ocr.py` | EasyOCR (Chinese + English) | Cross-platform |
| `highlight_detect.py` | Detect yellow highlighter regions | Cross-platform |
| `postprocess_notes.py` | Parse Apple OCR into dated two-column English word records | Cross-platform |
| `correct_words.py` | Compare local/network/hybrid dictionary correction strategies | Cross-platform |
| `dictionary_cache.py` | Local SQLite dictionary cache plus optional API lookup | Cross-platform |
| `compare_confidence.py` | Compare results & generate report | Cross-platform |

## CLI Options

All OCR scripts share these flags:

- `--input PATH` — Input PDF file
- `--output PATH` — Output JSON path
- `--dpi N` — Render DPI (`apple_ocr.py` default: 240; others default: 150)
- `--max-pages N` — Max pages; 0 = all (default: 0)

Engine-specific:

- `apple_ocr.py`: `--languages` (default: en-US), `--recognition-level accurate|fast`, `--language-correction on|off`, `--min-confidence`
- `paddle_ocr.py`: `--lang` (default: ch)
- `easy_ocr.py`: `--languages` (default: ch_sim en), `--canvas-size` (default: 1920)
- `highlight_detect.py`: HSV thresholds `--h-min`, `--h-max`, `--s-min`, `--v-min`, `--min-area`, plus morphology `--morph-kernel`, `--close-iterations`, `--open-iterations`
- `postprocess_notes.py`: `--highlight-json`, `--default-year`, `--allow-ambiguous-short-date`, `--column-split`, `--word-zone-ratio`, `--include-missing-ordinal`
- `correct_words.py`: `--mode compare|local|network|hybrid`, `--db`, `--custom-wordlist`, `--top-n`, correction thresholds, network cache controls
- `dictionary_cache.py`: `--notes`, `--db`, `--provider dictionaryapi`, `--offline`, `--deepseek on|off`

## Current Kept Outputs

The cleaned `outputs/` folder keeps only the current Apple-first pipeline artifacts:

- `apple_english.json` — raw Apple Vision OCR, English-first, first 3 pages.
- `highlights_5.json` — yellow highlight regions for the first 5 pages.
- `apple_words.json` — English word records, with short dates such as `5.5` left ambiguous.
- `apple_words_shortdate.json` — English word records, treating short dates such as `5.5` as valid dates.
- `apple_words_corrected_compare.json` — optional correction comparison output from `correct_words.py`.

## Output Format

All OCR scripts produce a unified JSON structure:

```json
{
  "engine": "apple_vision | paddleocr | easyocr",
  "input_pdf": "/abs/path/file.pdf",
  "dpi": 150,
  "created_at": "ISO time",
  "summary": {
    "pages": 3,
    "blank_pages": 1,
    "total_lines": 341,
    "avg_confidence": 0.82,
    "total_elapsed_seconds": 20.5
  },
  "pages": [...]
}
```

`apple_ocr.py` also includes per-line top candidates:

```json
{
  "text": "radiate",
  "confidence": 0.92,
  "is_low_confidence": false,
  "candidates": [
    {"text": "radiate", "confidence": 0.92},
    {"text": "radiale", "confidence": 0.41}
  ]
}
```

`postprocess_notes.py` reads the raw Apple OCR JSON and optional highlight JSON, then writes English-only word records:

```json
{
  "source_engine": "apple_vision",
  "summary": {
    "days": 1,
    "pages": 3,
    "blank_pages": 1,
    "words": 64,
    "highlighted_words": 3,
    "needs_review": 8
  },
  "days": [
    {
      "date": "2026-05-15",
      "pages": [2, 3],
      "words": [
        {
          "raw_ordinal": "b",
          "corrected_ordinal": 6,
          "word_normalized": "objection",
          "is_highlighted": false
        }
      ]
    }
  ]
}
```

The main word sequence excludes OCR lines without a raw ordinal by default so corrected ordinals stay accurate. Those lines are preserved under `unassigned_words`; pass `--include-missing-ordinal` only when you want them in the main sequence.

`correct_words.py` reads `apple_words_shortdate.json`, preserves the OCR fields, and adds three correction results for Web UI review:

```json
{
  "word_normalized_raw": "jonrney",
  "word_normalized": "journey",
  "correction_selected": "hybrid",
  "corrections": {
    "local": {"status": "auto_corrected", "word": "journey"},
    "network": {"status": "candidate_review", "word": "journey"},
    "hybrid": {"status": "auto_corrected", "word": "journey"}
  }
}
```

Local correction uses `wordfreq`; network checks use `dictionaryapi.dev` and are cached in `outputs/correction_cache.sqlite`. Add one word per line to `wordlists/custom_words.txt` for future exam-specific supplements.

`dictionary_cache.py` writes/reads `outputs/dictionary_cache.sqlite`. It first checks the local cache by `word_normalized`; cache misses call the configured provider unless `--offline` is set. Use `--deepseek on` with `DEEPSEEK_API_KEY` when you want Chinese meanings and exam-level examples.

## Caveats

- Confidence scores are **not directly comparable** across engines — each engine uses its own scoring metric.
- Apple Vision OCR only works on macOS.
- `paddleocr` / `paddlepaddle` installation can be slow on first run.
- The PDF `../单词.pdf` should be a standard (non-scanned) PDF to get comparable baseline results.
