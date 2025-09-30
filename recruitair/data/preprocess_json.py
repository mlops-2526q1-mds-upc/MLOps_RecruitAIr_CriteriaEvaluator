#!/usr/bin/env python3
"""
Preprocess job JSONs into JSONL with resume + criteria + score.

Input:
 - JSON files (ej: match_X.json / mismatch_X.json)
Output:
 - JSONL file, each line:
    {
      "resume": "...",
      "criteria": "leadership",
      "score": 2
    }
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional
from recruitair.config.data_preprocess_config import RAW_DATA_DIR, INTERIM_DATA_DIR

FNAME_RE = re.compile(r'^(?P<label>match|mismatch)_(?P<num>\d+)\.json$', re.IGNORECASE)


def find_target_json_files(input_dir: Path) -> List[Path]:
    """Find all match_X.json and mismatch_X.json files under input_dir."""
    return sorted([p for p in input_dir.rglob("*.json") if FNAME_RE.match(p.name)])


def process_file(path: Path) -> List[dict]:
    """Load JSON and extract resume + criteria + score."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception as e:
        print(f"WARNING: Failed to parse {path}: {e}", file=sys.stderr)
        return []

    inp = obj.get("input", {})
    out = obj.get("output", {})

    resume = inp.get("resume", "").strip()

    scores = out.get("scores", {})
    macro_scores = scores.get("macro_scores", []) or []
    micro_scores = scores.get("micro_scores", []) or []

    rows = []

    # Usamos macro_scores y micro_scores; puedes comentar uno si solo quieres uno
    for entry in macro_scores + micro_scores:
        criteria = entry.get("criteria")
        score = entry.get("score")
        if criteria is not None and score is not None:
            rows.append({
                "resume": resume,
                "criteria": criteria,
                "score": score
            })

    return rows


def main():
    p = argparse.ArgumentParser(description="Preprocess jobs JSONs into JSONL")
    p.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=RAW_DATA_DIR / "raw_jsons",
        help="Directory containing JSON files (default: RAW_DATA_DIR/raw_jsons)",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=INTERIM_DATA_DIR / "preprocessed_cvs.jsonl",
        help="Output JSONL file (default: INTERIM_DATA_DIR/preprocessed_cvs.jsonl)",
    )
    args = p.parse_args()

    input_dir: Path = args.input_dir
    out_jsonl: Path = args.output_jsonl

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(2)

    files = find_target_json_files(input_dir)
    if not files:
        print(f"No files matching match_X.json or mismatch_X.json found under {input_dir}", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(files)} JSON files. Processing...")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, fp in enumerate(files, start=1):
            rows = process_file(fp)
            for row in rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if idx % 100 == 0 or idx == len(files):
                print(f"Progress: {idx}/{len(files)} processed.")

    print(f"✅ Done. JSONL written to: {out_jsonl}")


if __name__ == "__main__":
    main()
