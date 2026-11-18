#!/usr/bin/env python3
"""Lightweight utility to clean raw .txt books and validate .jsonl dialogue datasets."""

import argparse
import json
import re
from pathlib import Path


def clean_text(raw: str) -> str:
    text = re.sub(r"\r\n?", "\n", raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare text and dialogue datasets for cognicore.")
    parser.add_argument("--input", required=True, help="Input .txt or .jsonl file")
    parser.add_argument("--output", required=True, help="Cleaned output path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix == ".jsonl":
        valid_lines = []
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" in obj and "intent" in obj:
                    obj["text"] = clean_text(str(obj["text"]))
                    valid_lines.append(json.dumps(obj, ensure_ascii=False))
        out_path.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
    else:
        raw = in_path.read_text(encoding="utf-8", errors="ignore")
        out_path.write_text(clean_text(raw) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
