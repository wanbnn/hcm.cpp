#!/usr/bin/env python3
"""Blind human-evaluation harness for fluency.

The project does not fabricate human ratings. This script creates a blind rating
sheet from candidate pairs and scores the returned sheet. Ratings are collected
outside the system (people read the sheet and fill the fields).

Input (generate): JSONL with {"id", "prompt", "system_a", "system_b"}.
Output (generate): JSONL rating sheet with empty rater/ratings.
Input (score): the filled rating sheet (one or more raters per id).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def generate(args):
    rows = read_jsonl(args.candidates)
    out = []
    for row in rows:
        out.append({
            "id": row["id"],
            "prompt": row.get("prompt", ""),
            "a": row["system_a"],
            "b": row["system_b"],
            "rater": "",
            "rating_a": None,   # 1..5
            "rating_b": None,   # 1..5
            "preference": None, # "a", "b" or "tie"
            "comment": "",
        })
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    print(f"rating sheet: {args.out} ({len(out)} items)")


def cohen_kappa(pairs):
    if not pairs:
        return None
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / len(pairs)
    counts = defaultdict(int)
    for a, b in pairs:
        counts[a] += 1
        counts[b] += 1
    total = 2 * len(pairs)
    pe = sum((c / total) ** 2 for c in counts.values())
    if math.isclose(pe, 1.0):
        return 1.0
    return (po - pe) / (1 - pe)


def score(args):
    rows = read_jsonl(args.ratings)
    by_id = defaultdict(list)
    for row in rows:
        if row.get("rating_a") is None or row.get("rating_b") is None:
            continue
        by_id[row["id"]].append(row)
    a_scores, b_scores, preferences = [], [], []
    for items in by_id.values():
        for item in items:
            a_scores.append(float(item["rating_a"]))
            b_scores.append(float(item["rating_b"]))
            preferences.append(item.get("preference") or "tie")
    # Inter-rater agreement for IDs rated by at least two raters.
    pairs = []
    for items in by_id.values():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pa = items[i].get("preference") or "tie"
                pb = items[j].get("preference") or "tie"
                pairs.append((pa, pb))
    report = {
        "items": len(by_id),
        "ratings": len(a_scores),
        "mean_rating_a": round(sum(a_scores) / len(a_scores), 4) if a_scores else None,
        "mean_rating_b": round(sum(b_scores) / len(b_scores), 4) if b_scores else None,
        "preference_a": preferences.count("a"),
        "preference_b": preferences.count("b"),
        "ties": preferences.count("tie"),
        "cohen_kappa_preference": None if cohen_kappa(pairs) is None else round(cohen_kappa(pairs), 4),
        "note": "Human ratings are only present when returned by real raters; empty sheets are not scored.",
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    Path(args.out).write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--candidates", required=True)
    gen.add_argument("--out", required=True)
    gen.set_defaults(func=generate)
    sc = sub.add_parser("score")
    sc.add_argument("--ratings", required=True)
    sc.add_argument("--out", required=True)
    sc.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
