#!/usr/bin/env python3
"""Script counterpart for notebooks/reddit-250k-analysis.ipynb."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl_records(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_subreddit_samples(
    records: Iterable[Dict[str, Any]], per_subreddit: int = 2
) -> Dict[str, List[Dict[str, Any]]]:
    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        subreddit = record.get("subreddit")
        if not subreddit:
            continue
        if len(samples[subreddit]) < per_subreddit:
            samples[subreddit].append(record)
    return dict(samples)


def percentile_filter(
    dataframe: "pd.DataFrame", quantile: float = 0.8
) -> tuple["pd.DataFrame", Dict[str, float]]:
    required_columns = ["z_score", "combined_score", "comment_normalized_score"]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Input data missing required score columns: {missing}")

    thresholds = {
        column: float(dataframe[column].quantile(quantile)) for column in required_columns
    }
    filtered = dataframe[
        (dataframe["z_score"] > thresholds["z_score"])
        & (dataframe["combined_score"] > thresholds["combined_score"])
        & (
            dataframe["comment_normalized_score"]
            > thresholds["comment_normalized_score"]
        )
    ]
    return filtered, thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Reddit finance JSONL data and optionally export percentile-filtered output."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to reddit-finance Data.jsonl file.",
    )
    parser.add_argument(
        "--filtered-output-jsonl",
        type=Path,
        default=None,
        help="Optional path to save percentile-filtered output JSONL.",
    )
    parser.add_argument(
        "--sample-per-subreddit",
        type=int,
        default=2,
        help="How many sample rows to retain per subreddit for display (default: 2).",
    )
    parser.add_argument(
        "--print-selftext-preview",
        type=int,
        default=0,
        help="If > 0, print this many selftext rows from filtered data.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    import pandas as pd

    records = load_jsonl_records(args.input_jsonl)
    print(f"Total samples: {len(records)}")
    if not records:
        print("No records found.")
        return 0

    print("First sample:")
    print(records[0])

    unique_subreddits = sorted(
        {record.get("subreddit") for record in records if record.get("subreddit")}
    )
    print(f"Number of unique subreddits: {len(unique_subreddits)}")
    print(f"Unique subreddits: {unique_subreddits}")

    subreddit_samples = collect_subreddit_samples(
        records, per_subreddit=max(args.sample_per_subreddit, 1)
    )
    for subreddit, sample_rows in subreddit_samples.items():
        print(f"Subreddit: {subreddit}, Samples: {len(sample_rows)}")

    dataframe = pd.DataFrame(records)
    filtered_df, thresholds = percentile_filter(dataframe, quantile=0.8)
    print(f"Thresholds: {thresholds}")
    print(f"Number of filtered samples: {len(filtered_df)}")

    if args.filtered_output_jsonl:
        args.filtered_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_json(
            args.filtered_output_jsonl, orient="records", lines=True, force_ascii=False
        )
        print(f"Saved filtered data to: {args.filtered_output_jsonl}")

    if args.print_selftext_preview > 0 and "selftext" in filtered_df.columns:
        preview = (
            filtered_df["selftext"]
            .head(args.print_selftext_preview)
            .fillna("")
            .tolist()
        )
        print("\nFiltered selftext preview:")
        for index, value in enumerate(preview, start=1):
            print(f"{index:02d}. {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
