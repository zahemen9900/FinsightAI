#!/usr/bin/env python3
"""Script counterpart for notebooks/fina_q_and_a.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List


def inspect_financial_qa_csv(dataset_path: Path, preview_rows: int = 5) -> Dict[str, Any]:
    """Load CSV and return notebook-equivalent inspection data."""
    import pandas as pd

    dataframe = pd.read_csv(dataset_path)
    return {
        "shape": dataframe.shape,
        "columns": dataframe.columns.tolist(),
        "preview": dataframe.head(preview_rows),
        "first_row": dataframe.iloc[0, :].to_dict() if not dataframe.empty else {},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the Financial-QA CSV similarly to fina_q_and_a.ipynb."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to Financial-QA CSV (for example: /path/to/Financial-QA-10k.csv).",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of rows to print in the preview table (default: 5).",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    result = inspect_financial_qa_csv(args.dataset_path, preview_rows=args.preview_rows)
    print(f"Dataset shape: {result['shape']}")
    print(f"Columns: {result['columns']}")
    print("\nPreview:")
    print(result["preview"].to_string(index=False))
    print("\nFirst row:")
    print(result["first_row"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
