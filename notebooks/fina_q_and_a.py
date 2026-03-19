#!/usr/bin/env python3
"""Backward-compatible wrapper for financial_q_and_a.py."""

from __future__ import annotations

try:
    from notebooks.financial_q_and_a import main
except ModuleNotFoundError:
    from financial_q_and_a import main


if __name__ == "__main__":
    raise SystemExit(main())
