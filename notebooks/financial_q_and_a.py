#!/usr/bin/env python3
"""Compatibility wrapper with clearer naming for Financial Q&A inspection."""

try:
    from notebooks.fina_q_and_a import main
except ModuleNotFoundError:
    from fina_q_and_a import main


if __name__ == "__main__":
    raise SystemExit(main())
