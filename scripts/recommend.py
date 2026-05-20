#!/usr/bin/env python3
"""Run recommendations from the command line. Prefer: rwrt-recommend"""

from rwrt.cli import recommend_main

if __name__ == "__main__":
    raise SystemExit(recommend_main())
