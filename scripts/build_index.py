#!/usr/bin/env python3
"""Build the FAISS embedding index. Prefer: rwrt-build-index"""

from rwrt.cli import build_index_main

if __name__ == "__main__":
    raise SystemExit(build_index_main())
