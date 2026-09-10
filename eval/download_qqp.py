"""Download the Quora Question Pairs dataset.

The canonical sources (huggingface.co, the original Quora/GLUE mirrors) are
unreachable from this sandbox's egress allowlist. This pulls the same
dataset from a GitHub-hosted mirror instead (raw.githubusercontent.com is
reachable) — same file, same 404,351 labeled question pairs, different host.
"""
from __future__ import annotations

import pathlib
import sys

import httpx

MIRROR_URL = (
    "https://raw.githubusercontent.com/tgaddair/quora-duplicate-question-detector/"
    "master/data/quora_duplicate_questions.tsv"
)
OUT_PATH = pathlib.Path(__file__).parent / "data" / "qqp_raw.tsv"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"already have {OUT_PATH}, skipping download")
        return
    print(f"downloading QQP mirror from {MIRROR_URL}")
    with httpx.stream("GET", MIRROR_URL, timeout=60.0, follow_redirects=True) as r:
        r.raise_for_status()
        with open(OUT_PATH, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
