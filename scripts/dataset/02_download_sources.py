"""Stage 2: download LaTeX source archives for the candidate query papers.

Reads stage 1's data/raw/query_candidates.jsonl, downloads each paper's
e-print source from arXiv, and records the outcome in an append-only index so
interrupted runs resume where they left off.

Outputs:
- data/raw/sources/{arxiv_id}.tar.gz   the raw archive (gzip; gitignored)
- data/raw/sources_index.jsonl         one record per attempted paper:
      {"arxiv_id", "status", "file", "size"}

Statuses: "ok" (gzip source saved), "pdf_only" (no LaTeX — excluded per the
inclusion rules), "not_found" (withdrawn/404), "unknown_format", "error"
(transient; retried on a rerun with --retry-failed). Only "ok" papers proceed
to stage 3.

This is the bandwidth-heavy stage (~1 paper / 3s). Pilot first:

    uv run --extra dataset python scripts/dataset/02_download_sources.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arxiv_api import ArxivClient  # noqa: E402
from common import RAW_DIR, read_jsonl  # noqa: E402

DEFAULT_CANDIDATES = RAW_DIR / "query_candidates.jsonl"
DEFAULT_SOURCES_DIR = RAW_DIR / "sources"
DEFAULT_INDEX = RAW_DIR / "sources_index.jsonl"

# Outcomes that should not be re-attempted on rerun ("error" is retryable).
_PERMANENT = {"ok", "pdf_only", "not_found", "unknown_format"}


def classify(content: bytes) -> str:
    """Identify the e-print payload by magic bytes."""
    if content[:2] == b"\x1f\x8b":
        return "gzip"
    if content[:4] == b"%PDF":
        return "pdf"
    if len(content) > 262 and content[257:262] == b"ustar":
        return "tar"
    return "unknown"


def load_index(path: Path) -> dict[str, dict]:
    """Latest outcome per arxiv_id (append-only file: last record wins)."""
    if not path.exists():
        return {}
    return {record["arxiv_id"]: record for record in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download arXiv LaTeX sources.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES),
                        help="Stage 1 output JSONL.")
    parser.add_argument("--sources-dir", default=str(DEFAULT_SOURCES_DIR),
                        help="Directory for downloaded archives.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX),
                        help="Append-only download index JSONL.")
    parser.add_argument("--limit", type=int,
                        help="Download at most this many new papers this run.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt papers previously recorded as 'error'.")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        sys.exit(f"Candidates file not found: {candidates_path} (run stage 1 first).")

    sources_dir = Path(args.sources_dir)
    sources_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index)
    index = load_index(index_path)

    client = ArxivClient()
    attempted = 0
    counts: dict[str, int] = {}

    with index_path.open("a", encoding="utf-8") as index_file:

        def record(arxiv_id: str, status: str, file: str | None = None, size: int = 0) -> None:
            entry = {"arxiv_id": arxiv_id, "status": status, "file": file, "size": size}
            index_file.write(json.dumps(entry) + "\n")
            index_file.flush()
            counts[status] = counts.get(status, 0) + 1

        for candidate in read_jsonl(candidates_path):
            arxiv_id = candidate["arxiv_id"]
            previous = index.get(arxiv_id)
            if previous is not None:
                if previous["status"] in _PERMANENT:
                    continue
                if previous["status"] == "error" and not args.retry_failed:
                    continue

            if args.limit is not None and attempted >= args.limit:
                break
            attempted += 1

            try:
                content = client.download_source(arxiv_id)
            except FileNotFoundError:
                record(arxiv_id, "not_found")
                continue
            except RuntimeError as exc:
                print(f"  {arxiv_id}: {exc}")
                record(arxiv_id, "error")
                continue

            kind = classify(content)
            if kind == "pdf":
                record(arxiv_id, "pdf_only")
                continue
            if kind == "unknown":
                record(arxiv_id, "unknown_format")
                continue

            # gzip (tarball or single gzipped .tex) or bare tar — stage 3 untangles.
            out_path = sources_dir / f"{arxiv_id.replace('/', '_')}.tar.gz"
            out_path.write_bytes(content)
            record(arxiv_id, "ok", file=out_path.name, size=len(content))
            print(f"  {arxiv_id}: saved {len(content) / 1024:.0f} KiB")

    total_ok = counts.get("ok", 0)
    print(
        f"Attempted {attempted} downloads: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if counts
        else "Nothing to do (all candidates already processed)."
    )
    print(f"Index: {index_path} | archives in {sources_dir} ({total_ok} new)")


if __name__ == "__main__":
    main()
