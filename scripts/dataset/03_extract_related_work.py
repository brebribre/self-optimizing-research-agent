"""Stage 3: extract related-work sections + citations from downloaded sources.

Reads stage 2's download index, parses each paper's LaTeX project, and applies
the inclusion filters from data/README.md:

- an explicit \\section-level Related Work section exists
- it cites 5-15 unique works
- the section is in English
- the cite keys resolve against the project's .bib/.bbl bibliography

Every attempted paper gets one record in the output (append-only, resumable);
rows with status "ok" carry the extracted section and bibliography and proceed
to stage 4 (citation resolution). Other statuses record why the paper was
rejected, which makes yield problems diagnosable.

    uv run --extra dataset python scripts/dataset/03_extract_related_work.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROCESSED_DIR, RAW_DIR, read_jsonl  # noqa: E402
from latex_extract import (  # noqa: E402
    build_document,
    extract_cite_keys,
    find_related_work,
    is_english,
    load_tex_project,
    parse_bibliographies,
)

DEFAULT_CANDIDATES = RAW_DIR / "query_candidates.jsonl"
DEFAULT_SOURCES_DIR = RAW_DIR / "sources"
DEFAULT_SOURCES_INDEX = RAW_DIR / "sources_index.jsonl"
DEFAULT_OUT = PROCESSED_DIR / "related_work_extracted.jsonl"

MIN_CITATIONS = 5
MAX_CITATIONS = 15

# Drop a paper when more than this fraction of its cite keys lack a bib entry.
MAX_MISSING_KEY_FRACTION = 0.2


def extract_one(archive_path: Path, candidate: dict) -> dict:
    """Run the full extraction for one paper; returns the output record."""
    record = {
        "arxiv_id": candidate["arxiv_id"],
        "title": candidate["title"],
        "abstract": candidate["abstract"],
        "submitted": candidate["submitted"],
    }

    try:
        files = load_tex_project(archive_path)
    except Exception as exc:  # noqa: BLE001 — malformed archives are data, not bugs
        return {**record, "status": "archive_error", "detail": f"{type(exc).__name__}: {exc}"}

    document = build_document(files)
    if not document:
        return {**record, "status": "no_tex_files"}

    section = find_related_work(document)
    if section is None:
        return {**record, "status": "no_related_work_section"}
    section_title, body = section

    cite_keys = extract_cite_keys(body)
    if len(cite_keys) < MIN_CITATIONS:
        return {**record, "status": "too_few_citations", "n_citations": len(cite_keys)}
    if len(cite_keys) > MAX_CITATIONS:
        return {**record, "status": "too_many_citations", "n_citations": len(cite_keys)}

    if not is_english(body):
        return {**record, "status": "not_english"}

    bibliography = parse_bibliographies(files)
    entries = {key: bibliography[key] for key in cite_keys if key in bibliography}
    missing = [key for key in cite_keys if key not in bibliography]
    if not entries:
        return {**record, "status": "no_bibliography"}
    if len(missing) / len(cite_keys) > MAX_MISSING_KEY_FRACTION:
        return {**record, "status": "missing_bib_keys", "missing": missing}

    return {
        **record,
        "status": "ok",
        "section_title": section_title,
        "related_work_latex": body,
        "cite_keys": cite_keys,
        "bib_entries": entries,
        "missing_keys": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract related-work sections from sources.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES),
                        help="Stage 1 output JSONL (query metadata).")
    parser.add_argument("--sources-index", default=str(DEFAULT_SOURCES_INDEX),
                        help="Stage 2 download index JSONL.")
    parser.add_argument("--sources-dir", default=str(DEFAULT_SOURCES_DIR),
                        help="Directory of downloaded archives.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL (append-only).")
    parser.add_argument("--limit", type=int, help="Process at most this many new papers.")
    args = parser.parse_args()

    candidates = {c["arxiv_id"]: c for c in read_jsonl(args.candidates)}
    downloaded = [
        entry for entry in read_jsonl(args.sources_index) if entry["status"] == "ok"
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = (
        {record["arxiv_id"] for record in read_jsonl(out_path)} if out_path.exists() else set()
    )

    processed = 0
    counts: dict[str, int] = {}
    sources_dir = Path(args.sources_dir)

    with out_path.open("a", encoding="utf-8") as out_file:
        for entry in downloaded:
            arxiv_id = entry["arxiv_id"]
            if arxiv_id in done or arxiv_id not in candidates:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1

            record = extract_one(sources_dir / entry["file"], candidates[arxiv_id])
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

            marker = "+" if record["status"] == "ok" else "-"
            extra = f" ({len(record['cite_keys'])} cites)" if record["status"] == "ok" else ""
            print(f"  {marker} {arxiv_id}: {record['status']}{extra}")

    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Processed {processed} papers: {summary or 'nothing new'}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
