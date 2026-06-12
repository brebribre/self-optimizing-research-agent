"""Stage 4: resolve cite keys to arXiv papers with abstracts.

Reads stage 3's "ok" rows and resolves each cite key to a concrete arXiv paper
(the final schema requires an arxiv_id + abstract per pool entry, and stage 5
needs publication dates for the distractor date filter). Three tiers:

1. arXiv id embedded in the bib entry (eprint field, "arXiv preprint", URLs)
   — free, no network.
2. Semantic Scholar title match (keyless OK; uses S2_API_KEY from .env if set).
3. arXiv API title search, fuzzy-matched against the bib title.

Every candidate id is then validated in one batched arXiv metadata fetch per
row: the resolved entry's canonical title must agree with the bib title
(token similarity), which catches wrong eprint fields and bad search hits.

Row filters (configurable below): >= 80% of cites resolved, and the resolved
count must still be within the 5-15 range from data/README.md.

Successful title resolutions are cached in data/raw/resolution_cache.jsonl so
famous papers cited by many rows are looked up once. Output is append-only and
resumable, like stages 2-3.

    uv run --extra dataset python scripts/dataset/04_resolve_citations.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arxiv_api import ArxivClient  # noqa: E402
from common import PROCESSED_DIR, RAW_DIR, read_jsonl  # noqa: E402
from s2_api import SemanticScholarClient  # noqa: E402

DEFAULT_IN = PROCESSED_DIR / "related_work_extracted.jsonl"
DEFAULT_OUT = PROCESSED_DIR / "citations_resolved.jsonl"
DEFAULT_CACHE = RAW_DIR / "resolution_cache.jsonl"

MIN_RESOLVED_FRACTION = 0.8
MIN_RESOLVED = 5
MAX_RESOLVED = 15

# Bib title vs. arXiv canonical title agreement (token-set similarity).
TITLE_SIMILARITY_THRESHOLD = 0.6

# New-style arXiv ids in bib entry text: eprint fields, "arXiv:NNNN.NNNNN",
# arxiv.org URLs. (Old-style ids like cs/0112017 are out of scope — this
# subfield's citations are post-2007.)
_ARXIV_ID = re.compile(
    r"(?:arxiv(?:\.org)?[:/\s]*(?:abs/|pdf/)?|eprint\s*[=:]?\s*)(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)


def _title_tokens(title: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", title.lower()) if len(t) > 2}


def titles_agree(a: str, b: str) -> bool:
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / max(len(ta), len(tb)) >= TITLE_SIMILARITY_THRESHOLD


def extract_embedded_arxiv_id(raw_entry: str) -> str | None:
    match = _ARXIV_ID.search(raw_entry)
    return match.group(1) if match else None


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {r["title_norm"]: r["arxiv_id"] for r in read_jsonl(path)}


def normalize_title(title: str) -> str:
    return " ".join(sorted(_title_tokens(title)))


def resolve_row(
    row: dict,
    arxiv: ArxivClient,
    s2: SemanticScholarClient | None,
    title_cache: dict[str, str],
    cache_file,
    metadata_cache: dict[str, dict],
) -> dict:
    """Resolve all cite keys of one stage-3 row. Returns the output record."""
    candidates: dict[str, tuple[str, str]] = {}  # key -> (arxiv_id, tier)
    unresolved: dict[str, str] = {}

    for key in row["cite_keys"]:
        entry = row["bib_entries"].get(key)
        if entry is None:
            unresolved[key] = "no_bib_entry"
            continue
        bib_title = entry.get("title", "")

        # Tier 1: id embedded in the bib entry itself.
        embedded = extract_embedded_arxiv_id(entry.get("raw", ""))
        if embedded:
            candidates[key] = (embedded, "bib")
            continue
        if not bib_title:
            unresolved[key] = "no_title"
            continue

        # Cache from earlier rows/runs.
        cached = title_cache.get(normalize_title(bib_title))
        if cached:
            candidates[key] = (cached, "cache")
            continue

        # Tier 2: Semantic Scholar title match.
        if s2 is not None:
            try:
                match = s2.match_title(bib_title)
            except RuntimeError as exc:
                print(f"    S2 gave up for {key!r}: {exc}")
                match = None
            if match and match["arxiv_id"] and titles_agree(bib_title, match["title"]):
                candidates[key] = (match["arxiv_id"], "s2")
                continue
            if match and not match["arxiv_id"] and titles_agree(bib_title, match["title"]):
                unresolved[key] = "not_on_arxiv"
                continue

        # Tier 3: arXiv title search.
        found = None
        for paper in arxiv.search(f'ti:"{bib_title}"', max_results=5):
            if titles_agree(bib_title, paper.title):
                found = paper.arxiv_id
                break
        if found:
            candidates[key] = (found, "arxiv_search")
        else:
            unresolved[key] = "no_match"

    # Validate every candidate id against canonical arXiv metadata (one
    # batched request per row for ids we haven't fetched before).
    to_fetch = sorted({aid for aid, _ in candidates.values() if aid not in metadata_cache})
    if to_fetch:
        for paper in arxiv.fetch_by_ids(to_fetch):
            metadata_cache[paper.arxiv_id] = {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "published": paper.submitted,
            }

    resolved: dict[str, dict] = {}
    tiers: dict[str, str] = {}
    for key, (arxiv_id, tier) in candidates.items():
        meta = metadata_cache.get(arxiv_id)
        if meta is None or not meta["abstract"]:
            unresolved[key] = "arxiv_fetch_failed"
            continue
        bib_title = row["bib_entries"][key].get("title", "")
        # Verify unless the bib entry had no usable title (id-only .bbl entry).
        if bib_title and not titles_agree(bib_title, meta["title"]):
            unresolved[key] = "title_mismatch"
            continue
        if arxiv_id == row["arxiv_id"]:
            unresolved[key] = "self_citation"
            continue
        resolved[key] = meta
        tiers[key] = tier
        if bib_title and tier in ("s2", "arxiv_search"):
            norm = normalize_title(bib_title)
            if norm not in title_cache:
                title_cache[norm] = arxiv_id
                cache_file.write(json.dumps({"title_norm": norm, "arxiv_id": arxiv_id}) + "\n")
                cache_file.flush()

    record = {
        "arxiv_id": row["arxiv_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "submitted": row["submitted"],
        "section_title": row["section_title"],
        "related_work_latex": row["related_work_latex"],
        "cite_keys": row["cite_keys"],
        "resolved": resolved,
        "unresolved": unresolved,
        "tiers": tiers,
    }

    fraction = len(resolved) / len(row["cite_keys"]) if row["cite_keys"] else 0.0
    if fraction < MIN_RESOLVED_FRACTION:
        return {**record, "status": "low_resolution", "resolved_fraction": round(fraction, 3)}
    if len(resolved) < MIN_RESOLVED:
        return {**record, "status": "too_few_resolved"}
    if len(resolved) > MAX_RESOLVED:  # defensive; stage 3 already capped raw cites
        return {**record, "status": "too_many_resolved"}
    return {**record, "status": "ok", "resolved_fraction": round(fraction, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve citations to arXiv papers.")
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN),
                        help="Stage 3 output JSONL.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL (append-only).")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE),
                        help="Title-resolution cache JSONL.")
    parser.add_argument("--limit", type=int, help="Process at most this many new rows.")
    parser.add_argument("--no-s2", action="store_true",
                        help="Skip Semantic Scholar (arXiv-only resolution).")
    args = parser.parse_args()

    load_dotenv()
    s2 = None
    if not args.no_s2:
        api_key = os.getenv("S2_API_KEY")
        s2 = SemanticScholarClient(api_key=api_key)
        print(f"Semantic Scholar: {'authenticated' if api_key else 'keyless (shared pool)'}")

    arxiv = ArxivClient()
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    title_cache = load_cache(cache_path)
    metadata_cache: dict[str, dict] = {}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["arxiv_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    rows = [r for r in read_jsonl(args.in_path) if r["status"] == "ok"]
    processed = 0
    counts: dict[str, int] = {}

    with out_path.open("a", encoding="utf-8") as out_file, \
            cache_path.open("a", encoding="utf-8") as cache_file:
        for row in rows:
            if row["arxiv_id"] in done:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1

            print(f"  {row['arxiv_id']}: resolving {len(row['cite_keys'])} citations...")
            record = resolve_row(row, arxiv, s2, title_cache, cache_file, metadata_cache)
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

            tier_summary = ", ".join(
                f"{t}={list(record['tiers'].values()).count(t)}"
                for t in ("bib", "cache", "s2", "arxiv_search")
                if t in record["tiers"].values()
            )
            print(
                f"    -> {record['status']}: {len(record['resolved'])}/{len(row['cite_keys'])} "
                f"resolved ({tier_summary or 'none'}); "
                f"unresolved: {sorted(set(record['unresolved'].values())) or 'none'}"
            )

    print(f"Processed {processed} rows: "
          + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing new"))
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
