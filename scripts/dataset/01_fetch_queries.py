"""Stage 1: discover candidate query papers on arXiv.

Searches cs.CL / cs.AI for papers submitted after the cutoff (default
2025-09-01, per data/README.md) that look like LLM-agent work, and writes
their metadata to data/raw/query_candidates.jsonl for stage 2 (LaTeX source
download).

The search is chunked into one query per calendar month. This keeps arXiv
pagination offsets small (deep offsets like start=5000 trigger rate limiting
and flaky responses) and makes the stage resumable: candidates are appended
and flushed as found, completed months are recorded in a sidecar progress
file, and a rerun after a crash skips finished months and dedupes the rest.

Two-level filtering keeps the arXiv traffic manageable:
- server side: category + submission date + a broad 'agent' keyword clause
- client side: a stricter heuristic requiring both an agent-ish and an
  LLM-ish term in the title+abstract

Pilot run (a few minutes, ~50 papers scanned):

    uv run --extra dataset python scripts/dataset/01_fetch_queries.py --limit 50

Full run: drop --limit. Expect tens of minutes (arXiv allows ~1 request / 3s);
interrupt and rerun freely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arxiv_api import ArxivClient, build_search_query  # noqa: E402
from common import RAW_DIR, read_jsonl  # noqa: E402

DEFAULT_OUT = RAW_DIR / "query_candidates.jsonl"
DEFAULT_CATEGORIES = ["cs.CL", "cs.AI"]
DEFAULT_FROM = "2025-09-01"

# Server-side clause: broad on purpose — precision comes from the client filter.
SERVER_KEYWORD_CLAUSE = "(ti:agent OR ti:agents OR ti:agentic OR abs:agent OR abs:agents OR abs:agentic)"

# Client-side heuristic: an LLM-agent paper should mention both concepts.
_AGENT_TERMS = re.compile(r"\bagent(s|ic)?\b|\bagent-based\b", re.IGNORECASE)
_LLM_TERMS = re.compile(
    r"\bLLMs?\b|\blarge language model|\blanguage model|\bfoundation model|\bGPT\b|\btool[ -]use\b",
    re.IGNORECASE,
)


def looks_like_llm_agent_paper(title: str, abstract: str) -> bool:
    text = f"{title} {abstract}"
    return bool(_AGENT_TERMS.search(text)) and bool(_LLM_TERMS.search(text))


def month_windows(date_from: str, date_to: str) -> list[tuple[str, str]]:
    """Split an inclusive date range into per-calendar-month (start, end) pairs."""
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    windows: list[tuple[str, str]] = []
    current = start
    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        window_end = min(next_month - timedelta(days=1), end)
        windows.append((current.isoformat(), window_end.isoformat()))
        current = next_month
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch candidate query papers from arXiv.")
    parser.add_argument("--from", dest="date_from", default=DEFAULT_FROM,
                        help=f"Earliest submission date, YYYY-MM-DD (default {DEFAULT_FROM}).")
    parser.add_argument("--to", dest="date_to", default=date.today().isoformat(),
                        help="Latest submission date, YYYY-MM-DD (default today).")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                        help=f"arXiv categories (default {' '.join(DEFAULT_CATEGORIES)}).")
    parser.add_argument("--limit", type=int,
                        help="Stop after scanning this many search results (pilot runs).")
    parser.add_argument("--broad", action="store_true",
                        help="Skip the server-side keyword clause (much more traffic).")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(str(out_path) + ".progress")

    seen = {r["arxiv_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    done_windows = (
        set(progress_path.read_text(encoding="utf-8").split()) if progress_path.exists() else set()
    )
    if seen:
        print(f"Resuming: {len(seen)} candidates already collected, "
              f"{len(done_windows)} month windows complete.")

    client = ArxivClient()
    scanned = 0
    kept = 0
    cut_short = False

    with out_path.open("a", encoding="utf-8") as out_file, \
            progress_path.open("a", encoding="utf-8") as progress_file:
        for window_start, window_end in month_windows(args.date_from, args.date_to):
            tag = f"{window_start}:{window_end}"
            if tag in done_windows:
                continue
            if cut_short:
                break

            query = build_search_query(
                args.categories,
                window_start,
                window_end,
                extra_clause=None if args.broad else SERVER_KEYWORD_CLAUSE,
            )
            print(f"window {tag}...")

            for paper in client.search(query):
                if args.limit is not None and scanned >= args.limit:
                    cut_short = True
                    break
                scanned += 1
                if scanned % 200 == 0:
                    print(f"  scanned {scanned} results, kept {kept}...")
                if paper.arxiv_id in seen or not paper.abstract:
                    continue
                if not looks_like_llm_agent_paper(paper.title, paper.abstract):
                    continue
                seen.add(paper.arxiv_id)
                out_file.write(json.dumps(paper.to_dict(), ensure_ascii=False) + "\n")
                out_file.flush()
                kept += 1

            if not cut_short:
                progress_file.write(tag + "\n")
                progress_file.flush()
                print(f"  window {tag} done (total: scanned {scanned}, kept {kept})")

    status = "stopped at --limit" if cut_short else "all windows complete"
    print(f"{status}: scanned {scanned} new results, kept {kept} -> {out_path}")
    print(f"Total candidates on disk: {len(seen)}")


if __name__ == "__main__":
    main()
