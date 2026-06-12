"""Stage 1: discover candidate query papers on arXiv.

Searches cs.CL / cs.AI for papers submitted after the cutoff (default
2025-09-01, per data/README.md) that look like LLM-agent work, and writes
their metadata to data/raw/query_candidates.jsonl for stage 2 (LaTeX source
download).

Two-level filtering keeps the arXiv traffic manageable:
- server side: category + submission date + a broad 'agent' keyword clause
- client side: a stricter heuristic requiring both an agent-ish and an
  LLM-ish term in the title+abstract

Pilot run (a few minutes, ~50 papers scanned):

    uv run --extra dataset python scripts/dataset/01_fetch_queries.py --limit 50

Full run: drop --limit. Expect tens of minutes (arXiv allows ~1 request / 3s).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arxiv_api import ArxivClient, build_search_query  # noqa: E402
from common import RAW_DIR, write_jsonl  # noqa: E402

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

    query = build_search_query(
        args.categories,
        args.date_from,
        args.date_to,
        extra_clause=None if args.broad else SERVER_KEYWORD_CLAUSE,
    )
    print(f"arXiv query: {query}")

    client = ArxivClient()
    seen: set[str] = set()
    candidates: list[dict] = []
    scanned = 0

    for paper in client.search(query, max_results=args.limit):
        scanned += 1
        if scanned % 200 == 0:
            print(f"  scanned {scanned} results, kept {len(candidates)}...")
        if paper.arxiv_id in seen:
            continue
        seen.add(paper.arxiv_id)
        if not paper.abstract:
            continue
        if not looks_like_llm_agent_paper(paper.title, paper.abstract):
            continue
        candidates.append(paper.to_dict())

    count = write_jsonl(args.out, candidates)
    print(f"Scanned {scanned} results; kept {count} LLM-agent candidates -> {args.out}")


if __name__ == "__main__":
    main()
