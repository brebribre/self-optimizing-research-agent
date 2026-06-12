"""Minimal arXiv API client for the dataset-collection pipeline.

Wraps the arXiv Atom API (https://info.arxiv.org/help/api/) with the two
operations the pipeline needs:

- `ArxivClient.search(query)`      — paginated search (stage 1: find queries)
- `ArxivClient.fetch_by_ids(ids)`  — metadata lookup (stage 4: resolve cites)

The client enforces arXiv's politeness rules: a single connection, >= 3 seconds
between requests, and exponential backoff on transient failures. Keep one
client instance per script run so the throttle spans all calls.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

API_URL = "https://export.arxiv.org/api/query"
EPRINT_URL = "https://export.arxiv.org/e-print/{arxiv_id}"

_ATOM = "{http://www.w3.org/2005/Atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# arXiv asks for no more than one request every 3 seconds.
DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_PAGE_SIZE = 200


@dataclass
class ArxivPaper:
    arxiv_id: str  # versionless, e.g. "2602.01234"
    version: int
    title: str
    abstract: str
    submitted: str  # YYYY-MM-DD of v1 submission
    primary_category: str
    categories: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "version": self.version,
            "title": self.title,
            "abstract": self.abstract,
            "submitted": self.submitted,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "authors": self.authors,
        }


def _collapse(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_entry(entry: ET.Element) -> ArxivPaper:
    raw_id = entry.findtext(f"{_ATOM}id") or ""
    # e.g. "http://arxiv.org/abs/2602.01234v2" -> ("2602.01234", 2)
    id_part = raw_id.rsplit("/abs/", 1)[-1]
    version_match = re.search(r"v(\d+)$", id_part)
    version = int(version_match.group(1)) if version_match else 1
    arxiv_id = re.sub(r"v\d+$", "", id_part)

    primary = entry.find(f"{_ARXIV}primary_category")
    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=version,
        title=_collapse(entry.findtext(f"{_ATOM}title")),
        abstract=_collapse(entry.findtext(f"{_ATOM}summary")),
        submitted=(entry.findtext(f"{_ATOM}published") or "")[:10],
        primary_category=primary.get("term", "") if primary is not None else "",
        categories=[
            c.get("term", "") for c in entry.findall(f"{_ATOM}category") if c.get("term")
        ],
        authors=[
            _collapse(a.findtext(f"{_ATOM}name"))
            for a in entry.findall(f"{_ATOM}author")
        ],
    )


def _parse_feed(xml_text: str) -> tuple[int, list[ArxivPaper]]:
    """Return (total_results, papers) from an Atom feed page."""
    root = ET.fromstring(xml_text)
    total = int(root.findtext(f"{_OPENSEARCH}totalResults") or 0)
    papers = [_parse_entry(e) for e in root.findall(f"{_ATOM}entry")]
    return total, papers


def build_search_query(
    categories: list[str],
    start_date: str,
    end_date: str,
    extra_clause: str | None = None,
) -> str:
    """Compose an arXiv `search_query` string.

    Dates are inclusive, formatted YYYY-MM-DD. `extra_clause` is ANDed on, e.g.
    a keyword restriction like '(ti:agent OR abs:agent)'.
    """
    cats = " OR ".join(f"cat:{c}" for c in categories)
    start = start_date.replace("-", "") + "0000"
    end = end_date.replace("-", "") + "2359"
    query = f"({cats}) AND submittedDate:[{start} TO {end}]"
    if extra_clause:
        query += f" AND {extra_clause}"
    return query


class ArxivClient:
    def __init__(
        self,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.page_size = page_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "self-optimizing-research-agent dataset pipeline "
            "(https://github.com/brebribre)"
        )
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

    def _get(self, params: dict) -> str:
        """One throttled GET with exponential backoff on transient failures.

        HTTP 429 gets a much longer cooldown than network errors — arXiv's
        rate limiter needs minutes, not seconds, to forgive sustained paging.
        """
        last_error: Exception | str | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                self._last_request_at = time.monotonic()
                response = self._session.get(API_URL, params=params, timeout=self.timeout)
                if response.status_code == 429:
                    last_error = "rate limited (429)"
                    cooldown = 60.0 * (attempt + 1)
                    print(f"  arXiv rate limit hit; cooling down {cooldown:.0f}s...")
                    time.sleep(cooldown)
                    continue
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = exc
                backoff = self.delay_seconds * 2**attempt
                print(f"  arXiv request failed ({exc}); retrying in {backoff:.0f}s...")
                time.sleep(backoff)
        raise RuntimeError(f"arXiv API request failed after {self.max_retries} retries: {last_error}")

    def search(self, query: str, max_results: int | None = None):
        """Yield `ArxivPaper`s for `query`, newest submissions first.

        Paginates transparently; stops at `max_results`, the feed's reported
        total, or an empty page (an occasional arXiv API hiccup — retried a
        few times before giving up).
        """
        start = 0
        yielded = 0
        empty_pages = 0
        while True:
            page_size = self.page_size
            if max_results is not None:
                page_size = min(page_size, max_results - yielded)
                if page_size <= 0:
                    return
            text = self._get(
                {
                    "search_query": query,
                    "start": start,
                    "max_results": page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
            )
            total, papers = _parse_feed(text)
            if not papers:
                # arXiv sometimes returns a transiently empty page mid-stream.
                empty_pages += 1
                if start >= total or empty_pages >= 3:
                    return
                continue
            empty_pages = 0
            for paper in papers:
                yield paper
                yielded += 1
                if max_results is not None and yielded >= max_results:
                    return
            start += len(papers)
            if start >= total:
                return

    def download_source(self, arxiv_id: str) -> bytes:
        """Download the e-print source archive for a paper.

        Returns the raw response bytes (usually a gzipped tarball, sometimes a
        gzipped single .tex file, or a PDF for PDF-only submissions — callers
        classify by magic bytes). Raises `FileNotFoundError` on HTTP 404
        (withdrawn / no source), which is permanent and must not be retried.
        """
        url = EPRINT_URL.format(arxiv_id=arxiv_id)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                self._last_request_at = time.monotonic()
                response = self._session.get(url, timeout=self.timeout)
                if response.status_code == 404:
                    raise FileNotFoundError(f"No e-print source for {arxiv_id} (HTTP 404)")
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                last_error = exc
                backoff = self.delay_seconds * 2**attempt
                print(f"  download {arxiv_id} failed ({exc}); retrying in {backoff:.0f}s...")
                time.sleep(backoff)
        raise RuntimeError(f"e-print download failed after {self.max_retries} retries: {last_error}")

    def fetch_by_ids(self, arxiv_ids: list[str], chunk_size: int = 100) -> list[ArxivPaper]:
        """Fetch metadata for specific arXiv ids (versionless ids are fine)."""
        papers: list[ArxivPaper] = []
        for i in range(0, len(arxiv_ids), chunk_size):
            chunk = arxiv_ids[i : i + chunk_size]
            text = self._get(
                {"id_list": ",".join(chunk), "max_results": len(chunk)}
            )
            _, page = _parse_feed(text)
            papers.extend(page)
        return papers
