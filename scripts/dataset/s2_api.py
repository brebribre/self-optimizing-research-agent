"""Minimal Semantic Scholar Graph API client for citation resolution (stage 4).

Works without an API key (shared global rate pool — expect occasional 429s,
handled with backoff). If S2_API_KEY is set in the environment/.env it is sent
as the x-api-key header, which gives a dedicated rate limit.
"""

from __future__ import annotations

import time

import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"
MATCH_FIELDS = "title,abstract,externalIds,publicationDate"

# Keyless access shares a global pool; stay polite.
DEFAULT_DELAY_SECONDS = 1.1
# 429 backoff is longer than network backoff — the shared pool needs time.
RATE_LIMIT_BACKOFF_SECONDS = 10.0


class SemanticScholarClient:
    def __init__(
        self,
        api_key: str | None = None,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        max_retries: int = 5,
        timeout: float = 30.0,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "self-optimizing-research-agent dataset pipeline"
        if api_key:
            self._session.headers["x-api-key"] = api_key
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)

    def _get(self, path: str, params: dict) -> dict | None:
        """Throttled GET. Returns parsed JSON, or None on HTTP 404 (no match)."""
        last_error: Exception | str | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            self._last_request_at = time.monotonic()
            try:
                response = self._session.get(
                    f"{BASE_URL}{path}", params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(self.delay_seconds * 2**attempt)
                continue
            if response.status_code == 404:
                return None
            if response.status_code == 429:
                last_error = "rate limited (429)"
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1))
                continue
            if response.status_code >= 500:
                last_error = f"server error ({response.status_code})"
                time.sleep(self.delay_seconds * 2**attempt)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"Semantic Scholar request failed after retries: {last_error}")

    def match_title(self, title: str) -> dict | None:
        """Resolve a paper title to its best match.

        Returns {"title", "abstract", "arxiv_id", "published"} or None when
        S2 has no match. `arxiv_id` is None for works not on arXiv.
        """
        payload = self._get("/paper/search/match", {"query": title, "fields": MATCH_FIELDS})
        if not payload or not payload.get("data"):
            return None
        best = payload["data"][0]
        external_ids = best.get("externalIds") or {}
        return {
            "title": best.get("title") or "",
            "abstract": best.get("abstract") or "",
            "arxiv_id": external_ids.get("ArXiv"),
            "published": best.get("publicationDate") or "",
        }
