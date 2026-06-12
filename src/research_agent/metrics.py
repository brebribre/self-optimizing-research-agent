"""Evaluation metrics for related-work generation.

DSPy optimizers need a metric `(example, prediction, trace) -> float|bool`.
This module provides:

- ``citation_f1``           — citation accuracy as F1 of cited vs. gold works.
- ``citation_faithfulness`` — whether generated claims are entailed by their
  cited sources (NLI-based).
- ``citation_f1_faithfulness`` — the product of the two, rewarding sections
  that cite the right papers *and* describe them faithfully.

`citation_f1` is implemented; `citation_faithfulness` is still a header pending
its NLI backend (so the combined product depends on that too). They are the
metrics for the planned optimization experiments (f1 only, faithfulness only,
and the combined product).
"""

from __future__ import annotations

import re

import dspy

# A candidate work counts as "cited" in a text when at least this fraction of
# its title's significant tokens appear there (fallback when the text contains
# no [pool_id] markers). 0.5 = a majority of the title.
_CITATION_MATCH_THRESHOLD = 0.5

# One candidate-pool line: "[P01] Title: <title> | Abstract: <abstract>".
# The pool_id prefix is optional so plain "Title: ..." lines still parse.
_POOL_LINE = re.compile(
    r"^(?:\[(?P<pid>[^\]]+)\]\s*)?Title:\s*(?P<title>.*?)\s*(?:\|\s*Abstract:.*)?$"
)


def _significant_tokens(text: str) -> set[str]:
    """Lowercased word tokens longer than three characters (drops most stopwords)."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3}


def _parse_candidates(cited_papers: str) -> list[tuple[str | None, str]]:
    """Extract (pool_id, title) pairs from the formatted `cited_papers` input.

    The input format (see `data.format_candidate_pool`) is one paper per line:
    ``[P01] Title: <title> | Abstract: <abstract>``. Lines that don't match
    fall back to (None, whole line).
    """
    candidates: list[tuple[str | None, str]] = []
    for line in cited_papers.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _POOL_LINE.match(line)
        if match and match.group("title"):
            candidates.append((match.group("pid"), match.group("title")))
        else:
            candidates.append((None, line))
    return candidates


def _cited_indices(candidates: list[tuple[str | None, str]], text: str) -> set[int]:
    """Indices of candidates judged to be cited in `text`.

    Primary signal: the candidate's bracketed pool_id (e.g. ``[P01]``) appears
    in the text. When at least one marker is found, markers are trusted
    exclusively — with a same-subfield candidate pool, fuzzy title matching
    would produce false positives across papers sharing common tokens.

    Fallback (no markers at all, e.g. the text cites by title): a candidate is
    cited when a majority of its title's significant tokens appear in the
    text's token set.
    """
    marker_hits = {
        i for i, (pid, _) in enumerate(candidates) if pid and f"[{pid}]" in text
    }
    if marker_hits:
        return marker_hits

    text_tokens = _significant_tokens(text)
    cited: set[int] = set()
    for i, (_, title) in enumerate(candidates):
        title_tokens = _significant_tokens(title)
        if not title_tokens:
            continue
        hits = sum(1 for tok in title_tokens if tok in text_tokens)
        if hits / len(title_tokens) >= _CITATION_MATCH_THRESHOLD:
            cited.add(i)
    return cited


def citation_f1(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Citation accuracy as the F1 of cited works against the gold answer key.

    The candidate pool (`example.cited_papers`) is the universe of citable
    works. The reference set is `example.gold_cited` (the pool_ids the real
    authors cited) when present, otherwise the works detected in the gold
    `related_work` text. The predicted set is the works the generated
    `related_work` cites — by ``[pool_id]`` marker, falling back to title
    matching when the text contains no markers. Then::

        precision = |predicted ∩ reference| / |predicted|
        recall    = |predicted ∩ reference| / |reference|
        f1        = 2 * precision * recall / (precision + recall)

    Returns a score in [0, 1]: rewards citing the right papers (recall) while
    penalizing spurious citations (precision). Returns 1.0 only when both sets
    are empty (nothing should be cited and nothing was) and 0.0 when one side is
    empty but the other isn't, or when there are no candidate works to match
    against.

    Note: only works present in the candidate pool can be detected — citations
    to papers outside that universe are invisible to this metric.
    """
    candidates = _parse_candidates(getattr(example, "cited_papers", "") or "")
    if not candidates:
        return 0.0

    generated = (getattr(prediction, "related_work", "") or "").strip()

    gold_cited = getattr(example, "gold_cited", None) or []
    if gold_cited:
        pid_to_index = {pid: i for i, (pid, _) in enumerate(candidates) if pid}
        reference = {pid_to_index[p] for p in gold_cited if p in pid_to_index}
    else:
        gold = (getattr(example, "related_work", "") or "").strip()
        reference = _cited_indices(candidates, gold)
    predicted = _cited_indices(candidates, generated)

    if not reference and not predicted:
        return 1.0
    if not reference or not predicted:
        return 0.0

    true_positives = len(predicted & reference)
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(predicted)
    recall = true_positives / len(reference)
    return 2 * precision * recall / (precision + recall)


def citation_faithfulness(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Faithfulness of generated claims to their cited source papers.

    For each claim in the generated `related_work`, an NLI model checks whether
    the claim is entailed by the corresponding cited paper's text (title /
    abstract). The metric is the fraction of claims judged entailed (not
    contradicted or unsupported), giving a score in [0, 1].

    The NLI backend is wired up separately; this is the metric header only.

    TODO: integrate the NLI model and implement claim extraction + scoring.
    """
    raise NotImplementedError("citation_faithfulness is not implemented yet")


def citation_f1_faithfulness(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Combined metric: the product of citation F1 and faithfulness.

    Multiplying the two scores requires a section to both cite the right works
    (`citation_f1`) and describe them faithfully (`citation_faithfulness`); a
    failure on either axis drives the combined score toward zero. Returns a
    score in [0, 1].
    """
    return citation_f1(example, prediction, trace) * citation_faithfulness(
        example, prediction, trace
    )
