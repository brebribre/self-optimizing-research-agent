"""LaTeX-source parsing for the dataset pipeline (used by stage 3).

Pure functions that take an arXiv e-print archive and produce the pieces the
dataset needs: the related-work section's LaTeX, the cite keys it uses, and a
bibliography map (key -> title / raw entry) from the project's .bib/.bbl files.

Scope notes:
- Only \\section-level related-work headings are recognized (papers that bury
  related work in a subsection are dropped — acceptable per the inclusion
  rules, which ask for an explicit section).
- .bbl title extraction is heuristic (text between the first two \\newblock
  commands); the raw entry is preserved so stage 4 can fall back to searching
  with it.
"""

from __future__ import annotations

import gzip
import io
import posixpath
import re
import tarfile
from pathlib import Path

import bibtexparser
from bibtexparser.bparser import BibTexParser

# Section headings that count as a related-work section.
RELATED_WORK_TITLES = re.compile(
    r"related\s+works?\b|related\s+literature|prior\s+work", re.IGNORECASE
)

# \cite, \citep, \citet, \citealp, \Citet, starred forms, optional [..] args.
_CITE = re.compile(r"\\[Cc]ite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^{}]+)\}")
_SECTION = re.compile(r"\\section\*?\s*(?:\[[^\]]*\])?\s*\{")
_BOUNDARY = re.compile(
    r"\\section\*?\s*[\[{]|\\appendix\b|\\bibliography\b"
    r"|\\begin\{thebibliography\}|\\end\{document\}"
)
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")
_COMMENT = re.compile(r"(?<!\\)%.*")
_BIBITEM = re.compile(r"\\bibitem\s*(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?\s*\{([^{}]+)\}")


# --- Archive handling -----------------------------------------------------------


def load_tex_project(archive_path: str | Path) -> dict[str, str]:
    """Read an e-print archive into {member_name: decoded_text}.

    Handles gzipped tarballs, bare tarballs, and single gzipped .tex files.
    Only text-like members (.tex/.bib/.bbl/.sty/.cls) are decoded; binary
    assets are skipped.
    """
    archive_path = Path(archive_path)
    raw = archive_path.read_bytes()

    files: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if not member.name.lower().endswith((".tex", ".bib", ".bbl", ".sty", ".cls")):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                files[member.name] = _decode(handle.read())
        return files
    except tarfile.ReadError:
        pass

    # Single gzipped file (no tar wrapper).
    text = _decode(gzip.decompress(raw))
    return {"main.tex": text}


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


# --- Document assembly ----------------------------------------------------------


def strip_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def _find_main_file(files: dict[str, str]) -> str | None:
    tex = {name: body for name, body in files.items() if name.lower().endswith(".tex")}
    if not tex:
        return None
    with_begin = [n for n, b in tex.items() if r"\begin{document}" in b]
    if with_begin:
        # Prefer the one that also declares the class; else the largest.
        with_class = [n for n in with_begin if r"\documentclass" in tex[n]]
        pool = with_class or with_begin
        return max(pool, key=lambda n: len(tex[n]))
    return max(tex, key=lambda n: len(tex[n]))


def _resolve_input(name: str, files: dict[str, str], base: str) -> str | None:
    """Match an \\input{...} target to an archive member name."""
    candidates = [name, f"{name}.tex"]
    base_dir = posixpath.dirname(base)
    if base_dir:
        candidates += [posixpath.join(base_dir, c) for c in list(candidates)]
    for candidate in candidates:
        normalized = posixpath.normpath(candidate)
        if normalized in files:
            return normalized
    return None


def build_document(files: dict[str, str]) -> str:
    """Inline \\input/\\include recursively from the main file, comments stripped."""
    main = _find_main_file(files)
    if main is None:
        return ""

    seen: set[str] = set()

    def expand(name: str) -> str:
        if name in seen:  # cycle guard
            return ""
        seen.add(name)
        body = strip_comments(files.get(name, ""))

        def replace(match: re.Match) -> str:
            target = _resolve_input(match.group(1).strip(), files, name)
            return expand(target) if target else ""

        return _INPUT.sub(replace, body)

    return expand(main)


# --- Related-work section -------------------------------------------------------


def _braced_argument(text: str, open_brace: int) -> tuple[str, int]:
    """Return (content, end_index) of the {...} group starting at `open_brace`."""
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : i], i + 1
    return text[open_brace + 1 :], len(text)


def find_related_work(document: str) -> tuple[str, str] | None:
    """Find the related-work section. Returns (section_title, body_latex) or None."""
    for match in _SECTION.finditer(document):
        title, body_start = _braced_argument(document, match.end() - 1)
        if not RELATED_WORK_TITLES.search(title):
            continue
        boundary = _BOUNDARY.search(document, body_start)
        body_end = boundary.start() if boundary else len(document)
        body = document[body_start:body_end].strip()
        if body:
            return re.sub(r"\s+", " ", title).strip(), body
    return None


def extract_cite_keys(latex: str) -> list[str]:
    """Unique cite keys in order of first appearance."""
    keys: list[str] = []
    seen: set[str] = set()
    for match in _CITE.finditer(latex):
        for key in match.group(1).split(","):
            key = key.strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


# --- Bibliography ---------------------------------------------------------------


def clean_latex_text(text: str) -> str:
    """Best-effort LaTeX-to-plain-text for titles."""
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)
    text = text.replace("~", " ").replace("{", "").replace("}", "")
    text = text.replace("``", '"').replace("''", '"')
    return re.sub(r"\s+", " ", text).strip()


def parse_bibliographies(files: dict[str, str]) -> dict[str, dict]:
    """Map cite key -> {"title", "raw", "source"} from all .bib and .bbl files.

    .bib entries win over .bbl ones for the same key (cleaner titles).
    """
    entries: dict[str, dict] = {}

    for name, body in files.items():
        if name.lower().endswith(".bbl"):
            entries.update(_parse_bbl(body))

    for name, body in files.items():
        if name.lower().endswith(".bib"):
            try:
                parser = BibTexParser(common_strings=True)
                parser.ignore_nonstandard_types = False
                database = bibtexparser.loads(body, parser=parser)
            except Exception:
                continue
            for entry in database.entries:
                key = entry.get("ID", "").strip()
                if not key:
                    continue
                # Keep venue/eprint/url fields in `raw` — stage 4 mines them
                # for embedded arXiv ids.
                extra = " ".join(
                    str(entry.get(field, ""))
                    for field in ("journal", "booktitle", "eprint", "archiveprefix",
                                  "url", "doi", "note", "howpublished")
                )
                raw = re.sub(r"\s+", " ", f"{entry.get('title', '')} {extra}").strip()
                entries[key] = {
                    "title": clean_latex_text(entry.get("title", "")),
                    "raw": raw[:500],
                    "source": "bib",
                }

    return entries


def _parse_bbl(body: str) -> dict[str, dict]:
    """Parse \\bibitem entries; title guessed from the first \\newblock chunk."""
    entries: dict[str, dict] = {}
    matches = list(_BIBITEM.finditer(body))
    for i, match in enumerate(matches):
        key = match.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw = body[match.end() : end].strip()
        raw = re.split(r"\\end\{thebibliography\}", raw)[0].strip()

        # natbib style: "Authors. \newblock Title. \newblock Venue."
        blocks = re.split(r"\\newblock", raw)
        title = clean_latex_text(blocks[1]).rstrip(".") if len(blocks) >= 2 else ""
        entries[key] = {
            "title": title,
            "raw": re.sub(r"\s+", " ", raw)[:500],
            "source": "bbl",
        }
    return entries


# --- Language check -------------------------------------------------------------

_STOPWORDS = ("the", "and", "of", "in", "to")


def is_english(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return sum(1 for w in _STOPWORDS if w in words) >= 3
