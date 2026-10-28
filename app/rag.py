"""Policy retrieval (section 2.4).

- Splits policies.md into documents at each top-level "# " title.
- Reads the Document ID line in each document and matches it to manifest.json.
- Splits each document into sections (### §x.y) which become the retrieval chunks.
- Only documents with status "current" and effective_date <= APP_TODAY are searchable.
  Documents whose effective_date is after APP_TODAY are kept separately as "future changes"
  and only surfaced as a dated notice, never as the answer.
- Search is a simple BM25 over normalised tokens (no external model needed, runs anywhere).
  Policy documents outrank FAQ documents when scores tie, so policy wins over FAQ.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

_TITLE_RE = re.compile(r"^# (?!#)(.+)$", re.MULTILINE)
_DOC_ID_RE = re.compile(r"\*\*Document ID:\*\*\s*([A-Za-z0-9_\-]+)")
_SECTION_RE = re.compile(r"^###\s+(§\d+(?:\.\d+)*)\s+(.*)$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(§\d+)\s+(.*)$", re.MULTILINE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP = set(
    "the a an and or of to in on for is are be can you your we our it its this that with "
    "do does how what when which who i my me if at by as from not no yes".split()
)

# Tiny stemmer: good enough to match "returns"/"returned"/"return", "refunded"/"refund".
def _stem(tok: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    doc_type: str
    section: str
    heading: str
    text: str
    tokens: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.doc_title,
            "doc_type": self.doc_type,
            "section": self.section,
            "heading": self.heading,
            "text": self.text,
        }


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    effective_date: date
    status: str
    body: str
    chunks: list[Chunk]


def _split_documents(markdown: str) -> list[tuple[str, str]]:
    """Returns [(title, body)] split at each '# ' line."""
    matches = list(_TITLE_RE.finditer(markdown))
    docs: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        docs.append((m.group(1).strip(), markdown[start:end].strip()))
    return docs


def _split_sections(doc_id: str, title: str, doc_type: str, body: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    h2_titles = {m.group(1): m.group(2).strip() for m in _H2_RE.finditer(body)}
    matches = list(_SECTION_RE.finditer(body))
    # Preamble (text before the first section) is useful, e.g. "this version replaces ..."
    preamble_end = matches[0].start() if matches else len(body)
    preamble = body[:preamble_end]
    preamble = re.sub(r"^\*\*Document ID:\*\*.*$", "", preamble, flags=re.MULTILINE)
    preamble = re.sub(r"^##\s+.*$", "", preamble, flags=re.MULTILINE).strip()
    if preamble:
        chunks.append(Chunk(doc_id, title, doc_type, "§0", "Introduction", preamble))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end]
        text = re.sub(r"^##\s+.*$", "", text, flags=re.MULTILINE).strip()  # drop next h2 header
        section = m.group(1)
        heading = m.group(2).strip()
        parent = section.rsplit(".", 1)[0]
        parent_title = h2_titles.get(parent, "")
        full_heading = f"{parent_title} — {heading}" if parent_title else heading
        chunks.append(Chunk(doc_id, title, doc_type, section, full_heading, text))
    for c in chunks:
        c.tokens = tokenize(f"{title} {c.heading} {c.heading} {c.text}")
    return chunks


class PolicyIndex:
    def __init__(self, policies_path: str, manifest_path: str, today: date) -> None:
        self.today = today
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = {m["doc_id"]: m for m in json.load(fh)}
        with open(policies_path, encoding="utf-8") as fh:
            markdown = fh.read()

        self.documents: dict[str, Document] = {}
        for title, body in _split_documents(markdown):
            m = _DOC_ID_RE.search(body)
            if not m:
                continue
            doc_id = m.group(1)
            meta = manifest.get(doc_id)
            if meta is None:
                continue  # not in manifest -> unknown status, do not use
            doc_type = str(meta.get("doc_type", "policy"))
            doc = Document(
                doc_id=doc_id,
                title=str(meta.get("title") or title),
                doc_type=doc_type,
                effective_date=date.fromisoformat(str(meta["effective_date"])[:10]),
                status=str(meta.get("status", "")).lower(),
                body=body,
                chunks=_split_sections(doc_id, title, doc_type, body),
            )
            self.documents[doc_id] = doc

        self.active: list[Document] = [
            d for d in self.documents.values() if d.status == "current" and d.effective_date <= today
        ]
        self.future: list[Document] = [
            d for d in self.documents.values() if d.status == "current" and d.effective_date > today
        ]
        self._chunks: list[Chunk] = [c for d in self.active for c in d.chunks]
        self._future_chunks: list[Chunk] = [c for d in self.future for c in d.chunks]
        self._build_bm25(self._chunks)

    # ---- BM25 ----------------------------------------------------------------
    def _build_bm25(self, chunks: list[Chunk]) -> None:
        self._df: Counter[str] = Counter()
        for c in chunks:
            self._df.update(set(c.tokens))
        self._n = max(len(chunks), 1)
        self._avgdl = sum(len(c.tokens) for c in chunks) / self._n if chunks else 1.0

    def _score(self, q: list[str], c: Chunk, k1: float = 1.5, b: float = 0.75) -> float:
        tf = Counter(c.tokens)
        dl = len(c.tokens) or 1
        score = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (self._n - self._df[t] + 0.5) / (self._df[t] + 0.5))
            f = tf[t]
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / self._avgdl))
        return score

    def search(self, query: str, top_k: int = 4) -> dict[str, Any]:
        q = tokenize(query)
        scored: list[tuple[float, Chunk]] = []
        for c in self._chunks:
            s = self._score(q, c)
            if s <= 0:
                continue
            if c.doc_type == "policy":
                s *= 1.15  # policy beats FAQ when they cover the same topic
            scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        results = [c.to_dict() | {"score": round(s, 3)} for s, c in scored[:top_k]]

        # Future changes: surface only if clearly relevant to the query
        future: list[dict[str, Any]] = []
        for d in self.future:
            best = max((self._score(q, c) for c in d.chunks), default=0.0)
            if best > 0 and results and best >= 0.5 * scored[0][0]:
                future.append(
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "effective_date": d.effective_date.isoformat(),
                        "note": "Not in force yet. Mention only as a future change with its start date.",
                        "sections": [c.to_dict() for c in d.chunks if self._score(q, c) > 0][:3],
                    }
                )
        return {"results": results, "future_changes": future, "as_of": self.today.isoformat()}
