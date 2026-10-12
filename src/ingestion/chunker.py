"""Split cleaned documents into section-aware chunks ready for embedding.

Input:  data/processed/clean/{doc_id}.json   (from cleaner)
Output: data/processed/chunks.jsonl          one chunk per line, with metadata

Strategy (decided in Phase 0.6, see CLAUDE.md):
  - Only `abstract` and `body` sections are chunked.
  - Chunks never cross section boundaries.
  - Whole paragraphs are packed greedily up to TARGET_TOKENS of new text.
    Paragraphs longer than that are split at sentence boundaries (and, as a
    last resort, at word boundaries).
  - Each chunk after the first in a section starts with up to OVERLAP_TOKENS of
    the previous chunk's last sentences, so an idea cut at a boundary is not lost.
  - A trailing chunk smaller than MIN_CHUNK_TOKENS is merged into the previous
    chunk of the same section.
  - `embedding_text` = contextual header + text, which is what Phase 1 embeds.

Token counts use the embedding model's own tokenizer (bge-small-en-v1.5).

Usage:
    python -m src.ingestion.chunker
"""

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass

from transformers import AutoTokenizer

from src.ingestion.cleaner import CLEAN_DIR
from src.ingestion.inventory import PROJECT_ROOT

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_MAX_TOKENS = 512          # includes the [CLS] and [SEP] special tokens
INDEXED_KINDS = ("abstract", "body")
TARGET_TOKENS = 300             # new text per chunk (overlap and header come on top)
OVERLAP_TOKENS = 45             # ~15% of TARGET_TOKENS, whole sentences only
MIN_CHUNK_TOKENS = 50

# Abbreviations that end with a period but do not end a sentence.
ABBREVIATIONS = re.compile(
    r"(?:\bet al|\bFigs?|\bEqs?|\bRefs?|\bTab|\bNo|\bvs|\bcf|\be\.g|\bi\.e|\bapprox|\bresp|\bSec)\.$",
    re.IGNORECASE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"\u201c])")


@dataclass
class Unit:
    """A piece of text that is never split further: a paragraph, sentence or word window."""
    text: str
    tokens: int
    page_start: int
    page_end: int
    paragraph: int          # index of the source paragraph within the section


class Chunker:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ----------------------------------------------------------- splitting ---

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        sentences: list[str] = []
        for piece in SENTENCE_BOUNDARY.split(text):
            if sentences and ABBREVIATIONS.search(sentences[-1]):
                sentences[-1] += " " + piece
            else:
                sentences.append(piece)
        return sentences

    def split_words(self, text: str, limit: int) -> list[str]:
        """Last resort for a single sentence longer than the limit."""
        windows, current = [], []
        for word in text.split():
            if current and self.count(" ".join(current + [word])) > limit:
                windows.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            windows.append(" ".join(current))
        return windows

    def section_units(self, paragraphs: list[dict]) -> list[Unit]:
        units = []
        for i, p in enumerate(paragraphs):
            pages = (p["page_start"], p["page_end"])
            tokens = self.count(p["text"])
            if tokens <= TARGET_TOKENS:
                units.append(Unit(p["text"], tokens, *pages, i))
                continue
            for sentence in self.split_sentences(p["text"]):
                n = self.count(sentence)
                pieces = [sentence] if n <= TARGET_TOKENS else self.split_words(sentence, TARGET_TOKENS)
                units += [Unit(s, self.count(s), *pages, i) for s in pieces]
        return units

    # ------------------------------------------------------------- packing ---

    @staticmethod
    def join(units: list[Unit]) -> str:
        """Same paragraph -> space; new paragraph -> blank line."""
        out = units[0].text
        for prev, unit in zip(units, units[1:]):
            out += (" " if unit.paragraph == prev.paragraph else "\n\n") + unit.text
        return out

    def overlap_from(self, units: list[Unit]) -> str:
        """Trailing whole sentences of the previous chunk, up to OVERLAP_TOKENS."""
        sentences = self.split_sentences(units[-1].text)
        taken: list[str] = []
        for sentence in reversed(sentences):
            if self.count(" ".join([sentence] + taken)) > OVERLAP_TOKENS:
                break
            taken.insert(0, sentence)
        return " ".join(taken)

    def pack_section(self, units: list[Unit]) -> list[list[Unit]]:
        groups: list[list[Unit]] = []
        current: list[Unit] = []
        size = 0
        for unit in units:
            if current and size + unit.tokens > TARGET_TOKENS:
                groups.append(current)
                current, size = [], 0
            current.append(unit)
            size += unit.tokens
        if current:
            # Merge a tiny trailing chunk into the previous one of the same section.
            if groups and size < MIN_CHUNK_TOKENS:
                groups[-1].extend(current)
            else:
                groups.append(current)
        return groups

    # ------------------------------------------------------------ document ---

    def chunk_document(self, doc: dict) -> list[dict]:
        chunks = []
        for section in doc["sections"]:
            if section["kind"] not in INDEXED_KINDS:
                continue
            header = f"[Paper: {doc['title']} | Section: {section['path']}]"
            groups = self.pack_section(self.section_units(section["paragraphs"]))
            for g, group in enumerate(groups):
                overlap = self.overlap_from(groups[g - 1]) if g > 0 else ""
                body = self.join(group)
                text = f"{overlap}\n\n{body}" if overlap else body
                embedding_text = f"{header}\n{text}"
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_{len(chunks):04d}",
                    "doc_id": doc["doc_id"],
                    "chunk_index": len(chunks),  # sequential within the document
                    "domain": doc["domain"],
                    "title": doc["title"],
                    "authors": doc["authors"],
                    "section": section["path"],
                    "section_kind": section["kind"],
                    "page_start": min(u.page_start for u in group),
                    "page_end": max(u.page_end for u in group),
                    # Characters at the start of `text` repeated from the previous
                    # chunk; strip them when stitching neighbours back together.
                    "overlap_chars": len(overlap) + 2 if overlap else 0,
                    "n_tokens": self.count(text),
                    "n_embedding_tokens": self.count(embedding_text) + 2,  # + [CLS], [SEP]
                    "text": text,
                    "embedding_text": embedding_text,
                })
        return chunks


def main() -> None:
    chunker = Chunker()
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CLEAN_DIR.glob("*.json"))]

    all_chunks = []
    for doc in docs:
        all_chunks += chunker.chunk_document(doc)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = [c["n_tokens"] for c in all_chunks]
    embed_sizes = [c["n_embedding_tokens"] for c in all_chunks]
    q = statistics.quantiles(sizes, n=10)
    truncated = sum(s > MODEL_MAX_TOKENS for s in embed_sizes)
    per_doc = Counter(c["doc_id"] for c in all_chunks)
    per_domain = Counter(c["domain"] for c in all_chunks)

    print(f"{len(all_chunks)} chunks -> {CHUNKS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"text tokens:      p10={q[0]:.0f}  p50={statistics.median(sizes):.0f}  "
          f"p90={q[8]:.0f}  max={max(sizes)}  (<{MIN_CHUNK_TOKENS}: "
          f"{sum(s < MIN_CHUNK_TOKENS for s in sizes)})")
    print(f"embedding tokens: max={max(embed_sizes)}  over {MODEL_MAX_TOKENS} (truncated): {truncated}")
    print(f"with overlap: {sum(c['overlap_chars'] > 0 for c in all_chunks)}")
    print("per domain: " + "  ".join(f"{d}={n}" for d, n in per_domain.most_common()))
    print("per doc:    " + "  ".join(f"{d[:4]}={n}" for d, n in per_doc.most_common()))


if __name__ == "__main__":
    main()
