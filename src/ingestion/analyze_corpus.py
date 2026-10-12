"""Exploratory analysis of the cleaned corpus, to choose chunk size and embedding model.

Token counts use the tokenizer shared by both candidate embedding models
(all-MiniLM-L6-v2 and bge-small-en-v1.5 use the same BERT uncased WordPiece
vocabulary), so the numbers are exact for either choice.

Reports:
  1. Indexable volume per document (abstract + body only).
  2. Paragraph and section length distributions.
  3. Cost of the contextual header "[Paper: ... | Section: ...]" in tokens.
  4. Simulated section-aware chunking at several target sizes.

Usage:
    python -m src.ingestion.analyze_corpus
"""

import json
import statistics
from collections import Counter

from transformers import AutoTokenizer

from src.ingestion.cleaner import CLEAN_DIR

TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"
INDEXED_KINDS = ("abstract", "body")
MODEL_LIMITS = {"all-MiniLM-L6-v2": 256, "bge-small-en-v1.5": 512}
CANDIDATE_CHUNK_SIZES = (128, 200, 256, 300, 400, 512)


def percentiles(values: list[int]) -> str:
    if not values:
        return "-"
    q = statistics.quantiles(values, n=10) if len(values) > 1 else [values[0]] * 9
    return f"p10={q[0]:.0f}  p50={statistics.median(values):.0f}  p90={q[8]:.0f}  max={max(values)}"


def context_header(doc: dict, section: dict) -> str:
    return f"[Paper: {doc['title']} | Section: {section['path']}]"


def simulate_chunks(paragraph_tokens: list[list[int]], target: int) -> list[int]:
    """Greedy section-aware packing: whole paragraphs into chunks of <= target tokens.

    Each inner list holds the paragraph sizes of one section; chunks never cross
    sections. Paragraphs longer than the target are split into target-size
    pieces. Returns the size of every resulting chunk. No overlap is modelled.
    """
    chunks = []
    for section in paragraph_tokens:
        current = 0
        for size in section:
            while size > target:  # oversized paragraph -> split
                if current:
                    chunks.append(current)
                    current = 0
                chunks.append(target)
                size -= target
            if current + size > target and current:
                chunks.append(current)
                current = 0
            current += size
        if current:
            chunks.append(current)
    return chunks


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    def n_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CLEAN_DIR.glob("*.json"))]

    all_paragraphs, all_sections, all_headers = [], [], []
    per_doc_sections: dict[str, list[list[int]]] = {}
    doc_tokens: Counter[str] = Counter()

    print("1. Indexable volume (abstract + body)\n")
    print(f"{'doc_id':<14}{'domain':<16}{'sections':>9}{'paras':>7}{'tokens':>9}{'share':>8}")
    rows = []
    for doc in docs:
        sections = [s for s in doc["sections"] if s["kind"] in INDEXED_KINDS]
        section_paras = [[n_tokens(p["text"]) for p in s["paragraphs"]] for s in sections]
        per_doc_sections[doc["doc_id"]] = section_paras
        doc_tokens[doc["doc_id"]] = sum(map(sum, section_paras))
        all_paragraphs += [t for sec in section_paras for t in sec]
        all_sections += [sum(sec) for sec in section_paras]
        all_headers += [n_tokens(context_header(doc, s)) for s in sections]
        rows.append((doc, len(sections), sum(map(len, section_paras))))

    total = sum(doc_tokens.values())
    for doc, n_sections, n_paras in rows:
        tokens = doc_tokens[doc["doc_id"]]
        print(f"{doc['doc_id']:<14}{doc['domain']:<16}{n_sections:>9}{n_paras:>7}"
              f"{tokens:>9,}{tokens / total:>8.1%}")
    by_domain: Counter[str] = Counter()
    for doc in docs:
        by_domain[doc["domain"]] += doc_tokens[doc["doc_id"]]
    print(f"\nTotal: {total:,} tokens  |  "
          + "  ".join(f"{d}: {t / total:.0%}" for d, t in by_domain.most_common()))

    print("\n2. Length distributions (tokens)\n")
    print(f"Paragraphs ({len(all_paragraphs)}): {percentiles(all_paragraphs)}")
    for model, limit in MODEL_LIMITS.items():
        over = sum(t > limit for t in all_paragraphs)
        print(f"   paragraphs > {limit} ({model} limit): {over} ({over / len(all_paragraphs):.1%})")
    print(f"Sections   ({len(all_sections)}): {percentiles(all_sections)}")
    tiny = sum(t < 50 for t in all_sections)
    print(f"   sections < 50 tokens: {tiny} ({tiny / len(all_sections):.0%})")

    print("\n3. Contextual header cost\n")
    print(f"Header tokens: {percentiles(all_headers)}")

    print("\n4. Simulated section-aware chunking (no overlap, header not included)\n")
    print(f"{'target':>7}{'chunks':>8}{'p50':>6}{'p10':>6}{'<50 tok':>9}{'largest doc share':>19}")
    for target in CANDIDATE_CHUNK_SIZES:
        per_doc = {d: simulate_chunks(secs, target) for d, secs in per_doc_sections.items()}
        sizes = [s for chunks in per_doc.values() for s in chunks]
        n = len(sizes)
        biggest = max(per_doc, key=lambda d: len(per_doc[d]))
        q = statistics.quantiles(sizes, n=10)
        print(f"{target:>7}{n:>8}{statistics.median(sizes):>6.0f}{q[0]:>6.0f}"
              f"{sum(s < 50 for s in sizes):>9}{len(per_doc[biggest]) / n:>13.0%} ({biggest[:4]})")


if __name__ == "__main__":
    main()
