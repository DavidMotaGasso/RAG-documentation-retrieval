"""Phase 1.5 sanity check of retrieval quality on a small hand-labeled query set.

This is not the Phase 4 evaluation: it is a quick check that the index works,
that filters help, and that the bge query instruction is worth keeping. Each
query lists the document(s) that should be retrieved; out-of-corpus queries
list none and are used to see how low scores get when nothing is relevant.

Metrics per configuration:
  hit@1 / hit@5  share of queries whose expected document appears in the top 1 / top 5
  MRR            mean reciprocal rank of the first expected document
  top1 score     cosine similarity of the best hit

Usage:
    python -m src.indexing.retrieval_check
"""

import statistics
from dataclasses import dataclass

from src.indexing.vector_store import VectorStore

TOP_K = 5


@dataclass
class Query:
    text: str
    expected: set[str]          # doc_id prefixes (4 chars); empty = out of corpus
    domain: str | None          # the domain a router would pick; None = ambiguous / none
    kind: str                   # easy | hard | identifier | multi | out


QUERIES = [
    # Healthcare: distinct papers
    Query("Which YOLO version achieved the best precision and recall for lung cancer detection?",
          {"2d21"}, "healthcare", "easy"),
    Query("How does scale-aware curriculum learning help nodule detection with little training data?",
          {"6fb4"}, "healthcare", "easy"),
    Query("What dice coefficient does the deeply supervised MultiResUNet reach for tumor segmentation?",
          {"9508"}, "healthcare", "easy"),
    Query("Does combining capsule networks with a YOLO detector reduce false positives?",
          {"a731"}, "healthcare", "easy"),
    Query("Which bacterial taxa differ between adenocarcinoma and squamous cell carcinoma tissue?",
          {"cf2e"}, "healthcare", "easy"),
    # Healthcare: the hard pair (both CNN-based CT classification)
    Query("What test accuracy did VGG16 achieve classifying CT scans into normal, benign and malignant?",
          {"b85a"}, "healthcare", "hard"),
    Query("What accuracy does the enhanced CNN reach on the four-class lung cancer CT dataset?",
          {"93c4"}, "healthcare", "hard"),
    Query("Which image enhancement techniques were applied to the CT images before training?",
          {"93c4"}, "healthcare", "hard"),
    Query("How was Grad-CAM used to explain the model's predictions on CT scans?",
          {"b85a"}, "healthcare", "hard"),
    # Exact identifiers
    Query("Which study uses the LIDC-IDRI dataset?", {"a731"}, "healthcare", "identifier"),
    Query("Results on the LOTUS benchmark", {"9508"}, "healthcare", "identifier"),
    # Remote sensing and AI security
    Query("How much did the built-up area grow in Gelephu between 2016 and 2023?",
          {"9a34"}, "remote_sensing", "easy"),
    Query("Which classifier performed best for land cover mapping in Google Earth Engine in a mountainous region?",
          {"564f"}, "remote_sensing", "easy"),
    Query("How do attention and transformer models compare to recurrent networks for satellite image time series?",
          {"36ec"}, "remote_sensing", "easy"),
    Query("Accuracy of Inception-v3 on the UC Merced land use dataset",
          {"3d40"}, "remote_sensing", "identifier"),
    Query("How do authenticated prompts prevent prompt injection in LLM agents?",
          {"c9d5"}, "ai_security", "easy"),
    # Cross-document
    Query("Which papers compare ResNet-50 with other CNN architectures?",
          {"3d40", "93c4", "b85a"}, None, "multi"),
    # Out of corpus
    Query("How do LSTM models forecast stock prices?", set(), None, "out"),
    Query("What are the side effects of statin drugs?", set(), None, "out"),
    Query("How to fine-tune BERT for sentiment analysis of tweets?", set(), None, "out"),
]


def evaluate(store: VectorStore, use_domain_filter: bool) -> list[dict]:
    rows = []
    for q in QUERIES:
        domain = q.domain if use_domain_filter else None
        hits = store.search(q.text, domain=domain, top_k=TOP_K)
        docs = [h["doc_id"][:4] for h in hits]
        rank = next((i + 1 for i, d in enumerate(docs) if d in q.expected), None)
        rows.append({"query": q, "docs": docs, "rank": rank, "top_score": hits[0]["score"],
                     "docs_found": len(q.expected & set(docs))})
    return rows


def summarize(name: str, rows: list[dict]) -> None:
    labeled = [r for r in rows if r["query"].expected]
    out = [r for r in rows if not r["query"].expected]
    hit1 = sum(r["rank"] == 1 for r in labeled) / len(labeled)
    hit5 = sum(r["rank"] is not None for r in labeled) / len(labeled)
    mrr = statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in labeled)
    in_scores = [r["top_score"] for r in labeled]
    out_scores = [r["top_score"] for r in out]
    print(f"{name:<34} hit@1={hit1:.2f}  hit@5={hit5:.2f}  MRR={mrr:.2f}  "
          f"top1 score: in-corpus min={min(in_scores):.3f}  out-of-corpus max={max(out_scores):.3f}")


def print_details(rows: list[dict]) -> None:
    for r in rows:
        q = r["query"]
        status = ("out " if not q.expected else
                  "OK  " if r["rank"] == 1 else f"#{r['rank']}  " if r["rank"] else "MISS")
        multi = f" ({r['docs_found']}/{len(q.expected)} docs)" if q.kind == "multi" else ""
        print(f"  {status} {q.kind:<10} {r['top_score']:.3f}  {' '.join(r['docs'])}{multi}  | {q.text[:60]}")


def main() -> None:
    store = VectorStore()
    configs = {}
    for instruction in (True, False):
        store.embedder.use_query_instruction = instruction
        for use_filter in (False, True):
            name = f"instruction={'on ' if instruction else 'off'} filter={'domain' if use_filter else 'none  '}"
            configs[name] = evaluate(store, use_filter)

    print(f"{len(QUERIES)} queries ({sum(bool(q.expected) for q in QUERIES)} in corpus, "
          f"{sum(not q.expected for q in QUERIES)} out of corpus), top_k={TOP_K}\n")
    for name, rows in configs.items():
        summarize(name, rows)
    for name in ("instruction=on  filter=none  ", "instruction=on  filter=domain"):
        print(f"\nDetails: {name.strip()}")
        print_details(configs[name])


if __name__ == "__main__":
    main()
