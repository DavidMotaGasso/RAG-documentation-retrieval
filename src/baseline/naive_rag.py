"""Phase 2 naive RAG baseline: one global top-k search, one LLM call.

This is the reference point the agent (Phase 3) is measured against in Phase 4,
so it is deliberately simple but not a straw man:
  - The user's question is embedded as-is: no rewriting, no translation to
    English, no domain filter, no follow-up searches.
  - Always TOP_K chunks from the whole corpus, whatever the question.
  - Same LLM, same index and same "say it is not covered" instruction as the
    agent will have, so the comparison isolates the retrieval strategy.

Every answer records what was retrieved, tokens, cost and latency, the same
fields the agent will report, so Phase 4 can compare both on equal terms.

Usage:
    python -m src.baseline.naive_rag "Which YOLO version worked best for lung cancer detection?"
    python -m src.baseline.naive_rag --show-context "..."
"""

import argparse
import time

import anthropic
from dotenv import load_dotenv

from src.indexing.vector_store import VectorStore
from src.ingestion.inventory import PROJECT_ROOT
from src.rag_answer import EFFORT, MODEL, RAGAnswer

TOP_K = 5
# Haiku 5.5 thinks by default and thinking counts toward max_tokens, so leave
# room above the ~300-token answers.
MAX_TOKENS = 4000

SYSTEM_PROMPT = """You answer questions about a collection of research papers.
Use only the numbered excerpts provided with the question; do not add outside knowledge.
Cite the excerpts you use as [1], [2], ...
If the excerpts do not contain the answer, say that the documents do not cover it
instead of guessing. Answer in the language of the question."""


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] Paper: {h['title']} | Section: {h['section']} | "
        f"pages {h['page_start']}-{h['page_end']}\n{h['text']}"
        for i, h in enumerate(hits, start=1))


class NaiveRAG:
    def __init__(self, store: VectorStore | None = None, top_k: int = TOP_K):
        load_dotenv(PROJECT_ROOT / ".env")
        self.store = store or VectorStore()
        self.client = anthropic.Anthropic()
        self.top_k = top_k

    def answer(self, question: str) -> RAGAnswer:
        start = time.perf_counter()
        hits = self.store.search(question, top_k=self.top_k)
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": EFFORT},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Excerpts:\n\n{format_context(hits)}\n\n"
                                                  f"Question: {question}"}],
        )
        if response.stop_reason not in ("end_turn", "max_tokens"):
            raise RuntimeError(f"Unexpected stop_reason {response.stop_reason}")
        return RAGAnswer(
            question=question,
            answer="".join(b.text for b in response.content if b.type == "text"),
            retrieved=[{"chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "score": h["score"],
                        "text": h["text"], "title": h["title"], "section": h["section"]}
                       for h in hits],
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            latency_s=round(time.perf_counter() - start, 2),
            meta={"stop_reason": response.stop_reason},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--show-context", action="store_true", help="print retrieved chunk texts")
    args = parser.parse_args()

    result = NaiveRAG(top_k=args.top_k).answer(args.question)
    print(result.answer)
    print("\nRetrieved:")
    for i, h in enumerate(result.retrieved, start=1):
        print(f"  [{i}] {h['score']:.3f}  {h['chunk_id']}  {h['title'][:50]} | {h['section'][:30]}")
        if args.show_context:
            print(f"      {h['text'][:400]}...\n")
    print(f"\n{result.input_tokens:,} in / {result.output_tokens:,} out tokens, "
          f"${result.cost_usd:.4f}, {result.latency_s:.1f}s")


if __name__ == "__main__":
    main()
