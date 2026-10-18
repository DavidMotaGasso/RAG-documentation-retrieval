"""The agent's retrieval tools (design: Phase 3.1 in CLAUDE.md).

  search(query, domain, doc_id?, top_k?)   semantic search; domain is required
  list_documents(domain)                   compact document cards, to pick a paper
  get_chunk_context(chunk_id, window?)     a chunk plus its neighbours

Each tool is a plain method returning (text, is_error), so it can be tested
without an LLM; `sdk_tools()` wraps them for the Claude Agent SDK. A RAGTools
instance also records, per question, every tool call (`trace`) and every chunk
the agent was shown (`retrieved`); call `reset()` before each question.

What the agent sees is deliberately lean:
  - no similarity scores (Phase 1: they mislead about relevance; kept in the trace),
  - cards without key_results (cards are for choosing a paper; answers must
    come from chunks).

Usage:
    python -m src.agent.tools      # prints sample output of each tool (no LLM)
"""

import json
import re
from typing import Any

from claude_agent_sdk import SdkMcpTool, tool

from src.indexing.vector_store import VectorStore
from src.ingestion.document_cards import CARDS_PATH
from src.ingestion.domains import DOC_DOMAINS, DOMAINS

DEFAULT_TOP_K = 5
MAX_TOP_K = 10
MAX_WINDOW = 2
ALL = "all"
CHUNK_ID = re.compile(r"^([0-9a-f]{12})_(\d{4})$")

SEARCH_DESCRIPTION = """Semantic search over chunks (~300 tokens each) of the research papers.
Write `query` in English as a descriptive phrase, not a bare name (e.g. "lung tumor
segmentation results on the LOTUS dataset", not "LOTUS").
`domain` is required: one of the corpus domains, or "all" only when the domain is unclear.
`doc_id` (optional) restricts the search to one paper. `top_k`: 1-10, default 5.
Each result is "[chunk_id] paper title | section | pages" followed by the text."""

LIST_DOCUMENTS_DESCRIPTION = """List the papers of a domain as short cards: doc_id, title, task,
methods, datasets and what distinguishes the paper. Use it to choose which paper a
question is about (then search with its doc_id), or to see which papers use a technique.
Cards are summaries for choosing papers: take facts and numbers from search results."""

CONTEXT_DESCRIPTION = """Return a chunk together with its neighbouring chunks (`window` on each
side, 1-2, default 1), as one continuous text. Use it when a retrieved chunk seems cut off."""


class RAGTools:
    def __init__(self, store: VectorStore | None = None, max_calls: int | None = None):
        self.store = store or VectorStore()
        self.max_calls = max_calls      # hard per-question limit; None = unlimited
        cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
        self.cards = sorted(cards.values(), key=lambda c: (c["domain"], c["doc_id"]))
        self.reset()

    def reset(self) -> None:
        """Start a new question: clear the trace and the retrieved chunks."""
        self.trace: list[dict] = []
        self.rejected_calls = 0                 # calls refused by the max_calls limit
        self._retrieved: dict[str, dict] = {}   # chunk_id -> hit, in first-seen order

    @property
    def retrieved(self) -> list[dict]:
        return list(self._retrieved.values())

    def _record(self, hits: list[dict]) -> None:
        for h in hits:
            self._retrieved.setdefault(h["chunk_id"], {
                "chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "score": h.get("score"),
                "text": h["text"], "title": h["title"], "section": h["section"]})

    def _error(self, name: str, args: dict, message: str) -> tuple[str, bool]:
        self.trace.append({"tool": name, "args": args, "error": message})
        return f"Error: {message}", True

    def _over_budget(self) -> bool:
        return self.max_calls is not None and len(self.trace) >= self.max_calls

    # --------------------------------------------------------------- tools ---

    def search(self, query: str, domain: str, doc_id: str | None = None,
               top_k: int = DEFAULT_TOP_K) -> tuple[str, bool]:
        args = {"query": query, "domain": domain, "doc_id": doc_id, "top_k": top_k}
        if domain != ALL and domain not in DOMAINS:
            return self._error("search", args, f"unknown domain {domain!r}; use one of "
                                               f"{', '.join([*DOMAINS, ALL])}")
        if doc_id:
            if doc_id not in DOC_DOMAINS:
                return self._error("search", args, f"unknown doc_id {doc_id!r}; "
                                                   "get valid ids from list_documents")
            if domain != ALL and DOC_DOMAINS[doc_id] != domain:
                return self._error("search", args, f"doc_id {doc_id} belongs to domain "
                                                   f"{DOC_DOMAINS[doc_id]!r}, not {domain!r}")
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        hits = self.store.search(query, domain=None if domain == ALL else domain,
                                 doc_id=doc_id or None, top_k=top_k)
        self._record(hits)
        self.trace.append({"tool": "search", "args": args,
                           "chunk_ids": [h["chunk_id"] for h in hits],
                           "doc_ids": [h["doc_id"] for h in hits],
                           "scores": [h["score"] for h in hits]})
        return "\n\n".join(
            f"[{h['chunk_id']}] {h['title']} | {h['section']} | "
            f"pages {h['page_start']}-{h['page_end']}\n{h['text']}" for h in hits), False

    def list_documents(self, domain: str) -> tuple[str, bool]:
        args = {"domain": domain}
        if domain != ALL and domain not in DOMAINS:
            return self._error("list_documents", args, f"unknown domain {domain!r}; use one "
                                                       f"of {', '.join([*DOMAINS, ALL])}")
        cards = [c for c in self.cards if domain == ALL or c["domain"] == domain]
        self.trace.append({"tool": "list_documents", "args": args,
                           "doc_ids": [c["doc_id"] for c in cards]})
        return "\n\n".join(self._format_card(c) for c in cards), False

    @staticmethod
    def _format_card(card: dict) -> str:
        lines = [f"{card['doc_id']} | {card['title']}",
                 f"  Task: {card['task']}",
                 f"  Methods: {', '.join(card['methods'])}"]
        if card["datasets"]:
            lines.append(f"  Datasets: {', '.join(card['datasets'])}")
        lines.append(f"  Distinctive: {card['distinguishing_features']}")
        return "\n".join(lines)

    def get_chunk_context(self, chunk_id: str, window: int = 1) -> tuple[str, bool]:
        args = {"chunk_id": chunk_id, "window": window}
        match = CHUNK_ID.match(chunk_id or "")
        if not match:
            return self._error("get_chunk_context", args, f"malformed chunk_id {chunk_id!r}; "
                                                          "expected e.g. 93c4d3a53ca1_0012")
        doc_id, index = match.group(1), int(match.group(2))
        window = max(1, min(int(window), MAX_WINDOW))
        try:
            context = self.store.get_context(doc_id, index, window)
        except KeyError:
            return self._error("get_chunk_context", args, f"no chunk {chunk_id!r} in the index")

        ids = [f"{doc_id}_{i:04d}" for i in context["chunk_indexes"]]
        rows = self.store.collection.get(ids=ids)
        self._record([{"chunk_id": cid, "doc_id": doc_id, "text": text, **meta}
                      for cid, text, meta in zip(rows["ids"], rows["documents"], rows["metadatas"])])
        self.trace.append({"tool": "get_chunk_context", "args": args, "chunk_ids": ids,
                           "doc_ids": [doc_id]})
        return (f"{context['title']} | pages {context['page_start']}-{context['page_end']} | "
                f"chunks {ids[0]} to {ids[-1]}\n{context['text']}"), False

    # ----------------------------------------------------------------- SDK ---

    def sdk_tools(self) -> list[SdkMcpTool]:
        """The tools wrapped for create_sdk_mcp_server()."""
        domain_enum = [*DOMAINS, ALL]

        def result(text_and_error: tuple[str, bool]) -> dict[str, Any]:
            text, is_error = text_and_error
            return {"content": [{"type": "text", "text": text}], "is_error": is_error}

        budget_message = result((f"Error: tool-call limit ({self.max_calls}) reached. "
                                 "Answer now with what you have.", True))

        @tool("search", SEARCH_DESCRIPTION, {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Descriptive search phrase in English"},
                "domain": {"type": "string", "enum": domain_enum},
                "doc_id": {"type": "string", "description": "Optional: restrict to one paper"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": MAX_TOP_K},
            },
            "required": ["query", "domain"],
        })
        async def search(args: dict) -> dict[str, Any]:
            if self._over_budget():
                self.rejected_calls += 1
                return budget_message
            return result(self.search(args["query"], args["domain"], args.get("doc_id"),
                                      args.get("top_k", DEFAULT_TOP_K)))

        @tool("list_documents", LIST_DOCUMENTS_DESCRIPTION, {
            "type": "object",
            "properties": {"domain": {"type": "string", "enum": domain_enum}},
            "required": ["domain"],
        })
        async def list_documents(args: dict) -> dict[str, Any]:
            if self._over_budget():
                self.rejected_calls += 1
                return budget_message
            return result(self.list_documents(args["domain"]))

        @tool("get_chunk_context", CONTEXT_DESCRIPTION, {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "window": {"type": "integer", "minimum": 1, "maximum": MAX_WINDOW},
            },
            "required": ["chunk_id"],
        })
        async def get_chunk_context(args: dict) -> dict[str, Any]:
            if self._over_budget():
                self.rejected_calls += 1
                return budget_message
            return result(self.get_chunk_context(args["chunk_id"], args.get("window", 1)))

        return [search, list_documents, get_chunk_context]


def main() -> None:
    tools = RAGTools()
    calls = [
        ("search", tools.search, ("CNN accuracy on four-class lung cancer CT dataset", "healthcare"),
         {"top_k": 2}),
        ("search + doc_id", tools.search, ("Grad-CAM explanations", "healthcare", "b85ac180014f"),
         {"top_k": 1}),
        ("list_documents", tools.list_documents, ("ai_security",), {}),
        ("get_chunk_context", tools.get_chunk_context, ("93c4d3a53ca1_0012",), {}),
        ("error: bad domain", tools.search, ("anything", "finance"), {}),
        ("error: doc in other domain", tools.search, ("anything", "healthcare", "c9d5afe5bfb8"), {}),
        ("error: bad chunk_id", tools.get_chunk_context, ("93c4_12",), {}),
    ]
    for label, fn, args, kwargs in calls:
        text, is_error = fn(*args, **kwargs)
        print(f"===== {label} (is_error={is_error}, ~{len(text) // 4} tokens)\n{text[:600]}\n")
    print(f"trace: {len(tools.trace)} calls, retrieved: {len(tools.retrieved)} chunks "
          f"from {sorted({h['doc_id'] for h in tools.retrieved})}")
    for entry in tools.trace:
        print("  ", {k: v for k, v in entry.items() if k not in ("chunk_ids", "scores")})


if __name__ == "__main__":
    main()
