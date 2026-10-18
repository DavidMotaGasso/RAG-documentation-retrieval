"""The RAG agent: Claude Haiku 5.5 deciding where and how to search (Phase 3).

Same model, index and answer rules as the naive baseline; what differs is that
the model drives retrieval through three tools (src/agent/tools.py): it picks
the domain, rewrites the query in English, can choose a paper from the
document cards first, and searches again if needed. Returns a RAGAnswer, like
the baseline, with the tool-call trace in `meta`.

Claude Agent SDK settings (checked in Phase 3.2, see CLAUDE.md): no built-in
Claude Code tools, no settings / CLAUDE.md loaded, API key passed explicitly
(bill the API, not a claude.ai login), and non-essential CLI model calls off.

Usage:
    python -m src.agent.agent "Which YOLO version worked best for lung cancer detection?"
    python -m src.agent.agent --show-trace "..."
"""

import argparse
import asyncio
import tempfile
import time

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock,
                              create_sdk_mcp_server, query)
from dotenv import dotenv_values

from src.agent.tools import RAGTools
from src.indexing.vector_store import VectorStore
from src.ingestion.domains import DOMAINS
from src.ingestion.inventory import PROJECT_ROOT
from src.rag_answer import EFFORT, MODEL, RAGAnswer

MAX_TOOL_CALLS = 6
SERVER = "rag"
TOOL_NAMES = [f"mcp__{SERVER}__{name}" for name in ("search", "list_documents", "get_chunk_context")]

SYSTEM_PROMPT = f"""You answer questions about a fixed collection of research papers, using search tools.

The collection has three domains:
{chr(10).join(f"- {name}: {description}" for name, description in DOMAINS.items())}
Nothing outside these topics is covered.

How to search:
1. Always search at least once before answering, even if the question looks unrelated.
2. Write search queries in English, whatever the language of the question, as
   descriptive phrases rather than bare names.
3. Search in the domain the question belongs to. If it spans several domains, search
   each one separately. Use domain "all" only when you cannot tell.
4. Call list_documents for the domain first, then search with the doc_id of each
   relevant paper, when:
   - the question asks which papers do something, or
   - it mentions a model, dataset or technique without saying which paper, and the
     domain has several similar papers (e.g. the lung-cancer CT papers): several
     papers may report it, and the answer must cover each of them.
5. If a retrieved chunk seems cut off, use get_chunk_context.
6. You have at most {MAX_TOOL_CALLS} tool calls. Stop as soon as you have what you need.

How to answer:
- Judge whether the results actually answer the question; similar wording is not
  enough. If they don't, rephrase the search once. If that fails too, say briefly that
  the documents do not cover it instead of guessing; don't describe unrelated material.
- Use only facts from search results: not your own knowledge, and not the document
  cards (they are only for choosing papers).
- Cite the chunks you use by their id, e.g. [93c4d3a53ca1_0012].
- When facts come from different papers, say which paper each one comes from.
- Keep metric names exact (accuracy, precision, recall and F1 are different). If the
  question's wording is ambiguous, say which metric each number is.
- Answer in the language of the question. Be concise."""


class RAGAgent:
    def __init__(self, store: VectorStore | None = None, max_tool_calls: int = MAX_TOOL_CALLS,
                 effort: str = EFFORT):
        self.tools = RAGTools(store, max_calls=max_tool_calls)
        self.server = create_sdk_mcp_server(SERVER, tools=self.tools.sdk_tools())
        env = dotenv_values(PROJECT_ROOT / ".env")
        self.options = ClaudeAgentOptions(
            model=MODEL,
            effort=effort,
            system_prompt=SYSTEM_PROMPT,
            tools=[],                           # no built-in Claude Code tools
            mcp_servers={SERVER: self.server},
            allowed_tools=TOOL_NAMES,
            strict_mcp_config=True,             # ignore user / project MCP servers
            setting_sources=[],                 # don't load CLAUDE.md or user settings
            # One turn per tool call plus the final answer; the hard limit on
            # tool calls is enforced by RAGTools.
            max_turns=max_tool_calls + 2,
            cwd=tempfile.gettempdir(),          # outside the repo: nothing to pick up
            env={"ANTHROPIC_API_KEY": env["ANTHROPIC_API_KEY"],
                 "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
                 "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
        )

    def answer(self, question: str) -> RAGAnswer:
        return asyncio.run(self._answer(question))

    async def _answer(self, question: str) -> RAGAnswer:
        self.tools.reset()
        start = time.perf_counter()
        texts: list[str] = []                   # last assistant text, fallback if no result
        result: ResultMessage | None = None
        async for message in query(prompt=question, options=self.options):
            if isinstance(message, AssistantMessage):
                turn_text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
                if turn_text:
                    texts.append(turn_text)
            elif isinstance(message, ResultMessage):
                result = message
        if result is None:
            raise RuntimeError("The agent finished without a result message")

        usage = (result.model_usage or {}).get(MODEL, {})
        return RAGAnswer(
            question=question,
            answer=result.result or (texts[-1] if texts else ""),
            retrieved=self.tools.retrieved,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            cache_write_tokens=usage.get("cacheCreationInputTokens", 0),
            cache_read_tokens=usage.get("cacheReadInputTokens", 0),
            latency_s=round(time.perf_counter() - start, 2),
            llm_calls=result.num_turns,
            meta={"trace": self.tools.trace,
                  "rejected_tool_calls": self.tools.rejected_calls,
                  "terminal_reason": result.terminal_reason,
                  "is_error": result.is_error,
                  "sdk_cost_usd": result.total_cost_usd,
                  "api_latency_s": result.duration_api_ms / 1000},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--show-trace", action="store_true", help="print each tool call")
    args = parser.parse_args()

    result = RAGAgent().answer(args.question)
    print(result.answer)
    if args.show_trace:
        print("\nTrace:")
        for i, call in enumerate(result.meta["trace"], start=1):
            shown = {k: v for k, v in call["args"].items() if v is not None}
            outcome = call.get("error") or " ".join(sorted({d[:4] for d in call["doc_ids"]}))
            print(f"  {i}. {call['tool']} {shown} -> {outcome}")
    print(f"\nRetrieved {len(result.retrieved)} chunks from "
          f"{', '.join(d[:4] for d in result.retrieved_doc_ids)}")
    print(f"{result.llm_calls} LLM calls, {result.input_tokens:,} in / "
          f"{result.cache_write_tokens:,} cache write / {result.cache_read_tokens:,} cache read / "
          f"{result.output_tokens:,} out tokens, ${result.cost_usd:.4f} "
          f"(SDK: ${result.meta['sdk_cost_usd']:.4f}), {result.latency_s:.1f}s")


if __name__ == "__main__":
    main()
