# rag-practice — Project Context

## Goal

Build an **AI agent** (not a static/naive RAG pipeline) that answers user questions
over a multi-domain corpus of research papers, retrieving only the data strictly
necessary per query — optimizing for efficiency, cost, and answer quality.

## Dataset

"rag-practice": a curated collection of research paper PDFs. Files are
raw/unprocessed, long-form, and mix relevant/irrelevant content — designed to
expose the limitations of naive RAG (retrieval noise, poor chunking, weak
embedding retrieval). PDFs live locally in `data/raw/` and are never committed.

Actual corpus (from `data/processed/inventory.csv`): 12 PDFs, 218 pages,
~220k tokens, all with a text layer (no OCR needed). One review paper
(`36ecde91c4ff`, ~62k tokens) is ~28% of the corpus.

### Domain taxonomy

Defined in `src/ingestion/domains.py`, assigned manually per document (no
automatic classifier at this corpus size). Domains are by **application area**,
not technique — nearly every paper is applied ML, so an "AI" domain would not
filter anything.

| Domain | Docs | Share of tokens |
|---|---|---|
| `healthcare` (lung cancer: CT detection/segmentation/classification, NSCLC microbiome) | 7 | ~44% |
| `remote_sensing` (satellite imagery, land cover, NDVI, Sentinel-2) | 4 | ~50% |
| `ai_security` (LLM prompt injection, authenticated prompts) | 1 | ~6% |

**Main retrieval challenge**: domain routing separates lung / satellite /
security well, but the hard part is *within* `healthcare` — 7 closely related
lung-cancer CT papers (hardest pair: `b85ac180014f` vs `93c4d3a53ca1`, both
CNN-based CT classification). Mitigation is layered:

1. (Phase 0) Every chunk carries `doc_id`, `title`, `authors`, `section`
   metadata, and a contextual header (`[Paper: ... | Section: ...]`) is
   prepended to the chunk text before embedding.
2. (Phase 0 → used in Phase 3) Per-document cards (task, method, dataset, key
   results) to support hierarchical routing: domain → document → chunk.
3. (Phase 5, if evals require) Hybrid BM25 + vector search for exact
   identifiers (`YOLOv11`, `VGG16`, `LIDC-IDRI`).
4. (Phase 4) Evals measure whether the *correct paper* was retrieved, not
   only answer quality.

No sub-domains or extra per-document tags (task/modality) for now.

## Key design decisions

- **Orchestration**: Claude Agent SDK (not LangChain / raw API loop). Rationale:
  same tool already used successfully in the `football-predictor` project; the
  task needs genuine agent autonomy (deciding which tool to call, in which order),
  which justifies an agent over a fixed pipeline.
- **Vector database**: Chroma (local, persisted to disk). Chosen over FAISS
  because it has native metadata filtering, which is essential for the
  domain-based routing strategy below. FAISS is a possible later exploration.
- **Embeddings model**: local, via `sentence-transformers`:
  **`BAAI/bge-small-en-v1.5`** (512-token limit, 384 dims). Chosen over
  `all-MiniLM-L6-v2` because MiniLM truncates at 256 tokens (trained on 128),
  which would leave ~200 tokens per chunk once the contextual header is added.
  No third-party paid APIs (OpenAI, Voyage) — consistent with the project
  owner's standing preference to avoid paid/external services and not send data
  off-machine when a local alternative exists. The corpus is English-only, so
  the agent must write search queries in English even when the user asks in
  another language.
- **Chunking** (decided in Phase 0.6 from `src/ingestion/analyze_corpus.py`):
  section-aware, **~300 tokens of text** per chunk plus the contextual header
  (`[Paper: ... | Section: ...]`, 30 tokens median, 58 max). Only `abstract` and
  `body` sections are indexed (front matter, back matter and references are
  excluded). Indexable corpus: ~129k tokens -> ~600 chunks. Revisit size (e.g.
  400) in Phase 5 using evals.
- **Vector index** (Phase 1, `src/indexing/`): one Chroma collection
  (`chunks`, 585 entries) with metadata filters on `domain` / `doc_id`; cosine
  distance; embeddings computed by our `Embedder` and passed explicitly
  (collection created with `embedding_function=None`, otherwise Chroma silently
  uses its default MiniLM); the embedding model name is stored on the
  collection and checked on open. Stored `document` = chunk `text` (no header).
  bge's query instruction is **off** by default (equal or slightly worse in the
  Phase 1.5 check; re-test in Phase 4). Document cards stay in
  `data/processed/document_cards.json` (no vectors; 12 cards are read whole).
- **LLM model**: `claude-haiku-5-5` for both the naive baseline and the agent
  (same model on both sides keeps the Phase 4 comparison fair). Switched from
  `claude-haiku-4-5` in Phase 2: ~10x cheaper per token ($0.10 / $0.50 per MTok
  for prompts up to 100K), though its tokenizer counts the same text as ~30-45%
  more tokens. It thinks by default; we run it at `effort: low` and leave
  `max_tokens` room for thinking. Reassess if answer quality is insufficient
  during evals — possibly upgrade only the final-generation step. (The
  document cards were generated earlier with Haiku 4.5 and were not redone.)
- **Ground truth for evals**: no existing Q&A pairs; will be generated
  synthetically from the corpus itself (via LLM) rather than hand-labeled.
- **Efficiency strategy**: the agent should route queries to a domain-filtered
  search whenever possible, and only fall back to a corpus-wide search when the
  domain is ambiguous — this is the core mechanism for avoiding "search
  everything, every time" naive RAG behavior.

## Out of scope (for now)

- Embedding fine-tuning
- Production deployment / polished UI
- Multi-user auth

## Project structure

```
rag-practice/
├── data/
│   ├── raw/                     # original PDFs
│   └── processed/                # extracted text / chunks (cache, optional)
├── src/
│   ├── ingestion/
│   │   ├── inventory.py          # corpus manifest -> data/processed/inventory.csv
│   │   ├── pdf_loader.py         # layout-aware block extraction -> data/processed/blocks/
│   │   ├── compare_extraction.py # one-off check of pdf_loader vs pymupdf4llm
│   │   ├── cleaner.py            # header/footer removal, de-hyphenation, sections -> data/processed/clean/
│   │   ├── analyze_corpus.py     # token-length analysis used to pick chunk size/model
│   │   ├── chunker.py            # section-aware chunking -> data/processed/chunks.jsonl
│   │   ├── pipeline.py           # runs inventory -> pdf_loader -> cleaner -> chunker
│   │   ├── document_cards.py     # one Haiku-generated card per paper (for list_documents)
│   │   └── domains.py            # domain taxonomy + manual doc -> domain map
│   ├── indexing/
│   │   ├── embeddings.py         # bge-small wrapper (documents vs. queries, normalized)
│   │   ├── vector_store.py       # Chroma: build index, search with filters, get_context
│   │   └── retrieval_check.py    # Phase 1.5 sanity check on 20 hand-labeled queries
│   ├── baseline/
│   │   └── naive_rag.py          # Phase 2: global top-5 search + one Haiku call
│   ├── agent/
│   │   ├── tools.py              # search / list_documents / get_chunk_context + trace (RAGTools)
│   │   └── agent.py              # RAGAgent: Agent SDK setup, system prompt, answer() -> RAGAnswer
│   ├── evals/
│   │   ├── generate_testset.py   # synthetic Q&A generation
│   │   ├── metrics.py            # evaluation metrics (e.g. via RAGAS)
│   │   └── run_evals.py
│   ├── rag_answer.py             # RAGAnswer result contract + shared LLM settings (model, effort, prices)
│   └── main.py                   # CLI entry point
├── chroma_db/                    # persisted vector index (gitignored)
├── requirements.txt
└── README.md
```

## Agent tools (decided in Phase 3.1)

| Tool | Purpose | When the agent calls it |
|---|---|---|
| `search(query, domain, doc_id?, top_k?)` | Semantic search. `domain` is **required**: `healthcare`, `remote_sensing`, `ai_security` or `all` (= global search). Optional `doc_id` restricts it to one paper. `top_k` default 5, max 10 | Every retrieval; one call per domain for cross-domain questions, `all` only when the domain is unclear |
| `list_documents(domain)` | Compact document cards: `doc_id`, title, task, methods, datasets, distinguishing features (~1.5k tokens for healthcare, ~2.7k for `all`) | When the question targets a specific paper among similar ones, or spans several papers, before a `doc_id`-filtered search |
| `get_chunk_context(chunk_id, window?)` | A chunk plus its neighbours, overlap removed | When a retrieved chunk seems cut off |

Phase 3.1 design decisions:
- One `search` tool with a required `domain` (instead of `search_by_domain` +
  `search_global`): forces an explicit routing decision on every call, fewer
  tools for Haiku to choose from; global searches stay visible in traces.
- `list_domains()` dropped: the 3 domains go in the system prompt.
- Results show `[chunk_id] title | section | pages` + text, **without
  similarity scores** (Phase 1: they mislead about relevance); scores are kept
  in the trace only.
- Cards omit `key_results`: cards are LLM summaries for *choosing* a paper; the
  agent must answer from chunks. `methods` is included (it identifies which
  papers use a technique, e.g. ResNet-50 -> `3d40`, `93c4`, `b85a`).
- `list_documents` is a hypothesis: ablate it (agent with vs. without) in
  Phase 4 and drop it if it doesn't help.
- The agent always runs at least one search, even for apparently
  out-of-corpus questions; reformulates at most once before "not covered".
- Max 6 tool calls per question; same model as the baseline (Haiku 5.5,
  `effort: low`; try `medium` if routing is poor). Stateless: one question per
  call, no conversation memory (Phase 6).
- Returns a `RAGAnswer` like the baseline: `retrieved` = every chunk returned
  by tools (deduplicated, in order), `llm_calls` = model turns,
  `meta.trace` = tool calls with arguments and returned papers.

Phase 3.2 Agent SDK check (`claude-agent-sdk` 0.2.165, Claude Code CLI 2.1.283):
- Works with `claude-haiku-5-5`, `effort="low"`, a custom `system_prompt`
  string and in-process tools (`create_sdk_mcp_server`, tool names
  `mcp__<server>__<tool>`, listed in `allowed_tools`).
- Required options to keep it a clean RAG agent: `tools=[]` (no built-in
  Bash/Read/...), `setting_sources=[]` (don't load CLAUDE.md / user settings),
  `strict_mcp_config=True`, `cwd` outside the repo, and
  `env={"ANTHROPIC_API_KEY": ..., "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
  "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}`. Passing the API key makes
  the CLI bill the API (not the user's claude.ai login); the two flags remove
  a hidden auxiliary model call (~1.2k in / 0.5k out tokens) seen otherwise.
- Overhead: ~360 fixed input tokens per turn beyond our own prompt; ~0.7 s
  process start per question (the very first cold start took ~6 s).
- The CLI applies prompt caching automatically (writes 1.25x, reads 0.1x), so
  turn 2+ and consecutive questions read the system prompt + tool definitions
  from cache. Cost must therefore be computed with cache tokens.
- Use `ResultMessage.model_usage[model]` / `total_cost_usd` as the source of
  truth; per-`AssistantMessage` usage is partial (streamed).
- A 2-turn question (1 search) cost ~$0.0004 and ~3.3 s, comparable to the
  baseline's ~$0.0004 and 1.5–5 s.

Notes from Phase 0:
- With only 3 static domains, the domain list + descriptions go directly in the
  system prompt; `list_domains()` becomes optional.
- Cross-domain queries should use one `search_by_domain` call per domain rather
  than `search_global`, so a single large domain can't crowd out the top-k.
- Out-of-corpus queries (e.g. finance) must yield "not covered", not a guess.
- Within-domain routing: `list_documents(domain)` and the `doc_id` filter on
  `search` (adopted in Phase 3.1, validate in Phase 4).

Notes from Phase 1 (retrieval check, `src/indexing/retrieval_check.py`):
- **Similarity scores cannot detect out-of-corpus questions**: best in-corpus
  top-1 score 0.669–0.706 vs. out-of-corpus up to 0.719. No threshold works;
  the agent must read the retrieved chunks and judge relevance itself (state
  this in the system prompt).
- The review paper `36ecde91c4ff` attracts generic ML questions (LSTM, BERT
  queries returned 5/5 chunks from it). Watch for this in evals.
- Bare identifiers retrieve poorly ("Results on the LOTUS benchmark" misses;
  "lung tumor segmentation on the LOTUS dataset" ranks the right paper first).
  The agent should write descriptive search queries, not bare names. The same
  model is spelled `ResNet-50` and `ResNet50` across papers (matters for BM25).
- Top-5 results tend to come from a single paper; fine for single-paper
  questions, a risk for multi-paper ones.
- `VectorStore.search` / `get_context` are the building blocks for
  `search_by_domain`, `search_global` and `get_chunk_context`.

Naive baseline (Phase 2, `src/baseline/naive_rag.py`): the raw question is
embedded as-is (no rewriting, no translation, no domain filter), top-5 chunks
from the whole corpus go to one `claude-haiku-5-5` call as numbered excerpts.
It is not a straw man: same index, same model and the same "say it is not
covered" instruction the agent will have, so Phase 4 isolates the retrieval
strategy. It returns a `RAGAnswer` (answer, retrieved chunks, tokens, cost,
latency; defined in `src/rag_answer.py` with the shared model, effort and
prices) — the agent returns the same object. First trial (5 questions,
~$0.0004 and ~2.5k input tokens each with Haiku 5.5): out-of-corpus correctly
refused; a Catalan question retrieved mostly the wrong paper (no translation);
a multi-paper question retrieved 3 papers but the answer only confirmed 2.

No reranking tool in this first version — revisit in the improvement phase once
baseline evals show whether it's worth the added cost.

## Development phases

0. **Data prep** — inventory PDFs, extract text, decide chunking strategy.
1. **Indexing** — generate embeddings, load into Chroma with domain metadata.
2. **Naive baseline** — simple top-k-over-everything RAG, for comparison.
3. **Agent build** — implement tools above + system prompt with domain-first
   routing logic.
4. **Evals** — generate synthetic Q&A set, implement metrics (retrieval
   precision/recall, faithfulness, answer relevancy, cost/latency), compare
   agent vs. naive baseline.
5. **Iterate** — based on eval results: tune chunking, consider reranking,
   consider hybrid search (BM25 + vector), refine system prompt.
6. **(Optional) Interface** — CLI or minimal API for interactive use.

## Conventions

- All code (comments, commit messages, variable/function names) in English.
- Keep this file current: when a decision is made or changed, update the
  relevant section here (it reflects the *current* state; the history and
  rationale go in the learning guide).
- The project owner keeps two personal **PDF guides** in `guide/` (gitignored,
  not part of the project), both built with `python guide/build_guide.py`:
  - `rag_learning_guide.md` → `RAG_Learning_Guide.pdf`: at the end of every
    phase, add what was built, the decisions with their alternatives, and the
    lessons learned. It is a learning project, so the "why" matters as much as
    the "what"; write it as a reading, not a log.
  - `project_structure.md` → `Project_Structure.pdf`: what every folder/file
    does and how data flows. Update it whenever files or folders are added,
    removed or change purpose.
- Discussion/planning with the project owner happens in Catalan.