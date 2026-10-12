# rag-practice — Project Context

## Goal

Build an **AI agent** (not a static/naive RAG pipeline) that answers user questions
over a multi-domain corpus of research papers, retrieving only the data strictly
necessary per query — optimizing for efficiency, cost, and answer quality.

## Dataset

"rag-practice": a curated collection of research paper PDFs spanning multiple
domains (AI, NLP, Software, Healthcare, Finance, etc.). Files are raw/unprocessed,
long-form, and mix relevant/irrelevant content across domains — designed to expose
the limitations of naive RAG (retrieval noise, poor chunking, weak embedding
retrieval).

## Key design decisions

- **Orchestration**: Claude Agent SDK (not LangChain / raw API loop). Rationale:
  same tool already used successfully in the `football-predictor` project; the
  task needs genuine agent autonomy (deciding which tool to call, in which order),
  which justifies an agent over a fixed pipeline.
- **Vector database**: Chroma (local, persisted to disk). Chosen over FAISS
  because it has native metadata filtering, which is essential for the
  domain-based routing strategy below. FAISS is a possible later exploration.
- **Embeddings model**: local, via `sentence-transformers` (e.g.
  `all-MiniLM-L6-v2` or `bge-small-en`). No third-party paid APIs (OpenAI,
  Voyage) — consistent with the project owner's standing preference to avoid
  paid/external services and not send data off-machine when a local alternative
  exists.
- **LLM model**: start with `claude-haiku-4-5` for cost efficiency (same choice
  made in `football-predictor`). Reassess if answer quality is insufficient
  during evals — possibly upgrade only the final-generation step.
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
│   │   ├── pdf_loader.py         # text extraction from PDFs
│   │   ├── chunker.py            # chunking strategy
│   │   └── domain_classifier.py  # tags each chunk/doc with a domain
│   ├── indexing/
│   │   ├── embeddings.py         # sentence-transformers wrapper
│   │   └── vector_store.py       # Chroma wrapper (index + metadata filters)
│   ├── agent/
│   │   ├── tools.py              # tool definitions for the Claude Agent SDK
│   │   └── agent.py              # agent setup, system prompt
│   ├── evals/
│   │   ├── generate_testset.py   # synthetic Q&A generation
│   │   ├── metrics.py            # evaluation metrics (e.g. via RAGAS)
│   │   └── run_evals.py
│   └── main.py                   # CLI entry point
├── chroma_db/                    # persisted vector index (gitignored)
├── requirements.txt
└── README.md
```

## Agent tools (draft)

| Tool | Purpose | When the agent calls it |
|---|---|---|
| `list_domains()` | Returns available domains in the corpus with a short descriptor each | At session start, or when unsure which domain a query belongs to |
| `search_by_domain(query, domain, top_k)` | Semantic search in Chroma filtered by `domain` metadata | When the query clearly points to one domain — the "cheap" path |
| `search_global(query, top_k)` | Semantic search across the whole corpus, no filter | Fallback for ambiguous or cross-domain queries |
| `get_chunk_context(doc_id, chunk_id)` | Returns neighboring chunks of a given result | When a retrieved chunk seems incomplete |

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
- Discussion/planning with the project owner happens in Catalan.