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
│   │   ├── inventory.py          # corpus manifest -> data/processed/inventory.csv
│   │   ├── pdf_loader.py         # layout-aware block extraction -> data/processed/blocks/
│   │   ├── compare_extraction.py # one-off check of pdf_loader vs pymupdf4llm
│   │   ├── cleaner.py            # header/footer removal, de-hyphenation, sections -> data/processed/clean/
│   │   ├── chunker.py            # chunking strategy
│   │   └── domains.py            # domain taxonomy + manual doc -> domain map
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

Notes from Phase 0:
- With only 3 static domains, the domain list + descriptions go directly in the
  system prompt; `list_domains()` becomes optional.
- Cross-domain queries should use one `search_by_domain` call per domain rather
  than `search_global`, so a single large domain can't crowd out the top-k.
- Out-of-corpus queries (e.g. finance) must yield "not covered", not a guess.
- Candidate additions for within-domain routing: `list_documents(domain)`
  (returns document cards) and an optional `doc_id` filter on
  `search_by_domain`. Decide in Phase 3, validate in Phase 4.

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