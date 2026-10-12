# RAG documentation retrieval

An AI agent (Claude Agent SDK) that answers questions over a multi-domain corpus
of research papers, retrieving only the data strictly necessary per query.
See [CLAUDE.md](CLAUDE.md) for design decisions and development phases.

## Setup

Requires Python 3.12+ (tested on 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp .env.example .env            # then set ANTHROPIC_API_KEY
```

## Data

Source PDFs are **not** committed. Copy them into `data/raw/` locally
(the folder is gitignored, except for its `.gitkeep`). Processed artifacts
are written to `data/processed/`, also gitignored.

## Data pipeline and outputs

```bash
python -m src.ingestion.pipeline         # PDFs -> chunks.jsonl (reproducible)
python -m src.ingestion.document_cards   # one card per paper (needs ANTHROPIC_API_KEY)
```

| Step | Module | Output in `data/processed/` |
|---|---|---|
| Inventory | `inventory.py` | `inventory.csv`: one row per PDF (`doc_id` = content hash) |
| Extraction | `pdf_loader.py` | `blocks/{doc_id}.jsonl`: text blocks in reading order, with position and font |
| Cleaning | `cleaner.py` | `clean/{doc_id}.json`: sections and paragraphs, noise removed; `cleaning_report.csv` |
| Chunking | `chunker.py` | `chunks.jsonl`: retrieval-ready chunks (schema below) |
| Document cards | `document_cards.py` | `document_cards.json`: per-paper summary (task, methods, datasets, results) |

Each step reads the previous step's output, so steps can also be run one by
one (`python -m src.ingestion.<module>`). Human-readable previews are written
to `preview/` and `clean_preview/`.

**`chunks.jsonl` schema** (one JSON object per line):

| Field | Description |
|---|---|
| `chunk_id` | `{doc_id}_{chunk_index:04d}` |
| `doc_id`, `chunk_index` | Source document; position within it (sequential, for fetching neighbours) |
| `domain` | `healthcare`, `remote_sensing` or `ai_security` (see `src/ingestion/domains.py`) |
| `title`, `authors` | From the PDF metadata |
| `section`, `section_kind` | Section path (e.g. `3. Results > 3.2 Models' Performance`); `abstract` or `body` |
| `page_start`, `page_end` | Source pages (1-based) |
| `text` | Chunk text; starts with up to 45 tokens of overlap from the previous chunk |
| `overlap_chars` | Length of that overlap prefix (strip it when stitching neighbours) |
| `embedding_text` | `[Paper: {title} \| Section: {section}]` + `text`: what gets embedded |
| `n_tokens`, `n_embedding_tokens` | Token counts with the `bge-small-en-v1.5` tokenizer |

## Vector index

```bash
python -m src.indexing.vector_store      # embed chunks.jsonl and (re)build chroma_db/ (~70 s on CPU)
python -m src.indexing.retrieval_check   # sanity check on 20 hand-labeled queries
```

The index is a single Chroma collection (`chunks`) in `chroma_db/` (gitignored),
using cosine distance and `BAAI/bge-small-en-v1.5` embeddings. Rebuild it
whenever `chunks.jsonl` or the embedding model changes; opening an index built
with a different model raises an error.

```python
from src.indexing.vector_store import VectorStore

store = VectorStore()
store.search("Which YOLO version performed best?", domain="healthcare", top_k=5)
store.get_context("93c4d3a53ca1", chunk_index=18, window=1)  # neighbours, overlap removed
```
