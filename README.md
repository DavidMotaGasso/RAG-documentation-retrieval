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
