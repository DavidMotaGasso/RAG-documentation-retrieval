"""Generate one structured "document card" per paper with Claude Haiku.

Cards summarize what makes each paper distinct (task, methods, datasets, key
results). They support hierarchical routing in the agent (domain -> document ->
chunk), which matters most inside `healthcare`, where 7 papers are closely related.

To keep cost low, the model sees the abstract, the section outline and the
conclusion/discussion sections, not the full paper.

Input:  data/processed/clean/{doc_id}.json
Output: data/processed/document_cards.json        (reviewed by hand afterwards)
        data/processed/document_cards_preview.md   (human-readable)

Existing cards are kept, so manual edits survive re-runs; pass --force to
regenerate all of them.

Usage:
    python -m src.ingestion.document_cards [--force]
"""

import argparse
import json
import re

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.ingestion.cleaner import CLEAN_DIR
from src.ingestion.inventory import PROJECT_ROOT

CARDS_PATH = PROJECT_ROOT / "data" / "processed" / "document_cards.json"
PREVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "document_cards_preview.md"

MODEL = "claude-haiku-4-5"
MAX_SOURCE_CHARS = 24_000  # ~6k tokens per paper
CLOSING_SECTIONS = re.compile(r"conclusion|discussion|outlook|summary", re.IGNORECASE)
PRICE_PER_MTOK = {"input": 1.00, "output": 5.00}

SYSTEM_PROMPT = """You write concise, factual "document cards" for research papers.
A retrieval agent reads these cards to decide which paper answers a user's question,
so the card must make the paper easy to tell apart from similar papers.

Rules:
- Use only information stated in the provided text. Do not add outside knowledge.
- Keep exact names of models, architectures, datasets and metrics (e.g. "YOLOv11",
  "LIDC-IDRI", "VGG16"), since users search for them literally.
- Key results must include the numbers as reported (metric, value, setting).
- If a field is not stated in the text, return an empty list rather than guessing.
- Write in English."""


class DocumentCard(BaseModel):
    summary: str = Field(description="One or two sentences: what the paper does and finds.")
    task: str = Field(description="The research task, e.g. 'lung nodule detection in chest CT'.")
    methods: list[str] = Field(description="Models, architectures and techniques used or proposed.")
    datasets: list[str] = Field(description="Datasets or study areas used.")
    key_results: list[str] = Field(description="Main quantitative results with their numbers.")
    distinguishing_features: str = Field(
        description="What sets this paper apart from other papers on the same topic.")
    keywords: list[str] = Field(description="5-10 search keywords.")


def source_text(doc: dict) -> str:
    """Abstract + section outline + closing sections, capped at MAX_SOURCE_CHARS."""
    body = [s for s in doc["sections"] if s["kind"] in ("abstract", "body")]
    abstract = "\n".join(p["text"] for s in body if s["kind"] == "abstract" for p in s["paragraphs"])
    outline = "\n".join(f"- {s['path']}" for s in body if s["kind"] == "body")
    closing = "\n\n".join(
        f"## {s['path']}\n" + "\n".join(p["text"] for p in s["paragraphs"])
        for s in body if s["kind"] == "body" and CLOSING_SECTIONS.search(s["heading"]))
    text = (f"# {doc['title']}\nAuthors: {doc['authors'] or 'unknown'}\n\n"
            f"## Abstract\n{abstract}\n\n## Section outline\n{outline}\n\n{closing}")
    return text[:MAX_SOURCE_CHARS]


def generate_card(client: anthropic.Anthropic, doc: dict) -> tuple[DocumentCard, dict]:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Write the document card for this paper.\n\n"
                                              f"{source_text(doc)}"}],
        output_format=DocumentCard,
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise RuntimeError(f"{doc['doc_id']}: unexpected stop_reason {response.stop_reason}")
    usage = {"input": response.usage.input_tokens, "output": response.usage.output_tokens}
    return response.parsed_output, usage


def write_preview(cards: dict[str, dict]) -> None:
    out = ["# Document cards", ""]
    for card in sorted(cards.values(), key=lambda c: (c["domain"], c["doc_id"])):
        out += [f"## {card['title']}",
                f"`{card['doc_id']}` · {card['domain']}", "",
                f"**Summary:** {card['summary']}", "",
                f"**Task:** {card['task']}", "",
                f"**Methods:** {', '.join(card['methods'])}", "",
                f"**Datasets:** {', '.join(card['datasets'])}", "",
                "**Key results:**", *[f"- {r}" for r in card["key_results"]], "",
                f"**Distinguishing features:** {card['distinguishing_features']}", "",
                f"**Keywords:** {', '.join(card['keywords'])}", ""]
    PREVIEW_PATH.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="regenerate existing cards")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    client = anthropic.Anthropic()

    cards = {} if args.force or not CARDS_PATH.exists() else json.loads(
        CARDS_PATH.read_text(encoding="utf-8"))
    total = {"input": 0, "output": 0}

    for path in sorted(CLEAN_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["doc_id"] in cards:
            continue
        card, usage = generate_card(client, doc)
        cards[doc["doc_id"]] = {"doc_id": doc["doc_id"], "title": doc["title"],
                                "domain": doc["domain"], **card.model_dump()}
        total = {k: total[k] + usage[k] for k in total}
        print(f"{doc['doc_id']}  in={usage['input']:>5} out={usage['output']:>4}  {card.task[:70]}")
        # Save after every card so a failure mid-run keeps the finished ones.
        CARDS_PATH.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")

    write_preview(cards)
    cost = sum(total[k] * PRICE_PER_MTOK[k] / 1_000_000 for k in total)
    print(f"\n{len(cards)} cards -> {CARDS_PATH.relative_to(PROJECT_ROOT)}  "
          f"(this run: {total['input']:,} in / {total['output']:,} out tokens, ~${cost:.3f})")


if __name__ == "__main__":
    main()
