"""One-off sanity check: compare our block extraction against pymupdf4llm.

Writes both outputs side by side to data/processed/comparison/ for a small
sample of documents (one per layout type) and prints simple quality signals.
pymupdf4llm is only used here as a reference, not in the pipeline.

Usage:
    python -m src.ingestion.compare_extraction
"""

import re

import pymupdf4llm

from src.ingestion.inventory import PROJECT_ROOT, RAW_DIR, compute_doc_id
from src.ingestion.pdf_loader import extract_blocks

COMPARISON_DIR = PROJECT_ROOT / "data" / "processed" / "comparison"

SAMPLE_DOC_IDS = {
    "93c4d3a53ca1": "two-column (Elsevier), footer between columns",
    "cf2ee32231c8": "two-column (npj), missing space characters",
    "9508bda7b433": "single-column",
}


def quality_signals(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+", text)
    return {
        "words": len(words),
        "glued_words(>=18)": sum(len(w) >= 18 for w in words),
        "ligatures": len(re.findall("[ﬀ-ﬆ]", text)),
        "soft_hyphens": text.count("­"),
    }


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    paths = {compute_doc_id(p): p for p in RAW_DIR.glob("*.pdf")}

    for doc_id, layout in SAMPLE_DOC_IDS.items():
        path = paths[doc_id]
        ours = "\n\n".join(b["text"] for b in extract_blocks(path, doc_id))
        reference = pymupdf4llm.to_markdown(str(path), show_progress=False)

        (COMPARISON_DIR / f"{doc_id}.ours.txt").write_text(ours, encoding="utf-8")
        (COMPARISON_DIR / f"{doc_id}.pymupdf4llm.md").write_text(reference, encoding="utf-8")

        print(f"\n{doc_id} — {layout}")
        print(f"  ours        {quality_signals(ours)}")
        print(f"  pymupdf4llm {quality_signals(reference)}  "
              f"headings={len(re.findall(r'^#+ ', reference, re.MULTILINE))}")

    print(f"\nOutputs written to {COMPARISON_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
