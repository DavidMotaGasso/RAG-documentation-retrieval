"""Run the whole Phase 0 ingestion pipeline: raw PDFs -> data/processed/chunks.jsonl.

Steps: inventory -> block extraction -> cleaning/sections -> chunking.
Each step reads the previous step's output from data/processed/, so steps can
also be run one by one (see each module's docstring).

Usage:
    python -m src.ingestion.pipeline
"""

from src.ingestion import chunker, cleaner, inventory, pdf_loader

STEPS = [
    ("0.3 inventory", inventory.main),
    ("0.4 block extraction", pdf_loader.main),
    ("0.5 cleaning and sections", cleaner.main),
    ("0.8 chunking", chunker.main),
]


def main() -> None:
    for name, step in STEPS:
        print(f"\n=== {name} ===")
        step()


if __name__ == "__main__":
    main()
