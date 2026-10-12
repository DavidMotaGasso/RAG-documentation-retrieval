"""Build an inventory (manifest) of the raw PDF corpus.

For each PDF in data/raw/ it records a stable doc_id, basic file/PDF stats,
whether the PDF has a usable text layer, a best-guess title and an abstract
snippet. The output (data/processed/inventory.csv) is the input for domain
labeling and gives a first picture of the corpus before extraction.

Usage:
    python -m src.ingestion.inventory
"""

import csv
import hashlib
import re
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "inventory.csv"

# A page with fewer extractable characters than this is treated as image-only.
MIN_CHARS_PER_TEXT_PAGE = 100
# Rough English heuristic; precise counts come later with the embedding tokenizer.
CHARS_PER_TOKEN = 4
SNIPPET_CHARS = 500

ABSTRACT_RE = re.compile(r"\babstract\b[\s.:—–-]*", re.IGNORECASE)
REFERENCES_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


def compute_doc_id(path: Path) -> str:
    """Content hash, so the id survives renames and exposes duplicates."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def guess_title(doc: pymupdf.Document) -> str:
    """Pick the text set in the largest font on page 1.

    PDF metadata titles are often empty or junk (e.g. 'Microsoft Word - draft.docx'),
    so the visual title is usually a better signal.
    """
    if doc.page_count == 0:
        return ""
    spans = []
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if line["dir"] != (1.0, 0.0):  # skip rotated text, e.g. the arXiv side stamp
                continue
            for span in line["spans"]:
                text = span["text"].strip()
                if len(text) > 3:
                    spans.append((round(span["size"], 1), text))
    if not spans:
        return ""
    max_size = max(size for size, _ in spans)
    title = " ".join(text for size, text in spans if size == max_size)
    return re.sub(r"\s+", " ", title)[:200]


def pick_title(meta_title: str, visual_title: str) -> str:
    """Prefer the metadata title unless it looks like a placeholder."""
    junk = ("microsoft word", "untitled", ".doc", ".pdf", ".tex")
    if len(meta_title) > 15 and not any(j in meta_title.lower() for j in junk):
        return meta_title
    return visual_title


def extract_abstract_snippet(first_pages_text: str) -> str:
    """Text right after an 'Abstract' marker, or the start of page 1 as fallback."""
    match = ABSTRACT_RE.search(first_pages_text)
    start = match.end() if match else 0
    snippet = re.sub(r"\s+", " ", first_pages_text[start:]).strip()
    return snippet[:SNIPPET_CHARS]


def inspect_pdf(path: Path) -> dict:
    with pymupdf.open(path) as doc:
        page_texts = [page.get_text() for page in doc]
        meta = doc.metadata or {}
        title_guess = guess_title(doc)

    chars_per_page = [len(t.strip()) for t in page_texts]
    total_chars = sum(chars_per_page)
    image_only_pages = sum(c < MIN_CHARS_PER_TEXT_PAGE for c in chars_per_page)
    full_text = "\n".join(page_texts)
    meta_title = (meta.get("title") or "").strip()

    return {
        "doc_id": compute_doc_id(path),
        "filename": path.name,
        "size_kb": round(path.stat().st_size / 1024),
        "n_pages": len(page_texts),
        "image_only_pages": image_only_pages,
        "needs_ocr": image_only_pages > len(page_texts) / 2,
        "total_chars": total_chars,
        "est_tokens": total_chars // CHARS_PER_TOKEN,
        "has_references_heading": bool(REFERENCES_RE.search(full_text)),
        "title": pick_title(meta_title, title_guess),
        "meta_title": meta_title,
        "meta_author": (meta.get("author") or "").strip(),
        "title_guess": title_guess,
        "abstract_snippet": extract_abstract_snippet("\n".join(page_texts[:2])),
    }


def main() -> None:
    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {RAW_DIR}")

    rows = []
    for path in pdf_paths:
        try:
            rows.append(inspect_pdf(path))
        except Exception as exc:  # keep going; one broken PDF shouldn't stop the inventory
            print(f"[ERROR] {path.name}: {exc}")

    seen: dict[str, str] = {}
    for row in rows:
        if row["doc_id"] in seen:
            print(f"[DUPLICATE] {row['filename']} == {seen[row['doc_id']]}")
        seen.setdefault(row["doc_id"], row["filename"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so the file opens correctly in Excel on Windows
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'doc_id':<13}{'pages':>6}{'tokens':>9}{'ocr':>5}  title")
    for r in rows:
        print(f"{r['doc_id']:<13}{r['n_pages']:>6}{r['est_tokens']:>9}"
              f"{'yes' if r['needs_ocr'] else 'no':>5}  {r['title'][:85]}")
    print(f"\n{len(rows)} PDFs, {sum(r['n_pages'] for r in rows)} pages, "
          f"~{sum(r['est_tokens'] for r in rows):,} tokens -> {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
