"""Extract layout-aware text blocks from the raw PDFs.

Each PDF becomes data/processed/blocks/{doc_id}.jsonl: one text block per line,
in reading order, with its page, position and dominant font. No cleaning
happens here; header/footer removal, de-hyphenation and section detection
(Phase 0.5) work on these blocks and need the layout information kept here.

A plain-text preview per document is written to data/processed/preview/ so the
reading order can be checked by eye.

Usage:
    python -m src.ingestion.pdf_loader
"""

import json
from collections import Counter
from pathlib import Path

import pymupdf

from src.ingestion.inventory import RAW_DIR, PROJECT_ROOT, compute_doc_id

BLOCKS_DIR = PROJECT_ROOT / "data" / "processed" / "blocks"
PREVIEW_DIR = PROJECT_ROOT / "data" / "processed" / "preview"

# Default "dict"/"rawdict" flags minus ligature preservation (so "ﬁ" comes out as "fi")
# and minus images (we only want text). De-hyphenation stays off on purpose:
# it would turn compounds like "Sentinel-\n2" into "Sentinel2" (see Phase 0.5).
EXTRACT_FLAGS = (
    pymupdf.TEXTFLAGS_DICT
    & ~pymupdf.TEXT_PRESERVE_LIGATURES
    & ~pymupdf.TEXT_PRESERVE_IMAGES
)

# A block counts as belonging to one column if it stays on its side of the
# page centre, allowing this fraction of page width as slack.
COLUMN_TOLERANCE = 0.02
BOLD_FLAG = 16  # bit set in span["flags"] for bold fonts
# Insert a space when the gap between two characters exceeds this fraction of
# the font size. Some PDFs (e.g. the npj microbiome paper) store justified text
# without space characters, and PyMuPDF's own space detection then glues words
# together ("Thelungmicrobiota"). 0.10 fixes that with negligible false splits
# on the rest of the corpus (measured: 5 extra tokens over 12 PDFs).
SPACE_GAP_RATIO = 0.10


def _line_text(line: dict) -> str:
    """Rebuild a line's text from character positions, inferring missing spaces."""
    out: list[str] = []
    prev_x1 = None
    for span in line["spans"]:
        for ch in span["chars"]:
            c = ch["c"]
            if (prev_x1 is not None and c != " " and out and out[-1] != " "
                    and ch["bbox"][0] - prev_x1 > SPACE_GAP_RATIO * span["size"]):
                out.append(" ")
            out.append(c)
            prev_x1 = ch["bbox"][2]
    return "".join(out)


def _block_text_and_font(block: dict) -> tuple[str, float, bool]:
    """Join the block's horizontal lines; return text, dominant size and boldness."""
    lines = []
    size_chars: Counter[float] = Counter()
    bold_chars = total_chars = 0
    for line in block["lines"]:
        if line["dir"] != (1.0, 0.0):  # rotated text, e.g. the arXiv side stamp
            continue
        line_text = _line_text(line)
        if not line_text.strip():
            continue
        lines.append(line_text)
        for span in line["spans"]:
            n = sum(1 for ch in span["chars"] if not ch["c"].isspace())
            size_chars[round(span["size"], 1)] += n
            total_chars += n
            if span["flags"] & BOLD_FLAG or "bold" in span["font"].lower():
                bold_chars += n
    text = "\n".join(lines)
    font_size = size_chars.most_common(1)[0][0] if size_chars else 0.0
    is_bold = total_chars > 0 and bold_chars / total_chars > 0.5
    return text, font_size, is_bold


def _column_of(bbox: tuple[float, float, float, float], page_width: float) -> str:
    x0, _, x1, _ = bbox
    mid, tol = page_width / 2, page_width * COLUMN_TOLERANCE
    if x1 <= mid + tol:
        return "left"
    if x0 >= mid - tol:
        return "right"
    return "full"


def _reading_order(blocks: list[dict]) -> list[dict]:
    """Order blocks for one page, handling two-column layouts.

    Full-width blocks (titles, wide figures/captions, single-column text) split
    the page into horizontal bands. Inside each band the left column is read
    top-to-bottom before the right column.
    """
    ordered, band = [], []

    def flush_band() -> None:
        for column in ("left", "right"):
            ordered.extend(sorted((b for b in band if b["column"] == column),
                                  key=lambda b: b["bbox"][1]))
        band.clear()

    for block in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        if block["column"] == "full":
            flush_band()
            ordered.append(block)
        else:
            band.append(block)
    flush_band()
    return ordered


def extract_blocks(path: Path, doc_id: str) -> list[dict]:
    blocks = []
    with pymupdf.open(path) as doc:
        for page in doc:
            width, height = page.rect.width, page.rect.height
            page_blocks = []
            for raw in page.get_text("rawdict", flags=EXTRACT_FLAGS)["blocks"]:
                if raw["type"] != 0:
                    continue
                text, font_size, is_bold = _block_text_and_font(raw)
                if not text.strip():
                    continue
                bbox = tuple(round(v, 1) for v in raw["bbox"])
                page_blocks.append({
                    "doc_id": doc_id,
                    "page": page.number + 1,  # 1-based, as printed in citations
                    "bbox": bbox,
                    "column": _column_of(bbox, width),
                    "page_width": round(width, 1),
                    "page_height": round(height, 1),
                    "font_size": font_size,
                    "is_bold": is_bold,
                    "text": text,
                })
            for block in _reading_order(page_blocks):
                block["block_no"] = len(blocks)  # position in the whole document
                blocks.append(block)
    return blocks


def write_preview(blocks: list[dict], out_path: Path) -> None:
    lines, current_page = [], None
    for b in blocks:
        if b["page"] != current_page:
            current_page = b["page"]
            lines.append(f"\n{'=' * 30} PAGE {current_page} {'=' * 30}")
        bold = " B" if b["is_bold"] else ""
        lines.append(f"--- #{b['block_no']} [{b['column']}] size={b['font_size']}{bold}")
        lines.append(b["text"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(RAW_DIR.glob("*.pdf")):
        doc_id = compute_doc_id(path)
        blocks = extract_blocks(path, doc_id)
        with (BLOCKS_DIR / f"{doc_id}.jsonl").open("w", encoding="utf-8") as f:
            for block in blocks:
                f.write(json.dumps(block, ensure_ascii=False) + "\n")
        write_preview(blocks, PREVIEW_DIR / f"{doc_id}.txt")

        ligatures = sum(ch in "ﬀﬁﬂﬃﬄﬅﬆ"
                        for b in blocks for ch in b["text"])
        columns = Counter(b["column"] for b in blocks)
        print(f"{doc_id}  blocks={len(blocks):<5} ligatures={ligatures:<3} "
              f"L/R/F={columns['left']}/{columns['right']}/{columns['full']}  {path.name[:45]}")


if __name__ == "__main__":
    main()
