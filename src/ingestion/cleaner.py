"""Clean extracted blocks and structure each document into sections.

Input:  data/processed/blocks/{doc_id}.jsonl   (from pdf_loader)
Output: data/processed/clean/{doc_id}.json     sections -> paragraphs, with pages
        data/processed/clean_preview/{doc_id}.md   human-readable version
        data/processed/cleaning_report.csv     per-document counts

Steps, in order:
  1. Drop running headers/footers and page numbers (page margins + repetition).
  2. Normalize characters (odd spaces, control chars, ligature fallback).
  3. Fix line-end hyphenation using the corpus vocabulary, and soft hyphens.
  4. Detect section headings at line level (style, numbering, canonical names).
  5. Classify sections: front_matter / abstract / body / back_matter / references.
  6. Merge paragraphs split across columns or pages.

Usage:
    python -m src.ingestion.cleaner
"""

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.domains import get_domain
from src.ingestion.inventory import OUTPUT_CSV as INVENTORY_CSV
from src.ingestion.inventory import PROJECT_ROOT
from src.ingestion.pdf_loader import BLOCKS_DIR

CLEAN_DIR = PROJECT_ROOT / "data" / "processed" / "clean"
PREVIEW_DIR = PROJECT_ROOT / "data" / "processed" / "clean_preview"
REPORT_CSV = PROJECT_ROOT / "data" / "processed" / "cleaning_report.csv"

# --- Header / footer detection ---
MARGIN_RATIO = 0.09          # block starts/ends within the top/bottom 9% of the page
MIN_REPEAT_RATIO = 0.3       # ...and its text (digits normalized) repeats on >=30% of pages
PAGE_NUMBER_MARGIN_RATIO = 0.15
PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?\s*$", re.IGNORECASE)

# --- Character normalization ---
SPACE_LIKE = dict.fromkeys(map(ord, "\u00a0\u2002\u2003\u2009\u200a\u202f\t"), " ")
DROP_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\x08"), None)
LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
             "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
HYPHENS = "-\u2010"
LINE_HYPHEN_RE = re.compile(rf"(\w+)[{HYPHENS}]\n(\w+)")
WORD_RE = re.compile(rf"\w+(?:[{HYPHENS}]\w+)*")

# --- Heading detection ---
NUMBERING_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]{1,4}\.|[A-H]\.)\s+(?P<rest>\S.*)$")
NOT_HEADING_RE = re.compile(r"^(fig\.?|figure|table|algorithm|eq\.?|equation)\s*\d", re.IGNORECASE)
SENTENCE_INSIDE_RE = re.compile(r"[.:;]\s+\S")
MAX_HEADING_CHARS = 90
MAX_HEADING_LINES = 3
STYLE_SIZE_DELTA = 0.8       # pt above body size to count as a larger font

ABSTRACT_LABEL_RE = re.compile(r"^\s*(a\s?b\s?s\s?t\s?r\s?a\s?c\s?t)\s*[:.\u2014\u2013-]?\s*", re.IGNORECASE)
KEYWORDS_RE = re.compile(r"^\s*(keywords?|index terms)\b", re.IGNORECASE)
INTRO_NAMES = {"introduction", "background"}
REFERENCE_NAMES = re.compile(r"^(references|bibliography|literature cited|works cited)$")
BACK_MATTER_NAMES = re.compile(
    r"^(acknowledge?ments?|funding|author contributions?|credit authorship.*|declaration.*"
    r"|conflicts? of interest|competing interests?|data availability.*|code availability.*"
    r"|ethic.*|institutional review board.*|informed consent.*|supplementary.*|abbreviations"
    r"|additional information|publisher.?s note|author information|consent for publication"
    r"|disclosure.*|use of ai.*|generative ai.*)$")
APPENDIX_NAMES = re.compile(r"^appendi(x|ces)\b")

# --- Abstract / front matter ---
MIN_ABSTRACT_BLOCK_CHARS = 200
FRONT_MATTER_JUNK_RE = re.compile(
    r"\u00a9|copyright|doi\.org|https?://|\S+@\S+|creative ?commons"
    r"|^\s*(received|accepted|revised|published|citation|academic editor|edited by"
    r"|reviewed by|\*?correspond)", re.IGNORECASE | re.MULTILINE)
SMALL_FONT_RATIO = 0.85

TERMINAL_PUNCT = tuple(".!?:;)]\"\u201d")
# Body paragraphs with fewer letters than this share of their characters are
# table cells or equation debris ("# # # # #", "= + TP Precision TP FP").
MIN_LETTER_RATIO = 0.5


@dataclass
class Section:
    heading: str
    level: int
    kind: str  # front_matter | abstract | body | back_matter | references
    path: str = ""  # e.g. "2. Materials and Methods > 2.1. Study Area"
    paragraphs: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- loading ---

def load_blocks(doc_id: str) -> list[dict]:
    with (BLOCKS_DIR / f"{doc_id}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_inventory() -> dict[str, dict]:
    with INVENTORY_CSV.open(encoding="utf-8-sig") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f)}


# ---------------------------------------------------- 1. headers/footers ---

def _normalize_for_repeat(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text)).strip().lower()


def _touches_top(block: dict, ratio: float = MARGIN_RATIO) -> bool:
    return block["bbox"][1] < block["page_height"] * ratio


def _touches_bottom(block: dict, ratio: float = MARGIN_RATIO) -> bool:
    return block["bbox"][3] > block["page_height"] * (1 - ratio)


def _margin_line_candidates(block: dict) -> list[int]:
    """Indexes of lines that could be a running header/footer inside this block.

    Headers are sometimes merged into the first paragraph block of the page
    ("2 / GRSM-2023-00112 / Google Scholar, ..."), so we look at lines, not blocks.
    """
    n = len(block["lines"])
    idx = set()
    if _touches_top(block):
        idx.update(range(min(2, n)))
    if _touches_bottom(block):
        idx.update(range(max(0, n - 2), n))
    return sorted(idx)


def remove_headers_footers(blocks: list[dict]) -> tuple[list[dict], int]:
    """Strip running header/footer lines and page numbers. Returns (blocks, lines removed)."""
    n_pages = max(b["page"] for b in blocks)
    pages_per_key: dict[str, set[int]] = {}
    for b in blocks:
        if not (_touches_top(b) or _touches_bottom(b)):
            continue
        keys = {_normalize_for_repeat(b["text"])}
        keys |= {_normalize_for_repeat(b["lines"][i]["text"]) for i in _margin_line_candidates(b)}
        for key in keys:
            pages_per_key.setdefault(key, set()).add(b["page"])
    min_pages = max(2, MIN_REPEAT_RATIO * n_pages)
    repeated = {k for k, pages in pages_per_key.items() if k and len(pages) >= min_pages}

    kept, removed = [], 0
    for b in blocks:
        # A page number alone in its block may sit a bit further from the edge.
        if PAGE_NUMBER_RE.match(b["text"]) and (_touches_top(b, PAGE_NUMBER_MARGIN_RATIO)
                                                 or _touches_bottom(b, PAGE_NUMBER_MARGIN_RATIO)):
            removed += 1
            continue
        if (_touches_top(b) or _touches_bottom(b)) and _normalize_for_repeat(b["text"]) in repeated:
            removed += len(b["lines"])
            continue
        drop = {i for i in _margin_line_candidates(b)
                if _normalize_for_repeat(b["lines"][i]["text"]) in repeated
                or PAGE_NUMBER_RE.match(b["lines"][i]["text"])}
        if drop:
            removed += len(drop)
            b["lines"] = [line for i, line in enumerate(b["lines"]) if i not in drop]
            if not b["lines"]:
                continue
            b["text"] = "\n".join(line["text"] for line in b["lines"])
        kept.append(b)
    return kept, removed


# ------------------------------------------------ 2-3. text normalization ---

def normalize_chars(text: str) -> tuple[str, int]:
    """Return normalized text and the number of soft hyphens removed."""
    text = text.translate(SPACE_LIKE).translate(DROP_CHARS)
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    soft = text.count("\u00ad")
    text = text.replace("\u00ad\n", "").replace("\u00ad", "")
    text = re.sub(r"[\x00-\x09\x0b-\x1f]", "", text)
    return text, soft


def build_vocabulary(texts: list[str]) -> tuple[set[str], set[str]]:
    """Lower-cased words and hyphenated compounds seen *outside* line breaks."""
    words, compounds = set(), set()
    for text in texts:
        for token in WORD_RE.findall(LINE_HYPHEN_RE.sub(" ", text)):
            token = token.lower().replace("\u2010", "-")
            (compounds if "-" in token else words).add(token)
    return words, compounds


class Dehyphenator:
    def __init__(self, words: set[str], compounds: set[str]):
        self.words, self.compounds = words, compounds
        self.stats: Counter[str] = Counter()

    def _replace(self, match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if f"{left}-{right}".lower() in self.compounds:
            decision = "kept_compound"
        elif f"{left}{right}".lower() in self.words:
            decision = "joined_known"
        elif right[0].isupper() or right[0].isdigit():
            decision = "kept_guess"
        else:
            decision = "joined_guess"
        self.stats[decision] += 1
        return f"{left}-{right}" if decision.startswith("kept") else f"{left}{right}"

    def __call__(self, text: str) -> str:
        return LINE_HYPHEN_RE.sub(self._replace, text)


def flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# -------------------------------------------------- 4. heading detection ---

def body_font_size(blocks: list[dict]) -> float:
    sizes: Counter[float] = Counter()
    for b in blocks:
        for line in b["lines"]:
            sizes[line["font_size"]] += len(line["text"])
    return sizes.most_common(1)[0][0]


def split_numbering(text: str) -> tuple[str | None, str]:
    match = NUMBERING_RE.match(text)
    return (match.group("num"), match.group("rest")) if match else (None, text)


def canonical_name(text: str) -> str:
    """Heading text without numbering, lower-cased, for name matching."""
    _, rest = split_numbering(text.strip())
    return re.sub(r"\s+", " ", rest).strip(" .:").lower()


def merge_numbering_lines(lines: list[dict]) -> list[dict]:
    """Join a numbering-only line with the next one ('1' + 'Introduction').

    LaTeX papers leave a wide gap after the section number, so PyMuPDF splits
    the number and the title into separate lines.
    """
    merged: list[dict] = []
    for line in lines:
        prev = merged[-1] if merged else None
        if prev is not None and re.fullmatch(r"\d{1,2}(\.\d{1,2}){0,3}\.?|[IVX]{1,4}\.",
                                             prev["text"].strip()):
            merged[-1] = {**line, "text": f"{prev['text'].strip()} {line['text'].strip()}"}
        else:
            merged.append(line)
    return merged


def heading_signal(line: dict, body_size: float) -> str | None:
    """Why a line looks like a heading: 'numbered', 'canonical', 'styled' or None."""
    text = line["text"].strip()
    if not 2 <= len(text) <= MAX_HEADING_CHARS or NOT_HEADING_RE.match(text):
        return None
    number, rest = split_numbering(text)
    if text[-1] in ".,;" or SENTENCE_INSIDE_RE.search(rest) or not rest[0].isupper():
        return None
    letters = sum(c.isalpha() for c in rest)
    if (letters < 3 or sum(c.isdigit() for c in rest) > letters / 3
            or re.search(r"[%=\u00b1\\/]", rest)):
        return None
    name = canonical_name(text)
    if (name in INTRO_NAMES | {"results", "discussion", "conclusion", "conclusions", "methods",
                               "abstract"}
            or REFERENCE_NAMES.match(name) or BACK_MATTER_NAMES.match(name)
            or APPENDIX_NAMES.match(name)):
        return "canonical"
    if number is not None and line["font_size"] >= body_size - 0.5:
        return "numbered"
    if (line["is_bold"] or line["font_size"] >= body_size + STYLE_SIZE_DELTA
            or (rest.isupper() and letters >= 4)):
        return "styled"
    return None


@dataclass
class HeadingStyle:
    """Document-level heading conventions, used to reject inconsistent candidates."""
    body_size: float
    numbered: bool = False          # the document numbers its section headings
    trailing_dot: bool | None = None  # "1. Introduction" (True) vs "1 Introduction" (False)


def is_heading_line(line: dict, style: HeadingStyle) -> bool:
    signal = heading_signal(line, style.body_size)
    if signal is None:
        return False
    # Documents that number their headings: unnumbered bold/large lines are
    # almost always table headers or figure labels, not sections.
    if style.numbered and signal == "styled":
        return False
    # Numbered lists inside sections use a different numbering style.
    if signal == "numbered" and style.trailing_dot is not None:
        number = split_numbering(line["text"].strip())[0]
        if number[0].isdigit() and number.endswith(".") != style.trailing_dot:
            return False
    return True


def detect_heading_style(blocks: list[dict], body_size: float) -> HeadingStyle:
    numbers = {split_numbering(line["text"].strip())[0]
               for b in blocks if b["page"] > 1  # page 1 holds numbered affiliations
               for line in b["lines"][:MAX_HEADING_LINES]
               if heading_signal(line, body_size) == "numbered"}
    numbers = {n for n in numbers if n[0].isdigit()}
    style = HeadingStyle(body_size, numbered=len(numbers) >= 3)
    if style.numbered:
        dotted = sum(n.endswith(".") for n in numbers)
        style.trailing_dot = dotted > len(numbers) / 2
    return style


def heading_level(text: str, font_size: float, heading_sizes: list[float]) -> int:
    number, rest = split_numbering(text)
    if number:
        if number[0] in "IVX" and number.endswith("."):
            return 1
        if number[0] in "ABCDEFGH" and number.endswith("."):
            return 2
        return min(number.rstrip(".").count(".") + 1, 3)
    if rest.isupper():
        return 1
    # Unnumbered: rank among the distinct heading font sizes of the document.
    return min(heading_sizes.index(font_size) + 1, 3) if font_size in heading_sizes else 2


def _continues_heading(line: dict, first: dict) -> bool:
    """A wrapped heading's next line: same font, short, not a sentence end."""
    text = line["text"].strip()
    return (line["font_size"] == first["font_size"] and line["is_bold"] == first["is_bold"]
            and 0 < len(text) <= MAX_HEADING_CHARS and text[-1] not in ".,;")


def split_heading(block: dict, next_block: dict | None,
                  style: HeadingStyle) -> tuple[list[dict], list[dict]]:
    """Split a block into (heading lines, body lines). Heading lines may be empty."""
    lines = block["lines"]
    body_size = style.body_size
    k = 0
    while k < min(len(lines), MAX_HEADING_LINES) and (
            is_heading_line(lines[k], style)
            or (k > 0 and _continues_heading(lines[k], lines[0])
                and heading_signal(lines[0], body_size) == "styled")):
        k += 1
    if k == 0:
        return [], lines
    if k < len(lines):
        # Heading followed by text in the same block: the text must start like prose.
        if len(lines[k]["text"]) >= 40:
            return lines[:k], lines[k:]
        return [], lines
    if len(lines) > MAX_HEADING_LINES:
        return [], lines
    # Numbered and canonical headings are strong signals on their own.
    if heading_signal(lines[0], body_size) in ("numbered", "canonical"):
        return lines, []
    # Style-only headings: reject table-header-like blocks by requiring that
    # what follows is prose or another heading.
    if next_block is None:
        return [], lines
    following = next_block["lines"][0]
    if len(following["text"]) >= 40 or is_heading_line(following, style):
        return lines, []
    return [], lines


def group_headings(head_lines: list[dict]) -> list[list[dict]]:
    """Split stacked headings ('2. Methods' + '2.1. Study Area') into separate groups."""
    groups: list[list[dict]] = []
    for line in head_lines:
        if not groups or split_numbering(line["text"].strip())[0] is not None:
            groups.append([line])
        else:
            groups[-1].append(line)
    return groups


# --------------------------------------------- 5-6. sections & paragraphs ---

def section_kind(name: str, current_kind: str) -> str:
    if REFERENCE_NAMES.match(name):
        return "references"
    if BACK_MATTER_NAMES.match(name):
        return "back_matter"
    if APPENDIX_NAMES.match(name):
        return "body"
    return "references" if current_kind == "references" else "body"


def keep_as_abstract(text: str, block: dict, body_size: float) -> bool:
    return (len(text) >= MIN_ABSTRACT_BLOCK_CHARS
            and SMALL_FONT_RATIO * body_size <= block["font_size"] <= 1.3 * body_size
            and not FRONT_MATTER_JUNK_RE.search(text)
            and not KEYWORDS_RE.match(text))


def append_paragraph(section: Section, text: str, page: int, dehyphenate: Dehyphenator) -> None:
    """Add text as a new paragraph, or continue the previous one if it was cut mid-sentence."""
    if not text:
        return
    if section.paragraphs:
        prev = section.paragraphs[-1]
        if not prev["text"].endswith(TERMINAL_PUNCT) and text[0].islower():
            if prev["text"][-1] in HYPHENS:
                merged = dehyphenate(prev["text"] + "\n" + text)
            else:
                merged = prev["text"] + " " + text
            prev.update(text=merged, page_end=page)
            return
    section.paragraphs.append({"text": text, "page_start": page, "page_end": page})


def structure_document(blocks: list[dict], dehyphenate: Dehyphenator) -> tuple[list[Section], Counter]:
    body_size = body_font_size(blocks)
    style = detect_heading_style(blocks, body_size)
    stats: Counter[str] = Counter()

    # Pre-compute heading splits so we know where the body starts.
    splits = [split_heading(b, blocks[i + 1] if i + 1 < len(blocks) else None, style)
              for i, b in enumerate(blocks)]
    def starts_body(i: int, head_lines: list[dict]) -> bool:
        name = canonical_name(head_lines[0]["text"])
        number = (split_numbering(head_lines[0]["text"].strip())[0] or "").rstrip(".")
        return (name in INTRO_NAMES or number in {"1", "I"}
                or (blocks[i]["page"] > 1 and name != "abstract"))

    body_start = next((i for i, (h, _) in enumerate(splits) if h and starts_body(i, h)), 0)
    # Font sizes of body headings only (front matter has titles in large fonts).
    heading_sizes = sorted({h[0]["font_size"] for h, _ in splits[body_start:] if h}, reverse=True)

    front = Section("Front matter", 0, "front_matter")
    abstract = Section("Abstract", 1, "abstract")
    sections: list[Section] = [front, abstract]
    current = abstract
    open_headings: dict[int, str] = {}  # by level, so empty parent headings survive in paths
    abstract_label_found = False

    for i, (block, (head_lines, body_lines)) in enumerate(zip(blocks, splits)):
        page = block["page"]
        small_page1 = page == 1 and block["font_size"] < SMALL_FONT_RATIO * body_size

        if i < body_start or small_page1:
            text = flatten(dehyphenate(block["text"]))
            if ABSTRACT_LABEL_RE.match(text):
                abstract_label_found = True
                text = ABSTRACT_LABEL_RE.sub("", text, count=1)
            if not small_page1 and keep_as_abstract(text, block, body_size):
                append_paragraph(abstract, text, page, dehyphenate)
            else:
                append_paragraph(front, text, page, dehyphenate)
            continue

        # Inside the reference list only back-matter/appendix headings can end it;
        # numbered entries would otherwise look like headings.
        if head_lines and current.kind == "references" and                 section_kind(canonical_name(head_lines[0]["text"]), "") == "body":
            head_lines, body_lines = [], block["lines"]

        for group in group_headings(head_lines):
            heading = flatten(" ".join(line["text"] for line in group))
            name = canonical_name(heading)
            if name == "abstract":
                abstract_label_found = True
                current = abstract
                continue
            level = heading_level(heading, group[0]["font_size"], heading_sizes)
            open_headings = {lvl: h for lvl, h in open_headings.items() if lvl < level}
            open_headings[level] = heading
            path = " > ".join(open_headings[lvl] for lvl in sorted(open_headings))
            current = Section(heading, level, section_kind(name, current.kind), path=path)
            sections.append(current)
            stats["headings"] += 1

        if body_lines:
            text = flatten(dehyphenate("\n".join(line["text"] for line in body_lines)))
            non_space = len(text.replace(" ", ""))
            if (current.kind in ("abstract", "body")
                    and sum(c.isalpha() for c in text) < MIN_LETTER_RATIO * non_space):
                stats["noise_paragraphs"] += 1
                continue
            append_paragraph(current, text, page, dehyphenate)

    if not abstract_label_found:
        abstract.heading = "Abstract / Introduction"
    for s in (front, abstract):
        s.path = s.heading
    return [s for s in sections if s.paragraphs], stats


# ------------------------------------------------------------- outputs ---

def write_preview(doc: dict, path: Path) -> None:
    out = [f"# {doc['title']}", f"`{doc['doc_id']}` · {doc['domain']} · {doc['authors']}", ""]
    for s in doc["sections"]:
        marker = "#" * (s["level"] + 1) if s["level"] else "##"
        out.append(f"{marker} {s['heading']}  _[{s['kind']}]_")
        for p in s["paragraphs"]:
            pages = (f"p{p['page_start']}" if p["page_start"] == p["page_end"]
                     else f"p{p['page_start']}-{p['page_end']}")
            out.append(f"<sub>{pages}</sub> {p['text']}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory()

    # Normalize every document first: the hyphenation vocabulary is corpus-wide.
    docs: dict[str, dict] = {}
    for doc_id in inventory:
        blocks, removed = remove_headers_footers(load_blocks(doc_id))
        soft = 0
        for b in blocks:
            b["text"], n = normalize_chars(b["text"])
            soft += n
            for line in b["lines"]:
                line["text"], _ = normalize_chars(line["text"])
            b["lines"] = merge_numbering_lines(b["lines"])
        docs[doc_id] = {"blocks": blocks, "removed": removed, "soft_hyphens": soft}

    words, compounds = build_vocabulary([b["text"] for d in docs.values() for b in d["blocks"]])

    report = []
    for doc_id, d in docs.items():
        dehyphenate = Dehyphenator(words, compounds)
        sections, stats = structure_document(d["blocks"], dehyphenate)
        meta = inventory[doc_id]
        doc = {
            "doc_id": doc_id,
            "title": meta["title"],
            "authors": meta["meta_author"],
            "domain": get_domain(doc_id),
            "sections": [s.__dict__ for s in sections],
        }
        (CLEAN_DIR / f"{doc_id}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        write_preview(doc, PREVIEW_DIR / f"{doc_id}.md")

        chars = Counter()
        for s in sections:
            chars[s.kind] += sum(len(p["text"]) for p in s.paragraphs)
        report.append({
            "doc_id": doc_id,
            "headers_footers_removed": d["removed"],
            "noise_paragraphs_dropped": stats["noise_paragraphs"],
            "soft_hyphens": d["soft_hyphens"],
            **{k: dehyphenate.stats[k] for k in
               ("joined_known", "joined_guess", "kept_compound", "kept_guess")},
            "headings": stats["headings"],
            **{f"chars_{k}": chars[k] for k in
               ("front_matter", "abstract", "body", "back_matter", "references")},
        })

    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    cols = ["doc_id", "headers_footers_removed", "noise_paragraphs_dropped", "joined_known", "kept_compound",
            "headings", "chars_abstract", "chars_body", "chars_references"]
    print("  ".join(c.replace("headers_footers_removed", "hdr/ftr").replace("noise_paragraphs_dropped", "noise")[:13] for c in cols))
    for r in report:
        print("  ".join(str(r[c]).ljust(len(c.replace("headers_footers_removed", "hdr/ftr").replace("noise_paragraphs_dropped", "noise")[:13]))
                        for c in cols))


if __name__ == "__main__":
    main()
