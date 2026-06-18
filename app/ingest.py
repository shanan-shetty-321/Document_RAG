"""
PDF ingestion: parse -> clean -> chunk.

Turns the AWS Customer Agreement PDF into clean, self-contained text chunks,
each tagged with the contract section it came from. Those chunks are what we
embed and retrieve over.

Design decisions (defended in the README/report):

1. Cleaning. Extracted text has noise that hurts retrieval:
   - a header/footer repeated on every page
     ("6/16/26, 12:40 PM AWS Customer Agreement" + ".../agreement/ X/19")
   - typographic ligatures: "defined" -> "deﬁned", "affiliates" -> "aﬃliates"
     (fixed with Unicode NFKC normalization)
   - smart quotes/dashes and private-use-area font garbage from the broken
     tables (folded to ASCII / stripped).

2. Section-aware chunking. The agreement numbers every clause (1, 1.1, ... 12).
   We split on those headings so each chunk is one coherent legal clause, and
   store the section label + title as metadata -> citations read
   "Section 5.2 - Term; Termination" instead of "chunk #14".

3. Definitions (Section 12) are split per defined term ("X" means ...), so a
   "what does X mean?" question retrieves the exact definition.

4. Size cap + overlap. A few sections are long (e.g. 11.5). We cap chunk size
   and split oversized sections with a small overlap so a clause straddling a
   split still has context on both sides.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pypdf

from app import config

# Highest real top-level section number in the agreement. Any "heading" with a
# bigger number (e.g. "35 TheGardens" from an address table) is a false match.
MAX_SECTION_NUMBER = 12


# --- 1. Parse ----------------------------------------------------------------

def load_pages(pdf_path) -> list[str]:
    """Extract raw text from every page of the PDF (native text layer; no OCR)."""
    reader = pypdf.PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


# --- 2. Clean ----------------------------------------------------------------

# Header/footer repeated on every page, e.g. "6/16/26, 12:40 PM AWS Customer Agreement"
_FOOTER_DATE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*[AP]M\s*AWS Customer Agreement"
)
# Page URL/number, e.g. "https://aws.amazon.com/agreement/ 3/19"
_FOOTER_URL = re.compile(r"https?://aws\.amazon\.com/agreement/\s*\d+/\d+")

# Smart-punctuation -> ASCII. NFKC leaves these alone (they're "valid"), but
# folding them to ASCII keeps stored text and citations clean and portable.
_PUNCT_MAP = {
    0x201C: '"', 0x201D: '"',   # curly double quotes
    0x2018: "'", 0x2019: "'",   # curly single quotes / apostrophe
    0x2013: "-", 0x2014: "-",   # en / em dash
    0x2026: "...",              # ellipsis
}
# Private-use-area characters (U+E000-U+F8FF) are font-encoding garbage left
# over from the broken tables; they carry no meaning, so we drop them. Written
# as an escape range so the source file stays plain ASCII.
_PUA_RE = re.compile(r"[-]")


def clean_text(text: str) -> str:
    """Normalize unicode and strip the repeating page header/footer noise."""
    # NFKC turns ligatures (ﬁ, ﬀ, ﬂ ...) into ASCII equivalents (fi, ff, fl).
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_PUNCT_MAP)
    text = _PUA_RE.sub("", text)
    text = _FOOTER_DATE.sub("", text)
    text = _FOOTER_URL.sub("", text)
    # Tidy whitespace: drop trailing spaces and collapse big blank gaps.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- 3. Section-aware splitting ----------------------------------------------

# A section heading at the start of a line: a number like "1" or "1.1", an
# optional dot, whitespace, then an uppercase letter or "(" (start of the
# title). The lookahead detects the position without consuming the title.
_HEADING_RE = re.compile(r"(?m)^(\d{1,2}(?:\.\d{1,2})?)\.?\s+(?=[A-Z(])")


def _major(label: str) -> int:
    """Top-level number of a section label, e.g. '5.2' -> 5, 'Preamble' -> 0."""
    try:
        return int(label.split(".")[0])
    except ValueError:
        return 0


def split_into_sections(text: str) -> list[dict]:
    """Slice cleaned text into one segment per contract section.

    Returns {"section", "title", "text"} dicts in document order. Bare
    top-level heading lines (e.g. "1. AWS Responsibilities" with no body of
    their own) are not emitted; their title is carried onto the sub-sections
    beneath them. Text before section 1 is "Preamble".
    """
    # Keep only headings whose number is a real section (1-12).
    matches = [m for m in _HEADING_RE.finditer(text)
               if 1 <= _major(m.group(1)) <= MAX_SECTION_NUMBER]

    segments: list[dict] = []
    group_title = ""  # title of the current top-level section (e.g. "Fees and Payment")

    # Everything before the first heading is the preamble.
    if matches and matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            segments.append({"section": "Preamble", "title": "Preamble", "text": preamble})

    for i, match in enumerate(matches):
        label = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        first_line, _, rest = body.partition("\n")
        is_top_level = "." not in label
        # The heading title is the first line with the "N." / "N.N" prefix removed.
        title = re.sub(r"^\d{1,2}(?:\.\d{1,2})?\.?\s*", "", first_line).strip(" .")

        if is_top_level:
            group_title = title  # remember it for the sub-sections below
            # If the top-level heading has no real body of its own (just a
            # title, like "1. AWS Responsibilities"), don't emit it as a chunk.
            if len(rest.strip()) < 30:
                continue

        segments.append({
            "section": label,
            "title": group_title if not is_top_level else title,
            "text": body,
        })

    return segments


# A defined term at the start of a line inside Section 12, e.g.
#   "Your Content" means ...     /     "Account Country" is ...
#   "Governing Laws" and "Governing Courts" mean ...
# The term is capitalized and followed by a definition verb.
_DEFINITION_RE = re.compile(r'(?m)^"([A-Z][^"\n]{1,70})"\s+(?:means|mean|is|and)\b')


def split_definitions(text: str) -> list[dict] | None:
    """Split the Section 12 block into one segment per defined term.

    Returns {"title": <term>, "text": <definition>} dicts, or None if no
    definitions are found (so the caller can fall back to the size splitter).
    """
    matches = list(_DEFINITION_RE.finditer(text))
    if not matches:
        return None

    entries: list[dict] = []
    for i, match in enumerate(matches):
        term = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            entries.append({"title": term, "text": body})
    return entries


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split an oversized segment into pieces <= max_chars with overlap.

    Prefer cutting on a paragraph / sentence boundary near the limit so we
    don't slice a sentence in half; hard-cut only if there's no good boundary.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            pieces.append(text[start:].strip())
            break
        window = text[start:end]
        cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
        if cut < max_chars // 2:  # no decent boundary -> hard cut at the limit
            cut = len(window)
        pieces.append(text[start : start + cut].strip())
        start = start + cut - overlap  # step back to create the overlap
    return [p for p in pieces if p]


# --- 4. Orchestrate ----------------------------------------------------------

@dataclass
class Chunk:
    """One retrievable unit of the document."""
    id: int
    section: str   # contract section label, e.g. "5.2", "12", or "Preamble"
    title: str     # human-readable title, e.g. "Term; Termination" or "Your Content"
    text: str      # the chunk's cleaned text
    char_len: int  # length in characters (handy for analysis)


def build_chunks(pdf_path=config.PDF_PATH) -> list[Chunk]:
    """Full pipeline: PDF -> cleaned, section-aware, size-capped chunks."""
    pages = load_pages(pdf_path)
    full_text = "\n".join(clean_text(p) for p in pages)
    sections = split_into_sections(full_text)

    chunks: list[Chunk] = []
    next_id = 0

    def _emit(section: str, title: str, text: str) -> None:
        nonlocal next_id
        for piece in _split_long(text, config.MAX_CHUNK_CHARS, config.CHUNK_OVERLAP_CHARS):
            chunks.append(Chunk(id=next_id, section=section, title=title,
                                text=piece, char_len=len(piece)))
            next_id += 1

    for seg in sections:
        # Section 12 is a glossary -> split per defined term for precise lookup.
        if seg["section"] == "12":
            definitions = split_definitions(seg["text"])
            if definitions:
                for d in definitions:
                    _emit("12", d["title"], d["text"])
                continue
        _emit(seg["section"], seg["title"], seg["text"])

    return chunks


def to_documents(chunks: list[Chunk]):
    """Convert Chunks to LangChain Documents for embedding / BM25.

    Imported lazily so the parsing/chunking core stays usable without LangChain
    installed (e.g. for `python -m app.ingest` inspection).
    """
    from langchain_core.documents import Document
    return [
        Document(
            page_content=c.text,
            metadata={"section": c.section, "title": c.title, "chunk_id": c.id},
        )
        for c in chunks
    ]


# --- Inspection entry point --------------------------------------------------
# Run `python -m app.ingest` to eyeball the chunking before we embed anything.

if __name__ == "__main__":
    chunks = build_chunks()
    lengths = [c.char_len for c in chunks]
    sections = sorted({c.section for c in chunks}, key=lambda s: (_major(s), s))
    definition_chunks = [c for c in chunks if c.section == "12"]

    print(f"Total chunks      : {len(chunks)}")
    print(f"Char length min/avg/max : "
          f"{min(lengths)} / {sum(lengths)//len(lengths)} / {max(lengths)}")
    print(f"Distinct sections : {len(sections)}")
    print(f"Sections found    : {', '.join(sections)}")
    print(f"Definition chunks : {len(definition_chunks)} "
          f"(e.g. {[c.title for c in definition_chunks[:5]]})")
    print("\n--- Sample chunks ---")
    for c in chunks[:3]:
        print(f"\n[id={c.id}  section={c.section} ({c.title})  chars={c.char_len}]")
        print(c.text[:300] + ("..." if c.char_len > 300 else ""))
    # Show one definition chunk to confirm per-term splitting.
    for c in definition_chunks:
        if c.title == "Your Content":
            print(f"\n[id={c.id}  section=12 ({c.title})  chars={c.char_len}]")
            print(c.text[:300])
            break
