"""Document parsing (PDF, DOCX, TXT/MD), normalisation and token-aware chunking."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import models
from .config import Settings
from .textutil import normalize_whitespace, split_sentences

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


class IngestError(Exception):
    """Raised for parser failures, kept distinct from empty results."""


@dataclass
class Block:
    text: str
    page: int | None = None
    section: str | None = None
    para: int = 0


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    version: int
    index: int
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    para_start: int
    para_end: int
    n_tokens: int
    overlap_tokens: int = 0

    def locator(self) -> str:
        parts = []
        if self.page_start is not None:
            parts.append(f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}")
        if self.section:
            parts.append(f"§ {self.section}")
        parts.append(f"¶ {self.para_start}" if self.para_start == self.para_end else f"¶ {self.para_start}-{self.para_end}")
        return " · ".join(parts)

    def embed_text(self) -> str:
        return f"{self.section}. {self.text}" if self.section else self.text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["locator"] = self.locator()
        return d


@dataclass
class Parsed:
    blocks: list[Block]
    method: str
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "doc"


# ---------------------------------------------------------------- parsers

def _looks_like_heading(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= 9 and not text.rstrip().endswith((".", ":", ";", ",")) and text[0].isupper()


def parse_pdf(data: bytes) -> Parsed:
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # noqa: BLE001
        raise IngestError(f"PDF could not be opened: {e}") from e
    if doc.needs_pass:
        raise IngestError("PDF is encrypted; supply an unlocked copy")
    blocks, warnings, section, para = [], [], None, 0
    for pno, page in enumerate(doc, start=1):
        page_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        texts = [normalize_whitespace(b[4]) for b in page_blocks]
        texts = [t for t in texts if t]
        if not texts:
            if page.get_images():
                warnings.append(f"page {pno} is image-only; no OCR stage is configured, so it was not indexed")
            else:
                warnings.append(f"page {pno} has no extractable text")
            continue
        for t in texts:
            if _looks_like_heading(t):
                section = t
                continue
            para += 1
            blocks.append(Block(t, page=pno, section=section, para=para))
    return Parsed(blocks, "pymupdf.get_text(blocks)", pages=len(doc), warnings=warnings)


def parse_docx(data: bytes) -> Parsed:
    import io

    import docx

    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise IngestError(f"DOCX could not be opened: {e}") from e
    blocks, section, para = [], None, 0
    body = d.element.body
    for el in body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = docx.text.paragraph.Paragraph(el, d)
            text = normalize_whitespace(p.text)
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            if style.lower().startswith(("heading", "title")):
                section = text
                continue
            para += 1
            blocks.append(Block(text, section=section, para=para))
        elif tag == "tbl":
            # Keep column associations: each row becomes "Header: value; Header: value".
            t = docx.table.Table(el, d)
            rows = [[normalize_whitespace(c.text) for c in r.cells] for r in t.rows]
            if not rows:
                continue
            header, body_rows = rows[0], rows[1:] or []
            for r in body_rows:
                para += 1
                cells = [f"{h}: {v}" for h, v in zip(header, r) if v]
                blocks.append(Block("; ".join(cells) + ".", section=section, para=para))
    return Parsed(blocks, "python-docx paragraphs+tables (section/paragraph locators)")


def parse_text(data: bytes) -> Parsed:
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:
        raw = data.decode("latin-1")
    blocks, section, para = [], None, 0
    for chunk in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        lines = [l for l in chunk.split("\n") if l.strip()]
        rest = []
        for line in lines:
            m = re.match(r"^\s*#{1,6}\s+(.*)", line)
            if m:
                if rest:
                    para += 1
                    blocks.append(Block(normalize_whitespace("\n".join(rest)), section=section, para=para))
                    rest = []
                section = m.group(1).strip()
            else:
                rest.append(re.sub(r"^\s*[-*]\s+", "", line))
        if rest:
            text = normalize_whitespace("\n".join(rest))
            if text:
                para += 1
                blocks.append(Block(text, section=section, para=para))
    return Parsed(blocks, "plain-text paragraphs (markdown headings as sections)")


def parse(filename: str, data: bytes) -> Parsed:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise IngestError(f"unsupported file type {ext or '(none)'}; use PDF, DOCX, TXT or MD")
    if not data:
        raise IngestError("file is empty")
    if ext == ".pdf":
        return parse_pdf(data)
    if ext == ".docx":
        return parse_docx(data)
    return parse_text(data)


# ---------------------------------------------------------------- chunking

def _ntok(text: str) -> int:
    return len(models.tokenizer().tokenize(text))


def _units(block: Block, limit: int) -> list[tuple[str, int]]:
    """Sentences of a block; any sentence longer than the limit is windowed by word pieces."""
    out = []
    for s in split_sentences(block.text):
        n = _ntok(s)
        if n <= limit:
            out.append((s, n))
            continue
        words = s.split()
        cur: list[str] = []
        for w in words:
            if cur and _ntok(" ".join(cur + [w])) > limit:
                piece = " ".join(cur)
                out.append((piece, _ntok(piece)))
                cur = []
            cur.append(w)
        if cur:
            piece = " ".join(cur)
            out.append((piece, _ntok(piece)))
    return out


def chunk_blocks(blocks: list[Block], doc_id: str, version: int, settings: Settings) -> list[Chunk]:
    limit = settings.chunk_tokens
    chunks: list[Chunk] = []
    cur: list[tuple[str, int, Block]] = []
    overlap_n = 0

    def flush(carry: bool):
        nonlocal cur, overlap_n
        if not cur:
            return
        text = " ".join(u[0] for u in cur)
        first, last = cur[0][2], cur[-1][2]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}@v{version}#c{len(chunks):03d}",
                doc_id=doc_id,
                version=version,
                index=len(chunks),
                text=text,
                section=first.section,
                page_start=first.page,
                page_end=last.page,
                para_start=first.para,
                para_end=last.para,
                n_tokens=sum(u[1] for u in cur),
                overlap_tokens=overlap_n,
            )
        )
        tail, tail_n = [], 0
        if carry and settings.overlap_tokens > 0:
            for u in reversed(cur):
                if tail_n + u[1] > settings.overlap_tokens:
                    break
                tail.insert(0, u)
                tail_n += u[1]
            if len(tail) == len(cur):  # never carry a whole chunk forward
                tail, tail_n = [], 0
        cur, overlap_n = tail, tail_n

    for block in blocks:
        if cur and block.section != cur[-1][2].section:
            flush(carry=False)  # never merge across a heading
        for text, n in _units(block, limit):
            if cur and sum(u[1] for u in cur) + n > limit:
                flush(carry=True)
            cur.append((text, n, block))
    flush(carry=False)

    max_len = models.max_seq_length()
    for c in chunks:
        if _ntok(c.embed_text()) + 2 > max_len:
            raise IngestError(f"chunk {c.chunk_id} would be truncated by the embedder ({max_len} word pieces)")
    return chunks
