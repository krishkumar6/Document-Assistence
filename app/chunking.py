from dataclasses import dataclass, field


@dataclass
class PageSpan:
    page: int
    start: int  # char offset within the chunk text
    end: int


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    spans: list[PageSpan] = field(default_factory=list)

    @property
    def page_start(self) -> int:
        return self.spans[0].page

    @property
    def page_end(self) -> int:
        return self.spans[-1].page


def chunk_pages(pages: list[tuple[int, str]], source: str, size: int, overlap: int) -> list[Chunk]:
    """Split a document into ~`size`-char chunks with `overlap` chars of overlap.

    Chunks may cross page boundaries (so a sentence split across pages stays
    together), but each chunk records which character ranges came from which
    page. That lets the answer step cite the exact page, not just a range.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    # Concatenate pages, remembering where each one lives in the full text.
    parts: list[str] = []
    page_offsets: list[tuple[int, int, int]] = []
    pos = 0
    for page_no, text in pages:
        if parts:
            parts.append(" ")
            pos += 1
        page_offsets.append((page_no, pos, pos + len(text)))
        parts.append(text)
        pos += len(text)
    full = "".join(parts)

    chunks: list[Chunk] = []
    start = 0
    while start < len(full):
        end = min(start + size, len(full))
        if end < len(full):
            end = _break_point(full, start, end)

        spans = []
        for page_no, p_start, p_end in page_offsets:
            s, e = max(p_start, start), min(p_end, end)
            if s < e and full[s:e].strip():
                spans.append(PageSpan(page_no, s - start, e - start))
        if spans:
            chunks.append(Chunk(full[start:end], source, len(chunks), spans))

        if end >= len(full):
            break
        start = _snap_to_word(full, max(end - overlap, start + 1), end)
    return chunks


def _break_point(text: str, start: int, end: int) -> int:
    """Prefer ending on a sentence, then on whitespace, within the back half of the window."""
    floor = start + (end - start) // 2
    sentence = text.rfind(". ", floor, end)
    if sentence != -1:
        return sentence + 1
    space = text.rfind(" ", floor, end)
    return space if space != -1 else end


def _snap_to_word(text: str, pos: int, limit: int) -> int:
    """Move an overlap start forward so chunks don't begin mid-word."""
    if pos > 0 and not text[pos - 1].isspace():
        nxt = text.find(" ", pos, limit)
        pos = nxt if nxt != -1 else pos
    while pos < limit and text[pos].isspace():
        pos += 1
    return pos
