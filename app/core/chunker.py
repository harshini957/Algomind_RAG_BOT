import re
import uuid
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class ParentChunk:
    id: str
    text: str
    chapter: str
    chapter_num: int
    page_start: int
    page_end: int
    source: str


@dataclass
class ChildChunk:
    id: str
    text: str
    parent_id: str
    chapter: str
    chapter_num: int
    section: str
    section_num: str
    page: int
    chunk_index: int
    source: str


class DoclingChunker:
    """
    Parent-child chunker built specifically for The Algorithm Design Manual.

    Derived from direct inspection of the PDF's raw text output.

    What the PDF actually looks like (verified by reading raw bytes):
    ----------------------------------------------------------------
    CHAPTER START — a page whose text begins with:
        "<digit(s)>\\n<Title Case title that may span 1 or 2 lines>"

        Single-line example:  "2\\nAlgorithm Analysis\\n"
        Two-line example:     "9\\nIntractable Problems and Approximation\\nAlgorithms\\n"

    RUNNING PAGE HEADER (MUST BE IGNORED):
        "<section_num> <SECTION TITLE IN ALL CAPS> <page_num>"
        e.g.  "1.1 ROBOT TOUR OPTIMIZATION 5"
              "3.1 CONTIGUOUS VS. LINKED DATA STRUCTURES 67"

    SECTION HEADING IN BODY (Title Case, what we want):
        "<n.n> <Title Case title>\\n"          e.g. "1.1 Robot Tour Optimization\\n"
        "<n.n.n> <Title Case title>\\n"        e.g. "1.3.1 Expressing Algorithms\\n"

    Parent  = one full chapter  (level: digit only)
    Child   = one section or sub-section (level: n.n or n.n.n)
              windowed into ≤child_max_tokens pieces if the section is large.

    No titles are hardcoded. Chapter detection uses structure:
      - standalone digit line followed by a Title Case line
      - title must NOT be all-caps (rules out running headers)
      - title must contain at least one lowercase letter
    """

    # ------------------------------------------------------------------ #
    # Chapter: digit(s) alone on a line, next non-empty line is Title Case
    # We capture: group(1)=number, group(2)=first title line,
    #             group(3)=optional second title line (for wrapped titles)
    # ------------------------------------------------------------------ #
    CHAPTER_RE = re.compile(
        r'(?m)'
        r'^(\d{1,2})\n'                          # chapter number alone
        r'([A-Z][A-Za-z ,:\-\']{3,60})\n'        # first title line (Title Case)
        r'([A-Za-z][A-Za-z ,:\-\']{3,50}\n)?'    # optional second title line
    )

    # ------------------------------------------------------------------ #
    # Section / Sub-section: "1.1 Robot Tour..." or "1.3.1 Expressing..."
    # Title Case — has at least one lowercase letter (excludes ALL CAPS headers)
    # ------------------------------------------------------------------ #
    SECTION_RE = re.compile(
        r'(?m)'
        r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)'    # n.n or n.n.n
        r'\s+'
        r'([A-Z][A-Za-z ,:\-\'\/\(\)\*\!]{2,70})'  # Title Case title
        r'\s*$'
    )

    # Running page header — "1.1 ROBOT TOUR OPTIMIZATION 5"
    # All-caps words after the section number → discard the whole line
    RUNNING_HEADER_RE = re.compile(
        r'(?m)^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+[A-Z][A-Z\s\-\':\/]{5,}\s+\d+\s*$'
    )

    # TOC dot-leader lines — "1.1 Robot Tour ... 5"
    TOC_LINE_RE = re.compile(r'(?m)^.*\.{3,}.*\d+\s*$')

    # Front matter ends at the first real chapter page
    FRONT_MATTER_END_PAGE = 14  # 0-indexed → page 15 in 1-indexed

    def __init__(
        self,
        child_max_tokens: int = 200,
        child_overlap: int = 20,
        source: str = "unknown",
    ):
        self.child_max_tokens = child_max_tokens
        self.child_overlap = child_overlap
        self.source = source

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _is_valid_chapter_title(self, title: str) -> bool:
        """
        Reject false positives from the chapter regex.
        A valid chapter title:
          - contains at least one lowercase letter  (not ALL CAPS)
          - is not a pure number sequence
          - has at least 4 characters
        """
        if len(title.strip()) < 4:
            return False
        if not any(c.islower() for c in title):
            return False
        return True

    def _clean(self, text: str) -> str:
        """Remove page markers, running headers, and TOC lines."""
        text = re.sub(r'<<<PAGE:\d+>>>', '', text)
        text = self.RUNNING_HEADER_RE.sub('', text)
        text = self.TOC_LINE_RE.sub('', text)
        # remove figure/table caption noise like "Figure 1.1: ..."
        # keep it — it is content
        # collapse 3+ blank lines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _window_slice(
        self,
        text: str,
        parent_id: str,
        chapter: str,
        chapter_num: int,
        section: str,
        section_num: str,
        page: int,
    ) -> list[ChildChunk]:
        """Slide a fixed window over section text → ChildChunks."""
        words = text.split()
        step = max(1, self.child_max_tokens - self.child_overlap)
        children = []

        for idx, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start: start + self.child_max_tokens]
            if len(chunk_words) < 15:
                continue
            children.append(ChildChunk(
                id=str(uuid.uuid4()),
                text=" ".join(chunk_words),
                parent_id=parent_id,
                chapter=chapter,
                chapter_num=chapter_num,
                section=section,
                section_num=section_num,
                page=page,
                chunk_index=idx,
                source=self.source,
            ))

        return children

    def _extract_full_text(self, pdf_path: str) -> tuple[str, list]:
        """
        Extract text from all content pages (skip front matter).
        Returns (full_text_with_markers, page_marker_list).
        """
        reader = PdfReader(pdf_path)
        full_text = ""
        for i, page in enumerate(reader.pages):
            if i < self.FRONT_MATTER_END_PAGE:
                continue
            text = page.extract_text() or ""
            full_text += f"\n<<<PAGE:{i+1}>>>\n{text}"

        page_markers = list(re.finditer(r'<<<PAGE:(\d+)>>>', full_text))
        return full_text, page_markers

    def _page_at(self, pos: int, page_markers: list) -> int:
        """Return page number for a character position in full_text."""
        current = self.FRONT_MATTER_END_PAGE + 1
        for m in page_markers:
            if m.start() > pos:
                break
            current = int(m.group(1))
        return current

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def chunk(
        self, pdf_path: str
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Parse the PDF and return (parent_chunks, child_chunks).

        Algorithm
        ---------
        1. Extract text page-by-page, embed <<<PAGE:N>>> markers.
        2. Find all chapter boundaries using CHAPTER_RE.
        3. For each chapter block:
           a. Build a ParentChunk from the full cleaned chapter text.
           b. Find all section / sub-section headings with SECTION_RE.
           c. For each section, window-slice its text into ChildChunks,
              each carrying parent_id back to the chapter.
        """
        print(f"[Chunker] Reading: {pdf_path}")
        full_text, page_markers = self._extract_full_text(pdf_path)
        print(f"[Chunker] Content pages extracted "
              f"(from page {self.FRONT_MATTER_END_PAGE + 1})")

        # -------------------------------------------------------------- #
        # Step 1 — locate chapters
        # -------------------------------------------------------------- #
        raw_chapter_matches = list(self.CHAPTER_RE.finditer(full_text))

        # filter false positives (figure labels, catalog entries, etc.)
        chapter_matches = []
        for m in raw_chapter_matches:
            title_line1 = m.group(2).strip()
            title_line2 = (m.group(3) or "").strip()
            full_title = (title_line1 + " " + title_line2).strip()

            if not self._is_valid_chapter_title(full_title):
                continue

            chapter_matches.append((m, full_title))

        print(f"[Chunker] Chapters detected: {len(chapter_matches)}")
        for m, title in chapter_matches:
            print(f"           Ch {m.group(1).strip()}: {title}")

        if not chapter_matches:
            print("[Chunker] ERROR: No chapters found.")
            return [], []

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        # -------------------------------------------------------------- #
        # Step 2 — build parent + children for each chapter
        # -------------------------------------------------------------- #
        for ci, (ch_match, ch_title) in enumerate(chapter_matches):
            ch_num_str = ch_match.group(1).strip()
            ch_num_int = int(ch_num_str)
            ch_name    = f"{ch_num_str} {ch_title}"
            ch_start   = ch_match.start()
            ch_end     = (
                chapter_matches[ci + 1][0].start()
                if ci + 1 < len(chapter_matches)
                else len(full_text)
            )
            ch_raw   = full_text[ch_start:ch_end]
            ch_clean = self._clean(ch_raw)
            ch_page  = self._page_at(ch_start, page_markers)

            parent = ParentChunk(
                id=str(uuid.uuid4()),
                text=ch_clean,
                chapter=ch_name,
                chapter_num=ch_num_int,
                page_start=ch_page,
                page_end=self._page_at(ch_end - 1, page_markers),
                source=self.source,
            )
            parents.append(parent)

            # ---------------------------------------------------------- #
            # Step 3 — find sections inside this chapter
            # ---------------------------------------------------------- #
            # work on raw (with page markers) so positions stay aligned
            section_matches = list(self.SECTION_RE.finditer(ch_raw))

            # filter: section number must start with this chapter number
            section_matches = [
                sm for sm in section_matches
                if sm.group(1).split('.')[0] == ch_num_str
            ]

            # deduplicate by section number — keep first occurrence only
            # (TOC remnants may cause the same number to appear twice)
            seen: set[str] = set()
            unique_sections = []
            for sm in section_matches:
                key = sm.group(1).strip()
                if key not in seen:
                    seen.add(key)
                    unique_sections.append(sm)
            section_matches = unique_sections

            if not section_matches:
                # no sub-sections found — whole chapter is one set of children
                children.extend(self._window_slice(
                    text=ch_clean,
                    parent_id=parent.id,
                    chapter=ch_name,
                    chapter_num=ch_num_int,
                    section=ch_name,
                    section_num=ch_num_str,
                    page=ch_page,
                ))
                continue

            for si, sec_m in enumerate(section_matches):
                sec_num   = sec_m.group(1).strip()
                sec_title = sec_m.group(2).strip()
                sec_name  = f"{sec_num} {sec_title}"
                sec_start = sec_m.start()
                sec_end   = (
                    section_matches[si + 1].start()
                    if si + 1 < len(section_matches)
                    else len(ch_raw)
                )
                sec_raw   = ch_raw[sec_start:sec_end]
                sec_clean = self._clean(sec_raw)
                sec_page  = self._page_at(ch_start + sec_start, page_markers)

                if not sec_clean:
                    continue

                children.extend(self._window_slice(
                    text=sec_clean,
                    parent_id=parent.id,
                    chapter=ch_name,
                    chapter_num=ch_num_int,
                    section=sec_name,
                    section_num=sec_num,
                    page=sec_page,
                ))

        print(f"\n[Chunker] ✓ {len(parents)} parent chunks (chapters)")
        print(f"[Chunker] ✓ {len(children)} child chunks (sections)")
        return parents, children