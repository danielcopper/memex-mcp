"""Split a Markdown note into search chunks: by heading, then by paragraph."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Linear patterns only: a note is untrusted input, and chunking runs under
# the index lock. The closing hashes of a heading are stripped in Python.
_HEADING = re.compile(r"(#{1,6})[ \t]+(.*)")
_FENCE = re.compile(r"[ \t]*(```|~~~)")


def _heading(line: str) -> tuple[int, str] | None:
    """Level and text of an ATX heading line, or None."""
    match = _HEADING.match(line)
    if match is None:
        return None
    text = match.group(2).strip()
    unclosed = text.rstrip("#")
    # "## Part ##" closes with hashes after a space; "# C#" keeps its hash.
    if not unclosed or unclosed.endswith((" ", "\t")):
        text = unclosed.rstrip()
    return len(match.group(1)), text


@dataclass(frozen=True)
class Chunk:
    heading: str
    body: str


def title_of(text: str, fallback: str) -> str:
    """The first level-one heading outside code fences, else ``fallback``."""
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _heading(line)
        if heading is not None and heading[0] == 1 and heading[1]:
            return heading[1]
    return fallback


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = ""
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        found = None if in_fence else _heading(line)
        if found is not None:
            sections.append((heading, "\n".join(lines)))
            heading = found[1]
            lines = []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines)))
    return sections


def _pieces(body: str, max_chars: int) -> list[str]:
    """Greedily pack paragraphs into pieces of at most ``max_chars``."""
    pieces: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            cut = paragraph.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(paragraph[:cut].rstrip())
            paragraph = paragraph[cut:].lstrip()
        if current and len(current) + 2 + len(paragraph) > max_chars:
            pieces.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def chunk_note(text: str, max_chars: int) -> list[Chunk]:
    """Chunks of a note; a note with no text at all still yields its headings."""
    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        pieces = _pieces(body, max_chars)
        if not pieces and heading:
            pieces = [""]
        chunks.extend(Chunk(heading=heading, body=piece) for piece in pieces)
    return chunks
