"""What a write does to the text of a note and of its index, without touching the clone.

The service reads the files, asks these functions for the new text, and
writes and commits the result. Everything here works on strings and path
segments only, so each rule is checked on its own.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from memex_mcp.markdown import headings
from memex_mcp.rights import (
    HIDDEN_CATEGORIES,
    HTML_COMMENT,
    IMPORT_AT,
    NOTE_SUFFIX,
    InvalidRequest,
)

type Folder = Literal["memory", "wiki"]
type Action = Literal["create", "update", "append", "delete"]

# Notes under any folder of this name are daily notes: they take no index line.
DAILY_FOLDER = "daily"
INDEX_FILE: dict[Folder, str] = {"memory": "MEMORY.md", "wiki": "_index.md"}
MASTER_INDEX = "_master-index.md"
MAX_INDEX_LINE_CHARS = 300
MAX_SECTION_CHARS = 100
# The level of the headings that open an index's sections (``## Section``).
SECTION_LEVEL = 2
# A topic folder's path: area, memory or wiki, topic.
_TOPIC_DEPTH = 3
# The git trailer that names the OAuth client of a write.
CLIENT_TRAILER = "Memex-Client"

CHANGED_SINCE_READ = "the note changed since you read it; read it again"
MISSING_TOPIC = (
    "the first note of a new wiki topic needs topic_description, one line that says "
    "what the topic holds, for the master index"
)

# Unambiguous machine secrets only, never passwords in prose. Each pattern is
# linear: the text is untrusted and checked while the clone is locked.
_SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a private key", re.compile(r"-----BEGIN [A-Z0-9 ]{0,40}PRIVATE KEY(?: BLOCK)?-----")),
    (
        "a GitHub token",
        re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})"),
    ),
    ("an API key (sk-)", re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{32,}")),
    ("an AWS access key", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![0-9A-Z])")),
    ("a Slack token", re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}")),
    ("a GitLab token", re.compile(r"(?<![A-Za-z0-9_])glpat-[A-Za-z0-9_-]{20,}")),
)


def version_of(data: bytes) -> str:
    """The note's version: git's blob id of its content, the same id ``git hash-object`` gives."""
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def check_secrets(texts: Iterable[str]) -> None:
    """Refuse text that holds a machine secret, naming its kind but never its value."""
    for text in texts:
        for kind, pattern in _SECRETS:
            if pattern.search(text):
                raise InvalidRequest(
                    f"the text contains what looks like {kind}; secrets do not belong in notes: "
                    + "leave it out, or write where it is kept instead"
                )


def check_size(data: bytes, limit: int) -> None:
    if len(data) > limit:
        raise InvalidRequest("the note would be too large; create a new note instead")


def check_version(current: bytes, version: str) -> None:
    if version_of(current) != version:
        raise InvalidRequest(CHANGED_SINCE_READ)


def decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidRequest("the note is not UTF-8 text and cannot be changed here") from None


def _ending(text: str) -> str:
    """``text`` with a line break at its end."""
    return text if text.endswith("\n") else text + "\n"


def new_note(content: str) -> str:
    if not content.strip():
        raise InvalidRequest("the content must not be empty")
    return _ending(content)


def appended(current: str, text: str) -> str:
    """``text`` after the note's last line, on a line of its own."""
    if not text.strip():
        raise InvalidRequest("the text to append must not be empty")
    head = _ending(current) if current else ""
    return head + _ending(text)


def edited(current: str, old_text: str, new_text: str) -> str:
    """The note with its one occurrence of ``old_text`` replaced by ``new_text``."""
    if not old_text:
        raise InvalidRequest("old_text must not be empty")
    first = current.find(old_text)
    if first < 0:
        raise InvalidRequest(
            "old_text does not occur in the note; read the note again and copy the text exactly"
        )
    # Counted from the next character, so overlapping occurrences count too.
    count, position = 1, current.find(old_text, first + 1)
    while position >= 0:
        count += 1
        position = current.find(old_text, position + 1)
    if count > 1:
        raise InvalidRequest(
            f"old_text occurs {count} times in the note; "
            + "include more of the surrounding text so that it occurs exactly once"
        )
    result = current[:first] + new_text + current[first + len(old_text) :]
    if result == current:
        raise InvalidRequest("the edit changes nothing")
    return result


# -- indexes -----------------------------------------------------------


def is_daily(parts: tuple[str, ...]) -> bool:
    """Whether a note lies in a ``daily/`` folder; such notes take no index line."""
    return DAILY_FOLDER in parts[2:-1]


@dataclass(frozen=True)
class IndexSpot:
    """The index a new note goes into: the one of its topic, by kind."""

    folder: Folder
    # The topic folder: <area>/<folder>/<topic>, or <area>/memory itself.
    topic: tuple[str, ...]

    @property
    def name(self) -> str:
        return INDEX_FILE[self.folder]


def index_spot(parts: tuple[str, ...]) -> IndexSpot:
    """The topic whose index a new note at ``parts`` is listed in.

    The topic is the folder directly under ``<area>/memory/`` or
    ``<area>/wiki/``: a note in a subfolder of a topic is listed in the
    topic's index, whatever index files the subfolders hold. A note directly
    in ``<area>/memory/`` is listed in ``<area>/memory/MEMORY.md``; one
    directly in ``<area>/wiki/`` is refused, since the master index there
    lists topics, not notes. ``parts`` passed ``Policy.may_write``.
    """
    folder: Folder = "memory" if parts[1] == "memory" else "wiki"
    note_folder = parts[:-1]
    if folder == "wiki" and len(note_folder) < _TOPIC_DEPTH:
        raise InvalidRequest(
            f"wiki notes go into a topic folder: {parts[0]}/wiki/<topic>/{parts[-1]}"
        )
    return IndexSpot(folder=folder, topic=note_folder[:_TOPIC_DEPTH])


def _lines(text: str) -> list[str]:
    """The lines of an index, split at line feeds only.

    Other characters Python takes for line breaks (form feed, U+0085,
    U+2028) stay inside their line, so adding an entry changes no other byte.
    """
    lines = text.split("\n")
    if lines and not lines[-1]:
        lines.pop()
    return [line.removesuffix("\r") for line in lines]


def section_names(text: str) -> list[str]:
    """The sections of an index: its level-two headings, in order."""
    return [name for _, level, name in headings(_lines(text)) if level == SECTION_LEVEL]


def _sections_hint(sections: Sequence[str]) -> str:
    if not sections:
        return "the index has no sections yet; name a new one"
    return "its sections: " + "; ".join(sections)


def _text_problem(text: str, limit: int) -> str | None:
    """Why ``text`` cannot be one line of an index, or None.

    Control characters and line breaks would split it; format characters
    (bidi overrides) and ``<!--`` can hide what follows in a rendered view;
    an ``@`` starting a word is what Claude Code reads as a file to load
    into the session, from a ``MEMORY.md`` it loads.
    """
    if len(text) > limit or _has_hidden(text):
        return (
            f"must be one line of at most {limit} characters, without control or format characters"
        )
    return _markup_problem(text)


def _has_hidden(text: str) -> bool:
    return any(unicodedata.category(c) in HIDDEN_CATEGORIES for c in text)


def _markup_problem(text: str) -> str | None:
    if HTML_COMMENT in text:
        return f"must not contain '{HTML_COMMENT}'"
    if IMPORT_AT.search(text):
        return "must not have '@' at the start of a word, which assistants read as a file to load"
    return None


def check_composed(lines: Iterable[str]) -> None:
    """Refuse index lines and headings, as composed from the path and the texts, that
    break the index's rules: a second guard behind the checks of each part."""
    for line in lines:
        problem = "must not hold control or format characters" if _has_hidden(line) else None
        problem = problem or _markup_problem(line)
        if problem is not None:
            raise InvalidRequest(
                f"the note cannot be listed in its index: a line or heading it needs {problem}"
            )


def checked_section(section: str | None, sections: Sequence[str]) -> str:
    """The section name without leading ``#`` marks; refuses one that cannot be a heading."""
    name = (section or "").strip().lstrip("#").strip()
    if not name:
        raise InvalidRequest(
            "a new note needs index_section, the section of its index to list it under; "
            + _sections_hint(sections)
        )
    problem = _text_problem(name, MAX_SECTION_CHARS)
    if problem is not None:
        raise InvalidRequest(f"index_section {problem}; " + _sections_hint(sections))
    return name


def checked_line(value: str | None, sections: Sequence[str] | None) -> str:
    """The note's line for its index; ``sections`` are the index's, for the refusal."""
    return _checked_description(
        value,
        "index_line",
        "a new note needs index_line, one line that says what the note holds; "
        + _sections_hint(sections or []),
    )


def checked_topic(value: str | None) -> str:
    """A new wiki topic's line for the master index."""
    return _checked_description(
        value,
        "topic_description",
        MISSING_TOPIC,
    )


def _checked_description(value: str | None, field: str, missing: str) -> str:
    line = (value or "").strip()
    if not line:
        raise InvalidRequest(missing)
    problem = _text_problem(line, MAX_INDEX_LINE_CHARS)
    if problem is not None:
        raise InvalidRequest(f"{field} {problem}")
    return line


def _link_target(relative: str) -> str:
    """A Markdown link target; one with whitespace, parentheses or angle brackets goes
    inside angle brackets."""
    return f"<{relative}>" if re.search(r"[\s()<>]", relative) else relative


def index_entry(
    folder: Folder, note: tuple[str, ...], topic: tuple[str, ...], description: str
) -> str:
    """The index line for ``note`` in its topic's index, in that index's format.

    Memory: ``- [<path>](<path>) — <description>``, the path relative to the
    index. Wiki: ``- [[<name>]] — <description>``, the note's name without
    ``.md``, the way Obsidian links it.
    """
    if folder == "wiki":
        name = note[-1].removesuffix(NOTE_SUFFIX)
        return f"- [[{name}]] — {description}"
    relative = "/".join(note[len(topic) :])
    label = relative.replace("[", "\\[").replace("]", "\\]")
    return f"- [{label}]({_link_target(relative)}) — {description}"


def new_index(spot: IndexSpot) -> str:
    """The heading of an index the server creates for a topic that has none."""
    topic = spot.topic
    # A topic folder gives its name; an area's memory folder itself gives "<area> memory".
    title = topic[-1] if len(topic) == _TOPIC_DEPTH else " ".join(topic)
    return f"# {title}\n"


def master_link(topic: tuple[str, ...]) -> str:
    """How the master index links a wiki topic, from the area's folder."""
    return f"[[{'/'.join(topic[1:])}/_index|{topic[-1]}]]"


def master_entry(topic: tuple[str, ...], description: str) -> str:
    return f"- {master_link(topic)} — {description}"


NEW_MASTER_INDEX = "# Wiki master index\n\n## Topics\n"


def _newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _find_section(found: list[tuple[int, int, str]], section: str) -> int | None:
    """The position in ``found`` of the section named ``section``, else one equal ignoring case."""
    sections = [(i, name) for i, (_, level, name) in enumerate(found) if level == SECTION_LEVEL]
    for i, name in sections:
        if name == section:
            return i
    wanted = section.casefold()
    for i, name in sections:
        if name.casefold() == wanted:
            return i
    return None


def with_entry(text: str, section: str, entry: str) -> str:
    """The index with ``entry`` as the last line of ``section``.

    A section whose name equals ``section`` ignoring case is that section; an
    unknown one is added at the end of the index. Every other line stays as it
    was, line endings included.
    """
    newline = _newline(text)
    lines = _lines(text)
    found = headings(lines)
    position = _find_section(found, section)
    if position is None:
        gap = [""] if lines and lines[-1].strip() else []
        lines += [*gap, f"## {section}", "", entry]
        return newline.join(lines) + newline
    start = found[position][0]
    following = [line for line, level, _ in found[position + 1 :] if level <= SECTION_LEVEL]
    end = following[0] if following else len(lines)
    last = end - 1
    while last > start and not lines[last].strip():
        last -= 1
    if last == start:
        # An empty section: the entry follows the heading after a blank line.
        lines[start + 1 : start + 1] = ["", entry]
    else:
        lines.insert(last + 1, entry)
    return newline.join(lines) + newline


def with_topic(text: str, entry: str) -> str:
    """The master index with ``entry`` as the last line of its first section,
    or of the file when it has no section."""
    sections = section_names(text)
    if sections:
        return with_entry(text, sections[0], entry)
    newline = _newline(text)
    return newline.join([*_lines(text), entry]) + newline


# -- commits -----------------------------------------------------------


def with_client(message: str, client: str) -> str:
    """``message`` with a last paragraph naming the client: the git trailer ``Memex-Client``.

    The client id comes from the verified token and equals a configured one;
    ``git log --format='%(trailers:key=Memex-Client)'`` lists it per write.
    """
    return f"{message}\n\n{CLIENT_TRAILER}: {client}"


def commit_message(writer: str, changes: Sequence[tuple[Action, str]]) -> str:
    """``<writer>: <action> <path>`` for one file; for several, a count and one line each.

    The first line of several names their deepest common folder, when they
    have one: ``<writer>: 3 files in alice/wiki``.
    """
    if len(changes) == 1:
        action, path = changes[0]
        return f"{writer}: {action} {path}"
    common = posixpath.commonpath([str(PurePosixPath(path).parent) for _, path in changes])
    head = f"{writer}: {len(changes)} files"
    if common not in {"", "."}:
        head += f" in {common}"
    return head + "\n\n" + "\n".join(f"{action} {path}" for action, path in changes)
