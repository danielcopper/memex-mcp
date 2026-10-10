"""What a write does to the text of a note and of its index, without touching the clone.

The service reads the files, asks these functions for the new text, and
writes and commits the result. Everything here works on strings and path
segments only, so each rule is checked on its own.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from memex_mcp.markdown import headings
from memex_mcp.rights import NOTE_SUFFIX, InvalidRequest

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

CHANGED_SINCE_READ = "the note changed since you read it; read it again"

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
    """Where the index of a note may be: each folder from the note's own up to its topic's."""

    folder: Folder
    # Nearest first; the last is the topic folder, where a missing index is created.
    candidates: tuple[tuple[str, ...], ...]

    @property
    def name(self) -> str:
        return INDEX_FILE[self.folder]

    @property
    def topic(self) -> tuple[str, ...]:
        return self.candidates[-1]


def index_spot(parts: tuple[str, ...]) -> IndexSpot:
    """The folders whose index file a new note at ``parts`` goes into, nearest first.

    The topic is the folder directly under ``<area>/memory/`` or
    ``<area>/wiki/``; a note directly in ``<area>/memory/`` has that folder
    as its topic. A note directly in ``<area>/wiki/`` has none: the master
    index there lists topics, not notes. ``parts`` passed ``Policy.may_write``.
    """
    folder: Folder = "memory" if parts[1] == "memory" else "wiki"
    note_folder = parts[:-1]
    if folder == "wiki" and len(note_folder) < 3:  # noqa: PLR2004 - area/wiki/topic
        raise InvalidRequest(
            f"wiki notes go into a topic folder: {parts[0]}/wiki/<topic>/{parts[-1]}"
        )
    topic_depth = 3 if len(note_folder) >= 3 else 2  # noqa: PLR2004 - area/folder/topic
    candidates = tuple(
        note_folder[:depth] for depth in range(len(note_folder), topic_depth - 1, -1)
    )
    return IndexSpot(folder=folder, candidates=candidates)


def section_names(text: str) -> list[str]:
    """The sections of an index: its level-two headings, in order."""
    return [name for _, level, name in headings(text.splitlines()) if level == SECTION_LEVEL]


def _sections_hint(sections: Sequence[str]) -> str:
    if not sections:
        return "the index has no sections yet; name a new one"
    return "its sections: " + "; ".join(sections)


def checked_section(section: str | None, sections: Sequence[str]) -> str:
    """The section name without leading ``#`` marks; refuses one that cannot be a heading."""
    name = (section or "").strip().lstrip("#").strip()
    if not name:
        raise InvalidRequest(
            "a new note needs index_section, the section of its index to list it under; "
            + _sections_hint(sections)
        )
    if "\n" in name or "\r" in name or len(name) > MAX_SECTION_CHARS:
        raise InvalidRequest(
            f"index_section must be one line of at most {MAX_SECTION_CHARS} characters; "
            + _sections_hint(sections)
        )
    return name


def checked_line(index_line: str | None, sections: Sequence[str]) -> str:
    line = (index_line or "").strip()
    if not line:
        raise InvalidRequest(
            "a new note needs index_line, one line that says what the note holds; "
            + _sections_hint(sections)
        )
    if "\n" in line or "\r" in line or len(line) > MAX_INDEX_LINE_CHARS:
        raise InvalidRequest(
            f"index_line must be one line of at most {MAX_INDEX_LINE_CHARS} characters"
        )
    return line


def _link_target(relative: str) -> str:
    """A Markdown link target; one with spaces or brackets goes inside angle brackets."""
    return f"<{relative}>" if re.search(r"[\s()<>]", relative) else relative


def index_entry(
    folder: Folder, note: tuple[str, ...], index_folder: tuple[str, ...], description: str
) -> str:
    """The index line for ``note`` in the index of ``index_folder``, in that index's format.

    Memory: ``- [<path>](<path>) — <description>``, the path relative to the
    index. Wiki: ``- [[<name>]] — <description>``, the note's name without
    ``.md``, the way Obsidian links it.
    """
    if folder == "wiki":
        name = note[-1].removesuffix(NOTE_SUFFIX)
        return f"- [[{name}]] — {description}"
    relative = "/".join(note[len(index_folder) :])
    label = relative.replace("[", "\\[").replace("]", "\\]")
    return f"- [{label}]({_link_target(relative)}) — {description}"


def new_index(spot: IndexSpot) -> str:
    """The heading of an index the server creates for a topic that has none."""
    topic = spot.topic
    # A topic folder gives its name; an area's memory folder itself gives "<area> memory".
    title = topic[-1] if len(topic) > 2 else " ".join(topic)  # noqa: PLR2004 - area/folder
    return f"# {title}\n"


def master_entry(topic: tuple[str, ...]) -> str:
    """The master index's line for a new wiki topic, linked from the area's folder."""
    return f"- [[{'/'.join(topic[1:])}/_index|{topic[-1]}]]"


NEW_MASTER_INDEX = "# Wiki master index\n"


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
    unknown one is added at the end of the index. The index keeps its line
    endings.
    """
    newline = _newline(text)
    lines = text.splitlines()
    found = headings(lines)
    position = _find_section(found, section)
    if position is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ["", f"## {section}", "", entry]
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


def with_line(text: str, line: str) -> str:
    """``text`` with ``line`` as its last line."""
    newline = _newline(text)
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return newline.join([*lines, line]) + newline


# -- commits -----------------------------------------------------------


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
