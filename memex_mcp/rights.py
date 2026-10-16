"""Who may read and write which area, and which paths inside an area are notes.

This module is the security core. Every tool asks it for the caller's areas
and for every path it touches; nothing reads the clone without passing here.

The rules (v1, no admin override):

- Without membership in the access group (``memex``) there is no access.
- A user reads the area their username maps to in ``[users]``.
- A member of the household group also reads the household area.
- A path is a relative POSIX path whose first segment is the area. It must
  not be absolute, contain ``..`` or a backslash or a NUL byte, or touch a
  hidden segment (``.git``, ``.obsidian``); after following symlinks it must
  still lie inside the same area and pass the same checks.
- What the caller cannot see answers exactly like what does not exist.

Writing follows the same rules, plus these: a caller writes exactly where
they read, and only below ``<area>/memory/`` or ``<area>/wiki/``; no write
names ``CLAUDE.md``, ``CLAUDE.local.md`` or ``AGENTS.md`` in any letter case,
as a file or as a folder; no write names an index file (``MEMORY.md``,
``_index.md``, ``_master-index.md``) either, since indexes change only
through ``create``, which adds a new note's line and creates the index, a
section or the topic's line in the main index of memory or the wiki's master
index when needed; and no write follows a symlink anywhere on its path.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from memex_mcp.config import Config

NOTE_SUFFIX = ".md"

# The folders directly inside an area that writes may go into.
WRITABLE_FOLDERS = frozenset({"memory", "wiki"})
# Loaded by assistants (Claude Code and others) that open the folder: never written through
# the server. Compared ignoring case, against every segment of a path.
PROTECTED_NAMES = frozenset({"CLAUDE.md", "CLAUDE.local.md", "AGENTS.md"})
# Changed only by create: a new note's line, and the index, a section or the topic's line
# in the main index of memory or the wiki's master index when needed. Compared like
# PROTECTED_NAMES.
INDEX_NAMES = frozenset({"MEMORY.md", "_index.md", "_master-index.md"})
_PROTECTED_FOLDED = frozenset(name.casefold() for name in PROTECTED_NAMES)
_INDEX_FOLDED = frozenset(name.casefold() for name in INDEX_NAMES)
# A writable note's path has its area, memory or wiki, and at least the file name.
_MIN_WRITE_SEGMENTS = 3

type WriteOperation = Literal["create", "append", "edit", "delete"]


# The one answer when the caller has no access at all, whichever setting is
# missing (the access group or an area for their username).
ACCESS_DENIED = "access denied for this account"


class AccessDenied(Exception):
    """The caller has no access at all, or sent a malformed path; the message is safe to show.

    Its messages depend only on the caller's identity and input, never on
    what the clone holds.
    """


class InvalidRequest(Exception):
    """The request itself is malformed or cannot be carried out; the message is safe to show.

    Raised only once the caller's rights for the path are settled, so its
    messages may describe what the caller's own areas hold.
    """


class NotFound(Exception):
    """No note, folder or area the caller can see answers to this name.

    Raised alike for what does not exist and for what lies outside the
    caller's areas; the message is one of the fixed ``*_NOT_FOUND`` texts.
    """


@dataclass(frozen=True)
class Identity:
    username: str
    groups: frozenset[str]
    # The OAuth client the caller came through (the verified token's `azp`); rights do
    # not depend on it, every write commit names it.
    client: str


@dataclass(frozen=True)
class Policy:
    """Maps identities to the areas they may read and write."""

    users: dict[str, str]
    access_group: str
    household_group: str
    household_area: str

    @classmethod
    def from_config(cls, config: Config) -> Policy:
        return cls(
            users=dict(config.users),
            access_group=config.rights.access_group,
            household_group=config.rights.household_group,
            household_area=config.rights.household_area,
        )

    def areas_for(self, identity: Identity) -> frozenset[str]:
        """The areas this identity reads; raises AccessDenied when there are none."""
        if self.access_group not in identity.groups:
            raise AccessDenied(ACCESS_DENIED)
        areas: set[str] = set()
        own = self.users.get(identity.username)
        if own is not None:
            areas.add(own)
        if self.household_group in identity.groups:
            areas.add(self.household_area)
        if not areas:
            raise AccessDenied(ACCESS_DENIED)
        return frozenset(areas)

    def may_write(
        self, identity: Identity, parts: tuple[str, ...], operation: WriteOperation
    ) -> str:
        """The area ``operation`` on the note at ``parts`` writes to; raises when it may not.

        The one decision every write tool asks for. A person writes exactly
        where they read; ``operation`` is where rules that differ by operation
        (devices without a person) will plug in. Reads only the given names,
        never the clone: a foreign area answers NotFound like a missing note,
        and only inside the caller's own areas do the messages get specific.
        """
        del operation  # every operation is allowed for a person; device rules will read it
        areas = self.areas_for(identity)
        if not parts:
            raise NotFound(NOTE_NOT_FOUND)
        area = check_area(parts[0], areas, NOTE_NOT_FOUND)
        if len(parts) < _MIN_WRITE_SEGMENTS or parts[1] not in WRITABLE_FOLDERS:
            raise AccessDenied(WRITE_OUTSIDE)
        # Trailing dots and spaces dropped, as Windows drops them from file names.
        folded = {part.rstrip(". ").casefold() for part in parts[2:]}
        if folded & _PROTECTED_FOLDED:
            raise AccessDenied(PROTECTED_FILE)
        if folded & _INDEX_FOLDED:
            raise AccessDenied(INDEX_FILE)
        return area


# The one answer for every path or area the caller cannot see, whatever the
# reason: outside their areas, absent, or a symlink leading elsewhere. A
# foreign note that exists and one that does not are indistinguishable, and
# no message names an area, a configured value or a server path.
NOTE_NOT_FOUND = "no such note"
FOLDER_NOT_FOUND = "no such folder"
AREA_NOT_FOUND = "no such area"

WRITE_OUTSIDE = (
    "notes can only be written below <area>/memory/ or <area>/wiki/, "
    "e.g. 'household/memory/wifi.md'"
)
PROTECTED_FILE = (
    "files and folders named CLAUDE.md, CLAUDE.local.md or AGENTS.md, in any letter case, "
    "cannot be written through memex"
)
INDEX_FILE = (
    "files and folders named MEMORY.md, _index.md or _master-index.md, in any letter case, "
    "cannot be written directly; create adds a new note's line to its index, creating the "
    "index when needed"
)
FORBIDDEN_CHARACTER = "the path contains a forbidden character"
WORD_START_AT = (
    "the path must not have '@' at the start of a word, which assistants read as a file to load"
)
# What no written path and no line of an index may contain: control characters and line
# breaks, which would split a commit message or an index line, and format characters such
# as bidi overrides, which can hide what follows where the text is shown.
HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
# Hides what follows in a rendered view.
HTML_COMMENT = "<!--"
# Claude Code reads "@path" at the start of a word, in a MEMORY.md it loads, as an import
# of that file; a path ends up in its index as a link.
IMPORT_AT = re.compile(r"(?<!\w)@")


def split_path(path: str) -> tuple[str, ...]:
    """Normalise a client-supplied relative path into its segments.

    Empty and ``.`` segments are dropped; everything that could leave the
    tree or name a hidden entry is refused outright rather than resolved.
    These checks read only the given string, never the clone, so their
    distinct messages say nothing about what exists.
    """
    if "\x00" in path or "\\" in path:
        raise AccessDenied(FORBIDDEN_CHARACTER)
    if path.startswith("/"):
        raise AccessDenied("absolute paths are not allowed; give a path like 'area/folder/note.md'")
    parts = tuple(part for part in path.split("/") if part not in {"", "."})
    for part in parts:
        if part == "..":
            raise AccessDenied("'..' is not allowed in a path")
        if part.startswith("."):
            raise AccessDenied("hidden files and folders are not accessible")
    return parts


def check_area(area: str, areas: frozenset[str], message: str = AREA_NOT_FOUND) -> str:
    """Return ``area`` when it is one of ``areas``.

    Anything else, a foreign area or one that does not exist, raises the same
    NotFound before the clone is touched, so neither the answer nor the work
    done differs between the two.
    """
    if area not in areas:
        raise NotFound(message)
    return area


@dataclass(frozen=True)
class Resolved:
    """A path that passed every check: its area, repo-relative form and real location."""

    area: str
    relative: str
    real: Path


def _resolve(root: Path, parts: tuple[str, ...], areas: frozenset[str], message: str) -> Resolved:
    if not parts:
        raise NotFound(message)
    area = check_area(parts[0], areas, message)
    real_root = root.resolve()
    area_root = (real_root / area).resolve()
    # The area directory itself must not be a symlink to somewhere else.
    if area_root != real_root / area:
        raise NotFound(message)
    try:
        real = real_root.joinpath(*parts).resolve(strict=True)
    except (OSError, RuntimeError):
        # Missing paths, symlink loops and unreadable components end here.
        raise NotFound(message) from None
    if not real.is_relative_to(area_root):
        raise NotFound(message)
    # A symlink may stay inside the area yet point at a hidden entry.
    if any(part.startswith(".") for part in real.relative_to(real_root).parts):
        raise NotFound(message)
    return Resolved(area=area, relative="/".join(parts), real=real)


def resolve_note(root: Path, path: str, areas: frozenset[str]) -> Resolved:
    """Resolve a note path for reading: a regular ``.md`` file inside an allowed area."""
    parts = split_path(path)
    if not parts or not parts[-1].endswith(NOTE_SUFFIX):
        raise AccessDenied(f"only notes ({NOTE_SUFFIX} files) can be read")
    resolved = _resolve(root, parts, areas, NOTE_NOT_FOUND)
    if not resolved.real.name.endswith(NOTE_SUFFIX) or not resolved.real.is_file():
        raise NotFound(NOTE_NOT_FOUND)
    return resolved


def resolve_folder(root: Path, area: str, folder: str | None, areas: frozenset[str]) -> Resolved:
    """Resolve a folder inside an allowed area for listing."""
    check_area(area, areas, FOLDER_NOT_FOUND)
    parts = (area, *split_path(folder or ""))
    resolved = _resolve(root, parts, areas, FOLDER_NOT_FOUND)
    if not resolved.real.is_dir():
        raise NotFound(FOLDER_NOT_FOUND)
    return resolved


def split_write_path(path: str) -> tuple[str, ...]:
    """``split_path`` for a note to write: an ``.md`` name, no control or format character,
    no line break, no ``<!--`` and no ``@`` starting a word.

    The path ends up in a commit message, one line per file, and in an index.
    """
    parts = split_path(path)
    if HTML_COMMENT in path or any(
        unicodedata.category(char) in HIDDEN_CATEGORIES for char in path
    ):
        raise AccessDenied(FORBIDDEN_CHARACTER)
    if IMPORT_AT.search(path):
        raise AccessDenied(WORD_START_AT)
    if not parts or not parts[-1].endswith(NOTE_SUFFIX):
        raise AccessDenied(f"only notes ({NOTE_SUFFIX} files) can be written")
    return parts


def resolve_write(root: Path, parts: tuple[str, ...], area: str) -> Path:
    """Where the note at ``parts`` lives in the clone; it may not exist yet.

    Call it once ``Policy.may_write`` has passed. No part of the path may be
    a symlink, the area directory included, and no part but the last may be
    anything other than a directory; whatever stands in the way answers like
    a missing note. After resolution the folder must still lie inside the area.
    """
    real_root = root.resolve()
    current = real_root
    for index, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            raise NotFound(NOTE_NOT_FOUND)
        last = index == len(parts) - 1
        if not last and current.exists() and not current.is_dir():
            raise NotFound(NOTE_NOT_FOUND)
    if not current.parent.resolve().is_relative_to(real_root / area):
        raise NotFound(NOTE_NOT_FOUND)
    return current


def is_utf8(name: str) -> bool:
    """False for a name the filesystem gave back with undecodable bytes (surrogates)."""
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def note_area(relative: str) -> str | None:
    """The area of a repo-relative note path, or None for a file at the repo root."""
    parts = PurePosixPath(relative).parts
    return parts[0] if len(parts) > 1 else None
