"""Who may read which area, and which paths inside an area are notes.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from memex_mcp.config import Config

NOTE_SUFFIX = ".md"


# The one answer when the caller has no access at all, whichever setting is
# missing (the access group or an area for their username).
ACCESS_DENIED = "access denied for this account"


class AccessDenied(Exception):
    """The caller has no access at all, or sent a malformed path; the message is safe to show.

    Its messages depend only on the caller's identity and input, never on
    what the clone holds.
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


@dataclass(frozen=True)
class Policy:
    """Maps identities to the areas they may read."""

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


# The one answer for every path or area the caller cannot see, whatever the
# reason: outside their areas, absent, or a symlink leading elsewhere. A
# foreign note that exists and one that does not are indistinguishable, and
# no message names an area, a configured value or a server path.
NOTE_NOT_FOUND = "no such note"
FOLDER_NOT_FOUND = "no such folder"
AREA_NOT_FOUND = "no such area"


def split_path(path: str) -> tuple[str, ...]:
    """Normalise a client-supplied relative path into its segments.

    Empty and ``.`` segments are dropped; everything that could leave the
    tree or name a hidden entry is refused outright rather than resolved.
    These checks read only the given string, never the clone, so their
    distinct messages say nothing about what exists.
    """
    if "\x00" in path or "\\" in path:
        raise AccessDenied("the path contains a forbidden character")
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
