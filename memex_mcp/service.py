"""The memex service: search, read, list and write under the caller's rights, and keeping
the clone fresh.

The MCP tools are thin wrappers around ``Memex``; every rights decision is
made here or in ``memex_mcp.rights``, never in the transport layer.

Every write is one commit, pushed within the call. One lock serialises
everything that moves the clone: the writes and the refresh loop's merging,
rebasing and pushing. The loop's fetch runs outside it, so writes go on while
the remote answers; a write whose push the remote turns down fetches inside
it, since it must apply its change again on exactly the state it fetched.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, NotRequired, TypedDict

from memex_mcp.config import Config
from memex_mcp.embed import Embedder, EmbeddingError
from memex_mcp.index import ChunkRow, Index, is_archived, query_words
from memex_mcp.markdown import title_of
from memex_mcp.notes import (
    MASTER_INDEX,
    NEW_MASTER_INDEX,
    Action,
    IndexSpot,
    appended,
    check_composed,
    check_secrets,
    check_size,
    check_version,
    checked_line,
    checked_section,
    checked_topic,
    commit_message,
    decode,
    edited,
    index_entry,
    index_spot,
    is_daily,
    master_entry,
    master_link,
    new_index,
    new_note,
    section_names,
    version_of,
    with_entry,
    with_topic,
)
from memex_mcp.repo import CONFLICT_PREFIX, WRITER, Author, GitError, GitRepo, Push
from memex_mcp.rights import (
    NOTE_NOT_FOUND,
    NOTE_SUFFIX,
    Identity,
    InvalidRequest,
    NotFound,
    Policy,
    WriteOperation,
    check_area,
    is_utf8,
    resolve_folder,
    resolve_note,
    resolve_write,
    split_write_path,
)

log = logging.getLogger(__name__)

# Reciprocal rank fusion constant (Cormack et al.); 60 is the usual choice.
RRF_K = 60
ELLIPSIS = "…"
# Every word becomes an FTS5 term and a snippet lookup; unbounded queries
# would let one caller stall search for everyone.
MAX_QUERY_CHARS = 500
MAX_QUERY_WORDS = 32

NOTICE_UNAVAILABLE = (
    "Semantic search is unavailable right now (the embedding service did not answer); "
    "these results come from keyword search only."
)
NOTICE_DISABLED = (
    "Semantic search is disabled on this server; these results come from keyword search only."
)
# What the check embeds while the last real embedding call has failed.
CHECK_TEXT = "memex health check"

# After the remote turned a push down for lacking its newer commits, a write
# fetches and applies its change again this many times.
PUSH_RETRIES = 3
NOT_PUSHED = (
    "Saved in the server's clone, not yet pushed: the git host did not take the commit; "
    "the server tries again in every refresh round."
)
PUSH_REFUSED = (
    "Saved in the server's clone, but the git host refused the push; the server tries "
    "again in every refresh round, and its health reports failing while the refusal lasts."
)
WRITING_DISABLED = "writing is disabled on this server"
NOT_SEARCHABLE = "Saved, but search does not show the change yet; it catches up by itself."
EXISTS = "a note already exists at this path; read it, then use edit or append"
CONFLICT_PARKED = (
    "the clone's own commits conflict with the remote; they are on branch %s, "
    "and the clone follows the remote again"
)
# What index_sections takes a folder for: a note of this name in it.
_PLACEHOLDER = "new-note.md"
# Reduced to these characters, a username fits into an e-mail address.
_EMAIL_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]")
# git refuses these in a name.
_NAME_UNSAFE = re.compile(r"[<>\x00-\x1f\x7f]")


class Hit(TypedDict):
    path: str
    area: str
    title: str
    snippet: str
    score: float
    archived: bool
    created: str | None
    updated: str | None


class SearchResult(TypedDict):
    semantic: bool
    hits: list[Hit]
    notice: NotRequired[str]


class NoteResult(TypedDict):
    path: str
    area: str
    title: str
    archived: bool
    content: str
    version: str
    created: str | None
    updated: str | None


class WriteResult(TypedDict):
    path: str
    commit: str
    pushed: bool
    searchable: bool
    version: NotRequired[str]
    index: NotRequired[str | None]
    notice: NotRequired[str]


class SectionsResult(TypedDict):
    index: str | None
    exists: bool
    sections: list[str]
    # The index is a new wiki topic's: create then also needs topic_description.
    new_topic: bool


class AreasResult(TypedDict):
    areas: list[str]


class Entry(TypedDict):
    name: str
    path: str
    type: Literal["folder", "note"]


class ListResult(TypedDict):
    area: str
    folder: str
    entries: list[Entry]


class Health(TypedDict):
    """What /healthz answers: fixed words only, since it needs no token."""

    status: Literal["ok"]
    embeddings: Literal["ok", "failing", "disabled"]
    repo: Literal["ok", "failing", "conflict"]


@dataclass(frozen=True)
class _Change:
    action: Action
    path: str
    content: bytes | None = None  # None deletes the file


@dataclass(frozen=True)
class IndexLine:
    """How create lists a new note: the section of its index, the note's line there,
    and, for the first note of a new wiki topic, the topic's line in the master index."""

    section: str | None = None
    line: str | None = None
    topic: str | None = None


@dataclass(frozen=True)
class _Plan:
    """What one write does: its files (the note first), the note's new version, its index."""

    changes: tuple[_Change, ...]
    version: str | None = None
    index: str | None = None


def _write_file(target: Path, data: bytes) -> None:
    """Replace ``target`` at once, so that a read never sees half a note."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # Hidden, so that neither the index nor a listing ever picks it up.
    handle, temporary = tempfile.mkstemp(prefix=".memex-", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        Path(temporary).chmod(0o644)
        Path(temporary).replace(target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _remove_empty_folders(root: Path, relative: str) -> None:
    """Remove the folders a deleted note leaves empty, up to its area's memory or wiki folder."""
    parts = PurePosixPath(relative).parts
    for depth in range(len(parts) - 1, 2, -1):
        folder = root.joinpath(*parts[:depth])
        try:
            folder.rmdir()
        except OSError:
            return


def _checked_query(query: str) -> str:
    """The stripped query; raises InvalidRequest when it is empty or too big."""
    query = query.strip()
    if not query:
        raise InvalidRequest("the query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidRequest(f"the query is too long: at most {MAX_QUERY_CHARS} characters")
    if len(query_words(query)) > MAX_QUERY_WORDS:
        raise InvalidRequest(f"the query has too many words: at most {MAX_QUERY_WORDS} words")
    return query


def _fuse(rankings: Iterable[list[int]]) -> dict[int, float]:
    """Reciprocal rank fusion: only ranks count, never scores from different tables."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] += 1.0 / (RRF_K + rank + 1)
    return fused


def make_snippet(text: str, words: list[str], size: int) -> str:
    """At most ``size`` characters of ``text`` around the first match of any word."""
    flat = " ".join(text.split())
    if len(flat) <= size:
        return flat
    lowered = flat.lower()
    positions = [p for p in (lowered.find(word) for word in words) if p >= 0]
    pos = min(positions, default=0)
    window = size - 2 * len(ELLIPSIS)
    start = max(0, pos - window // 3)
    end = min(len(flat), start + window)
    start = max(0, end - window)
    piece = flat[start:end]
    if start > 0:
        space = piece.find(" ")
        if 0 <= space < window // 4:
            piece = piece[space + 1 :]
        piece = ELLIPSIS + piece
    if end < len(flat):
        space = piece.rfind(" ")
        if space > len(piece) - window // 4:
            piece = piece[:space]
        piece = piece + ELLIPSIS
    return piece


@dataclass
class Memex:
    config: Config
    policy: Policy
    repo: GitRepo
    index: Index
    embedder: Embedder | None
    clock: Callable[[], float] = time.monotonic
    _embed_retry_at: float = 0.0
    # Written from worker threads, read by /healthz.
    _embed_failed: bool = False  # a real embedding call failed, and none has succeeded since
    _check_failed: bool = False  # a check failed, and nothing has succeeded since
    _repo_failed: bool = False  # the last refresh round failed
    # The remote holds a conflict branch of this server, as last fetched.
    _conflict: bool = False
    # Held by everything that moves the clone; see the module's docstring.
    _git_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_config(cls, config: Config, embedder: Embedder | None) -> Memex:
        return cls(
            config=config,
            policy=Policy.from_config(config),
            repo=GitRepo(
                Path(config.repo.path), config.repo.branch, config.repo.git_timeout_seconds
            ),
            index=Index(
                Path(config.index.path),
                config.index.chunk_chars,
                frozenset(config.users.values()) | {config.rights.household_area},
            ),
            embedder=embedder,
        )

    @property
    def root(self) -> Path:
        return self.repo.path

    # -- tools -----------------------------------------------------------

    def search(
        self, identity: Identity, query: str, area: str | None = None, limit: int = 10
    ) -> SearchResult:
        areas = self.policy.areas_for(identity)
        if area is not None:
            areas = frozenset({check_area(area, areas)})
        query = _checked_query(query)
        limit = max(1, min(limit, self.config.index.max_limit))
        pool = max(limit * 5, 50)

        keyword_rankings = self.index.keyword_ranked(query, areas, pool)
        vector_ids, notice = self._semantic(query, areas, pool)

        # One keyword ranking per area plus the vector ranking.
        fused = _fuse((*keyword_rankings, vector_ids))
        rows = self.index.chunks(fused)
        best = self._best_per_note(fused, rows, areas)

        ordered = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
        words = query_words(query)
        hits: list[Hit] = []
        for path, (score, chunk_id) in ordered:
            row = rows[chunk_id]
            text = row.body or row.heading
            hits.append(
                {
                    "path": path,
                    "area": row.area,
                    "title": row.title,
                    "snippet": make_snippet(text, words, self.config.index.snippet_chars),
                    "score": round(score, 6),
                    "archived": row.archived,
                    "created": row.created,
                    "updated": row.updated,
                }
            )
        result: SearchResult = {"semantic": notice is None, "hits": hits}
        if notice is not None:
            result["notice"] = notice
        return result

    def _best_per_note(
        self, fused: dict[int, float], rows: dict[int, ChunkRow], areas: frozenset[str]
    ) -> dict[str, tuple[float, int]]:
        """Each note's best chunk as (score, chunk id); rows outside ``areas`` are dropped."""
        best: dict[str, tuple[float, int]] = {}
        for chunk_id, fused_score in fused.items():
            row = rows.get(chunk_id)
            if row is None or row.area not in areas:
                continue
            score = fused_score
            if row.archived:
                score *= self.config.index.archive_factor
            if row.path not in best or score > best[row.path][0]:
                best[row.path] = (score, chunk_id)
        return best

    def _semantic(
        self, query: str, areas: frozenset[str], pool: int
    ) -> tuple[list[int], str | None]:
        """Vector ranking, or an empty one plus the notice why there is none."""
        if self.embedder is None:
            return [], NOTICE_DISABLED
        now = self.clock()
        if now < self._embed_retry_at:
            return [], NOTICE_UNAVAILABLE
        try:
            vector = self.embedder.embed(
                [query], timeout=self.config.embeddings.query_timeout_seconds
            )[0]
        except EmbeddingError as exc:
            log.warning("query embedding failed, keyword search only: %s", exc)
            self._embed_retry_at = now + self.config.embeddings.retry_after_seconds
            self._embed_failed = True
            return [], NOTICE_UNAVAILABLE
        self._embedded()
        return self.index.vector_ranked(vector, areas, pool), None

    def read(self, identity: Identity, path: str) -> NoteResult:
        areas = self.policy.areas_for(identity)
        note = resolve_note(self.root, path, areas)
        data = note.real.read_bytes()
        content = data.decode("utf-8", errors="replace")
        created, updated = self.index.dates_of(note.relative) or (None, None)
        return {
            "path": note.relative,
            "area": note.area,
            "title": title_of(content, PurePosixPath(note.relative).stem),
            "archived": is_archived(note.relative),
            "content": content,
            "version": version_of(data),
            "created": created,
            "updated": updated,
        }

    def areas(self, identity: Identity) -> AreasResult:
        """The caller's own areas, and nothing about anyone else's."""
        return {"areas": sorted(self.policy.areas_for(identity))}

    def list(self, identity: Identity, area: str, folder: str | None = None) -> ListResult:
        areas = self.policy.areas_for(identity)
        resolved = resolve_folder(self.root, area, folder, areas)
        entries: list[Entry] = []
        for child in sorted(resolved.real.iterdir(), key=lambda p: p.name):
            # Hidden entries, symlinks and names that are not UTF-8 are never
            # listed; a listing shows only what reading and the index accept.
            hidden = child.name.startswith(".")
            if hidden or child.is_symlink() or not is_utf8(child.name):
                continue
            relative = f"{resolved.relative}/{child.name}"
            if child.is_dir():
                entries.append({"name": child.name, "path": relative, "type": "folder"})
            elif child.is_file() and child.name.endswith(NOTE_SUFFIX):
                entries.append({"name": child.name, "path": relative, "type": "note"})
        return {"area": resolved.area, "folder": resolved.relative, "entries": entries}

    # -- writing ---------------------------------------------------------

    def create(
        self,
        identity: Identity,
        path: str,
        content: str,
        listing: IndexLine | None = None,
    ) -> WriteResult:
        """A new note plus its line in its topic's index; daily notes take no index line.

        The first note of a new wiki topic also needs ``listing.topic``, the
        topic's line in the master index.
        """
        texts = listing or IndexLine()

        def plan(parts: tuple[str, ...], target: Path) -> _Plan:
            return self._plan_create(parts, target, content, texts)

        return self._write(identity, path, "create", plan)

    def append(self, identity: Identity, path: str, text: str) -> WriteResult:
        """``text`` added at the end of an existing note."""

        def plan(parts: tuple[str, ...], target: Path) -> _Plan:
            data = appended(decode(self._existing(target)), text).encode()
            check_secrets([text])
            return self._plan_change("append", parts, data)

        return self._write(identity, path, "append", plan)

    def edit(
        self,
        identity: Identity,
        path: str,
        old_text: str,
        new_text: str,
        version: str | None = None,
    ) -> WriteResult:
        """The one occurrence of ``old_text`` in a note replaced by ``new_text``."""

        def plan(parts: tuple[str, ...], target: Path) -> _Plan:
            current = self._existing(target)
            if version is not None:
                check_version(current, version)
            data = edited(decode(current), old_text, new_text).encode()
            check_secrets([new_text])
            return self._plan_change("update", parts, data)

        return self._write(identity, path, "edit", plan)

    def delete(self, identity: Identity, path: str, version: str) -> WriteResult:
        """A note removed, if it is still the version the caller read."""

        def plan(parts: tuple[str, ...], target: Path) -> _Plan:
            if not version:
                raise InvalidRequest("delete needs the note's version, as read returns it")
            check_version(self._existing(target), version)
            return _Plan(changes=(_Change("delete", "/".join(parts)),))

        return self._write(identity, path, "delete", plan)

    def index_sections(self, identity: Identity, path: str) -> SectionsResult:
        """The index a new note at ``path`` (a note's path, or a folder) would be listed in."""
        self.policy.areas_for(identity)
        note = path if path.endswith(NOTE_SUFFIX) else f"{path.rstrip('/')}/{_PLACEHOLDER}"
        parts = split_write_path(note)
        area = self.policy.may_write(identity, parts, "create")
        resolve_write(self.root, parts, area)
        if is_daily(parts):
            return {"index": None, "exists": False, "sections": [], "new_topic": False}
        spot = index_spot(parts)
        index_parts, current = self._find_index(spot)
        return {
            "index": "/".join(index_parts),
            "exists": current is not None,
            "sections": section_names(current or ""),
            "new_topic": current is None and spot.folder == "wiki",
        }

    def _plan_create(
        self, parts: tuple[str, ...], target: Path, content: str, texts: IndexLine
    ) -> _Plan:
        # Checked first: where a note may go depends on its path alone.
        spot = None if is_daily(parts) else index_spot(parts)
        if target.exists():
            raise InvalidRequest(EXISTS)
        data = new_note(content).encode()
        self._check_size(data)
        note = _Change("create", "/".join(parts), data)
        if spot is None:
            check_secrets([content])
            return _Plan(changes=(note,), version=version_of(data))
        index_parts, current = self._find_index(spot)
        sections = section_names(current or "")
        section = checked_section(texts.section, sections)
        line = checked_line(texts.line, sections)
        new_topic = current is None and spot.folder == "wiki"
        topic = ""
        if new_topic:
            topic = checked_topic(texts.topic)
        check_secrets([content, section, line, topic])
        entry = index_entry(spot.folder, parts, spot.topic, line)
        heading = new_index(spot) if current is None else ""
        check_composed([entry, heading.rstrip("\n"), f"## {section}"])
        text = with_entry(heading or current or "", section, entry)
        index_path = "/".join(index_parts)
        changes = [
            note,
            _Change("create" if current is None else "update", index_path, text.encode()),
        ]
        if new_topic:
            changes += self._master_change(spot, topic)
        return _Plan(changes=tuple(changes), version=version_of(data), index=index_path)

    def _plan_change(self, action: Action, parts: tuple[str, ...], data: bytes) -> _Plan:
        self._check_size(data)
        return _Plan(changes=(_Change(action, "/".join(parts), data),), version=version_of(data))

    def _check_size(self, data: bytes) -> None:
        check_size(data, self.config.write.max_note_bytes)

    @staticmethod
    def _existing(target: Path) -> bytes:
        """The content of the note at ``target``, which ``resolve_write`` returned."""
        if not target.is_file():
            raise NotFound(NOTE_NOT_FOUND)
        return target.read_bytes()

    def _index_file(self, parts: tuple[str, ...]) -> str | None:
        """The text of the index file at ``parts``, or None when there is none.

        Anything there but a regular file answers like a missing note.
        """
        file = self.root.joinpath(*parts)
        if file.is_symlink() or (file.exists() and not file.is_file()):
            raise NotFound(NOTE_NOT_FOUND)
        return decode(file.read_bytes()) if file.exists() else None

    def _find_index(self, spot: IndexSpot) -> tuple[tuple[str, ...], str | None]:
        """The topic's index and its text, None while the topic has none."""
        parts = (*spot.topic, spot.name)
        return parts, self._index_file(parts)

    def _master_change(self, spot: IndexSpot, description: str) -> list[_Change]:
        """The master index's line for a wiki topic whose index is new, unless it has one.

        The line goes to the end of the master index's first section.
        """
        parts = (*spot.topic[:2], MASTER_INDEX)
        current = self._index_file(parts)
        if current is not None and master_link(spot.topic) in current:
            return []
        # The topic's name was checked in the new index's heading, the description on its own.
        entry = master_entry(spot.topic, description)
        text = with_topic(NEW_MASTER_INDEX if current is None else current, entry)
        action: Action = "create" if current is None else "update"
        return [_Change(action, "/".join(parts), text.encode())]

    def _author(self, identity: Identity) -> Author:
        name = _NAME_UNSAFE.sub("", identity.username) or "unknown"
        local = _EMAIL_UNSAFE.sub("-", identity.username)
        return Author(name, self.config.write.author_email.replace("{username}", local))

    def _write(
        self,
        identity: Identity,
        path: str,
        operation: WriteOperation,
        plan_of: Callable[[tuple[str, ...], Path], _Plan],
    ) -> WriteResult:
        """Check the caller's right, then plan, commit and push the write under the lock.

        The rights are settled from the path alone before the clone is
        touched; the plan reads the clone, and is made again on the new state
        each time the remote turns the push down.
        """
        self.policy.areas_for(identity)
        if not self.config.write.enabled:
            raise InvalidRequest(WRITING_DISABLED)
        parts = split_write_path(path)
        area = self.policy.may_write(identity, parts, operation)
        # The file name goes into the commit message and the index.
        check_secrets(["/".join(parts)])
        author = self._author(identity)
        with self._git_lock:
            try:
                plan, commit, outcome = self._commit_and_push(parts, area, plan_of, author)
            finally:
                searchable = self._reindex()
        result: WriteResult = {
            "path": "/".join(parts),
            "commit": commit,
            "pushed": outcome is Push.PUSHED,
            "searchable": searchable,
        }
        if plan.version is not None:
            result["version"] = plan.version
        if operation == "create":
            result["index"] = plan.index
        notices: list[str] = []
        if outcome is Push.REFUSED:
            notices.append(PUSH_REFUSED)
        elif outcome is not Push.PUSHED:
            notices.append(NOT_PUSHED)
        if not searchable:
            notices.append(NOT_SEARCHABLE)
        if notices:
            result["notice"] = " ".join(notices)
        return result

    def _commit_and_push(
        self,
        parts: tuple[str, ...],
        area: str,
        plan_of: Callable[[tuple[str, ...], Path], _Plan],
        author: Author,
    ) -> tuple[_Plan, str, Push]:
        """Commit the write and push it; when the remote has newer commits, apply it again
        on them, up to PUSH_RETRIES times.

        When the clone's earlier saved commits cannot be brought onto the remote
        (they conflict, and no conflict branch could be pushed), the write is
        applied once more on top of them and saved without another push.
        """
        # A rebase a timeout left behind would make every later step fail.
        self.repo.abort_rebase()
        attempt = 0
        stuck = False
        while True:
            base = self.repo.head()
            plan = plan_of(parts, resolve_write(self.root, parts, area))
            commit = self._commit(plan, author, base)
            outcome = Push.REJECTED if stuck else self.repo.push()
            if outcome is not Push.REJECTED or attempt == PUSH_RETRIES or stuck:
                break
            attempt += 1
            log.info("the remote has newer commits; applying the write again on them (%d)", attempt)
            fetched = self._refetch(base)
            if fetched is None:
                break
            stuck = not fetched
        if outcome is Push.REFUSED:
            self._repo_failed = True
            log.error("the remote refused the push of %s; it stays in the clone", commit[:12])
        elif outcome is not Push.PUSHED:
            log.warning(
                "saved %s locally; the remote did not take it (%s)", commit[:12], outcome.value
            )
        return plan, commit, outcome

    def _commit(self, plan: _Plan, author: Author, base: str) -> str:
        """Write the plan's files and commit them; on any failure the tree goes back to ``base``."""
        message = commit_message(WRITER, [(change.action, change.path) for change in plan.changes])
        try:
            for change in plan.changes:
                target = self.root / change.path
                if change.content is None:
                    target.unlink()
                else:
                    _write_file(target, change.content)
            return self.repo.commit([change.path for change in plan.changes], message, author)
        except BaseException:
            self.repo.reset_hard(base)
            for change in plan.changes:
                if change.action == "create":
                    (self.root / change.path).unlink(missing_ok=True)
            raise
        finally:
            for change in plan.changes:
                if change.content is None:
                    _remove_empty_folders(self.root, change.path)

    def _refetch(self, base: str) -> bool | None:
        """Fetch after a turned-down push, drop the write's commit and bring ``base`` onto
        the new remote state.

        Returns None, keeping the commit to push later, when the fetch fails;
        else whether the clone now sits on the remote (False: its earlier saved
        commits conflict and could not be put on a conflict branch).
        """
        try:
            self.repo.fetch()
        except GitError as exc:
            log.warning("could not fetch after a turned-down push: %s", exc)
            return None
        self.repo.reset_hard(base)
        return self._integrate(push=False)

    def _reindex(self) -> bool:
        """Bring the search index to the clone's HEAD; a failure leaves it to the next round."""
        try:
            self._index_to(self.repo.head())
        except Exception:  # the write is committed; search catches up in a later round
            log.exception("a write is committed, but the search index could not take it yet")
            return False
        return True

    # -- freshness -------------------------------------------------------

    def prepare(self) -> None:
        """Clone if needed, open the index and bring it to the clone's HEAD."""
        if self.repo.is_repo():
            # The server owns its clone: whatever a crash left between writing and
            # committing (a changed note, a temporary file, a rebase) is dropped.
            self.repo.restore()
        else:
            remote = self.config.repo.remote
            if not remote:
                raise RuntimeError(
                    f"{self.root} is not a git clone and no [repo] remote is configured"
                )
            log.info("cloning the memex repository into %s", self.root)
            self.repo.clone(remote)
        self.index.open()
        if self.embedder is not None:
            self.index.ensure_embedding_model(self.embedder.model, self.embedder.dimensions)
        # A changed set of areas (a user added or remapped) needs a full build.
        full = self.index.fresh or self.index.areas_changed()
        self._index_to(self.repo.head(), force_full=full)
        self._conflict = bool(self.repo.conflict_branches())

    def _index_to(self, head: str, force_full: bool = False) -> None:
        stored = self.index.get_meta("head")
        if force_full or stored is None or not self.repo.has_commit(stored):
            count = self.index.rebuild(self.root)
            self.index.set_dates(self.repo.dates())
            log.info("indexed %d notes at %s (full build)", count, head[:12])
        elif stored != head:
            changed = self.repo.changed_paths(stored, head)
            count = self.index.update(self.root, changed)
            self.index.set_dates(self.repo.dates(changed), changed)
            log.info("re-indexed %d of %d changed paths at %s", count, len(changed), head[:12])
        self.index.set_meta("head", head)

    def refresh(self) -> bool:
        """Fetch, bring the clone onto the remote and push its own commits, then re-index.

        Without commits of its own the clone fast-forwards. Commits a write
        could not push are rebased onto the remote and pushed; when they
        conflict, they go to a conflict branch and the clone follows the
        remote again. A failure is logged and leaves the last state serving;
        returns whether the clone and the remote now agree.
        """
        try:
            self.repo.fetch()
        except GitError as exc:
            log.warning("could not update the clone, serving the last state: %s", exc)
            return False
        with self._git_lock:
            try:
                self.repo.abort_rebase()
                synced = self._integrate(push=True)
            except GitError as exc:
                log.warning("could not update the clone, serving the last state: %s", exc)
                synced = False
            self._conflict = bool(self.repo.conflict_branches())
            self._index_to(self.repo.head())
        return synced

    def _integrate(self, push: bool) -> bool:
        """Bring the clone onto the fetched remote; with ``push``, push its own commits.

        Returns whether nothing of the clone's own is left to push. With writing
        off, the clone only fast-forwards.
        """
        ahead = self.repo.ahead() if self.config.write.enabled else 0
        if ahead == 0:
            self.repo.fast_forward()
            return True
        if not self.repo.rebase():
            return self._park_conflict()
        if not push:
            return True
        outcome = self.repo.push()
        if outcome is Push.PUSHED:
            log.info("pushed %d commits saved while the remote was not taking them", ahead)
            return True
        log.warning(
            "could not push %d saved commits (%s); trying again next round", ahead, outcome.value
        )
        return False

    def _park_conflict(self) -> bool:
        """Push the clone's conflicting state to a new conflict branch, then follow the remote.

        Returns False, keeping the state, when no branch could be pushed.
        """
        day = datetime.now(UTC).date().isoformat()
        taken = set(self.repo.conflict_branches())
        names = [f"{CONFLICT_PREFIX}{day}"] + [f"{CONFLICT_PREFIX}{day}-{n}" for n in range(2, 100)]
        for name in names:
            if name in taken:
                continue
            outcome = self.repo.push("HEAD", name)
            if outcome is Push.PUSHED:
                log.error(CONFLICT_PARKED, name)
                self.repo.reset_hard(f"origin/{self.repo.branch}")
                self._conflict = True
                return True
            if outcome is not Push.REJECTED:
                break  # only a name taken meanwhile is worth the next one
        log.error(
            "the clone's own commits conflict with the remote and could not be saved to a branch"
        )
        return False

    def backfill(self) -> int:
        """Embed chunks that have no vector yet; stops at the first embedder failure."""
        if self.embedder is None:
            return 0
        done = 0
        batch_size = self.config.embeddings.batch_size
        while batch := self.index.missing_vectors(batch_size):
            try:
                vectors = self.embedder.embed(
                    [text for _, text in batch],
                    timeout=self.config.embeddings.index_timeout_seconds,
                )
            except EmbeddingError as exc:
                log.warning("embedding paused, %d chunks done this round: %s", done, exc)
                self._embed_failed = True
                return done
            self._embedded()
            self.index.store_vectors(
                [(chunk_id, vector) for (chunk_id, _), vector in zip(batch, vectors, strict=True)]
            )
            done += len(batch)
        if done:
            log.info("embedded %d chunks", done)
        return done

    # -- health ----------------------------------------------------------

    def health(self) -> Health:
        embeddings: Literal["ok", "failing", "disabled"]
        if self.embedder is None:
            embeddings = "disabled"
        elif self._embed_failed or self._check_failed:
            embeddings = "failing"
        else:
            embeddings = "ok"
        repo: Literal["ok", "failing", "conflict"] = "failing" if self._repo_failed else "ok"
        if self._conflict:
            repo = "conflict"
        return {"status": "ok", "embeddings": embeddings, "repo": repo}

    def _embedded(self) -> None:
        """An embedding succeeded (a real call or the check's own), which proves more than the
        model list.
        """
        if self._embed_failed or self._check_failed:
            log.info("the embedder works again")
        self._embed_failed = False
        self._check_failed = False

    def check_embeddings(self) -> None:
        """Check the embedder, once per refresh round.

        Usually this asks the embedder whether it offers the model, which loads no
        model. While the last real call has failed, that would not show whether
        embedding works again, so the check embeds a short text instead; only then
        does a check load the model. A failure is logged when it begins, not every round.
        """
        if self.embedder is None:
            return
        timeout = self.config.embeddings.query_timeout_seconds
        if self._embed_failed:
            try:
                self.embedder.embed([CHECK_TEXT], timeout=timeout)
            except EmbeddingError:
                return  # still failing; the failed call that began it was logged
            self._embedded()
            # It answered within a search's timeout, so searches stop skipping the embedder;
            # a backfill batch, with its longer timeout, does not show that.
            self._embed_retry_at = 0.0
            return
        try:
            self.embedder.check(timeout=timeout)
        except EmbeddingError as exc:
            if not self._check_failed:
                log.warning("the embedder check failed: %s", exc)
            self._check_failed = True
            return
        if self._check_failed:
            log.info("the embedder check passes again")
        self._check_failed = False

    def refresh_round(self, first: bool) -> None:
        """One round of the refresh loop: fetch, bring the clone onto the remote, push saved
        commits and re-index (not in the first round, which follows ``prepare``), check the
        embedder, fill in vectors.

        The repo counts as failing until a round passes without a failed fetch,
        merge, rebase or push and without raising; a conflict branch on the
        remote shows as conflict instead (see ``health``).
        """
        try:
            fresh = first or self.refresh()
            self.check_embeddings()
            self.backfill()
        except Exception:
            self._repo_failed = True
            raise
        self._repo_failed = not fresh
