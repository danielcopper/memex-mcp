"""The memex service: search, read and list under the caller's rights, and keeping the clone fresh.

The MCP tools are thin wrappers around ``Memex``; every rights decision is
made here or in ``memex_mcp.rights``, never in the transport layer.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, NotRequired, TypedDict

from memex_mcp.config import Config
from memex_mcp.embed import Embedder, EmbeddingError
from memex_mcp.index import Index, is_archived, query_words
from memex_mcp.markdown import title_of
from memex_mcp.repo import GitError, GitRepo
from memex_mcp.rights import (
    NOTE_SUFFIX,
    Identity,
    Policy,
    check_area,
    is_utf8,
    resolve_folder,
    resolve_note,
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


class InvalidRequest(Exception):
    """The request itself is malformed (an empty query); the message is safe to show."""


class Hit(TypedDict):
    path: str
    area: str
    title: str
    snippet: str
    score: float
    archived: bool


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
        query = query.strip()
        if not query:
            raise InvalidRequest("the query must not be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise InvalidRequest(f"the query is too long: at most {MAX_QUERY_CHARS} characters")
        if len(query_words(query)) > MAX_QUERY_WORDS:
            raise InvalidRequest(f"the query has too many words: at most {MAX_QUERY_WORDS} words")
        limit = max(1, min(limit, self.config.index.max_limit))
        pool = max(limit * 5, 50)

        keyword_rankings = self.index.keyword_ranked(query, areas, pool)
        vector_ids, notice = self._semantic(query, areas, pool)

        # Reciprocal rank fusion over one keyword ranking per area plus the
        # vector ranking: only ranks count, never scores from different tables.
        fused: dict[int, float] = defaultdict(float)
        for ranking in (*keyword_rankings, vector_ids):
            for rank, chunk_id in enumerate(ranking):
                fused[chunk_id] += 1.0 / (RRF_K + rank + 1)
        rows = self.index.chunks(fused)

        best: dict[str, tuple[float, int]] = {}
        for chunk_id, score in fused.items():
            row = rows.get(chunk_id)
            if row is None or row.area not in areas:
                continue
            if row.archived:
                score *= self.config.index.archive_factor
            if row.path not in best or score > best[row.path][0]:
                best[row.path] = (score, chunk_id)

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
                }
            )
        result: SearchResult = {"semantic": notice is None, "hits": hits}
        if notice is not None:
            result["notice"] = notice
        return result

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
            return [], NOTICE_UNAVAILABLE
        return self.index.vector_ranked(vector, areas, pool), None

    def read(self, identity: Identity, path: str) -> NoteResult:
        areas = self.policy.areas_for(identity)
        note = resolve_note(self.root, path, areas)
        content = note.real.read_text(encoding="utf-8", errors="replace")
        return {
            "path": note.relative,
            "area": note.area,
            "title": title_of(content, PurePosixPath(note.relative).stem),
            "archived": is_archived(note.relative),
            "content": content,
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

    # -- freshness -------------------------------------------------------

    def prepare(self) -> None:
        """Clone if needed, open the index and bring it to the clone's HEAD."""
        if not self.repo.is_repo():
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

    def _index_to(self, head: str, force_full: bool = False) -> None:
        stored = self.index.get_meta("head")
        if force_full or stored is None or not self.repo.has_commit(stored):
            count = self.index.rebuild(self.root)
            log.info("indexed %d notes at %s (full build)", count, head[:12])
        elif stored != head:
            changed = self.repo.changed_paths(stored, head)
            count = self.index.update(self.root, changed)
            log.info("re-indexed %d of %d changed paths at %s", count, len(changed), head[:12])
        self.index.set_meta("head", head)

    def refresh(self) -> bool:
        """Fetch and fast-forward the clone, then re-index what changed.

        A failed fetch or merge is logged and leaves the last state serving;
        returns whether the clone is now at the remote's head.
        """
        try:
            self.repo.fetch()
            self.repo.fast_forward()
        except GitError as exc:
            log.warning("could not update the clone, serving the last state: %s", exc)
            return False
        self._index_to(self.repo.head())
        return True

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
                return done
            self.index.store_vectors(
                [(chunk_id, vector) for (chunk_id, _), vector in zip(batch, vectors, strict=True)]
            )
            done += len(batch)
        if done:
            log.info("embedded %d chunks", done)
        return done
