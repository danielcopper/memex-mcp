"""The search index: a SQLite cache of the clone's notes, never committed.

Each note is split into chunks. A chunk is a row in ``chunks``, its text a row
of its area's own FTS5 table (same rowid) and, once the embedder has answered,
its vector a row of ``chunk_vectors``. Vectors are compared with sqlite-vec's
``vec_distance_cosine`` over the rows of the caller's areas only, so filtering
by area is exact rather than a post-filter of a global top-k.

One FTS5 table per area, not one for all: bm25 weighs a word by how often it
occurs in the table's documents, so with a shared table the order of a caller's
own hits would shift with what other people's areas contain. Per area, a
caller's ranking depends only on the areas they may read.

Text is indexed synchronously; vectors are filled in afterwards by
``missing_vectors``/``store_vectors``, so an unreachable embedder never holds
up the keyword index.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import sqlite_vec

from memex_mcp.markdown import chunk_note, title_of
from memex_mcp.rights import NOTE_SUFFIX, is_utf8, note_area

log = logging.getLogger(__name__)

# Bump whenever the tables or what goes into them change; a database with a
# different version is discarded and rebuilt.
SCHEMA_VERSION = 3

ARCHIVE_SEGMENT = "archive"

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    area TEXT NOT NULL,
    title TEXT NOT NULL,
    heading TEXT NOT NULL,
    body TEXT NOT NULL,
    ord INTEGER NOT NULL,
    archived INTEGER NOT NULL
);
CREATE INDEX chunks_path ON chunks(path);
CREATE INDEX chunks_area ON chunks(area);
CREATE TABLE areas (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);
CREATE TABLE chunk_vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL
);
"""

# The columns of every area's FTS5 table, and their bm25 weights in that order.
_FTS_COLUMNS = "path, title, heading, body, tokenize = 'unicode61 remove_diacritics 2'"
_BM25_WEIGHTS = "2.0, 5.0, 3.0, 1.0"


@dataclass(frozen=True)
class ChunkRow:
    id: int
    path: str
    area: str
    title: str
    heading: str
    body: str
    archived: bool


def is_archived(relative: str) -> bool:
    """True when any directory on the path is named ``archive``."""
    return ARCHIVE_SEGMENT in PurePosixPath(relative).parts[:-1]


def is_indexable(root: Path, relative: str, areas: frozenset[str]) -> bool:
    """Whether a repo-relative path is a note the index carries.

    The same rule the full walk applies: an ``.md`` regular file inside one of
    the configured ``areas``, no hidden segment, and no symlink anywhere on
    the way.
    """
    if not is_utf8(relative):
        # Git and the filesystem allow any bytes; SQLite and MCP need UTF-8.
        log.warning("skipping a file name that is not valid UTF-8: %r", relative)
        return False
    parts = PurePosixPath(relative).parts
    if len(parts) < 2 or parts[0] not in areas or not parts[-1].endswith(NOTE_SUFFIX):
        return False
    if any(part.startswith(".") or part == ".." for part in parts):
        return False
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            return False
    return current.is_file()


def iter_notes(root: Path, areas: frozenset[str]) -> Iterator[str]:
    """Every indexable note in the configured areas, as a repo-relative POSIX path.

    Only the area directories are walked: other top-level folders are neither
    read nor indexed, however many a writer pushes.
    """
    for area in sorted(areas):
        top = root / area
        if top.is_symlink() or not top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            base = Path(dirpath)
            for name in sorted(filenames):
                relative = (base / name).relative_to(root).as_posix()
                if is_indexable(root, relative, areas):
                    yield relative


def query_words(text: str) -> list[str]:
    """The distinct words of a query, lower-cased, in order."""
    return list(dict.fromkeys(re.findall(r"\w+", text.lower())))


def fts_query(text: str) -> str | None:
    """A safe FTS5 query: every word quoted, joined with OR (bm25 ranks)."""
    words = query_words(text)
    if not words:
        return None
    return " OR ".join(f'"{word}"' for word in words)


class Index:
    """Thread-safe access to the index database; every method takes the lock."""

    def __init__(self, path: Path, chunk_chars: int, areas: frozenset[str]) -> None:
        self._path = path
        self._chunk_chars = chunk_chars
        # The configured areas: the only ones indexed, each with its FTS5 table.
        self._areas = areas
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.fresh = False

    # -- lifecycle -------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
        except BaseException:
            conn.close()
            raise
        return conn

    def open(self) -> None:
        """Open the database; a missing or outdated one is created anew.

        ``fresh`` tells the caller that the index is empty and needs a full
        build.
        """
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                try:
                    conn = self._connect()
                except sqlite3.DatabaseError:
                    version = None
                else:
                    version = self._read_schema_version(conn)
                    if version == SCHEMA_VERSION:
                        self._conn = conn
                        self.fresh = False
                        return
                    conn.close()
                log.info("index schema %s differs from %s; rebuilding", version, SCHEMA_VERSION)
                for suffix in ("", "-wal", "-shm"):
                    Path(f"{self._path}{suffix}").unlink(missing_ok=True)
            conn = self._connect()
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
            self._conn = conn
            self.fresh = True

    @staticmethod
    def _read_schema_version(conn: sqlite3.Connection) -> int | None:
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        except sqlite3.DatabaseError:
            return None
        try:
            return int(row[0]) if row else None
        except ValueError:
            return None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("the index is not open")
        return self._conn

    # -- metadata --------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def ensure_embedding_model(self, model: str, dimensions: int) -> None:
        """Drop all vectors when the model or its dimensions changed."""
        wanted = f"{model}:{dimensions}"
        with self._lock:
            if self.get_meta("embedding_model") != wanted:
                self.conn.execute("DELETE FROM chunk_vectors")
                self.set_meta("embedding_model", wanted)

    # -- writing ---------------------------------------------------------

    def _fts_table(self, area: str, create: bool) -> str | None:
        """The name of the area's FTS5 table; created on first use when ``create``.

        The name is built from the area's integer id, never from its text.
        """
        row = self.conn.execute("SELECT id FROM areas WHERE name = ?", (area,)).fetchone()
        if row is not None:
            return f"fts_{int(row[0])}"
        if not create:
            return None
        cursor = self.conn.execute("INSERT INTO areas (name) VALUES (?)", (area,))
        table = f"fts_{int(cursor.lastrowid or 0)}"
        self.conn.execute(f"CREATE VIRTUAL TABLE {table} USING fts5({_FTS_COLUMNS})")
        return table

    def _delete_paths(self, paths: Iterable[str]) -> None:
        for path in paths:
            rows = self.conn.execute("SELECT id, area FROM chunks WHERE path = ?", (path,))
            for chunk_id, area in rows.fetchall():
                table = self._fts_table(area, create=False)
                if table is not None:
                    self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (chunk_id,))  # noqa: S608 - the name is fts_<integer>
            self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))

    def _insert_note(self, root: Path, relative: str) -> None:
        area = note_area(relative)
        if area is None:
            return
        text = (root / relative).read_text(encoding="utf-8", errors="replace")
        title = title_of(text, PurePosixPath(relative).stem)
        archived = int(is_archived(relative))
        table = self._fts_table(area, create=False)
        if table is None:
            raise RuntimeError(f"no search table for area {area!r}")
        for ord_, chunk in enumerate(chunk_note(text, self._chunk_chars)):
            cursor = self.conn.execute(
                "INSERT INTO chunks (path, area, title, heading, body, ord, archived) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (relative, area, title, chunk.heading, chunk.body, ord_, archived),
            )
            self.conn.execute(
                f"INSERT INTO {table} (rowid, path, title, heading, body) VALUES (?, ?, ?, ?, ?)",  # noqa: S608 - the name is fts_<integer>
                (cursor.lastrowid, relative, title, chunk.heading, chunk.body),
            )

    def _insert_note_or_skip(self, root: Path, relative: str) -> bool:
        """Insert one note inside its own savepoint; a failure skips only that note."""
        self.conn.execute("SAVEPOINT note")
        try:
            self._insert_note(root, relative)
        except Exception as exc:
            self.conn.execute("ROLLBACK TO note")
            self.conn.execute("RELEASE note")
            log.warning("skipping %s: %s", relative, exc)
            return False
        self.conn.execute("RELEASE note")
        return True

    def _ensure_area_tables(self) -> None:
        """Create every configured area's FTS5 table up front.

        Never inside a note's savepoint: rolling back to a savepoint that
        created and wrote a virtual table leaves SQLite unable to open the
        next savepoint ("SQL logic error").
        """
        for area in sorted(self._areas):
            self._fts_table(area, create=True)

    def _areas_key(self) -> str:
        return json.dumps(sorted(self._areas))

    def areas_changed(self) -> bool:
        """Whether the index was built for a different set of areas than configured now."""
        return self.get_meta("areas") != self._areas_key()

    def rebuild(self, root: Path) -> int:
        """Index every note from scratch; returns the number of notes indexed."""
        with self._lock:
            notes = list(iter_notes(root, self._areas))
            self.conn.execute("BEGIN")
            try:
                for (area_id,) in self.conn.execute("SELECT id FROM areas").fetchall():
                    self.conn.execute(f"DROP TABLE fts_{int(area_id)}")
                self.conn.execute("DELETE FROM areas")
                self.conn.execute("DELETE FROM chunks")
                self._ensure_area_tables()
                self.set_meta("areas", self._areas_key())
                indexed = sum(self._insert_note_or_skip(root, relative) for relative in notes)
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            return indexed

    def update(self, root: Path, paths: Iterable[str]) -> int:
        """Re-index exactly these repo-relative paths: changed ones are re-read,
        removed or no longer indexable ones dropped. Returns how many are indexed."""
        with self._lock:
            unique: list[str] = []
            for path in dict.fromkeys(paths):
                if is_utf8(path):
                    unique.append(path)
                else:
                    # Never indexed, so there is nothing to delete either.
                    log.warning("skipping a file name that is not valid UTF-8: %r", path)
            self.conn.execute("BEGIN")
            try:
                # The area tables exist: rebuild creates all of them, and a
                # changed set of areas always leads to a rebuild.
                self._delete_paths(unique)
                indexed = 0
                for relative in unique:
                    if is_indexable(root, relative, self._areas):
                        indexed += self._insert_note_or_skip(root, relative)
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            return indexed

    def missing_vectors(self, limit: int) -> list[tuple[int, str]]:
        """Chunks without a vector yet, with the text to embed."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT c.id, c.title, c.heading, c.body FROM chunks c "
                "LEFT JOIN chunk_vectors v ON v.chunk_id = c.id "
                "WHERE v.chunk_id IS NULL ORDER BY c.id LIMIT ?",
                (limit,),
            ).fetchall()
        return [(row[0], "\n".join(part for part in row[1:] if part)) for row in rows]

    def store_vectors(self, vectors: Sequence[tuple[int, list[float]]]) -> None:
        """Store vectors; a chunk deleted meanwhile is skipped (ids are never reused)."""
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding) "
                "SELECT ?, ? WHERE EXISTS (SELECT 1 FROM chunks WHERE id = ?)",
                [(i, sqlite_vec.serialize_float32(v), i) for i, v in vectors],
            )

    # -- reading ---------------------------------------------------------

    def keyword_ranked(self, query: str, areas: frozenset[str], limit: int) -> list[list[int]]:
        """One bm25 ranking of chunk ids per area in ``areas``, each best-first.

        Each ranking is computed in that area's own table, so it depends only
        on that area's notes; the caller fuses them.
        """
        match = fts_query(query)
        if match is None:
            return []
        rankings: list[list[int]] = []
        with self._lock:
            for area in sorted(areas):
                table = self._fts_table(area, create=False)
                if table is None:
                    continue
                rows = self.conn.execute(
                    f"SELECT c.id FROM {table} f JOIN chunks c ON c.id = f.rowid "  # noqa: S608 - the name is fts_<integer>
                    f"WHERE {table} MATCH ? "
                    f"ORDER BY bm25({table}, {_BM25_WEIGHTS}), c.path, c.ord LIMIT ?",
                    (match, limit),
                ).fetchall()
                rankings.append([row[0] for row in rows])
        return rankings

    def vector_ranked(self, vector: list[float], areas: frozenset[str], limit: int) -> list[int]:
        """Chunk ids nearest-first by cosine distance, restricted to ``areas``."""
        if not areas:
            return []
        marks = ",".join("?" * len(areas))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT c.id FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id "  # noqa: S608 - only placeholders are interpolated
                f"WHERE c.area IN ({marks}) "
                f"ORDER BY vec_distance_cosine(v.embedding, ?), c.path, c.ord LIMIT ?",
                (*sorted(areas), sqlite_vec.serialize_float32(vector), limit),
            ).fetchall()
        return [row[0] for row in rows]

    def chunks(self, ids: Iterable[int]) -> dict[int, ChunkRow]:
        wanted = list(ids)
        if not wanted:
            return {}
        marks = ",".join("?" * len(wanted))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id, path, area, title, heading, body, archived FROM chunks "  # noqa: S608 - only placeholders are interpolated
                f"WHERE id IN ({marks})",
                wanted,
            ).fetchall()
        return {
            row[0]: ChunkRow(row[0], row[1], row[2], row[3], row[4], row[5], bool(row[6]))
            for row in rows
        }

    def chunk_ids_for(self, path: str) -> list[int]:
        with self._lock:
            return [
                row[0]
                for row in self.conn.execute(
                    "SELECT id FROM chunks WHERE path = ? ORDER BY ord", (path,)
                )
            ]

    def paths(self) -> list[str]:
        with self._lock:
            return [row[0] for row in self.conn.execute("SELECT DISTINCT path FROM chunks")]
