"""Freshness: clone, fetch, fast-forward, incremental re-index, rebuilds, failures."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

import memex_mcp.index as index_module
from memex_mcp.index import SCHEMA_VERSION
from memex_mcp.markdown import Chunk, chunk_note
from memex_mcp.repo import GitError, GitRepo, redact
from memex_mcp.rights import Identity
from memex_mcp.service import Memex
from tests.conftest import (
    ALICE,
    FailingEmbedder,
    FakeEmbedder,
    Origin,
    git,
    make_config,
    malformed_ollama,
)


def paths(memex: Memex, query: str) -> list[str]:
    return [hit["path"] for hit in memex.search(ALICE, query, limit=50)["hits"]]


def test_prepare_clones_from_the_remote(memex: Memex, origin: Origin) -> None:
    assert (memex.root / ".git").exists()
    assert memex.index.get_meta("head") == git(origin.work, "rev-parse", "HEAD").strip()
    assert "alice/memory/bike-repair.md" in paths(memex, "degreaser")


def test_refresh_reindexes_only_what_changed(memex: Memex, origin: Origin) -> None:
    untouched = memex.index.chunk_ids_for("alice/memory/zebra.md")
    changed_before = memex.index.chunk_ids_for("alice/memory/bike-repair.md")
    git(origin.work, "mv", "alice/wiki/networking.md", "alice/wiki/network.md")
    new_head = origin.commit(
        "edit, add, delete, rename",
        notes={
            "alice/memory/bike-repair.md": "# Bike repair\n\nNow the workshop does it.\n",
            "alice/memory/new.md": "# New\n\nA fresh quokka note.\n",
            "household/memory/küche.md": "# Küche\n\nDer Herd braucht einen Kundendienst.\n",
        },
        remove=("bob/memory/garden.md",),
    )
    assert memex.refresh() is True
    assert memex.index.get_meta("head") == new_head
    assert paths(memex, "workshop") == ["alice/memory/bike-repair.md"]
    assert paths(memex, "tube") == []
    assert paths(memex, "quokka") == ["alice/memory/new.md"]
    assert paths(memex, "dhcp") == ["alice/wiki/network.md"]
    assert paths(memex, "kundendienst") == ["household/memory/küche.md"]
    assert "bob/memory/garden.md" not in memex.index.paths()
    assert "alice/wiki/networking.md" not in memex.index.paths()
    # The untouched note kept its rows; the edited one got new ones.
    assert memex.index.chunk_ids_for("alice/memory/zebra.md") == untouched
    assert not set(memex.index.chunk_ids_for("alice/memory/bike-repair.md")) & set(changed_before)


def test_refresh_without_news_changes_nothing(memex: Memex) -> None:
    before = memex.index.chunk_ids_for("alice/memory/zebra.md")
    assert memex.refresh() is True
    assert memex.index.chunk_ids_for("alice/memory/zebra.md") == before


def test_failed_fetch_keeps_serving_the_last_state(
    memex: Memex, origin: Origin, caplog: pytest.LogCaptureFixture
) -> None:
    head = memex.index.get_meta("head")
    git(memex.root, "remote", "set-url", "origin", str(origin.bare.parent / "gone.git"))
    origin.commit("not reachable", {"alice/memory/later.md": "# Later\n\nwombat\n"})
    with caplog.at_level(logging.WARNING, logger="memex_mcp.service"):
        assert memex.refresh() is False
    assert "serving the last state" in caplog.text
    assert memex.index.get_meta("head") == head
    assert paths(memex, "tube") == ["alice/memory/bike-repair.md"]
    assert paths(memex, "wombat") == []


def test_restart_catches_up_incrementally(tmp_path: Path, origin: Origin) -> None:
    config = make_config(tmp_path, origin)
    first = Memex.from_config(config, embedder=None)
    first.prepare()
    untouched = first.index.chunk_ids_for("alice/memory/zebra.md")
    first.index.close()
    origin.commit("while down", {"alice/memory/new.md": "# New\n\nplatypus\n"})
    git(Path(config.repo.path), "pull", "-q", "--ff-only")
    second = Memex.from_config(config, embedder=None)
    second.prepare()
    try:
        assert second.index.fresh is False
        assert second.index.chunk_ids_for("alice/memory/zebra.md") == untouched
        assert paths(second, "platypus") == ["alice/memory/new.md"]
    finally:
        second.index.close()


def _reopen(tmp_path: Path, origin: Origin) -> Memex:
    memex = Memex.from_config(make_config(tmp_path, origin), embedder=None)
    memex.prepare()
    return memex


def test_schema_version_change_rebuilds(memex: Memex, tmp_path: Path, origin: Origin) -> None:
    memex.index.set_meta("schema_version", str(SCHEMA_VERSION - 1))
    memex.index.close()
    reopened = _reopen(tmp_path, origin)
    try:
        assert reopened.index.fresh is True
        assert reopened.index.get_meta("schema_version") == str(SCHEMA_VERSION)
        assert "alice/memory/bike-repair.md" in paths(reopened, "degreaser")
    finally:
        reopened.index.close()


def test_missing_database_rebuilds(memex: Memex, tmp_path: Path, origin: Origin) -> None:
    memex.index.close()
    db = Path(memex.config.index.path)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    reopened = _reopen(tmp_path, origin)
    try:
        assert reopened.index.fresh is True
        assert "alice/memory/bike-repair.md" in paths(reopened, "degreaser")
    finally:
        reopened.index.close()


def test_corrupt_database_rebuilds(memex: Memex, tmp_path: Path, origin: Origin) -> None:
    memex.index.close()
    db = Path(memex.config.index.path)
    for suffix in ("-wal", "-shm"):
        Path(f"{db}{suffix}").unlink(missing_ok=True)
    db.write_bytes(b"this is not a database" * 100)
    reopened = _reopen(tmp_path, origin)
    try:
        assert reopened.index.fresh is True
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    finally:
        reopened.index.close()


def test_new_chunks_get_vectors_and_a_failed_round_resumes(tmp_path: Path, origin: Origin) -> None:
    embedder = FakeEmbedder()
    memex = Memex.from_config(make_config(tmp_path, origin), embedder=embedder)
    memex.prepare()
    try:
        assert memex.backfill() > 0
        assert memex.index.missing_vectors(10) == []
        origin.commit("new", {"household/memory/plants.md": "# Plants\n\nwatering plan\n"})
        memex.embedder = FailingEmbedder()
        memex.refresh()
        assert memex.backfill() == 0
        assert len(memex.index.missing_vectors(10)) > 0
        memex.embedder = embedder
        assert memex.backfill() > 0
        assert memex.index.missing_vectors(10) == []
    finally:
        memex.index.close()


def test_a_malformed_embedder_answer_pauses_embedding(tmp_path: Path, origin: Origin) -> None:
    embedder = malformed_ollama()
    memex = Memex.from_config(make_config(tmp_path, origin), embedder=embedder)
    memex.prepare()
    try:
        assert memex.backfill() == 0
        assert len(memex.index.missing_vectors(10)) > 0
    finally:
        memex.index.close()
        embedder.close()


def test_a_different_embedding_model_drops_the_vectors(tmp_path: Path, origin: Origin) -> None:
    memex = Memex.from_config(make_config(tmp_path, origin), embedder=FakeEmbedder())
    memex.prepare()
    memex.backfill()
    memex.index.close()
    other = FakeEmbedder()
    other.model = "another-model"
    second = Memex.from_config(make_config(tmp_path, origin), embedder=other)
    second.prepare()
    try:
        assert len(second.index.missing_vectors(1000)) > 0
    finally:
        second.index.close()


def test_no_clone_and_no_remote_is_an_error(tmp_path: Path) -> None:
    memex = Memex.from_config(make_config(tmp_path), embedder=None)
    with pytest.raises(RuntimeError, match="no \\[repo\\] remote"):
        memex.prepare()


def test_git_errors_hide_credentials(tmp_path: Path) -> None:
    assert redact("unable to access 'https://bot:s3cret@git.example.org/memex.git/'") == (
        "unable to access 'https://***@git.example.org/memex.git/'"
    )
    repo = GitRepo(tmp_path / "clone", "main", 10.0)
    with pytest.raises(GitError) as error:
        repo.clone("https://bot:s3cret@127.0.0.1:9/memex.git")
    assert "s3cret" not in str(error.value)


def _commit_undecodable_name(origin: Origin) -> str:
    """A writer commits a note whose file name is not UTF-8, next to a good one."""
    # os.fsdecode turns the byte 0xff into a surrogate, the way os.walk reports it.
    bad = origin.work / "alice" / "memory" / os.fsdecode(b"\xff-latin1.md")
    bad.write_bytes(b"# Bad name\n\nhidden marmot\n")
    return origin.commit("bad name", {"alice/memory/good.md": "# Good\n\nvisible ocelot\n"})


def test_a_non_utf8_file_name_is_skipped_on_refresh_and_restart(
    memex: Memex, origin: Origin, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    head = _commit_undecodable_name(origin)
    with caplog.at_level(logging.WARNING, logger="memex_mcp.index"):
        assert memex.refresh() is True
    assert memex.index.get_meta("head") == head
    assert paths(memex, "ocelot") == ["alice/memory/good.md"]
    assert paths(memex, "marmot") == []
    assert "not valid UTF-8" in caplog.text
    memex.index.close()
    restarted = Memex.from_config(make_config(tmp_path, origin), embedder=None)
    restarted.prepare()
    try:
        assert paths(restarted, "ocelot") == ["alice/memory/good.md"]
    finally:
        restarted.index.close()


def test_a_non_utf8_file_name_is_skipped_on_a_fresh_index(
    tmp_path: Path, origin: Origin, caplog: pytest.LogCaptureFixture
) -> None:
    _commit_undecodable_name(origin)
    memex = Memex.from_config(make_config(tmp_path, origin), embedder=None)
    with caplog.at_level(logging.WARNING, logger="memex_mcp.index"):
        memex.prepare()
    try:
        # Recognised by its name, before anything tries to read or store it.
        assert "not valid UTF-8" in caplog.text
        assert memex.index.fresh is True
        assert paths(memex, "ocelot") == ["alice/memory/good.md"]
        assert paths(memex, "degreaser")[0] == "alice/memory/bike-repair.md"
    finally:
        memex.index.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files without read permission")
def test_an_unreadable_note_does_not_stop_indexing(
    memex: Memex, caplog: pytest.LogCaptureFixture
) -> None:
    locked = memex.root / "alice/memory/zebra.md"
    locked.chmod(0)
    try:
        with caplog.at_level(logging.WARNING, logger="memex_mcp.index"):
            assert memex.index.rebuild(memex.root) > 0
        assert "alice/memory/zebra.md" not in memex.index.paths()
        assert paths(memex, "degreaser")[0] == "alice/memory/bike-repair.md"
        assert "skipping alice/memory/zebra.md" in caplog.text
    finally:
        locked.chmod(0o644)


POISON = "POISON"


def _fail_after_first_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make chunking of a poisoned note fail after its first chunk was written."""
    real = chunk_note

    def flaky(text: str, max_chars: int) -> Iterator[Chunk]:
        chunks = real(text, max_chars)
        if POISON not in text:
            yield from chunks
            return
        yield chunks[0]
        raise RuntimeError("chunking failed half-way")

    monkeypatch.setattr(index_module, "chunk_note", flaky)


POISONED = "# Poisoned\n\n" + f"{POISON} first chunk\n\n" + "more text " * 400


def test_first_note_of_an_area_failing_half_way_does_not_stop_prepare(
    tmp_path: Path, origin: Origin, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sorted first in alice's area, so its failure comes before any other note there.
    origin.commit("poison", {"alice/0-poisoned.md": POISONED})
    _fail_after_first_chunk(monkeypatch)
    memex = Memex.from_config(make_config(tmp_path, origin), embedder=None)
    memex.prepare()
    try:
        assert "alice/0-poisoned.md" not in memex.index.paths()
        assert paths(memex, "degreaser")[0] == "alice/memory/bike-repair.md"
        assert paths(memex, POISON) == []
    finally:
        memex.index.close()


def test_first_note_of_a_new_area_failing_half_way_does_not_stop_refresh(
    tmp_path: Path, origin: Origin, monkeypatch: pytest.MonkeyPatch
) -> None:
    # dave is configured but has no note yet; this commit brings the first ones.
    memex = Memex.from_config(make_config(tmp_path, origin, users={"dave": "dave"}), None)
    memex.prepare()
    try:
        origin.commit(
            "dave arrives",
            {"dave/0-poisoned.md": POISONED, "dave/memory/later.md": "# Later\n\nquokka\n"},
        )
        _fail_after_first_chunk(monkeypatch)
        assert memex.refresh() is True
        dave = Identity("dave", frozenset({"memex"}))
        hits = memex.search(dave, "quokka")["hits"]
        assert [h["path"] for h in hits] == ["dave/memory/later.md"]
    finally:
        memex.index.close()


def test_only_configured_areas_are_indexed(
    tmp_path: Path, origin: Origin, caplog: pytest.LogCaptureFixture
) -> None:
    """A pusher's thousands of top-level folders neither get indexed nor slow the build."""
    many = {f"folder{i:04d}/note.md": "# n\n\nkiwi\n" for i in range(3000)}
    origin.commit("many folders", many)
    memex = Memex.from_config(make_config(tmp_path, origin), embedder=None)
    started = time.perf_counter()
    memex.prepare()
    elapsed = time.perf_counter() - started
    try:
        assert not [p for p in memex.index.paths() if p.startswith("folder")]
        assert elapsed < 10.0
        origin.commit("one more", {"folder9999/note.md": "# n\n\nkiwi\n"})
        with caplog.at_level(logging.WARNING, logger="memex_mcp.index"):
            assert memex.refresh() is True
        assert "folder9999/note.md" not in memex.index.paths()
        # Passed over as not indexable, not attempted and then skipped as a failure.
        assert "folder9999" not in caplog.text
    finally:
        memex.index.close()


def test_a_changed_area_set_rebuilds(tmp_path: Path, origin: Origin) -> None:
    # dave's notes are in the clone before dave is configured: no new commit
    # will bring them, only the changed configuration can.
    origin.commit("dave", {"dave/memory/note.md": "# Dave\n\nwombat\n"})
    first = Memex.from_config(make_config(tmp_path, origin), embedder=None)
    first.prepare()
    assert not [p for p in first.index.paths() if p.startswith("dave/")]
    first.index.close()
    second = Memex.from_config(make_config(tmp_path, origin, users={"dave": "dave"}), None)
    second.prepare()
    try:
        dave = Identity("dave", frozenset({"memex"}))
        assert [h["path"] for h in second.search(dave, "wombat")["hits"]] == ["dave/memory/note.md"]
    finally:
        second.index.close()
