"""Path attacks: nothing outside the caller's areas, nothing hidden, nothing but notes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from memex_mcp.rights import AccessDenied, Identity, NotFound, split_path
from memex_mcp.service import Memex
from tests.conftest import ALICE, MARKER, Origin, make_config

ATTACKS = [
    "../etc/passwd",
    "../bob/memory/zebra.md",
    "alice/../bob/memory/zebra.md",
    "alice/./../bob/memory/zebra.md",
    "alice/memory/../../bob/memory/zebra.md",
    "/etc/passwd",
    "/alice/memory/zebra.md",
    "alice/.git/config",
    ".git/config",
    "alice/.obsidian/workspace.md",
    "alice\\memory\\zebra.md",
    "alice/memory/zebra.md\x00.md",
    "alice/memory/zebra.txt",
    "alice/memory",
    "alice",
    "",
    ".",
    "..",
    "README.md",
    "bob/memory/zebra.md",
]


@pytest.mark.parametrize("path", ATTACKS)
def test_read_refuses_attacks(memex: Memex, path: str) -> None:
    with pytest.raises((AccessDenied, NotFound)):
        memex.read(ALICE, path)


@pytest.mark.parametrize(
    "folder",
    ["..", "../bob", "memory/../../bob", "/etc", ".obsidian", ".git", "memory/zebra.md"],
)
def test_list_refuses_attacks(memex: Memex, folder: str) -> None:
    with pytest.raises((AccessDenied, NotFound)):
        memex.list(ALICE, "alice", folder)


def test_normalisation_drops_empty_and_dot_segments() -> None:
    assert split_path("alice//memory/./zebra.md") == ("alice", "memory", "zebra.md")


def test_redundant_segments_still_read(memex: Memex) -> None:
    note = memex.read(ALICE, "alice//memory/./zebra.md")
    assert note["path"] == "alice/memory/zebra.md"


def _link(memex: Memex, relative: str, target: str | Path) -> None:
    (memex.root / relative).symlink_to(target)


def test_symlink_to_a_foreign_area_is_refused(memex: Memex) -> None:
    _link(memex, "alice/memory/leak.md", "../../bob/memory/zebra.md")
    with pytest.raises(NotFound):
        memex.read(ALICE, "alice/memory/leak.md")


def test_symlink_out_of_the_repository_is_refused(memex: Memex, tmp_path: Path) -> None:
    secret = tmp_path / "secret.md"
    secret.write_text("# secret\n", encoding="utf-8")
    _link(memex, "alice/memory/outside.md", secret)
    with pytest.raises(NotFound):
        memex.read(ALICE, "alice/memory/outside.md")


def test_symlinked_folder_to_a_foreign_area_is_refused(memex: Memex) -> None:
    _link(memex, "alice/shared", "../bob")
    with pytest.raises(NotFound):
        memex.read(ALICE, "alice/shared/memory/zebra.md")
    with pytest.raises(NotFound):
        memex.list(ALICE, "alice", "shared")
    assert "shared" not in {e["name"] for e in memex.list(ALICE, "alice")["entries"]}


def test_symlink_inside_the_area_onto_a_hidden_file_is_refused(memex: Memex) -> None:
    _link(memex, "alice/memory/state.md", "../.obsidian/workspace.md")
    with pytest.raises(NotFound):
        memex.read(ALICE, "alice/memory/state.md")


def test_symlink_onto_a_non_note_is_refused(memex: Memex) -> None:
    (memex.root / "alice/memory/raw.txt").write_text("not a note", encoding="utf-8")
    _link(memex, "alice/memory/raw.md", "raw.txt")
    with pytest.raises(NotFound):
        memex.read(ALICE, "alice/memory/raw.md")


def test_symlink_staying_inside_the_area_reads(memex: Memex) -> None:
    _link(memex, "alice/memory/alias.md", "zebra.md")
    assert MARKER in memex.read(ALICE, "alice/memory/alias.md")["content"]


def test_an_area_directory_that_is_a_symlink_is_refused(tmp_path: Path, origin: Origin) -> None:
    config = make_config(tmp_path, origin, users={"eve": "evil"})
    memex = Memex.from_config(config, embedder=None)
    memex.prepare()
    try:
        (memex.root / "evil").symlink_to("bob")
        eve = Identity("eve", frozenset({"memex"}))
        with pytest.raises(NotFound):
            memex.read(eve, "evil/memory/zebra.md")
        with pytest.raises(NotFound):
            memex.list(eve, "evil")
    finally:
        memex.index.close()


def test_listing_shows_only_notes_and_visible_folders(memex: Memex) -> None:
    (memex.root / "alice/memory/image.png").write_bytes(b"\x89PNG")
    listing = memex.list(ALICE, "alice")
    assert {(e["name"], e["type"]) for e in listing["entries"]} == {
        ("memory", "folder"),
        ("wiki", "folder"),
    }
    names = {e["name"] for e in memex.list(ALICE, "alice", "memory")["entries"]}
    assert "image.png" not in names
    assert {"zebra.md", "archive"} <= names


def test_missing_note_in_an_allowed_area_is_not_found(memex: Memex) -> None:
    with pytest.raises(NotFound):
        memex.read(ALICE, "alice/memory/nothing.md")


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("..", "'..'"),
        ("a/../b", "'..'"),
        ("a/..", "'..'"),
        ("/abs", "absolute"),
        (".git", "hidden"),
        ("alice/.obsidian/x.md", "hidden"),
        ("a\\b", "forbidden character"),
        ("a\x00b", "forbidden character"),
    ],
)
def test_split_path_refuses_on_its_own(path: str, reason: str) -> None:
    # Checked before the filesystem is consulted, so even a path whose
    # resolution would land inside the area is refused.
    with pytest.raises(AccessDenied, match=reason):
        split_path(path)


def test_index_filters_areas_in_the_query(memex: Memex) -> None:
    rankings = memex.index.keyword_ranked(MARKER, frozenset({"household"}), 50)
    ids = [chunk_id for ranking in rankings for chunk_id in ranking]
    assert ids
    assert {row.area for row in memex.index.chunks(ids).values()} == {"household"}


def test_search_drops_rows_of_foreign_areas_even_if_the_index_returns_them(
    memex: Memex, monkeypatch: pytest.MonkeyPatch
) -> None:
    every = memex.index.keyword_ranked(
        MARKER, frozenset({"alice", "bob", "household", "carol"}), 50
    )
    monkeypatch.setattr(memex.index, "keyword_ranked", lambda *_args: every)
    result = memex.search(ALICE, MARKER, limit=50)
    assert {hit["area"] for hit in result["hits"]} == {"alice", "household"}


def test_listing_skips_names_that_are_not_utf8(memex: Memex) -> None:
    # os.fsdecode turns the byte 0xff into a surrogate, the way os.walk reports it.
    (memex.root / "alice" / "memory" / os.fsdecode(b"\xff.md")).write_bytes(b"# x\n")
    listing = memex.list(ALICE, "alice", "memory")
    # What reaches the client must encode as UTF-8.
    json.dumps(listing, ensure_ascii=False).encode("utf-8")
    assert "zebra.md" in {e["name"] for e in listing["entries"]}
