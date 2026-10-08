"""What the caller cannot see answers exactly like what does not exist.

For every tool: a foreign note, folder or area that exists gets the same
exception type and message as one that does not, the clone is not touched
for either, and no message names an area, a configured value or a path on
the server.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from memex_mcp.embed import Embedder
from memex_mcp.rights import (
    ACCESS_DENIED,
    AREA_NOT_FOUND,
    FOLDER_NOT_FOUND,
    NOTE_NOT_FOUND,
    AccessDenied,
    Identity,
    NotFound,
)
from memex_mcp.service import Memex
from tests.conftest import (
    ALICE,
    CAROL,
    MALLORY,
    MARKER,
    STRANGER,
    FakeEmbedder,
    Origin,
    git,
    make_config,
    write_notes,
)

FOREIGN_AREAS = ("bob", "carol")


def refusal(call: Callable[[], object]) -> tuple[type[BaseException], str]:
    with pytest.raises((AccessDenied, NotFound)) as caught:
        call()
    return caught.type, str(caught.value)


def assert_reveals_nothing(message: str) -> None:
    assert "/" not in message  # no path of any kind, the clone's or the index's
    for name in (*FOREIGN_AREAS, "household", "alice", "mallory"):
        assert name not in message


@pytest.mark.parametrize(
    "path",
    [
        "bob/memory/zebra.md",  # foreign, exists
        "bob/memory/no-such-note.md",  # foreign, missing
        "nobody/memory/zebra.md",  # no such area
        "alice/memory/no-such-note.md",  # own area, missing
        "alice/no-such-folder/zebra.md",  # own area, missing folder
    ],
)
def test_read_answers_alike(memex: Memex, path: str) -> None:
    kind, message = refusal(lambda: memex.read(ALICE, path))
    assert (kind, message) == (NotFound, NOTE_NOT_FOUND)
    assert_reveals_nothing(message)


@pytest.mark.parametrize(
    ("area", "folder"),
    [
        ("bob", None),  # foreign, exists
        ("bob", "memory"),  # foreign folder, exists
        ("bob", "no-such-folder"),  # foreign folder, missing
        ("nobody", None),  # no such area
        ("alice", "no-such-folder"),  # own area, missing folder
        ("alice", "memory/zebra.md"),  # own area, a note, not a folder
    ],
)
def test_list_answers_alike(memex: Memex, area: str, folder: str | None) -> None:
    kind, message = refusal(lambda: memex.list(ALICE, area, folder))
    assert (kind, message) == (NotFound, FOLDER_NOT_FOUND)
    assert_reveals_nothing(message)


@pytest.mark.parametrize("area", ["bob", "carol", "nobody", ".git", ""])
def test_search_area_filter_answers_alike(memex: Memex, area: str) -> None:
    kind, message = refusal(lambda: memex.search(ALICE, MARKER, area=area))
    assert (kind, message) == (NotFound, AREA_NOT_FOUND)


def test_symlinks_to_existing_and_missing_foreign_notes_answer_alike(memex: Memex) -> None:
    (memex.root / "alice/memory/to-existing.md").symlink_to("../../bob/memory/zebra.md")
    (memex.root / "alice/memory/to-missing.md").symlink_to("../../bob/memory/nothing.md")
    existing = refusal(lambda: memex.read(ALICE, "alice/memory/to-existing.md"))
    missing = refusal(lambda: memex.read(ALICE, "alice/memory/to-missing.md"))
    assert existing == missing == (NotFound, NOTE_NOT_FOUND)


def test_foreign_areas_are_refused_without_touching_the_clone(
    memex: Memex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No filesystem work for a foreign area, so timing cannot tell existing from missing."""
    touched: list[str] = []
    original = Path.resolve

    def spy(self: Path, strict: bool = False) -> Path:
        touched.append(str(self))
        return original(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", spy)
    for call in (
        lambda: memex.read(ALICE, "bob/memory/zebra.md"),
        lambda: memex.read(ALICE, "bob/memory/no-such-note.md"),
        lambda: memex.list(ALICE, "bob", "memory"),
        lambda: memex.list(ALICE, "nobody"),
        lambda: memex.search(ALICE, MARKER, area="bob"),
    ):
        refusal(call)
    assert touched == []


def test_search_reveals_no_foreign_area(memex: Memex) -> None:
    """Hits, and nothing else in the result, come from the caller's areas only."""
    result = memex.search(CAROL, MARKER, limit=50)
    assert set(result) <= {"semantic", "notice", "hits"}
    assert {hit["area"] for hit in result["hits"]} == {"carol"}
    rendered = repr(result)
    for name in ("bob", "household", "alice"):
        assert name not in rendered


def test_no_access_answers_with_one_fixed_text(memex: Memex) -> None:
    """Missing group and missing area mapping read the same: no hint at the configuration."""
    unmapped = refusal(lambda: memex.read(STRANGER, "household/memory/zebra.md"))
    no_group = refusal(lambda: memex.read(MALLORY, "mallory/memory/zebra.md"))
    assert unmapped == no_group == (AccessDenied, ACCESS_DENIED)
    assert "memex" not in ACCESS_DENIED and "stranger" not in ACCESS_DENIED


def _world(base: Path, foreign: dict[str, str], embedder: Embedder | None) -> str:
    """One memex where only bob's area differs; alice's search result, serialised."""
    base.mkdir()
    bare, work = base / "origin.git", base / "writer"
    git(base, "init", "-q", "--bare", "--initial-branch=main", str(bare))
    git(base, "clone", "-q", str(bare), str(work))
    git(work, "checkout", "-q", "-B", "main")
    own = {
        "alice/memory/one.md": "# One\n\nkestrel filler words\n",
        "alice/memory/two.md": "# Two\n\nzzqqx filler words\n",
        "alice/memory/three.md": "# Three\n\nkestrel zzqqx and more filler\n",
        "household/memory/shared.md": "# Shared\n\nkestrel in the shared area\n",
    }
    write_notes(work, {**own, **foreign})
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "world")
    git(work, "push", "-q", "origin", "main")
    memex = Memex.from_config(make_config(base, Origin(bare, work)), embedder=embedder)
    memex.prepare()
    try:
        memex.backfill()
        caller = Identity("alice", frozenset({"memex", "household"}))
        return json.dumps(memex.search(caller, "kestrel zzqqx filler", limit=50), sort_keys=True)
    finally:
        memex.index.close()


@pytest.mark.parametrize("embedder", [None, FakeEmbedder()], ids=["keyword", "hybrid"])
def test_ranking_does_not_depend_on_foreign_areas(
    tmp_path: Path, embedder: Embedder | None
) -> None:
    """Two worlds that differ only in bob's area give alice byte-identical results."""
    kestrels = {f"bob/memory/k{i}.md": f"# K{i}\n\nkestrel kestrel kestrel\n" for i in range(8)}
    zzqqx = {"bob/memory/z.md": "# Z\n\nzzqqx\n", "bob/aaa/first.md": "# A\n\nfiller\n"}
    first = _world(tmp_path / "kestrels", kestrels, embedder)
    second = _world(tmp_path / "zzqqx", zzqqx, embedder)
    assert first == second


def test_no_access_messages_name_no_area(memex: Memex) -> None:
    for call in (
        lambda: memex.read(STRANGER, "household/memory/zebra.md"),
        lambda: memex.list(STRANGER, "household"),
        lambda: memex.search(STRANGER, MARKER),
    ):
        kind, message = refusal(call)
        assert kind is AccessDenied
        assert_reveals_nothing(message)
