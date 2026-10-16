"""Writing through the server: rights, path rules, guards, indexes, commits."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

import memex_mcp.rights as rights_module
from memex_mcp.config import build_config, load_config
from memex_mcp.notes import CHANGED_SINCE_READ, version_of
from memex_mcp.rights import (
    ACCESS_DENIED,
    FORBIDDEN_CHARACTER,
    INDEX_FILE,
    NOTE_NOT_FOUND,
    PROTECTED_FILE,
    WORD_START_AT,
    WRITE_OUTSIDE,
    AccessDenied,
    Identity,
    InvalidRequest,
    NotFound,
)
from memex_mcp.service import EXISTS, NOT_SEARCHABLE, IndexLine, Memex, WriteResult
from tests.conftest import (
    ALICE,
    BOB,
    CAROL,
    GUEST,
    MALLORY,
    STRANGER,
    Origin,
    git,
    make_config,
    raw_config,
)
from tests.test_config import EXAMPLE
from tests.test_paths import ATTACKS

# Topics in alice's area, with indexes in both formats.
TOPICS: dict[str, str] = {
    "alice/memory/homelab/MEMORY.md": (
        "# homelab\n\n## Services\n\n- [dns.md](dns.md) — DNS\n\n"
        "## Hardware\n\n- [ups.md](ups.md) — UPS\n"
    ),
    "alice/memory/homelab/dns.md": "# DNS\n\nThe resolver runs on the router.\n",
    "alice/memory/homelab/ups.md": "# UPS\n\nThe battery lasts ten minutes.\n",
    "alice/wiki/_master-index.md": "# Wiki\n\n- [[wiki/networking/_index|Networking]]\n",
    "alice/wiki/networking/_index.md": "# Networking\n\n## Basics\n- [[dhcp]] — leases\n",
    "alice/wiki/networking/dhcp.md": "# DHCP\n\nLeases last a day.\n",
    "alice/memory/CLAUDE.md": "# Instructions\n",
}

LINE = "what the note holds"


class Remote:
    """The origin as the server's pushes reach it: its head and how many pushes arrived."""

    def __init__(self, origin: Origin) -> None:
        self.origin: Origin = origin
        self.log: Path = origin.bare.parent / "pushes.log"
        hook = origin.bare / "hooks" / "post-receive"
        hook.write_text(f'#!/bin/sh\necho push >> "{self.log}"\n', encoding="utf-8")
        hook.chmod(0o755)

    def head(self) -> str:
        return git(self.origin.bare, "rev-parse", "main").strip()

    def refs(self) -> str:
        return git(self.origin.bare, "for-each-ref")

    def pushes(self) -> int:
        if not self.log.exists():
            return 0
        return len(self.log.read_text(encoding="utf-8").splitlines())

    def show(self, path: str) -> str:
        return git(self.origin.bare, "show", f"main:{path}")

    def log_of(self, fmt: str, count: int = 1) -> str:
        return git(self.origin.bare, "log", f"-{count}", f"--format={fmt}", "main").rstrip("\n")


@pytest.fixture
def remote(origin: Origin) -> Remote:
    return Remote(origin)


@pytest.fixture
def topics(memex: Memex, origin: Origin, remote: Remote) -> Memex:
    """``memex`` once the topics above are on the origin and in the clone."""
    origin.commit("topics", TOPICS)
    assert memex.refresh() is True
    del remote
    return memex


def clone_file(memex: Memex, path: str) -> str:
    return (memex.root / path).read_text(encoding="utf-8")


def create(
    memex: Memex, identity: Identity, path: str, content: str = "# New\n\nText.\n"
) -> WriteResult:
    return memex.create(identity, path, content, IndexLine("Notes", LINE))


# -- rights ------------------------------------------------------------

AREAS = ("alice", "bob", "household", "carol")
IDENTITIES = (ALICE, BOB, CAROL, MALLORY, STRANGER, GUEST)
WRITES: dict[str, frozenset[str] | None] = {
    "alice": frozenset({"alice", "household"}),
    "bob": frozenset({"bob", "household"}),
    "carol": frozenset({"carol"}),
    "mallory": None,
    "stranger": None,
    "guest": frozenset({"household"}),
}
CASES = [(identity, area) for identity in IDENTITIES for area in AREAS]


@pytest.mark.parametrize("case", CASES, ids=[f"{i.username}-{a}" for i, a in CASES])
def test_create_matrix(memex: Memex, remote: Remote, case: tuple[Identity, str]) -> None:
    identity, area = case
    path = f"{area}/memory/from-{identity.username}.md"
    allowed = WRITES[identity.username]
    head, pushes = remote.head(), remote.pushes()
    if allowed is not None and area in allowed:
        result = create(memex, identity, path)
        assert result["pushed"] is True
        assert remote.head() == result["commit"]
        assert "Text." in remote.show(path)
        return
    refusal = AccessDenied if allowed is None else NotFound
    with pytest.raises(refusal):
        create(memex, identity, path)
    assert (remote.head(), remote.pushes()) == (head, pushes)
    assert not (memex.root / path).exists()


WRITE_CALLS: dict[str, Callable[[Memex, Identity, str], object]] = {
    "create": lambda m, i, p: m.create(i, p, "# x\n", IndexLine("Notes", LINE)),
    "append": lambda m, i, p: m.append(i, p, "more"),
    "edit": lambda m, i, p: m.edit(i, p, "zebracode", "changed"),
    "delete": lambda m, i, p: m.delete(i, p, "0" * 40),
}
EXISTING_CALLS = {name: call for name, call in WRITE_CALLS.items() if name != "create"}


def refusal(call: Callable[[], object]) -> tuple[type[BaseException], str]:
    with pytest.raises((AccessDenied, NotFound, InvalidRequest)) as caught:
        call()
    return caught.type, str(caught.value)


@pytest.mark.parametrize("tool", list(EXISTING_CALLS))
def test_foreign_and_missing_notes_answer_alike(memex: Memex, remote: Remote, tool: str) -> None:
    call = EXISTING_CALLS[tool]
    head = remote.head()
    answers = {
        refusal(lambda path=path: call(memex, ALICE, path))
        for path in (
            "bob/memory/zebra.md",  # foreign, exists
            "bob/memory/no-such-note.md",  # foreign, missing
            "nobody/memory/zebra.md",  # no such area
            "alice/memory/no-such-note.md",  # own area, missing
            "alice/memory/no-such-folder/zebra.md",  # own area, missing folder
        )
    }
    assert answers == {(NotFound, NOTE_NOT_FOUND)}
    assert remote.head() == head


def test_create_in_a_foreign_area_answers_like_a_missing_area(memex: Memex) -> None:
    answers = {
        refusal(lambda path=path: create(memex, ALICE, path))
        for path in ("bob/memory/zebra.md", "bob/memory/new.md", "nobody/memory/new.md")
    }
    assert answers == {(NotFound, NOTE_NOT_FOUND)}


@pytest.mark.parametrize("tool", list(WRITE_CALLS))
def test_foreign_writes_are_refused_without_touching_the_clone(
    memex: Memex, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    touched: list[str] = []
    for name in ("resolve", "stat", "lstat"):
        original = cast("Callable[..., object]", getattr(Path, name))

        def spy(
            self: Path, *args: object, _original: Callable[..., object] = original, **kw: object
        ) -> object:
            touched.append(str(self))
            return _original(self, *args, **kw)

        monkeypatch.setattr(Path, name, spy)
    for path in ("bob/memory/zebra.md", "bob/memory/missing.md", "nobody/memory/x.md"):
        refusal(lambda path=path: WRITE_CALLS[tool](memex, ALICE, path))
    assert touched == []


@pytest.mark.parametrize("tool", list(WRITE_CALLS))
def test_no_access_gets_the_one_fixed_text(memex: Memex, tool: str) -> None:
    for identity in (MALLORY, STRANGER):
        kind, message = refusal(
            lambda i=identity: WRITE_CALLS[tool](memex, i, "household/memory/x.md")
        )
        assert (kind, message) == (AccessDenied, ACCESS_DENIED)


def test_household_needs_the_group_to_write_even_for_the_owner(memex: Memex) -> None:
    alice_alone = Identity("alice", frozenset({"memex"}), "claude-code")
    with pytest.raises(NotFound):
        create(memex, alice_alone, "household/memory/new.md")
    assert create(memex, alice_alone, "alice/memory/new.md")["pushed"] is True


# -- path rules --------------------------------------------------------


@pytest.mark.parametrize("tool", list(WRITE_CALLS))
@pytest.mark.parametrize("path", [*ATTACKS, "alice/memory/a\nb.md", "alice/memory/a\rb.md"])
def test_writes_refuse_path_attacks(memex: Memex, remote: Remote, tool: str, path: str) -> None:
    head = remote.head()
    with pytest.raises((AccessDenied, NotFound)):
        WRITE_CALLS[tool](memex, ALICE, path)
    assert remote.head() == head


@pytest.mark.parametrize("char", ["\n", "\r", "\t", "\x1b", "\u2028"], ids=repr)
def test_a_control_character_or_line_break_in_the_path_is_refused(memex: Memex, char: str) -> None:
    with pytest.raises(AccessDenied, match=re.escape(FORBIDDEN_CHARACTER)):
        create(memex, ALICE, f"alice/memory/a{char}b.md")


@pytest.mark.parametrize(
    "path", ["alice/notes.md", "alice/archive/x.md", "alice/memory.md", "household/x.md"]
)
@pytest.mark.parametrize("tool", list(WRITE_CALLS))
def test_writes_outside_memory_and_wiki_are_refused(
    memex: Memex, origin: Origin, tool: str, path: str
) -> None:
    origin.commit("outside", {path: "# Outside\n\nzebracode\n"})
    memex.refresh()
    with pytest.raises(AccessDenied, match=re.escape(WRITE_OUTSIDE)):
        WRITE_CALLS[tool](memex, ALICE, path)
    assert clone_file(memex, path) == "# Outside\n\nzebracode\n"


def test_a_symlink_on_the_path_is_refused_for_writing(memex: Memex) -> None:
    (memex.root / "alice/memory/elsewhere").symlink_to("../../bob/memory")
    (memex.root / "alice/memory/alias.md").symlink_to("zebra.md")
    with pytest.raises(NotFound):
        create(memex, ALICE, "alice/memory/elsewhere/new.md")
    with pytest.raises(NotFound):
        memex.append(ALICE, "alice/memory/alias.md", "more")
    assert not (memex.root / "bob/memory/new.md").exists()


def test_an_area_directory_that_is_a_symlink_is_refused(tmp_path: Path, origin: Origin) -> None:
    memex = Memex.from_config(make_config(tmp_path, origin, users={"eve": "evil"}), None)
    memex.prepare()
    try:
        (memex.root / "evil").symlink_to("bob")
        eve = Identity("eve", frozenset({"memex"}), "claude-code")
        with pytest.raises(NotFound):
            create(memex, eve, "evil/memory/new.md")
        assert not (memex.root / "bob/memory/new.md").exists()
    finally:
        memex.index.close()


# -- protected files ---------------------------------------------------


@pytest.mark.parametrize("tool", list(WRITE_CALLS))
@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("alice/memory/CLAUDE.md", PROTECTED_FILE),
        ("alice/wiki/networking/AGENTS.md", PROTECTED_FILE),
        ("alice/memory/homelab/CLAUDE.md", PROTECTED_FILE),
        ("alice/memory/homelab/MEMORY.md", INDEX_FILE),
        ("alice/wiki/networking/_index.md", INDEX_FILE),
        ("alice/wiki/_master-index.md", INDEX_FILE),
        ("alice/memory/MEMORY.md", INDEX_FILE),
    ],
)
def test_protected_and_index_files_are_never_written(
    topics: Memex, remote: Remote, tool: str, path: str, message: str
) -> None:
    before = (topics.root / path).read_bytes() if (topics.root / path).exists() else None
    head = remote.head()
    with pytest.raises(AccessDenied, match=re.escape(message)):
        WRITE_CALLS[tool](topics, ALICE, path)
    after = (topics.root / path).read_bytes() if (topics.root / path).exists() else None
    assert after == before
    assert remote.head() == head


# -- the operations ----------------------------------------------------


def test_create_refuses_an_existing_note(memex: Memex, remote: Remote) -> None:
    head = remote.head()
    with pytest.raises(InvalidRequest, match=re.escape(EXISTS)):
        create(memex, ALICE, "alice/memory/zebra.md")
    assert "zebracode" in clone_file(memex, "alice/memory/zebra.md")
    assert remote.head() == head


def test_append_adds_at_the_end_and_never_creates(memex: Memex, remote: Remote) -> None:
    result = memex.append(ALICE, "alice/memory/zebra.md", "- a new line")
    assert remote.show("alice/memory/zebra.md").endswith("alice private note\n- a new line\n")
    assert result.get("version") == version_of((memex.root / "alice/memory/zebra.md").read_bytes())
    with pytest.raises(NotFound, match=re.escape(NOTE_NOT_FOUND)):
        memex.append(ALICE, "alice/memory/nothing.md", "text")
    assert not (memex.root / "alice/memory/nothing.md").exists()


def test_edit_replaces_one_passage(memex: Memex, remote: Remote) -> None:
    memex.edit(ALICE, "alice/memory/bike-repair.md", "every week", "every month")
    assert "cleaned every month with degreaser" in remote.show("alice/memory/bike-repair.md")


@pytest.mark.parametrize(
    ("old_text", "message"),
    [("no such words", "does not occur in the note"), ("bike", "occurs 2 times in the note")],
)
def test_edit_needs_exactly_one_occurrence(
    memex: Memex, remote: Remote, old_text: str, message: str
) -> None:
    head = remote.head()
    with pytest.raises(InvalidRequest, match=re.escape(message)):
        memex.edit(ALICE, "alice/memory/bike-repair.md", old_text, "x")
    assert remote.head() == head


def test_edit_checks_the_version_when_given(memex: Memex, remote: Remote) -> None:
    read = memex.read(ALICE, "alice/memory/zebra.md")
    head = remote.head()
    with pytest.raises(InvalidRequest, match=re.escape(CHANGED_SINCE_READ)):
        memex.edit(ALICE, "alice/memory/zebra.md", "private", "secret", version="0" * 40)
    assert remote.head() == head
    result = memex.edit(ALICE, "alice/memory/zebra.md", "private", "own", version=read["version"])
    assert result["pushed"] is True
    assert memex.read(ALICE, "alice/memory/zebra.md")["version"] == result.get("version")


def test_delete_needs_the_version_read_returned(memex: Memex, remote: Remote) -> None:
    path = "alice/memory/archive/old-bike.md"
    read = memex.read(ALICE, path)
    head = remote.head()
    for wrong, message in (("0" * 40, CHANGED_SINCE_READ), ("", "needs the note's version")):
        with pytest.raises(InvalidRequest, match=re.escape(message)):
            memex.delete(ALICE, path, wrong)
    assert remote.head() == head
    memex.append(ALICE, path, "one more line")
    with pytest.raises(InvalidRequest, match=re.escape(CHANGED_SINCE_READ)):
        memex.delete(ALICE, path, read["version"])
    result = memex.delete(ALICE, path, memex.read(ALICE, path)["version"])
    assert "version" not in result
    assert remote.log_of("%s") == f"memex-mcp: delete {path}"
    # The emptied folder goes too; the area's memory folder stays.
    assert not (memex.root / "alice/memory/archive").exists()
    assert (memex.root / "alice/memory").is_dir()
    with pytest.raises(NotFound):
        memex.read(ALICE, path)


def test_delete_leaves_the_index_line(topics: Memex, remote: Remote) -> None:
    path = "alice/memory/homelab/dns.md"
    topics.delete(ALICE, path, topics.read(ALICE, path)["version"])
    assert "[dns.md](dns.md)" in remote.show("alice/memory/homelab/MEMORY.md")


# -- size and secrets --------------------------------------------------


@pytest.fixture
def small(tmp_path: Path, origin: Origin) -> Iterator[Memex]:
    memex = Memex.from_config(make_config(tmp_path, origin, write={"max_note_bytes": 100}), None)
    memex.prepare()
    yield memex
    memex.index.close()


TOO_LARGE = "the note would be too large; create a new note instead"


def test_a_note_may_reach_the_size_limit_but_not_pass_it(small: Memex) -> None:
    exactly = "x" * 99 + "\n"
    assert small.create(ALICE, "alice/memory/daily/full.md", exactly)["pushed"] is True
    with pytest.raises(InvalidRequest, match=re.escape(TOO_LARGE)):
        small.create(ALICE, "alice/memory/daily/over.md", "x" * 100 + "\n")
    # Bytes, not characters: 50 two-byte letters are 100 bytes, plus the line break.
    with pytest.raises(InvalidRequest, match=re.escape(TOO_LARGE)):
        small.create(ALICE, "alice/memory/daily/umlaut.md", "ä" * 50)


def test_append_and_edit_may_not_pass_the_size_limit(small: Memex, remote: Remote) -> None:
    path = "alice/memory/zebra.md"
    size = len((small.root / path).read_bytes())
    head = remote.head()
    with pytest.raises(InvalidRequest, match=re.escape(TOO_LARGE)):
        small.append(ALICE, path, "y" * (100 - size))
    with pytest.raises(InvalidRequest, match=re.escape(TOO_LARGE)):
        small.edit(ALICE, path, "private", "y" * (100 - size + 8))
    assert remote.head() == head
    assert small.append(ALICE, path, "y" * (100 - size - 1))["pushed"] is True


def test_the_size_limit_is_configured(tmp_path: Path) -> None:
    assert make_config(tmp_path).write.max_note_bytes == 65536
    assert make_config(tmp_path, write={"max_note_bytes": 10}).write.max_note_bytes == 10


TOKEN = "gh" + "p_" + "Zz9" * 12


SECRET_CALLS: list[Callable[[Memex], object]] = [
    lambda m: m.create(ALICE, "alice/memory/t.md", f"token {TOKEN}\n", IndexLine("Notes", LINE)),
    lambda m: m.create(ALICE, "alice/memory/t.md", "# T\n", IndexLine("Notes", f"token {TOKEN}")),
    lambda m: m.create(ALICE, "alice/memory/t.md", "# T\n", IndexLine(TOKEN, LINE)),
    lambda m: m.create(ALICE, "alice/memory/daily/t.md", f"token {TOKEN}\n"),
    lambda m: m.append(ALICE, "alice/memory/zebra.md", f"token {TOKEN}"),
    lambda m: m.edit(ALICE, "alice/memory/zebra.md", "private", TOKEN),
]


@pytest.mark.parametrize(
    "call",
    SECRET_CALLS,
    ids=["content", "index_line", "index_section", "daily", "append", "edit"],
)
def test_a_secret_in_any_new_text_is_refused_without_its_value(
    memex: Memex, remote: Remote, call: Callable[[Memex], object]
) -> None:
    head = remote.head()
    kind, message = refusal(lambda: call(memex))
    assert kind is InvalidRequest
    assert "a GitHub token" in message
    assert "Zz9" not in message
    assert remote.head() == head


def test_a_secret_already_in_a_note_does_not_block_appending(memex: Memex, origin: Origin) -> None:
    origin.commit("old", {"alice/memory/keys.md": f"# Keys\n\nold {TOKEN}\n"})
    memex.refresh()
    assert memex.append(ALICE, "alice/memory/keys.md", "a harmless line")["pushed"] is True


def test_a_password_in_prose_is_allowed(memex: Memex) -> None:
    result = memex.append(ALICE, "household/memory/wifi.md", "Guest password: Sommer-2026!")
    assert result["pushed"] is True


# -- indexes -----------------------------------------------------------


def test_create_adds_the_line_at_the_end_of_an_existing_section(
    topics: Memex, remote: Remote
) -> None:
    result = topics.create(
        ALICE, "alice/memory/homelab/vpn.md", "# VPN\n", IndexLine("Services", "VPN")
    )
    assert result.get("index") == "alice/memory/homelab/MEMORY.md"
    index = remote.show("alice/memory/homelab/MEMORY.md")
    assert "- [dns.md](dns.md) — DNS\n- [vpn.md](vpn.md) — VPN\n\n## Hardware" in index
    assert remote.log_of("%B").strip() == (
        "memex-mcp: 2 files in alice/memory/homelab\n\n"
        "create alice/memory/homelab/vpn.md\n"
        "update alice/memory/homelab/MEMORY.md\n\n"
        "Memex-Client: claude-code"
    )


def test_a_section_equal_ignoring_case_is_the_existing_one(topics: Memex, remote: Remote) -> None:
    topics.create(ALICE, "alice/memory/homelab/nas.md", "# NAS\n", IndexLine("hardware", "NAS"))
    index = remote.show("alice/memory/homelab/MEMORY.md")
    assert index.count("## ") == 2
    assert index.endswith("- [ups.md](ups.md) — UPS\n- [nas.md](nas.md) — NAS\n")


def test_a_new_section_is_added_at_the_end(topics: Memex, remote: Remote) -> None:
    topics.create(ALICE, "alice/memory/homelab/plan.md", "# Plan\n", IndexLine("Plans", "the plan"))
    assert remote.show("alice/memory/homelab/MEMORY.md").endswith(
        "- [ups.md](ups.md) — UPS\n\n## Plans\n\n- [plan.md](plan.md) — the plan\n"
    )


def test_a_wiki_note_gets_a_wiki_line(topics: Memex, remote: Remote) -> None:
    topics.create(
        ALICE, "alice/wiki/networking/vlan.md", "# VLAN\n", IndexLine("Basics", "tagging")
    )
    assert remote.show("alice/wiki/networking/_index.md").endswith(
        "## Basics\n- [[dhcp]] — leases\n- [[vlan]] — tagging\n"
    )


def test_a_note_below_a_topic_goes_into_the_topics_index(
    topics: Memex, origin: Origin, remote: Remote
) -> None:
    # An index file in a subfolder is not the topic's index, and is left alone.
    git(origin.work, "pull", "-q", "--ff-only")
    origin.commit("own index", {"alice/memory/homelab/net/MEMORY.md": "# net\n"})
    topics.refresh()
    deep = topics.create(
        ALICE, "alice/memory/homelab/net/vlan.md", "# V\n", IndexLine("Services", "vlans")
    )
    assert deep.get("index") == "alice/memory/homelab/MEMORY.md"
    assert "- [net/vlan.md](net/vlan.md) — vlans" in remote.show("alice/memory/homelab/MEMORY.md")
    assert remote.show("alice/memory/homelab/net/MEMORY.md") == "# net\n"
    assert topics.index_sections(ALICE, "alice/memory/homelab/net")["index"] == (
        "alice/memory/homelab/MEMORY.md"
    )


def test_a_new_memory_topic_gets_its_index_in_the_same_commit(
    topics: Memex, remote: Remote
) -> None:
    pushes = remote.pushes()
    listing = IndexLine("Plants", "roses", "plants and beds")
    result = topics.create(ALICE, "alice/memory/garden/roses.md", "# Roses\n", listing)
    assert remote.show("alice/memory/garden/MEMORY.md") == (
        "# garden\n\n## Plants\n\n- [roses.md](roses.md) — roses\n"
    )
    # The main index has no section: the section of topics goes to its end.
    assert remote.show("alice/memory/MEMORY.md") == (
        "# Memory index\n\n- bike repair notes\n\n"
        "## Topics\n\n- [garden/MEMORY.md](garden/MEMORY.md) — plants and beds\n"
    )
    assert remote.pushes() == pushes + 1
    assert remote.log_of("%B").strip() == (
        "memex-mcp: 3 files in alice/memory\n\n"
        "create alice/memory/garden/roses.md\n"
        "create alice/memory/garden/MEMORY.md\n"
        "update alice/memory/MEMORY.md\n\n"
        "Memex-Client: claude-code"
    )
    assert result.get("index") == "alice/memory/garden/MEMORY.md"


def test_a_missing_main_index_is_created_with_its_topics(memex: Memex, remote: Remote) -> None:
    memex.create(BOB, "bob/memory/bees/hive.md", "# Hive\n", IndexLine("Hives", "hive", "bees"))
    assert remote.show("bob/memory/MEMORY.md") == (
        "# bob memory\n\n## Topics\n\n- [bees/MEMORY.md](bees/MEMORY.md) — bees\n"
    )


def test_the_topics_go_before_the_notes_of_the_main_index(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    main = "# alice memory\n\n## Bikes\n\n- [bike-repair.md](bike-repair.md) — chain\n"
    origin.commit("main index", {"alice/memory/MEMORY.md": main})
    memex.refresh()
    memex.create(ALICE, "alice/memory/garden/roses.md", "# R\n", IndexLine("Plants", LINE, "beds"))
    memex.create(ALICE, "alice/memory/house/roof.md", "# R\n", IndexLine("Roof", LINE, "house"))
    assert remote.show("alice/memory/MEMORY.md") == (
        "# alice memory\n\n## Topics\n\n"
        "- [garden/MEMORY.md](garden/MEMORY.md) — beds\n"
        "- [house/MEMORY.md](house/MEMORY.md) — house\n\n"
        "## Bikes\n\n- [bike-repair.md](bike-repair.md) — chain\n"
    )


def test_a_note_in_the_main_index_cannot_take_the_section_of_topics(
    memex: Memex, remote: Remote
) -> None:
    head = remote.head()
    for section in ("Topics", "topics", "## TOPICS"):
        with pytest.raises(
            InvalidRequest, match="^the section Topics lists topics; choose another"
        ):
            memex.create(ALICE, "alice/memory/bikes.md", "# Bikes\n", IndexLine(section, LINE))
    assert remote.head() == head
    # In a topic's own index, a section of that name is an ordinary one.
    listing = IndexLine("Topics", LINE, "beds")
    memex.create(ALICE, "alice/memory/garden/roses.md", "# Roses\n", listing)
    assert "## Topics\n\n- [roses.md](roses.md)" in remote.show("alice/memory/garden/MEMORY.md")


def test_an_archive_folder_in_memory_is_a_topic_like_any_other(
    memex: Memex, remote: Remote
) -> None:
    assert memex.index_sections(ALICE, "alice/memory/archive")["new_topic"] is True
    listing = IndexLine("Bikes", "the old bike", "notes that no longer hold")
    memex.create(ALICE, "alice/memory/archive/old-car.md", "# Car\n", listing)
    assert remote.show("alice/memory/MEMORY.md").endswith(
        "## Topics\n\n- [archive/MEMORY.md](archive/MEMORY.md) — notes that no longer hold\n"
    )


def test_a_new_wiki_topic_gets_its_index_and_a_master_index_line(
    topics: Memex, remote: Remote
) -> None:
    listing = IndexLine("Baking", "sourdough", "bread and cakes")
    topics.create(ALICE, "alice/wiki/cooking/bread.md", "# Bread\n", listing)
    assert remote.show("alice/wiki/cooking/_index.md") == (
        "# cooking\n\n## Baking\n\n- [[bread]] — sourdough\n"
    )
    # The master index has no section: the line goes to its end.
    assert remote.show("alice/wiki/_master-index.md") == (
        "# Wiki\n\n- [[wiki/networking/_index|Networking]]\n"
        "- [[wiki/cooking/_index|cooking]] — bread and cakes\n"
    )
    assert remote.log_of("%B").strip() == (
        "memex-mcp: 3 files in alice/wiki\n\n"
        "create alice/wiki/cooking/bread.md\n"
        "create alice/wiki/cooking/_index.md\n"
        "update alice/wiki/_master-index.md\n\n"
        "Memex-Client: claude-code"
    )


def test_a_missing_master_index_is_created(memex: Memex, remote: Remote) -> None:
    memex.create(BOB, "bob/wiki/garden/soil.md", "# Soil\n", IndexLine("Basics", "soil", "beds"))
    assert remote.show("bob/wiki/_master-index.md") == (
        "# Wiki master index\n\n## Topics\n\n- [[wiki/garden/_index|garden]] — beds\n"
    )


def test_the_first_note_of_an_area_memory_folder_creates_its_index(
    memex: Memex, remote: Remote
) -> None:
    result = memex.create(
        GUEST, "household/memory/bins.md", "# Bins\n", IndexLine("House", "bin days")
    )
    assert result.get("index") == "household/memory/MEMORY.md"
    assert remote.show("household/memory/MEMORY.md") == (
        "# household memory\n\n## House\n\n- [bins.md](bins.md) — bin days\n"
    )


def test_a_daily_note_takes_no_index_line(memex: Memex, remote: Remote) -> None:
    index = clone_file(memex, "alice/memory/MEMORY.md")
    result = memex.create(ALICE, "alice/memory/daily/2026-10-10.md", "## 09:00 — start\n")
    assert result.get("index") is None
    # Given anyway, the index arguments change nothing.
    memex.create(
        ALICE, "alice/memory/daily/2026-10-11.md", "## 10:00\n", IndexLine("Days", "a day")
    )
    assert clone_file(memex, "alice/memory/MEMORY.md") == index
    assert remote.log_of("%s") == "memex-mcp: create alice/memory/daily/2026-10-11.md"


def test_a_note_without_index_arguments_is_refused_naming_the_sections(
    topics: Memex, remote: Remote
) -> None:
    head = remote.head()
    for section, line in ((None, LINE), ("Services", None), ("", "")):
        with pytest.raises(InvalidRequest) as caught:
            topics.create(ALICE, "alice/memory/homelab/x.md", "# X\n", IndexLine(section, line))
        assert "Services; Hardware" in str(caught.value)
    with pytest.raises(InvalidRequest, match="at most 300 characters"):
        topics.create(ALICE, "alice/memory/homelab/x.md", "# X\n", IndexLine("Services", "x" * 301))
    assert remote.head() == head
    assert not (topics.root / "alice/memory/homelab/x.md").exists()


def test_index_sections_names_the_index_and_its_sections(topics: Memex) -> None:
    expected = {
        "index": "alice/memory/homelab/MEMORY.md",
        "exists": True,
        "sections": ["Services", "Hardware"],
        "new_topic": False,
    }
    assert topics.index_sections(ALICE, "alice/memory/homelab/new.md") == expected
    assert topics.index_sections(ALICE, "alice/memory/homelab") == expected
    assert topics.index_sections(ALICE, "alice/memory/homelab/deeper/new.md") == expected
    assert topics.index_sections(ALICE, "alice/wiki/cooking") == {
        "index": "alice/wiki/cooking/_index.md",
        "exists": False,
        "sections": [],
        "new_topic": True,
    }
    # A new memory topic needs a topic description like a new wiki topic; an existing
    # topic does not, nor does an area's memory folder itself, even without its index.
    assert topics.index_sections(ALICE, "alice/memory/garden") == {
        "index": "alice/memory/garden/MEMORY.md",
        "exists": False,
        "sections": [],
        "new_topic": True,
    }
    assert topics.index_sections(ALICE, "alice/wiki/networking")["new_topic"] is False
    assert topics.index_sections(ALICE, "household/memory") == {
        "index": "household/memory/MEMORY.md",
        "exists": False,
        "sections": [],
        "new_topic": False,
    }
    assert topics.index_sections(ALICE, "alice/memory/daily") == {
        "index": None,
        "exists": False,
        "sections": [],
        "new_topic": False,
    }


def test_index_sections_answers_foreign_areas_like_missing_ones(memex: Memex) -> None:
    for path in ("bob/memory", "bob/memory/zebra.md", "nobody/memory"):
        assert refusal(lambda path=path: memex.index_sections(ALICE, path)) == (
            NotFound,
            NOTE_NOT_FOUND,
        )
    with pytest.raises(AccessDenied, match=re.escape(WRITE_OUTSIDE)):
        memex.index_sections(ALICE, "alice")


# -- commits and search ------------------------------------------------


def test_a_write_is_committed_by_the_service_for_the_caller(memex: Memex, remote: Remote) -> None:
    memex.append(ALICE, "alice/memory/zebra.md", "line")
    assert remote.log_of("%an|%ae|%cn|%ce|%B").strip() == (
        "alice|alice@memex.invalid|memex-mcp|memex-mcp@memex.invalid|"
        "memex-mcp: append alice/memory/zebra.md\n\n"
        "Memex-Client: claude-code"
    )


def test_every_write_commit_names_its_client_as_a_git_trailer(
    topics: Memex, remote: Remote
) -> None:
    via_chat = Identity(ALICE.username, ALICE.groups, "memex-openwebui")
    topics.create(via_chat, "alice/memory/homelab/vpn.md", "# VPN\n", IndexLine("Services", "VPN"))
    topics.append(ALICE, "alice/memory/homelab/vpn.md", "more")
    trailers = remote.log_of("%(trailers:key=Memex-Client,valueonly)", count=2)
    assert trailers.split() == ["claude-code", "memex-openwebui"]
    # git reads the client as the only trailer, not the file lines above it; the author stays
    # the caller.
    every_trailer = "%an|%(trailers:only,unfold,separator=%x2C)"
    assert remote.log_of(every_trailer, count=2).split("\n") == [
        "alice|Memex-Client: claude-code",
        "alice|Memex-Client: memex-openwebui",
    ]


def test_the_author_address_comes_from_the_configured_pattern(
    tmp_path: Path, origin: Origin, remote: Remote
) -> None:
    config = make_config(tmp_path, origin, write={"author_email": "{username}@users.example.org"})
    memex = Memex.from_config(config, None)
    memex.prepare()
    try:
        odd = Identity("ali ce<x>", frozenset({"memex", "household"}), "claude-code")
        memex.append(odd, "household/memory/wifi.md", "line")
        assert remote.log_of("%an|%ae") == "ali cex|ali-ce-x-@users.example.org"
    finally:
        memex.index.close()


def test_one_commit_and_one_push_per_write(memex: Memex, remote: Remote) -> None:
    head, pushes = remote.head(), remote.pushes()
    result = memex.append(ALICE, "alice/memory/zebra.md", "line")
    assert remote.pushes() == pushes + 1
    assert git(memex.root, "rev-parse", "HEAD^").strip() == head
    assert result["commit"] == remote.head()
    assert git(memex.root, "status", "--porcelain") == ""


def test_a_write_is_searchable_at_once(memex: Memex) -> None:
    result = memex.create(ALICE, "alice/memory/daily/today.md", "# Today\n\nquetzal sighting\n")
    assert result["searchable"] is True
    hits = memex.search(ALICE, "quetzal")["hits"]
    assert [hit["path"] for hit in hits] == ["alice/memory/daily/today.md"]


def test_a_failed_reindex_after_the_commit_is_not_an_error(
    memex: Memex, remote: Remote, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: object) -> int:
        raise RuntimeError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(memex.index, "update", broken)
        result = memex.create(ALICE, "alice/memory/daily/today.md", "# Today\n\nkinkajou\n")
    assert (result["pushed"], result["searchable"]) == (True, False)
    assert result.get("notice") == NOT_SEARCHABLE
    assert remote.head() == result["commit"]
    assert memex.search(ALICE, "kinkajou")["hits"] == []
    memex.refresh()
    assert [h["path"] for h in memex.search(ALICE, "kinkajou")["hits"]] == [
        "alice/memory/daily/today.md"
    ]


def test_a_failed_commit_leaves_the_clone_as_it_was(
    memex: Memex, remote: Remote, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memex_mcp.repo import GitError, GitRepo  # noqa: PLC0415

    def broken(*_args: object) -> str:
        raise GitError("git commit failed: disk full")

    head = remote.head()
    with monkeypatch.context() as patch:
        patch.setattr(GitRepo, "commit", broken)
        with pytest.raises(GitError):
            listing = IndexLine("Notes", LINE, "the homelab")
            memex.create(ALICE, "alice/memory/homelab/new.md", "# New\n", listing)
    assert git(memex.root, "status", "--porcelain", "--untracked-files=all") == ""
    assert git(memex.root, "rev-parse", "HEAD").strip() == head


def _result_keys(result: WriteResult) -> set[str]:
    return set(result)


def test_a_write_result_states_commit_push_and_version(memex: Memex) -> None:
    result = memex.create(ALICE, "alice/memory/daily/d.md", "# D\n")
    assert _result_keys(result) == {"path", "commit", "pushed", "searchable", "version", "index"}
    assert result["path"] == "alice/memory/daily/d.md"


def test_the_folder_check_after_resolution_catches_a_symlink_the_walk_missed(
    memex: Memex, monkeypatch: pytest.MonkeyPatch
) -> None:
    (memex.root / "alice/memory/elsewhere").symlink_to("../../bob/memory")

    # As if the link appeared between the walk and the write.
    def never(_self: Path) -> bool:
        return False

    monkeypatch.setattr(Path, "is_symlink", never)
    with pytest.raises(NotFound):
        create(memex, ALICE, "alice/memory/elsewhere/new.md")
    assert not (memex.root / "bob/memory/new.md").exists()


def test_a_file_in_the_way_of_a_folder_answers_like_a_missing_note(memex: Memex) -> None:
    assert refusal(lambda: create(memex, ALICE, "alice/memory/zebra.md/inner.md")) == (
        NotFound,
        NOTE_NOT_FOUND,
    )


def test_an_index_that_is_a_symlink_is_never_written(topics: Memex) -> None:
    index = topics.root / "alice/memory/homelab/MEMORY.md"
    index.unlink()
    index.symlink_to("../../../bob/memory/garden.md")
    with pytest.raises(NotFound):
        topics.create(ALICE, "alice/memory/homelab/new.md", "# New\n", IndexLine("Notes", LINE))
    assert (
        clone_file(topics, "bob/memory/garden.md")
        == "# Garden\n\nThe tomatoes need water every evening.\n"
    )


def test_a_master_index_line_is_not_added_twice(topics: Memex, remote: Remote) -> None:
    # The topic's own index went missing, its master index line did not.
    topics.create(
        ALICE, "alice/wiki/networking/vlan.md", "# VLAN\n", IndexLine("Basics", "tagging")
    )
    (topics.root / "alice/wiki/networking/_index.md").unlink()
    git(topics.root, "commit", "-q", "-am", "index gone")
    master = remote.show("alice/wiki/_master-index.md").replace(
        "[[wiki/networking/_index|Networking]]", "[[wiki/cooking/_index|cooking]]"
    )
    (topics.root / "alice/wiki/_master-index.md").write_text(master, encoding="utf-8")
    git(topics.root, "commit", "-q", "-am", "master names cooking")
    listing = IndexLine("Baking", "sourdough", "bread")
    topics.create(ALICE, "alice/wiki/cooking/bread.md", "# Bread\n", listing)
    assert remote.show("alice/wiki/_master-index.md").count("wiki/cooking/_index") == 1


def test_a_main_index_line_is_not_added_twice(
    topics: Memex, origin: Origin, remote: Remote
) -> None:
    # The main index lists the topic already, though the topic has no index of its own.
    git(origin.work, "pull", "-q", "--ff-only")
    main = "# alice memory\n\n- [garden/MEMORY.md](garden/MEMORY.md) — the garden\n"
    origin.commit("main index", {"alice/memory/MEMORY.md": main})
    topics.refresh()
    listing = IndexLine("Plants", "roses", "plants and beds")
    topics.create(ALICE, "alice/memory/garden/roses.md", "# Roses\n", listing)
    assert remote.show("alice/memory/MEMORY.md") == main
    assert remote.log_of("%s") == "memex-mcp: 2 files in alice/memory/garden"


def test_a_note_that_is_not_utf8_is_not_changed(memex: Memex, origin: Origin) -> None:
    (origin.work / "alice/memory/latin1.md").write_bytes(b"# Caf\xe9\n")
    origin.commit("latin-1")
    memex.refresh()
    for call in (
        lambda: memex.append(ALICE, "alice/memory/latin1.md", "more"),
        lambda: memex.edit(ALICE, "alice/memory/latin1.md", "Caf", "Bar"),
    ):
        with pytest.raises(InvalidRequest, match="not UTF-8"):
            call()
    assert (memex.root / "alice/memory/latin1.md").read_bytes() == b"# Caf\xe9\n"


def test_a_note_needs_content(memex: Memex, remote: Remote) -> None:
    head = remote.head()
    for empty in ("", "  \n\n"):
        with pytest.raises(InvalidRequest, match="content must not be empty"):
            memex.create(ALICE, "alice/memory/daily/empty.md", empty)
    assert remote.head() == head


def test_a_wiki_note_needs_a_topic_folder(topics: Memex, remote: Remote) -> None:
    head = remote.head()
    guidance = "wiki notes go into a topic folder: alice/wiki/<topic>/glossary.md"
    with pytest.raises(InvalidRequest, match=re.escape(guidance)):
        topics.create(ALICE, "alice/wiki/glossary.md", "# Glossary\n", IndexLine("Notes", LINE))
    with pytest.raises(InvalidRequest, match="wiki notes go into a topic folder"):
        topics.index_sections(ALICE, "alice/wiki")
    assert remote.head() == head
    assert not (topics.root / "alice/wiki/glossary.md").exists()
    assert not (topics.root / "alice/wiki/_index.md").exists()


def test_notes_already_at_the_wiki_root_can_still_be_changed(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    origin.commit("log", {"alice/wiki/log.md": "# Log\n\n- started\n"})
    memex.refresh()
    memex.append(ALICE, "alice/wiki/log.md", "- went on")
    memex.edit(ALICE, "alice/wiki/log.md", "started", "began")
    assert remote.show("alice/wiki/log.md") == "# Log\n\n- began\n- went on\n"
    memex.delete(ALICE, "alice/wiki/log.md", memex.read(ALICE, "alice/wiki/log.md")["version"])
    assert remote.log_of("%s") == "memex-mcp: delete alice/wiki/log.md"


@pytest.mark.parametrize("tool", list(WRITE_CALLS))
@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("alice/memory/claude.md", PROTECTED_FILE),
        ("alice/memory/CLAUDE.local.md", PROTECTED_FILE),
        ("alice/memory/homelab/claude.LOCAL.md", PROTECTED_FILE),
        ("alice/wiki/networking/Agents.md", PROTECTED_FILE),
        ("alice/memory/CLAUDE.md/inner.md", PROTECTED_FILE),
        ("alice/wiki/AGENTS.md/networking/x.md", PROTECTED_FILE),
        ("alice/memory/homelab/memory.md", INDEX_FILE),
        ("alice/wiki/networking/_INDEX.md", INDEX_FILE),
        ("alice/wiki/_Master-Index.md", INDEX_FILE),
        ("household/memory/MEMORY.md/x.md", INDEX_FILE),
        ("alice/wiki/_index.md/notes/x.md", INDEX_FILE),
    ],
)
def test_protected_and_index_names_are_refused_in_any_case_and_as_folders(
    memex: Memex, remote: Remote, tool: str, path: str, message: str
) -> None:
    head = remote.head()
    with pytest.raises(AccessDenied, match=re.escape(message)):
        WRITE_CALLS[tool](memex, ALICE, path)
    assert remote.head() == head
    assert not (memex.root / path).exists()


def test_a_secret_in_the_path_is_refused(memex: Memex, remote: Remote) -> None:
    head = remote.head()
    kind, message = refusal(lambda: create(memex, ALICE, f"alice/memory/daily/{TOKEN}.md"))
    assert (kind, "a GitHub token" in message, "Zz9" in message) == (InvalidRequest, True, False)
    assert remote.head() == head


@pytest.mark.parametrize("bad", ["\u202e", "\u200b", "<!--"], ids=repr)
def test_format_characters_and_html_comments_are_refused_everywhere(
    topics: Memex, remote: Remote, bad: str
) -> None:
    head = remote.head()
    with pytest.raises(AccessDenied, match=re.escape(FORBIDDEN_CHARACTER)):
        create(topics, ALICE, f"alice/memory/daily/a{bad}b.md")
    for listing in (
        IndexLine(f"Se{bad}ction", LINE),
        IndexLine("Services", f"li{bad}ne"),
    ):
        with pytest.raises(InvalidRequest):
            topics.create(ALICE, "alice/memory/homelab/x.md", "# X\n", listing)
    with pytest.raises(InvalidRequest, match="topic_description"):
        topics.create(ALICE, "alice/wiki/new/x.md", "# X\n", IndexLine("S", LINE, f"t{bad}t"))
    assert remote.head() == head


@pytest.mark.parametrize("char", ["\x0c", "\x85", "\u2028"], ids=repr)
def test_a_line_breaking_character_never_reaches_an_index(
    topics: Memex, remote: Remote, char: str
) -> None:
    for listing in (IndexLine("Services", f"one{char}## Two"), IndexLine(f"A{char}B", LINE)):
        with pytest.raises(InvalidRequest, match="must be one line"):
            topics.create(ALICE, "alice/memory/homelab/x.md", "# X\n", listing)
    assert remote.show("alice/memory/homelab/MEMORY.md") == TOPICS["alice/memory/homelab/MEMORY.md"]


def test_an_at_sign_starting_a_word_never_reaches_an_index(topics: Memex, remote: Remote) -> None:
    head = remote.head()
    for listing in (
        IndexLine("Services", "see @~/.ssh/id_ed25519"),
        IndexLine("@imports", LINE),
        IndexLine("Basics", LINE, "pulls in @secrets.md"),
    ):
        with pytest.raises(InvalidRequest, match="'@' at the start of a word"):
            topics.create(ALICE, "alice/wiki/new/x.md", "# X\n", listing)
    assert remote.head() == head


def test_a_new_topic_needs_its_description(topics: Memex, remote: Remote) -> None:
    head = remote.head()
    for path in ("alice/wiki/cooking/bread.md", "alice/memory/garden/roses.md"):
        with pytest.raises(InvalidRequest, match="needs topic_description"):
            topics.create(ALICE, path, "# X\n", IndexLine("Plants", LINE))
    assert remote.head() == head
    assert not (topics.root / "alice/memory/garden").exists()
    # An existing topic, or an area's memory folder itself, needs none, even
    # while that folder has no index.
    topics.create(ALICE, "alice/wiki/networking/vlan.md", "# V\n", IndexLine("Basics", LINE))
    topics.create(GUEST, "household/memory/bins.md", "# B\n", IndexLine("House", LINE))


def test_the_master_line_goes_to_the_end_of_its_first_section(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    origin.commit(
        "master",
        {
            "alice/wiki/_master-index.md": (
                "# Wiki\n\n## Areas\n\n- [[wiki/a/_index|a]] — a\n\n## Log\n"
            )
        },
    )
    memex.refresh()
    memex.create(ALICE, "alice/wiki/b/n.md", "# N\n", IndexLine("Notes", LINE, "the b topic"))
    assert remote.show("alice/wiki/_master-index.md") == (
        "# Wiki\n\n## Areas\n\n- [[wiki/a/_index|a]] — a\n"
        "- [[wiki/b/_index|b]] — the b topic\n\n## Log\n"
    )


def test_writing_is_off_unless_switched_on(tmp_path: Path) -> None:
    raw = raw_config(tmp_path)
    raw["write"] = {}
    assert build_config(raw, {}).write.enabled is False
    assert load_config(EXAMPLE, env={}).write.enabled is False


@pytest.mark.parametrize(
    "path",
    [
        "alice/memory/topic/x @/etc/passwd y/n.md",
        "alice/memory/x @big.md y/n.md",
        "alice/memory/a @b/n.md",
        "alice/memory/@notes.md",
        "alice/wiki/@topic/n.md",
    ],
)
def test_an_at_sign_starting_a_word_in_the_path_is_refused(
    topics: Memex, remote: Remote, path: str
) -> None:
    head = remote.head()
    with pytest.raises(AccessDenied, match=re.escape(WORD_START_AT)):
        topics.create(ALICE, path, "# N\n", IndexLine("Notes", LINE, "a topic"))
    assert remote.head() == head
    with pytest.raises(AccessDenied, match=re.escape(WORD_START_AT)):
        topics.index_sections(ALICE, path)


def test_an_at_sign_inside_a_word_in_the_path_is_fine(memex: Memex) -> None:
    listing = IndexLine("Notes", LINE, "mail")
    result = memex.create(ALICE, "alice/memory/mail/alice@example.org.md", "# New\n", listing)
    assert result["pushed"] is True


def test_a_composed_index_line_is_checked_again(
    topics: Memex, remote: Remote, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As if the path check let a word-start "@" through.
    monkeypatch.setattr(rights_module, "IMPORT_AT", re.compile(r"(?!)"))
    head = remote.head()
    for path in ("alice/memory/homelab/x @y/n.md", "alice/memory/x @big.md y/n.md"):
        with pytest.raises(InvalidRequest, match="cannot be listed in its index"):
            topics.create(ALICE, path, "# New\n", IndexLine("Notes", LINE, "a topic"))
    assert remote.head() == head


@pytest.mark.parametrize(
    "path",
    [
        "alice/memory/CLAUDE.md./x.md",
        "alice/memory/CLAUDE.md /x.md",
        "alice/memory/agents.md ./x.md",
        "alice/memory/MEMORY.md../x.md",
        "alice/wiki/_index.md /x.md",
    ],
)
def test_trailing_dots_and_spaces_do_not_hide_a_protected_name(memex: Memex, path: str) -> None:
    with pytest.raises(AccessDenied):
        create(memex, ALICE, path)
    assert not (memex.root / path).exists()
