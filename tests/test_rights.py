"""The rights matrix: every identity against every area through every tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex_mcp.config import ConfigError
from memex_mcp.rights import ACCESS_DENIED, AccessDenied, Identity, NotFound, Policy
from memex_mcp.service import Memex
from tests.conftest import (
    ALICE,
    BOB,
    CAROL,
    GUEST,
    MALLORY,
    MARKER,
    STRANGER,
    make_config,
)

AREAS = ("alice", "bob", "household", "carol")

# identity -> the areas it reads; None means no access at all.
EXPECTED: dict[str, frozenset[str] | None] = {
    "alice": frozenset({"alice", "household"}),
    "bob": frozenset({"bob", "household"}),
    "carol": frozenset({"carol"}),
    "mallory": None,
    "stranger": None,
    "guest": frozenset({"household"}),
}
IDENTITIES = (ALICE, BOB, CAROL, MALLORY, STRANGER, GUEST)
CASES = [(identity, area) for identity in IDENTITIES for area in AREAS]


def _ids(case: tuple[Identity, str]) -> str:
    return f"{case[0].username}-{case[1]}"


def allowed(identity: Identity, area: str) -> bool:
    areas = EXPECTED[identity.username]
    return areas is not None and area in areas


def refusal(identity: Identity) -> type[Exception]:
    """No access at all is AccessDenied; a foreign area answers like a missing one."""
    return AccessDenied if EXPECTED[identity.username] is None else NotFound


@pytest.mark.parametrize("case", CASES, ids=[_ids(c) for c in CASES])
def test_read_matrix(memex: Memex, case: tuple[Identity, str]) -> None:
    identity, area = case
    path = f"{area}/memory/zebra.md"
    if allowed(identity, area):
        note = memex.read(identity, path)
        assert note["area"] == area
        assert f"{MARKER} {area}" in note["content"]
    else:
        with pytest.raises(refusal(identity)):
            memex.read(identity, path)


@pytest.mark.parametrize("case", CASES, ids=[_ids(c) for c in CASES])
def test_list_matrix(memex: Memex, case: tuple[Identity, str]) -> None:
    identity, area = case
    if allowed(identity, area):
        listing = memex.list(identity, area, "memory")
        assert f"{area}/memory/zebra.md" in {e["path"] for e in listing["entries"]}
    else:
        with pytest.raises(refusal(identity)):
            memex.list(identity, area, "memory")


@pytest.mark.parametrize("case", CASES, ids=[_ids(c) for c in CASES])
def test_search_with_area_matrix(memex: Memex, case: tuple[Identity, str]) -> None:
    identity, area = case
    if allowed(identity, area):
        result = memex.search(identity, MARKER, area=area)
        assert {hit["area"] for hit in result["hits"]} == {area}
    else:
        with pytest.raises(refusal(identity)):
            memex.search(identity, MARKER, area=area)


@pytest.mark.parametrize("identity", IDENTITIES, ids=[i.username for i in IDENTITIES])
def test_search_without_area_returns_exactly_the_allowed_areas(
    memex: Memex, identity: Identity
) -> None:
    expected = EXPECTED[identity.username]
    if expected is None:
        with pytest.raises(AccessDenied):
            memex.search(identity, MARKER)
        return
    result = memex.search(identity, MARKER, limit=50)
    assert {hit["area"] for hit in result["hits"]} == expected
    # Nothing from the hidden folder or the repository root, either.
    assert all(".obsidian" not in hit["path"] for hit in result["hits"])
    assert all(hit["path"] != "README.md" for hit in result["hits"])


def test_no_memex_group_is_rejected_by_every_tool(memex: Memex) -> None:
    for call in (
        lambda: memex.read(MALLORY, "mallory/memory/zebra.md"),
        lambda: memex.list(MALLORY, "mallory"),
        lambda: memex.search(MALLORY, MARKER),
    ):
        with pytest.raises(AccessDenied, match=ACCESS_DENIED):
            call()


def test_household_needs_the_group_even_for_the_owner(memex: Memex) -> None:
    alice_alone = Identity("alice", frozenset({"memex"}))
    with pytest.raises(NotFound):
        memex.read(alice_alone, "household/memory/zebra.md")
    assert memex.read(alice_alone, "alice/memory/zebra.md")["area"] == "alice"


def test_policy_without_access_group_gives_nothing() -> None:
    policy = Policy(
        users={"alice": "alice"},
        access_group="memex",
        household_group="household",
        household_area="household",
    )
    assert policy.areas_for(ALICE) == {"alice", "household"}
    with pytest.raises(AccessDenied):
        policy.areas_for(Identity("alice", frozenset({"household", "admins"})))


def test_config_refuses_a_user_mapped_to_the_household_area(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="household area"):
        make_config(tmp_path, users={"eve": "household"})


@pytest.mark.parametrize("area", ["", ".git", "..", "a/b", ".obsidian"])
def test_config_refuses_area_names_that_are_not_plain_directories(
    tmp_path: Path, area: str
) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, users={"eve": area})


@pytest.mark.parametrize("identity", IDENTITIES, ids=[i.username for i in IDENTITIES])
def test_areas_lists_exactly_the_callers_areas(memex: Memex, identity: Identity) -> None:
    expected = EXPECTED[identity.username]
    if expected is None:
        with pytest.raises(AccessDenied, match=ACCESS_DENIED):
            memex.areas(identity)
        return
    assert memex.areas(identity) == {"areas": sorted(expected)}
