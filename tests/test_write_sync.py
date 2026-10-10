"""Writes and the remote: turned-down pushes, an unreachable remote, conflicts, dates from git."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memex_mcp.repo import CONFLICT_PREFIX, GitRepo, Push
from memex_mcp.rights import InvalidRequest
from memex_mcp.service import NOT_PUSHED, PUSH_RETRIES, Memex
from tests.conftest import ALICE, BOB, Origin, git
from tests.test_write import Remote


@pytest.fixture
def remote(origin: Origin) -> Remote:
    return Remote(origin)


def writer_commit(origin: Origin, message: str, notes: dict[str, str]) -> str:
    """Someone else pushes, on top of whatever the remote has now."""
    git(origin.work, "pull", "-q", "--ff-only")
    return origin.commit(message, notes)


def break_remote(memex: Memex, origin: Origin) -> None:
    git(memex.root, "remote", "set-url", "origin", str(origin.bare.parent / "gone.git"))


def mend_remote(memex: Memex, origin: Origin) -> None:
    git(memex.root, "remote", "set-url", "origin", str(origin.bare))


def paths(memex: Memex, query: str) -> list[str]:
    return [hit["path"] for hit in memex.search(ALICE, query, limit=50)["hits"]]


# -- a remote that moved on ----------------------------------------------


def test_a_turned_down_push_applies_the_write_again_on_the_new_state(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    theirs = writer_commit(
        origin, "theirs", {"alice/memory/zebra.md": "# Alice zebra\n\nrewritten by the deck\n"}
    )
    result = memex.append(ALICE, "alice/memory/zebra.md", "- from the server")
    assert result["pushed"] is True
    assert remote.head() == result["commit"]
    assert git(origin.bare, "rev-parse", "main^").strip() == theirs
    # Applied to the note as it is now, not merged as text.
    assert remote.show("alice/memory/zebra.md") == (
        "# Alice zebra\n\nrewritten by the deck\n- from the server\n"
    )
    assert paths(memex, "rewritten") == ["alice/memory/zebra.md"]


def test_a_write_no_longer_valid_on_the_new_state_fails_with_its_reason(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    theirs = writer_commit(origin, "theirs", {"alice/memory/zebra.md": "# Alice zebra\n\ngone\n"})
    pushes = remote.pushes()
    with pytest.raises(InvalidRequest, match="does not occur in the note"):
        memex.edit(ALICE, "alice/memory/zebra.md", "private note", "x")
    assert remote.head() == theirs
    assert remote.pushes() == pushes
    # The clone follows the remote, and so does search.
    assert git(memex.root, "rev-parse", "HEAD").strip() == theirs
    assert git(memex.root, "status", "--porcelain") == ""
    assert paths(memex, "gone") == ["alice/memory/zebra.md"]


def test_a_create_whose_path_was_taken_meanwhile_is_refused(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    theirs = writer_commit(origin, "theirs", {"alice/memory/daily/d.md": "# Theirs\n"})
    with pytest.raises(InvalidRequest, match="already exists"):
        memex.create(ALICE, "alice/memory/daily/d.md", "# Mine\n")
    assert remote.head() == theirs
    assert remote.show("alice/memory/daily/d.md") == "# Theirs\n"


def test_a_delete_of_a_note_changed_meanwhile_is_refused(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    version = memex.read(ALICE, "alice/memory/zebra.md")["version"]
    theirs = writer_commit(origin, "theirs", {"alice/memory/zebra.md": "# Changed\n"})
    with pytest.raises(InvalidRequest, match="changed since you read it"):
        memex.delete(ALICE, "alice/memory/zebra.md", version)
    assert remote.head() == theirs


def test_after_the_last_try_the_commit_stays_to_be_pushed_later(
    memex: Memex, remote: Remote, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def turned_down(_self: GitRepo, ref: str = "HEAD", branch: str | None = None) -> Push:
        calls.append(f"{ref}:{branch}")
        return Push.REJECTED

    head = remote.head()
    with monkeypatch.context() as patch:
        patch.setattr(GitRepo, "push", turned_down)
        result = memex.append(ALICE, "alice/memory/zebra.md", "- line")
    assert len(calls) == 1 + PUSH_RETRIES
    assert (result["pushed"], result.get("notice")) == (False, NOT_PUSHED)
    assert remote.head() == head
    assert git(memex.root, "rev-parse", "HEAD").strip() == result["commit"]
    assert memex.refresh() is True
    assert remote.head() == result["commit"]


def test_writes_from_several_callers_at_once_all_arrive(memex: Memex, remote: Remote) -> None:
    errors: list[BaseException] = []

    def write(n: int) -> None:
        try:
            identity = ALICE if n % 2 else BOB
            area = identity.username
            memex.create(identity, f"{area}/memory/daily/n{n}.md", f"# N{n}\n")
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion below
            errors.append(exc)

    head = remote.head()
    threads = [threading.Thread(target=write, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    bare = remote.origin.bare
    assert git(bare, "rev-list", "--count", f"{head}..main").strip() == "6"
    # One line of history: no merges.
    assert git(bare, "rev-list", "--merges", f"{head}..main").strip() == ""


# -- an unreachable remote -----------------------------------------------


def test_an_unreachable_remote_keeps_the_commit_and_the_loop_pushes_it(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    head, pushes = remote.head(), remote.pushes()
    break_remote(memex, origin)
    result = memex.create(ALICE, "alice/memory/daily/offline.md", "# Offline\n\nokapi\n")
    assert (result["pushed"], result.get("notice")) == (False, NOT_PUSHED)
    assert result["searchable"] is True
    assert paths(memex, "okapi") == ["alice/memory/daily/offline.md"]
    assert (remote.head(), remote.pushes()) == (head, pushes)
    assert memex.refresh() is False
    mend_remote(memex, origin)
    assert memex.refresh() is True
    assert remote.head() == result["commit"]
    assert memex.health()["repo"] == "ok"


def test_saved_commits_are_rebased_onto_a_remote_that_moved_on(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    break_remote(memex, origin)
    mine = memex.append(ALICE, "alice/memory/zebra.md", "- mine")
    theirs = writer_commit(origin, "theirs", {"bob/memory/garden.md": "# Garden\n\nbeans\n"})
    mend_remote(memex, origin)
    assert memex.refresh() is True
    assert git(origin.bare, "rev-parse", "main^").strip() == theirs
    assert remote.log_of("%s|%an|%cn") == "memex-mcp: append alice/memory/zebra.md|alice|memex-mcp"
    assert remote.show("alice/memory/zebra.md").endswith("- mine\n")
    assert remote.head() != mine["commit"]  # rebased: a new commit, same change
    assert git(memex.root, "rev-parse", "HEAD").strip() == remote.head()


def test_a_later_write_while_unreachable_keeps_the_earlier_one(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    break_remote(memex, origin)
    memex.append(ALICE, "alice/memory/zebra.md", "- first")
    memex.append(ALICE, "alice/memory/zebra.md", "- second")
    mend_remote(memex, origin)
    assert memex.refresh() is True
    assert remote.show("alice/memory/zebra.md").endswith("- first\n- second\n")


def test_a_turned_down_push_keeps_earlier_saved_commits(
    memex: Memex, origin: Origin, remote: Remote
) -> None:
    break_remote(memex, origin)
    memex.append(ALICE, "alice/memory/zebra.md", "- saved earlier")
    writer_commit(origin, "theirs", {"bob/memory/garden.md": "# Garden\n\nbeans\n"})
    mend_remote(memex, origin)
    result = memex.append(ALICE, "household/memory/wifi.md", "- now")
    assert result["pushed"] is True
    assert remote.show("alice/memory/zebra.md").endswith("- saved earlier\n")
    assert remote.show("household/memory/wifi.md").endswith("- now\n")
    assert remote.show("bob/memory/garden.md") == "# Garden\n\nbeans\n"


# -- conflicts -----------------------------------------------------------


def conflict_branches(origin: Origin) -> list[str]:
    out = git(origin.bare, "for-each-ref", "--format=%(refname:short)", "refs/heads/conflict/")
    return out.split()


def make_conflict(memex: Memex, origin: Origin) -> str:
    """A saved edit and someone else's edit of the same line; returns the saved commit."""
    break_remote(memex, origin)
    mine = memex.edit(ALICE, "alice/memory/zebra.md", "alice private note", "mine")
    writer_commit(origin, "theirs", {"alice/memory/zebra.md": "# Alice zebra\n\ntheirs\n"})
    mend_remote(memex, origin)
    return mine["commit"]


def test_a_conflicting_saved_commit_goes_to_a_conflict_branch(
    memex: Memex, origin: Origin, remote: Remote, caplog: pytest.LogCaptureFixture
) -> None:
    mine = make_conflict(memex, origin)
    theirs = remote.head()
    with caplog.at_level(logging.ERROR, logger="memex_mcp.service"):
        assert memex.refresh() is True
    day = datetime.now(UTC).date().isoformat()
    branch = f"{CONFLICT_PREFIX}{day}"
    assert conflict_branches(origin) == [branch]
    assert git(origin.bare, "rev-parse", branch).strip() == mine
    assert remote.head() == theirs
    # The clone follows the remote again and keeps serving.
    assert git(memex.root, "rev-parse", "HEAD").strip() == theirs
    assert paths(memex, "theirs") == ["alice/memory/zebra.md"]
    assert memex.health()["repo"] == "conflict"
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert branch in errors[0]
    # Writes go on.
    assert memex.append(ALICE, "alice/memory/zebra.md", "- later")["pushed"] is True
    assert memex.health()["repo"] == "conflict"


def test_the_conflict_state_ends_when_its_branch_is_gone(memex: Memex, origin: Origin) -> None:
    make_conflict(memex, origin)
    memex.refresh()
    assert memex.health()["repo"] == "conflict"
    (branch,) = conflict_branches(origin)
    git(origin.bare, "branch", "-D", branch)
    assert memex.refresh() is True
    assert memex.health()["repo"] == "ok"


def test_a_second_conflict_on_the_same_day_gets_its_own_branch(
    memex: Memex, origin: Origin
) -> None:
    make_conflict(memex, origin)
    memex.refresh()
    break_remote(memex, origin)
    memex.edit(ALICE, "alice/memory/zebra.md", "theirs", "mine again")
    writer_commit(origin, "theirs again", {"alice/memory/zebra.md": "# Alice zebra\n\nagain\n"})
    mend_remote(memex, origin)
    memex.refresh()
    day = datetime.now(UTC).date().isoformat()
    assert conflict_branches(origin) == [f"{CONFLICT_PREFIX}{day}", f"{CONFLICT_PREFIX}{day}-2"]


def test_the_conflict_state_survives_a_restart(memex: Memex, origin: Origin) -> None:
    make_conflict(memex, origin)
    memex.refresh()
    memex.index.close()
    again = Memex.from_config(memex.config, None)
    again.prepare()
    try:
        assert again.health()["repo"] == "conflict"
    finally:
        again.index.close()


# -- dates ---------------------------------------------------------------


def commit_at(origin: Origin, when: str, notes: dict[str, str]) -> None:
    git(origin.work, "pull", "-q", "--ff-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("GIT_AUTHOR_DATE", when)
        patch.setenv("GIT_COMMITTER_DATE", when)
        origin.commit(f"at {when}", notes)


FIRST = "2026-03-01T08:00:00+01:00"
SECOND = "2026-04-02T09:30:00+02:00"


def test_read_and_search_give_the_dates_of_the_first_and_last_commit(
    memex: Memex, origin: Origin
) -> None:
    commit_at(origin, FIRST, {"alice/memory/tapir.md": "# Tapir\n\ntapir one\n"})
    commit_at(origin, SECOND, {"alice/memory/tapir.md": "# Tapir\n\ntapir two\n"})
    memex.refresh()
    note = memex.read(ALICE, "alice/memory/tapir.md")
    assert (note["created"], note["updated"]) == (FIRST, SECOND)
    (hit,) = memex.search(ALICE, "tapir")["hits"]
    assert (hit["created"], hit["updated"]) == (FIRST, SECOND)


def test_a_full_build_finds_the_same_dates(memex: Memex, origin: Origin) -> None:
    commit_at(origin, FIRST, {"alice/memory/tapir.md": "# Tapir\n\ntapir one\n"})
    commit_at(origin, SECOND, {"alice/memory/tapir.md": "# Tapir\n\ntapir two\n"})
    memex.refresh()
    memex.index.close()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{memex.config.index.path}{suffix}").unlink(missing_ok=True)
    fresh = Memex.from_config(memex.config, None)
    fresh.prepare()
    try:
        assert fresh.index.fresh is True
        note = fresh.read(ALICE, "alice/memory/tapir.md")
        assert (note["created"], note["updated"]) == (FIRST, SECOND)
    finally:
        fresh.index.close()


def test_a_write_moves_the_updated_date_only(memex: Memex, origin: Origin) -> None:
    commit_at(origin, FIRST, {"alice/memory/tapir.md": "# Tapir\n\ntapir\n"})
    memex.refresh()
    memex.append(ALICE, "alice/memory/tapir.md", "- more")
    note = memex.read(ALICE, "alice/memory/tapir.md")
    assert note["created"] == FIRST
    assert note["updated"] is not None
    assert datetime.fromisoformat(note["updated"]) > datetime.fromisoformat(FIRST)
    created = memex.create(ALICE, "alice/memory/daily/new.md", "# New\n")
    new = memex.read(ALICE, created["path"])
    assert new["created"] == new["updated"] is not None


def test_a_note_without_dates_reads_none(memex: Memex) -> None:
    memex.index.set_dates({}, ["alice/memory/zebra.md"])
    note = memex.read(ALICE, "alice/memory/zebra.md")
    assert (note["created"], note["updated"]) == (None, None)


def test_dates_cost_no_git_call_at_query_time(
    memex: Memex, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_git(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("git was called")

    monkeypatch.setattr(GitRepo, "_run", no_git)
    hits = memex.search(ALICE, "zebracode")["hits"]
    assert hits
    assert all(hit["created"] is not None for hit in hits)
    assert memex.read(ALICE, "alice/memory/zebra.md")["created"] is not None
