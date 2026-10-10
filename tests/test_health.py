"""The health state /healthz reports: the embedder and the clone's refresh."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import pytest

from memex_mcp.embed import Embedder, EmbeddingError
from memex_mcp.service import CHECK_TEXT, Memex
from tests.conftest import ALICE, FakeEmbedder, Origin, git, make_config

QUERY_TIMEOUT = 0.5


@dataclass
class Switchable(Embedder):
    """An embedder whose real calls and checks fail while told to; records both."""

    model: str = "fake-embedder"
    dimensions: int = 16
    embed_fails: bool = False
    check_fails: bool = False
    embedded: list[tuple[list[str], float]] = field(default_factory=list[tuple[list[str], float]])
    checks: list[float] = field(default_factory=list[float])

    @override
    def embed(self, texts: list[str], timeout: float) -> list[list[float]]:
        self.embedded.append((texts, timeout))
        if self.embed_fails:
            raise EmbeddingError("ReadTimeout: timed out")
        return FakeEmbedder().embed(texts, timeout)

    @override
    def check(self, timeout: float) -> None:
        self.checks.append(timeout)
        if self.check_fails:
            raise EmbeddingError("Ollama does not list the model 'fake-embedder'")


class Clock:
    def __init__(self) -> None:
        self.now: float = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def embedder() -> Switchable:
    return Switchable()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def service(tmp_path: Path, origin: Origin, embedder: Switchable, clock: Clock) -> Iterator[Memex]:
    """A prepared service with a working embedder; every chunk still lacks its vector."""
    config = make_config(tmp_path, origin, embeddings={"query_timeout_seconds": QUERY_TIMEOUT})
    memex = Memex.from_config(config, embedder=embedder)
    memex.clock = clock
    memex.prepare()
    yield memex
    memex.index.close()


def embeddings(memex: Memex) -> str:
    return memex.health()["embeddings"]


def repo(memex: Memex) -> str:
    return memex.health()["repo"]


def fail_a_search(memex: Memex, embedder: Switchable, clock: Clock) -> None:
    """A search whose query embedding fails; the clock then moves past the retry pause."""
    embedder.embed_fails = True
    assert memex.search(ALICE, "degreaser")["semantic"] is False
    clock.now += memex.config.embeddings.retry_after_seconds + 1


# -- embeddings --------------------------------------------------------


def test_embeddings_off_read_disabled(memex: Memex) -> None:
    assert memex.health() == {"status": "ok", "embeddings": "disabled", "repo": "ok"}
    memex.check_embeddings()
    assert memex.health() == {"status": "ok", "embeddings": "disabled", "repo": "ok"}


def test_a_working_embedder_reads_ok(service: Memex) -> None:
    assert service.health() == {"status": "ok", "embeddings": "ok", "repo": "ok"}
    service.check_embeddings()
    assert embeddings(service) == "ok"


def test_a_failed_search_reads_failing_until_a_search_embeds(
    service: Memex, embedder: Switchable, clock: Clock
) -> None:
    fail_a_search(service, embedder, clock)
    assert embeddings(service) == "failing"
    embedder.embed_fails = False
    assert service.search(ALICE, "degreaser")["semantic"] is True
    assert embeddings(service) == "ok"


def test_a_search_resting_after_a_failure_changes_nothing(
    service: Memex, embedder: Switchable
) -> None:
    embedder.embed_fails = True
    service.search(ALICE, "degreaser")
    embedder.embed_fails = False
    # Still resting: no call is made, so nothing shows that embedding works again.
    service.search(ALICE, "degreaser")
    assert len(embedder.embedded) == 1
    assert embeddings(service) == "failing"


def test_a_failed_backfill_reads_failing_until_a_backfill_embeds(
    service: Memex, embedder: Switchable
) -> None:
    embedder.embed_fails = True
    assert service.backfill() == 0
    assert embeddings(service) == "failing"
    embedder.embed_fails = False
    assert service.backfill() > 0
    assert embeddings(service) == "ok"


def test_a_failed_check_reads_failing_until_a_check_passes(
    service: Memex, embedder: Switchable
) -> None:
    embedder.check_fails = True
    service.check_embeddings()
    assert embeddings(service) == "failing"
    embedder.check_fails = False
    service.check_embeddings()
    assert embeddings(service) == "ok"
    # The model list was asked each time; nothing was embedded.
    assert embedder.checks == [QUERY_TIMEOUT, QUERY_TIMEOUT]
    assert embedder.embedded == []


def test_a_real_embedding_clears_a_failed_check(service: Memex, embedder: Switchable) -> None:
    embedder.check_fails = True
    service.check_embeddings()
    assert embeddings(service) == "failing"
    assert service.search(ALICE, "degreaser")["semantic"] is True
    assert embeddings(service) == "ok"


def test_while_a_real_call_has_failed_the_check_embeds_a_short_text(
    service: Memex, embedder: Switchable, clock: Clock
) -> None:
    fail_a_search(service, embedder, clock)
    embedder.embed_fails = False
    service.check_embeddings()
    assert embedder.embedded[-1] == ([CHECK_TEXT], QUERY_TIMEOUT)
    assert embedder.checks == []
    assert embeddings(service) == "ok"
    # Once it works again, the check asks the model list again.
    service.check_embeddings()
    assert embedder.checks == [QUERY_TIMEOUT]
    assert embedder.embedded[-1] == ([CHECK_TEXT], QUERY_TIMEOUT)
    assert len(embedder.embedded) == 2


def test_while_a_real_call_fails_the_check_keeps_embedding_and_stays_failing(
    service: Memex, embedder: Switchable, clock: Clock
) -> None:
    # Ollama lists the model, but its answers are refused (too large, too late).
    fail_a_search(service, embedder, clock)
    for _ in range(3):
        service.check_embeddings()
        assert embeddings(service) == "failing"
    assert embedder.checks == []
    assert [texts for texts, _ in embedder.embedded[1:]] == [[CHECK_TEXT]] * 3


def test_a_recovery_by_the_check_ends_the_retry_pause(service: Memex, embedder: Switchable) -> None:
    embedder.embed_fails = True
    assert service.search(ALICE, "degreaser")["semantic"] is False
    embedder.embed_fails = False
    service.check_embeddings()
    # The clock has not moved: without the recovery, searches would still skip the embedder.
    assert service.search(ALICE, "degreaser")["semantic"] is True


def test_a_backfill_clears_the_state_but_not_the_retry_pause(
    service: Memex, embedder: Switchable
) -> None:
    embedder.embed_fails = True
    assert service.search(ALICE, "degreaser")["semantic"] is False
    embedder.embed_fails = False
    assert service.backfill() > 0
    assert embeddings(service) == "ok"
    calls = len(embedder.embedded)
    # The clock has not moved: a batch that took the longer index timeout says nothing
    # about a search's, so searches keep skipping the embedder until the pause is over.
    assert service.search(ALICE, "degreaser")["semantic"] is False
    assert len(embedder.embedded) == calls


def test_a_failure_is_logged_once_and_the_recovery_once(
    service: Memex, embedder: Switchable, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="memex_mcp.service"):
        embedder.check_fails = True
        for _ in range(3):
            service.check_embeddings()
        embedder.check_fails = False
        for _ in range(3):
            service.check_embeddings()
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("WARNING", "the embedder check failed: Ollama does not list the model 'fake-embedder'"),
        ("INFO", "the embedder check passes again"),
    ]


def test_a_failed_real_call_is_not_logged_again_by_the_checks(
    service: Memex, embedder: Switchable, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="memex_mcp.service"):
        fail_a_search(service, embedder, clock)
        for _ in range(3):
            service.check_embeddings()
        embedder.embed_fails = False
        for _ in range(3):
            service.check_embeddings()
        assert service.search(ALICE, "degreaser")["semantic"] is True
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ("WARNING", "query embedding failed, keyword search only: ReadTimeout: timed out"),
        ("INFO", "the embedder works again"),
    ]


# -- refresh rounds ------------------------------------------------------


def break_remote(memex: Memex, origin: Origin) -> None:
    git(memex.root, "remote", "set-url", "origin", str(origin.bare.parent / "gone.git"))


def mend_remote(memex: Memex, origin: Origin) -> None:
    git(memex.root, "remote", "set-url", "origin", str(origin.bare))


def test_a_failed_fetch_reads_failing_until_a_round_fetches(memex: Memex, origin: Origin) -> None:
    memex.refresh_round(first=True)
    assert repo(memex) == "ok"
    break_remote(memex, origin)
    memex.refresh_round(first=False)
    assert repo(memex) == "failing"
    memex.refresh_round(first=False)
    assert repo(memex) == "failing"
    mend_remote(memex, origin)
    memex.refresh_round(first=False)
    assert repo(memex) == "ok"


@pytest.mark.parametrize("first", [True, False], ids=["first-round", "later-round"])
def test_a_round_that_raises_reads_failing_until_one_passes(
    memex: Memex, monkeypatch: pytest.MonkeyPatch, first: bool
) -> None:
    def broken() -> int:
        raise RuntimeError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(memex, "backfill", broken)
        with pytest.raises(RuntimeError, match="database is locked"):
            memex.refresh_round(first)
        assert repo(memex) == "failing"
    memex.refresh_round(first)
    assert repo(memex) == "ok"


def test_every_round_checks_the_embedder_before_filling_in_vectors(
    service: Memex, embedder: Switchable
) -> None:
    calls: list[str] = []
    real_backfill = service.backfill

    def backfill() -> int:
        calls.append(f"backfill after {len(embedder.checks)} checks")
        return real_backfill()

    service.backfill = backfill
    service.refresh_round(first=True)
    service.refresh_round(first=False)
    assert calls == ["backfill after 1 checks", "backfill after 2 checks"]


def test_a_failed_fetch_does_not_touch_the_embedding_state(service: Memex, origin: Origin) -> None:
    break_remote(service, origin)
    service.refresh_round(first=False)
    assert service.health() == {"status": "ok", "embeddings": "ok", "repo": "failing"}
