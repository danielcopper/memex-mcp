"""Search: ranking, archive down-weighting, hybrid fusion, keyword fallback, snippets."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from memex_mcp.embed import Embedder
from memex_mcp.service import (
    ELLIPSIS,
    NOTICE_DISABLED,
    NOTICE_UNAVAILABLE,
    InvalidRequest,
    Memex,
    SearchResult,
    make_snippet,
)
from tests.conftest import (
    ALICE,
    FailingEmbedder,
    FakeEmbedder,
    Origin,
    make_config,
    malformed_ollama,
)


class Clock:
    def __init__(self) -> None:
        self.now: float = 100.0

    def __call__(self) -> float:
        return self.now


def service(
    tmp_path: Path, origin: Origin, embedder: Embedder | None, **sections: dict[str, object]
) -> Memex:
    memex = Memex.from_config(make_config(tmp_path, origin, **sections), embedder=embedder)
    memex.prepare()
    return memex


@pytest.fixture
def hybrid(tmp_path: Path, origin: Origin) -> Iterator[Memex]:
    memex = service(tmp_path, origin, FakeEmbedder())
    memex.backfill()
    yield memex
    memex.index.close()


def paths(result: SearchResult) -> list[str]:
    return [hit["path"] for hit in result["hits"]]


def test_keyword_ranking_puts_the_best_match_first(memex: Memex) -> None:
    result = memex.search(ALICE, "degreaser tube spare")
    assert paths(result)[0] == "alice/memory/bike-repair.md"
    hit = result["hits"][0]
    assert hit["title"] == "Bike repair"
    assert hit["area"] == "alice"
    assert hit["archived"] is False
    assert hit["score"] > 0


def test_one_hit_per_note(memex: Memex) -> None:
    found = paths(memex.search(ALICE, "degreaser", limit=50))
    assert len(found) == len(set(found))


def test_archived_notes_rank_lower_and_are_flagged(tmp_path: Path, origin: Origin) -> None:
    text = "# Printer\n\nThe printer toner is in the hallway cupboard.\n"
    origin.commit(
        "same note twice",
        {"alice/memory/printer.md": text, "alice/memory/archive/printer.md": text},
    )
    memex = service(tmp_path, origin, None, index={"archive_factor": 0.5})
    try:
        hits = memex.search(ALICE, "printer toner")["hits"]
        assert [h["path"] for h in hits[:2]] == [
            "alice/memory/printer.md",
            "alice/memory/archive/printer.md",
        ]
        current, archived = hits[0], hits[1]
        assert archived["archived"] is True and current["archived"] is False
        assert archived["score"] == pytest.approx(current["score"] * 0.5, rel=0.05)
    finally:
        memex.index.close()


def test_an_archived_note_still_wins_when_only_it_matches(memex: Memex) -> None:
    result = memex.search(ALICE, "diesel")
    assert paths(result) == ["alice/memory/archive/old-bike.md"]
    assert result["hits"][0]["archived"] is True


def test_semantic_search_finds_a_note_without_shared_words(hybrid: Memex) -> None:
    result = hybrid.search(ALICE, "internet wlan")
    assert result["semantic"] is True
    assert "notice" not in result
    assert "household/memory/wifi.md" in paths(result)
    # Keyword search alone does not find it: the note shares no word with the query.
    assert hybrid.index.keyword_ranked("internet wlan", frozenset({"household"}), 50) == [[]]


def test_hybrid_keeps_keyword_hits(hybrid: Memex) -> None:
    result = hybrid.search(ALICE, "fridge")
    assert result["semantic"] is True
    assert paths(result)[0] == "household/memory/wifi.md"


def test_failing_embedder_falls_back_to_keywords(tmp_path: Path, origin: Origin) -> None:
    embedder = FailingEmbedder()
    memex = service(tmp_path, origin, embedder)
    try:
        result = memex.search(ALICE, "degreaser")
        assert result["semantic"] is False
        assert result.get("notice") == NOTICE_UNAVAILABLE
        assert paths(result)[0] == "alice/memory/bike-repair.md"
    finally:
        memex.index.close()


def test_after_a_failure_the_embedder_rests_for_a_while(tmp_path: Path, origin: Origin) -> None:
    embedder = FailingEmbedder()
    clock = Clock()
    memex = service(tmp_path, origin, embedder, embeddings={"retry_after_seconds": 60.0})
    memex.clock = clock
    try:
        memex.search(ALICE, "degreaser")
        assert embedder.calls == 1
        clock.now += 30
        assert memex.search(ALICE, "degreaser")["semantic"] is False
        assert embedder.calls == 1  # skipped: still resting
        clock.now += 31
        memex.search(ALICE, "degreaser")
        assert embedder.calls == 2
    finally:
        memex.index.close()


def test_a_malformed_embedder_answer_falls_back_to_keywords(tmp_path: Path, origin: Origin) -> None:
    embedder = malformed_ollama()
    memex = service(tmp_path, origin, embedder)
    try:
        result = memex.search(ALICE, "degreaser")
        assert result["semantic"] is False
        assert result.get("notice") == NOTICE_UNAVAILABLE
        assert paths(result)[0] == "alice/memory/bike-repair.md"
    finally:
        memex.index.close()
        embedder.close()


def test_disabled_embeddings_say_so(memex: Memex) -> None:
    result = memex.search(ALICE, "degreaser")
    assert result["semantic"] is False
    assert result.get("notice") == NOTICE_DISABLED


def test_empty_query_is_refused(memex: Memex) -> None:
    with pytest.raises(InvalidRequest):
        memex.search(ALICE, "   ")


@pytest.mark.parametrize(
    "query", ['"unbalanced', "NEAR(a b", "a AND OR b", "*", "col:value", "-x", "(((", "^"]
)
def test_fts_syntax_in_queries_is_harmless(memex: Memex, query: str) -> None:
    result = memex.search(ALICE, query)
    assert isinstance(result["hits"], list)


def test_limit_is_respected(memex: Memex) -> None:
    assert len(memex.search(ALICE, "the", limit=2)["hits"]) <= 2
    assert len(memex.search(ALICE, "the", limit=0)["hits"]) <= 1


def test_snippet_is_short_and_centred_on_the_match(tmp_path: Path, origin: Origin) -> None:
    filler = " ".join(f"word{i}" for i in range(400))
    origin.commit(
        "long note",
        {"alice/memory/long.md": f"# Long\n\n{filler} needle-in-the-hay {filler}\n"},
    )
    memex = service(tmp_path, origin, None, index={"chunk_chars": 20000})
    try:
        hit = memex.search(ALICE, "needle")["hits"][0]
        assert hit["path"] == "alice/memory/long.md"
        assert len(hit["snippet"]) <= 300
        assert "needle" in hit["snippet"]
        assert hit["snippet"].startswith(ELLIPSIS) and hit["snippet"].endswith(ELLIPSIS)
    finally:
        memex.index.close()


def test_make_snippet_edges() -> None:
    assert make_snippet("short  text\n here", ["text"], 300) == "short text here"
    text = "alpha " * 100 + "omega"
    snippet = make_snippet(text, ["omega"], 100)
    assert len(snippet) <= 100 and snippet.endswith("omega") and snippet.startswith(ELLIPSIS)
    # Without spaces to trim at, the window itself must leave room for both ellipses.
    unbroken = make_snippet("a" * 500 + "needle" + "b" * 500, ["needle"], 100)
    assert len(unbroken) == 100 and "needle" in unbroken
    start = make_snippet(text, ["nothing"], 100)
    assert len(start) <= 100 and start.startswith("alpha") and start.endswith(ELLIPSIS)


def test_overlong_queries_are_refused_with_a_clear_message(memex: Memex) -> None:
    with pytest.raises(InvalidRequest, match="at most 500 characters"):
        memex.search(ALICE, "a" * 501)
    with pytest.raises(InvalidRequest, match="at most 32 words"):
        memex.search(ALICE, " ".join(f"w{i}" for i in range(33)))
    assert isinstance(memex.search(ALICE, "b" * 500)["hits"], list)
    assert isinstance(memex.search(ALICE, " ".join(f"w{i}" for i in range(32)))["hits"], list)
