"""The Ollama client, against a mocked /api/embed (no network)."""

from __future__ import annotations

import json
import re
from typing import cast

import httpx2
import pytest

from memex_mcp.embed import EmbeddingError, OllamaEmbedder


def raw_answer(body: bytes) -> httpx2.Response:
    """An answer whose body is not strict JSON (NaN, Infinity) but parses in Python."""
    return httpx2.Response(200, content=body, headers={"content-type": "application/json"})


def embedder(handler: httpx2.MockTransport) -> OllamaEmbedder:
    return OllamaEmbedder("http://ollama.example.org:11434/", "bge-m3", 3, transport=handler)


def test_embeds_a_batch_in_order() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/embed"
        body = cast("object", json.loads(request.content))
        assert body == {"model": "bge-m3", "input": ["a", "bb"]}
        return httpx2.Response(
            200, json={"model": "bge-m3", "embeddings": [[1, 0, 0], [0, 1, 0.5]]}
        )

    assert embedder(httpx2.MockTransport(handler)).embed(["a", "bb"], timeout=1.0) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.5],
    ]


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(500, text="model not loaded"),
        httpx2.Response(200, text="not json"),
        httpx2.Response(200, json={"embeddings": [[1, 0]]}),
        httpx2.Response(200, json={"embeddings": []}),
        httpx2.Response(200, json=["unexpected"]),
        httpx2.Response(200, json={"embeddings": None}),
        httpx2.Response(200, json={"embeddings": [None]}),
        httpx2.Response(200, json={"embeddings": [[1, None, 0]]}),
        httpx2.Response(200, json={"embeddings": [[1, "a", 0]]}),
        httpx2.Response(200, json={"embeddings": [["1", "0", "0"]]}),
        httpx2.Response(200, json={"embeddings": [[1, True, 0]]}),
        httpx2.Response(200, json={"embeddings": [[1, [0], 0]]}),
        raw_answer(b'{"embeddings": [[1, NaN, 0]]}'),
        raw_answer(b'{"embeddings": [[1, -Infinity, 0]]}'),
        httpx2.Response(200, json={"embeddings": [[1, 1e39, 0]]}),
        raw_answer(b'{"embeddings": [[1, 1' + b"0" * 400 + b", 0]]}"),
        httpx2.Response(200, json={"embeddings": [[0, 0.0, -0.0]]}),
    ],
    ids=[
        "http-500",
        "not-json",
        "wrong-dimensions",
        "missing-vector",
        "not-an-object",
        "embeddings-null",
        "vector-null",
        "value-null",
        "value-text",
        "values-numeric-text",
        "value-boolean",
        "value-list",
        "value-nan",
        "value-infinite",
        "value-too-large",
        "value-huge-int",
        "vector-zero",
    ],
)
def test_bad_answers_raise_embedding_error(response: httpx2.Response) -> None:
    with pytest.raises(EmbeddingError):
        embedder(httpx2.MockTransport(lambda _request: response)).embed(["a"], timeout=1.0)


@pytest.mark.parametrize(
    ("response", "texts", "problem"),
    [
        (
            httpx2.Response(200, json={"embeddings": [[1, 0, 0]]}),
            ["a", "b"],
            "1 vectors for 2 texts",
        ),
        (httpx2.Response(200, json={"embeddings": None}), ["a"], "embeddings is null, not a list"),
        (httpx2.Response(200, json={"embeddings": [7]}), ["a"], "vector 0 is a number, not a list"),
        (
            httpx2.Response(200, json={"embeddings": [[1, 0]]}),
            ["a"],
            "vector 0 has 2 values, expected 3",
        ),
        (
            httpx2.Response(200, json={"embeddings": [[1, "a", 0]]}),
            ["a"],
            "value 1 of vector 0 is a string",
        ),
        (
            httpx2.Response(200, json={"embeddings": [[1, 0, 1e39]]}),
            ["a"],
            "value 2 of vector 0 is 1e+39",
        ),
        (httpx2.Response(200, json={"embeddings": [[0, 0, 0]]}), ["a"], "vector 0 is all zeros"),
    ],
    ids=["count", "embeddings-null", "vector-number", "dimensions", "type", "range", "zero"],
)
def test_embedding_errors_name_the_problem(
    response: httpx2.Response, texts: list[str], problem: str
) -> None:
    with pytest.raises(EmbeddingError, match=re.escape(problem)):
        embedder(httpx2.MockTransport(lambda _request: response)).embed(texts, timeout=1.0)


def test_unreachable_and_slow_hosts_raise_embedding_error() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    for handler in (refuse, slow):
        with pytest.raises(EmbeddingError):
            embedder(httpx2.MockTransport(handler)).embed(["a"], timeout=0.1)


def test_nothing_to_embed_makes_no_request() -> None:
    def fail(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request expected")

    assert embedder(httpx2.MockTransport(fail)).embed([], timeout=1.0) == []
