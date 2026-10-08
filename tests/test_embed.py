"""The Ollama client, against a mocked /api/embed (no network)."""

from __future__ import annotations

import json

import httpx2
import pytest

from memex_mcp.embed import EmbeddingError, OllamaEmbedder


def embedder(handler: httpx2.MockTransport) -> OllamaEmbedder:
    return OllamaEmbedder("http://ollama.example.org:11434/", "bge-m3", 3, transport=handler)


def test_embeds_a_batch_in_order() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
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
    ],
    ids=["http-500", "not-json", "wrong-dimensions", "missing-vector", "not-an-object"],
)
def test_bad_answers_raise_embedding_error(response: httpx2.Response) -> None:
    with pytest.raises(EmbeddingError):
        embedder(httpx2.MockTransport(lambda _request: response)).embed(["a"], timeout=1.0)


def test_unreachable_and_slow_hosts_raise_embedding_error() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    for handler in (refuse, slow):
        with pytest.raises(EmbeddingError):
            embedder(httpx2.MockTransport(handler)).embed(["a"], timeout=0.1)


def test_nothing_to_embed_makes_no_request() -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request expected")

    assert embedder(httpx2.MockTransport(fail)).embed([], timeout=1.0) == []
