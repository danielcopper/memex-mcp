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


# A closed document, so that nothing but the depth can fail it.
NESTED_TOO_DEEPLY = b"[" * 100_000 + b"]" * 100_000


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
        httpx2.Response(200, json={"embeddings": [[1e-30, 0, 0]]}),
        httpx2.Response(200, json={"embeddings": [[1e-46, 0, 0]]}),
        httpx2.Response(200, json={"embeddings": [[1.9e19, 0, 0]]}),
        httpx2.Response(200, json={"model": "bge-m3", "error": "no such route"}),
        raw_answer(NESTED_TOO_DEEPLY),
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
        "vector-near-zero",
        "vector-subnormal",
        "vector-too-long",
        "no-embeddings-member",
        "nested-too-deeply",
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
        (
            httpx2.Response(200, json={"embeddings": [[0, 0, 0]]}),
            ["a"],
            "vector 0 has (almost) no length",
        ),
        (
            httpx2.Response(200, json={"embeddings": [[0, 1e-30, 0]]}),
            ["a"],
            "vector 0 has (almost) no length",
        ),
        (
            httpx2.Response(200, json={"embeddings": [[0, 1.9e19, 0]]}),
            ["a"],
            "vector 0 is too long: its squared length overflows float32",
        ),
        (raw_answer(NESTED_TOO_DEEPLY), ["a"], "the answer is nested too deeply"),
        (
            httpx2.Response(200, json={"model": "bge-m3", "error": "no such route"}),
            ["a"],
            "the answer has no 'embeddings' member (it has: 'error', 'model'); "
            + "is [embeddings] url Ollama's API?",
        ),
    ],
    ids=[
        "count",
        "embeddings-null",
        "vector-number",
        "dimensions",
        "type",
        "range",
        "zero",
        "near-zero",
        "too-long",
        "nested-too-deeply",
        "no-embeddings-member",
    ],
)
def test_embedding_errors_name_the_problem(
    response: httpx2.Response, texts: list[str], problem: str
) -> None:
    with pytest.raises(EmbeddingError, match=re.escape(problem)):
        embedder(httpx2.MockTransport(lambda _request: response)).embed(texts, timeout=1.0)


def test_a_vector_just_inside_float32_is_accepted() -> None:
    # Its squared length, 3.24e38, is just below float32's largest value.
    response = httpx2.Response(200, json={"embeddings": [[1.8e19, 0, 0]]})
    vectors = embedder(httpx2.MockTransport(lambda _request: response)).embed(["a"], timeout=1.0)
    assert vectors == [[1.8e19, 0.0, 0.0]]


def message_for(response: httpx2.Response) -> str:
    with pytest.raises(EmbeddingError) as raised:
        embedder(httpx2.MockTransport(lambda _request: response)).embed(["a"], timeout=1.0)
    return str(raised.value)


def test_member_names_are_escaped_and_at_most_ten() -> None:
    members = {f"m{index:02}": 1 for index in range(14)} | {"a\nforged": 1, "b" * 100: 1}
    assert message_for(httpx2.Response(200, json=members)) == (
        "the answer has no 'embeddings' member (it has: 'a\\nforged', "
        + repr("b" * 64 + "…")
        + ", "
        + ", ".join(repr(f"m{index:02}") for index in range(8))
        + " and 6 more); is [embeddings] url Ollama's API?"
    )


OLLAMA_ERROR = 'model "bge-m3" not found, try pulling it first'


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            httpx2.Response(404, json={"error": OLLAMA_ERROR}),
            f"HTTP 404 Not Found; Ollama says {OLLAMA_ERROR!r}",
        ),
        (
            httpx2.Response(500, json={"error": "line\nforged" + "x" * 300}),
            "HTTP 500 Internal Server Error; Ollama says "
            + repr("line\nforged" + "x" * (200 - len("line\nforged")) + "…"),
        ),
        (httpx2.Response(500, text="model not loaded"), "HTTP 500 Internal Server Error"),
        (httpx2.Response(500, json={"error": 5}), "HTTP 500 Internal Server Error"),
        (httpx2.Response(500, json=["error"]), "HTTP 500 Internal Server Error"),
        (httpx2.Response(500, content=NESTED_TOO_DEEPLY), "HTTP 500 Internal Server Error"),
    ],
    ids=[
        "model-missing",
        "escaped-and-cut",
        "not-json",
        "error-not-text",
        "not-an-object",
        "nested-too-deeply",
    ],
)
def test_an_error_status_carries_ollamas_own_error_text(
    response: httpx2.Response, message: str
) -> None:
    assert message_for(response) == message


LOCATION = "http://ollama.example.org:11434/api/embed/"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            httpx2.Response(500, extensions={"reason_phrase": b"\x1b[31mforged" + b"x" * 1000}),
            "HTTP 500 Internal Server Error",
        ),
        (httpx2.Response(599), "HTTP 599"),
        (
            httpx2.Response(308, headers={"location": LOCATION}),
            f"HTTP 308 Permanent Redirect; redirected to {LOCATION!r}",
        ),
        (
            httpx2.Response(302, headers={"location": "/x'\\" + "y" * 300}),
            "HTTP 302 Found; redirected to " + repr("/x'\\" + "y" * 196 + "…"),
        ),
        (httpx2.Response(304), "HTTP 304 Not Modified"),
        (httpx2.Response(300, headers={"location": LOCATION}), "HTTP 300 Multiple Choices"),
    ],
    ids=[
        "server-phrase",
        "unknown-code",
        "redirect",
        "redirect-escaped-and-cut",
        "no-location",
        "multiple-choices",
    ],
)
def test_an_error_status_is_named_by_its_code(response: httpx2.Response, message: str) -> None:
    assert message_for(response) == message


def raising(exc: Exception) -> httpx2.MockTransport:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise exc

    return httpx2.MockTransport(handler)


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (
            # h11 quotes the bytes of a bad line from the wire; they can run to hundreds of KB.
            httpx2.RemoteProtocolError("illegal header line: \x1b[31m" + "x" * 50_000),
            "RemoteProtocolError: illegal header line: \\x1b[31m"
            + "x" * (200 - len("illegal header line: \x1b[31m"))
            + "…",
        ),
        (httpx2.ConnectError(""), "ConnectError: no detail"),
        # Not the answer: the request itself ran out of stack.
        (RecursionError("maximum recursion depth"), "RecursionError: maximum recursion depth"),
    ],
    ids=["long-protocol-error", "no-text", "recursion-while-sending"],
)
def test_a_failed_request_names_its_cause_escaped_and_cut(exc: Exception, message: str) -> None:
    with pytest.raises(EmbeddingError) as raised:
        embedder(raising(exc)).embed(["a"], timeout=1.0)
    assert str(raised.value) == message


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
