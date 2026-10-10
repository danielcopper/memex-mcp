"""The Ollama client, against a mocked /api/embed (no network)."""

from __future__ import annotations

import gzip
import itertools
import json
import re
import time
from collections.abc import Iterator
from typing import cast

import httpx2
import pytest

from memex_mcp.embed import EmbeddingError, OllamaEmbedder
from tests.conftest import Trickle, streamed


def raw_answer(body: bytes) -> httpx2.Response:
    """An answer whose body is not strict JSON (NaN, Infinity) but parses in Python."""
    return httpx2.Response(200, content=body, headers={"content-type": "application/json"})


# Deep enough that json runs out of an 8 MB stack; with more stack it parses
# into nested lists. Either way the answer is an EmbeddingError.
NESTED_TOO_DEEPLY = b"[" * 100_000 + b"]" * 100_000
# A request for one vector of this many values admits that document's 200,000
# bytes (its limit is 327,680), so the document reaches the decoder.
NESTED_DIMENSIONS = 8192


def embedder(handler: httpx2.MockTransport, dimensions: int = 3) -> OllamaEmbedder:
    return OllamaEmbedder(
        "http://ollama.example.org:11434/", "bge-m3", dimensions, transport=streamed(handler)
    )


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


def message_for(response: httpx2.Response, timeout: float = 1.0, dimensions: int = 3) -> str:
    client = embedder(httpx2.MockTransport(lambda _request: response), dimensions)
    with pytest.raises(EmbeddingError) as raised:
        client.embed(["a"], timeout=timeout)
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
    ],
    ids=[
        "model-missing",
        "escaped-and-cut",
        "not-json",
        "error-not-text",
        "not-an-object",
    ],
)
def test_an_error_status_carries_ollamas_own_error_text(
    response: httpx2.Response, message: str
) -> None:
    assert message_for(response) == message


@pytest.mark.parametrize(
    ("status", "message"),
    [(200, None), (500, "HTTP 500 Internal Server Error")],
    ids=["answer", "error-answer"],
)
def test_a_deeply_nested_answer_raises_embedding_error(status: int, message: str | None) -> None:
    raised = message_for(
        httpx2.Response(status, content=NESTED_TOO_DEEPLY), dimensions=NESTED_DIMENSIONS
    )
    # The document must reach the decoder, not stop at the size limit.
    assert "larger than" not in raised
    if message is not None:
        assert raised == message


@pytest.mark.usefixtures("too_deep_to_decode")
@pytest.mark.parametrize(
    ("status", "message"),
    [(200, "the answer is nested too deeply"), (500, "HTTP 500 Internal Server Error")],
    ids=["answer", "error-answer"],
)
def test_an_answer_too_deep_to_decode_raises_embedding_error(status: int, message: str) -> None:
    assert message_for(httpx2.Response(status, content=b"[]")) == message


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


# For one text of 3 values: 3 x 32 bytes plus 64 KiB.
LIMIT = 65_632
KIB = b" " * 1024
ANSWER = b'{"embeddings": [[1, 0, 0]]}'


def padded(size: int) -> bytes:
    """A good answer for one text, padded with whitespace to ``size`` bytes."""
    return ANSWER + b" " * (size - len(ANSWER))


@pytest.mark.parametrize("size", [LIMIT - 1, LIMIT], ids=["just-under", "exactly"])
@pytest.mark.parametrize("declared", [True, False], ids=["declared", "read"])
def test_an_answer_within_the_size_limit_is_read(size: int, declared: bool) -> None:
    body = padded(size)
    response = (
        httpx2.Response(200, content=body)  # declares its Content-Length
        if declared
        else httpx2.Response(200, stream=Trickle([body[:1000], body[1000:]]))
    )
    vectors = embedder(httpx2.MockTransport(lambda _request: response)).embed(["a"], timeout=1.0)
    assert vectors == [[1.0, 0.0, 0.0]]


def test_an_answer_past_the_size_limit_is_not_read_further() -> None:
    # 1 KiB at a time, 1000 times the limit on offer: reading stops at the chunk
    # that crosses it, the 65th.
    body = Trickle(itertools.repeat(KIB, 1000 * LIMIT // len(KIB)))
    response = httpx2.Response(200, stream=body)
    assert message_for(response) == "the answer is larger than 65,632 bytes (66,560 bytes received)"
    assert body.sent == LIMIT // len(KIB) + 1


def test_a_declared_size_past_the_limit_is_refused_before_reading() -> None:
    body = Trickle(itertools.repeat(KIB, 5000))
    response = httpx2.Response(200, headers={"content-length": "5000000"}, stream=body)
    assert (
        message_for(response) == "the answer is larger than 65,632 bytes (Content-Length 5,000,000)"
    )
    assert body.sent == 0


def test_an_answer_one_byte_past_the_limit_is_refused() -> None:
    body = Trickle([padded(LIMIT + 1)])
    assert message_for(httpx2.Response(200, stream=body)) == (
        "the answer is larger than 65,632 bytes (65,633 bytes received)"
    )


@pytest.mark.parametrize(
    ("texts", "dimensions", "limit"),
    [(1, 3, "65,632"), (2, 3, "65,728"), (1, 1024, "98,304"), (32, 1024, "1,114,112")],
)
def test_the_size_limit_scales_with_texts_and_dimensions(
    texts: int, dimensions: int, limit: str
) -> None:
    response = httpx2.Response(200, headers={"content-length": "5000000"}, stream=Trickle([]))
    client = embedder(httpx2.MockTransport(lambda _request: response), dimensions)
    with pytest.raises(EmbeddingError) as raised:
        client.embed(["a"] * texts, timeout=1.0)
    assert (
        str(raised.value) == f"the answer is larger than {limit} bytes (Content-Length 5,000,000)"
    )


@pytest.mark.parametrize(
    ("headers", "sent", "limit_hit"),
    [
        ({}, LIMIT // len(KIB) + 1, "66,560 bytes received"),
        ({"content-length": "5000000"}, 0, "Content-Length 5,000,000"),
    ],
    ids=["read", "declared"],
)
def test_an_error_answer_past_the_size_limit_names_the_limit_after_the_status(
    headers: dict[str, str], sent: int, limit_hit: str
) -> None:
    error = b'{"error": "' + b"x" * (2 * LIMIT) + b'"}'
    body = Trickle(error[start : start + len(KIB)] for start in range(0, len(error), len(KIB)))
    response = httpx2.Response(500, headers=headers, stream=body)
    assert message_for(response) == (
        f"HTTP 500 Internal Server Error; the answer is larger than 65,632 bytes ({limit_hit})"
    )
    assert body.sent == sent


def test_a_dripping_answer_ends_near_the_deadline() -> None:
    # The whole answer would take 2 s; the deadline is 0.2 s and a chunk 0.05 s.
    body = Trickle(itertools.chain([ANSWER], itertools.repeat(b" ", 40)), pause=0.05)
    started = time.monotonic()
    message = message_for(httpx2.Response(200, stream=body), timeout=0.2)
    assert message == "no complete answer within 0.2 s"
    assert time.monotonic() - started < 1.0


def test_the_wait_for_the_answer_counts_toward_the_deadline() -> None:
    body = Trickle([ANSWER])

    def answer_late(_request: httpx2.Request) -> httpx2.Response:
        time.sleep(0.3)
        return httpx2.Response(200, stream=body)

    with pytest.raises(EmbeddingError) as raised:
        embedder(httpx2.MockTransport(answer_late)).embed(["a"], timeout=0.2)
    assert str(raised.value) == "no complete answer within 0.2 s"
    assert body.sent == 0


def test_an_error_answer_past_the_deadline_names_the_timeout_after_the_status() -> None:
    body = Trickle([b'{"error": ', b'"slow"}'], pause=0.15)
    response = httpx2.Response(500, stream=body)
    assert message_for(response, timeout=0.2) == (
        "HTTP 500 Internal Server Error; no complete answer within 0.2 s"
    )


def test_an_error_answer_whose_body_fails_names_the_failure_after_the_status() -> None:
    def stalled() -> Iterator[bytes]:
        yield b'{"error": '
        raise httpx2.ReadTimeout("timed out")

    response = httpx2.Response(500, stream=Trickle(stalled()))
    assert message_for(response) == "HTTP 500 Internal Server Error; ReadTimeout: timed out"


def test_the_request_asks_for_an_uncompressed_answer() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx2.Response(200, content=ANSWER)

    assert embedder(httpx2.MockTransport(handler)).embed(["a"], timeout=1.0) == [[1.0, 0.0, 0.0]]


@pytest.mark.parametrize("encoding", [" Identity ", ""], ids=["identity", "empty"])
def test_an_answer_declared_uncompressed_is_read(encoding: str) -> None:
    response = httpx2.Response(200, headers={"content-encoding": encoding}, content=ANSWER)
    vectors = embedder(httpx2.MockTransport(lambda _request: response)).embed(["a"], timeout=1.0)
    assert vectors == [[1.0, 0.0, 0.0]]


@pytest.mark.parametrize(
    ("status", "encoding", "message"),
    [
        (200, "gzip", "the answer is compressed ('gzip'), which the embedder should not do"),
        (
            200,
            "\x1b[31m" + "x" * 100,
            "the answer is compressed ("
            + repr("\x1b[31m" + "x" * (64 - len("\x1b[31m")) + "…")
            + "), which the embedder should not do",
        ),
        (
            500,
            "gzip",
            "HTTP 500 Internal Server Error; "
            + "the answer is compressed ('gzip'), which the embedder should not do",
        ),
    ],
    ids=["gzip", "escaped-and-cut", "error-answer"],
)
def test_a_compressed_answer_is_refused_unread(status: int, encoding: str, message: str) -> None:
    body = Trickle([gzip.compress(ANSWER)])
    response = httpx2.Response(status, headers={"content-encoding": encoding}, stream=body)
    assert message_for(response) == message
    assert body.sent == 0


def test_a_compressed_answer_is_not_decoded_while_the_deadline_runs() -> None:
    # A gzip header announcing a comment, then the comment one byte per 0.05 s:
    # decoded, it yields nothing, so a reader that decodes would wait the whole
    # 2 s before any check could see it.
    header = b"\x1f\x8b\x08\x10" + b"\x00" * 6
    body = Trickle(itertools.chain([header], itertools.repeat(b"a", 40)), pause=0.05)
    response = httpx2.Response(200, headers={"content-encoding": "gzip"}, stream=body)
    started = time.monotonic()
    message = message_for(response, timeout=0.2)
    assert message == "the answer is compressed ('gzip'), which the embedder should not do"
    assert time.monotonic() - started < 1.0


# -- the check: Ollama's model list ---------------------------------------


def listing(*names: str) -> dict[str, object]:
    """A model list as Ollama answers it, each entry naming its model twice."""
    return {"models": [{"name": name, "model": name, "size": 1} for name in names]}


def checker(handler: httpx2.MockTransport, model: str = "bge-m3") -> OllamaEmbedder:
    return OllamaEmbedder("http://ollama.example.org:11434/", model, 3, transport=streamed(handler))


def check_message(answer: httpx2.Response | httpx2.MockTransport, timeout: float = 1.0) -> str:
    """The EmbeddingError of a check against ``answer``, a fixed answer or a transport."""
    transport = (
        answer
        if isinstance(answer, httpx2.MockTransport)
        else httpx2.MockTransport(lambda _request: answer)
    )
    with pytest.raises(EmbeddingError) as raised:
        checker(transport).check(timeout=timeout)
    return str(raised.value)


def test_the_check_reads_the_model_list_and_loads_nothing() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=listing("nomic-embed-text:latest", "bge-m3:latest"))

    checker(httpx2.MockTransport(handler)).check(timeout=1.0)
    assert [(r.method, r.url.path, r.content) for r in requests] == [("GET", "/api/tags", b"")]
    assert requests[0].headers["accept-encoding"] == "identity"


@pytest.mark.parametrize(
    ("model", "entry", "listed"),
    [
        ("bge-m3", {"name": "bge-m3:latest", "model": "bge-m3:latest"}, True),
        ("bge-m3", {"name": "bge-m3", "model": "bge-m3"}, True),
        ("bge-m3:latest", {"name": "bge-m3:latest", "model": "bge-m3:latest"}, True),
        ("bge-m3:567m", {"name": "bge-m3:567m", "model": "bge-m3:567m"}, True),
        ("bge-m3", {"model": "bge-m3:latest"}, True),
        ("bge-m3", {"name": "bge-m3:latest"}, True),
        (
            "registry.example.org:5000/team/bge-m3",
            {"name": "registry.example.org:5000/team/bge-m3:latest"},
            True,
        ),
        ("bge-m3:567m", {"name": "bge-m3:latest", "model": "bge-m3:latest"}, False),
        ("bge-m3:latest", {"name": "bge-m3", "model": "bge-m3"}, False),
        ("bge-m3", {"name": "bge-m3:567m", "model": "bge-m3:567m"}, False),
        ("bge-m3", {"name": "bge-m3-large:latest", "model": "bge-m3-large:latest"}, False),
        ("bge", {"name": "bge-m3:latest", "model": "bge-m3:latest"}, False),
        ("bge-m3", {"name": "team/bge-m3:latest", "model": "team/bge-m3:latest"}, False),
        ("bge-m3", {"name": ["bge-m3:latest"], "model": None}, False),
        ("library/bge-m3", {"name": "bge-m3:latest", "model": "bge-m3:latest"}, True),
        (
            "registry.ollama.ai/library/bge-m3",
            {"name": "bge-m3:latest", "model": "bge-m3:latest"},
            True,
        ),
        (
            "registry.ollama.ai/library/bge-m3:567m",
            {"name": "bge-m3:567m", "model": "bge-m3:567m"},
            True,
        ),
        (
            "registry.ollama.ai/team/bge-m3",
            {"name": "team/bge-m3:latest", "model": "team/bge-m3:latest"},
            True,
        ),
        ("team/bge-m3", {"name": "bge-m3:latest", "model": "bge-m3:latest"}, False),
        (
            "registry.example.org/library/bge-m3",
            {"name": "bge-m3:latest", "model": "bge-m3:latest"},
            False,
        ),
        (
            "registry.example.org/library/bge-m3",
            {"name": "registry.example.org/library/bge-m3:latest"},
            True,
        ),
        ("registry.ollama.ai/bge-m3", {"name": "bge-m3:latest", "model": "bge-m3:latest"}, False),
    ],
    ids=[
        "no-tag-listed-latest",
        "no-tag-listed-bare",
        "latest",
        "other-tag",
        "model-member-only",
        "name-member-only",
        "colon-in-host",
        "other-tag-not-latest",
        "tag-needs-exact",
        "no-tag-is-not-any-tag",
        "longer-name",
        "prefix",
        "other-namespace",
        "names-not-text",
        "default-namespace",
        "default-host-and-namespace",
        "default-host-and-namespace-tagged",
        "default-host-other-namespace",
        "other-namespace-configured",
        "other-host-keeps-namespace",
        "other-host-listed-whole",
        "two-parts-are-namespace-and-model",
    ],
)
def test_the_check_matches_the_model_name_as_ollama_lists_it(
    model: str, entry: dict[str, object], listed: bool
) -> None:
    response = httpx2.Response(200, json={"models": [entry]})
    client = checker(httpx2.MockTransport(lambda _request: response), model)
    if listed:
        client.check(timeout=1.0)
    else:
        with pytest.raises(EmbeddingError, match="Ollama does not list the model"):
            client.check(timeout=1.0)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx2.Response(200, json=listing()), "Ollama does not list the model 'bge-m3'"),
        (
            httpx2.Response(200, json=listing("nomic-embed-text:latest")),
            "Ollama does not list the model 'bge-m3'",
        ),
        (httpx2.Response(200, json={"models": None}), "models is null, not a list"),
        (httpx2.Response(200, json={"models": ["bge-m3:latest"]}), "Ollama does not list"),
        (httpx2.Response(200, json=["bge-m3:latest"]), "the answer is a list, not an object"),
        (
            httpx2.Response(200, json={"embeddings": []}),
            "the answer has no 'models' member (it has: 'embeddings'); "
            + "is [embeddings] url Ollama's API?",
        ),
        (httpx2.Response(200, text="not json"), "JSONDecodeError"),
        (
            httpx2.Response(500, json={"error": "out of memory"}),
            "HTTP 500 Internal Server Error; Ollama says 'out of memory'",
        ),
        (httpx2.Response(404), "HTTP 404 Not Found"),
    ],
    ids=[
        "empty",
        "not-listed",
        "models-null",
        "entries-not-objects",
        "not-an-object",
        "no-models-member",
        "not-json",
        "error-status",
        "no-such-route",
    ],
)
def test_a_bad_model_list_fails_the_check(response: httpx2.Response, message: str) -> None:
    assert message in check_message(response)


def test_an_unreachable_ollama_fails_the_check() -> None:
    message = check_message(raising(httpx2.ConnectError("connection refused")))
    assert message == "ConnectError: connection refused"


# The model list's size limit: 1 MiB.
TAGS_LIMIT = 1_048_576
TAGS_ANSWER = json.dumps(listing("bge-m3:latest")).encode()


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "read"])
def test_a_model_list_of_exactly_the_limit_is_read(declared: bool) -> None:
    body = TAGS_ANSWER + b" " * (TAGS_LIMIT - len(TAGS_ANSWER))
    response = (
        httpx2.Response(200, content=body)
        if declared
        else httpx2.Response(200, stream=Trickle([body[:1000], body[1000:]]))
    )
    checker(httpx2.MockTransport(lambda _request: response)).check(timeout=1.0)


@pytest.mark.parametrize(
    ("headers", "limit_hit"),
    [({}, "1,048,577 bytes received"), ({"content-length": "1048577"}, "Content-Length 1,048,577")],
    ids=["read", "declared"],
)
def test_a_model_list_past_the_limit_fails_the_check(
    headers: dict[str, str], limit_hit: str
) -> None:
    body = TAGS_ANSWER + b" " * (TAGS_LIMIT + 1 - len(TAGS_ANSWER))
    response = httpx2.Response(200, headers=headers, stream=Trickle([body]))
    assert check_message(response) == f"the answer is larger than 1,048,576 bytes ({limit_hit})"


def test_a_compressed_model_list_fails_the_check_unread() -> None:
    body = Trickle([gzip.compress(TAGS_ANSWER)])
    response = httpx2.Response(200, headers={"content-encoding": "gzip"}, stream=body)
    assert check_message(response) == (
        "the answer is compressed ('gzip'), which the embedder should not do"
    )
    assert body.sent == 0


def test_a_late_model_list_fails_the_check_unread() -> None:
    body = Trickle([TAGS_ANSWER])

    def answer_late(_request: httpx2.Request) -> httpx2.Response:
        time.sleep(0.3)
        return httpx2.Response(200, stream=body)

    message = check_message(httpx2.MockTransport(answer_late), timeout=0.2)
    assert message == "no complete answer within 0.2 s"
    assert body.sent == 0


def test_a_dripping_model_list_ends_near_the_deadline() -> None:
    body = Trickle(itertools.chain([TAGS_ANSWER], itertools.repeat(b" ", 40)), pause=0.05)
    started = time.monotonic()
    message = check_message(httpx2.Response(200, stream=body), timeout=0.2)
    assert message == "no complete answer within 0.2 s"
    assert time.monotonic() - started < 1.0
