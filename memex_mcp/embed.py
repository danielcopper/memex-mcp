"""Text embeddings from Ollama, and the failure every caller must survive."""

from __future__ import annotations

import json
import math
import time
from http import HTTPStatus
from typing import Protocol, cast

import httpx2

from memex_mcp.json_kind import json_kind

# sqlite-vec stores float32: a larger value would become infinite there.
_FLOAT32_MAX = 3.4028234663852886e38
# sqlite-vec computes cosine distance in float32, where a vector of almost
# no length gives an infinite or NULL distance that sorts first (a length of
# 1e-30 gives -inf). A squared length below this bound, a length of 1e-6,
# is refused with a wide margin.
_MIN_SQUARED_LENGTH = 1e-12
# A squared length beyond float32 overflows there, and the distance to every
# vector becomes a flat 1.0 (cosine distance runs from 0 to 2), as if the
# vector were orthogonal to all of them.
_MAX_SQUARED_LENGTH = _FLOAT32_MAX
# An error message names at most this many of the answer's members, each cut
# to this many characters, and carries at most this much of Ollama's own error
# text or of a redirect's target.
_MEMBER_NAMES = 10
_NAME_CHARS = 64
_ERROR_CHARS = 200
# ... and at most this much of an exception's text, which can quote what came over the wire.
_TEXT_CHARS = 200
# The codes after which a Location header says where the answer moved (not 300, which
# offers a choice).
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
# An answer may hold this many bytes per value it should carry, plus this much
# for the rest. Ollama writes each float32 in its shortest form, at most nine
# digits: a value of a unit vector takes at most 17 characters and a comma
# (shaped like -0.00000123456789), about 12.4 on average, so 32 is about twice
# the worst case.
_BYTES_PER_VALUE = 32
_ANSWER_ALLOWANCE = 64 * 1024
# Ollama's model list may hold this many bytes. An entry takes about 350 bytes
# (the example in Ollama's API docs), so this holds about three thousand models,
# while an endpoint that is not Ollama cannot make the check read more.
_TAGS_LIMIT = 1024 * 1024
# The parts Ollama fills into a model's name when it gives none.
_DEFAULT_HOST = "registry.ollama.ai"
_DEFAULT_NAMESPACE = "library"


def _clip(text: str, chars: int) -> str:
    """``text`` cut to ``chars`` characters, escaped, for a message."""
    return repr(text if len(text) <= chars else text[:chars] + "…")


def _escaped(text: str, chars: int) -> str:
    """``text`` cut to ``chars`` characters, backslashes and all but printable ASCII escaped.

    Unlike ``repr`` it adds no quotes, so a message that needs no escaping reads as before.
    """
    escaped = text[:chars].encode("unicode_escape").decode("ascii")
    return escaped if len(text) <= chars else escaped + "…"


class EmbeddingError(Exception):
    """The embedder did not answer, answered too slowly, or answered nonsense or too much."""


def _failure(exc: Exception) -> EmbeddingError:
    """An exception's type and text, escaped and cut, as an EmbeddingError."""
    return EmbeddingError(f"{type(exc).__name__}: {_escaped(str(exc), _TEXT_CHARS) or 'no detail'}")


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str], timeout: float) -> list[list[float]]:
        """One vector of ``dimensions`` floats per text, in order."""
        ...

    def check(self, timeout: float) -> None:
        """EmbeddingError unless the embedder answers and offers ``model``; loads no model."""
        ...


def _vector(position: int, vector: object, dimensions: int) -> list[float]:
    """One vector of the answer as floats; EmbeddingError naming what is wrong with it."""
    if not isinstance(vector, list):
        raise EmbeddingError(f"vector {position} is {json_kind(vector)}, not a list")
    values = cast("list[object]", vector)  # a JSON array's items are any JSON value
    if len(values) != dimensions:
        raise EmbeddingError(f"vector {position} has {len(values)} values, expected {dimensions}")
    result: list[float] = []
    for index, value in enumerate(values):
        where = f"value {index} of vector {position}"
        # A JSON number arrives as int or float; bool is an int subclass but a JSON true/false.
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise EmbeddingError(f"{where} is {json_kind(value)}, not a number")
        try:
            number = float(value)
        except OverflowError:  # an integer beyond any float
            raise EmbeddingError(f"{where} is too large for a float") from None
        if not math.isfinite(number) or abs(number) > _FLOAT32_MAX:
            raise EmbeddingError(f"{where} is {number!r}, outside the finite float32 range")
        result.append(number)
    squared_length = math.fsum(v * v for v in result)
    if squared_length < _MIN_SQUARED_LENGTH:
        raise EmbeddingError(f"vector {position} has (almost) no length")
    if squared_length > _MAX_SQUARED_LENGTH:
        raise EmbeddingError(f"vector {position} is too long: its squared length overflows float32")
    return result


def _in_time(deadline: float, timeout: float) -> None:
    """EmbeddingError once ``deadline`` has passed."""
    if time.monotonic() > deadline:
        raise EmbeddingError(f"no complete answer within {timeout:g} s")


def _body(response: httpx2.Response, limit: int, deadline: float, timeout: float) -> bytes:
    """The answer's body; EmbeddingError if compressed, over ``limit`` bytes or past ``deadline``.

    The body is read as it comes over the wire, undecoded, so a compressed answer is refused
    before it is read. The deadline is checked once the headers have arrived and after each
    part of the answer, so a server that goes silent can stretch the call by up to one more
    timeout, and one that trickles its headers or the framing of its answer byte by byte
    for longer still.
    """
    encoding = response.headers.get("content-encoding", "")
    if encoding.strip().lower() not in {"", "identity"}:
        raise EmbeddingError(
            f"the answer is compressed ({_clip(encoding, _NAME_CHARS)}), "
            + "which the embedder should not do"
        )
    declared = response.headers.get("content-length", "")
    if declared.isdecimal() and int(declared) > limit:
        raise EmbeddingError(
            f"the answer is larger than {limit:,} bytes (Content-Length {int(declared):,})"
        )
    _in_time(deadline, timeout)
    chunks: list[bytes] = []
    received = 0
    for chunk in response.iter_raw():
        received += len(chunk)
        if received > limit:
            raise EmbeddingError(
                f"the answer is larger than {limit:,} bytes ({received:,} bytes received)"
            )
        _in_time(deadline, timeout)
        chunks.append(chunk)
    return b"".join(chunks)


def _ollama_says(body: bytes) -> str:
    """Ollama's own error text from an error answer's body, as a message suffix; empty if none."""
    try:
        payload = cast("object", json.loads(body))
    except (ValueError, RecursionError):
        return ""
    error = cast("dict[str, object]", payload).get("error") if isinstance(payload, dict) else None
    return f"; Ollama says {_clip(error, _ERROR_CHARS)}" if isinstance(error, str) else ""


def _error_text(response: httpx2.Response, limit: int, deadline: float, timeout: float) -> str:
    """What follows the status of an error answer: Ollama's text, or why its body was not read."""
    try:
        return _ollama_says(_body(response, limit, deadline, timeout))
    except EmbeddingError as exc:
        return f"; {exc}"
    except httpx2.HTTPError as exc:
        return f"; {_failure(exc)}"


def _status(response: httpx2.Response) -> str:
    """``HTTP <code> <phrase>`` for an error answer, plus where a redirect points.

    The phrase comes from the code, never from the server.
    """
    try:
        phrase = HTTPStatus(response.status_code).phrase
    except ValueError:  # a code HTTP does not define
        phrase = ""
    status = f"HTTP {response.status_code} {phrase}".rstrip()
    location = response.headers.get("location")
    if response.status_code in _REDIRECTS and location is not None:
        status += f"; redirected to {_clip(location, _ERROR_CHARS)}"
    return status


def _listed_as(model: str) -> frozenset[str]:
    """The names under which Ollama's model list can show ``model``.

    The list shows each model in its shortest form (Ollama's ``Name.DisplayShortest``):
    without the default host ``registry.ollama.ai`` and the default namespace ``library``,
    and always with its tag, ``latest`` if the name gives none. So ``bge-m3``,
    ``library/bge-m3`` and ``registry.ollama.ai/library/bge-m3`` are all listed as
    ``bge-m3:latest``; a name without a tag also matches its untagged form. As in Ollama's
    parser (``ParseNameBare``), a colon after the last slash starts the tag, the part after
    the last slash is the model, the part before it the namespace, and what is left the host.
    """
    name, tag = model, ""
    if name.rfind(":") > name.rfind("/"):
        name, _, tag = name.rpartition(":")
    rest, _, base = name.rpartition("/")
    host, _, namespace = rest.rpartition("/")
    if host not in {"", _DEFAULT_HOST}:
        shortest = f"{host}/{namespace}/{base}"
    elif namespace not in {"", _DEFAULT_NAMESPACE}:
        shortest = f"{namespace}/{base}"
    else:
        shortest = base
    if tag:
        return frozenset({f"{shortest}:{tag}"})
    return frozenset({shortest, f"{shortest}:latest"})


def _names_of(entry: object) -> set[str]:
    """The names an entry of Ollama's model list gives its model (``name`` and ``model``)."""
    if not isinstance(entry, dict):
        return set()
    members = cast("dict[str, object]", entry)
    return {name for name in (members.get("name"), members.get("model")) if isinstance(name, str)}


def _member_names(members: dict[str, object]) -> str:
    """The answer's member names for a message, escaped, cut and counted."""
    names = sorted(members)
    shown = ", ".join(_clip(name, _NAME_CHARS) for name in names[:_MEMBER_NAMES]) or "nothing"
    more = len(names) - _MEMBER_NAMES
    return f"{shown} and {more} more" if more > 0 else shown


class OllamaEmbedder:
    """Calls Ollama's ``POST /api/embed``, and ``GET /api/tags`` to check it."""

    def __init__(
        self,
        url: str,
        model: str,
        dimensions: int,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self.model: str = model
        self.dimensions: int = dimensions
        # The answer is read undecoded, so it must not be compressed.
        self._client: httpx2.Client = httpx2.Client(
            base_url=url.rstrip("/"),
            headers={"accept-encoding": "identity"},
            transport=transport,
        )

    def _answer(
        self, path: str, member: str, limit: int, timeout: float, request: object = None
    ) -> object:
        """The answer's ``member``, read within ``limit`` bytes and ``timeout`` seconds.

        A ``request`` is POSTed as JSON; without one the call is a GET.
        """
        method = "GET" if request is None else "POST"
        # The whole call must finish within the timeout; httpx2 applies it to each step,
        # so _body also checks it against this deadline.
        deadline = time.monotonic() + timeout
        try:
            with self._client.stream(method, path, json=request, timeout=timeout) as response:
                if not response.is_success:
                    raise EmbeddingError(
                        _status(response) + _error_text(response, limit, deadline, timeout)
                    )
                body = _body(response, limit, deadline, timeout)
        except (httpx2.HTTPError, ValueError, RecursionError) as exc:
            raise _failure(exc) from exc
        try:
            payload = cast("object", json.loads(body))
        except ValueError as exc:
            raise _failure(exc) from exc
        except RecursionError as exc:
            raise EmbeddingError("the answer is nested too deeply") from exc
        # A JSON object's keys are strings.
        if not isinstance(payload, dict):
            raise EmbeddingError(f"the answer is {json_kind(payload)}, not an object")
        members = cast("dict[str, object]", payload)
        if member not in members:
            raise EmbeddingError(
                f"the answer has no {member!r} member (it has: {_member_names(members)}); "
                + "is [embeddings] url Ollama's API?"
            )
        return members[member]

    def embed(self, texts: list[str], timeout: float) -> list[list[float]]:
        if not texts:
            return []
        limit = len(texts) * self.dimensions * _BYTES_PER_VALUE + _ANSWER_ALLOWANCE
        vectors = self._answer(
            "/api/embed", "embeddings", limit, timeout, {"model": self.model, "input": texts}
        )
        if not isinstance(vectors, list):
            raise EmbeddingError(f"embeddings is {json_kind(vectors)}, not a list")
        items = cast("list[object]", vectors)
        if len(items) != len(texts):
            raise EmbeddingError(
                f"the embedder returned {len(items)} vectors for {len(texts)} texts"
            )
        return [_vector(position, vector, self.dimensions) for position, vector in enumerate(items)]

    def check(self, timeout: float) -> None:
        """Asks ``GET /api/tags``, the list of the models Ollama holds, which loads none of them.

        Each entry names its model in ``name`` and in ``model`` (Ollama's ``ListModelResponse``).
        """
        models = self._answer("/api/tags", "models", _TAGS_LIMIT, timeout)
        if not isinstance(models, list):
            raise EmbeddingError(f"models is {json_kind(models)}, not a list")
        wanted = _listed_as(self.model)
        if not any(_names_of(entry) & wanted for entry in cast("list[object]", models)):
            raise EmbeddingError(f"Ollama does not list the model {_clip(self.model, _NAME_CHARS)}")

    def close(self) -> None:
        self._client.close()
