"""Text embeddings from Ollama, and the failure every caller must survive."""

from __future__ import annotations

import math
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


class EmbeddingError(Exception):
    """The embedder did not answer, answered too slowly, or answered nonsense."""


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str], timeout: float) -> list[list[float]]:
        """One vector of ``dimensions`` floats per text, in order."""
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
    if math.fsum(v * v for v in result) < _MIN_SQUARED_LENGTH:
        raise EmbeddingError(f"vector {position} has (almost) no length")
    return result


class OllamaEmbedder:
    """Calls Ollama's ``POST /api/embed``."""

    def __init__(
        self,
        url: str,
        model: str,
        dimensions: int,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self.model: str = model
        self.dimensions: int = dimensions
        self._client: httpx2.Client = httpx2.Client(base_url=url.rstrip("/"), transport=transport)

    def embed(self, texts: list[str], timeout: float) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.post(
                "/api/embed",
                json={"model": self.model, "input": texts},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = cast("object", response.json())
        except (httpx2.HTTPError, ValueError) as exc:
            raise EmbeddingError(f"{type(exc).__name__}: {exc}") from exc
        # A JSON object's keys are strings.
        if not isinstance(payload, dict):
            raise EmbeddingError(f"the answer is {json_kind(payload)}, not an object")
        members = cast("dict[str, object]", payload)
        if "embeddings" not in members:
            names = ", ".join(sorted(members)) or "nothing"
            raise EmbeddingError(
                f"the answer has no 'embeddings' member (it has: {names}); "
                + "is [embeddings] url Ollama's API?"
            )
        vectors = members["embeddings"]
        if not isinstance(vectors, list):
            raise EmbeddingError(f"embeddings is {json_kind(vectors)}, not a list")
        items = cast("list[object]", vectors)
        if len(items) != len(texts):
            raise EmbeddingError(
                f"the embedder returned {len(items)} vectors for {len(texts)} texts"
            )
        return [_vector(position, vector, self.dimensions) for position, vector in enumerate(items)]

    def close(self) -> None:
        self._client.close()
