"""Text embeddings from Ollama, and the failure every caller must survive."""

from __future__ import annotations

from typing import Protocol, cast

import httpx2


class EmbeddingError(Exception):
    """The embedder did not answer, answered too slowly, or answered nonsense."""


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str], timeout: float) -> list[list[float]]:
        """One vector of ``dimensions`` floats per text, in order."""
        ...


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
        # A JSON object's keys are strings, an array's items any JSON value.
        vectors = (
            cast("dict[str, object]", payload).get("embeddings")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(vectors, list) or len(cast("list[object]", vectors)) != len(texts):
            raise EmbeddingError("the embedder returned no embedding per input")
        result: list[list[float]] = []
        for vector in cast("list[object]", vectors):
            if not isinstance(vector, list) or len(cast("list[object]", vector)) != self.dimensions:
                raise EmbeddingError(
                    f"the embedder returned a vector without {self.dimensions} dimensions"
                )
            # Typed as the numbers it should hold; each value goes through float() as before.
            result.append([float(value) for value in cast("list[float]", vector)])
        return result

    def close(self) -> None:
        self._client.close()
