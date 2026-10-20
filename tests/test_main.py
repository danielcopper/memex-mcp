"""The command line: from the config file to the parts of the server."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import uvicorn

import memex_mcp.__main__ as cli
from memex_mcp.config import Config
from memex_mcp.embed import Embedder, OllamaEmbedder
from memex_mcp.service import Memex

EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.toml"


def test_the_embedder_gets_the_configured_keep_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    assert text.count('# keep_alive = "30m"\n') == 1
    path = tmp_path / "config.toml"
    path.write_text(text.replace('# keep_alive = "30m"\n', 'keep_alive = "30m"\n'), "utf-8")
    embedders: list[Embedder | None] = []

    def from_config(_config: Config, embedder: Embedder | None) -> None:
        embedders.append(embedder)

    def build_app(_config: Config, _memex: Memex) -> None:
        return None

    def ignore(*_args: object, **_kwargs: object) -> None:
        return None

    for name in [name for name in os.environ if name.startswith("MEMEX_")]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(Memex, "from_config", from_config)
    monkeypatch.setattr(cli, "build_app", build_app)
    monkeypatch.setattr(uvicorn, "run", ignore)
    monkeypatch.setattr(logging, "basicConfig", ignore)
    assert cli.main(["--config", str(path)]) == 0
    [embedder] = embedders
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.keep_alive == "30m"
    embedder.close()
