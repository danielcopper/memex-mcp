"""Command line: ``memex-mcp [--config PATH]`` runs the server."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import cast

import uvicorn

from memex_mcp.config import ConfigError, load_config
from memex_mcp.embed import OllamaEmbedder
from memex_mcp.server import build_app
from memex_mcp.service import Memex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memex-mcp", description=__doc__)
    parser.add_argument("--config", help="the config file (default: $MEMEX_CONFIG)")
    args = parser.parse_args(argv)
    try:
        config = load_config(cast("str | None", args.config))
    except ConfigError as exc:
        sys.stderr.write(f"memex-mcp: {exc}\n")
        return 2
    logging.basicConfig(
        level=config.server.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    embeddings = config.embeddings
    embedder = (
        OllamaEmbedder(embeddings.url, embeddings.model, embeddings.dimensions)
        if embeddings.enabled
        else None
    )
    app = build_app(config, Memex.from_config(config, embedder))
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
