"""Shared fixtures: an isolated HOME, a memex clone with an origin, identities, fake embedders."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, override

import httpx2
import pytest

from memex_mcp.config import Config, build_config
from memex_mcp.embed import Embedder, EmbeddingError, OllamaEmbedder
from memex_mcp.rights import Identity
from memex_mcp.service import Memex

# Every area carries a note `memory/zebra.md` with this word, so one search
# shows exactly which areas a caller reaches.
MARKER = "zebracode"


@pytest.fixture(scope="session", autouse=True)
def isolated_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """No test reads or writes the real HOME, and git sees no user or system config.

    Session-scoped so that it is in force before any other fixture runs.
    """
    home = tmp_path_factory.mktemp("home")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(home))
        patch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
        patch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
        patch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        patch.setenv("GIT_AUTHOR_NAME", "Test Author")
        patch.setenv("GIT_AUTHOR_EMAIL", "author@example.org")
        patch.setenv("GIT_COMMITTER_NAME", "Test Author")
        patch.setenv("GIT_COMMITTER_EMAIL", "author@example.org")
        for name in list(os.environ):
            if name.startswith("MEMEX_"):
                patch.delenv(name)
        yield home


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )
    return done.stdout


NOTES: dict[str, str] = {
    "alice/memory/MEMORY.md": "# Memory index\n\n- bike repair notes\n",
    "alice/memory/zebra.md": f"# Alice zebra\n\n{MARKER} alice private note\n",
    "alice/memory/bike-repair.md": (
        "---\nname: bike-repair\n---\n# Bike repair\n\n"
        "The chain of the bike is cleaned every week with degreaser.\n\n"
        "## Spare parts\n\nKeep a spare inner tube next to the pump.\n"
    ),
    "alice/memory/archive/old-bike.md": (
        "# Old bike\n\nBefore the degreaser, diesel cleaned the chain of the bike by hand.\n"
    ),
    "alice/wiki/networking.md": "# Networking\n\nThe router hands out addresses by DHCP.\n",
    "alice/.obsidian/workspace.md": f"# hidden\n\n{MARKER} obsidian state\n",
    "bob/memory/zebra.md": f"# Bob zebra\n\n{MARKER} bob private note\n",
    "bob/memory/garden.md": "# Garden\n\nThe tomatoes need water every evening.\n",
    "household/memory/zebra.md": f"# Household zebra\n\n{MARKER} household shared note\n",
    "household/memory/wifi.md": "# Wifi\n\nThe guest wifi password is on the fridge.\n",
    "carol/memory/zebra.md": f"# Carol zebra\n\n{MARKER} carol private note\n",
    "README.md": f"# memex\n\n{MARKER} repository readme outside every area\n",
}


def write_notes(root: Path, notes: dict[str, str]) -> None:
    for relative, text in notes.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@dataclass
class Origin:
    """A bare origin plus a working copy that pushes to it (the writers' side)."""

    bare: Path
    work: Path

    def commit(
        self, message: str, notes: dict[str, str] | None = None, remove: tuple[str, ...] = ()
    ) -> str:
        write_notes(self.work, notes or {})
        for relative in remove:
            git(self.work, "rm", "-q", relative)
        git(self.work, "add", "-A")
        git(self.work, "commit", "-q", "-m", message)
        git(self.work, "push", "-q", "origin", "main")
        return git(self.work, "rev-parse", "HEAD").strip()


@pytest.fixture
def origin(tmp_path: Path) -> Origin:
    bare = tmp_path / "origin.git"
    work = tmp_path / "writer"
    git(tmp_path, "init", "-q", "--bare", "--initial-branch=main", str(bare))
    git(tmp_path, "clone", "-q", str(bare), str(work))
    git(work, "checkout", "-q", "-B", "main")
    result = Origin(bare=bare, work=work)
    result.commit("initial notes", NOTES)
    return result


def raw_config(tmp_path: Path, origin: Origin | None = None) -> dict[str, dict[str, object]]:
    return {
        "server": {"public_url": "https://memex.example.org"},
        "auth": {
            "issuer": "https://auth.example.org/application/o/memex/",
            "client_ids": ["claude-code"],
        },
        "repo": {
            "path": str(tmp_path / "clone"),
            "remote": str(origin.bare) if origin else "",
        },
        "index": {"path": str(tmp_path / "index" / "memex.sqlite3")},
        "embeddings": {"dimensions": FakeEmbedder.dimensions},
        "users": {
            "alice": "alice",
            "bob": "bob",
            "carol": "carol",
            "mallory": "mallory",
        },
    }


def make_config(
    tmp_path: Path, origin: Origin | None = None, **sections: dict[str, object]
) -> Config:
    raw = raw_config(tmp_path, origin)
    for name, values in sections.items():
        raw.setdefault(name, {}).update(values)
    return build_config(raw, {})


# Concepts the fake embedder maps several words onto, so that a query can be
# close in meaning to a note that shares no word with it.
CONCEPTS: dict[str, int] = {
    "wifi": 0, "wlan": 0, "internet": 0, "network": 0, "password": 0,
    "garden": 1, "tomatoes": 1, "plants": 1, "watering": 1,
    "bike": 2, "chain": 2, "degreaser": 2, "tube": 2, "diesel": 2,
}  # fmt: skip


class FakeEmbedder(Embedder):
    """Deterministic embeddings: one dimension per concept, the rest hashed."""

    model: str = "fake-embedder"
    dimensions: int = 16

    def __init__(self) -> None:
        self.calls: int = 0

    @override
    def embed(self, texts: list[str], timeout: float) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            vector[-1] = 0.01  # never the zero vector
            for word in cast("list[str]", re.findall(r"\w+", text.lower())):
                if word in CONCEPTS:
                    vector[CONCEPTS[word]] += 1.0
                else:
                    vector[4 + sum(map(ord, word)) % (self.dimensions - 5)] += 0.1
            vectors.append(vector)
        return vectors


@dataclass
class FailingEmbedder(Embedder):
    """An embedder whose host is down."""

    model: str = "fake-embedder"
    dimensions: int = 16
    calls: int = 0
    message: str = "ConnectError: connection refused"
    texts: list[str] = field(default_factory=list[str])

    @override
    def embed(self, texts: list[str], timeout: float) -> list[list[float]]:
        self.calls += 1
        self.texts.extend(texts)
        raise EmbeddingError(self.message)


def malformed_ollama() -> OllamaEmbedder:
    """Ollama on a mocked transport that answers every text with a vector of nulls."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        texts = cast("dict[str, list[str]]", json.loads(request.content))["input"]
        vectors = [[None] * FakeEmbedder.dimensions for _ in texts]
        return httpx2.Response(200, json={"embeddings": vectors})

    return OllamaEmbedder(
        "http://ollama.example.org:11434",
        FakeEmbedder.model,
        FakeEmbedder.dimensions,
        transport=httpx2.MockTransport(handler),
    )


@pytest.fixture
def memex(tmp_path: Path, origin: Origin) -> Iterator[Memex]:
    """A prepared service over a fresh clone of ``origin``, keyword search only."""
    service = Memex.from_config(make_config(tmp_path, origin), embedder=None)
    service.prepare()
    yield service
    service.index.close()


# Identities as Authentik would describe them.
ALICE = Identity("alice", frozenset({"memex", "household"}))
BOB = Identity("bob", frozenset({"memex", "household"}))
CAROL = Identity("carol", frozenset({"memex"}))
# Mapped to an area, but not in the memex group.
MALLORY = Identity("mallory", frozenset({"household"}))
# In the memex group, but no area is configured for them.
STRANGER = Identity("stranger", frozenset({"memex"}))
# Unmapped, but in memex and household: reaches the shared area only.
GUEST = Identity("guest", frozenset({"memex", "household"}))
