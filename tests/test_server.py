"""The HTTP surface end to end: discovery, 401s, and tools called with real signed tokens."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict, cast

import anyio
import httpx2
import pytest

from memex_mcp.rights import ACCESS_DENIED
from memex_mcp.server import build_app
from memex_mcp.service import Hit, Memex
from tests.conftest import MARKER, Origin, make_config
from tests.test_auth import ISSUER, KEY, FakeJwks, make_verifier, token

BASE = "https://memex.example.org"
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


class ListedTool(TypedDict):
    """The parts of a tools/list entry these tests read."""

    name: str
    annotations: dict[str, object]


class TextContent(TypedDict):
    text: str


class ToolCall(TypedDict):
    """The parts of a tools/call result these tests read."""

    isError: bool
    content: list[TextContent]
    structuredContent: dict[str, object]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def serve(
    tmp_path: Path, origin: Origin, run_loop: bool = False
) -> AsyncGenerator[httpx2.AsyncClient, None]:
    """The app with its lifespan, entered and left in the test's own task."""
    config = make_config(tmp_path, origin, repo={"fetch_interval_seconds": 0.05})
    memex = Memex.from_config(config, embedder=None)
    with FakeJwks((KEY, "k1")) as jwks:
        verifier = make_verifier(jwks, algorithms=("RS256",))
        app = build_app(config, memex, verifier=verifier, run_loop=run_loop)
        async with app.router.lifespan_context(app):
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(transport=transport, base_url=BASE) as http:
                yield http


async def rpc(
    client: httpx2.AsyncClient,
    method: str,
    params: dict[str, object] | None = None,
    bearer: str | None = None,
) -> httpx2.Response:
    headers = dict(HEADERS)
    if bearer is not None:
        headers["authorization"] = f"Bearer {bearer}"
    body: dict[str, object] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    return await client.post("/mcp", headers=headers, json=body)


async def call(client: httpx2.AsyncClient, bearer: str, tool: str, **arguments: object) -> ToolCall:
    response = await rpc(client, "tools/call", {"name": tool, "arguments": arguments}, bearer)
    assert response.status_code == 200, response.text
    return cast("ToolCall", response.json()["result"])


@pytest.mark.anyio
@pytest.mark.parametrize(
    "path", ["/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"]
)
async def test_protected_resource_metadata_names_authentik(
    tmp_path: Path, origin: Origin, path: str
) -> None:
    async with serve(tmp_path, origin) as client:
        response = await client.get(path)
        assert response.status_code == 200
        metadata = cast("dict[str, object]", response.json())
        assert metadata["resource"] == f"{BASE}/mcp"
        assert metadata["authorization_servers"] == [ISSUER]
        assert metadata["scopes_supported"] == ["openid", "profile", "offline_access"]


@pytest.mark.anyio
async def test_missing_token_is_401_with_discovery_hint(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin) as client:
        response = await rpc(client, "tools/list")
        assert response.status_code == 401
        assert "resource_metadata=" in response.headers["www-authenticate"]


@pytest.mark.anyio
async def test_invalid_token_is_401(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin) as client:
        for bad in ("garbage", token(iss="https://evil.example.org/"), token(groups=None)):
            assert (await rpc(client, "tools/list", bearer=bad)).status_code == 401


@pytest.mark.anyio
async def test_health_needs_no_token(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin) as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_tools_are_listed_read_only(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin) as client:
        response = await rpc(client, "tools/list", bearer=token())
        listed = cast("list[ListedTool]", response.json()["result"]["tools"])
        tools = {tool["name"]: tool for tool in listed}
        assert set(tools) == {"search", "read", "list", "areas"}
        assert all(tool["annotations"]["readOnlyHint"] is True for tool in tools.values())


@pytest.mark.anyio
async def test_own_note_reads_and_foreign_note_does_not(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin) as client:
        alice = token()
        own = await call(client, alice, "read", path="alice/memory/zebra.md")
        assert own["isError"] is False
        assert MARKER in cast("str", own["structuredContent"]["content"])
        foreign = await call(client, alice, "read", path="bob/memory/zebra.md")
        assert foreign["isError"] is True
        assert foreign["content"][0]["text"] == "no such note"
        assert MARKER not in json.dumps(foreign)


@pytest.mark.anyio
async def test_search_over_http_stays_inside_the_callers_areas(
    tmp_path: Path, origin: Origin
) -> None:
    async with serve(tmp_path, origin) as client:
        bob = token(preferred_username="bob")
        result = await call(client, bob, "search", query=MARKER, limit=50)
        hits = cast("list[Hit]", result["structuredContent"]["hits"])
        assert {hit["area"] for hit in hits} == {"bob", "household"}
        assert result["structuredContent"]["semantic"] is False


@pytest.mark.anyio
async def test_member_without_memex_group_gets_a_tool_error(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin) as client:
        outsider = token(groups=["household"])
        for tool, arguments in (
            ("search", {"query": MARKER}),
            ("read", {"path": "alice/memory/zebra.md"}),
            ("list", {"area": "alice"}),
        ):
            result = await call(client, outsider, tool, **arguments)
            assert result["isError"] is True
            assert result["content"][0]["text"] == ACCESS_DENIED


@pytest.mark.anyio
async def test_background_loop_picks_up_new_commits(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin, run_loop=True) as client:
        origin.commit("later", {"household/memory/later.md": "# Later\n\nnumbat sighting\n"})
        deadline = time.monotonic() + 10
        hits: list[Hit] = []
        while time.monotonic() < deadline and not hits:
            result = await call(client, token(), "search", query="numbat")
            hits = cast("list[Hit]", result["structuredContent"]["hits"])
            await anyio.sleep(0.05)
        assert [hit["path"] for hit in hits] == ["household/memory/later.md"]


@pytest.mark.anyio
async def test_foreign_and_missing_notes_answer_alike_over_http(
    tmp_path: Path, origin: Origin
) -> None:
    async with serve(tmp_path, origin) as client:
        alice = token()
        existing = await call(client, alice, "read", path="bob/memory/zebra.md")
        missing = await call(client, alice, "read", path="bob/memory/no-such-note.md")
        own_missing = await call(client, alice, "read", path="alice/memory/no-such-note.md")
        assert existing == missing == own_missing
        assert existing["isError"] is True


@pytest.mark.anyio
async def test_overlong_query_is_a_tool_error(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin) as client:
        result = await call(client, token(), "search", query="a" * 501)
        assert result["isError"] is True


@pytest.mark.anyio
async def test_areas_tool_answers_the_callers_areas_only(tmp_path: Path, origin: Origin) -> None:
    async with serve(tmp_path, origin) as client:
        bob = await call(client, token(preferred_username="bob"), "areas")
        assert bob["isError"] is False
        assert bob["structuredContent"] == {"areas": ["bob", "household"]}
        outsider = await call(client, token(groups=["household"]), "areas")
        assert outsider["isError"] is True
        assert outsider["content"][0]["text"] == ACCESS_DENIED
