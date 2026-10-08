"""The MCP surface: three read-only tools over Streamable HTTP, behind Authentik tokens.

No ``from __future__ import annotations`` here: the tool signatures are read
by pydantic at registration, and one bound depends on the config.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from functools import partial
from typing import Annotated

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import RemoteAuthProvider
from fastmcp.server.dependencies import get_access_token
from pydantic import AnyHttpUrl, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from memex_mcp.auth import AuthentikTokenVerifier, identity_from
from memex_mcp.config import Config
from memex_mcp.rights import AccessDenied, Identity, NotFound
from memex_mcp.service import MAX_QUERY_CHARS, MAX_QUERY_WORDS, InvalidRequest, Memex

log = logging.getLogger(__name__)

PROTECTED_RESOURCE_PATH = "/.well-known/oauth-protected-resource"
HEALTH_PATH = "/healthz"

INSTRUCTIONS = """\
memex is a shared memory of Markdown notes, organised in areas: your own personal
area and, if you belong to the household, the shared area `household`. Use `search`
to find notes (it returns short snippets), `read` to get one note in full, `list`
to browse a folder, and `areas` to see which areas you can use. Notes under an
`archive/` folder are historical: prefer current notes and treat archived ones as
background."""

AREA_FILTER_HELP = "Search only this area, e.g. 'household'. Omit to search all your areas."
PATH_HELP = "The note's path as search or list return it, e.g. 'household/memory/wifi.md'."
FOLDER_HELP = "A folder inside the area, e.g. 'memory/archive'. Omit for the area's top level."

READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}


def build_auth(
    config: Config, verifier: AuthentikTokenVerifier | None = None
) -> RemoteAuthProvider:
    verifier = verifier or AuthentikTokenVerifier(
        issuer=config.auth.issuer,
        jwks_url=config.jwks_url,
        client_ids=config.auth.client_ids,
        algorithms=config.auth.algorithms,
        leeway_seconds=config.auth.leeway_seconds,
        jwks_min_refetch_seconds=config.auth.jwks_min_refetch_seconds,
        timeout_seconds=config.auth.timeout_seconds,
    )
    return RemoteAuthProvider(
        token_verifier=verifier,
        authorization_servers=[AnyHttpUrl(config.auth.issuer)],
        base_url=config.server.public_url,
        scopes_supported=list(config.auth.scopes),
        resource_name="memex",
    )


async def refresh_loop(memex: Memex, interval: float) -> None:
    """Keep the clone and the index fresh; the first round only fills in vectors."""
    first = True
    while True:
        try:
            if not first:
                await anyio.to_thread.run_sync(memex.refresh)
            await anyio.to_thread.run_sync(memex.backfill)
        except Exception:
            log.exception("refresh round failed; serving the last state")
        first = False
        await anyio.sleep(interval)


def _caller() -> Identity:
    try:
        return identity_from(get_access_token())
    except PermissionError as exc:
        raise ToolError(str(exc)) from None


async def _run[T](call: Callable[[], T]) -> T:
    """Run a blocking service call off the event loop, turning refusals into tool errors."""
    try:
        return await anyio.to_thread.run_sync(call)
    except (AccessDenied, NotFound, InvalidRequest) as exc:
        raise ToolError(str(exc)) from None


def build_mcp(
    config: Config, memex: Memex, auth: RemoteAuthProvider, run_loop: bool = True
) -> FastMCP:
    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncGenerator[None, None]:
        await anyio.to_thread.run_sync(memex.prepare)
        # A plain task rather than a task group: FastMCP may enter and leave
        # this lifespan from different tasks, which a cancel scope forbids.
        loop = (
            asyncio.create_task(refresh_loop(memex, config.repo.fetch_interval_seconds))
            if run_loop
            else None
        )
        try:
            yield
        finally:
            if loop is not None:
                loop.cancel()
                with suppress(asyncio.CancelledError):
                    await loop
            memex.index.close()

    mcp = FastMCP(
        "memex",
        instructions=INSTRUCTIONS,
        auth=auth,
        lifespan=lifespan,
        mask_error_details=True,
    )

    # The tools return Mapping[str, object] so that their advertised output
    # schema stays a free-form object; memex_mcp.service types each result.
    @mcp.tool(name="search", annotations=READ_ONLY)
    async def search(
        query: Annotated[
            str,
            Field(
                description=(
                    f"What to look for: keywords or a question (at most {MAX_QUERY_WORDS} words)."
                ),
                max_length=MAX_QUERY_CHARS,
            ),
        ],
        area: Annotated[
            str | None,
            Field(description=AREA_FILTER_HELP),
        ] = None,
        limit: Annotated[
            int, Field(description="Maximum number of hits.", ge=1, le=config.index.max_limit)
        ] = 10,
    ) -> Mapping[str, object]:
        """Search your memory notes by keywords and meaning.

        Returns one hit per note: path, area, title, a short snippet around the best
        match, a score (higher is better) and `archived` (true for historical notes,
        which rank lower). Call `read` with a hit's path for the full note. The
        top-level `semantic` flag says whether meaning-based search took part; when
        false, `notice` explains why and only keyword matches were found.
        """
        identity = _caller()
        return await _run(partial(memex.search, identity, query, area, limit))

    @mcp.tool(name="read", annotations=READ_ONLY)
    async def read(
        path: Annotated[
            str,
            Field(description=PATH_HELP),
        ],
    ) -> Mapping[str, object]:
        """Read one memory note in full (Markdown), with its title and whether it is archived."""
        identity = _caller()
        return await _run(partial(memex.read, identity, path))

    @mcp.tool(name="list", annotations=READ_ONLY)
    async def list_entries(
        area: Annotated[
            str, Field(description="The area to browse, e.g. your own area or 'household'.")
        ],
        folder: Annotated[
            str | None,
            Field(description=FOLDER_HELP),
        ] = None,
    ) -> Mapping[str, object]:
        """List the notes and folders in a folder of one of your areas."""
        identity = _caller()
        return await _run(partial(memex.list, identity, area, folder))

    @mcp.tool(name="areas", annotations=READ_ONLY)
    async def areas() -> Mapping[str, object]:
        """List the memory areas you can search, read and list: your own and, if you
        belong to the household, the shared one."""
        identity = _caller()
        return await _run(partial(memex.areas, identity))

    @mcp.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    return mcp


def build_app(
    config: Config,
    memex: Memex,
    verifier: AuthentikTokenVerifier | None = None,
    run_loop: bool = True,
) -> Starlette:
    auth = build_auth(config, verifier)
    mcp = build_mcp(config, memex, auth, run_loop=run_loop)
    app = mcp.http_app(path=config.server.mcp_path, stateless_http=True, json_response=True)
    # The SDK serves the metadata at the path-scoped location (RFC 9728 §3.1,
    # /.well-known/oauth-protected-resource/mcp); clients that look at the
    # root location get the same document there.
    scoped = PROTECTED_RESOURCE_PATH + config.server.mcp_path
    for route in list(app.router.routes):
        if isinstance(route, Route) and route.path == scoped:
            app.router.routes.append(
                Route(PROTECTED_RESOURCE_PATH, endpoint=route.endpoint, methods=["GET", "OPTIONS"])
            )
            break
    else:
        raise RuntimeError(f"no protected resource metadata route at {scoped}")
    return app
