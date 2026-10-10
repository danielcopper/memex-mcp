"""The MCP surface: tools to search, read and write notes over Streamable HTTP, behind
Authentik tokens.

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
from memex_mcp.rights import AccessDenied, Identity, InvalidRequest, NotFound
from memex_mcp.service import MAX_QUERY_CHARS, MAX_QUERY_WORDS, IndexLine, Memex

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

WRITE_INSTRUCTIONS = """

You can write where you can read, below `<area>/memory/` and `<area>/wiki/`:
`create` a note (call `index_sections` first to pick the section of its index),
`append` to one, `edit` a passage of one, or `delete` one. Every write is saved as
its own commit. Never write keys or access tokens into a note; a password the user
asks you to note down is fine."""

AREA_FILTER_HELP = "Search only this area, e.g. 'household'. Omit to search all your areas."
PATH_HELP = "The note's path as search or list return it, e.g. 'household/memory/wifi.md'."
FOLDER_HELP = "A folder inside the area, e.g. 'memory/archive'. Omit for the area's top level."

READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
# create and append add text and keep what is there; edit and delete change or remove it.
ADDS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}
CHANGES = {**ADDS, "destructiveHint": True}

WRITE_PATH_HELP = (
    "The note's path, below <area>/memory/ or <area>/wiki/, e.g. 'household/memory/wifi.md'."
)
VERSION_HELP = "The note's version as read returned it."


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
    """Keep the clone, the index and the health state fresh; the first round does not fetch."""
    first = True
    while True:
        try:
            await anyio.to_thread.run_sync(partial(memex.refresh_round, first))
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
        instructions=INSTRUCTIONS + (WRITE_INSTRUCTIONS if config.write.enabled else ""),
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
        """Read one memory note in full (Markdown), with its title, whether it is archived,
        its `version` (which edit and delete take) and when it was created and last updated."""
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
        """List the memory areas you can use: your own and, if you belong to the
        household, the shared one."""
        identity = _caller()
        return await _run(partial(memex.areas, identity))

    if config.write.enabled:
        _add_write_tools(mcp, config, memex)

    @mcp.custom_route(HEALTH_PATH, methods=["GET"], include_in_schema=False)
    async def health(_request: Request) -> Response:
        # Always 200: the container healthcheck asks only whether the server answers.
        return JSONResponse(memex.health())

    return mcp


def _add_write_tools(mcp: FastMCP, config: Config, memex: Memex) -> None:
    """The tools that change notes, and index_sections, which create relies on."""
    longest = config.write.max_note_bytes

    @mcp.tool(name="create", annotations=ADDS)
    async def create(
        path: Annotated[str, Field(description=WRITE_PATH_HELP)],
        content: Annotated[str, Field(description="The note in Markdown.", max_length=longest)],
        index_section: Annotated[
            str | None,
            Field(
                description=(
                    "The section of the note's index to list it under, as index_sections "
                    "returns them; a new name adds a section."
                )
            ),
        ] = None,
        index_line: Annotated[
            str | None,
            Field(
                description=(
                    "What the note holds, for its index: one line of at most 300 characters."
                )
            ),
        ] = None,
        topic_description: Annotated[
            str | None,
            Field(
                description=(
                    "Only for the first note of a new wiki topic: what the topic holds, "
                    "for the master index; one line of at most 300 characters."
                )
            ),
        ] = None,
    ) -> Mapping[str, object]:
        """Create a new note and add its line to its topic's index, in one commit.

        Refuses a path where a note exists. Notes in a `daily/` folder take no
        index line; every other note needs index_section and index_line, and the
        first note of a new wiki topic also topic_description. Returns the commit,
        whether it was pushed, and the note's version.
        """
        identity = _caller()
        listing = IndexLine(index_section, index_line, topic_description)
        return await _run(partial(memex.create, identity, path, content, listing))

    @mcp.tool(name="append", annotations=ADDS)
    async def append(
        path: Annotated[str, Field(description=WRITE_PATH_HELP)],
        text: Annotated[str, Field(description="The text to add.", max_length=longest)],
    ) -> Mapping[str, object]:
        """Add text at the end of an existing note, on a line of its own."""
        identity = _caller()
        return await _run(partial(memex.append, identity, path, text))

    @mcp.tool(name="edit", annotations=CHANGES)
    async def edit(
        path: Annotated[str, Field(description=WRITE_PATH_HELP)],
        old_text: Annotated[
            str,
            Field(
                description="The passage to replace, exactly as in the note; it must occur once.",
                max_length=longest,
            ),
        ],
        new_text: Annotated[str, Field(description="What replaces it.", max_length=longest)],
        version: Annotated[
            str | None,
            Field(description=VERSION_HELP + " If given, the edit is refused if the note changed."),
        ] = None,
    ) -> Mapping[str, object]:
        """Replace one passage of a note: old_text must occur in it exactly once."""
        identity = _caller()
        return await _run(partial(memex.edit, identity, path, old_text, new_text, version))

    @mcp.tool(name="delete", annotations=CHANGES)
    async def delete(
        path: Annotated[str, Field(description=WRITE_PATH_HELP)],
        version: Annotated[str, Field(description=VERSION_HELP)],
    ) -> Mapping[str, object]:
        """Delete a note, if it is still the version you read. Its index line stays."""
        identity = _caller()
        return await _run(partial(memex.delete, identity, path, version))

    @mcp.tool(name="index_sections", annotations=READ_ONLY)
    async def index_sections(
        path: Annotated[
            str,
            Field(
                description=(
                    "The new note's path, or the folder it goes into, "
                    "e.g. 'household/memory' or 'alice/wiki/networking'."
                )
            ),
        ],
    ) -> Mapping[str, object]:
        """Show the index a new note there would be listed in: its path, whether it exists
        yet, and its sections. A note in a `daily/` folder has no index."""
        identity = _caller()
        return await _run(partial(memex.index_sections, identity, path))


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
