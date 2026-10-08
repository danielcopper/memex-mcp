# CLAUDE.md — how to work in memex-mcp

memex-mcp is a read-only MCP server over a git repository of Markdown notes, one area per person plus a shared one. The
README describes the architecture, the trust model and the setup.

## Gates

Every change passes all of these, run in the checkout you changed:

```bash
mise run setup      # editable install + pinned dev tools into .venv
mise run lint       # ruff check + ruff format --check
mise run typecheck  # basedpyright --warnings: zero errors and zero warnings
mise run complexity # cognitive complexity of every function at most 15
mise run test       # pytest, no network
deno fmt --check    # markdown formatting (CI-enforced)
```

CI runs the same on Python 3.12 (the floor) and 3.14, and builds the container image on pull requests.

## Ground rules

- **The rights core is `memex_mcp/rights.py`.** Every tool asks it for the caller's areas and for every path; nothing
  reads the clone without passing there. Search reads only the FTS5 tables of the caller's areas, filters vectors by
  area in SQL (`memex_mcp/index.py`), and checks the area again in the service.
- **What the caller cannot see answers like what does not exist**: same exception (`NotFound`), same fixed message,
  refused before the clone is touched. Error messages never name an area, a configured value or a server path. Messages
  that depend only on the caller's input (absolute path, `..`, hidden segment) may be specific. A caller without any
  access gets one fixed text (`ACCESS_DENIED`), whichever setting is missing.
- **A caller's ranking depends only on the areas they may read.** Each area has its own FTS5 table, because bm25's word
  statistics are per table; rankings from different areas are fused by rank, never compared by score.
  `test_ranking_does_not_depend_on_foreign_areas` holds this.
- **Notes and file names are untrusted input**, processed under the index lock: regular expressions over them must be
  linear, a file name that is not UTF-8 is skipped, and one note that fails never stops a batch.
- **A new guard gets a test that is seen failing**: break the guard on a copy of the tree, watch the test go red,
  restore. `tests/test_rights.py`, `tests/test_paths.py` and `tests/test_disclosure.py` are the rights tests.
- **Tests never touch the network or the real HOME.** `tests/conftest.py` points HOME and git's global config at a
  temporary directory for the whole session; origins are local bare repositories; JWKS and Ollama are mocked transports.
- Runtime dependencies are pinned exactly in `pyproject.toml` (this is an application), dev tools too.
- Version 1 is read-only. Writing (a commit per write, push, generated index files) comes later; keep the service layer
  the one place that would grow it.

## Pitfalls

- **Worktrees** need `mise trust && mise run setup` once. The harness's LSP diagnostics resolve against the main
  checkout; trust `mise run typecheck` run in the worktree.
- **No `from __future__ import annotations` in `memex_mcp/server.py`**: pydantic reads the tool signatures at
  registration, and the `limit` bound comes from the config in a local scope.
- **FastMCP may enter and leave the server lifespan in different tasks**, so the lifespan must not hold an anyio task
  group or cancel scope across its `yield`; the refresh loop is a plain asyncio task.
- **Close every SQLite connection.** Python 3.14 (the newer CI leg) warns about unclosed ones, and the suite turns
  warnings into errors; `with sqlite3.connect(...)` commits but does not close (use `contextlib.closing`).
- **Never create a virtual table inside a note's savepoint.** Rolling back to a savepoint that created and wrote an FTS5
  table leaves SQLite unable to open the next savepoint; `rebuild` creates every configured area's table first.
- **sqlite-vec ships no types**: `typings/sqlite_vec/__init__.pyi` copies its signatures. A bump of its pin checks the
  stub against the new `__init__.py`.
- `Index` ids are `AUTOINCREMENT` so a vector computed for a chunk that was replaced meanwhile can never attach to a new
  chunk.
