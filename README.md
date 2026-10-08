# memex-mcp

MCP server for a shared, per-person Markdown memory in git: search and read with rights from your identity provider.

The memory ("memex") is a git repository of Markdown notes with one top-level directory per area: a private area per
person (`alice/`, `bob/`) and a shared one (`household/`). Assistants such as Claude Code reach it through this server,
which searches the notes by keyword and by meaning, returns short snippets, reads single notes, and lets each caller see
only their own areas. Version 1 is read-only; people (or a local clone with its own sync) write to the repository
directly.

## Architecture

**Clone and index.** The server owns a clone of the memex repository. At start it clones it if needed and builds a
SQLite search index of the configured areas (the users' areas and the household area; other top-level folders are not
read); the index is a cache on a volume, never committed, and rebuilt from scratch when it is missing, its schema
version differs, or the set of configured areas changed. Every 60 seconds a background loop runs `git fetch` and
`git merge --ff-only` and re-indexes exactly the paths that changed between the old and the new HEAD. A failed fetch is
logged and the last state keeps serving.

**Search.** Notes are split into chunks by heading and then by paragraph. Each chunk is indexed twice: in its area's own
FTS5 table (bm25 ranking) and, once the embedder has answered, as a vector (Ollama, model `bge-m3`) compared with
sqlite-vec's cosine distance. One FTS5 table per area keeps bm25's word statistics to the areas the caller may read, so
nobody's ranking shifts with what other people's notes contain. The keyword ranking of each of the caller's areas and
the vector ranking over those areas are fused by rank with reciprocal rank fusion; a note scores by its best chunk. A
query holds at most 500 characters and 32 words. Notes under any `archive/` directory are multiplied by a configurable
factor (0.5 by default) and flagged `archived`. If the embedder does not answer within the query timeout, the search
runs on keywords alone, says so in the result (`semantic: false` plus a `notice`), and skips the embedder for a minute
instead of waiting on every query. Vectors for new or changed chunks are filled in by the background loop, so an
embedder that is off does not hold up the keyword index.

**Authentication.** The server is an OAuth 2.1 resource server; Authentik is the authorization server. Clients find
Authentik through the protected resource metadata (RFC 9728) at `/.well-known/oauth-protected-resource` and obtain
tokens there; the server only validates them. Authentik issues access tokens as JWTs signed with the provider's signing
key, so each token is checked locally against the provider's JWKS: signature, `iss`, `exp`, `aud` and `azp` (both must
be one of the configured client ids). The caller's username comes from `preferred_username`, their groups from `groups`.
A token must name its signing key (`kid`). The key set is cached for five minutes, so a key removed from the provider
stops verifying within five minutes. A token naming a key the cached set lacks fetches the set again, but not sooner
than `auth.jwks_min_refetch_seconds` (60 by default) after the last successful fetch. When the set cannot be fetched
once the five minutes are over, every token is rejected until a fetch succeeds.

**Rights.** Access requires membership in the group `memex`. A user reads the area their username maps to in the config
and, as a member of the group `household`, the shared area too. Nothing else; there is no admin override. Every tool
enforces this on the server, and paths are checked before and after following symlinks: only `.md` files, inside an
allowed area, no `..`, no absolute paths, no hidden segments (`.git`, `.obsidian`). A note, folder or area outside the
caller's rights gets exactly the answer a missing one gets, so the server does not reveal what other people have.

**Tools.**

| Tool                                 | Returns                                                                                                                                                                                   |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `search(query, area=None, limit=10)` | `semantic`, optional `notice`, and `hits`: one per note with `path`, `area`, `title`, `snippet` (up to 300 characters, `index.snippet_chars`, around the best match), `score`, `archived` |
| `read(path)`                         | the full note: `path`, `area`, `title`, `archived`, `content`                                                                                                                             |
| `list(area, folder=None)`            | the notes and folders of one folder: `name`, `path`, `type` (`note` or `folder`)                                                                                                          |
| `areas()`                            | the caller's own areas: `areas` (the own area, plus `household` for members of the household group)                                                                                       |

## Trust model

- memex-mcp enforces the per-person rights only for access **through the server**, that is, for assistants.
- git has no read permissions per directory: anyone with the repository (the operator, every full clone, the git host's
  admin) can read every area.
- Whoever runs the server can read everything, since the server must read the notes to index them. Privacy from the
  operator would need client-side encryption, which rules out server-side search.
- Response times are not separated: while the server indexes someone's new notes, searches wait for it, so a caller can
  notice that other areas are being written, though not what they contain.
- Operators who need people separated at the git level would use one repository per person plus one shared repository.
  Version 1 does not support that; it is planned for a later version.

## Configuration

One TOML file, named by `--config` or `MEMEX_CONFIG`; [`config.example.toml`](config.example.toml) lists every setting.
Any scalar or list setting can be overridden by an environment variable `MEMEX_<SECTION>_<KEY>` (lists comma-separated),
for example `MEMEX_REPO_REMOTE` for a remote URL that carries a credential. The `[users]` table lives in the file only.

| Setting                                                    | Default                                    | Meaning                                                                   |
| ---------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| `server.public_url`                                        | required                                   | URL clients use; the advertised resource is `public_url` + `mcp_path`     |
| `server.host`, `server.port`, `server.mcp_path`            | `0.0.0.0`, `8000`, `/mcp`                  | where the server listens                                                  |
| `auth.issuer`                                              | required                                   | Authentik issuer, e.g. `https://auth.example.org/application/o/memex/`    |
| `auth.client_ids`                                          | required                                   | client ids whose tokens are accepted (`aud` and `azp`)                    |
| `auth.jwks_url`                                            | `<issuer>jwks/`                            | the provider's JWKS endpoint                                              |
| `auth.algorithms`                                          | `RS256`, `ES256`                           | accepted signing algorithms                                               |
| `auth.scopes`                                              | `openid`, `profile`, `offline_access`      | scopes advertised to clients                                              |
| `rights.access_group`                                      | `memex`                                    | group required for any access                                             |
| `rights.household_group`, `rights.household_area`          | `household`, `household`                   | group and directory of the shared area                                    |
| `repo.path`                                                | required                                   | the server's clone                                                        |
| `repo.remote`                                              | none                                       | cloned from here when `repo.path` holds no clone                          |
| `repo.branch`, `repo.fetch_interval_seconds`               | `main`, `60`                               | what to follow and how often                                              |
| `index.path`                                               | required                                   | the SQLite index (a cache)                                                |
| `index.archive_factor`                                     | `0.5`                                      | score factor for notes under `archive/`                                   |
| `index.snippet_chars`, `index.chunk_chars`                 | `300`, `1500`                              | snippet and chunk size                                                    |
| `embeddings.enabled`, `embeddings.url`, `embeddings.model` | `true`, `http://localhost:11434`, `bge-m3` | the Ollama embedder                                                       |
| `embeddings.dimensions`                                    | `1024`                                     | vector size of the model; changing model or size drops the stored vectors |
| `embeddings.query_timeout_seconds`                         | `3`                                        | how long a search waits before falling back to keywords                   |
| `embeddings.retry_after_seconds`                           | `60`                                       | how long searches skip the embedder after a failure                       |
| `users.<username>`                                         | none                                       | Authentik username to the user's own area directory                       |

## Authentik setup

Create an application `memex` with an **OAuth2/OpenID provider**:

- **Client type: Public.** Claude Code uses the authorization code flow with PKCE and no secret.
- **Redirect URIs:** `http://localhost:<port>/callback`, strict, with the port you give Claude Code as
  `--callback-port`.
- **Signing key:** select a key pair; Authentik lists them under **System > Certificates**. Only the key pair matters:
  Authentik signs the tokens with the private key and publishes the public key in the provider's JWKS. The certificate
  Authentik stores with the key plays no part in verification; Authentik also puts it into the JWKS as `x5c`, which this
  server ignores. RSA with at least 2048 bits (RS256) and EC P-256 (ES256) are both fine; they are the algorithms
  `auth.algorithms` accepts by default. Generate a key pair for this provider alone, so it can be rotated without
  touching other applications, and pick RSA or ECDSA (P-256) in the form: Ed25519 and Ed448 keys sign EdDSA tokens, and
  imported P-384 or P-521 keys sign ES384 or ES512, which the default `auth.algorithms` rejects. Without a signing key
  Authentik signs with HS256 and the client secret and publishes no JWKS, which this server does not accept. Leave
  **Encryption key** empty.
- **Scopes:** the default mappings `openid`, `profile` (carries `preferred_username` and `groups`) and `offline_access`
  (refresh tokens).
- **Include claims in id_token:** on (the default). Authentik builds the access token from the same claims, so with this
  off the token carries no username or groups and every request is rejected.
- **Issuer mode:** each provider has its own issuer (the default); `auth.issuer` is then
  `https://<authentik>/application/o/<slug>/`.
- Put the client id into `auth.client_ids`.
- **Usernames must not be changeable by users.** Rights follow `preferred_username`, which is the Authentik username: a
  user who could rename themselves to someone else's mapped name would read that person's area. Keep **Allow users to
  change username** off under **System > Settings** (it is off by default). Mapping by the immutable `sub` claim instead
  is the alternative; version 1 maps by username.
- Create the groups `memex` (everyone who uses memex) and `household` (who reads the shared area). Binding `memex` to
  the application as well keeps everyone else from even obtaining a token.

These follow Authentik 2026.8's provider code and docs (`authentik/providers/oauth2/id_token.py`, `models.py`,
`blueprints/system/providers-oauth2.yaml`); check a decoded token from your instance once, as described in the
[setup check](#checking-a-live-token).

### Checking a live token

After logging in, decode the access token's payload (the middle part, base64url) and check that it carries `iss` equal
to `auth.issuer`, `aud` and `azp` equal to the client id, `preferred_username`, and `groups` with `memex`.

## Claude Code setup

```bash
claude mcp add --transport http --client-id <client-id> --callback-port <port> memex https://<host>/mcp
```

Then run `/mcp` in Claude Code and log in. Claude Code finds Authentik through `/.well-known/oauth-protected-resource`.
If its discovery of Authentik's metadata fails, point it at
`https://<authentik>/application/o/<slug>/.well-known/openid-configuration` with the `oauth.authServerMetadataUrl`
setting.

## Running

Locally, with [mise](https://mise.jdx.dev):

```bash
mise run setup
cp config.example.toml config.toml   # edit: issuer, client ids, users, paths, remote
MEMEX_CONFIG=config.toml mise run serve
```

As a container (`ghcr.io/danielcopper/memex-mcp`), with a read-only root filesystem; the clone and the index live on the
`/data` volume, which must be writable by uid 10001:

```bash
docker run -d --name memex-mcp --read-only --tmpfs /tmp \
  -v memex-data:/data -v ./config.toml:/config/config.toml:ro \
  -e MEMEX_REPO_REMOTE=https://<user>:<token>@git.example.org/alice/memex.git \
  -p 8000:8000 ghcr.io/danielcopper/memex-mcp:latest
```

In the config, set `repo.path = "/data/clone"` and `index.path = "/data/index/memex.sqlite3"`. The healthcheck calls
`/healthz` on port 8000.

Image tags: `latest` and `sha-<short>` follow `main`; each release adds `<version>` and `<major>.<minor>` (`0.1.0`,
`0.1`). release-please cuts the releases from the Conventional Commits merged to `main`: it bumps the version in
`pyproject.toml` and writes `CHANGELOG.md`.

## Development

```bash
mise run setup      # editable install + dev tools into .venv
mise run lint       # ruff check + ruff format --check
mise run typecheck  # basedpyright, zero warnings
mise run complexity # cognitive complexity of every function at most 15
mise run test       # pytest, no network
```

The tests cover the rights matrix (each identity against each area through each tool), path attacks, token validation
against a locally generated key and a JWKS endpoint on 127.0.0.1, search ranking with archive down-weighting and the
keyword fallback, incremental re-indexing against a local bare repository, and the HTTP surface end to end.
