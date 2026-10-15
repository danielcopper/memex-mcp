# memex-mcp

MCP server for a shared, per-person Markdown memory in git: search, read and write with rights from your identity
provider.

The memory ("memex") is a git repository of Markdown notes with one top-level directory per area: a private area per
person (`alice/`, `bob/`) and a shared one (`household/`). Assistants such as Claude Code reach it through this server,
which searches the notes by keyword and by meaning, returns short snippets, reads single notes, writes notes as commits
once writing is switched on (`write.enabled`), and lets each caller see and change only their own areas. People with a
clone of their own write to the repository directly; [docs/memex-format.md](docs/memex-format.md) is the convention both
follow.

## Architecture

**Clone and index.** The server owns a clone of the memex repository. At start it clones it if needed and builds a
SQLite search index of the configured areas (the users' areas and the household area; other top-level folders are not
read); the index is a cache on a volume, never committed, and rebuilt from scratch when it is missing, its schema
version differs, or the set of configured areas changed. Since the clone is the server's alone, the server drops at
start whatever was never committed (a crash between writing a note and committing it), and it ends a rebase left behind
before each write and before a refresh round moves the clone. Every 60 seconds a background loop runs `git fetch`,
brings the clone onto the remote (`git merge --ff-only`, or, with writing on, a rebase of commits the server could not
push yet, which it then pushes) and re-indexes exactly the paths that changed between the old and the new HEAD. A failed
fetch is logged and the last state keeps serving. The index also keeps each note's dates from git: `created` is the
author date of the first commit touching it, `updated` the commit date of the last one, so a change rebased onto newer
commits counts from when it landed.

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

**Writing.** With writing off (`write.enabled = false`, the default), the server offers no write tools and names none in
its instructions, and a read-only credential suffices. With writing on, each write is one commit in the server's clone,
pushed within the call: the caller is the commit's author (their username, and an e-mail address from
`write.author_email`), the server its committer (`memex-mcp`). The message names the action and the path, never the
content, and ends with a `Memex-Client` git trailer naming the client of the caller's token (its `azp`);
[docs/memex-format.md](docs/memex-format.md#commits) describes the format and what the trailer is for. The answer says
whether the push went through. When the remote has commits the clone lacks, the server fetches and applies the write
again on that state, up to three times; a write that no longer fits (the passage to replace is gone, the note changed
since it was read, the path is taken) fails with that reason. When the remote cannot be reached, the commit stays in the
clone, the answer says it is saved but not yet pushed, and every refresh round tries to push it again. When the remote
refuses the push (a hook, a protected branch, missing rights), the answer says so and the commit stays in the clone;
every refresh round tries again, and `/healthz` reports `failing` while the refusal lasts. If saved commits conflict
with the remote when they are rebased, the server pushes them to a branch `conflict/memex-mcp-<date>`, logs an error,
follows the remote again and reports `conflict` on `/healthz` until that branch is deleted; should even that push fail,
a write waiting behind them is saved without further tries. One lock serialises the writes and the loop's rebasing and
pushing; the loop's fetch runs outside it. After the commit the server re-indexes the touched notes, so search shows the
change at once; should that fail, the write still counts and search catches up in the next round.

Guards on every write: a note may hold at most `write.max_note_bytes` (64 KiB) afterwards; new text or a path that
contains a private key block or an access token with a well-known prefix (`ghp_`, `github_pat_`, `sk-`, `AKIA`, `xoxb-`,
`glpat-` and similar) is refused, naming the kind but never the value, while passwords written in prose are allowed. A
new note's line goes into its topic's index in the same commit, and the first note of a new wiki topic brings the
topic's line for the master index. Index lines, section names, topic descriptions and paths hold no control or format
characters, no `<!--` and no `@` at the start of a word (which Claude Code would follow as a file to load); a new note's
index line and every heading it adds are checked again as composed, before they are written.
[docs/memex-format.md](docs/memex-format.md) describes indexes, sections and line formats.

**Authentication.** The server is an OAuth 2.1 resource server; Authentik is the authorization server. Clients find
Authentik through the protected resource metadata (RFC 9728) at `/.well-known/oauth-protected-resource` and obtain
tokens there; the server only validates them. Authentik issues access tokens as JWTs signed with the provider's signing
key, so each token is checked locally against the provider's JWKS: signature, `iss`, `exp`, `aud` and `azp` (both must
be one of the configured client ids). The caller's username comes from `preferred_username`, their groups from `groups`.
A token must name its signing key (`kid`). The key set is cached for five minutes, so a key removed from the provider
stops verifying within five minutes. A token naming a key the cached set lacks fetches the set again, but not sooner
than `auth.jwks_min_refetch_seconds` (60 by default) after the last successful fetch. After a failed fetch the next one
waits as long, counted from the failure and at least `auth.timeout_seconds`. When the set cannot be fetched once the
five minutes are over, every token is rejected until a fetch succeeds; a token whose key is cached never waits for a
fetch.

Several providers may issue tokens. `[auth]` is the provider Claude Code logs in at (see
[Claude Code setup](#claude-code-setup)), and each `[[auth.extra_issuers]]` entry is a further provider dedicated to
memex for another front end. A token goes to the provider whose issuer equals its `iss` exactly and is checked against
that provider's key set, algorithms and client ids only: a client id or algorithm allowed for one provider is never
allowed for another. A token naming any other issuer is refused, with one warning, before any key set is fetched.

Each provider has its own cached key set and its own wait after a failed fetch, so one whose key set cannot be fetched
does not hold up the others; the leeway, the refetch wait and the timeout are those of `[auth]`. The protected resource
metadata lists the issuer of `[auth]` first, then each further one in the order of the file. Users of every provider go
through the same `[users]` mapping and group checks.

**Rights.** Access requires membership in the group `memex`. A user reads the area their username maps to in the config
and, as a member of the group `household`, the shared area too. Nothing else; there is no admin override. A user writes
exactly where they read, and only below `<area>/memory/` or `<area>/wiki/`. Every tool enforces this on the server, and
paths are checked before and after following symlinks: only `.md` files, inside an allowed area, no `..`, no absolute
paths, no hidden segments (`.git`, `.obsidian`). A write follows no symlink at all and takes no control or format
character, line break or `<!--` in its path. No write names `CLAUDE.md`, `CLAUDE.local.md` or `AGENTS.md`, in any letter
case, as a file or a folder. Index files (`MEMORY.md`, `_index.md`, `_master-index.md`, matched the same way) change
only through `create`, which adds a new note's line and creates the index, a section or the master index line when
needed. A note, folder or area outside the caller's rights gets exactly the answer a missing one gets, so the server
does not reveal what other people have.

**Tools.**

The write tools and `index_sections` exist only with `write.enabled` on.

| Tool                                                                  | Does                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `search(query, area=None, limit=10)`                                  | returns `semantic`, optional `notice`, and `hits`: one per note with `path`, `area`, `title`, `snippet` (up to 300 characters, `index.snippet_chars`, around the best match), `score`, `archived`, `created`, `updated`                                                                                                                                          |
| `read(path)`                                                          | returns the full note: `path`, `area`, `title`, `archived`, `content`, `version` (git's blob id of the content), `created`, `updated`                                                                                                                                                                                                                            |
| `list(area, folder=None)`                                             | returns the notes and folders of one folder: `name`, `path`, `type` (`note` or `folder`)                                                                                                                                                                                                                                                                         |
| `areas()`                                                             | returns the caller's own areas: `areas` (the own area, plus `household` for members of the household group)                                                                                                                                                                                                                                                      |
| `index_sections(path)`                                                | returns the index a new note at `path` (a note's path or a folder) would be listed in, its topic's: `index` (its path, `null` for a daily note), `exists`, `sections`, `new_topic` (true when the note would start a new wiki topic, so `create` needs `topic_description`)                                                                                      |
| `create(path, content, index_section, index_line, topic_description)` | creates a note, refusing an existing path, and adds `index_line` (a description of at most 300 characters) at the end of `index_section` in its topic's index; the first note of a new wiki topic also needs `topic_description` for the master index; a note under a `daily/` folder takes no index line and ignores the rest; a wiki note needs a topic folder |
| `append(path, text)`                                                  | adds `text` at the end of an existing note, on a line of its own                                                                                                                                                                                                                                                                                                 |
| `edit(path, old_text, new_text, version=None)`                        | replaces `old_text`, which must occur exactly once; with `version`, refuses a note that changed since it was read                                                                                                                                                                                                                                                |
| `delete(path, version)`                                               | deletes a note if it still has `version`; its index line stays until a cleanup                                                                                                                                                                                                                                                                                   |

The write tools return `path`, `commit`, `pushed`, `searchable`, the note's new `version` (not after `delete`), for
`create` the `index` it added a line to, and a `notice` when the commit is not pushed yet or search does not show it
yet, or the git host refused it. `created` and `updated` are ISO 8601 timestamps, or `null` while the index has no dates
for the note.

## Trust model

- memex-mcp enforces the per-person rights only for access **through the server**, that is, for assistants.
- git has no read permissions per directory: anyone with the repository (the operator, every full clone, the git host's
  admin) can read every area.
- Whoever runs the server can read everything, since the server must read the notes to index them. Privacy from the
  operator would need client-side encryption, which rules out server-side search.
- Response times are not separated: while the server indexes someone's new notes, searches wait for it, and writes wait
  for each other, so a caller can notice that other areas are being written, though not what they contain.
- With writing on, the server pushes with a credential that can write the whole repository; the per-area write rights
  hold only for writes through the server. A commit's author is what the server sets from the verified token, but git
  itself does not verify authors: anyone else who can push can name any author.
- A further issuer (`[[auth.extra_issuers]]`) must be the same identity provider installation as `[auth]` (the same
  Authentik), with the same usernames: rights follow `preferred_username`, and the server cannot tell whether two
  providers mean the same person by one name. The server does not check this; another identity provider, or one whose
  users can pick their own names, could hand a person someone else's area.
- Each further issuer is a provider and client dedicated to memex: its tokens are issued for memex alone. Never
  configure a provider that also serves another application, such as the one a front end's own login uses. Its tokens
  are meant for that application, which may hand them on to other tools; whoever holds one could then use memex in that
  person's name. The MCP specification rules this out: a server takes only tokens issued for it as their audience, and
  no others
  ([token handling](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization#token-handling)).
  Membership in `rights.access_group` stays the gate for every provider: the server checks it on every call.
- The checks for secrets catch only unambiguous machine secrets in the text a write adds and in its path. Whatever a
  note holds is readable by everyone with the repository, as above.
- Operators who need people separated at the git level would use one repository per person plus one shared repository.
  Version 1 does not support that; it is planned for a later version.

## Configuration

One TOML file, named by `--config` or `MEMEX_CONFIG`; [`config.example.toml`](config.example.toml) lists every setting.
Any scalar or list setting can be overridden by an environment variable `MEMEX_<SECTION>_<KEY>` (lists comma-separated),
for example `MEMEX_REPO_REMOTE` for a remote URL that carries a credential. The `[users]` table and the
`[[auth.extra_issuers]]` entries live in the file only. At startup the server checks types, ranges, URLs, names and
required settings, and refuses to start on a value that fails, naming the setting and the file or variable it came from;
the example file states the rules. A `MEMEX_` variable that names no setting, other than `MEMEX_CONFIG`, also stops the
start.

| Setting                                                    | Default                                    | Meaning                                                                   |
| ---------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| `server.public_url`                                        | required                                   | URL clients use; the advertised resource is `public_url` + `mcp_path`     |
| `server.host`, `server.port`, `server.mcp_path`            | `0.0.0.0`, `8000`, `/mcp`                  | where the server listens                                                  |
| `auth.issuer`                                              | required                                   | Authentik issuer, e.g. `https://auth.example.org/application/o/memex/`    |
| `auth.client_ids`                                          | required                                   | client ids whose tokens are accepted (`aud` and `azp`)                    |
| `auth.jwks_url`                                            | `<issuer>jwks/`                            | the provider's JWKS endpoint                                              |
| `auth.algorithms`                                          | `RS256`, `ES256`                           | accepted signing algorithms                                               |
| `auth.scopes`                                              | `openid`, `profile`, `offline_access`      | scopes advertised to clients                                              |
| `[[auth.extra_issuers]]`                                   | none                                       | further providers (file only), each with its own issuer and client ids    |
| `rights.access_group`                                      | `memex`                                    | group required for any access                                             |
| `rights.household_group`, `rights.household_area`          | `household`, `household`                   | group and directory of the shared area                                    |
| `repo.path`                                                | required                                   | the server's clone                                                        |
| `repo.remote`                                              | none                                       | cloned from here when `repo.path` holds no clone                          |
| `repo.branch`, `repo.fetch_interval_seconds`               | `main`, `60`                               | what to follow and how often                                              |
| `index.path`                                               | required                                   | the SQLite index (a cache); must lie outside `repo.path`                  |
| `index.archive_factor`                                     | `0.5`                                      | score factor for notes under `archive/`                                   |
| `index.snippet_chars`, `index.chunk_chars`                 | `300`, `1500`                              | snippet and chunk size                                                    |
| `embeddings.enabled`, `embeddings.url`, `embeddings.model` | `true`, `http://localhost:11434`, `bge-m3` | the Ollama embedder                                                       |
| `embeddings.dimensions`                                    | `1024`                                     | vector size of the model; changing model or size drops the stored vectors |
| `embeddings.query_timeout_seconds`                         | `3`                                        | a search's embedding call must finish within this, else keywords only     |
| `embeddings.retry_after_seconds`                           | `60`                                       | how long searches skip the embedder after a failure                       |
| `write.enabled`                                            | `false`                                    | offer the write tools; needs a remote the server may push to              |
| `write.max_note_bytes`                                     | `65536`                                    | the largest a note may be after a write through the server, in bytes      |
| `write.author_email`                                       | `{username}@memex.invalid`                 | commit author's e-mail; `{username}` becomes the caller's username        |
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

### A provider for another front end

A front end that connects to memex with its own OAuth client, such as Open WebUI's MCP connection in its "OAuth 2.1
(Static)" mode, gets a provider of its own: Authentik has one client per provider. Its tokens then carry that front
end's client id in `aud` and `azp`, and the commits of its writes name it in the `Memex-Client` trailer.

- Create a further application (for example `memex-openwebui`) with an **OAuth2/OpenID provider**. **Client type:
  Confidential**, since the front end keeps a client secret; the front end's callback URL as a strict redirect URI.
- Bind the group `memex` to this application as well.
- **Scopes** and **Include claims in id_token** as above.
- **Signing key:** an asymmetric key pair, as above; an ECDSA (P-256) key signs ES256.
- **Issuer mode:** each provider has its own issuer. Put that issuer, the client id and the algorithm into an
  `[[auth.extra_issuers]]` entry; its `jwks_url` defaults to `<issuer>jwks/`:

  ```toml
  [[auth.extra_issuers]]
  issuer = "https://auth.example.org/application/o/memex-openwebui/"
  client_ids = ["memex-openwebui"]
  algorithms = ["ES256"]
  ```

- Give the front end this provider's issuer as its OAuth server, with the client id and secret.

### Checking a live token

After logging in, decode the access token's payload (the middle part, base64url) and check that it carries `iss` equal
to `auth.issuer`, `aud` and `azp` equal to the client id, `preferred_username`, and `groups` with `memex`. Check one
token from each further provider the same way: `iss` equal to its entry's `issuer`, `aud` and `azp` one of the entry's
`client_ids`, and the same `preferred_username` as the person has with `[auth]`.

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

### Health

`/healthz` needs no token and answers HTTP 200 whenever the server answers at all, so the container healthcheck marks
the container unhealthy only when the server stops answering. The body says more, in fixed words only:

```text
{"status":"ok","embeddings":"ok","repo":"ok"}
```

| Field        | Value      | Meaning                                                                                                                                                              |
| ------------ | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `status`     | `ok`       | the server answers                                                                                                                                                   |
| `embeddings` | `ok`       | the latest embedding call or check succeeded, or none has run yet                                                                                                    |
|              | `failing`  | the latest embedding call (a search's query or the vectors of new notes) or check failed                                                                             |
|              | `disabled` | `embeddings.enabled` is off                                                                                                                                          |
| `repo`       | `ok`       | the last refresh round fetched, brought the clone onto the remote and pushed what it had to without an error, or only the first round, which does not fetch, has run |
|              | `failing`  | the last round could not fetch, bring the clone onto the remote or push, or broke off with an error, or a write's push was refused since; the last state is served   |
|              | `conflict` | a branch `conflict/memex-mcp-<date>` with commits that conflicted is on the remote, as last fetched; shown instead of `failing`, until the branch is deleted         |

Every refresh round (`repo.fetch_interval_seconds`) checks the embedder. Usually the check asks Ollama for its model
list (`GET /api/tags`) and looks for `embeddings.model` there, which loads no model. It compares names as Ollama
resolves them: `bge-m3`, `library/bge-m3` and `registry.ollama.ai/library/bge-m3` all match `bge-m3:latest`, and a name
with a tag must match that tag. After an embedding call has failed, the check embeds a short text instead, so that the
state recovers without anyone searching; only then does a check load the model. A model list that Ollama answers does
not clear a failed embedding call: Ollama can list the model and still fail to embed. `failing` turns back to `ok` with
the next embedding call or check that succeeds. A search that embeds, or the check's embedding of a short text, also
ends the pause in which searches skip the embedder after a failure; vectors for new notes do not, as they are allowed a
longer timeout than a search. The log says why: a failed check is logged when it begins and its recovery once, not every
round.

Monitors should match the body text, not the status code: for example, a keyword monitor that alerts unless the body
contains `"embeddings":"ok"` (on a server with embeddings `disabled`, that one alerts too), and one for `"repo":"ok"`,
which also alerts on `conflict`; the body has no spaces, as shown above.

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
keyword fallback, incremental re-indexing against a local bare repository, writes with their guards, indexes and
commits, pushes that are turned down or cannot reach the remote, conflict branches, and the HTTP surface end to end.
