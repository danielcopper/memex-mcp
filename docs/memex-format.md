# The memex format

The shared convention for everyone who writes to a memex repository: memex-mcp, and clients that keep their own clone (a
sync script on a laptop, an editor such as Obsidian, an assistant working in the clone). memex-mcp enforces part of it
for the writes that go through it; a client with a clone has to follow it on its own. The
[last section](#what-the-server-enforces) says which is which.

## Layout

The repository has one top-level folder per area: a private area per person (`alice/`, `bob/`) and a shared one
(`household/`). memex-mcp reads only the areas it is configured for and ignores every other top-level folder.

Each area holds two folders:

- `memory/`: working notes, what an assistant remembers.
- `wiki/`: reference knowledge that has proven durable.

A **topic** is a folder directly under `<area>/memory/` or `<area>/wiki/`, for example `alice/memory/homelab/` or
`alice/wiki/networking/`. A topic may have subfolders; they belong to it. Notes directly in `<area>/memory/` belong to
that folder itself, which then acts as a topic of its own.

Two folder names have a meaning wherever they appear:

- `daily/` holds daily notes, one file per day, for example `alice/memory/daily/2026-10-10.md`. Daily notes are not
  listed in an index.
- `archive/` holds historical notes. Search ranks them lower and marks them `archived`.

Notes are Markdown files ending in `.md`, in UTF-8. Hidden files and folders (`.git`, `.obsidian`) are not notes.

```text
alice/
  memory/
    homelab/
      MEMORY.md          index of the topic
      dns.md
      network/vlan.md    listed in homelab/MEMORY.md
    daily/2026-10-10.md  no index line
  wiki/
    _master-index.md     the area's list of wiki topics
    networking/
      _index.md
      dhcp.md
household/
  memory/
    MEMORY.md            index of household/memory itself
    wifi.md
```

## Indexes

Each topic has one index file: `MEMORY.md` in memory, `_index.md` in the wiki. An area's wiki also has
`<area>/wiki/_master-index.md`, which lists its topics.

The index of a note is the index file in the note's own folder; if there is none, the nearest one above it, up to the
topic folder. A note in `alice/memory/homelab/network/` is listed in `alice/memory/homelab/MEMORY.md` unless `network/`
has its own `MEMORY.md`.

An index is split into **sections** by level-two headings (`## Services`). Its lines are list items in one of two
formats:

| Index        | Line                                 | Example                                        |
| ------------ | ------------------------------------ | ---------------------------------------------- |
| `MEMORY.md`  | `- [<path>](<path>) — <description>` | `- [network/vlan.md](network/vlan.md) — VLANs` |
| `_index.md`  | `- [[<name>]] — <description>`       | `- [[dhcp]] — leases and reservations`         |
| master index | `- [[wiki/<topic>/_index\|<topic>]]` | `- [[wiki/networking/_index\|networking]]`     |

- In `MEMORY.md` the path is relative to the index's folder; a path with spaces or brackets goes in angle brackets:
  `- [my note.md](<my note.md>) — …`.
- In `_index.md` the name is the note's file name without `.md`, the way Obsidian links it.
- Master index links start at the area folder.
- The description is one line of at most 300 characters.

**Adding a note.** Whoever creates a note adds its line in the same commit, at the end of a section they choose. A
section whose name equals an existing one ignoring case is that section; a section that does not exist yet is added at
the end of the index. When the topic has no index yet, the writer creates it with a heading naming the topic
(`# networking`; for `<area>/memory/` itself, `# <area> memory`). For a new wiki topic, the writer also adds the topic's
line at the end of the master index and creates the master index (`# Wiki master index`) if it is missing.

Reordering, rewording and removing index lines are left to a cleanup run started by a person. Deleting a note leaves its
line in place until then.

## Frontmatter

A note may begin with a YAML block:

```markdown
---
type: project
name: dns
description: The resolver, its upstreams and the local zone
---
```

`name` equals the file name without `.md`. Dates are not part of the frontmatter: when a note was created and last
changed comes from git (the author dates of the first and the last commit touching it), and memex-mcp returns both with
every search hit and `read`.

Planned, not yet in use: `supersedes:` and `superseded-by:`, which will link a note to a revised version written as a
new note.

## Protected files

- `CLAUDE.md` and `AGENTS.md`, in any folder, are loaded into every assistant session that opens the folder. memex-mcp
  never creates, changes or deletes them; in a clone, only a person changes them on purpose.
- Index files (`MEMORY.md`, `_index.md`, `_master-index.md`) change only by the line added for a new note, and by
  cleanup.

## Commits

One commit per change. The message is English, with no type or scope:

- one file: `<writer>: <action> <path>`, for example `memex-mcp: append alice/memory/daily/2026-10-10.md`;
- several files: `<writer>: <N> files in <folder>`, naming the deepest folder they share (or `<writer>: <N> files` when
  they share none), then a blank line and one `<action> <path>` per file:

  ```text
  memex-mcp: 3 files in alice/wiki

  create alice/wiki/cooking/bread.md
  create alice/wiki/cooking/_index.md
  update alice/wiki/_master-index.md
  ```

The writer is the program that commits: `memex-mcp` for the server, the machine's name for a clone (`laptop`). The
actions are `create`, `update`, `append` and `delete`. The commit's author is who stands behind the change; memex-mcp
sets the calling person (their username, and an e-mail address from `write.author_email`) and commits as itself. A
message never carries a note's content.

## Sync and conflicts

Writers push to the followed branch (`main`); nobody force-pushes it. Before pushing, a writer fetches and rebases its
own commits onto the remote.

When that rebase conflicts, the writer pushes its own state to a branch `conflict/<writer>-<date>`, for example
`conflict/memex-mcp-2026-10-10` (`-2`, `-3` for further conflicts on the same day), goes on from the remote's state, and
tells a person. memex-mcp reports `"repo":"conflict"` on `/healthz` while one of its conflict branches exists. A person
resolves the conflict and deletes the branch.

memex-mcp itself works like this: it commits and pushes within each write. When the remote turns the push down because
it has newer commits, memex-mcp fetches and applies the write again on the new state, up to three times; when the remote
cannot be reached, the commit stays in its clone and the refresh loop pushes it later, rebasing it as above.

## What the server enforces

| Rule                                                             | memex-mcp, for writes through it | A client with a clone |
| ---------------------------------------------------------------- | -------------------------------- | --------------------- |
| Write only in your own areas                                     | enforced                         | must follow           |
| Write only `.md` notes below `<area>/memory/` or `<area>/wiki/`  | enforced                         | must follow           |
| No hidden files, no symlinks                                     | enforced                         | must follow           |
| Protected files                                                  | enforced                         | must follow           |
| A new note's index line, in the same commit                      | enforced                         | must follow           |
| A note at most 64 KiB after a write (`write.max_note_bytes`)     | enforced                         | recommended           |
| No machine secrets (private keys, access tokens) in written text | enforced                         | recommended           |
| Commit messages, conflict branches                               | done                             | must follow           |

git has no rights per folder: whoever has a clone can read and write every area. The rights hold only for access through
memex-mcp.
