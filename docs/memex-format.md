# The memex format

The shared convention for everyone who writes to a memex repository: memex-mcp, and clients that keep their own clone (a
sync script on a laptop, an editor such as Obsidian, an assistant working in the clone). memex-mcp enforces it for the
writes that go through it, and enforces more than a client has to follow; the [last section](#who-follows-what) says
which rule binds whom. How the server itself writes, pushes and reports is described under
[Writing](../README.md#architecture) in the README.

## Layout

The repository has one top-level folder per area: a private area per person (`alice/`, `bob/`) and a shared one
(`household/`). memex-mcp reads only the areas it is configured for and ignores every other top-level folder.

Each area holds two folders:

- `memory/`: working notes, what an assistant remembers.
- `wiki/`: reference knowledge that has proven durable.

A **topic** is a folder directly under `<area>/memory/` or `<area>/wiki/`, for example `alice/memory/homelab/` or
`alice/wiki/networking/`. A topic may have subfolders; they belong to it. Notes directly in `<area>/memory/` belong to
that folder itself, which then acts as a topic of its own. Wiki notes belong in a topic folder: memex-mcp creates none
directly in `<area>/wiki/`, whose only index is the master index of topics, but notes already there can still be
changed.

Two folder names have a meaning wherever they appear:

- `daily/` holds daily notes, for example `alice/memory/daily/2026-10-10.md`. Daily notes are not listed in an index.
- `archive/` holds historical notes. Search ranks them lower and marks them `archived`.

Notes are Markdown files ending in `.md`, in UTF-8. Hidden files and folders (`.git`, `.obsidian`) are not notes. A
note's title is its first level-one heading (`# DNS`) outside fenced code blocks, else its file name without `.md`.

```text
alice/
  memory/
    MEMORY.md            main index: the topics, and the notes directly in memory/
    profile.md           short facts about alice
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
    MEMORY.md            main index of household/memory
    wifi.md
```

A person may keep a **profile** at `<area>/memory/profile.md`: short facts about them only (who they are, their home and
what they own and use, habits, the people around them), no links to other notes, under 300 words. memex-mcp's
instructions tell assistants to read it at the start of a conversation. memex-mcp treats it as an ordinary note directly
in `<area>/memory/`.

## Indexes

Each topic has one index file: `MEMORY.md` in memory, `_index.md` in the wiki, in the topic folder. A note is listed in
its topic's index, also when it lies in a subfolder of the topic; an index file in a subfolder is not used.

Each kind also has a list of the area's topics, one line per topic. In memory it is the **main index**
`<area>/memory/MEMORY.md`, which also lists the notes directly in `<area>/memory/`; in the wiki it is the **master
index** `<area>/wiki/_master-index.md`, which lists topics only.

An index is split into **sections** by level-two headings (`## Services`). A section name is one line of at most 100
characters, under the same character rules as a description (below). The lines of an index are list items in these
formats:

| Index               | Line                                                       | Example                                                   |
| ------------------- | ---------------------------------------------------------- | --------------------------------------------------------- |
| `MEMORY.md`         | `- [<path>](<path>) — <description>`                       | `- [network/vlan.md](network/vlan.md) — VLANs`            |
| `_index.md`         | `- [[<name>]] — <description>`                             | `- [[dhcp]] — leases and reservations`                    |
| master index        | `- [[wiki/<topic>/_index\|<topic>]] — <description>`       | `- [[wiki/networking/_index\|networking]] — home network` |
| main index, a topic | `- [<topic>/MEMORY.md](<topic>/MEMORY.md) — <description>` | `- [homelab/MEMORY.md](homelab/MEMORY.md) — servers`      |

- In `MEMORY.md` the path is relative to the folder of the index; a path with whitespace, parentheses or angle brackets
  goes in angle brackets: `- [my note.md](<my note.md>) — …`.
- In `_index.md` the name is the note's file name without `.md`, the way Obsidian links it.
- Master index links start at the area folder.
- The description is one line of at most 300 characters; the limit counts the description alone, not the whole line. It
  holds no control or format characters (such as bidi overrides), no `<!--`, and no `@` at the start of a word: Claude
  Code reads `@path` in a `MEMORY.md` it loads as a file to load as well.

**Adding a note.** Whoever creates a note adds its line in the same commit, at the end of a section they choose. A
section whose name equals an existing one ignoring case is that section; a section that does not exist yet is added at
the end of the index. When the topic has no index yet, the writer creates it with a heading naming the topic
(`# networking`; for `<area>/memory/` itself, `# <area> memory`).

**Adding a topic.** The first note of a new topic, one whose folder has no index yet, also brings a description of the
topic, and the writer adds the topic's line in the same commit:

- In memory, at the end of the main index's section `## Topics` (matched ignoring case), which stands before the
  sections of notes. When the main index has no such section, the writer inserts it after the heading and any text
  before the first section, or at the end of an index without sections; an existing `## Topics` stays where it is. A
  missing main index is created as `# <area> memory` with a section `## Topics`.
- In the wiki, at the end of the master index's first section (at the end of the file when it has no section). A missing
  master index is created as `# Wiki master index` with a section `## Topics`.

A topic the list already names gets no second line. `archive/` is a topic like any other; notes in `daily/` take no
index line, so a `daily/` folder never starts a topic.

Reordering, rewording and removing index lines are left to a cleanup run started by a person. Deleting a note leaves its
line in place until then.

## Frontmatter

A note may begin with a YAML block, for example:

```markdown
---
type: project
name: dns
description: The resolver, its upstreams and the local zone
---
```

memex-mcp neither reads nor checks it. Dates are not part of the frontmatter: memex-mcp takes them from git and returns
them with every search hit and `read`, as the README defines them under [Clone and index](../README.md#architecture).

Planned, not yet in use: `supersedes:` and `superseded-by:`, which will link a note to a revised version written as a
new note.

## Protected files

- `CLAUDE.md`, `CLAUDE.local.md` and `AGENTS.md`, in any folder, are read by assistants (Claude Code and others) that
  open the folder. memex-mcp never creates, changes or deletes them, nor a folder of such a name, in any letter case and
  also with trailing dots or spaces.
- Index files (`MEMORY.md`, `_index.md`, `_master-index.md`) change only through creating a note, which adds the note's
  line and, when needed, creates the index, a section or the topic's line in the main index or the master index; and
  through cleanup. memex-mcp treats these names like the protected ones: in any letter case, and as folder names too.

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
commits as itself. A message never carries a note's content. Since version 0.4.0, memex-mcp ends each message with a git
trailer naming the OAuth client the write came through, for example `Memex-Client: memex-openwebui`, so that the writes
from one front end, such as a chat, can be found and reverted: `git log --format='%(trailers:key=Memex-Client)'` lists
the client of every commit that carries one.

## Sync and conflicts

Writers push to the followed branch (`main`); nobody force-pushes it. Before pushing, a writer fetches and rebases its
own commits onto the remote.

When that rebase conflicts, the writer pushes its own state to a branch `conflict/<writer>-<date>`, for example
`conflict/memex-mcp-2026-10-10` (`-2`, `-3` for further conflicts on the same day), goes on from the remote's state, and
tells a person. Resolving the conflict and then deleting the branch is up to a person; memex-mcp reports
`"repo":"conflict"` on `/healthz` while one of its own conflict branches exists.

## Who follows what

A client with a clone must follow three rules: the commit message format, the conflict branches, and a new note's index
line in the commit that creates the note. Every other rule holds only for writes through memex-mcp.

| Rule                                                                                   | memex-mcp, for writes through it | A client with a clone |
| -------------------------------------------------------------------------------------- | -------------------------------- | --------------------- |
| Commit messages, conflict branches                                                     | done                             | must follow           |
| A new note's index line, in the same commit                                            | enforced                         | must follow           |
| Write only in your own areas                                                           | enforced                         | —                     |
| Write only `.md` notes below `<area>/memory/` or `<area>/wiki/`                        | enforced                         | —                     |
| No hidden files, no symlinks                                                           | enforced                         | —                     |
| Protected files and index files                                                        | enforced                         | —                     |
| New wiki notes only in a topic folder                                                  | enforced                         | —                     |
| A note at most 64 KiB after a write (`write.max_note_bytes`)                           | enforced                         | —                     |
| No machine secrets (private keys, access tokens) in written text or paths              | enforced                         | —                     |
| Index texts and paths without control or format characters, `<!--` or a word-start `@` | enforced                         | —                     |

git has no rights per folder: whoever has a clone can read every area, and whoever may push can write every area. The
rights hold only for access through memex-mcp.
