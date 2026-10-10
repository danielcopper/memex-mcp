"""The server's own clone: fetch, commit, push, rebase, and what changed between two commits."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Not the shorter ``://\S*@``: on a long token with many ``://`` and no ``@``
# it rescans the token from each ``://``, which takes quadratic time.
_URL_TAIL = re.compile(r"://\S*")


def _hide_credential(match: re.Match[str]) -> str:
    tail = match[0]
    if "@" not in tail:
        return tail
    return "://***@" + tail.rpartition("@")[2]


def redact(text: str) -> str:
    """Hide a credential embedded in a remote URL (``https://user:token@host``).

    Everything from ``://`` up to the last ``@`` before the next whitespace
    becomes ``***``, so the credential may hold any character but whitespace
    (``/`` and ``@`` too). The price: any later ``@`` before the next
    whitespace, in the path or in text glued to the URL, hides what lies
    before it too (``https://host/a@b/r.git`` becomes ``https://***@b/r.git``).
    """
    return _URL_TAIL.sub(_hide_credential, text)


class GitError(Exception):
    """A git command failed; the message carries git's own stderr."""


# The server's name as a writer: the committer of every write, the first word of
# its commit messages and of its conflict branches.
WRITER = "memex-mcp"
COMMITTER_EMAIL = "memex-mcp@memex.invalid"
CONFLICT_PREFIX = f"conflict/{WRITER}-"
# A path list longer than this reads the dates from the whole history at once.
_MAX_DATE_PATHS = 100


class Push(Enum):
    PUSHED = "pushed"
    # The remote has commits the clone lacks: fetch, and try again on top of them.
    REJECTED = "rejected"
    # Anything else: unreachable, refused by a hook, timed out.
    FAILED = "failed"


@dataclass(frozen=True)
class Author:
    name: str
    email: str


def _identity_env(author: Author | None) -> dict[str, str]:
    """The service commits; the author is the caller, or the service when there is none."""
    author = author or Author(WRITER, COMMITTER_EMAIL)
    return {
        "GIT_AUTHOR_NAME": author.name,
        "GIT_AUTHOR_EMAIL": author.email,
        "GIT_COMMITTER_NAME": WRITER,
        "GIT_COMMITTER_EMAIL": COMMITTER_EMAIL,
    }


def _parse_dates(out: str) -> dict[str, tuple[str, str]]:
    """``git log -z --name-only --format=%x01%aI``, newest first, as path -> (created, updated)."""
    dates: dict[str, tuple[str, str]] = {}
    date = ""
    for field in out.split("\x00"):
        name = field.lstrip("\n")
        if name.startswith("\x01"):
            date = name[1:]
        elif name:
            # Newest first: the first date seen is the last change, the last one the first.
            updated = dates[name][1] if name in dates else date
            dates[name] = (date, updated)
    return dates


@dataclass(frozen=True)
class GitRepo:
    path: Path
    branch: str
    timeout: float

    def _git(
        self, *args: str, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> str:
        done = self._run(*args, cwd=cwd, env=env)
        if done.returncode != 0:
            raise GitError(f"git {args[0]} failed: {redact(done.stderr.strip())}")
        return done.stdout

    def _run(
        self, *args: str, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run git; raises GitError only when it does not finish in time."""
        command = ["git", "-C", str(cwd or self.path), *args]
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                encoding="utf-8",
                errors="surrogateescape",
                timeout=self.timeout,
                check=False,
                # Never stop to ask for a password (a fetch without credentials
                # fails instead), and report errors in English for the log.
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **(env or {})},
            )
        except subprocess.TimeoutExpired:
            done = None
        if done is None:
            # Raised outside the except block: TimeoutExpired carries the
            # command, which for a clone holds the remote and its credential,
            # and would stay attached as __context__ even with ``from None``.
            raise GitError(f"git {args[0]} timed out after {self.timeout:g}s")
        return done

    def is_repo(self) -> bool:
        return (self.path / ".git").exists()

    def clone(self, remote: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._git(
            "clone", "--branch", self.branch, "--", remote, str(self.path), cwd=self.path.parent
        )

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    def has_commit(self, sha: str) -> bool:
        try:
            self._git("cat-file", "-e", f"{sha}^{{commit}}")
        except GitError:
            return False
        return True

    def fetch(self) -> None:
        """Fetch the branch and the server's conflict branches; gone ones are pruned."""
        self._git(
            "fetch",
            "--quiet",
            "--prune",
            "origin",
            f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}",
            f"+refs/heads/{CONFLICT_PREFIX}*:refs/remotes/origin/{CONFLICT_PREFIX}*",
        )

    def fast_forward(self) -> None:
        self._git("merge", "--ff-only", "--quiet", f"origin/{self.branch}")

    def ahead(self) -> int:
        """How many commits the clone has that the remote, as last fetched, lacks."""
        return int(self._git("rev-list", "--count", f"origin/{self.branch}..HEAD").strip())

    def commit(self, paths: Sequence[str], message: str, author: Author) -> str:
        """Commit exactly ``paths`` as they are in the working tree; returns the commit."""
        # Literal: a file name with "*" or "[" names only itself.
        literal = {"GIT_LITERAL_PATHSPECS": "1"}
        self._git("add", "--all", "--", *paths, env=literal)
        env = {**_identity_env(author), **literal}
        self._git("commit", "--quiet", "--no-verify", "-m", message, "--", *paths, env=env)
        return self.head()

    def push(self, ref: str = "HEAD", branch: str | None = None) -> Push:
        """Push ``ref`` to ``branch`` (the followed branch by default), never forcing."""
        target = f"refs/heads/{branch or self.branch}"
        done = self._run("push", "--porcelain", "origin", f"{ref}:{target}")
        if done.returncode == 0:
            return Push.PUSHED
        # A ref the remote turned down for lack of commits reads "!<tab>…<tab>[rejected] …".
        rejected = any(
            line.startswith("!") and "\t[rejected]" in line for line in done.stdout.splitlines()
        )
        return Push.REJECTED if rejected else Push.FAILED

    def reset_hard(self, ref: str) -> None:
        self._git("reset", "--quiet", "--hard", ref)

    def rebase(self) -> bool:
        """Replay the clone's own commits on the fetched remote; on a conflict, undo and say so."""
        done = self._run("rebase", "--quiet", f"origin/{self.branch}", env=_identity_env(None))
        if done.returncode == 0:
            return True
        self._run("rebase", "--abort")
        return False

    def conflict_branches(self) -> list[str]:
        """The server's conflict branches on the remote, as last fetched."""
        out = self._git(
            "for-each-ref",
            "--format=%(refname:lstrip=3)",
            # A pattern without a glob matches only up to a slash.
            f"refs/remotes/origin/{CONFLICT_PREFIX}*",
        )
        return [name for name in out.splitlines() if name]

    def dates(self, paths: Sequence[str] | None = None) -> dict[str, tuple[str, str]]:
        """Path -> (created, updated): the author dates of the first and the last commit
        touching it, in ISO 8601. ``None`` reads every path in one pass over the history.
        """
        command = ("log", "-z", "--name-only", "--no-renames", "--format=%x01%aI", "HEAD")
        if paths is None or len(paths) > _MAX_DATE_PATHS:
            dates = _parse_dates(self._git(*command))
            return dates if paths is None else {p: dates[p] for p in paths if p in dates}
        if not paths:
            return {}
        # Literal: a file name with "*" or "[" names only itself.
        out = self._git(*command, "--", *paths, env={"GIT_LITERAL_PATHSPECS": "1"})
        return _parse_dates(out)

    def changed_paths(self, old: str, new: str) -> list[str]:
        """Every path added, modified or deleted between two commits.

        ``--no-renames`` reports a rename as a deletion plus an addition, so
        both the old and the new path come back.
        """
        out = self._git("diff", "--name-only", "--no-renames", "-z", old, new)
        return [path for path in out.split("\x00") if path]
