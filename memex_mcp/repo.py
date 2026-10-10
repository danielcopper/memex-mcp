"""The server's own clone: fetch, fast-forward, and what changed between two commits."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
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


@dataclass(frozen=True)
class GitRepo:
    path: Path
    branch: str
    timeout: float

    def _git(self, *args: str, cwd: Path | None = None) -> str:
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
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"},
            )
        except subprocess.TimeoutExpired:
            done = None
        if done is None:
            # Raised outside the except block: TimeoutExpired carries the
            # command, which for a clone holds the remote and its credential,
            # and would stay attached as __context__ even with ``from None``.
            raise GitError(f"git {args[0]} timed out after {self.timeout:g}s")
        if done.returncode != 0:
            raise GitError(f"git {args[0]} failed: {redact(done.stderr.strip())}")
        return done.stdout

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
        self._git("fetch", "--quiet", "origin", self.branch)

    def fast_forward(self) -> None:
        self._git("merge", "--ff-only", "--quiet", f"origin/{self.branch}")

    def changed_paths(self, old: str, new: str) -> list[str]:
        """Every path added, modified or deleted between two commits.

        ``--no-renames`` reports a rename as a deletion plus an addition, so
        both the old and the new path come back.
        """
        out = self._git("diff", "--name-only", "--no-renames", "-z", old, new)
        return [path for path in out.split("\x00") if path]
