"""The suite never touches the real HOME or the user's git configuration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_home_is_a_temporary_directory(isolated_home: Path) -> None:
    assert os.environ["HOME"] == str(isolated_home)
    assert Path.home() == isolated_home
    assert not any(name.startswith("MEMEX_") for name in os.environ)


def test_git_sees_no_user_or_system_config(isolated_home: Path) -> None:
    done = subprocess.run(
        ["git", "config", "--show-origin", "--list"],
        capture_output=True,
        text=True,
        check=False,
        cwd=isolated_home,  # outside any repository: only global and system config apply
    )
    origins = {line.split("\t", 1)[0] for line in done.stdout.splitlines()}
    assert all(str(isolated_home) in origin or "command line" in origin for origin in origins), (
        origins
    )
