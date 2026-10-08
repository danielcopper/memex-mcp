"""Configuration: the example file, environment overrides, validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from memex_mcp.config import ConfigError, build_config, load_config
from tests.conftest import raw_config

EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.toml"


def test_example_config_loads() -> None:
    config = load_config(EXAMPLE, env={})
    assert config.auth.client_ids == ("memex-claude-code",)
    assert config.jwks_url == "https://auth.example.org/application/o/memex/jwks/"
    assert config.users == {"alice": "alice", "bob": "bob"}
    assert config.embeddings.model == "bge-m3"


def test_config_path_comes_from_the_environment() -> None:
    config = load_config(env={"MEMEX_CONFIG": str(EXAMPLE)})
    assert config.server.port == 8000


def test_environment_overrides_the_file() -> None:
    env = {
        "MEMEX_SERVER_PORT": "9000",
        "MEMEX_AUTH_CLIENT_IDS": "a, b",
        "MEMEX_EMBEDDINGS_ENABLED": "false",
        "MEMEX_REPO_REMOTE": "https://bot:token@git.example.org/memex.git",
        "MEMEX_INDEX_ARCHIVE_FACTOR": "0.25",
    }
    config = load_config(EXAMPLE, env=env)
    assert config.server.port == 9000
    assert config.auth.client_ids == ("a", "b")
    assert config.embeddings.enabled is False
    assert config.repo.remote.endswith("@git.example.org/memex.git")
    assert config.index.archive_factor == 0.25


@pytest.mark.parametrize(
    ("name", "value"),
    [("MEMEX_SERVER_PORT", "eighty"), ("MEMEX_EMBEDDINGS_ENABLED", "maybe")],
)
def test_bad_environment_values_are_refused(name: str, value: str) -> None:
    with pytest.raises(ConfigError):
        load_config(EXAMPLE, env={name: value})


def test_explicit_jwks_url_wins(tmp_path: Path) -> None:
    raw = raw_config(tmp_path)
    raw["auth"]["jwks_url"] = "https://auth.example.org/keys"
    assert build_config(raw, {}).jwks_url == "https://auth.example.org/keys"


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("server", "public_url"),
        ("auth", "issuer"),
        ("auth", "client_ids"),
        ("repo", "path"),
        ("index", "path"),
    ],
)
def test_required_settings(tmp_path: Path, section: str, key: str) -> None:
    raw = raw_config(tmp_path)
    del raw[section][key]
    with pytest.raises(ConfigError, match=key):
        build_config(raw, {})


def test_unknown_keys_and_sections_are_refused(tmp_path: Path) -> None:
    raw = raw_config(tmp_path)
    raw["auth"]["isuer"] = "typo"
    with pytest.raises(ConfigError, match="isuer"):
        build_config(raw, {})
    raw = raw_config(tmp_path)
    raw["admin"] = {}
    with pytest.raises(ConfigError, match="admin"):
        build_config(raw, {})


def test_missing_or_broken_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no config file"):
        load_config(env={})
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "absent.toml", env={})
    broken = tmp_path / "broken.toml"
    broken.write_text("[server\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(broken, env={})
