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


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("server", "port", [8000], "must be an integer"),
        ("server", "port", {"value": 8000}, "must be an integer"),
        ("server", "port", float("inf"), "must be an integer"),
        ("index", "archive_factor", [0.5], "must be a number"),
        ("embeddings", "enabled", 1, "must be a boolean"),
        ("auth", "client_ids", ["claude-code", 7], "must be a list of strings"),
        ("auth", "client_ids", 7, "must be a list of strings"),
        ("server", "host", 5, "must be a string"),
    ],
)
def test_values_of_the_wrong_type_are_refused(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    raw = raw_config(tmp_path)
    raw[section][key] = value
    with pytest.raises(ConfigError, match=rf"^\[{section}\] {key} {message}$"):
        build_config(raw, {})


@pytest.mark.parametrize("section", ["server", "auth", "rights", "repo", "index", "embeddings"])
@pytest.mark.parametrize("value", [5, "auth", [1], [{"issuer": "x"}]], ids=repr)
def test_a_section_that_is_not_a_table_is_refused(
    tmp_path: Path, section: str, value: object
) -> None:
    raw: dict[str, object] = {**raw_config(tmp_path), section: value}
    with pytest.raises(ConfigError, match=rf"^\[{section}\] must be a table$"):
        build_config(raw, {})


@pytest.mark.parametrize("value", [5, "alice", [1]], ids=repr)
def test_users_that_are_not_a_table_are_refused(tmp_path: Path, value: object) -> None:
    raw: dict[str, object] = {**raw_config(tmp_path), "users": value}
    with pytest.raises(ConfigError, match=r"^\[users\] must map usernames to area directories$"):
        build_config(raw, {})


def test_an_unknown_log_level_is_refused(tmp_path: Path) -> None:
    raw = raw_config(tmp_path)
    raw["server"]["log_level"] = "debug"
    assert build_config(raw, {}).server.log_level == "debug"
    raw["server"]["log_level"] = "loud"
    with pytest.raises(ConfigError, match=r"^\[server\] log_level must be a logging level"):
        build_config(raw, {})
