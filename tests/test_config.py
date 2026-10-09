"""Configuration: the example file, environment overrides, validation."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

from memex_mcp.config import SHOWN_CHARS, Config, ConfigError, build_config, load_config
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
    ("setting", "value", "message", "shown"),
    [
        (("server", "port"), [8000], "must be a whole number", "[8000]"),
        (("server", "port"), {"value": 8000}, "must be a whole number", "{'value': 8000}"),
        (("server", "port"), float("inf"), "must be a whole number", "inf"),
        (("index", "archive_factor"), [0.5], "must be a number", "[0.5]"),
        (("embeddings", "enabled"), 1, "must be a boolean", "1"),
        (("auth", "client_ids"), ["claude-code", 7], "must be a list of strings", None),
        (("auth", "client_ids"), 7, "must be a list of strings", None),
        (("server", "host"), 5, "must be a string", None),
    ],
)
def test_values_of_the_wrong_type_are_refused(
    tmp_path: Path, setting: tuple[str, str], value: object, message: str, shown: str | None
) -> None:
    section, key = setting
    raw = raw_config(tmp_path)
    raw[section][key] = value
    expected = message if shown is None else f"{message}, got {shown}"
    with pytest.raises(ConfigError, match=rf"^\[{section}\] {key} {re.escape(expected)}$"):
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


def _refusal(raw: dict[str, dict[str, object]], env: dict[str, str] | None = None) -> str:
    with pytest.raises(ConfigError) as caught:
        build_config(raw, env or {})
    return str(caught.value)


@pytest.mark.parametrize(
    ("section", "key", "message"),
    [
        ("server", "port", "must be a whole number"),
        ("index", "archive_factor", "must be a number"),
    ],
)
def test_a_boolean_is_not_a_number(tmp_path: Path, section: str, key: str, message: str) -> None:
    raw = raw_config(tmp_path)
    raw[section][key] = True
    assert _refusal(raw) == f"[{section}] {key} {message}, got True"
    env_name = f"MEMEX_{section.upper()}_{key.upper()}"
    assert _refusal(raw_config(tmp_path), {env_name: "true"}) == (
        f"[{section}] {key} from {env_name} {message}, got 'true'"
    )


@pytest.mark.parametrize("value", [8000.0, 8000.9], ids=repr)
def test_a_float_is_refused_for_a_whole_number_setting(tmp_path: Path, value: float) -> None:
    raw = raw_config(tmp_path)
    raw["server"]["port"] = value
    assert _refusal(raw) == f"[server] port must be a whole number, got {value!r}"
    assert _refusal(raw_config(tmp_path), {"MEMEX_SERVER_PORT": str(value)}) == (
        f"[server] port from MEMEX_SERVER_PORT must be a whole number, got '{value}'"
    )


def test_whole_numbers_come_from_the_file_or_the_environment(tmp_path: Path) -> None:
    raw = raw_config(tmp_path)
    raw["server"]["port"] = 8080
    assert build_config(raw, {}).server.port == 8080
    assert build_config(raw, {"MEMEX_SERVER_PORT": " 9090 "}).server.port == 9090


# Each number setting: values just inside its bound, values just outside it,
# and how the refusal words the bound.
BOUNDS: dict[tuple[str, str], tuple[list[float], list[float], str]] = {
    ("server", "port"): ([1, 65535], [0, 65536], "at least 1 and at most 65535"),
    ("auth", "leeway_seconds"): ([0], [-1], "at least 0"),
    ("auth", "jwks_min_refetch_seconds"): ([0], [-1], "at least 0"),
    ("auth", "timeout_seconds"): ([0.001], [0, -0.001], "greater than 0"),
    ("repo", "fetch_interval_seconds"): ([1], [0.999, 0], "at least 1"),
    ("repo", "git_timeout_seconds"): ([0.001], [0, -0.001], "greater than 0"),
    ("index", "archive_factor"): ([0.001, 1], [0, 1.001], "greater than 0 and at most 1"),
    ("index", "snippet_chars"): ([40], [39], "at least 40"),
    ("index", "chunk_chars"): ([100], [99], "at least 100"),
    ("index", "max_limit"): ([1], [0], "at least 1"),
    ("embeddings", "dimensions"): ([1], [0], "at least 1"),
    ("embeddings", "query_timeout_seconds"): ([0.001], [0, -0.001], "greater than 0"),
    ("embeddings", "index_timeout_seconds"): ([0.001], [0, -0.001], "greater than 0"),
    ("embeddings", "retry_after_seconds"): ([0], [-0.001], "at least 0"),
    ("embeddings", "batch_size"): ([1], [0], "at least 1"),
}


def _settings(config: Config) -> dict[str, dict[str, object]]:
    return cast("dict[str, dict[str, object]]", asdict(config))


def _settings_of_type(kind: type) -> list[tuple[str, str]]:
    """The (section, key) of every setting whose default has exactly this type."""
    return [
        (section, key)
        for section, values in _settings(Config()).items()
        if section != "users"
        for key, value in values.items()
        if type(value) is kind
    ]


FLOAT_SETTINGS = _settings_of_type(float)


def test_every_number_setting_has_a_bound() -> None:
    assert {*_settings_of_type(int), *FLOAT_SETTINGS} == set(BOUNDS)


@pytest.mark.parametrize(
    ("section", "key", "value", "inside"),
    [
        (section, key, value, inside)
        for (section, key), (inside_values, outside_values, _) in BOUNDS.items()
        for inside, values in ((True, inside_values), (False, outside_values))
        for value in values
    ],
    ids=str,
)
def test_each_number_setting_is_bounded(
    tmp_path: Path, section: str, key: str, value: float, inside: bool
) -> None:
    raw = raw_config(tmp_path)
    raw[section][key] = value
    if inside:
        assert _settings(build_config(raw, {}))[section][key] == value
    else:
        bound = BOUNDS[section, key][2]
        assert _refusal(raw) == f"[{section}] {key} must be {bound}, got {value!r}"


@pytest.mark.parametrize(("section", "key"), FLOAT_SETTINGS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "nan", "inf"])
def test_a_number_setting_must_be_finite(
    tmp_path: Path, section: str, key: str, value: float | str
) -> None:
    raw = raw_config(tmp_path)
    if isinstance(value, str):
        env_name = f"MEMEX_{section.upper()}_{key.upper()}"
        where = f"[{section}] {key} from {env_name}"
        message = _refusal(raw, {env_name: value})
    else:
        raw[section][key] = value
        where = f"[{section}] {key}"
        message = _refusal(raw)
    assert message == f"{where} must be a finite number, got {value!r}"


# The URL settings that need https, except to a loopback host.
HTTPS_URLS = [("server", "public_url"), ("auth", "issuer"), ("auth", "jwks_url")]


@pytest.mark.parametrize(("section", "key"), HTTPS_URLS)
@pytest.mark.parametrize(
    "url",
    [
        "https://auth.example.org/application/o/memex/",
        "http://localhost:9000/",
        "http://127.0.0.1:9000/",
        "http://[::1]:9000/",
    ],
)
def test_https_urls_accept_https_and_loopback_http(
    tmp_path: Path, section: str, key: str, url: str
) -> None:
    raw = raw_config(tmp_path)
    raw[section][key] = url
    assert _settings(build_config(raw, {}))[section][key] == url


@pytest.mark.parametrize(("section", "key"), HTTPS_URLS)
@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("auth.example.org/application/o/memex/", "must be an https URL"),
        ("ftp://auth.example.org/", "must be an https URL"),
        ("https:///application/o/memex/", "has no host"),
        ("https://", "has no host"),
        ("http://auth.example.org/", "uses http, needs https (http only for localhost)"),
        ("http://127.0.0.2/", "uses http, needs https (http only for localhost)"),
        ("https://[::1/", "is not a valid URL"),
        ("https://auth.example.org:99999/", "is not a valid URL"),
    ],
)
def test_https_urls_refuse_the_rest(
    tmp_path: Path, section: str, key: str, url: str, problem: str
) -> None:
    raw = raw_config(tmp_path)
    raw[section][key] = url
    assert _refusal(raw) == f"[{section}] {key} {problem}"


@pytest.mark.parametrize("url", ["http://ollama.example.org:11434", "https://ollama.example.org"])
def test_the_embeddings_url_takes_http_or_https(tmp_path: Path, url: str) -> None:
    raw = raw_config(tmp_path)
    raw["embeddings"]["url"] = url
    assert build_config(raw, {}).embeddings.url == url


@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("ollama.example.org:11434", "must be an http or https URL"),
        ("ftp://ollama.example.org", "must be an http or https URL"),
        ("http://", "has no host"),
    ],
)
def test_the_embeddings_url_is_checked_while_enabled(
    tmp_path: Path, url: str, problem: str
) -> None:
    raw = raw_config(tmp_path)
    raw["embeddings"]["url"] = url
    assert _refusal(raw) == f"[embeddings] url {problem}"
    raw["embeddings"]["enabled"] = False
    assert build_config(raw, {}).embeddings.url == url


@pytest.mark.parametrize(
    ("section", "key", "url"),
    [
        ("server", "public_url", "http://user:secret@memex.example.org/"),
        ("auth", "issuer", "ftp://user:secret@auth.example.org/"),
        ("auth", "jwks_url", "https://user:secret@[::1/"),
        ("embeddings", "url", "user:secret@ollama.example.org"),
    ],
)
def test_a_refused_url_is_never_shown(tmp_path: Path, section: str, key: str, url: str) -> None:
    env_name = f"MEMEX_{section.upper()}_{key.upper()}"
    for message in (
        _refusal({**raw_config(tmp_path), section: {**raw_config(tmp_path)[section], key: url}}),
        _refusal(raw_config(tmp_path), {env_name: url}),
    ):
        assert message.startswith(f"[{section}] {key} ")
        for part in ("user", "secret", "example.org", "::1"):
            assert part not in message


def test_a_refused_string_is_never_shown(tmp_path: Path) -> None:
    message = _refusal(raw_config(tmp_path), {"MEMEX_SERVER_MCP_PATH": "secret"})
    assert message == "[server] mcp_path from MEMEX_SERVER_MCP_PATH must start with '/'"
    raw = raw_config(tmp_path)
    raw["server"]["log_level"] = "secret"
    assert "secret" not in _refusal(raw)


def _example_with(tmp_path: Path, old: str, new: str) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_a_refusal_names_the_file_or_the_variable(tmp_path: Path) -> None:
    path = _example_with(tmp_path, "port = 8000\n", "port = 0\n")
    with pytest.raises(ConfigError) as caught:
        load_config(path, env={})
    assert (
        str(caught.value) == f"[server] port in {path} must be at least 1 and at most 65535, got 0"
    )
    with pytest.raises(ConfigError) as caught:
        load_config(path, env={"MEMEX_SERVER_PORT": "65536"})
    assert str(caught.value) == (
        "[server] port from MEMEX_SERVER_PORT must be at least 1 and at most 65535, got '65536'"
    )
    with pytest.raises(ConfigError) as caught:
        load_config(EXAMPLE, env={"MEMEX_AUTH_ISSUER": "http://auth.example.org/"})
    assert str(caught.value) == (
        "[auth] issuer from MEMEX_AUTH_ISSUER uses http, needs https (http only for localhost)"
    )


def test_a_refused_area_names_the_file_but_not_the_area(tmp_path: Path) -> None:
    path = _example_with(tmp_path, 'bob = "bob"\n', 'bob = ".secret"\n')
    with pytest.raises(ConfigError) as caught:
        load_config(path, env={})
    assert str(caught.value) == f"[users] bob in {path} is not a single, visible directory name"
    path = _example_with(tmp_path, 'bob = "bob"\n', 'bob = "household"\n')
    with pytest.raises(ConfigError) as caught:
        load_config(path, env={})
    assert str(caught.value) == f"[users] bob in {path} is the household area"


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        ("\x1b[31m", r"'\x1b[31m'"),
        ("x" * 78, "'" + "x" * 78 + "'"),
        ("x" * 79, "'" + "x" * 76 + "..."),
        ("\n" * 100, "'" + r"\n" * 38 + "..."),
    ],
    ids=["control", "80 shown whole", "81 cut", "escaped then cut"],
)
def test_a_shown_value_is_escaped_and_cut(tmp_path: Path, value: str, shown: str) -> None:
    assert len(shown) <= SHOWN_CHARS
    message = _refusal(raw_config(tmp_path), {"MEMEX_SERVER_PORT": value})
    assert message == f"[server] port from MEMEX_SERVER_PORT must be a whole number, got {shown}"
