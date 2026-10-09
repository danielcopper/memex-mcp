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
SECONDS = "at most 86400"
WHOLE = "at most 2147483647"
BOUNDS: dict[tuple[str, str], tuple[list[float], list[float], str]] = {
    ("server", "port"): ([1, 65535], [0, 65536], "at least 1 and at most 65535"),
    ("auth", "leeway_seconds"): ([0, 86400], [-1, 86401], f"at least 0 and {SECONDS}"),
    ("auth", "jwks_min_refetch_seconds"): ([0, 86400], [-1, 86401], f"at least 0 and {SECONDS}"),
    ("auth", "timeout_seconds"): ([0.1, 86400], [0.099, 86400.001], f"at least 0.1 and {SECONDS}"),
    ("repo", "fetch_interval_seconds"): (
        [1, 86400],
        [0.999, 86400.001],
        f"at least 1 and {SECONDS}",
    ),
    ("repo", "git_timeout_seconds"): (
        [0.1, 86400],
        [0.099, 86400.001],
        f"at least 0.1 and {SECONDS}",
    ),
    ("index", "archive_factor"): ([0.001, 1], [0, 1.001], "greater than 0 and at most 1"),
    ("index", "snippet_chars"): ([40, 2**31 - 1], [39, 2**31], f"at least 40 and {WHOLE}"),
    ("index", "chunk_chars"): ([100, 2**31 - 1], [99, 2**31], f"at least 100 and {WHOLE}"),
    ("index", "max_limit"): ([1, 2**31 - 1], [0, 2**31], f"at least 1 and {WHOLE}"),
    ("embeddings", "dimensions"): ([1, 2**31 - 1], [0, 2**31], f"at least 1 and {WHOLE}"),
    ("embeddings", "query_timeout_seconds"): (
        [0.1, 86400],
        [0.099, 86400.001],
        f"at least 0.1 and {SECONDS}",
    ),
    ("embeddings", "index_timeout_seconds"): (
        [0.1, 86400],
        [0.099, 86400.001],
        f"at least 0.1 and {SECONDS}",
    ),
    ("embeddings", "retry_after_seconds"): (
        [0, 86400],
        [-0.001, 86400.001],
        f"at least 0 and {SECONDS}",
    ),
    ("embeddings", "batch_size"): ([1, 2**31 - 1], [0, 2**31], f"at least 1 and {WHOLE}"),
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
        "https://auth.example.org/",
        "https://auth.example.org:8443",
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
    assert message == "[server] mcp_path from MEMEX_SERVER_MCP_PATH must be a path such as /mcp"
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
    assert str(caught.value) == f"[users] 'bob' in {path} is not a single, visible directory name"
    path = _example_with(tmp_path, 'bob = "bob"\n', 'bob = "household"\n')
    with pytest.raises(ConfigError) as caught:
        load_config(path, env={})
    assert str(caught.value) == f"[users] 'bob' in {path} is the household area"


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


URL_SETTINGS = [*HTTPS_URLS, ("embeddings", "url")]


def _url_refusal(tmp_path: Path, section: str, key: str, url: str, *, env: bool) -> ConfigError:
    raw = raw_config(tmp_path)
    environment: dict[str, str] = {}
    if env:
        environment[f"MEMEX_{section.upper()}_{key.upper()}"] = url
    else:
        raw[section][key] = url
    with pytest.raises(ConfigError) as caught:
        build_config(raw, environment)
    return caught.value


@pytest.mark.parametrize(("section", "key"), URL_SETTINGS)
@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("https://a<b.example.org/", "is not a valid URL"),
        ("https://xn--zz.example.org/", "is not a valid URL"),
        ("https://u:s3cret@auth exa.org/", "contains whitespace or control characters"),
        ("https://a b/", "contains whitespace or control characters"),
    ],
)
def test_a_url_the_server_cannot_parse_is_refused(
    tmp_path: Path, section: str, key: str, url: str, problem: str
) -> None:
    for env in (False, True):
        refusal = _url_refusal(tmp_path, section, key, url, env=env)
        assert str(refusal).startswith(f"[{section}] {key} ")
        assert str(refusal).endswith(f" {problem}")
        assert "s3cret" not in str(refusal)
        assert refusal.__context__ is None
        assert refusal.__cause__ is None


@pytest.mark.parametrize(("section", "key"), URL_SETTINGS)
@pytest.mark.parametrize(
    "url",
    [
        "https://auth.example.org/application/o/memex/\n",
        " https://auth.example.org/",
        "https://auth.exa\tmple.org/",
        "\x01https://auth.example.org/",
    ],
    ids=["trailing newline", "leading space", "tab inside", "control prefix"],
)
def test_a_url_with_whitespace_or_control_characters_is_refused(
    tmp_path: Path, section: str, key: str, url: str
) -> None:
    for env in (False, True):
        message = str(_url_refusal(tmp_path, section, key, url, env=env))
        assert message.endswith(" contains whitespace or control characters")
        assert "example" not in message


HUGE = int("f" * 4000, 16)  # past the digit limit of str(); TOML reads it as 0xfff...


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (
            "server",
            "port",
            HUGE,
            "must be at least 1 and at most 65535, got <int too long to show>",
        ),
        ("server", "port", [HUGE], "must be a whole number, got <list too long to show>"),
        ("auth", "timeout_seconds", HUGE, "is too large, got <int too long to show>"),
        ("embeddings", "enabled", HUGE, "must be a boolean, got <int too long to show>"),
    ],
    ids=["int setting", "in a list", "float setting", "bool setting"],
)
def test_a_value_too_long_to_show_is_described(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    raw = raw_config(tmp_path)
    raw[section][key] = value
    assert _refusal(raw) == f"[{section}] {key} {message}"


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("index", "chunk_chars", "9" * 5000, "is too large"),
        ("index", "chunk_chars", " +9_" + "9" * 5000, "is too large"),
        ("index", "chunk_chars", "1__0", "must be a whole number"),
        ("auth", "timeout_seconds", "9" * 400, "is too large"),
        ("auth", "timeout_seconds", "-infinity", "must be a finite number"),
    ],
    ids=["digits", "sign and underscore", "not a number", "float setting", "infinity"],
)
def test_a_number_past_what_python_reads_is_too_large(
    tmp_path: Path, section: str, key: str, value: str, message: str
) -> None:
    env_name = f"MEMEX_{section.upper()}_{key.upper()}"
    shown = repr(value) if len(repr(value)) <= SHOWN_CHARS else repr(value)[:77] + "..."
    assert _refusal(raw_config(tmp_path), {env_name: value}) == (
        f"[{section}] {key} from {env_name} {message}, got {shown}"
    )


def test_a_missing_setting_names_its_origin(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_config(EXAMPLE, env={"MEMEX_SERVER_PUBLIC_URL": "", "MEMEX_AUTH_CLIENT_IDS": ","})
    assert str(caught.value) == (
        "missing required settings: [server] public_url from MEMEX_SERVER_PUBLIC_URL, "
        "[auth] client_ids from MEMEX_AUTH_CLIENT_IDS"
    )
    path = _example_with(tmp_path, 'path = "/data/clone"\n', "")
    with pytest.raises(ConfigError) as caught:
        load_config(path, env={})
    assert str(caught.value) == f"missing required settings: [repo] path in {path}"


def test_key_names_are_escaped_and_cut(tmp_path: Path) -> None:
    key = "x\x1b[2J" * 30
    shown = repr(key)[: SHOWN_CHARS - 3] + "..."
    raw = raw_config(tmp_path)
    raw["server"][key] = 1
    assert _refusal(raw) == f"[server] has unknown keys: {shown}"
    raw = raw_config(tmp_path)
    raw[key] = {}
    assert _refusal(raw) == f"unknown sections: {shown}"
    raw = raw_config(tmp_path)
    raw["users"] = {"al\x1b[31mice": "household"}
    assert _refusal(raw) == r"[users] 'al\x1b[31mice' is the household area"


def test_a_household_clash_names_the_variable_that_set_the_area() -> None:
    with pytest.raises(ConfigError) as caught:
        load_config(EXAMPLE, env={"MEMEX_RIGHTS_HOUSEHOLD_AREA": "alice"})
    assert str(caught.value) == (
        f"[users] 'alice' in {EXAMPLE} is the household area set by MEMEX_RIGHTS_HOUSEHOLD_AREA"
    )


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("# M\xfcller s3cret\n".encode("latin-1"), "is not UTF-8: invalid byte at offset 3"),
        (f"port = 1{'9' * 5000}\n".encode(), "is not valid TOML: a value is out of range"),
        (f"a = {'[' * 5000}{']' * 5000}\n".encode(), "is not valid TOML: nested too deeply"),
    ],
    ids=["not utf-8", "digit limit", "deep nesting"],
)
def test_a_file_tomllib_cannot_decode_is_refused(
    tmp_path: Path, content: bytes, problem: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(content)
    with pytest.raises(ConfigError) as caught:
        load_config(path, env={})
    assert str(caught.value) == f"config {path} {problem}"


@pytest.mark.parametrize(
    ("setting", "value", "kind", "read"),
    [
        (("server", "port"), "8000", "a whole number", 8000),
        (("auth", "timeout_seconds"), "5", "a number", 5.0),
        (("embeddings", "enabled"), "false", "a boolean", False),
    ],
)
def test_a_quoted_number_or_boolean_comes_only_from_the_environment(
    tmp_path: Path, setting: tuple[str, str], value: str, kind: str, read: object
) -> None:
    section, key = setting
    raw = raw_config(tmp_path)
    raw[section][key] = value
    assert _refusal(raw) == f"[{section}] {key} must be {kind} without quotes, got {value!r}"
    env_name = f"MEMEX_{section.upper()}_{key.upper()}"
    assert _settings(build_config(raw_config(tmp_path), {env_name: value}))[section][key] == read


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("rights", "access_group"),
        ("rights", "household_group"),
        ("repo", "branch"),
        ("embeddings", "url"),
        ("embeddings", "model"),
    ],
)
def test_an_empty_setting_the_server_needs_is_missing(
    tmp_path: Path, section: str, key: str
) -> None:
    env_name = f"MEMEX_{section.upper()}_{key.upper()}"
    message = _refusal(raw_config(tmp_path), {env_name: ""})
    assert message == f"missing required settings: [{section}] {key} from {env_name}"
    raw = raw_config(tmp_path)
    raw.setdefault(section, {})[key] = ""
    assert _refusal(raw) == f"missing required settings: [{section}] {key}"


@pytest.mark.parametrize("key", ["url", "model"])
def test_the_embedder_settings_may_be_empty_while_disabled(tmp_path: Path, key: str) -> None:
    raw = raw_config(tmp_path)
    raw["embeddings"].update({key: "", "enabled": False})
    assert _settings(build_config(raw, {}))["embeddings"][key] == ""


@pytest.mark.parametrize("branch", ["--upload-pack=s3cret", "-x"])
def test_a_branch_that_git_would_read_as_an_option_is_refused(tmp_path: Path, branch: str) -> None:
    raw = raw_config(tmp_path)
    raw["repo"]["branch"] = branch
    assert _refusal(raw) == "[repo] branch must not start with '-'"
    assert _refusal(raw_config(tmp_path), {"MEMEX_REPO_BRANCH": branch}) == (
        "[repo] branch from MEMEX_REPO_BRANCH must not start with '-'"
    )


def test_an_environment_variable_no_setting_reads_is_refused(tmp_path: Path) -> None:
    env = {"MEMEX_SERVER_PROT": "s3cret", "MEMEX_EMBEDDINGS_ENABLE": "false"}
    message = _refusal(raw_config(tmp_path), env)
    assert message == (
        "unknown environment variables: 'MEMEX_EMBEDDINGS_ENABLE', 'MEMEX_SERVER_PROT'"
    )
    assert "s3cret" not in message
    message = _refusal(raw_config(tmp_path), {"MEMEX_USERS_ALICE": "alice"})
    assert message == "[users] lives in the file only, not in 'MEMEX_USERS_ALICE'"
    message = _refusal(raw_config(tmp_path), {"MEMEX_\x1b[2J": ""})
    assert message == r"unknown environment variables: 'MEMEX_\x1b[2J'"


def test_the_environment_may_name_the_file_and_carry_other_variables(tmp_path: Path) -> None:
    env = {"MEMEX_CONFIG": str(EXAMPLE), "MEMEXX_PORT": "1", "PATH": "/usr/bin", "memex_x": "1"}
    assert build_config(raw_config(tmp_path), env).server.port == 8000


def _as_env_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, tuple):
        return ",".join(cast("tuple[str, ...]", value))
    return str(value)


def test_every_setting_reads_its_environment_variable(tmp_path: Path) -> None:
    built = _settings(build_config(raw_config(tmp_path), {}))
    for section, values in built.items():
        if section == "users":
            continue
        for key, value in values.items():
            env = {f"MEMEX_{section.upper()}_{key.upper()}": _as_env_text(value)}
            assert _settings(build_config(raw_config(tmp_path), env)) == built


ALLOWED_ALGORITHMS = (
    "ES256, ES256K, ES384, ES512, ES521, EdDSA, PS256, PS384, PS512, RS256, RS384, RS512"
)


@pytest.mark.parametrize("algorithms", [["RS256"], ["ES256", "PS512", "EdDSA"]])
def test_asymmetric_algorithms_are_accepted(tmp_path: Path, algorithms: list[str]) -> None:
    raw = raw_config(tmp_path)
    raw["auth"]["algorithms"] = algorithms
    assert build_config(raw, {}).auth.algorithms == tuple(algorithms)


@pytest.mark.parametrize("algorithm", ["HS256", "none", "RS257", "rs256"])
def test_an_algorithm_that_is_not_asymmetric_is_refused(tmp_path: Path, algorithm: str) -> None:
    raw = raw_config(tmp_path)
    raw["auth"]["algorithms"] = ["RS256", algorithm]
    assert _refusal(raw) == (
        f"[auth] algorithms has {algorithm!r}, not one of {ALLOWED_ALGORITHMS}"
    )
    message = _refusal(raw_config(tmp_path), {"MEMEX_AUTH_ALGORITHMS": f"RS256,{algorithm}"})
    assert message.startswith(f"[auth] algorithms from MEMEX_AUTH_ALGORITHMS has {algorithm!r}")


@pytest.mark.parametrize("value", ["", ","])
def test_no_algorithm_is_missing(tmp_path: Path, value: str) -> None:
    assert _refusal(raw_config(tmp_path), {"MEMEX_AUTH_ALGORITHMS": value}) == (
        "missing required settings: [auth] algorithms from MEMEX_AUTH_ALGORITHMS"
    )
    raw = raw_config(tmp_path)
    raw["auth"]["algorithms"] = []
    assert _refusal(raw) == "missing required settings: [auth] algorithms"


@pytest.mark.parametrize("url", ["https://u:s3cret@a<b.example.org/", "http://u:s3cret@xn--zz"])
def test_a_credential_in_an_embeddings_url_the_server_cannot_parse_is_not_shown(
    tmp_path: Path, url: str
) -> None:
    for env in (False, True):
        refusal = _url_refusal(tmp_path, "embeddings", "url", url, env=env)
        assert str(refusal).endswith(" is not a valid URL")
        assert "s3cret" not in str(refusal)
        assert refusal.__context__ is None


def test_issuer_and_jwks_url_may_have_a_path(tmp_path: Path) -> None:
    raw = raw_config(tmp_path)
    raw["server"]["public_url"] = "https://memex.example.org"
    raw["auth"]["issuer"] = "https://auth.example.org/application/o/memex/"
    raw["auth"]["jwks_url"] = "https://auth.example.org/application/o/memex/jwks/?x=1#k"
    config = build_config(raw, {})
    assert config.server.public_url == "https://memex.example.org"
    assert config.auth.issuer == "https://auth.example.org/application/o/memex/"
    assert config.jwks_url == "https://auth.example.org/application/o/memex/jwks/?x=1#k"


@pytest.mark.parametrize(
    "url",
    [
        "https://memex.example.org/sub",
        "https://memex.example.org/mcp/",
        "https://memex.example.org?x",
        "https://memex.example.org/?",
        "https://memex.example.org#f",
        "https://memex.example.org/#",
    ],
)
def test_the_public_url_has_no_path_query_or_fragment(tmp_path: Path, url: str) -> None:
    for env in (False, True):
        message = str(_url_refusal(tmp_path, "server", "public_url", url, env=env))
        assert message.endswith(
            " must not have a path, query or fragment; the server serves at mcp_path"
        )
        assert "example" not in message


@pytest.mark.parametrize("section, key", HTTPS_URLS)
@pytest.mark.parametrize(
    "url",
    [
        "https://u:s3cret@auth.example.org/",
        "https://u@auth.example.org/",
        "https://:s3cret@auth.example.org/",
        "https://@auth.example.org/",
        "http://u:s3cret@localhost:9000/",
    ],
)
def test_an_https_url_carries_no_user_or_password(
    tmp_path: Path, section: str, key: str, url: str
) -> None:
    for env in (False, True):
        message = str(_url_refusal(tmp_path, section, key, url, env=env))
        assert message.endswith(" must not carry a user or password")
        assert "s3cret" not in message
        assert "example" not in message


@pytest.mark.parametrize("path", ["/mcp", "/mcp/", "/a/b-c_d.e~f", "/...", "/.x"])
def test_a_simple_mcp_path_is_accepted(tmp_path: Path, path: str) -> None:
    raw = raw_config(tmp_path)
    raw["server"]["mcp_path"] = path
    assert build_config(raw, {}).server.mcp_path == path


@pytest.mark.parametrize(
    "path",
    [
        "mcp",
        "/",
        "//mcp",
        "/mcp?x=1",
        "/mcp#f",
        "/m c p",
        "/{x}",
        "/mcp/{x}",
        "/mcp\x00",
        "/mcp\n",
        "/..",
        "/.",
        "/mcp/..",
        "/./mcp",
    ],
)
def test_an_mcp_path_the_server_cannot_mount_is_refused(tmp_path: Path, path: str) -> None:
    raw = raw_config(tmp_path)
    raw["server"]["mcp_path"] = path
    assert _refusal(raw) == "[server] mcp_path must be a path such as /mcp"


@pytest.mark.parametrize(
    ("section", "key"), [("repo", "branch"), ("repo", "path"), ("index", "path")]
)
@pytest.mark.parametrize("value", ["main\n", " main", "ma in", "ma\x00in", "ma\tin"])
def test_a_name_git_or_the_file_system_takes_as_given_is_refused(
    tmp_path: Path, section: str, key: str, value: str
) -> None:
    raw = raw_config(tmp_path)
    raw[section][key] = value
    assert _refusal(raw) == f"[{section}] {key} contains whitespace or control characters"
    env_name = f"MEMEX_{section.upper()}_{key.upper()}"
    assert _refusal(raw_config(tmp_path), {env_name: value}) == (
        f"[{section}] {key} from {env_name} contains whitespace or control characters"
    )


@pytest.mark.parametrize("area", ["household ", "house\nhold", "house\x00hold", "\u200bhousehold"])
def test_an_area_name_with_whitespace_or_control_characters_is_refused(
    tmp_path: Path, area: str
) -> None:
    raw = raw_config(tmp_path)
    raw["rights"] = {"household_area": area}
    assert _refusal(raw) == "[rights] household_area contains whitespace or control characters"
    raw = raw_config(tmp_path)
    raw["users"] = {"eve": area}
    assert _refusal(raw) == "[users] 'eve' contains whitespace or control characters"


@pytest.mark.parametrize("value", ["-1e999", "-" + "9" * 400])
def test_a_string_that_reads_as_negative_infinity_is_not_finite(tmp_path: Path, value: str) -> None:
    message = _refusal(raw_config(tmp_path), {"MEMEX_AUTH_TIMEOUT_SECONDS": value})
    assert message.startswith(
        "[auth] timeout_seconds from MEMEX_AUTH_TIMEOUT_SECONDS must be a finite number, got "
    )


def test_a_file_with_a_byte_order_mark_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xef\xbb\xbf" + EXAMPLE.read_bytes())
    with pytest.raises(ConfigError) as caught:
        load_config(path, env={})
    assert str(caught.value) == f"config {path} starts with a byte-order mark; save it without one"
