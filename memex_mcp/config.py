"""Configuration: one TOML file plus environment overrides.

The file holds everything; an environment variable ``MEMEX_<SECTION>_<KEY>``
overrides a scalar or list value of that section (lists comma-separated), so
secrets and URLs need not live in the file. The identity-to-area mapping
(``[users]``) lives in the file only.
"""

from __future__ import annotations

import logging
import math
import os
import re
import tomllib
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast, override
from urllib.parse import urlsplit

from jwt.algorithms import HMACAlgorithm, NoneAlgorithm, get_default_algorithms
from pydantic import AnyHttpUrl, ValidationError

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

ENV_PREFIX = "MEMEX_"
CONFIG_ENV = "MEMEX_CONFIG"
MIN_SNIPPET_CHARS = 40
# The smallest chunk_chars; at 0 the chunker would never end.
MIN_CHUNK_CHARS = 100
# The longest any seconds setting may be: one day.
MAX_SECONDS = 86400
# The shortest any timeout may be.
MIN_TIMEOUT_SECONDS = 0.1
# The largest chunk_chars, snippet_chars, max_limit, dimensions and batch_size:
# the largest signed 32-bit integer.
MAX_WHOLE = 2**31 - 1
# A refusal shows a given number or boolean value, escaped and cut to this length.
SHOWN_CHARS = 80


class ConfigError(ValueError):
    """The configuration is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"  # noqa: S104 - the container listens on all interfaces
    port: int = 8000
    # The public base URL clients reach the server under; the protected
    # resource metadata advertises "<public_url><mcp_path>" as the resource.
    public_url: str = ""
    mcp_path: str = "/mcp"
    log_level: str = "INFO"


@dataclass(frozen=True)
class AuthConfig:
    # The issuer exactly as Authentik puts it into `iss`, e.g.
    # https://auth.example.org/application/o/memex/ (per-provider issuer mode).
    issuer: str = ""
    # Defaults to "<issuer>jwks/", Authentik's per-application JWKS endpoint.
    jwks_url: str = ""
    # The OAuth client ids whose tokens are accepted (`aud` and `azp`).
    client_ids: tuple[str, ...] = ()
    algorithms: tuple[str, ...] = ("RS256", "ES256")
    # Scopes advertised in the protected resource metadata; `profile` carries
    # the username and the groups, `offline_access` a refresh token.
    scopes: tuple[str, ...] = ("openid", "profile", "offline_access")
    leeway_seconds: int = 30
    # A refetch of the key set for an unknown key id happens at most this long
    # after the last successful fetch, and a failed fetch is not tried again
    # for this long, nor sooner than timeout_seconds.
    jwks_min_refetch_seconds: int = 60
    timeout_seconds: float = 5.0


@dataclass(frozen=True)
class RightsConfig:
    # Membership in this group is required for any access.
    access_group: str = "memex"
    # Members of this group also read the shared area.
    household_group: str = "household"
    household_area: str = "household"


@dataclass(frozen=True)
class RepoConfig:
    # The server's own clone of the memex repository.
    path: str = ""
    # Cloned from here when `path` holds no repository yet; may carry a
    # credential, so set it through MEMEX_REPO_REMOTE rather than the file.
    remote: str = ""
    branch: str = "main"
    fetch_interval_seconds: float = 60.0
    git_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class IndexConfig:
    # The SQLite search index: a cache, rebuilt when missing or outdated.
    path: str = ""
    # Hits under an `archive/` directory have their score multiplied by this.
    archive_factor: float = 0.5
    snippet_chars: int = 300
    chunk_chars: int = 1500
    max_limit: int = 50


@dataclass(frozen=True)
class EmbeddingsConfig:
    enabled: bool = True
    url: str = "http://localhost:11434"
    model: str = "bge-m3"
    dimensions: int = 1024
    # A search waits this long for the query embedding before it falls back
    # to keyword search only.
    query_timeout_seconds: float = 3.0
    index_timeout_seconds: float = 120.0
    # After a failure, searches skip the embedder for this long.
    retry_after_seconds: float = 60.0
    batch_size: int = 32


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    rights: RightsConfig = field(default_factory=RightsConfig)
    repo: RepoConfig = field(default_factory=RepoConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    # username (as Authentik sends it in `preferred_username`) -> area directory
    users: Mapping[str, str] = field(default_factory=dict[str, str])

    @property
    def jwks_url(self) -> str:
        return self.auth.jwks_url or self.auth.issuer.rstrip("/") + "/jwks/"


# The top-level tables a config file may have: one per field of Config.
_SECTIONS = frozenset(f.name for f in fields(Config))


@dataclass(frozen=True)
class _Bound:
    """The range a number setting must lie in."""

    low: float
    above: bool = False  # the value must be greater than `low`, not equal to it
    high: float | None = None

    def holds(self, number: float) -> bool:
        if number < self.low or (self.above and number == self.low):
            return False
        return self.high is None or number <= self.high

    @override
    def __str__(self) -> str:
        low = f"greater than {self.low}" if self.above else f"at least {self.low}"
        return low if self.high is None else f"{low} and at most {self.high}"


# Every number setting has a bound, so that the server refuses to start on a
# value it cannot run with instead of hanging, busy-looping or failing every
# call.
_BOUNDS: Mapping[tuple[str, str], _Bound] = {
    ("server", "port"): _Bound(1, high=65535),
    ("auth", "leeway_seconds"): _Bound(0, high=MAX_SECONDS),
    ("auth", "jwks_min_refetch_seconds"): _Bound(0, high=MAX_SECONDS),
    ("auth", "timeout_seconds"): _Bound(MIN_TIMEOUT_SECONDS, high=MAX_SECONDS),
    ("repo", "fetch_interval_seconds"): _Bound(1, high=MAX_SECONDS),
    ("repo", "git_timeout_seconds"): _Bound(MIN_TIMEOUT_SECONDS, high=MAX_SECONDS),
    ("index", "archive_factor"): _Bound(0, above=True, high=1),
    ("index", "snippet_chars"): _Bound(MIN_SNIPPET_CHARS, high=MAX_WHOLE),
    ("index", "chunk_chars"): _Bound(MIN_CHUNK_CHARS, high=MAX_WHOLE),
    ("index", "max_limit"): _Bound(1, high=MAX_WHOLE),
    ("embeddings", "dimensions"): _Bound(1, high=MAX_WHOLE),
    ("embeddings", "query_timeout_seconds"): _Bound(MIN_TIMEOUT_SECONDS, high=MAX_SECONDS),
    ("embeddings", "index_timeout_seconds"): _Bound(MIN_TIMEOUT_SECONDS, high=MAX_SECONDS),
    ("embeddings", "retry_after_seconds"): _Bound(0, high=MAX_SECONDS),
    ("embeddings", "batch_size"): _Bound(1, high=MAX_WHOLE),
}

# What int() reads as a decimal whole number; it refuses one only past its
# digit limit.
_DECIMAL = re.compile(r"\s*[+-]?[0-9]+(?:_[0-9]+)*\s*")

# The algorithms PyJWT verifies with a public key: every one it supports but
# the shared-secret HMAC ones and "none".
_ASYMMETRIC_ALGORITHMS = frozenset(
    name
    for name, algorithm in get_default_algorithms().items()
    if not isinstance(algorithm, HMACAlgorithm | NoneAlgorithm)
)

# Hosts a URL that needs https may still reach over plain http.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _env_name(section: str, key: str) -> str:
    return f"{ENV_PREFIX}{section.upper()}_{key.upper()}"


def _setting_env_names() -> frozenset[str]:
    defaults = Config()
    return frozenset(
        _env_name(section, f.name)
        for section in _SECTIONS - {"users"}
        for f in fields(cast("DataclassInstance", getattr(defaults, section)))
    )


# The environment variables the configuration reads: one per setting, and the
# one naming the file. Any other MEMEX_ variable is a typo or a stray.
_ENV_NAMES = _setting_env_names() | {CONFIG_ENV}


@dataclass(frozen=True)
class _Origins:
    """Where the settings came from, for refusals that name a setting.

    A refusal names the setting with the environment variable that set it, or
    else with the config file: where it is set, or where a required setting
    nobody set is missing.
    """

    env: Mapping[str, str]
    source: str | None

    def where(self, section: str, key: str) -> str:
        env_name = _env_name(section, key)
        if env_name in self.env:
            return f"[{section}] {key} from {env_name}"
        return self.in_file(f"[{section}] {key}")

    def in_file(self, name: str) -> str:
        return name if self.source is None else f"{name} in {self.source}"


def _shown(value: object) -> str:
    """A given value for a refusal: escaped like repr, at most SHOWN_CHARS long."""
    try:
        text = repr(value)
    except ValueError:  # an int past the digit limit of str(), alone or in a list
        return f"<{type(value).__name__} too long to show>"
    return text if len(text) <= SHOWN_CHARS else text[: SHOWN_CHARS - 3] + "..."


def _key(name: object) -> str:
    """A key name from the file for a refusal, escaped and cut like a value."""
    return _shown(str(name))


def _coerce_bool(where: str, value: object) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{where} must be a boolean, got {_shown(value)}")


def _coerce_int(where: str, value: object) -> int:
    # A float is refused even when whole: int() would cut 8000.9 to 8000.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        with suppress(ValueError):
            return int(value)
        if _DECIMAL.fullmatch(value):
            raise ConfigError(f"{where} is too large, got {_shown(value)}")
    raise ConfigError(f"{where} must be a whole number, got {_shown(value)}")


def _coerce_float(where: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ConfigError(f"{where} must be a number, got {_shown(value)}")
    try:
        number = float(value)
    except ValueError:
        problem = "must be a number"
    except OverflowError:  # float() of an int past the float range
        problem = "is too large"
    else:
        if math.isfinite(number):
            return number
        # float() reads a string of digits past the float range as infinity.
        spelled = not isinstance(value, str) or "inf" in value.lower()
        problem = "must be a finite number" if math.isnan(number) or spelled else "is too large"
    raise ConfigError(f"{where} {problem}, got {_shown(value)}")


def _coerce_strings(where: str, value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, list):
        items = cast("list[object]", value)
        strings = [item for item in items if isinstance(item, str)]
        if len(strings) == len(items):
            return tuple(strings)
    raise ConfigError(f"{where} must be a list of strings")


def _refuse_quoted(where: str, value: object, default: object) -> None:
    """Refuse a quoted number or boolean in the file; only the environment gives strings."""
    if not isinstance(value, str) or not isinstance(default, bool | int | float):
        return
    if isinstance(default, bool):
        kind = "a boolean"
    else:
        kind = "a whole number" if isinstance(default, int) else "a number"
    raise ConfigError(f"{where} must be {kind} without quotes, got {_shown(value)}")


def _coerce(section: str, key: str, where: str, value: object, default: object) -> object:
    if isinstance(default, bool):
        return _coerce_bool(where, value)
    if isinstance(default, int | float):
        # TOML and the environment give strings, numbers, booleans, lists,
        # tables and dates; a string here came from the environment. A
        # whole-number setting takes an integer or a string of one, never a
        # float; any other number setting takes an integer, a float or a string
        # of either. Neither takes a boolean, though Python counts it as an int.
        number = (
            _coerce_int(where, value) if isinstance(default, int) else _coerce_float(where, value)
        )
        bound = _BOUNDS[section, key]
        if not bound.holds(number):
            raise ConfigError(f"{where} must be {bound}, got {_shown(value)}")
        return number
    if isinstance(default, tuple):
        return _coerce_strings(where, value)
    # A string setting's value is never shown: it may be a URL with a credential.
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string")
    return value


def _section[S: DataclassInstance](
    name: str, defaults: S, raw: Mapping[str, object], origins: _Origins
) -> S:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{origins.in_file(f'[{name}]')} must be a table")
    table = cast("dict[str, object]", value)  # a TOML table's keys are strings
    known = {f.name for f in fields(defaults)}
    unknown = set(table) - known
    if unknown:
        where = origins.in_file(f"[{name}]")
        raise ConfigError(f"{where} has unknown keys: {', '.join(map(_key, sorted(unknown)))}")
    values: dict[str, object] = {}
    for f in fields(defaults):
        default = cast("object", getattr(defaults, f.name))
        env_name = _env_name(name, f.name)
        where = origins.where(name, f.name)
        if env_name in origins.env:
            values[f.name] = _coerce(name, f.name, where, origins.env[env_name], default)
        elif f.name in table:
            _refuse_quoted(where, table[f.name], default)
            values[f.name] = _coerce(name, f.name, where, table[f.name], default)
    return replace(defaults, **values)


def _check_area_name(name: str, where: str) -> None:
    if not name or "/" in name or "\\" in name or name.startswith(".") or name in {"..", "."}:
        raise ConfigError(f"{where} is not a single, visible directory name")


def _users(raw: Mapping[str, object], origins: _Origins) -> dict[str, str]:
    users_raw = raw.get("users", {})
    if not isinstance(users_raw, dict):
        where = origins.in_file("[users]")
        raise ConfigError(f"{where} must map usernames to area directories")
    users: dict[str, str] = {}
    for username, area in cast("dict[object, object]", users_raw).items():
        where = origins.in_file(f"[users] {_key(username)}")
        if not isinstance(area, str):
            raise ConfigError(f"{where} must be a string")
        _check_area_name(area, where)
        users[str(username)] = area
    return users


def _check_env_names(env: Mapping[str, str]) -> None:
    """Refuse a MEMEX_ variable no setting reads, naming it but never its value."""
    unknown = sorted(name for name in env if name.startswith(ENV_PREFIX) and name not in _ENV_NAMES)
    users = [name for name in unknown if name.startswith(_env_name("users", ""))]
    if users:
        raise ConfigError(f"[users] lives in the file only, not in {', '.join(map(_key, users))}")
    if unknown:
        raise ConfigError(f"unknown environment variables: {', '.join(map(_key, unknown))}")


def build_config(
    raw: Mapping[str, object], env: Mapping[str, str], source: str | Path | None = None
) -> Config:
    """Build and validate a Config from parsed TOML and an environment.

    ``source`` names the file ``raw`` was read from, for the refusals.
    """
    origins = _Origins(env, None if source is None else str(source))
    _check_env_names(env)
    unknown = set(raw) - _SECTIONS
    if unknown:
        where = origins.in_file("unknown sections")
        raise ConfigError(f"{where}: {', '.join(map(_key, sorted(unknown)))}")
    config = Config(
        server=_section("server", ServerConfig(), raw, origins),
        auth=_section("auth", AuthConfig(), raw, origins),
        rights=_section("rights", RightsConfig(), raw, origins),
        repo=_section("repo", RepoConfig(), raw, origins),
        index=_section("index", IndexConfig(), raw, origins),
        embeddings=_section("embeddings", EmbeddingsConfig(), raw, origins),
        users=_users(raw, origins),
    )
    _validate(config, origins)
    return config


def _parses_as_http_url(url: str) -> bool:
    """Whether pydantic, which the server hands the URLs to, accepts ``url``."""
    try:
        AnyHttpUrl(url)
    except ValidationError:
        return False
    return True


def _scheme_and_host(url: str) -> tuple[str, str] | None:
    """The scheme and host of ``url``, or None when urlsplit cannot read it."""
    try:
        parts = urlsplit(url)
        _ = parts.port  # raises for a port that is not a number in range
    except ValueError:
        return None
    return parts.scheme, parts.hostname or ""


def _url_problem(url: str, *, https: bool) -> str | None:
    """What is wrong with ``url``, never quoting it; None when nothing is."""
    # urlsplit drops these before it parses; the server uses the URL as given.
    if any(char.isspace() or not char.isprintable() for char in url):
        return "contains whitespace or control characters"
    split = _scheme_and_host(url)
    if split is None:
        return "is not a valid URL"
    scheme, host = split
    if scheme not in {"http", "https"}:
        return "must be an https URL" if https else "must be an http or https URL"
    if not host:
        return "has no host"
    if https and scheme == "http" and host not in _LOOPBACK_HOSTS:
        return "uses http, needs https (http only for localhost)"
    return None if _parses_as_http_url(url) else "is not a valid URL"


def _check_urls(config: Config, origins: _Origins) -> None:
    # (section, key, value, whether it needs https); an empty value is not
    # checked here (the required ones are refused as missing), and a derived
    # jwks_url follows the issuer.
    urls = [
        ("server", "public_url", config.server.public_url, True),
        ("auth", "issuer", config.auth.issuer, True),
        ("auth", "jwks_url", config.auth.jwks_url, True),
    ]
    if config.embeddings.enabled:
        urls.append(("embeddings", "url", config.embeddings.url, False))
    for section, key, url, https in urls:
        problem = _url_problem(url, https=https) if url else None
        if problem:
            raise ConfigError(f"{origins.where(section, key)} {problem}")


def _check_required(config: Config, origins: _Origins) -> None:
    required: list[tuple[str, str, object]] = [
        ("server", "public_url", config.server.public_url),
        ("auth", "issuer", config.auth.issuer),
        ("auth", "client_ids", config.auth.client_ids),
        ("auth", "algorithms", config.auth.algorithms),
        ("rights", "access_group", config.rights.access_group),
        ("rights", "household_group", config.rights.household_group),
        ("repo", "path", config.repo.path),
        ("repo", "branch", config.repo.branch),
        ("index", "path", config.index.path),
    ]
    if config.embeddings.enabled:
        required += [
            ("embeddings", "url", config.embeddings.url),
            ("embeddings", "model", config.embeddings.model),
        ]
    missing = [origins.where(section, key) for section, key, value in required if not value]
    if missing:
        raise ConfigError(f"missing required settings: {', '.join(missing)}")


def _check_algorithms(config: Config, origins: _Origins) -> None:
    for algorithm in config.auth.algorithms:
        if algorithm not in _ASYMMETRIC_ALGORITHMS:
            where = origins.where("auth", "algorithms")
            allowed = ", ".join(sorted(_ASYMMETRIC_ALGORITHMS))
            raise ConfigError(f"{where} has {_shown(algorithm)}, not one of {allowed}")


def _validate(config: Config, origins: _Origins) -> None:
    _check_required(config, origins)
    _check_algorithms(config, origins)
    # git would read such a branch as an option, not a ref.
    if config.repo.branch.startswith("-"):
        raise ConfigError(f"{origins.where('repo', 'branch')} must not start with '-'")
    _check_urls(config, origins)
    if not config.server.mcp_path.startswith("/"):
        raise ConfigError(f"{origins.where('server', 'mcp_path')} must start with '/'")
    # The server hands the upper-cased name to logging.basicConfig.
    if config.server.log_level.upper() not in logging.getLevelNamesMapping():
        where = origins.where("server", "log_level")
        raise ConfigError(f"{where} must be a logging level such as INFO or DEBUG")
    household = config.rights.household_area
    _check_area_name(household, origins.where("rights", "household_area"))
    household_env = _env_name("rights", "household_area")
    set_by = f" set by {household_env}" if household_env in origins.env else ""
    for username, area in config.users.items():
        # A personal area named like the shared one would hand the shared area
        # to that user without the household group.
        if area == household:
            where = origins.in_file(f"[users] {_key(username)}")
            raise ConfigError(f"{where} is the household area{set_by}")


def load_config(path: str | Path | None = None, env: Mapping[str, str] | None = None) -> Config:
    """Load the config file named by ``path`` or ``MEMEX_CONFIG``."""
    env = os.environ if env is None else env
    chosen = path if path is not None else env.get(CONFIG_ENV)
    if not chosen:
        raise ConfigError(f"no config file given (set {CONFIG_ENV} or pass --config)")
    try:
        with Path(chosen).open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config {chosen}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config {chosen} is not valid TOML: {exc}") from None
    # tomllib decodes the bytes itself and lets these through; none of the
    # messages quotes the file's content.
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"config {chosen} is not UTF-8: invalid byte at offset {exc.start}"
        ) from None
    except ValueError:  # an integer past the digit limit of int()
        raise ConfigError(f"config {chosen} is not valid TOML: a value is out of range") from None
    except RecursionError:
        raise ConfigError(f"config {chosen} is not valid TOML: nested too deeply") from None
    return build_config(raw, env, chosen)
