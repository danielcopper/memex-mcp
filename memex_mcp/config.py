"""Configuration: one TOML file plus environment overrides.

The file holds everything; an environment variable ``MEMEX_<SECTION>_<KEY>``
overrides a scalar or list value of that section (lists comma-separated), so
secrets and URLs need not live in the file. The identity-to-area mapping
(``[users]``) lives in the file only.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "MEMEX_"
CONFIG_ENV = "MEMEX_CONFIG"


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
    # A refetch of the key set for an unknown key id happens at most this often.
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
    users: Mapping[str, str] = field(default_factory=dict)

    @property
    def jwks_url(self) -> str:
        return self.auth.jwks_url or self.auth.issuer.rstrip("/") + "/jwks/"


_SECTIONS: dict[str, type] = {
    "server": ServerConfig,
    "auth": AuthConfig,
    "rights": RightsConfig,
    "repo": RepoConfig,
    "index": IndexConfig,
    "embeddings": EmbeddingsConfig,
}


def _coerce(section: str, key: str, value: Any, default: Any) -> Any:
    where = f"[{section}] {key}"
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where} must be a boolean")
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where} must be an integer") from None
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where} must be a number") from None
    if isinstance(default, tuple):
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return tuple(value)
        raise ConfigError(f"{where} must be a list of strings")
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string")
    return value


def _section(name: str, raw: Mapping[str, Any], env: Mapping[str, str]) -> Any:
    cls = _SECTIONS[name]
    defaults = cls()
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"[{name}] has unknown keys: {', '.join(sorted(unknown))}")
    values: dict[str, Any] = {}
    for f in fields(cls):
        default = getattr(defaults, f.name)
        env_name = f"{ENV_PREFIX}{name.upper()}_{f.name.upper()}"
        if env_name in env:
            values[f.name] = _coerce(name, f.name, env[env_name], default)
        elif f.name in raw:
            values[f.name] = _coerce(name, f.name, raw[f.name], default)
    return cls(**values)


def _check_area_name(name: str, where: str) -> None:
    if not name or "/" in name or "\\" in name or name.startswith(".") or name in {"..", "."}:
        raise ConfigError(f"{where}: {name!r} is not a single, visible directory name")


def build_config(raw: Mapping[str, Any], env: Mapping[str, str]) -> Config:
    """Build and validate a Config from parsed TOML and an environment."""
    unknown = set(raw) - set(_SECTIONS) - {"users"}
    if unknown:
        raise ConfigError(f"unknown sections: {', '.join(sorted(unknown))}")
    sections = {name: _section(name, raw.get(name, {}), env) for name in _SECTIONS}
    users_raw = raw.get("users", {})
    if not isinstance(users_raw, dict):
        raise ConfigError("[users] must map usernames to area directories")
    users: dict[str, str] = {}
    for username, area in users_raw.items():
        if not isinstance(area, str):
            raise ConfigError(f"[users] {username}: the area must be a string")
        _check_area_name(area, f"[users] {username}")
        users[str(username)] = area
    config = Config(**sections, users=users)
    _validate(config)
    return config


def _validate(config: Config) -> None:
    missing = [
        name
        for name, value in (
            ("[server] public_url", config.server.public_url),
            ("[auth] issuer", config.auth.issuer),
            ("[repo] path", config.repo.path),
            ("[index] path", config.index.path),
        )
        if not value
    ]
    if not config.auth.client_ids:
        missing.append("[auth] client_ids")
    if missing:
        raise ConfigError(f"missing required settings: {', '.join(missing)}")
    if not config.server.mcp_path.startswith("/"):
        raise ConfigError("[server] mcp_path must start with '/'")
    household = config.rights.household_area
    _check_area_name(household, "[rights] household_area")
    for username, area in config.users.items():
        # A personal area named like the shared one would hand the shared area
        # to that user without the household group.
        if area == household:
            raise ConfigError(f"[users] {username}: {area!r} is the household area")
    if not 0 < config.index.archive_factor <= 1:
        raise ConfigError("[index] archive_factor must be in (0, 1]")
    if config.index.snippet_chars < 40:
        raise ConfigError("[index] snippet_chars must be at least 40")


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
    return build_config(raw, env)
