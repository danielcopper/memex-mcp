"""Access-token validation against Authentik, done locally.

Authentik's OAuth2 provider issues its access tokens as signed JWTs: the
token carries the ID-token claims plus ``azp``, ``uid`` and ``scope``, signed
with the provider's signing key, whose public half Authentik publishes at the
application's JWKS endpoint. So the server validates each token itself —
signature against the JWKS, ``iss``, ``exp``, ``aud`` and ``azp`` against the
configured client ids — and reads the username (``preferred_username``) and
groups (``groups``) from it; no introspection call per request. A token must
name its signing key (``kid``); the key set is cached for
``JWKS_LIFESPAN_SECONDS``.

A token without a well-formed ``preferred_username`` or ``groups`` claim is
rejected: the provider is then missing the ``profile`` scope mapping, and
without groups no rights decision is possible.
"""

from __future__ import annotations

import logging
import time
from typing import cast, override

import anyio
import jwt
from fastmcp.server.auth import AccessToken, TokenVerifier

from memex_mcp.json_kind import json_kind
from memex_mcp.rights import Identity

log = logging.getLogger(__name__)

USERNAME_CLAIM = "preferred_username"
GROUPS_CLAIM = "groups"
REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "sub"]


# The key set is cached this long, then fetched again: a key the provider has
# revoked stops verifying at most this long after it left the key set.
JWKS_LIFESPAN_SECONDS = 300

# What a fetch was seen to raise besides PyJWT's own errors at PyJWT 2.15.1
# (tests/test_auth.py tries every member of each key type): OSError for a
# connection reset while the answer is read, ValueError for an answer that is
# not JSON, RecursionError for one nested too deeply, and, out of a key entry,
# TypeError for an `alg` that is not a string, NotImplementedError for `alg`
# "none" and KeyError for an `oct` key without `k`. Each fails the whole set.
_FETCH_ERRORS = (
    jwt.PyJWTError,
    OSError,
    ValueError,
    RecursionError,
    TypeError,
    NotImplementedError,
    KeyError,
)

_HINT = "; is [auth] jwks_url the provider's JWKS endpoint?"

# Log lines carry at most this much of a kid.
_KID_CHARS = 64


def _clip(text: str) -> str:
    return text if len(text) <= _KID_CHARS else text[:_KID_CHARS] + "…"


def _member(value: object) -> str:
    """A key set member for a log line: a clipped string, or the kind of anything else."""
    return repr(_clip(value)) if isinstance(value, str) else json_kind(value)


def _unused(entry: object) -> tuple[int, str] | None:
    """Why PyJWKClient will not use a key set entry, with the log level; None for a signing key."""
    if not isinstance(entry, dict):
        return logging.WARNING, f"is {json_kind(entry)}, not an object"
    data = cast("dict[str, object]", entry)  # a JSON object's keys are strings
    about = ", ".join(f"{name} {_member(data.get(name))}" for name in ("kid", "kty", "alg"))
    use = data.get("use")
    if use == "enc":
        # Authentik lists its encryption key here when one is set.
        return logging.DEBUG, f"({about}) is an encryption key"
    try:
        key = jwt.PyJWK(data)
    except jwt.PyJWTError as exc:
        # Only the type: PyJWT's messages may quote the whole key.
        return logging.WARNING, f"({about}) is unusable: {type(exc).__name__}"
    # What PyJWKClient keeps as signing keys: `use` "sig" or absent, and a kid.
    if use not in ("sig", None):
        return logging.WARNING, f"({about}) has use {_member(use)}, not 'sig'"
    if not key.key_id:
        return logging.WARNING, f"({about}) has no kid"
    return None


class _KeySetUnavailable(jwt.PyJWKClientError):
    """The key set could not be fetched; the fetch that failed has logged why."""


class _ProviderJwks(jwt.PyJWKClient):
    """PyJWKClient that waits a cooldown after a failed fetch and logs what it will not use.

    PyJWT still fetches, parses and caches the set; this class only adds to
    ``fetch_data``, which PyJWT calls under the client's lock.
    """

    def __init__(self, uri: str, *, timeout: float, cooldown: float) -> None:
        # A token with a kid the cached set lacks refetches the set, but not
        # sooner than the cooldown after the last successful fetch.
        super().__init__(
            uri,
            cache_jwk_set=True,
            lifespan=JWKS_LIFESPAN_SECONDS,
            timeout=timeout,
            cooldown_duration=cooldown,
        )
        # No fetch before this time on the monotonic clock, after a failed one.
        self._retry_at: float = -float("inf")
        self._failures: int = 0
        self._kids: tuple[object, ...] | None = None

    @override
    def fetch_data(self) -> dict[str, object]:
        if time.monotonic() < self._retry_at:
            raise _KeySetUnavailable("waiting after a failed fetch")
        try:
            data = cast("dict[str, object]", super().fetch_data())
        except _FETCH_ERRORS as exc:
            # Counted from the failure: a fetch that timed out took that long.
            failed_at = time.monotonic()
            self._retry_at = failed_at + self.cooldown_duration
            self._failures += 1
            self._log_failure(exc, failed_at)
            raise _KeySetUnavailable(type(exc).__name__) from exc
        self._log_success(self._signing_kids(cast("list[object]", data["keys"])))
        self._retry_at = -float("inf")
        self._failures = 0
        return data

    def _log_failure(self, exc: Exception, now: float) -> None:
        cache = self.jwk_set_cache
        cached = cache.jwk_set_with_timestamp if cache else None
        if cache is None or cached is None or cache.is_expired():
            consequence = "every token is rejected until a fetch succeeds"
        else:
            left = cached.get_timestamp() + cache.lifespan - now
            consequence = f"the cached key set stays in use for another {left:.0f} s"
        log.warning(
            "cannot load the key set from %s: %s: %s%s; %s; no new fetch for %g s",
            self.uri,
            type(exc).__name__,
            str(exc) or "no detail",
            # A connection error says nothing about what the endpoint is.
            "" if isinstance(exc, jwt.PyJWKClientConnectionError | OSError) else _HINT,
            consequence,
            self.cooldown_duration,
        )

    def _signing_kids(self, entries: list[object]) -> tuple[object, ...]:
        """The kids of the signing keys in a fetched set; every other entry is logged."""
        kids: list[object] = []
        for index, entry in enumerate(entries):
            unused = _unused(entry)
            if unused is None:
                kids.append(cast("dict[str, object]", entry)["kid"])
                continue
            level, why = unused
            log.log(level, "key set entry %d %s; not used", index, why)
        return tuple(kids)

    def _log_success(self, kids: tuple[object, ...]) -> None:
        listed = ", ".join(_member(kid) for kid in kids)
        if not kids:
            log.warning(
                "the key set from %s has no signing key: every token is rejected for %d s%s",
                self.uri,
                JWKS_LIFESPAN_SECONDS,
                _HINT,
            )
        elif self._failures:
            log.info(
                "loaded the key set from %s after %d failed fetch(es); signing keys %s",
                self.uri,
                self._failures,
                listed,
            )
        elif kids != self._kids:
            log.info("loaded the key set from %s; signing keys %s", self.uri, listed)
        self._kids = kids


class AuthentikTokenVerifier(TokenVerifier):
    """Validates Authentik-issued JWT access tokens with the provider's JWKS."""

    # Not FastMCP's JWTVerifier: it accepts a single algorithm and does not
    # check `azp`. PyJWKClient fetches, parses and caches the key set; the
    # checks on the token are this class's own.

    def __init__(  # noqa: PLR0913 - keyword-only: every setting is named where it is passed
        self,
        *,
        issuer: str,
        jwks_url: str,
        client_ids: tuple[str, ...],
        algorithms: tuple[str, ...],
        leeway_seconds: int,
        jwks_min_refetch_seconds: int,
        timeout_seconds: float,
    ) -> None:
        super().__init__()
        self.issuer: str = issuer
        self.jwks_url: str = jwks_url
        self.client_ids: tuple[str, ...] = client_ids
        self.algorithms: tuple[str, ...] = algorithms
        self.leeway: int = leeway_seconds
        self._jwks: _ProviderJwks = _ProviderJwks(
            jwks_url, timeout=timeout_seconds, cooldown=jwks_min_refetch_seconds
        )
        # PyJWKClient blocks on its own lock while it fetches; one thread at a
        # time keeps the waiting requests off the shared worker threads.
        self._jwks_limiter: anyio.CapacityLimiter = anyio.CapacityLimiter(1)

    async def _signing_key(self, kid: str | None) -> jwt.PyJWK | None:
        """The provider's signing key for ``kid``, or None with one log line saying why."""
        if not kid:
            # Authentik always names the signing key; PyJWKClient looks keys up by it.
            log.info("bearer token rejected: no kid header")
            return None
        try:
            return await anyio.to_thread.run_sync(
                self._jwks.get_signing_key, kid, limiter=self._jwks_limiter
            )
        except jwt.PyJWKClientError as exc:
            # A problem with the endpoint or the set has been logged by the fetch.
            log.info(
                "bearer token rejected: no signing key for kid %r%s",
                _clip(kid),
                ": the key set could not be fetched" if isinstance(exc, _KeySetUnavailable) else "",
            )
            return None

    @override
    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            header: dict[str, object] = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            log.info("bearer token rejected: not a JWT")
            return None
        alg = header.get("alg")
        # Every allowed algorithm is a string, so the type check rejects nothing more.
        if not isinstance(alg, str) or alg not in self.algorithms:
            log.info("bearer token rejected: algorithm %r not allowed", alg)
            return None
        # PyJWT has already refused a token whose `kid` header is not a string.
        key = await self._signing_key(cast("str | None", header.get("kid")))
        if key is None:
            return None
        if key.algorithm_name != alg:
            # The token must name the algorithm its key is for; this is what
            # stops a token "signed" with HS256 and the public key.
            log.info("bearer token rejected: %s token for a %s key", alg, key.algorithm_name)
            return None
        try:
            claims = cast(
                "dict[str, object]",
                jwt.decode(
                    token,
                    key=key.key,  # pyright: ignore[reportAny] - PyJWT types PyJWK.key as Any
                    algorithms=[alg],
                    audience=list(self.client_ids),
                    issuer=self.issuer,
                    leeway=self.leeway,
                    options={"require": REQUIRED_CLAIMS},
                ),
            )
        except jwt.PyJWTError as exc:
            log.info("bearer token rejected: %s", exc)
            return None
        return self._access_token(token, claims)

    def _access_token(self, token: str, claims: dict[str, object]) -> AccessToken | None:
        azp = claims.get("azp")
        if azp not in self.client_ids:
            # Authentik puts `azp` only into access tokens, so this also keeps
            # an ID token from passing as one.
            log.info("bearer token rejected: azp %r is not a configured client", azp)
            return None
        username = claims.get(USERNAME_CLAIM)
        groups = claims.get(GROUPS_CLAIM)
        if not isinstance(username, str) or not username:
            log.info("bearer token rejected: no %s claim", USERNAME_CLAIM)
            return None
        if not isinstance(groups, list) or not all(
            isinstance(g, str) for g in cast("list[object]", groups)
        ):
            log.info("bearer token rejected for %s: no %s claim", username, GROUPS_CLAIM)
            return None
        scope = claims.get("scope")
        return AccessToken(
            token=token,
            client_id=str(azp),
            scopes=scope.split() if isinstance(scope, str) else [],
            # PyJWT has already checked that `exp` converts to an integer.
            expires_at=int(cast("int | float | str", claims["exp"])),
            subject=str(claims["sub"]),
            claims=claims,
        )


def identity_from(token: AccessToken | None) -> Identity:
    """The caller's identity from a token this module validated."""
    if token is None:
        raise PermissionError("not authenticated")
    username = token.claims.get(USERNAME_CLAIM)
    groups = token.claims.get(GROUPS_CLAIM)
    if not isinstance(username, str) or not isinstance(groups, list):
        raise PermissionError("the token carries no identity")
    return Identity(
        username=username, groups=frozenset(str(g) for g in cast("list[object]", groups))
    )
