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
from typing import cast, override

import anyio
import jwt
from fastmcp.server.auth import AccessToken, TokenVerifier

from memex_mcp.rights import Identity

log = logging.getLogger(__name__)

USERNAME_CLAIM = "preferred_username"
GROUPS_CLAIM = "groups"
REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "sub"]


# The key set is cached this long, then fetched again: a key the provider has
# revoked stops verifying at most this long after it left the key set.
JWKS_LIFESPAN_SECONDS = 300

# What a key lookup can raise besides PyJWT's own errors (measured at PyJWT
# 2.15.1): OSError for a connection reset while the answer is read, ValueError
# for an answer that is not JSON, RecursionError for one nested too deeply, and,
# out of a key entry, TypeError for an `alg` that is not a string and
# NotImplementedError for `alg` "none". Each of these fails the whole key set.
_LOOKUP_ERRORS = (
    jwt.PyJWTError,
    OSError,
    ValueError,
    RecursionError,
    TypeError,
    NotImplementedError,
)

# How PyJWKClient.get_signing_key says that the key set has no key for the kid.
_UNKNOWN_KID = "Unable to find a signing key that matches"


def _answered_without_keys(exc: Exception) -> bool:
    """Whether a failed lookup means the endpoint answered, but not with a usable key set."""
    if isinstance(exc, jwt.PyJWKClientConnectionError | OSError):
        return False
    return not str(exc).startswith(_UNKNOWN_KID)


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
        # A token with a kid the cached set lacks refetches the set, but not
        # sooner than jwks_min_refetch_seconds after the last successful fetch.
        self._jwks: jwt.PyJWKClient = jwt.PyJWKClient(
            jwks_url,
            cache_jwk_set=True,
            lifespan=JWKS_LIFESPAN_SECONDS,
            timeout=timeout_seconds,
            cooldown_duration=jwks_min_refetch_seconds,
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
        except _LOOKUP_ERRORS as exc:
            # PyJWT puts the token's kid into its message unescaped; the escape
            # keeps a line break in it from forging a log line.
            detail = str(exc).encode("unicode_escape").decode("ascii") or "no detail"
            hint = "; is [auth] jwks_url the provider's JWKS endpoint?"
            log.warning(
                "bearer token rejected: no signing key from %s: %s: %s%s",
                self.jwks_url,
                type(exc).__name__,
                detail,
                hint if _answered_without_keys(exc) else "",
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
