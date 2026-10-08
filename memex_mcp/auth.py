"""Access-token validation against Authentik, done locally.

Authentik's OAuth2 provider issues its access tokens as signed JWTs: the
token carries the ID-token claims plus ``azp``, ``uid`` and ``scope``, signed
with the provider's signing key, whose public half Authentik publishes at the
application's JWKS endpoint. So the server validates each token itself —
signature against the JWKS, ``iss``, ``exp``, ``aud`` and ``azp`` against the
configured client ids — and reads the username (``preferred_username``) and
groups (``groups``) from it; no introspection call per request.

A token without a well-formed ``preferred_username`` or ``groups`` claim is
rejected: the provider is then missing the ``profile`` scope mapping, and
without groups no rights decision is possible.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import cast, override

import anyio
import httpx2
import jwt
from fastmcp.server.auth import AccessToken, TokenVerifier

from memex_mcp.rights import Identity

log = logging.getLogger(__name__)

USERNAME_CLAIM = "preferred_username"
GROUPS_CLAIM = "groups"
REQUIRED_CLAIMS = ["exp", "iat", "iss", "aud", "sub"]


def _key_entries(document: object) -> list[object]:
    """The ``keys`` of a JWKS document as they come; PyJWKSetError when it holds no list."""
    if not isinstance(document, dict):
        raise jwt.PyJWKSetError("the key set is not a JSON object")
    # A JSON object's keys are strings.
    entries = cast("dict[str, object]", document).get("keys", [])
    if not isinstance(entries, list):
        raise jwt.PyJWKSetError("the key set's keys are not a list")
    return cast("list[object]", entries)


def _signing_key(entry: object) -> jwt.PyJWK | None:
    """The signing key a JWKS entry describes, or None; a malformed entry is logged."""
    if not isinstance(entry, dict):
        log.warning("skipping an entry that is not a JSON object in the key set")
        return None
    data = cast("dict[str, object]", entry)  # a JSON object's keys are strings
    # Authentik also lists the encryption key (use "enc") when one is set.
    if data.get("use", "sig") != "sig":
        return None
    kid = data.get("kid")
    # Both are strings (RFC 7517); PyJWT looks `alg` up and the key set is keyed by `kid`.
    if not isinstance(kid, str | None) or not isinstance(data.get("alg", ""), str):
        log.warning("skipping a signing key whose kid or alg is not a string")
        return None
    # PyJWT has no key form for alg "none" and says so with NotImplementedError.
    try:
        return jwt.PyJWK(data)
    except (jwt.PyJWTError, NotImplementedError) as exc:
        log.warning("skipping an unusable signing key %r: %s", kid, exc)
        return None


class AuthentikTokenVerifier(TokenVerifier):
    """Validates Authentik-issued JWT access tokens with the provider's JWKS."""

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
        transport: httpx2.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.issuer: str = issuer
        self.jwks_url: str = jwks_url
        self.client_ids: tuple[str, ...] = client_ids
        self.algorithms: tuple[str, ...] = algorithms
        self.leeway: int = leeway_seconds
        self.min_refetch: int = jwks_min_refetch_seconds
        self.timeout: float = timeout_seconds
        self._transport: httpx2.AsyncBaseTransport | None = transport
        self._clock: Callable[[], float] = clock
        self._keys: dict[str | None, jwt.PyJWK] = {}
        self._fetched_at: float | None = None
        self._fetch_lock: anyio.Lock = anyio.Lock()

    async def _fetch_keys(self) -> None:
        async with httpx2.AsyncClient(transport=self._transport, timeout=self.timeout) as client:
            response = await client.get(self.jwks_url)
            response.raise_for_status()
            document = cast("object", response.json())
        keys: dict[str | None, jwt.PyJWK] = {}
        for entry in _key_entries(document):
            key = _signing_key(entry)
            if key is not None:
                keys[key.key_id] = key
        self._keys = keys

    async def _key_for(self, kid: str | None) -> jwt.PyJWK | None:
        if kid in self._keys:
            return self._keys[kid]
        async with self._fetch_lock:
            if kid not in self._keys:
                now = self._clock()
                if self._fetched_at is None or now - self._fetched_at >= self.min_refetch:
                    self._fetched_at = now
                    try:
                        await self._fetch_keys()
                    except (httpx2.HTTPError, ValueError, jwt.PyJWTError) as exc:
                        log.warning("cannot load the signing keys from %s: %s", self.jwks_url, exc)
                        return None
        if kid in self._keys:
            return self._keys[kid]
        if kid is None and len(self._keys) == 1:
            return next(iter(self._keys.values()))
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
        key = await self._key_for(cast("str | None", header.get("kid")))
        if key is None:
            log.info("bearer token rejected: no signing key for kid %r", header.get("kid"))
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
