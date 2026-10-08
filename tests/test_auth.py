"""Token validation against a local key and a mocked JWKS endpoint (no network)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from base64 import urlsafe_b64encode
from typing import Any

import httpx2
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from memex_mcp.auth import AuthentikTokenVerifier, identity_from

ISSUER = "https://auth.example.org/application/o/memex/"
JWKS_URL = ISSUER + "jwks/"
CLIENT = "claude-code"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def new_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwk_of(key: rsa.RSAPrivateKey, kid: str) -> dict[str, Any]:
    jwk = RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    return {**jwk, "kid": kid, "alg": "RS256", "use": "sig"}


class FakeJwks:
    """The provider's JWKS endpoint; counts fetches and can go down."""

    def __init__(self, *keys: tuple[rsa.RSAPrivateKey, str]) -> None:
        self.keys = list(keys)
        self.fetches = 0
        self.down = False

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == JWKS_URL
        self.fetches += 1
        if self.down:
            raise httpx2.ConnectError("connection refused", request=request)
        return httpx2.Response(200, json={"keys": [jwk_of(k, kid) for k, kid in self.keys]})


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


KEY = new_key()


def claims(**overrides: Any) -> dict[str, Any]:
    """What an Authentik access token carries for a user with the profile scope."""
    now = int(time.time())
    base: dict[str, Any] = {
        "iss": ISSUER,
        "sub": "8f0c1d",
        "aud": CLIENT,
        "azp": CLIENT,
        "exp": now + 300,
        "iat": now,
        "auth_time": now,
        "uid": "abc123",
        "scope": "openid profile offline_access",
        "preferred_username": "alice",
        "groups": ["memex", "household"],
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def token(
    key: rsa.RSAPrivateKey = KEY, kid: str = "k1", alg: str = "RS256", **overrides: Any
) -> str:
    return jwt.encode(claims(**overrides), key, algorithm=alg, headers={"kid": kid})


def make_verifier(jwks: FakeJwks, clock: Clock | None = None) -> AuthentikTokenVerifier:
    return AuthentikTokenVerifier(
        issuer=ISSUER,
        jwks_url=JWKS_URL,
        client_ids=(CLIENT,),
        algorithms=("RS256", "ES256"),
        leeway_seconds=30,
        jwks_min_refetch_seconds=60,
        timeout_seconds=1.0,
        transport=httpx2.MockTransport(jwks.handler),
        clock=clock or Clock(),
    )


@pytest.fixture
def jwks() -> FakeJwks:
    return FakeJwks((KEY, "k1"))


@pytest.mark.anyio
async def test_valid_token_yields_the_identity(jwks: FakeJwks) -> None:
    access = await make_verifier(jwks).verify_token(token())
    assert access is not None
    identity = identity_from(access)
    assert identity.username == "alice"
    assert identity.groups == {"memex", "household"}
    assert access.client_id == CLIENT
    assert "profile" in access.scopes


@pytest.mark.anyio
async def test_keys_are_cached(jwks: FakeJwks) -> None:
    verifier = make_verifier(jwks)
    for _ in range(3):
        assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 1


REJECTED = {
    "expired": {"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200},
    "wrong issuer": {"iss": "https://auth.example.org/application/o/other/"},
    "global issuer": {"iss": "https://auth.example.org/"},
    "wrong audience": {"aud": "open-webui", "azp": "open-webui"},
    "audience ok but azp foreign": {"azp": "open-webui"},
    "no azp (an ID token)": {"azp": None, "uid": None, "scope": None},
    "missing groups": {"groups": None},
    "groups not a list": {"groups": "memex"},
    "groups with a non-string": {"groups": ["memex", 7]},
    "missing username": {"preferred_username": None},
    "empty username": {"preferred_username": ""},
    "missing exp": {"exp": None},
    "missing sub": {"sub": None},
}


@pytest.mark.anyio
@pytest.mark.parametrize("overrides", REJECTED.values(), ids=list(REJECTED))
async def test_bad_claims_are_rejected(jwks: FakeJwks, overrides: dict[str, Any]) -> None:
    assert await make_verifier(jwks).verify_token(token(**overrides)) is None


@pytest.mark.anyio
async def test_expiry_within_leeway_is_accepted(jwks: FakeJwks) -> None:
    now = int(time.time())
    assert await make_verifier(jwks).verify_token(token(exp=now - 5)) is not None


@pytest.mark.anyio
async def test_token_signed_by_another_key_is_rejected(jwks: FakeJwks) -> None:
    assert await make_verifier(jwks).verify_token(token(key=new_key())) is None


@pytest.mark.anyio
async def test_unsigned_token_is_rejected(jwks: FakeJwks) -> None:
    header = b64(json.dumps({"alg": "none", "kid": "k1"}).encode())
    unsigned = f"{header}.{b64(json.dumps(claims()).encode())}."
    assert await make_verifier(jwks).verify_token(unsigned) is None


def b64(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode()


def hmac_with_public_key() -> str:
    """The classic algorithm confusion: HS256 keyed with the published public key."""
    public_pem = KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    header = b64(json.dumps({"alg": "HS256", "kid": "k1", "typ": "JWT"}).encode())
    payload = b64(json.dumps(claims()).encode())
    signature = hmac.new(public_pem, f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{b64(signature)}"


@pytest.mark.anyio
async def test_hmac_with_the_public_key_is_rejected(jwks: FakeJwks) -> None:
    forged = hmac_with_public_key()
    assert await make_verifier(jwks).verify_token(forged) is None
    # Even a misconfiguration that allows HS256 does not let it through.
    permissive = AuthentikTokenVerifier(
        issuer=ISSUER,
        jwks_url=JWKS_URL,
        client_ids=(CLIENT,),
        algorithms=("RS256", "HS256"),
        leeway_seconds=30,
        jwks_min_refetch_seconds=60,
        timeout_seconds=1.0,
        transport=httpx2.MockTransport(jwks.handler),
    )
    assert await permissive.verify_token(forged) is None


@pytest.mark.anyio
async def test_garbage_is_rejected(jwks: FakeJwks) -> None:
    verifier = make_verifier(jwks)
    for value in ("", "not-a-jwt", "a.b.c", "Bearer xyz"):
        assert await verifier.verify_token(value) is None


@pytest.mark.anyio
async def test_unknown_kid_refetches_at_most_once_per_interval() -> None:
    jwks = FakeJwks((KEY, "k1"))
    clock = Clock()
    verifier = make_verifier(jwks, clock)
    assert await verifier.verify_token(token()) is not None
    rotated = new_key()
    assert await verifier.verify_token(token(key=rotated, kid="k2")) is None
    assert jwks.fetches == 1  # within the interval: no refetch for an unknown kid
    jwks.keys.append((rotated, "k2"))
    clock.now += 61
    assert await verifier.verify_token(token(key=rotated, kid="k2")) is not None
    assert jwks.fetches == 2


@pytest.mark.anyio
async def test_jwks_endpoint_down_rejects_and_recovers() -> None:
    jwks = FakeJwks((KEY, "k1"))
    jwks.down = True
    clock = Clock()
    verifier = make_verifier(jwks, clock)
    assert await verifier.verify_token(token()) is None
    jwks.down = False
    clock.now += 61
    assert await verifier.verify_token(token()) is not None


def test_identity_needs_a_token() -> None:
    with pytest.raises(PermissionError):
        identity_from(None)


@pytest.mark.anyio
async def test_encryption_keys_and_keys_without_alg_are_handled() -> None:
    """Authentik lists an encryption key (use "enc") beside the signing key when one is set."""
    enc_key = new_key()

    def handler(request: httpx2.Request) -> httpx2.Response:
        signing = jwk_of(KEY, "k1")
        del signing["alg"]  # PyJWT derives RS256 from an RSA key
        encryption = {**jwk_of(enc_key, "e1"), "use": "enc", "alg": "RSA-OAEP-256"}
        return httpx2.Response(200, json={"keys": [encryption, signing]})

    verifier = AuthentikTokenVerifier(
        issuer=ISSUER,
        jwks_url=JWKS_URL,
        client_ids=(CLIENT,),
        algorithms=("RS256",),
        leeway_seconds=30,
        jwks_min_refetch_seconds=60,
        timeout_seconds=1.0,
        transport=httpx2.MockTransport(handler),
    )
    assert await verifier.verify_token(token()) is not None
    assert await verifier.verify_token(token(key=enc_key, kid="e1")) is None
