"""Token validation against a local key and a mocked JWKS endpoint (no network)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from base64 import urlsafe_b64encode
from collections.abc import Callable
from typing import TypedDict, cast

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


def jwk_of(key: rsa.RSAPrivateKey, kid: str) -> dict[str, object]:
    jwk = RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    return {**jwk, "kid": kid, "alg": "RS256", "use": "sig"}


class FakeJwks:
    """The provider's JWKS endpoint; counts fetches and can go down."""

    def __init__(self, *keys: tuple[rsa.RSAPrivateKey, str]) -> None:
        self.keys: list[tuple[rsa.RSAPrivateKey, str]] = list(keys)
        self.fetches: int = 0
        self.down: bool = False

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == JWKS_URL
        self.fetches += 1
        if self.down:
            raise httpx2.ConnectError("connection refused", request=request)
        return httpx2.Response(200, json={"keys": [jwk_of(k, kid) for k, kid in self.keys]})


class Clock:
    def __init__(self) -> None:
        self.now: float = 1000.0

    def __call__(self) -> float:
        return self.now


KEY = new_key()


def claims(**overrides: object) -> dict[str, object]:
    """What an Authentik access token carries for a user with the profile scope."""
    now = int(time.time())
    base: dict[str, object] = {
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
    key: rsa.RSAPrivateKey = KEY, kid: str = "k1", alg: str = "RS256", **overrides: object
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


class ClaimOverrides(TypedDict, total=False):
    """The claims a rejected case changes; None leaves a claim out."""

    iss: str
    aud: str
    azp: str | None
    uid: None
    scope: None
    exp: int | None
    iat: int
    groups: str | list[str | int] | None
    preferred_username: str | None
    sub: None


REJECTED: dict[str, ClaimOverrides] = {
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
async def test_bad_claims_are_rejected(jwks: FakeJwks, overrides: ClaimOverrides) -> None:
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

    def handler(_request: httpx2.Request) -> httpx2.Response:
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


def verifier_serving(document: object) -> AuthentikTokenVerifier:
    """A verifier whose JWKS endpoint answers with ``document``."""
    return AuthentikTokenVerifier(
        issuer=ISSUER,
        jwks_url=JWKS_URL,
        client_ids=(CLIENT,),
        algorithms=("RS256", "ES256"),
        leeway_seconds=30,
        jwks_min_refetch_seconds=60,
        timeout_seconds=1.0,
        transport=httpx2.MockTransport(lambda _request: httpx2.Response(200, json=document)),
    )


def without(jwk: dict[str, object], name: str) -> dict[str, object]:
    return {k: v for k, v in jwk.items() if k != name}


GOOD_JWK = jwk_of(KEY, "k1")
ENC_JWK = {**jwk_of(new_key(), "e1"), "use": "enc", "alg": "RSA-OAEP-256"}
UNUSABLE = "unusable signing key"

MALFORMED_KEY_SETS: dict[str, tuple[object, str]] = {
    "not an object": ([GOOD_JWK], "the key set is a list, not an object"),
    "no keys member": ({}, "the key set has no 'keys' member"),
    "keys a number": ({"keys": 5}, "keys is a number, not a list"),
    "keys null": ({"keys": None}, "keys is null, not a list"),
    "keys an object": ({"keys": GOOD_JWK}, "keys is an object, not a list"),
    "keys empty": ({"keys": []}, "no usable signing key"),
    "encryption key only": ({"keys": [ENC_JWK]}, "no usable signing key"),
    "entries not objects": ({"keys": [5, None, "k1"]}, "entry 1 is null, not an object"),
    "kid a list": ({"keys": [{**GOOD_JWK, "kid": ["k1"]}]}, "entry 0: kid is a list, not a string"),
    "alg a list": (
        {"keys": [{**GOOD_JWK, "alg": ["RS256"]}]},
        "entry 0: alg is a list, not a string",
    ),
    "use a list": (
        {"keys": [{**GOOD_JWK, "use": ["sig"]}]},
        "entry 0: use is a list, not a string",
    ),
    "use unknown": ({"keys": [{**GOOD_JWK, "use": "foo"}]}, "entry 0: use 'foo' is neither"),
    "use null": ({"keys": [{**GOOD_JWK, "use": None}]}, "entry 0: use is null, not a string"),
    "alg none": (
        {"keys": [{**GOOD_JWK, "alg": "none"}]},
        f"entry 0: {UNUSABLE} (kid 'k1', kty 'RSA', alg 'none'): NotImplementedError",
    ),
    "no kty": (
        {"keys": [without(GOOD_JWK, "kty")]},
        f"entry 0: {UNUSABLE} (kid 'k1', kty None, alg 'RS256'): InvalidKeyError",
    ),
    "no modulus": (
        {"keys": [without(GOOD_JWK, "n")]},
        f"entry 0: {UNUSABLE} (kid 'k1', kty 'RSA', alg 'RS256'): InvalidKeyError",
    ),
}


def warned(caplog: pytest.LogCaptureFixture, text: str) -> bool:
    return any(r.levelno == logging.WARNING and text in r.getMessage() for r in caplog.records)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("document", "problem"), MALFORMED_KEY_SETS.values(), ids=list(MALFORMED_KEY_SETS)
)
async def test_a_malformed_key_set_rejects_the_token_and_logs_why(
    document: object, problem: str, caplog: pytest.LogCaptureFixture
) -> None:
    bearer = token()
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await verifier_serving(document).verify_token(bearer) is None
    assert warned(caplog, problem)
    # However the set is broken, no usable key is left: the fetch counts as failed.
    [failed] = [r.getMessage() for r in caplog.records if "cannot load" in r.getMessage()]
    assert failed.startswith(f"cannot load the signing keys from {JWKS_URL}: PyJWKSetError: ")
    assert failed.count(JWKS_URL) == 1
    assert "is [auth] jwks_url the provider's JWKS endpoint?" in failed
    assert failed.endswith("no cached key: every token is rejected until a fetch succeeds")
    assert "no signing key for kid 'k1'" in caplog.text
    assert bearer not in caplog.text


@pytest.mark.anyio
async def test_a_malformed_entry_does_not_hide_the_good_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    document = {"keys": [5, {**GOOD_JWK, "kid": ["k1"]}, {**GOOD_JWK, "alg": "none"}, GOOD_JWK]}
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await verifier_serving(document).verify_token(token()) is not None
    assert warned(caplog, "entry 0 is a number, not an object")
    assert warned(caplog, "entry 1: kid is a list, not a string")
    assert warned(caplog, f"entry 2: {UNUSABLE}")


@pytest.mark.anyio
async def test_a_null_alg_counts_as_absent() -> None:
    assert await verifier_serving({"keys": [{**GOOD_JWK, "alg": None}]}).verify_token(token())


@pytest.mark.anyio
async def test_a_null_use_is_skipped() -> None:
    assert (
        await verifier_serving({"keys": [{**GOOD_JWK, "use": None}]}).verify_token(token()) is None
    )


@pytest.mark.anyio
async def test_a_skipped_key_never_logs_its_key_material(caplog: pytest.LogCaptureFixture) -> None:
    private = cast("dict[str, str]", RSAAlgorithm.to_jwk(new_key(), as_dict=True))
    members = [private[name] for name in ("n", "d", "p", "q", "dp", "dq", "qi")]
    # Without `kty` PyJWT's own message would carry the whole key.
    document = {"keys": [without(cast("dict[str, object]", private), "kty"), GOOD_JWK]}
    with caplog.at_level(logging.DEBUG, logger="memex_mcp.auth"):
        assert await verifier_serving(document).verify_token(token()) is not None
    assert warned(caplog, f"entry 0: {UNUSABLE}")
    assert not any(member in caplog.text for member in members)


def answer_json(document: object) -> Callable[[], httpx2.Response]:
    return lambda: httpx2.Response(200, json=document)


def answer_html() -> httpx2.Response:
    return httpx2.Response(
        200, content=b"<html>login</html>", headers={"content-type": "text/html"}
    )


def refuse() -> httpx2.Response:
    raise httpx2.ConnectError("connection refused")


def refuse_silently() -> httpx2.Response:
    raise httpx2.ReadTimeout("")


HINT = "; is [auth] jwks_url the provider's JWKS endpoint?"

BAD_REFETCHES: dict[str, tuple[Callable[[], httpx2.Response], str]] = {
    "no keys member": (answer_json({}), f"PyJWKSetError: the key set has no 'keys' member{HINT}"),
    "keys empty": (answer_json({"keys": []}), f"the key set has no usable signing key{HINT}"),
    "encryption key only": (answer_json({"keys": [ENC_JWK]}), f"no usable signing key{HINT}"),
    "keys a number": (answer_json({"keys": 5}), f"keys is a number, not a list{HINT}"),
    "not JSON": (answer_html, f"not JSON (content-type 'text/html'){HINT}"),
    "endpoint down": (refuse, "ConnectError: connection refused; keeping"),
    "endpoint silent": (refuse_silently, "ReadTimeout: no detail; keeping"),
}


@pytest.mark.anyio
@pytest.mark.parametrize(("bad_answer", "problem"), BAD_REFETCHES.values(), ids=list(BAD_REFETCHES))
async def test_a_bad_refetch_keeps_the_cached_keys(
    bad_answer: Callable[[], httpx2.Response], problem: str, caplog: pytest.LogCaptureFixture
) -> None:
    answers = [answer_json({"keys": [GOOD_JWK]}), bad_answer]
    clock = Clock()
    verifier = AuthentikTokenVerifier(
        issuer=ISSUER,
        jwks_url=JWKS_URL,
        client_ids=(CLIENT,),
        algorithms=("RS256",),
        leeway_seconds=30,
        jwks_min_refetch_seconds=60,
        timeout_seconds=1.0,
        transport=httpx2.MockTransport(lambda _request: answers.pop(0)()),
        clock=clock,
    )
    assert await verifier.verify_token(token()) is not None
    clock.now += 61
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await verifier.verify_token(token(key=new_key(), kid="k2")) is None
    assert warned(caplog, problem)
    assert warned(caplog, "; keeping 1 cached key(s)")
    assert answers == []
    assert await verifier.verify_token(token()) is not None


@pytest.mark.anyio
async def test_a_failed_refetch_keeps_the_single_key_fallback() -> None:
    """A token without `kid` still verifies against the one cached key while the JWKS is down."""
    jwks = FakeJwks((KEY, "k1"))
    clock = Clock()
    verifier = make_verifier(jwks, clock)
    kidless = jwt.encode(claims(), KEY, algorithm="RS256")
    assert await verifier.verify_token(kidless) is not None
    jwks.down = True
    clock.now += 61
    assert await verifier.verify_token(kidless) is not None
    assert jwks.fetches == 2
