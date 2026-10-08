"""Token validation against a local key and a JWKS endpoint on 127.0.0.1 (no network)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import socket
import struct
import threading
import time
import uuid
from base64 import urlsafe_b64encode
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Self, TypedDict, cast, override

import anyio
import jwt
import jwt.api_jwk
import jwt.jwk_set_cache
import jwt.jwks_client
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from memex_mcp.auth import JWKS_LIFESPAN_SECONDS, AuthentikTokenVerifier, identity_from

ISSUER = "https://auth.example.org/application/o/memex/"
JWKS_PATH = "/application/o/memex/jwks/"
CLIENT = "claude-code"
MIN_REFETCH = 60


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def new_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwk_of(key: rsa.RSAPrivateKey, kid: str) -> dict[str, object]:
    jwk = RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    return {**jwk, "kid": kid, "alg": "RS256", "use": "sig"}


Answer = Callable[[BaseHTTPRequestHandler], None]


def respond(handler: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("content-type", content_type)
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def answer_json(document: object) -> Answer:
    return lambda handler: respond(handler, 200, json.dumps(document).encode(), "application/json")


def answer_raw(body: bytes, status: int = 200, content_type: str = "text/html") -> Answer:
    return lambda handler: respond(handler, status, body, content_type)


def hang_up(handler: BaseHTTPRequestHandler) -> None:
    """Close the connection without an answer, as an endpoint that is down."""
    handler.close_connection = True


def reset_mid_answer(handler: BaseHTTPRequestHandler) -> None:
    """Send half an answer, then reset the connection."""
    handler.send_response(200)
    handler.send_header("content-length", "1000")
    handler.end_headers()
    handler.wfile.write(b'{"keys": [')
    handler.wfile.flush()
    connection = cast("socket.socket", handler.connection)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    connection.close()


CONNECTION_ERROR = "PyJWKClientConnectionError: Fail to fetch data from the url, err: "
HUNG_UP = CONNECTION_ERROR + '"Remote end closed connection without response"'


class FakeJwks:
    """The provider's JWKS endpoint, served on 127.0.0.1; counts fetches and can go down."""

    def __init__(self, *keys: tuple[rsa.RSAPrivateKey, str]) -> None:
        self.keys: list[tuple[rsa.RSAPrivateKey, str]] = list(keys)
        self.fetches: int = 0
        self.down: bool = False
        # Answers every fetch with this instead of the keys above.
        self.answer: Answer | None = None
        self.delay: float = 0.0
        jwks = self

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                assert self.path == JWKS_PATH
                jwks.fetches += 1
                time.sleep(jwks.delay)
                if jwks.down:
                    hang_up(self)
                elif jwks.answer is not None:
                    jwks.answer(self)
                else:
                    document = {"keys": [jwk_of(k, kid) for k, kid in jwks.keys]}
                    answer_json(document)(self)

        self._server: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # A short poll interval, since shutdown waits for the next poll.
        self._thread: threading.Thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        self.url: str = f"http://127.0.0.1:{self._server.server_address[1]}{JWKS_PATH}"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


class Clock:
    """Stands in for the ``time`` module where PyJWT reads ``time.monotonic``."""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """PyJWT's key set cache and refetch cooldown, on a clock the test moves.

    These three modules are where PyJWT 2.15.1 reads ``time.monotonic`` for
    them; only their name ``time`` is replaced, so any other use of it fails.
    """
    fake = Clock()
    for module in (jwt.jwks_client, jwt.jwk_set_cache, jwt.api_jwk):
        monkeypatch.setattr(module, "time", fake)
    return fake


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


def make_verifier(
    jwks: FakeJwks, algorithms: tuple[str, ...] = ("RS256", "ES256")
) -> AuthentikTokenVerifier:
    return AuthentikTokenVerifier(
        issuer=ISSUER,
        jwks_url=jwks.url,
        client_ids=(CLIENT,),
        algorithms=algorithms,
        leeway_seconds=30,
        jwks_min_refetch_seconds=MIN_REFETCH,
        timeout_seconds=1.0,
    )


@pytest.fixture
def jwks() -> Iterator[FakeJwks]:
    with FakeJwks((KEY, "k1")) as endpoint:
        yield endpoint


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
    permissive = make_verifier(jwks, algorithms=("RS256", "HS256"))
    assert await permissive.verify_token(forged) is None


@pytest.mark.anyio
async def test_a_token_must_name_the_algorithm_of_its_key(jwks: FakeJwks) -> None:
    """PS256 verifies with an RSA key too, but this key is published for RS256."""
    verifier = make_verifier(jwks, algorithms=("RS256", "PS256"))
    assert await verifier.verify_token(token(alg="PS256")) is None
    assert await verifier.verify_token(token()) is not None


@pytest.mark.anyio
async def test_garbage_is_rejected(jwks: FakeJwks) -> None:
    verifier = make_verifier(jwks)
    for value in ("", "not-a-jwt", "a.b.c", "Bearer xyz"):
        assert await verifier.verify_token(value) is None


def test_identity_needs_a_token() -> None:
    with pytest.raises(PermissionError):
        identity_from(None)


@pytest.mark.anyio
async def test_a_token_without_kid_is_rejected_without_a_fetch(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    verifier = make_verifier(jwks)
    kidless = jwt.encode(claims(), KEY, algorithm="RS256")
    empty_kid = jwt.encode(claims(), KEY, algorithm="RS256", headers={"kid": ""})
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await verifier.verify_token(kidless) is None
        assert await verifier.verify_token(empty_kid) is None
    assert jwks.fetches == 0
    assert caplog.messages == ["bearer token rejected: no kid header"] * 2


@pytest.mark.anyio
async def test_a_rotated_key_is_fetched(jwks: FakeJwks, clock: Clock) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    rotated = new_key()
    jwks.keys.append((rotated, "k2"))
    clock.now += MIN_REFETCH + 1
    assert await verifier.verify_token(token(key=rotated, kid="k2")) is not None
    assert jwks.fetches == 2


@pytest.mark.anyio
async def test_a_revoked_key_stops_verifying_once_the_cached_set_expires(
    jwks: FakeJwks, clock: Clock
) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.keys = [(new_key(), "k2")]  # k1 revoked at the provider
    clock.now += JWKS_LIFESPAN_SECONDS - 1
    assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 1
    clock.now += 2
    assert await verifier.verify_token(token()) is None
    assert jwks.fetches == 2


@pytest.mark.anyio
async def test_a_cached_key_verifies_while_the_endpoint_is_down(
    jwks: FakeJwks, clock: Clock
) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.down = True
    clock.now += JWKS_LIFESPAN_SECONDS - 1
    assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 1


@pytest.mark.anyio
async def test_an_expired_key_set_is_not_used_while_the_endpoint_is_down(
    jwks: FakeJwks, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.down = True
    clock.now += JWKS_LIFESPAN_SECONDS + 1
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await verifier.verify_token(token()) is None
    assert caplog.messages == [f"bearer token rejected: no signing key from {jwks.url}: {HUNG_UP}"]


@pytest.mark.anyio
async def test_the_cooldown_limits_refetches_for_unknown_kids(jwks: FakeJwks, clock: Clock) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    for _ in range(5):
        assert await verifier.verify_token(token(kid=uuid.uuid4().hex)) is None
    assert jwks.fetches == 1
    clock.now += MIN_REFETCH + 1
    for _ in range(5):
        assert await verifier.verify_token(token(kid=uuid.uuid4().hex)) is None
    assert jwks.fetches == 2


@pytest.mark.anyio
async def test_jwks_endpoint_down_rejects_and_recovers(jwks: FakeJwks) -> None:
    jwks.down = True
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is None
    jwks.down = False
    assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 2


@pytest.mark.anyio
async def test_concurrent_requests_fetch_the_key_set_once(jwks: FakeJwks) -> None:
    jwks.delay = 0.2
    verifier = make_verifier(jwks)
    results: list[bool] = []

    async def verify() -> None:
        results.append(await verifier.verify_token(token()) is not None)

    async with anyio.create_task_group() as group:
        for _ in range(5):
            group.start_soon(verify)
    assert results == [True] * 5
    assert jwks.fetches == 1


@pytest.mark.anyio
async def test_encryption_keys_and_keys_without_alg_are_handled(jwks: FakeJwks) -> None:
    """Authentik lists an encryption key (use "enc") beside the signing key when one is set."""
    enc_key = new_key()
    signing = jwk_of(KEY, "k1")
    del signing["alg"]  # PyJWT derives RS256 from an RSA key
    encryption = {**jwk_of(enc_key, "e1"), "use": "enc", "alg": "RSA-OAEP-256"}
    # Without `alg` PyJWT would make an RS256 key of it; `use` keeps it out.
    encryption_without_alg = {**without(jwk_of(enc_key, "e2"), "alg"), "use": "enc"}
    jwks.answer = answer_json({"keys": [encryption, encryption_without_alg, signing]})
    verifier = make_verifier(jwks, algorithms=("RS256",))
    assert await verifier.verify_token(token()) is not None
    assert await verifier.verify_token(token(key=enc_key, kid="e1")) is None
    assert await verifier.verify_token(token(key=enc_key, kid="e2")) is None


def without(jwk: dict[str, object], name: str) -> dict[str, object]:
    return {k: v for k, v in jwk.items() if k != name}


GOOD_JWK = jwk_of(KEY, "k1")
ENC_JWK = {**jwk_of(new_key(), "e1"), "use": "enc", "alg": "RSA-OAEP-256"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "signing_key",
    [without(GOOD_JWK, "use"), {**GOOD_JWK, "use": None}],
    ids=["use absent", "use null"],
)
async def test_a_signing_key_without_use_verifies(
    jwks: FakeJwks, signing_key: dict[str, object]
) -> None:
    jwks.answer = answer_json({"keys": [signing_key]})
    assert await make_verifier(jwks).verify_token(token()) is not None


@pytest.mark.anyio
async def test_a_null_alg_counts_as_absent(jwks: FakeJwks) -> None:
    jwks.answer = answer_json({"keys": [{**GOOD_JWK, "alg": None}]})
    assert await make_verifier(jwks).verify_token(token()) is not None


@pytest.mark.anyio
async def test_a_malformed_entry_does_not_hide_the_good_key(jwks: FakeJwks) -> None:
    jwks.answer = answer_json(
        {
            "keys": [
                5,
                {**GOOD_JWK, "kid": ["k1"]},
                {**GOOD_JWK, "alg": 5, "kid": "k2"},
                {**without(GOOD_JWK, "kty"), "kid": "k3"},
                GOOD_JWK,
            ]
        }
    )
    assert await make_verifier(jwks).verify_token(token()) is not None


HINT = "; is [auth] jwks_url the provider's JWKS endpoint?"
NO_SIGNING_KEYS = "PyJWKClientError: The JWKS endpoint did not contain any signing keys"
NO_USABLE_KEYS = "PyJWKSetError: The JWK Set did not contain any usable keys."

# What the endpoint answers -> the cause the warning names, and whether the
# endpoint answered, but not with a usable key set (the jwks_url hint).
LOOKUP_FAILURES: dict[str, tuple[Answer, str, bool]] = {
    "not JSON": (answer_raw(b"<html>login</html>"), "JSONDecodeError: Expecting value", True),
    "not UTF-8": (answer_raw(b"\xff\xfe\x00"), "UnicodeDecodeError: ", True),
    "nested too deeply": (answer_raw(b"[" * 100_000), "RecursionError: maximum recursion", True),
    "not an object": (
        answer_json([GOOD_JWK]),
        "PyJWKClientError: The JWKS endpoint did not return a JSON object",
        True,
    ),
    "no keys member": (
        answer_json({}),
        "PyJWKSetError: The JWK Set did not contain any keys",
        True,
    ),
    "keys a number": (answer_json({"keys": 5}), "PyJWKSetError: Invalid JWK Set value", True),
    "keys an object": (
        answer_json({"keys": GOOD_JWK}),
        "PyJWKSetError: Invalid JWK Set value",
        True,
    ),
    "no usable key": (answer_json({"keys": [without(GOOD_JWK, "kty")]}), NO_USABLE_KEYS, True),
    "entries not objects": (answer_json({"keys": [5, None, "k1"]}), NO_USABLE_KEYS, True),
    "encryption key only": (answer_json({"keys": [ENC_JWK]}), NO_USABLE_KEYS, True),
    "only keys marked enc": (
        answer_json({"keys": [{**GOOD_JWK, "use": "enc"}]}),
        NO_SIGNING_KEYS,
        True,
    ),
    "keys without kid": (answer_json({"keys": [without(GOOD_JWK, "kid")]}), NO_SIGNING_KEYS, True),
    "alg none beside a good key": (
        answer_json({"keys": [{**GOOD_JWK, "alg": "none", "kid": "k0"}, GOOD_JWK]}),
        "NotImplementedError: no detail",
        True,
    ),
    "alg a list beside a good key": (
        answer_json({"keys": [{**GOOD_JWK, "alg": ["RS256"], "kid": "k0"}, GOOD_JWK]}),
        "TypeError: unhashable type: 'list'",
        True,
    ),
    "not found": (
        answer_raw(b"", status=404),
        CONNECTION_ERROR + '"HTTP Error 404',
        False,
    ),
    "redirect": (
        answer_raw(b"", status=302),
        CONNECTION_ERROR + '"HTTP Error 302',
        False,
    ),
    "endpoint down": (hang_up, HUNG_UP, False),
    "reset while answering": (reset_mid_answer, "ConnectionResetError: ", False),
}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("answer", "cause", "hint"), LOOKUP_FAILURES.values(), ids=list(LOOKUP_FAILURES)
)
async def test_a_failed_key_lookup_rejects_the_token_with_one_warning(
    jwks: FakeJwks, answer: Answer, cause: str, hint: bool, caplog: pytest.LogCaptureFixture
) -> None:
    jwks.answer = answer
    bearer = token()
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(bearer) is None
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert message.startswith(f"bearer token rejected: no signing key from {jwks.url}: {cause}")
    assert message.count(jwks.url) == 1
    assert message.endswith(HINT) is hint
    assert bearer not in caplog.text


@pytest.mark.anyio
async def test_an_unknown_kid_names_the_kid_without_the_hint(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token(kid="k9")) is None
    unknown = 'PyJWKClientError: Unable to find a signing key that matches: "k9"'
    assert caplog.messages == [f"bearer token rejected: no signing key from {jwks.url}: {unknown}"]


@pytest.mark.anyio
async def test_a_kid_cannot_forge_a_log_line(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token(kid="k9\nforged")) is None
    [message] = caplog.messages
    assert "\n" not in message
    assert message.endswith('matches: "k9\\nforged"')


@pytest.mark.anyio
@pytest.mark.parametrize("beside", [[], [GOOD_JWK]], ids=["the only entry", "beside a good key"])
async def test_a_private_key_in_the_set_never_reaches_the_log(
    jwks: FakeJwks, beside: list[dict[str, object]], caplog: pytest.LogCaptureFixture
) -> None:
    private = cast("dict[str, str]", RSAAlgorithm.to_jwk(new_key(), as_dict=True))
    members = [private[name] for name in ("n", "d", "p", "q", "dp", "dq", "qi")]
    # Without `kty` PyJWT's own message for the entry carries the whole key.
    entry = {**without(cast("dict[str, object]", private), "kty"), "kid": "k0"}
    jwks.answer = answer_json({"keys": [entry, *beside]})
    with caplog.at_level(logging.DEBUG):
        verified = await make_verifier(jwks).verify_token(token())
    assert (verified is not None) is bool(beside)
    assert not any(member in caplog.text for member in members)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "bad_answer",
    [answer_json({"keys": []}), answer_raw(b"<html>login</html>"), hang_up],
    ids=["keys empty", "not JSON", "endpoint down"],
)
async def test_a_bad_refetch_keeps_the_cached_key_set(
    jwks: FakeJwks, clock: Clock, bad_answer: Answer
) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.answer = bad_answer
    clock.now += MIN_REFETCH + 1
    assert await verifier.verify_token(token(key=new_key(), kid="k2")) is None
    assert jwks.fetches == 2
    assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 2
