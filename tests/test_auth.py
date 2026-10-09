"""Token validation against a local key and a JWKS endpoint on 127.0.0.1 (no network)."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import hmac
import inspect
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
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from jwt.algorithms import ECAlgorithm, OKPAlgorithm, RSAAlgorithm

import memex_mcp.auth
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


# PyJWT's wording; what follows it is Python's own, which changes between versions.
CONNECTION_ERROR = "PyJWKClientConnectionError: Fail to fetch data from the url, err: "


class FakeJwks:
    """The provider's JWKS endpoint, served on 127.0.0.1; counts fetches and can go down."""

    def __init__(self, *keys: tuple[rsa.RSAPrivateKey, str]) -> None:
        self.keys: list[tuple[rsa.RSAPrivateKey, str]] = list(keys)
        self.fetches: int = 0
        self.down: bool = False
        # Answers every fetch with this instead of the keys above.
        self.answer: Answer | None = None
        self.delay: float = 0.0
        # Runs while a fetch is being answered.
        self.on_fetch: Callable[[], None] = lambda: None
        jwks = self

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                assert self.path == JWKS_PATH
                jwks.fetches += 1
                jwks.on_fetch()
                time.sleep(jwks.delay)
                answer = jwks.answer or answer_json(
                    {"keys": [jwk_of(k, kid) for k, kid in jwks.keys]}
                )
                # The client may have given up waiting.
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    (hang_up if jwks.down else answer)(self)

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
    """The key set cache, the refetch cooldown and the wait after a failed fetch, on a
    clock the test moves.

    These are the modules that read ``time.monotonic`` for them (PyJWT's at
    2.15.1); only their name ``time`` is replaced, so any other use of it fails.
    """
    fake = Clock()
    for module in (jwt.jwks_client, jwt.jwk_set_cache, jwt.api_jwk, memex_mcp.auth):
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
    jwks: FakeJwks,
    algorithms: tuple[str, ...] = ("RS256", "ES256"),
    timeout: float = 1.0,
    min_refetch: int = MIN_REFETCH,
) -> AuthentikTokenVerifier:
    return AuthentikTokenVerifier(
        issuer=ISSUER,
        jwks_url=jwks.url,
        client_ids=(CLIENT,),
        algorithms=algorithms,
        leeway_seconds=30,
        jwks_min_refetch_seconds=min_refetch,
        timeout_seconds=timeout,
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


WRONG_ISSUERS: dict[str, tuple[object, str]] = {
    "without the trailing slash": (ISSUER.rstrip("/"), repr(ISSUER.rstrip("/"))),
    "with a line break": (f"{ISSUER}\nforged", repr(f"{ISSUER}\nforged")),
    "long": (ISSUER + "x" * 1000, repr(ISSUER + "x" * (200 - len(ISSUER)) + "…")),
    "not a string": ([ISSUER], "a list"),
}


@pytest.mark.anyio
@pytest.mark.parametrize(("iss", "shown"), WRONG_ISSUERS.values(), ids=list(WRONG_ISSUERS))
async def test_a_wrong_issuer_is_logged_at_warning_with_both_issuers(
    jwks: FakeJwks, iss: object, shown: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Signed as JWS: PyJWT's encode refuses an `iss` that is not a string.
    payload = json.dumps(claims(iss=iss)).encode()
    bearer = jwt.PyJWS().encode(payload, KEY, algorithm="RS256", headers={"kid": "k1"})
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(bearer) is None
    assert warnings_in(caplog) == [f"bearer token rejected: issuer {shown}, expected {ISSUER!r}"]
    assert bearer not in caplog.text


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


REJECTED_UNFETCHED = (
    "bearer token rejected: no signing key for kid 'k1': the key set could not be fetched"
)
ALL_REJECTED = "every token is rejected until a fetch succeeds"
NO_FETCH_FOR = f"no new fetch for {MIN_REFETCH} s"


def warnings_in(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.anyio
async def test_an_expired_key_set_is_not_used_while_the_endpoint_is_down(
    jwks: FakeJwks, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.down = True
    clock.now += JWKS_LIFESPAN_SECONDS + 1
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await verifier.verify_token(token()) is None
    [warning, rejected] = caplog.messages
    assert warning.startswith(f"cannot load the key set from {jwks.url}: {CONNECTION_ERROR}")
    assert warning.endswith(f"; {ALL_REJECTED}; {NO_FETCH_FOR}")
    assert rejected == REJECTED_UNFETCHED


@pytest.mark.anyio
async def test_a_failed_fetch_is_not_retried_before_the_cooldown(
    jwks: FakeJwks, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.down = True
    clock.now += JWKS_LIFESPAN_SECONDS + 1
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        for _ in range(5):
            assert await verifier.verify_token(token()) is None
        assert jwks.fetches == 2
        clock.now += MIN_REFETCH - 1
        assert await verifier.verify_token(token()) is None
        assert jwks.fetches == 2
        clock.now += 2
        assert await verifier.verify_token(token()) is None
        assert jwks.fetches == 3
    assert len(warnings_in(caplog)) == 2
    assert all(ALL_REJECTED in message for message in warnings_in(caplog))


@pytest.mark.anyio
async def test_the_wait_counts_from_the_failure(jwks: FakeJwks, clock: Clock) -> None:
    """A fetch that takes as long as the cooldown does not let the next one through."""
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    clock.now += JWKS_LIFESPAN_SECONDS + 1
    jwks.down = True

    def slow_fetch() -> None:
        clock.now += MIN_REFETCH

    jwks.on_fetch = slow_fetch
    assert await verifier.verify_token(token()) is None
    assert await verifier.verify_token(token()) is None
    assert jwks.fetches == 2


@pytest.mark.anyio
async def test_a_failed_fetch_never_extends_the_cached_set(jwks: FakeJwks, clock: Clock) -> None:
    """A wait after a failed fetch never serves a key set past its lifespan."""
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.down = True
    clock.now += JWKS_LIFESPAN_SECONDS - 1
    assert await verifier.verify_token(token(kid="k9")) is None  # a refetch, which fails
    assert jwks.fetches == 2
    clock.now += 2
    assert await verifier.verify_token(token()) is None
    assert jwks.fetches == 2


@pytest.mark.anyio
async def test_unknown_kids_while_the_endpoint_hangs_cost_one_fetch(
    jwks: FakeJwks, clock: Clock
) -> None:
    timeout = 0.2
    verifier = make_verifier(jwks, timeout=timeout)
    assert await verifier.verify_token(token()) is not None
    clock.now += MIN_REFETCH + 1
    jwks.delay = 2 * timeout
    waited: list[float] = []

    async def unknown_kid() -> None:
        assert await verifier.verify_token(token(kid=uuid.uuid4().hex)) is None

    async def cached_kid() -> None:
        start = time.monotonic()
        assert await verifier.verify_token(token()) is not None
        waited.append(time.monotonic() - start)

    async with anyio.create_task_group() as group:
        for _ in range(5):
            group.start_soon(unknown_kid)
        group.start_soon(cached_kid)
    assert jwks.fetches == 2
    [cached_wait] = waited
    assert cached_wait < 4 * timeout


@pytest.mark.anyio
async def test_a_zero_cooldown_still_waits_the_timeout_after_a_failure(
    jwks: FakeJwks, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    timeout = 0.2
    verifier = make_verifier(jwks, timeout=timeout, min_refetch=0)
    assert await verifier.verify_token(token()) is not None
    clock.now += JWKS_LIFESPAN_SECONDS + 1
    jwks.delay = 2 * timeout
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        for _ in range(4):
            assert await verifier.verify_token(token()) is None
        assert jwks.fetches == 2
        clock.now += timeout
        assert await verifier.verify_token(token()) is None
        assert jwks.fetches == 3
    assert len(warnings_in(caplog)) == 2
    assert all(message.endswith(f"no new fetch for {timeout} s") for message in warnings_in(caplog))


@pytest.mark.anyio
async def test_a_cached_kid_never_waits_for_a_fetch(jwks: FakeJwks, clock: Clock) -> None:
    timeout = 0.6
    verifier = make_verifier(jwks, timeout=timeout)
    assert await verifier.verify_token(token()) is not None
    clock.now += MIN_REFETCH + 1
    jwks.delay = 2 * timeout
    waited: list[float] = []

    async def cached_kid() -> None:
        start = time.monotonic()
        assert await verifier.verify_token(token()) is not None
        waited.append(time.monotonic() - start)

    async with anyio.create_task_group() as group:
        group.start_soon(verifier.verify_token, token(kid="k9"))
        await anyio.sleep(timeout / 6)  # the refetch for k9 is waiting on the endpoint
        group.start_soon(cached_kid)
    assert jwks.fetches == 2
    [cached_wait] = waited
    assert cached_wait < timeout / 4


@pytest.mark.anyio
async def test_a_cached_key_is_served_until_the_set_expires_and_not_after(
    jwks: FakeJwks, clock: Clock
) -> None:
    """The fast path for cached keys stops exactly where the cache does."""
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.down = True
    clock.now += JWKS_LIFESPAN_SECONDS
    assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 1
    clock.now += 0.001
    assert await verifier.verify_token(token()) is None
    assert jwks.fetches == 2


@pytest.mark.anyio
async def test_a_slow_endpoint_times_out(jwks: FakeJwks, caplog: pytest.LogCaptureFixture) -> None:
    timeout = 0.2
    jwks.delay = 2 * timeout
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await make_verifier(jwks, timeout=timeout).verify_token(token()) is None
    [warning] = warnings_in(caplog)
    assert warning.startswith(f"cannot load the key set from {jwks.url}: {CONNECTION_ERROR}")
    assert "timed out" in warning


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
async def test_jwks_endpoint_down_rejects_and_recovers(
    jwks: FakeJwks, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    jwks.down = True
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is None
    jwks.down = False
    assert await verifier.verify_token(token()) is None  # still waiting after the failure
    assert jwks.fetches == 1
    clock.now += MIN_REFETCH
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 2
    assert caplog.messages[-1] == (
        f"loaded the key set from {jwks.url} after 1 failed fetch(es); signing keys 'k1'"
    )


@pytest.mark.anyio
async def test_a_changed_key_set_is_logged_once(
    jwks: FakeJwks, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    verifier = make_verifier(jwks)
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await verifier.verify_token(token()) is not None
        jwks.keys.append((new_key(), "k2"))
        for _ in range(2):
            clock.now += JWKS_LIFESPAN_SECONDS + 1
            assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 3
    assert caplog.messages == [
        f"loaded the key set from {jwks.url}; signing keys 'k1'",
        f"loaded the key set from {jwks.url}; signing keys 'k1', 'k2'",
    ]


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
async def test_encryption_keys_and_keys_without_alg_are_handled(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    """Authentik lists an encryption key (use "enc") beside the signing key when one is set."""
    enc_key = new_key()
    signing = jwk_of(KEY, "k1")
    del signing["alg"]  # PyJWT derives RS256 from an RSA key
    encryption = {**jwk_of(enc_key, "e1"), "use": "enc", "alg": "RSA-OAEP-256"}
    # Without `alg` PyJWT would make an RS256 key of it; `use` keeps it out.
    encryption_without_alg = {**without(jwk_of(enc_key, "e2"), "alg"), "use": "enc"}
    jwks.answer = answer_json({"keys": [encryption, encryption_without_alg, signing]})
    verifier = make_verifier(jwks, algorithms=("RS256",))
    with caplog.at_level(logging.DEBUG, logger="memex_mcp.auth"):
        assert await verifier.verify_token(token()) is not None
        assert await verifier.verify_token(token(key=enc_key, kid="e1")) is None
        assert await verifier.verify_token(token(key=enc_key, kid="e2")) is None
    assert warnings_in(caplog) == []
    assert caplog.messages[:2] == [
        "key set entry 0 (kid 'e1', kty 'RSA', alg 'RSA-OAEP-256') is an encryption key; not used",
        "key set entry 1 (kid 'e2', kty 'RSA', alg null) is an encryption key; not used",
    ]


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
async def test_a_malformed_entry_does_not_hide_the_good_key(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    jwks.answer = answer_json(
        {
            "keys": [
                5,
                {**GOOD_JWK, "kid": ["k1"]},
                {**GOOD_JWK, "alg": 5, "kid": "k2"},
                {**without(GOOD_JWK, "kty"), "kid": "k3"},
                {**GOOD_JWK, "use": "foo", "kid": "k4"},
                without(GOOD_JWK, "kid"),
                GOOD_JWK,
            ]
        }
    )
    verifier = make_verifier(jwks)
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        for _ in range(2):
            assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 1
    assert warnings_in(caplog) == [
        "key set entry 1 has kid a list, not a string; no token can name it",
        "key set entry 2 (kid 'k2', kty 'RSA', alg a number) is unusable: PyJWKError; not used",
        "key set entry 3 (kid 'k3', kty null, alg 'RS256') is unusable: InvalidKeyError; not used",
        "key set entry 4 (kid 'k4', kty 'RSA', alg 'RS256') has use 'foo', not 'sig'; not used",
        "key set entry 5 (kid null, kty 'RSA', alg 'RS256') has no kid; not used",
        "key set entry 0 is a number, not an object; not used",
    ]


@pytest.mark.anyio
async def test_unused_entries_are_logged_when_the_set_changes(
    jwks: FakeJwks, clock: Clock, caplog: pytest.LogCaptureFixture
) -> None:
    bad = {**GOOD_JWK, "use": "foo", "kid": "k4"}
    jwks.answer = answer_json({"keys": [bad, GOOD_JWK]})
    verifier = make_verifier(jwks)
    line = "key set entry 0 (kid 'k4', kty 'RSA', alg 'RS256') has use 'foo', not 'sig'; not used"
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        for _ in range(2):
            assert await verifier.verify_token(token()) is not None
            clock.now += JWKS_LIFESPAN_SECONDS + 1
        jwks.answer = answer_json({"keys": [GOOD_JWK, bad]})
        assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 3
    assert warnings_in(caplog) == [line, line.replace("entry 0", "entry 1")]


@pytest.mark.anyio
async def test_unused_entries_are_logged_up_to_ten(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    jwks.answer = answer_json({"keys": [*range(15), GOOD_JWK]})
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token()) is not None
    assert warnings_in(caplog) == [
        *(f"key set entry {index} is a number, not an object; not used" for index in range(10)),
        "and 5 more key set entries not used",
    ]


UNUSABLE_K0 = {**without(GOOD_JWK, "kty"), "kid": "k0"}
UNUSABLE_K0_LINE = "(kid 'k0', kty null, alg 'RS256') is unusable: InvalidKeyError; not used"


@pytest.mark.anyio
async def test_lines_about_objects_come_before_other_entries(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    use_foo = {**GOOD_JWK, "use": "foo", "kid": "k4"}
    jwks.answer = answer_json({"keys": [*range(12), UNUSABLE_K0, use_foo, GOOD_JWK, GOOD_JWK]})
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token()) is not None
    assert warnings_in(caplog) == [
        f"key set entry 12 {UNUSABLE_K0_LINE}",
        "key set entry 13 (kid 'k4', kty 'RSA', alg 'RS256') has use 'foo', not 'sig'; not used",
        "key set entry 15 repeats kid 'k1'; PyJWT uses the first",
        *(f"key set entry {index} is a number, not an object; not used" for index in range(7)),
        "and 5 more key set entries not used",
    ]


@pytest.mark.anyio
async def test_debug_lines_never_take_the_place_of_a_warning(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    encryption_keys = [{**ENC_JWK, "kid": f"e{index}"} for index in range(12)]
    jwks.answer = answer_json({"keys": [*encryption_keys, UNUSABLE_K0, GOOD_JWK]})
    with caplog.at_level(logging.DEBUG, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token()) is not None
    about = "(kid 'e{0}', kty 'RSA', alg 'RSA-OAEP-256') is an encryption key"
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, f"key set entry 12 {UNUSABLE_K0_LINE}"),
        *(
            (logging.DEBUG, f"key set entry {index} {about.format(index)}; not used")
            for index in range(10)
        ),
        (logging.DEBUG, "and 2 more key set entries not used"),
        (logging.INFO, f"loaded the key set from {jwks.url}; signing keys 'k1'"),
    ]


@pytest.mark.anyio
async def test_a_repeated_kid_is_reported(jwks: FakeJwks, caplog: pytest.LogCaptureFixture) -> None:
    other = jwk_of(new_key(), "k1")
    jwks.answer = answer_json({"keys": [GOOD_JWK, other]})
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token()) is not None
        assert await make_verifier(jwks).verify_token(token(key=new_key())) is None
    assert warnings_in(caplog) == ["key set entry 1 repeats kid 'k1'; PyJWT uses the first"] * 2


VALUES: dict[str, object] = {
    "null": None,
    "integer": 5,
    "float": 1.5,
    "true": True,
    "list": [1],
    "object": {},
    "empty": "",
    "not base64": "!!!",
}
MISSING = "missing"


def fuzz_bases() -> dict[str, dict[str, object]]:
    """One key of each type PyJWT knows, kid k0, as a provider would list it."""
    rsa_jwk = RSAAlgorithm.to_jwk(new_key().public_key(), as_dict=True)
    ec_jwk = ECAlgorithm.to_jwk(ec.generate_private_key(ec.SECP256R1()).public_key(), as_dict=True)
    okp_key = ed25519.Ed25519PrivateKey.generate().public_key()
    okp_jwk = OKPAlgorithm.to_jwk(okp_key, as_dict=True)
    oct_jwk = {"kty": "oct", "k": "c2VjcmV0"}
    entries = {
        "RSA": (rsa_jwk, "RS256"),
        "EC": (ec_jwk, "ES256"),
        "OKP": (okp_jwk, "EdDSA"),
        "oct": (oct_jwk, "HS256"),
    }
    return {
        kty: {**jwk, "kid": "k0", "alg": alg, "use": "sig"} for kty, (jwk, alg) in entries.items()
    }


FUZZ_BASES = fuzz_bases()
FUZZ_CASES = [
    (kty, member, value)
    for kty, base in FUZZ_BASES.items()
    for member in base
    for value in [*VALUES, MISSING]
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("kty", "member", "value"),
    FUZZ_CASES,
    ids=[f"{kty}-{member}-{value}" for kty, member, value in FUZZ_CASES],
)
async def test_a_malformed_member_never_escapes(
    jwks: FakeJwks, kty: str, member: str, value: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Every member of every key type, broken beside a good key: rejected or verified, logged."""
    entry = without(FUZZ_BASES[kty], member)
    if value != MISSING:
        entry[member] = VALUES[value]
    jwks.answer = answer_json({"keys": [entry, GOOD_JWK]})
    with caplog.at_level(logging.DEBUG, logger="memex_mcp.auth"):
        verified = await make_verifier(jwks).verify_token(token())
    warnings = warnings_in(caplog)
    assert len(warnings) <= 1
    # A rejection always says why.
    assert verified is not None or len(warnings) == 1


# The drift test compares auth.py's signing-key filter with PyJWT's directly, to catch drift on a
# PyJWT bump; it needs these private names.
ProviderJwks = memex_mcp.auth._ProviderJwks  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
ENTRY_ERRORS = memex_mcp.auth._ENTRY_ERRORS  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
review = memex_mcp.auth._review  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("kty", "member", "value"),
    FUZZ_CASES,
    ids=[f"{kty}-{member}-{value}" for kty, member, value in FUZZ_CASES],
)
def test_the_signing_keys_are_the_ones_pyjwt_picks(kty: str, member: str, value: str) -> None:
    """The kids the fetch reports and the keys served from the cache are PyJWT's signing keys.

    The fast path and the review of a fetched set filter the set themselves;
    this ties both to PyJWT's own filter, so a change of it on a pin bump fails here.
    """
    entry = without(FUZZ_BASES[kty], member)
    if value != MISSING:
        entry[member] = VALUES[value]
    document: dict[str, object] = {"keys": [entry, GOOD_JWK]}
    client = ProviderJwks("http://127.0.0.1/jwks/", timeout=1.0, cooldown=1.0)
    assert client.jwk_set_cache is not None
    try:
        client.jwk_set_cache.put(document)
    except ENTRY_ERRORS:
        return  # PyJWT refuses the whole set; a fetch of it fails
    # Served from the cache just filled, without a fetch.
    pyjwt_keys = client.get_signing_keys()
    entries = cast("list[object]", document["keys"])
    kids, _unusable, _notes = review(entries)
    assert list(kids) == [key.key_id for key in pyjwt_keys]
    for kid in ("k0", "k1", "k9"):
        assert client.cached_key(kid) is client.match_kid(pyjwt_keys, kid)


CACHE_ATTRIBUTES = frozenset({"jwk_set_cache", "jwk_set_with_timestamp"})


def assigns_none_to_the_cache(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, ast.AnnAssign):
        targets, value = [node.target], node.value
    else:
        return False
    return (
        isinstance(value, ast.Constant)
        and value.value is None
        and any(isinstance(t, ast.Attribute) and t.attr in CACHE_ATTRIBUTES for t in targets)
    )


def test_pyjwt_never_clears_the_cached_key_set() -> None:
    """The cache is read without the lock, which holds while PyJWT only replaces the set.

    Clearing it would empty the cache while a fetch runs, and a failed fetch would
    drop a set that is still valid; this fails on a pin bump that adds either.

    It sees, in ``jwt/jwks_client.py``, the first argument of every ``.put(`` call
    and every assignment of None to an attribute named ``jwk_set_cache`` or
    ``jwk_set_with_timestamp`` outside ``__init__``. It does not see None passed
    through a variable, ``setattr`` or ``del``, other methods of the cache, or code
    in any other module.
    """
    tree = ast.parse(inspect.getsource(jwt.jwks_client))
    puts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "put"
    ]
    assert puts  # the check below would hold for no call at all
    assert [ast.unparse(node.args[0]) for node in puts] == ["jwk_set"] * len(puts)
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    in_init = [
        node
        for method in methods
        if method.name == "__init__"
        for node in ast.walk(method)
        if assigns_none_to_the_cache(node)
    ]
    # __init__ starts without a cache, in both forms: the match sees each.
    assert {type(node) for node in in_init} == {ast.Assign, ast.AnnAssign}
    assert [
        ast.unparse(node)
        for method in methods
        if method.name != "__init__"
        for node in ast.walk(method)
        if assigns_none_to_the_cache(node)
    ] == []


HINT = "; is [auth] jwks_url the provider's JWKS endpoint?"
REFUSED = (
    '; an entry makes PyJWT refuse the whole set (alg "none", an alg that is not a string, '
    + "or an oct key without k)"
)
NO_USABLE_KEYS = "PyJWKSetError: The JWK Set did not contain any usable keys."
NOT_FOUND = "; check [auth] jwks_url, by default the issuer's path plus jwks/"

# What the endpoint answers -> how the cause the warning names begins (its
# type, plus PyJWT's message, but never Python's wording, which changes between
# versions), and what the warning adds: the jwks_url hint when the endpoint
# answered, but not with a usable key set; what PyJWT refused when one entry
# fails the set; where the path comes from for a 404; nothing for any other
# connection error.
FETCH_FAILURES: dict[str, tuple[Answer, str, str]] = {
    "not JSON": (answer_raw(b"<html>login</html>"), "JSONDecodeError: ", HINT),
    "not UTF-8": (answer_raw(b"\xff\xfe\x00"), "UnicodeDecodeError: ", HINT),
    # A closed document, so that nothing but the depth can fail it.
    "nested too deeply": (
        answer_raw(b"[" * 100_000 + b"]" * 100_000),
        "RecursionError: ",
        HINT,
    ),
    "not an object": (
        answer_json([GOOD_JWK]),
        "PyJWKClientError: The JWKS endpoint did not return a JSON object",
        HINT,
    ),
    "no keys member": (
        answer_json({}),
        "PyJWKSetError: The JWK Set did not contain any keys",
        HINT,
    ),
    "keys a number": (answer_json({"keys": 5}), "PyJWKSetError: Invalid JWK Set value", HINT),
    "keys an object": (
        answer_json({"keys": GOOD_JWK}),
        "PyJWKSetError: Invalid JWK Set value",
        HINT,
    ),
    "no usable key": (answer_json({"keys": [without(GOOD_JWK, "kty")]}), NO_USABLE_KEYS, HINT),
    "entries not objects": (answer_json({"keys": [5, None, "k1"]}), NO_USABLE_KEYS, HINT),
    "encryption key only": (answer_json({"keys": [ENC_JWK]}), NO_USABLE_KEYS, HINT),
    "alg none beside a good key": (
        answer_json({"keys": [{**GOOD_JWK, "alg": "none", "kid": "k0"}, GOOD_JWK]}),
        "NotImplementedError: no detail",
        REFUSED,
    ),
    "alg a list beside a good key": (
        answer_json({"keys": [{**GOOD_JWK, "alg": ["RS256"], "kid": "k0"}, GOOD_JWK]}),
        "TypeError: ",
        REFUSED,
    ),
    "oct key without k beside a good key": (
        answer_json({"keys": [{"kty": "oct", "kid": "k0"}, GOOD_JWK]}),
        "KeyError: 'k'",
        REFUSED,
    ),
    "not found": (answer_raw(b"", status=404), CONNECTION_ERROR, NOT_FOUND),
    "redirect": (
        answer_raw(b"", status=302),
        CONNECTION_ERROR,
        "",
    ),
    "endpoint down": (hang_up, CONNECTION_ERROR, ""),
    "reset while answering": (reset_mid_answer, "ConnectionResetError: ", ""),
}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("answer", "cause", "explanation"), FETCH_FAILURES.values(), ids=list(FETCH_FAILURES)
)
async def test_a_failed_fetch_rejects_the_token_with_one_warning(
    jwks: FakeJwks,
    answer: Answer,
    cause: str,
    explanation: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    jwks.answer = answer
    bearer = token()
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(bearer) is None
    [warning] = warnings_in(caplog)
    assert warning.startswith(f"cannot load the key set from {jwks.url}: {cause}")
    assert warning.count(jwks.url) == 1
    assert (HINT in warning) is (explanation == HINT)
    assert (REFUSED in warning) is (explanation == REFUSED)
    assert (NOT_FOUND in warning) is (explanation == NOT_FOUND)
    assert warning.endswith(f"{explanation}; {ALL_REJECTED}; {NO_FETCH_FOR}")
    assert caplog.messages[-1] == REJECTED_UNFETCHED
    assert bearer not in caplog.text


def not_found_with_a_forged_reason(handler: BaseHTTPRequestHandler) -> None:
    """A 404 whose reason phrase carries a terminal escape and thousands of characters."""
    handler.send_response(404, "\x1b[31m" + "x" * 3000)
    handler.send_header("content-length", "0")
    handler.end_headers()


@pytest.mark.anyio
async def test_the_cause_of_a_failed_fetch_is_escaped_and_cut(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    jwks.answer = not_found_with_a_forged_reason
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token()) is None
    [warning] = warnings_in(caplog)
    cause = f'{CONNECTION_ERROR}"HTTP Error 404: \\x1b[31m'
    assert warning.startswith(f"cannot load the key set from {jwks.url}: {cause}")
    assert "\x1b" not in warning
    shown = 200 - len('Fail to fetch data from the url, err: "HTTP Error 404: \x1b[31m')
    assert f"{'x' * shown}…{NOT_FOUND}" in warning
    assert "x" * (shown + 1) not in warning


@pytest.mark.anyio
@pytest.mark.parametrize(
    "entry",
    [{**GOOD_JWK, "use": "enc"}, without(GOOD_JWK, "kid")],
    ids=["only a key marked enc", "only a key without kid"],
)
async def test_a_key_set_without_signing_keys_is_reported(
    jwks: FakeJwks, entry: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    jwks.answer = answer_json({"keys": [entry]})
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token()) is None
    assert warnings_in(caplog)[-1] == (
        f"the key set from {jwks.url} has no signing key: every token is rejected for "
        + f"{JWKS_LIFESPAN_SECONDS} s; check the provider's signing key"
    )
    assert caplog.messages[-1] == (
        "bearer token rejected: no signing key for kid 'k1': the key set has no signing key"
    )


@pytest.mark.anyio
async def test_an_unknown_kid_is_logged_at_info(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token(kid="k9")) is None
    assert caplog.messages == [
        f"loaded the key set from {jwks.url}; signing keys 'k1'",
        "bearer token rejected: no signing key for kid 'k9'",
    ]
    assert warnings_in(caplog) == []


USE_FOO_K0 = {**GOOD_JWK, "use": "foo", "kid": "k0"}
ENC_K0 = {**ENC_JWK, "kid": "k0"}
NOT_USED = (
    "bearer token rejected: no signing key for kid 'k0': the set names the kid, but its entry"
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("entry", "why"),
    [
        (UNUSABLE_K0, "is unusable: InvalidKeyError"),
        (USE_FOO_K0, "has use 'foo', not 'sig'"),
        (ENC_K0, "is an encryption key"),
    ],
    ids=["unusable", "use foo", "encryption key"],
)
async def test_a_kid_whose_entry_is_not_used_says_so(
    jwks: FakeJwks, entry: dict[str, object], why: str, caplog: pytest.LogCaptureFixture
) -> None:
    jwks.answer = answer_json({"keys": [entry, GOOD_JWK]})
    verifier = make_verifier(jwks)
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        # The reason comes with every token that names the kid, not only the first.
        for _ in range(2):
            assert await verifier.verify_token(token(kid="k0")) is None
        assert await verifier.verify_token(token(kid="k9")) is None
    rejections = [m for m in caplog.messages if m.startswith("bearer token rejected")]
    assert rejections == [f"{NOT_USED} {why}"] * 2 + [
        "bearer token rejected: no signing key for kid 'k9'"
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("later", "reason"),
    [
        ([GOOD_JWK], ""),
        ([ENC_K0, GOOD_JWK], ": the set names the kid, but its entry is an encryption key"),
    ],
    ids=["entry gone", "entry unused for another reason"],
)
async def test_a_refetch_replaces_what_is_known_about_unused_entries(
    jwks: FakeJwks,
    clock: Clock,
    later: list[dict[str, object]],
    reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    jwks.answer = answer_json({"keys": [USE_FOO_K0, GOOD_JWK]})
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token(kid="k0")) is None
    jwks.answer = answer_json({"keys": later})
    clock.now += JWKS_LIFESPAN_SECONDS + 1
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await verifier.verify_token(token(kid="k0")) is None
    assert jwks.fetches == 2
    assert caplog.messages[-1] == f"bearer token rejected: no signing key for kid 'k0'{reason}"


@pytest.mark.anyio
async def test_a_long_kid_is_clipped(jwks: FakeJwks, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token(kid="x" * 20_000)) is None
    assert caplog.messages[-1] == f"bearer token rejected: no signing key for kid '{'x' * 64}…'"


@pytest.mark.anyio
async def test_a_kid_cannot_forge_a_log_line(
    jwks: FakeJwks, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await make_verifier(jwks).verify_token(token(kid="k9\nforged")) is None
    assert not any("\n" in message for message in caplog.messages)
    assert caplog.messages[-1].endswith("kid 'k9\\nforged'")


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
    jwks: FakeJwks, clock: Clock, bad_answer: Answer, caplog: pytest.LogCaptureFixture
) -> None:
    verifier = make_verifier(jwks)
    assert await verifier.verify_token(token()) is not None
    jwks.answer = bad_answer
    clock.now += MIN_REFETCH + 1
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await verifier.verify_token(token(key=new_key(), kid="k2")) is None
    assert jwks.fetches == 2
    assert await verifier.verify_token(token()) is not None
    assert jwks.fetches == 2
    left = JWKS_LIFESPAN_SECONDS - MIN_REFETCH - 1
    [warning] = warnings_in(caplog)
    assert warning.endswith(
        f"; the cached key set stays in use for another {left} s; {NO_FETCH_FOR}"
    )
