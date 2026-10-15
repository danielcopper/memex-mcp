"""Tokens from several providers: [auth] and an [[auth.extra_issuers]] entry, each with its own
key set, client ids and algorithms, through the verifier and the server the config builds."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx2
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from memex_mcp.auth import AuthentikTokenVerifier, IssuerVerifier, identity_from
from memex_mcp.config import Config
from memex_mcp.rights import ACCESS_DENIED
from memex_mcp.server import build_app, build_verifier
from memex_mcp.service import Memex
from tests.conftest import MARKER, Origin, git, make_config
from tests.test_auth import (
    CLIENT,
    ISSUER,
    KEY,
    FakeJwks,
    SigningKey,
    claims,
    new_key,
    warnings_in,
)
from tests.test_server import BASE, call, rpc

# A provider dedicated to memex for another front end, in the same Authentik.
OTHER_ISSUER = "https://auth.example.org/application/o/memex-openwebui/"
OTHER_JWKS_PATH = "/application/o/memex-openwebui/jwks/"
OTHER_CLIENT = "memex-openwebui"
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())

# [auth] signs RS256 with kid k1, the extra issuer ES256 with kid e1.
ALGORITHM = {ISSUER: "RS256", OTHER_ISSUER: "ES256"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class Providers:
    """[auth]'s key set endpoint and the extra issuer's, and the config naming both."""

    auth: FakeJwks
    other: FakeJwks
    config: Config


@pytest.fixture
def providers(tmp_path: Path, origin: Origin) -> Iterator[Providers]:
    with (
        FakeJwks((KEY, "k1")) as auth,
        FakeJwks((OTHER_KEY, "e1"), path=OTHER_JWKS_PATH) as other,
    ):
        config = make_config(
            tmp_path,
            origin,
            auth={
                "jwks_url": auth.url,
                "algorithms": ["RS256"],
                "timeout_seconds": 1.0,
                "extra_issuers": [
                    {
                        "issuer": OTHER_ISSUER,
                        "jwks_url": other.url,
                        "client_ids": [OTHER_CLIENT],
                        "algorithms": ["ES256"],
                    }
                ],
            },
        )
        yield Providers(auth, other, config)


def signed(key: SigningKey = KEY, kid: str = "k1", alg: str = "RS256", **overrides: object) -> str:
    return jwt.encode(claims(**overrides), key, algorithm=alg, headers={"kid": kid})


def other_token(**overrides: object) -> str:
    """A token as the extra issuer's provider issues it for its own client."""
    base: dict[str, object] = {"iss": OTHER_ISSUER, "aud": OTHER_CLIENT, "azp": OTHER_CLIENT}
    return signed(OTHER_KEY, "e1", "ES256", **{**base, **overrides})


def raw_signed(payload: dict[str, object], key: SigningKey = KEY, kid: str = "k1") -> str:
    """Signed as JWS: PyJWT's encode refuses an `iss` that is not a string."""
    alg = "ES256" if isinstance(key, ec.EllipticCurvePrivateKey) else "RS256"
    return jwt.PyJWS().encode(
        json.dumps(payload).encode(), key, algorithm=alg, headers={"kid": kid}
    )


@pytest.mark.anyio
async def test_a_token_of_each_issuer_yields_the_same_identity(providers: Providers) -> None:
    verifier = build_verifier(providers.config)
    ours = await verifier.verify_token(signed())
    theirs = await verifier.verify_token(other_token())
    assert ours is not None
    assert theirs is not None
    assert identity_from(ours).username == identity_from(theirs).username == "alice"
    assert identity_from(ours).groups == identity_from(theirs).groups
    assert identity_from(ours).client == CLIENT
    assert identity_from(theirs).client == OTHER_CLIENT


# Client ids allowed for one issuer, in a token of the other: never a union.
MIXED_CLIENTS: dict[str, dict[str, object]] = {
    "[auth] token for the extra client": {"aud": OTHER_CLIENT, "azp": OTHER_CLIENT},
    "[auth] token, audience ours, azp the extra client": {"azp": OTHER_CLIENT},
    "[auth] token, azp ours, audience the extra client": {"aud": OTHER_CLIENT},
}
OTHER_MIXED: dict[str, dict[str, object]] = {
    "extra token for [auth]'s client": {"aud": CLIENT, "azp": CLIENT},
    "extra token, audience its own, azp [auth]'s client": {"azp": CLIENT},
    "extra token, azp its own, audience [auth]'s client": {"aud": CLIENT},
}


@pytest.mark.anyio
@pytest.mark.parametrize("overrides", MIXED_CLIENTS.values(), ids=list(MIXED_CLIENTS))
async def test_an_auth_token_for_the_extra_client_is_refused(
    providers: Providers, overrides: dict[str, object]
) -> None:
    bearer = jwt.encode(claims(**overrides), KEY, algorithm="RS256", headers={"kid": "k1"})
    assert await build_verifier(providers.config).verify_token(bearer) is None


@pytest.mark.anyio
@pytest.mark.parametrize("overrides", OTHER_MIXED.values(), ids=list(OTHER_MIXED))
async def test_an_extra_token_for_the_auth_client_is_refused(
    providers: Providers, overrides: dict[str, object]
) -> None:
    assert await build_verifier(providers.config).verify_token(other_token(**overrides)) is None


@pytest.mark.anyio
async def test_an_algorithm_allowed_only_for_the_other_issuer_is_refused(
    providers: Providers, caplog: pytest.LogCaptureFixture
) -> None:
    """Each key set also lists a key for the other algorithm; only the issuer's own list counts."""
    ec_key, rsa_key = ec.generate_private_key(ec.SECP256R1()), new_key()
    providers.auth.keys.append((ec_key, "e9"))
    providers.other.keys.append((rsa_key, "k9"))
    verifier = build_verifier(providers.config)
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await verifier.verify_token(signed(ec_key, "e9", "ES256")) is None
        theirs = signed(rsa_key, "k9", iss=OTHER_ISSUER, aud=OTHER_CLIENT, azp=OTHER_CLIENT)
        assert await verifier.verify_token(theirs) is None
    assert caplog.messages == [
        "bearer token rejected: algorithm 'ES256' not allowed",
        "bearer token rejected: algorithm 'RS256' not allowed",
    ]


UNKNOWN_ISSUERS: dict[str, tuple[object, str]] = {
    "a third provider": (
        "https://auth.example.org/application/o/openwebui/",
        "'https://auth.example.org/application/o/openwebui/'",
    ),
    "the global issuer": ("https://auth.example.org/", "'https://auth.example.org/'"),
    "without the trailing slash": (OTHER_ISSUER.rstrip("/"), repr(OTHER_ISSUER.rstrip("/"))),
    "with a line break": (f"{OTHER_ISSUER}\nforged", repr(f"{OTHER_ISSUER}\nforged")),
    "not a string": ([ISSUER], "a list"),
    "missing": (None, "null"),
}
EXPECTED = (
    f"expected {ISSUER!r} or {OTHER_ISSUER!r} (a token for another application, "
    + "or a wrong issuer in [auth] or [[auth.extra_issuers]] entry 1)"
)


@pytest.mark.anyio
@pytest.mark.parametrize(("iss", "shown"), UNKNOWN_ISSUERS.values(), ids=list(UNKNOWN_ISSUERS))
@pytest.mark.parametrize("key", [KEY, OTHER_KEY], ids=["auth's key", "the extra issuer's key"])
async def test_an_unknown_issuer_is_refused_before_any_fetch(
    providers: Providers,
    iss: object,
    shown: str,
    key: SigningKey,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {k: v for k, v in {**claims(), "iss": iss}.items() if v is not None}
    bearer = raw_signed(payload, key, "k1" if key is KEY else "e1")
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        assert await build_verifier(providers.config).verify_token(bearer) is None
    assert caplog.messages == [f"bearer token rejected: issuer {shown}, {EXPECTED}"]
    assert warnings_in(caplog) == caplog.messages
    assert providers.auth.fetches == providers.other.fetches == 0
    assert bearer not in caplog.text


@pytest.mark.anyio
async def test_without_extra_issuers_the_line_names_auth_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with FakeJwks((KEY, "k1")) as jwks:
        config = make_config(tmp_path, auth={"jwks_url": jwks.url})
        with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
            assert await build_verifier(config).verify_token(signed(iss=OTHER_ISSUER)) is None
        assert jwks.fetches == 0
    assert caplog.messages == [
        f"bearer token rejected: issuer {OTHER_ISSUER!r}, expected {ISSUER!r} "
        + "(a token for another application, or a wrong issuer in [auth])"
    ]


@pytest.mark.anyio
async def test_garbage_is_refused_by_the_dispatcher_without_a_fetch(
    providers: Providers, caplog: pytest.LogCaptureFixture
) -> None:
    verifier = build_verifier(providers.config)
    with caplog.at_level(logging.INFO, logger="memex_mcp.auth"):
        for value in ("", "not-a-jwt", "a.b.c", "Bearer xyz"):
            assert await verifier.verify_token(value) is None
    assert caplog.messages == ["bearer token rejected: not a JWT"] * 4
    assert providers.auth.fetches == providers.other.fetches == 0


@pytest.mark.anyio
async def test_a_token_naming_auth_but_signed_with_the_extra_key_is_refused(
    providers: Providers,
) -> None:
    """Checked against [auth]'s key set only: the extra issuer's is never fetched for it."""
    rsa_key = new_key()
    providers.other.keys.append((rsa_key, "b1"))
    verifier = build_verifier(providers.config)
    assert await verifier.verify_token(signed(rsa_key, "b1")) is None  # iss and aud are [auth]'s
    assert await verifier.verify_token(signed(OTHER_KEY, "e1", "ES256")) is None
    assert providers.auth.fetches == 1
    assert providers.other.fetches == 0


@pytest.mark.anyio
async def test_the_same_kid_with_different_keys_stays_with_its_issuer(
    providers: Providers, tmp_path: Path
) -> None:
    """Both providers name their key k1; a token is checked only against its issuer's k1."""
    other_rsa = new_key()
    providers.other.keys = [(other_rsa, "k1")]
    extra = providers.config.auth.extra_issuers[0]
    config = make_config(
        tmp_path,
        auth={
            "jwks_url": providers.auth.url,
            "algorithms": ["RS256"],
            "extra_issuers": [
                {
                    "issuer": extra.issuer,
                    "jwks_url": extra.jwks_url,
                    "client_ids": list(extra.client_ids),
                    "algorithms": ["RS256"],
                }
            ],
        },
    )
    verifier = build_verifier(config)
    theirs = {"iss": OTHER_ISSUER, "aud": OTHER_CLIENT, "azp": OTHER_CLIENT}
    assert await verifier.verify_token(signed()) is not None
    assert await verifier.verify_token(signed(other_rsa, **theirs)) is not None
    assert await verifier.verify_token(signed(other_rsa)) is None
    assert await verifier.verify_token(signed(KEY, **theirs)) is None


@pytest.mark.anyio
async def test_one_key_set_down_leaves_the_other_issuer_working(
    providers: Providers, caplog: pytest.LogCaptureFixture
) -> None:
    providers.other.down = True
    verifier = build_verifier(providers.config)
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await verifier.verify_token(other_token()) is None
        assert await verifier.verify_token(signed()) is not None
    assert providers.other.fetches == 1
    [warning] = warnings_in(caplog)
    assert warning.startswith(f"cannot load the key set from {providers.other.url}: ")


@pytest.mark.anyio
async def test_a_hint_names_the_extra_issuer_entry(
    providers: Providers, caplog: pytest.LogCaptureFixture
) -> None:
    providers.other.answer = lambda handler: handler.send_error(404)
    with caplog.at_level(logging.WARNING, logger="memex_mcp.auth"):
        assert await build_verifier(providers.config).verify_token(other_token()) is None
    [warning] = warnings_in(caplog)
    assert (
        "; check [[auth.extra_issuers]] entry 1 jwks_url, by default the issuer's path plus jwks/"
        in warning
    )


def test_two_verifiers_for_one_issuer_are_refused(providers: Providers) -> None:
    def verifier() -> AuthentikTokenVerifier:
        return AuthentikTokenVerifier(
            issuer=ISSUER,
            jwks_url=providers.auth.url,
            client_ids=(CLIENT,),
            algorithms=("RS256",),
            leeway_seconds=30,
            jwks_min_refetch_seconds=60,
            timeout_seconds=1.0,
        )

    with pytest.raises(ValueError, match="two verifiers for one issuer"):
        IssuerVerifier([verifier(), verifier()])


# -- through the server --------------------------------------------------


@asynccontextmanager
async def serving(providers: Providers) -> AsyncGenerator[tuple[httpx2.AsyncClient, Memex], None]:
    """The app the config builds, its verifier included, with its lifespan and its service."""
    memex = Memex.from_config(providers.config, embedder=None)
    app = build_app(providers.config, memex, run_loop=False)
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url=BASE) as http:
            yield http, memex


@pytest.mark.anyio
async def test_the_metadata_names_auth_first_then_each_extra_issuer(
    providers: Providers,
) -> None:
    async with serving(providers) as (client, _memex):
        for path in (
            "/.well-known/oauth-protected-resource",
            "/.well-known/oauth-protected-resource/mcp",
        ):
            metadata = cast("dict[str, object]", (await client.get(path)).json())
            assert metadata["authorization_servers"] == [ISSUER, OTHER_ISSUER]


@pytest.mark.anyio
async def test_without_extra_issuers_the_metadata_names_auth_alone(
    tmp_path: Path, origin: Origin
) -> None:
    config = make_config(tmp_path, origin)
    memex = Memex.from_config(config, embedder=None)
    app = build_app(config, memex, run_loop=False)
    async with app.router.lifespan_context(app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url=BASE) as client:
            metadata = cast(
                "dict[str, object]",
                (await client.get("/.well-known/oauth-protected-resource")).json(),
            )
    assert metadata["authorization_servers"] == [ISSUER]


@pytest.mark.anyio
async def test_refusals_answer_alike_whatever_the_reason(providers: Providers) -> None:
    refused = {
        "unknown issuer": signed(iss="https://auth.example.org/application/o/openwebui/"),
        "foreign client": other_token(aud=CLIENT, azp=CLIENT),
        "bad signature": other_token()[:-4] + "AAAA",
        "another key": signed(new_key()),
    }
    async with serving(providers) as (client, _memex):
        answers = {
            reason: await rpc(client, "tools/list", bearer=bearer)
            for reason, bearer in refused.items()
        }
    first = answers["unknown issuer"]
    assert first.status_code == 401
    for reason, answer in answers.items():
        assert answer.status_code == first.status_code, reason
        assert answer.headers["www-authenticate"] == first.headers["www-authenticate"], reason
        assert answer.content == first.content, reason


@pytest.mark.anyio
async def test_a_token_of_the_extra_issuer_reads_the_same_area(providers: Providers) -> None:
    async with serving(providers) as (client, _memex):
        own = await call(client, other_token(), "read", path="alice/memory/zebra.md")
        assert own["isError"] is False
        assert MARKER in cast("str", own["structuredContent"]["content"])
        foreign = await call(client, other_token(), "read", path="bob/memory/zebra.md")
        assert foreign["content"][0]["text"] == "no such note"


@pytest.mark.anyio
async def test_the_access_group_holds_for_the_extra_issuer(providers: Providers) -> None:
    async with serving(providers) as (client, _memex):
        outsider = other_token(groups=["household"])
        for tool, arguments in (
            ("read", {"path": "household/memory/wifi.md"}),
            ("areas", {}),
            ("append", {"path": "household/memory/wifi.md", "text": "more"}),
        ):
            result = await call(client, outsider, tool, **arguments)
            assert result["isError"] is True, tool
            assert result["content"][0]["text"] == ACCESS_DENIED, tool


@pytest.mark.anyio
async def test_each_write_names_the_client_it_came_through(
    providers: Providers, origin: Origin
) -> None:
    async with serving(providers) as (client, _memex):
        for bearer in (signed(), other_token()):
            result = await call(
                client, bearer, "append", path="alice/memory/zebra.md", text="one more line"
            )
            assert result["isError"] is False
    trailers = git(
        origin.bare, "log", "-2", "--format=%(trailers:key=Memex-Client,valueonly)", "main"
    )
    assert trailers.split() == [OTHER_CLIENT, CLIENT]
    assert (
        git(origin.bare, "log", "-2", "--format=%an|%cn", "main").split() == ["alice|memex-mcp"] * 2
    )
