# Changelog

## [0.1.3](https://github.com/danielcopper/memex-mcp/compare/v0.1.2...v0.1.3) (2026-10-10)


### Bug Fixes

* make the auth and embedding log lines say what went wrong ([#15](https://github.com/danielcopper/memex-mcp/issues/15)) ([a03e984](https://github.com/danielcopper/memex-mcp/commit/a03e984e08ebac3245a2ded8c3a5d0d72a49763a))
* refuse config values that break the server at startup ([#13](https://github.com/danielcopper/memex-mcp/issues/13)) ([df869d6](https://github.com/danielcopper/memex-mcp/commit/df869d67eb7a63148463e2bcc17ae0544d2d7d8f))

## [0.1.2](https://github.com/danielcopper/memex-mcp/compare/v0.1.1...v0.1.2) (2026-10-09)


### Bug Fixes

* let cached jwks signing keys expire so revoked keys stop verifying ([#10](https://github.com/danielcopper/memex-mcp/issues/10)) ([441d3a1](https://github.com/danielcopper/memex-mcp/commit/441d3a1d00eb6b8e644c3313c257d202e90f6d28))

## [0.1.1](https://github.com/danielcopper/memex-mcp/compare/v0.1.0...v0.1.1) (2026-10-08)


### Bug Fixes

* refuse malformed key sets, embedder answers and config values with a clear error ([#8](https://github.com/danielcopper/memex-mcp/issues/8)) ([4e6f70e](https://github.com/danielcopper/memex-mcp/commit/4e6f70e8e58303c5d29b94ef9269873695c6b197))

## 0.1.0 (2026-10-08)


### Features

* serve a per-person markdown memory over mcp ([#1](https://github.com/danielcopper/memex-mcp/issues/1)) ([9822b66](https://github.com/danielcopper/memex-mcp/commit/9822b66ac893c49b235b6186795bef269f5996dc))


### Documentation

* **readme:** explain which signing key the authentik provider needs ([#4](https://github.com/danielcopper/memex-mcp/issues/4)) ([4a90f6c](https://github.com/danielcopper/memex-mcp/commit/4a90f6c0141129818e20e46c60b0330725cc8df3))
