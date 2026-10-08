# memex-mcp: the read-only MCP server over a memex clone.
# Runs as an unprivileged user with a read-only root filesystem: the clone and
# the index live on a volume (/data), git and Python only need a writable /tmp.

FROM python:3.12.15-slim-trixie AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY memex_mcp ./memex_mcp
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --disable-pip-version-check .

FROM python:3.12.15-slim-trixie
# git fetches the memex repository; openssh-client for ssh:// remotes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --no-create-home --home-dir /data --shell /usr/sbin/nologin memex \
    && mkdir -p /data \
    && chown memex:memex /data
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MEMEX_CONFIG=/config/config.toml
USER memex
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
ENTRYPOINT ["memex-mcp"]
