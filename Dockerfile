# syntax=docker/dockerfile:1

# --- builder --------------------------------------------------------------------------
# uv resolves and installs from the locked dependency set. Locked, not `pip install -e .`
# against pyproject.toml directly, because reproducibility is the whole premise of this
# project -- a build that re-resolves versions is a build that can silently change what
# "111 tests, fully offline" was actually run against.
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv==0.9.7

WORKDIR /app
COPY pyproject.toml uv.lock ./
# --no-install-project: cache the dependency layer independently of application code, so an
# edit to src/ doesn't invalidate and re-resolve the entire dependency set on every build.
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY README.md ./
RUN uv sync --frozen --no-dev

# --- runtime ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root: a compromised research-fetching process (this one talks to arbitrary URLs on
# the open web by design) should not be root in its own container on top of that.
RUN groupadd -r gateway && useradd -r -g gateway -d /app -s /usr/sbin/nologin gateway

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN chown -R gateway:gateway /app
USER gateway

EXPOSE 8000

# Uses urllib rather than curl/wget: neither is in python:slim, and adding one is an extra
# package for a single GET this image can already do with its own interpreter.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as r; r.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Runs uvicorn directly rather than through `gateway serve`: the CLI wrapper's argparse
# defaults to 127.0.0.1, which is unreachable from outside the container.
CMD ["uvicorn", "gateway.api:app", "--host", "0.0.0.0", "--port", "8000"]
