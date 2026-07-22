# Dockerfile
#
# Multi-stage build for the SmartStock Intelligence Platform.
#
#
# Stage 1 (builder) installs build tools + all Python dependencies into
# a venv; Stage 2 (runtime) copies only that venv and the application
# code into a clean slim base — keeping build-essential and pip's
# download/wheel cache out of the final image.

# ---------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------
FROM python:3.11-slim AS builder

# build-essential + libpq-dev: some of this project's dependencies
# (chromadb's hnswlib in particular) may need to compile a C extension
# if a prebuilt wheel isn't available for the target platform; libpq-dev
# is included defensively even though psycopg2-binary ships its own
# libpq, in case of an architecture where only the source distribution
# resolves. Both are discarded when Stage 2 copies only the venv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# ---------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# libpq5: the runtime C library psycopg2-binary's compiled extension
# links against. build-essential/libpq-dev from Stage 1 are not needed
# here — only the shared library itself.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user rather than the container's default root, per
# standard container hardening practice.
RUN useradd --create-home --shell /bin/bash smartstock

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app


COPY . .


RUN mkdir -p data/reports vector_db models logs \
    && chown -R smartstock:smartstock /app

USER smartstock

EXPOSE 8000

