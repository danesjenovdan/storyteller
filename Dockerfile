# syntax=docker/dockerfile:1
FROM python:3.14

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y ffmpeg gettext \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
	UV_LINK_MODE=copy \
	UV_PYTHON_DOWNLOADS=0 \
	UV_PROJECT_ENVIRONMENT=/opt/venv \
	PATH="/opt/venv/bin:$PATH"

# Install dependencies first (cached separately from app code changes).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
	uv sync --locked --no-dev --no-install-project

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
	uv sync --locked --no-dev

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

RUN python3 manage.py compilemessages

CMD exec python manage.py runserver 0.0.0.0:8000
