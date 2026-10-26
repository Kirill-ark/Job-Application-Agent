# The web app with the agent and its MCP server, in one image.
# Render (and docker compose) start it; Render sets PORT itself.
# Images are pinned by digest: a moved tag cannot change what gets built.
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

COPY --from=ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff /uv /bin/uv

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first: this layer is rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data/cv.yaml data/skills.yaml data/demo_run.json ./data/
RUN uv sync --frozen --no-dev && mkdir -p data/db && chown -R app:app /app

USER app
ENV PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "job_agent.api"]
