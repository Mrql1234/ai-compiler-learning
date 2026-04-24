# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

AI Robot Agent service — an LLM-driven conversational agent that generates replies directly via System Prompt (no DAG orchestration). Built with FastAPI, deployed on K8s with Gunicorn+Uvicorn.

## Agent Framework

The project uses a custom agent abstraction (`BaseAgent` / `SimpleChatAgent`) rather than a full framework. The only AgentScope dependency is `OpenAIChatModel` from `agentscope.model`, used in `app/agents/llm_client.py` for streaming LLM calls.

- AgentScope docs: https://doc.agentscope.io/
- AgentScope repo: https://github.com/agentscope-ai/agentscope

## Commands

```bash
# Install dependencies
uv sync

# Run locally (port 8008, requires Docker MySQL + Redis + Ollama)
env=dev \
  DATABASE_URL="mysql+pymysql://root:root123@127.0.0.1:3306/test" \
  REDIS_URL="redis://127.0.0.1:6379" \
  LLM_BASE_URL="http://127.0.0.1:11434/v1" \
  LLM_API_KEY="EMPTY" \
  uv run python run_local.py

# Lint & format (ALWAYS run this when a task finished)
uv run ruff check .
uv run black .
uv run mypy .

# Run unit tests
./scripts/run_tests.sh
# or directly:
uv sync --extra test && uv run pytest -q

# Run a single test
uv run pytest tests/path/to/test_file.py::test_name -v

# Run E2E endpoint tests (server must be running)
bash scripts/test_endpoints.sh
```

## Architecture

### Layers

- **app/main.py** — FastAPI app with lifespan management. Entry point is `app.main:app`. Installs handlers for uncaught exceptions, signals, and asyncio errors.
- **app/core/config.py** — Config loaded from remote config center via `yqg_common.config_center`. Environments (dev/test/prod) defined in `config.ini`.
- **app/core/logging.py** — Logging initialized via `yqg_common` with rotating file handlers. Log level controlled by `LOG_LEVEL` env var.
- **app/core/middleware.py** — `TraceIdMiddleware` extracts/generates `X-Trace-Id` for distributed tracing.
- **app/api/routes/agent.py** — Route handlers for `/healthCheck`, `/sync_api`, `/streaming_api`. Single router re-exported from `app/api/__init__.py`.
- **app/services/chat_service.py** — Core orchestration: loads agent config, renders prompts, drives retry loop (sync) or stream loop (streaming), records metrics.
- **app/services/prompt_renderer.py** — Renders prompt templates substituting `{language}` and `{meta_data}` placeholders. Protects literal `{` / `}` from accidental substitution.
- **app/services/errors.py** — `LLMServiceError` domain exception carrying HTTP status code and caller-safe message.
- **app/agents/base.py** — `BaseAgent` ABC + `AgentResult` dataclass (`reply`, `intent`, `action`, `label`, `buttons`).
- **app/agents/simple_chat_agent.py** — `SimpleChatAgent`: parses ND-JSON event stream from LLM, aggregates into `AgentResult` (sync) or yields lines (streaming). Injects localised time suffix based on `__REGION` env var. Extracts `<Button>opt1;opt2</Button>` tags. Raises `NoStructuredReplyError` when no `text` events are present.
- **app/agents/llm_client.py** — Low-level LLM integration using AgentScope `OpenAIChatModel`. Caches model instances per `(model_name, stream)`. Supports per-model URL routing via `llm.model.url.mapping`. Maps OpenAI SDK exceptions to `LLMServiceError`.
- **app/repositories/robot_repository.py** — `get_agent_config(agent_id)` async wrapper (offloads blocking SQLAlchemy Core query to thread pool). Queries `ai_call_robot_agent`, excludes soft-deleted rows (`deleted=0`). Monitors connection pool exhaustion.
- **app/schemas/chat.py** — `ChatRequest` (camelCase aliases), `ChatResponse`, streaming event models, `HistoryMessage` (normalises `seat` role → `assistant`).
- **app/schemas/common.py** — `AgentConfig` TypedDict, `ErrorOut`.
- **app/utils/metrics.py** — InfluxDB metrics via `yqg_common.influxdb_batch_client` (batch 100, interval 1 s). Global `metrics` singleton. Tracks TTFT and total LLM latency tagged by model and agent_id.

### Request Flow

**Sync (`POST /sync_api`)**
```
ChatRequest → validate → handle_chat()
  → get_agent_config() → render_prompt()
  → retry loop (llm.retry.maxtimes, default 3):
      SimpleChatAgent.run()
        → chat_completion_stream() [AgentScope OpenAIChatModel]
        → parse ND-JSON lines → aggregate AgentResult
        → NoStructuredReplyError if no "text" events → retry
  → record sync_total latency metric
  → ChatResponse (HTTP 200)
```

**Streaming (`POST /streaming_api`)**
```
ChatRequest → validate → pre-load agent_config (404 before stream)
  → StreamingResponse (application/x-ndjson)
      handle_chat_stream()
        → SimpleChatAgent.run_stream()
            → chat_completion_stream()
            → buffer deltas → emit validated ND-JSON lines
            → record streaming_ttft on first line
        → errors yielded as {"type": "error", ...} events
```

### ND-JSON Event Types

| type | fields | description |
|------|--------|-------------|
| `text` | `delta` | Reply text chunk (LLM `reply` events are normalised to `text`) |
| `intent` | `name` | User intent classification |
| `action` | `name`, `action_params` | Actionable instruction |
| `label` | `name` | Response classification label |
| `error` | `message` | Error event (streaming only) |

### Error Handling

| Exception | Trigger | HTTP |
|-----------|---------|------|
| `AgentNotFoundError` | agent_id not in DB or soft-deleted | 404 |
| `LLMOutputError` | no structured reply after all retries | 502 |
| `LLMServiceError` | OpenAI SDK error (rate limit, timeout, etc.) | varies |
| `NoStructuredReplyError` | no `text` events in one LLM call | triggers retry |
| `EmptyReplyError` | stream yields no events at all (streaming path) | stream error event |

### Key Config Keys

| Key | Default | Description |
|-----|---------|-------------|
| `llm.retry.maxtimes` | `3` | Max retries on `NoStructuredReplyError` (sync only) |
| `llm.model.url.mapping` | — | JSON map of model name → base URL |
| `llm_base_url` / `LLM_BASE_URL` env | — | Fallback LLM base URL |
| `llm_api_key` / `LLM_API_KEY` env | `EMPTY` | LLM API key |
| `spring.datasource.url/username/password` | — | DB config (JDBC URL auto-converted) |
| `DATABASE_URL` env | — | Direct SQLAlchemy DB URL (takes priority) |
| `DB_POOL_SIZE` | `20` | SQLAlchemy pool size |
| `DB_MAX_OVERFLOW` | `40` | Max overflow connections |
| `__REGION` env | — | Timezone for time suffix injection (`cn`, `indo`, `mex`) |
| `LOG_LEVEL` env | `INFO` | Logging level |

## Key Conventions
- Python 3.11 (strict: `>=3.11,<3.12`)
- Dependency management via `uv` and `pyproject.toml` only
- Line length: 120 (black + ruff)
- Internal library `py-smart-common` (aliased as `yqg_common`) sourced from private Nexus registry
- Config values come from remote config center at runtime, not `.env` files — use `config.get("KEY")` pattern
- For Mysql, use SQLAlchemy Core

## Local Development

Config values come from a remote config center at runtime, which is unreachable locally. Use environment variables to override:

```bash
# Start service locally
env=dev \
  DATABASE_URL="mysql+pymysql://root:root123@127.0.0.1:3306/test" \
  REDIS_URL="redis://127.0.0.1:6379" \
  LLM_BASE_URL="http://127.0.0.1:11434/v1" \
  LLM_API_KEY="EMPTY" \
  uv run python run_local.py

# Run E2E endpoint tests (server must be running, in another terminal)
bash scripts/test_endpoints.sh
# Supports: BASE_URL (default http://localhost:8008), AGENT_ID (default 1)
```

**Local prerequisites:**
- Docker MySQL on port 3306 (password: `root123`, database: `test`), table `ai_call_robot_agent` must exist with at least one active record
- Docker Redis on port 6379
- Ollama on port 11434 with `qwen2.5:3b` model

**After finishing a task**, run the service locally and run `bash scripts/test_endpoints.sh`. Check both server log (`logs/application.log`) and client output for errors.
