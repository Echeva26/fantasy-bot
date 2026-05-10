"""
Configuración y utilidades seguras para LangSmith.

Este módulo está diseñado para ser opcional:
- si LangSmith no está instalado, no rompe nada;
- si LANGSMITH_TRACING=false o falta LANGSMITH_API_KEY, no envía trazas;
- si hay cualquier error de observabilidad, la lógica principal sigue funcionando.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any, Iterator

logger = logging.getLogger(__name__)

APP_NAME = "fantasy-bot"
COMPONENT_NAME = "langchain-agent"
REDACTED = "[REDACTED]"
SENSITIVE_KEY_MARKERS = (
    "token",
    "accesstoken",
    "authorization",
    "password",
    "cookie",
    "setcookie",
    "xapikey",
    "apikey",
    "secret",
    "bearer",
    "refreshtoken",
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.getenv(name, "")
        if raw and raw.strip():
            return raw.strip()
    return default


def get_langsmith_project_name() -> str:
    return _env_first("LANGSMITH_PROJECT", default=APP_NAME)


def _normalize_key(key: Any) -> str:
    return (
        str(key or "")
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _is_sensitive_key(key: Any) -> bool:
    norm = _normalize_key(key)
    return any(marker in norm for marker in SENSITIVE_KEY_MARKERS)


def sanitize_langsmith_metadata(value: Any) -> Any:
    """
    Elimina o enmascara datos sensibles en estructuras arbitrarias.
    """
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                sanitized[str(key)] = REDACTED
            else:
                sanitized[str(key)] = sanitize_langsmith_metadata(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [sanitize_langsmith_metadata(item) for item in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return value


def _load_langsmith_runtime() -> tuple[Any | None, Any | None]:
    try:
        import langsmith as ls
        from langsmith import trace as ls_trace

        return ls, ls_trace
    except Exception:
        return None, None


def is_langsmith_enabled() -> bool:
    if not _env_bool("LANGSMITH_TRACING", default=False):
        return False
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        return False
    ls, _ = _load_langsmith_runtime()
    return ls is not None


def build_langsmith_config(
    run_name: str,
    phase: str | None = None,
    command: str | None = None,
    league_id: str | None = None,
    market_cycle_id: str | None = None,
    dry_run: bool | None = None,
    extra_metadata: dict | None = None,
) -> dict[str, Any]:
    phase_value = str(phase or "").strip().lower()
    command_value = str(command or "").strip().lower()

    tags = [APP_NAME, COMPONENT_NAME]
    if phase_value:
        tags.append(phase_value)
    if command_value:
        tags.append(command_value)
    if dry_run:
        tags.append("dry-run")

    metadata = sanitize_langsmith_metadata(extra_metadata or {})
    if not isinstance(metadata, dict):
        metadata = {"extra": metadata}

    environment = _env_first("ENVIRONMENT", "APP_ENV")
    model_name = str(
        metadata.get("model")
        or metadata.get("llm_model")
        or os.getenv("LANGCHAIN_LLM_MODEL", "")
    ).strip()
    provider = str(
        metadata.get("provider")
        or metadata.get("model_provider")
        or ("openai" if model_name else "")
    ).strip()

    metadata.update(
        {
            "app": APP_NAME,
            "component": COMPONENT_NAME,
        }
    )
    if phase_value:
        metadata["phase"] = phase_value
    if command_value:
        metadata["command"] = command_value
    if league_id:
        metadata["league_id"] = str(league_id)
    if market_cycle_id:
        metadata["market_cycle_id"] = str(market_cycle_id)
    if dry_run is not None:
        metadata["dry_run"] = bool(dry_run)
    if environment:
        metadata["environment"] = environment
    if model_name:
        metadata["model"] = model_name
    if provider:
        metadata["provider"] = provider

    return {
        "run_name": run_name,
        "tags": list(dict.fromkeys(tag for tag in tags if tag)),
        "metadata": sanitize_langsmith_metadata(metadata),
    }


@contextlib.contextmanager
def langsmith_trace_context(
    run_name: str,
    phase: str | None = None,
    command: str | None = None,
    league_id: str | None = None,
    market_cycle_id: str | None = None,
    dry_run: bool | None = None,
    extra_metadata: dict | None = None,
) -> Iterator[None]:
    if not is_langsmith_enabled():
        yield
        return

    ls, _ = _load_langsmith_runtime()
    if ls is None or not hasattr(ls, "tracing_context"):
        yield
        return

    config = build_langsmith_config(
        run_name=run_name,
        phase=phase,
        command=command,
        league_id=league_id,
        market_cycle_id=market_cycle_id,
        dry_run=dry_run,
        extra_metadata=extra_metadata,
    )
    try:
        ctx = ls.tracing_context(
            enabled=True,
            project_name=get_langsmith_project_name(),
            tags=config["tags"],
            metadata=config["metadata"],
        )
    except Exception as exc:
        logger.debug("No se pudo crear tracing_context de LangSmith: %s", exc)
        yield
        return

    try:
        ctx.__enter__()
    except Exception as exc:
        logger.debug("No se pudo entrar en tracing_context de LangSmith: %s", exc)
        yield
        return

    try:
        yield
    except Exception:
        exc_info = sys.exc_info()
        try:
            ctx.__exit__(*exc_info)
        except Exception as exc:
            logger.debug("No se pudo cerrar tracing_context de LangSmith tras excepción: %s", exc)
        raise
    else:
        try:
            ctx.__exit__(None, None, None)
        except Exception as exc:
            logger.debug("No se pudo cerrar tracing_context de LangSmith: %s", exc)


class _NoOpRun:
    def end(self, *args: Any, **kwargs: Any) -> None:
        return None


@contextlib.contextmanager
def langsmith_run_span(
    run_name: str,
    *,
    run_type: str = "chain",
    phase: str | None = None,
    command: str | None = None,
    league_id: str | None = None,
    market_cycle_id: str | None = None,
    dry_run: bool | None = None,
    extra_metadata: dict | None = None,
    inputs: dict | None = None,
) -> Iterator[Any]:
    if not is_langsmith_enabled():
        yield _NoOpRun()
        return

    _, ls_trace = _load_langsmith_runtime()
    if ls_trace is None:
        yield _NoOpRun()
        return

    config = build_langsmith_config(
        run_name=run_name,
        phase=phase,
        command=command,
        league_id=league_id,
        market_cycle_id=market_cycle_id,
        dry_run=dry_run,
        extra_metadata=extra_metadata,
    )
    safe_inputs = sanitize_langsmith_metadata(inputs or {})

    try:
        trace_ctx = ls_trace(
            name=run_name,
            run_type=run_type,
            project_name=get_langsmith_project_name(),
            inputs=safe_inputs,
            tags=config["tags"],
            metadata=config["metadata"],
        )
    except Exception as exc:
        logger.debug("No se pudo crear span manual de LangSmith: %s", exc)
        yield _NoOpRun()
        return

    try:
        run_tree = trace_ctx.__enter__()
    except Exception as exc:
        logger.debug("No se pudo entrar en span manual de LangSmith: %s", exc)
        yield _NoOpRun()
        return

    try:
        yield run_tree
    except Exception:
        exc_info = sys.exc_info()
        try:
            trace_ctx.__exit__(*exc_info)
        except Exception as exc:
            logger.debug("No se pudo cerrar span manual de LangSmith tras excepción: %s", exc)
        raise
    else:
        try:
            trace_ctx.__exit__(None, None, None)
        except Exception as exc:
            logger.debug("No se pudo cerrar span manual de LangSmith: %s", exc)


def finish_langsmith_span(run_tree: Any, outputs: dict | None = None) -> None:
    end = getattr(run_tree, "end", None)
    if not callable(end):
        return
    try:
        end(outputs=sanitize_langsmith_metadata(outputs or {}))
    except Exception as exc:
        logger.debug("No se pudo finalizar span de LangSmith: %s", exc)


__all__ = [
    "build_langsmith_config",
    "finish_langsmith_span",
    "get_langsmith_project_name",
    "is_langsmith_enabled",
    "langsmith_run_span",
    "langsmith_trace_context",
    "sanitize_langsmith_metadata",
]
