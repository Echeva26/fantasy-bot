"""
Runner Docker para agente LangChain autónomo.

Lanza en un solo proceso:
- Daemon LangChain (scheduler PRE/POST)
- Bot de Telegram para renovación de token (opcional)
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from types import SimpleNamespace

from prediction.langchain_autonomous import run_daemon
from prediction.token_bot import run_token_bot

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runner Docker LangChain: daemon + token bot"
    )
    parser.add_argument(
        "--league",
        default="",
        help="Liga fija opcional (modo avanzado). Si se omite, usa la selección de Telegram.",
    )
    parser.add_argument(
        "--state-file",
        default=os.getenv("LANGCHAIN_STATE_FILE", ".langchain_agent_state.json"),
    )
    parser.add_argument(
        "--token-alert-cooldown-minutes",
        type=int,
        default=int(os.getenv("TOKEN_ALERT_COOLDOWN_MINUTES", "360")),
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LANGCHAIN_LLM_MODEL", "gpt-5-mini"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LANGCHAIN_TEMPERATURE", "0.1")),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.getenv("LANGCHAIN_MAX_ITERATIONS", "20")),
    )
    parser.add_argument(
        "--pre-objective",
        default=os.getenv("LANGCHAIN_PRE_OBJECTIVE", ""),
    )
    parser.add_argument(
        "--post-objective",
        default=os.getenv("LANGCHAIN_POST_OBJECTIVE", ""),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--token-bot-enabled",
        action="store_true",
        default=_env_bool("TOKEN_BOT_ENABLED", True),
    )
    parser.add_argument("--token-bot-disabled", action="store_true")
    parser.add_argument("--bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument("--allowed-chat-id", default=os.getenv("TELEGRAM_ALLOWED_CHAT_ID", ""))
    parser.add_argument("--notify-chat-id", default=os.getenv("TELEGRAM_CHAT_ID", ""))
    parser.add_argument(
        "--token-bot-poll-timeout",
        type=int,
        default=int(os.getenv("TOKEN_BOT_POLL_TIMEOUT", "50")),
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    stop_event = threading.Event()
    token_bot_enabled = bool(args.token_bot_enabled and not args.token_bot_disabled)

    daemon_args = SimpleNamespace(
        league=args.league,
        state_file=args.state_file,
        token_alert_cooldown_minutes=args.token_alert_cooldown_minutes,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        pre_objective=args.pre_objective,
        post_objective=args.post_objective,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    threads: list[threading.Thread] = []

    daemon_thread = threading.Thread(
        target=run_daemon,
        kwargs={"args": daemon_args, "stop_event": stop_event},
        name="langchain-daemon",
        daemon=True,
    )
    daemon_thread.start()
    threads.append(daemon_thread)
    logger.info("LangChain daemon thread iniciada.")

    if token_bot_enabled:
        if not args.bot_token:
            logger.warning(
                "TOKEN_BOT_ENABLED activo pero falta TELEGRAM_BOT_TOKEN. "
                "Se ejecutará solo el daemon LangChain."
            )
        else:
            token_thread = threading.Thread(
                target=run_token_bot,
                kwargs={
                    "bot_token": args.bot_token,
                    "allowed_chat_id": args.allowed_chat_id,
                    "poll_timeout": args.token_bot_poll_timeout,
                    "notify_chat_id": args.notify_chat_id,
                    "stop_event": stop_event,
                },
                name="token-bot",
                daemon=True,
            )
            token_thread.start()
            threads.append(token_thread)
            logger.info("Token bot thread iniciada.")
    else:
        logger.info("Token bot deshabilitado.")

    try:
        while True:
            for thread in threads:
                if not thread.is_alive():
                    raise RuntimeError(f"Thread detenida inesperadamente: {thread.name}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Parada solicitada por usuario.")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
