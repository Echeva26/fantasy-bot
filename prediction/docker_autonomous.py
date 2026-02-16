"""
Docker Autonomous Runner
========================
Lanza en un solo proceso:
- Autopilot daemon (scheduler pre/post + alerta de token)
- Telegram token bot (renovacion de token)
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from types import SimpleNamespace

from prediction.autopilot_daemon import run_daemon
from prediction.token_bot import run_token_bot

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Runner unico para Docker: autopilot + telegram token bot"
    )
    # Daemon config
    parser.add_argument("--league", default=os.getenv("LALIGA_LEAGUE_ID", ""))
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "lightgbm"],
    )
    parser.add_argument("--pre-time", default="07:50")
    parser.add_argument("--post-time", default="08:10")
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--token-max-age-hours",
        type=float,
        default=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")),
    )
    parser.add_argument(
        "--token-alert-cooldown-minutes",
        type=int,
        default=int(os.getenv("TOKEN_ALERT_COOLDOWN_MINUTES", "360")),
    )
    parser.add_argument(
        "--state-file",
        default=os.getenv("AUTOPILOT_STATE_FILE", ".autopilot_state.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--lineup-auto-enabled",
        action="store_true",
        default=_env_bool("LINEUP_AUTO_ENABLED", True),
        help="Activar guardado automático de alineación",
    )
    parser.add_argument(
        "--lineup-auto-disabled",
        action="store_true",
        help="Desactivar guardado automático de alineación",
    )
    parser.add_argument(
        "--lineup-after-time",
        default=os.getenv("LINEUP_AUTO_AFTER_TIME", "08:10"),
        help="Hora local mínima para auto-set de alineación (HH:MM)",
    )
    parser.add_argument(
        "--lineup-day-before-only",
        action="store_true",
        default=_env_bool("LINEUP_AUTO_DAY_BEFORE_ONLY", True),
        help="Guardar alineación solo D-1",
    )
    parser.add_argument(
        "--lineup-no-day-before-check",
        action="store_true",
        help="Ignorar condición D-1 para alineación",
    )

    # Telegram token bot config
    parser.add_argument(
        "--token-bot-enabled",
        action="store_true",
        default=_env_bool("TOKEN_BOT_ENABLED", True),
        help="Habilitar bot de Telegram para renovar token",
    )
    parser.add_argument(
        "--token-bot-disabled",
        action="store_true",
        help="Deshabilitar bot de Telegram para renovar token",
    )
    parser.add_argument(
        "--bot-token",
        default=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        help="Token del bot de Telegram",
    )
    parser.add_argument(
        "--allowed-chat-id",
        default=os.getenv("TELEGRAM_ALLOWED_CHAT_ID", ""),
        help="Si se define, solo este chat puede renovar token",
    )
    parser.add_argument(
        "--notify-chat-id",
        default=os.getenv("TELEGRAM_CHAT_ID", ""),
        help="Chat para notificaciones de estado/error",
    )
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

    if not args.league:
        raise RuntimeError("Falta league_id: usa --league o LALIGA_LEAGUE_ID.")

    token_bot_enabled = bool(args.token_bot_enabled and not args.token_bot_disabled)
    stop_event = threading.Event()

    daemon_args = SimpleNamespace(
        league=args.league,
        model=args.model,
        pre_time=args.pre_time,
        post_time=args.post_time,
        poll_seconds=args.poll_seconds,
        token_max_age_hours=args.token_max_age_hours,
        token_alert_cooldown_minutes=args.token_alert_cooldown_minutes,
        state_file=args.state_file,
        dry_run=args.dry_run,
        analysis_only=args.analysis_only,
        lineup_auto_enabled=args.lineup_auto_enabled,
        lineup_auto_disabled=args.lineup_auto_disabled,
        lineup_after_time=args.lineup_after_time,
        lineup_day_before_only=args.lineup_day_before_only,
        lineup_no_day_before_check=args.lineup_no_day_before_check,
    )

    threads: list[threading.Thread] = []

    daemon_thread = threading.Thread(
        target=run_daemon,
        kwargs={"args": daemon_args, "stop_event": stop_event},
        name="autopilot-daemon",
        daemon=True,
    )
    daemon_thread.start()
    threads.append(daemon_thread)
    logger.info("Autopilot daemon thread iniciada.")

    if token_bot_enabled:
        if not args.bot_token:
            logger.warning(
                "TOKEN_BOT_ENABLED activo pero falta TELEGRAM_BOT_TOKEN. "
                "Se ejecutara solo el scheduler."
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
            for t in threads:
                if not t.is_alive():
                    raise RuntimeError(f"Thread detenida inesperadamente: {t.name}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Parada solicitada por usuario.")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5)


if __name__ == "__main__":
    main()
