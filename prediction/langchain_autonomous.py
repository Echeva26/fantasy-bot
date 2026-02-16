"""
Daemon autónomo para el agente LangChain.

Ejemplo:
  python -m prediction.langchain_autonomous --league 016615640
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from zoneinfo import ZoneInfo

from laliga_fantasy_client import load_token
from prediction.langchain_agent import run_agent_objective, run_agent_phase
from prediction.telegram_notify import send_telegram_message

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = Path(".langchain_agent_state.json")


def _parse_hhmm(raw: str) -> tuple[int, int]:
    s = (raw or "").strip()
    try:
        hh, mm = s.split(":", 1)
        h = int(hh)
        m = int(mm)
    except Exception as exc:
        raise ValueError(f"Hora inválida '{raw}', se esperaba HH:MM.") from exc
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Hora inválida '{raw}'.")
    return h, m


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def _notify(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        send_telegram_message(token, chat_id, text)


def _should_run(last_run_date: str | None, now_local: datetime, target_h: int, target_m: int) -> bool:
    today = now_local.date().isoformat()
    if last_run_date == today:
        return False
    return (now_local.hour, now_local.minute) >= (target_h, target_m)


def _token_ok() -> bool:
    return bool(load_token())


def _maybe_alert_token_issue(state: dict, cooldown_minutes: int) -> dict:
    now_utc = datetime.now(timezone.utc)
    new_state = dict(state)

    if _token_ok():
        new_state.pop("token_alert_at", None)
        return new_state

    cooldown = timedelta(minutes=max(1, int(cooldown_minutes)))
    last_alert_raw = new_state.get("token_alert_at")
    should_alert = False
    if not last_alert_raw:
        should_alert = True
    else:
        try:
            last_alert = datetime.fromisoformat(last_alert_raw)
            if last_alert.tzinfo is None:
                last_alert = last_alert.replace(tzinfo=timezone.utc)
            if now_utc - last_alert >= cooldown:
                should_alert = True
        except Exception:
            should_alert = True

    if should_alert:
        msg = (
            "Fantasy LangChain Agent: token ausente/expirado.\n"
            "Renueva token enviándolo al bot de Telegram antes del próximo ciclo."
        )
        logger.warning(msg.replace("\n", " | "))
        _notify(msg)
        new_state["token_alert_at"] = now_utc.isoformat()

    return new_state


def _build_objective(phase: str, custom_pre: str, custom_post: str) -> str | None:
    if phase == "pre" and custom_pre.strip():
        return custom_pre.strip()
    if phase == "post" and custom_post.strip():
        return custom_post.strip()
    return None


def _run_phase(phase: str, args: argparse.Namespace) -> dict:
    custom_objective = _build_objective(phase, args.pre_objective, args.post_objective)
    if custom_objective:
        return run_agent_objective(
            league_id=args.league,
            objective=custom_objective,
            model_type=args.model,
            llm_model=args.llm_model,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    return run_agent_phase(
        league_id=args.league,
        phase=phase,
        model_type=args.model,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def run_daemon(args: argparse.Namespace, stop_event: Event | None = None) -> None:
    if not args.league:
        raise RuntimeError("Falta league_id: usa --league o LALIGA_LEAGUE_ID.")

    pre_h, pre_m = _parse_hhmm(args.pre_time)
    post_h, post_m = _parse_hhmm(args.post_time)

    tz_name = os.getenv("TZ", "Europe/Madrid")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning("TZ inválida (%s), usando UTC.", tz_name)
        tz = timezone.utc

    state_path = Path(args.state_file)
    state = _load_state(state_path)

    _notify(
        "Fantasy LangChain daemon iniciado\n"
        f"Liga: {args.league}\n"
        f"PRE: {args.pre_time} | POST: {args.post_time} | LLM: {args.llm_model}"
    )
    logger.info(
        "LangChain daemon iniciado liga=%s pre=%s post=%s model=%s llm=%s",
        args.league,
        args.pre_time,
        args.post_time,
        args.model,
        args.llm_model,
    )

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Daemon LangChain detenido por stop_event.")
            return
        try:
            now_local = datetime.now(tz)
            now_utc = datetime.now(timezone.utc)
            state = _maybe_alert_token_issue(state, args.token_alert_cooldown_minutes)
            has_token = _token_ok()

            if _should_run(state.get("last_pre_run"), now_local, pre_h, pre_m):
                if has_token:
                    logger.info("Lanzando fase PRE con agente LangChain...")
                    res = _run_phase("pre", args)
                    output = res.get("output", "")
                    state["last_pre_run"] = now_local.date().isoformat()
                    state["last_pre_run_at"] = now_utc.isoformat()
                    state["last_pre_output"] = output[:2000]
                    _notify(
                        "Fantasy LangChain (PRE) completado\n"
                        f"Tools: {len(res.get('steps', []))}\n"
                        f"Resumen:\n{output[:1200]}"
                    )
                else:
                    logger.warning("PRE omitido: token no válido.")

            if _should_run(state.get("last_post_run"), now_local, post_h, post_m):
                if has_token:
                    logger.info("Lanzando fase POST con agente LangChain...")
                    res = _run_phase("post", args)
                    output = res.get("output", "")
                    state["last_post_run"] = now_local.date().isoformat()
                    state["last_post_run_at"] = now_utc.isoformat()
                    state["last_post_output"] = output[:2000]
                    _notify(
                        "Fantasy LangChain (POST) completado\n"
                        f"Tools: {len(res.get('steps', []))}\n"
                        f"Resumen:\n{output[:1200]}"
                    )
                else:
                    logger.warning("POST omitido: token no válido.")

            _save_state(state_path, state)
            sleep_seconds = max(10, int(args.poll_seconds))
            if stop_event:
                stop_event.wait(timeout=sleep_seconds)
            else:
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("Daemon LangChain detenido por usuario.")
            return
        except Exception as exc:
            logger.exception("Error en bucle LangChain daemon: %s", exc)
            _notify(f"Fantasy LangChain daemon ERROR\n{type(exc).__name__}: {exc}")
            sleep_seconds = max(10, int(args.poll_seconds))
            if stop_event:
                stop_event.wait(timeout=sleep_seconds)
            else:
                time.sleep(sleep_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daemon autónomo con agente LangChain")
    parser.add_argument("--league", default=os.getenv("LALIGA_LEAGUE_ID", ""))
    parser.add_argument(
        "--model",
        default=os.getenv("AUTOPILOT_MODEL", "xgboost"),
        choices=["xgboost", "lightgbm"],
    )
    parser.add_argument("--pre-time", default=os.getenv("AUTOPILOT_PRE_TIME", "07:50"))
    parser.add_argument("--post-time", default=os.getenv("AUTOPILOT_POST_TIME", "08:10"))
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=int(os.getenv("AUTOPILOT_POLL_SECONDS", "30")),
    )
    parser.add_argument(
        "--state-file",
        default=os.getenv("LANGCHAIN_STATE_FILE", str(DEFAULT_STATE_FILE)),
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
        help="Objetivo custom para fase PRE.",
    )
    parser.add_argument(
        "--post-objective",
        default=os.getenv("LANGCHAIN_POST_OBJECTIVE", ""),
        help="Objetivo custom para fase POST.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_daemon(args=args, stop_event=None)


if __name__ == "__main__":
    main()
