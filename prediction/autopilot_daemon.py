"""
Autopilot Daemon
================
Servicio persistente:
- Vigila caducidad de token y avisa por Telegram.
- Ejecuta PRE mercado a una hora fija diaria.
- Ejecuta POST mercado a una hora fija diaria.
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
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from prediction.autopilot import run_post_market, run_pre_market
from prediction.telegram_notify import GUIDA_RENOVACION_TOKEN, send_telegram_message

logger = logging.getLogger(__name__)

TOKEN_FILE = Path(".laliga_token")
DEFAULT_STATE_FILE = Path(".autopilot_state.json")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_hhmm(raw: str) -> tuple[int, int]:
    s = (raw or "").strip()
    try:
        hh, mm = s.split(":", 1)
        h = int(hh)
        m = int(mm)
    except Exception as exc:
        raise ValueError(f"Hora invalida '{raw}'. Formato esperado HH:MM") from exc
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Hora invalida '{raw}'.")
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


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _notify(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        send_telegram_message(token, chat_id, text)


def _token_status(max_age_hours: float) -> tuple[str, float | None, str | None]:
    """
    status: ok | missing | invalid | expired
    age_h: edad estimada en horas (si disponible)
    saved_at_iso: timestamp de guardado
    """
    if not TOKEN_FILE.exists():
        return "missing", None, None

    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        saved_at_raw = data.get("saved_at")
        if not saved_at_raw:
            return "invalid", None, None
        saved_at = datetime.fromisoformat(saved_at_raw)
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600.0
        if age_h >= float(max_age_hours):
            return "expired", age_h, saved_at.isoformat()
        return "ok", age_h, saved_at.isoformat()
    except Exception:
        return "invalid", None, None


def _maybe_alert_token_issue(
    state: dict,
    cooldown_minutes: int,
    token_status: str,
    age_h: float | None,
    saved_at_iso: str | None,
) -> dict:
    """
    Envia alertas de token de forma controlada para evitar spam.
    """
    now_utc = datetime.now(timezone.utc)
    new_state = dict(state)

    if token_status == "ok":
        if new_state.get("token_alert_key"):
            logger.info("Token recuperado; se limpia estado de alerta.")
        new_state.pop("token_alert_key", None)
        new_state.pop("token_alert_at", None)
        return new_state

    alert_key = f"{token_status}:{saved_at_iso or 'none'}"
    last_key = str(new_state.get("token_alert_key", ""))
    last_at = _parse_iso_dt(new_state.get("token_alert_at"))
    cooldown = timedelta(minutes=max(1, int(cooldown_minutes)))

    should_alert = False
    if alert_key != last_key:
        should_alert = True
    elif not last_at:
        should_alert = True
    elif now_utc - last_at >= cooldown:
        should_alert = True

    if should_alert:
        if token_status == "expired":
            msg = (
                f"Fantasy Autopilot TOKEN CADUCADO (edad ~{age_h:.1f}h)"
                + GUIDA_RENOVACION_TOKEN
            )
        elif token_status == "missing":
            msg = "Fantasy Autopilot TOKEN AUSENTE" + GUIDA_RENOVACION_TOKEN
        else:
            msg = "Fantasy Autopilot TOKEN INVALIDO" + GUIDA_RENOVACION_TOKEN
        logger.warning(msg.replace("\n", " | "))
        _notify(msg)
        new_state["token_alert_key"] = alert_key
        new_state["token_alert_at"] = now_utc.isoformat()

    return new_state


def _should_run(
    last_run_date: str | None,
    now_local: datetime,
    target_h: int,
    target_m: int,
) -> bool:
    today = now_local.date().isoformat()
    if last_run_date == today:
        return False
    return (now_local.hour, now_local.minute) >= (target_h, target_m)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daemon autonomo pre/post mercado")
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
        default=os.getenv("AUTOPILOT_STATE_FILE", str(DEFAULT_STATE_FILE)),
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
        default=os.getenv("LINEUP_AUTO_AFTER_TIME", os.getenv("AUTOPILOT_POST_TIME", "08:10")),
        help="Hora local mínima para guardar alineación (HH:MM)",
    )
    parser.add_argument(
        "--lineup-day-before-only",
        action="store_true",
        default=_env_bool("LINEUP_AUTO_DAY_BEFORE_ONLY", True),
        help="Guardar alineación solo en D-1",
    )
    parser.add_argument(
        "--lineup-no-day-before-check",
        action="store_true",
        help="Ignorar la condición D-1 para alineación",
    )
    return parser


def run_daemon(args: argparse.Namespace, stop_event: Event | None = None) -> None:
    if not args.league:
        raise RuntimeError("Falta league_id: usa --league o LALIGA_LEAGUE_ID.")

    pre_h, pre_m = _parse_hhmm(args.pre_time)
    post_h, post_m = _parse_hhmm(args.post_time)

    tz_name = os.getenv("TZ", "Europe/Madrid")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning("TZ invalida (%s), usando UTC.", tz_name)
        tz = timezone.utc

    state_path = Path(args.state_file)
    state = _load_state(state_path)

    _notify(
        "Fantasy Autopilot daemon iniciado\n"
        f"Liga: {args.league}\n"
        f"PRE: {args.pre_time} | POST: {args.post_time} | TZ: {tz_name}"
    )
    logger.info(
        "Daemon iniciado liga=%s pre=%s post=%s tz=%s state=%s",
        args.league,
        args.pre_time,
        args.post_time,
        tz_name,
        state_path,
    )

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Daemon detenido por stop_event.")
            return
        try:
            now_local = datetime.now(tz)
            now_utc = datetime.now(timezone.utc)

            # 1) Token health / alertas
            token_st, age_h, saved_at = _token_status(args.token_max_age_hours)
            state = _maybe_alert_token_issue(
                state=state,
                cooldown_minutes=args.token_alert_cooldown_minutes,
                token_status=token_st,
                age_h=age_h,
                saved_at_iso=saved_at,
            )

            # 2) PRE mercado
            if _should_run(state.get("last_pre_run"), now_local, pre_h, pre_m):
                if token_st == "ok":
                    pre_args = SimpleNamespace(
                        league=args.league,
                        model=args.model,
                        snapshot=None,
                        output=None,
                        dry_run=args.dry_run,
                        analysis_only=args.analysis_only,
                    )
                    logger.info("Lanzando PRE mercado...")
                    run_pre_market(pre_args)
                    state["last_pre_run"] = now_local.date().isoformat()
                    state["last_pre_run_at"] = now_utc.isoformat()
                else:
                    logger.warning("PRE omitido: token_status=%s", token_st)

            # 3) POST mercado
            if _should_run(state.get("last_post_run"), now_local, post_h, post_m):
                if token_st == "ok":
                    post_args = SimpleNamespace(league=args.league)
                    logger.info("Lanzando POST mercado...")
                    run_post_market(post_args)
                    state["last_post_run"] = now_local.date().isoformat()
                    state["last_post_run_at"] = now_utc.isoformat()

                    lineup_enabled = bool(
                        getattr(args, "lineup_auto_enabled", True)
                        and not getattr(args, "lineup_auto_disabled", False)
                    )
                    if lineup_enabled:
                        try:
                            from prediction.lineup_autoset import autoset_best_lineup

                            day_before_only = bool(
                                getattr(args, "lineup_day_before_only", True)
                                and not getattr(args, "lineup_no_day_before_check", False)
                            )
                            lineup_res = autoset_best_lineup(
                                league_id=args.league,
                                model=args.model,
                                day_before_only=day_before_only,
                                after_market_time=getattr(args, "lineup_after_time", "08:10"),
                                timezone_name=tz_name,
                                force=False,
                                dry_run=args.dry_run,
                            )
                            if lineup_res.get("applied"):
                                msg = (
                                    "Fantasy Autopilot LINEUP\n"
                                    f"Jornada: {lineup_res.get('jornada')}\n"
                                    f"Formacion: {'-'.join(str(x) for x in lineup_res.get('formation', []))}\n"
                                    f"xP once: {lineup_res.get('xp_once')}"
                                )
                                _notify(msg)
                                logger.info("Alineación guardada: %s", msg.replace("\n", " | "))
                            elif lineup_res.get("skipped"):
                                logger.info("Lineup skip: %s", lineup_res.get("reason"))
                            elif lineup_res.get("dry_run"):
                                logger.info(
                                    "Lineup dry-run jornada=%s forma=%s",
                                    lineup_res.get("jornada"),
                                    lineup_res.get("formation"),
                                )
                        except Exception as exc:
                            logger.exception("Error auto-set lineup: %s", exc)
                            _notify(
                                f"Fantasy Autopilot LINEUP ERROR\n"
                                f"{type(exc).__name__}: {exc}"
                            )
                else:
                    logger.warning("POST omitido: token_status=%s", token_st)

            _save_state(state_path, state)
            sleep_seconds = max(10, int(args.poll_seconds))
            if stop_event:
                stop_event.wait(timeout=sleep_seconds)
            else:
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("Daemon detenido por usuario.")
            return
        except Exception as exc:
            logger.exception("Error en daemon loop: %s", exc)
            _notify(f"Fantasy Autopilot daemon ERROR\n{type(exc).__name__}: {exc}")
            sleep_seconds = max(10, int(args.poll_seconds))
            if stop_event:
                stop_event.wait(timeout=sleep_seconds)
            else:
                time.sleep(sleep_seconds)


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
