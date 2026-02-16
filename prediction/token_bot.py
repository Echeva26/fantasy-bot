"""
Telegram Token Bot
==================
Recibe tokens por Telegram y actualiza .laliga_token.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from threading import Event
from urllib.parse import parse_qs, urlparse

import requests

from laliga_fantasy_client import LaLigaFantasyClient, TOKEN_FILE, load_token, save_token
from prediction.autopilot import run_pre_market
from prediction.league_selection import (
    load_selected_league,
    resolve_league_id,
    save_selected_league,
)
from prediction.market_schedule import build_market_schedule, schedule_message
from prediction.telegram_notify import GUIDA_RENOVACION_TOKEN, send_telegram_message

logger = logging.getLogger(__name__)

JWT_RE = re.compile(r"(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)")


def _api(bot_token: str, method: str, payload: dict | None = None) -> dict:
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    resp = requests.post(url, json=payload or {}, timeout=65)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data


def _extract_token_from_text(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None

    # 1) JWT directo
    match = JWT_RE.search(text)
    if match:
        return match.group(1)

    # 2) URL con fragment/query id_token/access_token
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        frag = parse_qs(parsed.fragment or "")
        qry = parse_qs(parsed.query or "")
        for src in (frag, qry):
            for key in ("id_token", "access_token"):
                values = src.get(key, [])
                if values and values[0].startswith("eyJ"):
                    return values[0]
    return None


def _token_age_hours() -> float | None:
    try:
        if not TOKEN_FILE.exists():
            return None
        data = TOKEN_FILE.read_text(encoding="utf-8")
        token = load_token()
        if not token:
            return None
        raw = json.loads(data)
        saved_at = datetime.fromisoformat(raw["saved_at"])
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600.0
        return age
    except Exception:
        return None


def _token_status(max_age_hours: float = 23.0) -> tuple[str, float | None]:
    """
    Devuelve (status, age_h) con status: "ok" | "missing" | "expired" | "invalid"
    """
    if not TOKEN_FILE.exists():
        return "missing", None
    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        saved_at_raw = data.get("saved_at")
        if not saved_at_raw:
            return "invalid", None
        saved_at = datetime.fromisoformat(saved_at_raw)
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600.0
        if age_h >= max_age_hours:
            return "expired", age_h
        return "ok", age_h
    except Exception:
        return "invalid", None


def _list_user_leagues() -> tuple[list[dict], str]:
    """
    Devuelve (ligas, error).
    ligas: [{"id": "...", "name": "..."}, ...]
    """
    status, _ = _token_status(max_age_hours=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")))
    if status != "ok":
        return [], f"token no válido ({status})"
    try:
        client = LaLigaFantasyClient.from_saved_token(league_id="")
        raw = client.get_leagues() or []
        leagues = []
        for lg in raw:
            lid = str(lg.get("id", "")).strip()
            if not lid:
                continue
            leagues.append(
                {
                    "id": lid,
                    "name": str(lg.get("name", "")).strip() or "(Sin nombre)",
                }
            )
        return leagues, ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _resolve_operational_league_id() -> tuple[str, str]:
    """
    Resuelve league_id para ejecutar comandos operativos.
    Si hay una sola liga en la cuenta, la selecciona automáticamente.
    """
    league_id = resolve_league_id("")
    if league_id:
        return league_id, ""

    leagues, err = _list_user_leagues()
    if err:
        return "", f"No se pudo resolver la liga: {err}"
    if not leagues:
        return "", "No se encontraron ligas para este usuario."
    if len(leagues) == 1:
        lg = leagues[0]
        save_selected_league(lg["id"], lg["name"], source="auto_single_league")
        return lg["id"], ""

    return (
        "",
        "No hay liga seleccionada. Usa /ligas para verlas y /liga <nombre> para elegir.",
    )


def _selected_league_text() -> str:
    selected = load_selected_league()
    if not selected:
        fallback = resolve_league_id("")
        if fallback:
            return "Liga activa: definida por configuracion avanzada."
        return "Liga seleccionada: (ninguna)"
    name = selected.get("league_name", "")
    if name:
        return f"Liga activa: {name}"
    return "Liga activa: seleccionada"


def _norm(text: str) -> str:
    s = (text or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(s.split())


def _cmd_list_leagues() -> str:
    leagues, err = _list_user_leagues()
    if err:
        return f"No se pudieron listar ligas: {err}"
    if not leagues:
        return "No se encontraron ligas para este usuario."

    selected_id = resolve_league_id("")
    lines = ["Ligas disponibles:"]
    for i, lg in enumerate(leagues, 1):
        mark = " ✅" if str(lg["id"]) == str(selected_id) else ""
        lines.append(f"{i}. {lg['name']}{mark}")
    lines.append("")
    lines.append("Usa /liga <nombre> para seleccionarla.")
    return "\n".join(lines)


def _cmd_select_league(raw_text: str) -> str:
    parts = raw_text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        current = _selected_league_text()
        return (
            f"{current}\n"
            "Para seleccionar liga: /liga <nombre>\n"
            "Para ver ligas: /ligas"
        )
    target = parts[1].strip()

    leagues, err = _list_user_leagues()
    if err:
        return f"No se pudo seleccionar liga: {err}"
    if not leagues:
        return "No se encontraron ligas para este usuario."

    chosen = None
    if target.isdigit():
        idx = int(target)
        if 1 <= idx <= len(leagues):
            chosen = leagues[idx - 1]
    else:
        norm_target = _norm(target)
        exact = [lg for lg in leagues if _norm(lg.get("name", "")) == norm_target]
        if len(exact) == 1:
            chosen = exact[0]
        elif len(exact) > 1:
            names = "\n".join(f"- {lg['name']}" for lg in exact[:10])
            return (
                "Hay varias ligas con ese nombre exacto. "
                "Escribe el nombre completo tal cual o usa el número:\n"
                f"{names}"
            )
        else:
            partial = [lg for lg in leagues if norm_target in _norm(lg.get("name", ""))]
            if len(partial) == 1:
                chosen = partial[0]
            elif len(partial) > 1:
                names = "\n".join(f"- {lg['name']}" for lg in partial[:10])
                return (
                    "Coinciden varias ligas con ese texto. "
                    "Escribe un nombre más específico o usa el número:\n"
                    f"{names}"
                )

    if not chosen:
        return (
            f"No encontré la liga '{target}'.\n"
            "Usa /ligas para ver opciones y luego /liga <nombre>."
        )

    save_selected_league(chosen["id"], chosen["name"], source="telegram")
    sched, sched_err = build_market_schedule(
        chosen["id"],
        timezone_name=os.getenv("TZ", "Europe/Madrid"),
    )
    schedule_txt = (
        "\n\n" + schedule_message(sched)
        if sched
        else f"\n\nNo pude calcular la hora de mercado ahora: {sched_err}"
    )
    return (
        "Liga seleccionada correctamente.\n"
        f"{chosen['name']}"
        f"{schedule_txt}"
    )


def _run_pre_market_cmd(analysis_only: bool, bot_token: str, chat_id: str) -> str:
    """Ejecuta informe o compraventa, envía mensaje intermedio y devuelve resumen."""
    cmd = "informe" if analysis_only else "compraventa"
    send_telegram_message(
        bot_token,
        chat_id,
        "Generando informe..." if analysis_only
        else "Ejecutando compraventa (recomendaciones del bot)...",
    )
    league_id, league_err = _resolve_operational_league_id()
    if not league_id:
        return f"Error de liga: {league_err}"

    status, _ = _token_status(max_age_hours=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")))
    if status != "ok":
        return f"Error: token no valido ({status}). Renuevalo con /help."

    from types import SimpleNamespace

    args = SimpleNamespace(
        league=league_id,
        model="xgboost",
        snapshot=None,
        output=None,
        dry_run=False,
        analysis_only=analysis_only,
    )
    try:
        result = run_pre_market(args, skip_notify=True)
        # Enviar informe completo como mensaje para verlo en el chat
        if result.report_content:
            send_telegram_message(bot_token, chat_id, result.report_content)
        return result.message
    except Exception as exc:
        logger.exception("Error en /%s: %s", cmd, exc)
        return f"Error ejecutando /{cmd}: {type(exc).__name__}: {exc}"


def _handle_text(
    text: str,
    *,
    bot_token: str = "",
    chat_id: str = "",
) -> tuple[bool, str]:
    t = text.strip().lower()
    if t.startswith("/start") or t.startswith("/help"):
        return (
            False,
            "Comandos:\n"
            "• /help - Esta ayuda\n"
            "• /status - Estado del token\n"
            "• /ligas - Listar ligas disponibles\n"
            "• /liga <nombre> - Seleccionar liga activa\n"
            "• /informe - Generar informe de predicciones ahora\n"
            "• /compraventa - Ejecutar recomendaciones (ventas+compras)\n\n"
            "Para renovar token: envia JWT (eyJ...) o URL de jwt.ms con id_token.",
        )
    if t.startswith("/status"):
        age = _token_age_hours()
        if age is None:
            return False, "No hay token valido guardado."
        return False, f"Token guardado. Edad aproximada: {age:.1f}h.\n{_selected_league_text()}"

    if t.startswith("/ligas"):
        return False, _cmd_list_leagues()

    if t.startswith("/liga"):
        return False, _cmd_select_league(text)

    if t.startswith("/informe"):
        if bot_token and chat_id:
            msg = _run_pre_market_cmd(analysis_only=True, bot_token=bot_token, chat_id=chat_id)
        else:
            msg = "Comando /informe requiere contexto de bot."
        return False, msg

    if t.startswith("/compraventa"):
        if bot_token and chat_id:
            msg = _run_pre_market_cmd(analysis_only=False, bot_token=bot_token, chat_id=chat_id)
        else:
            msg = "Comando /compraventa requiere contexto de bot."
        return False, msg

    token = _extract_token_from_text(text.strip())
    if not token:
        return False, "No detecte un token valido. Usa /help para instrucciones."

    save_token(token)
    leagues, err = _list_user_leagues()
    if not err and len(leagues) == 1:
        lg = leagues[0]
        save_selected_league(lg["id"], lg["name"], source="auto_single_league")
        sched, sched_err = build_market_schedule(
            lg["id"],
            timezone_name=os.getenv("TZ", "Europe/Madrid"),
        )
        schedule_txt = (
            "\n\n" + schedule_message(sched)
            if sched
            else f"\n\nNo pude calcular la hora de mercado ahora: {sched_err}"
        )
        return (
            True,
            "Token guardado correctamente en .laliga_token.\n"
            f"Liga auto-seleccionada: {lg['name']}"
            f"{schedule_txt}",
        )
    return (
        True,
        "Token guardado correctamente en .laliga_token.\n"
        "Si tienes varias ligas, usa /ligas y /liga para elegir la activa.",
    )


def _set_bot_commands(bot_token: str) -> None:
    """Configura el menú de comandos predefinidos en Telegram."""
    commands = [
        {"command": "informe", "description": "Generar informe de predicciones"},
        {"command": "compraventa", "description": "Ejecutar recomendaciones ventas+compras"},
        {"command": "ligas", "description": "Listar ligas disponibles"},
        {"command": "liga", "description": "Seleccionar liga por nombre"},
        {"command": "status", "description": "Estado del token"},
        {"command": "help", "description": "Ayuda y lista de comandos"},
    ]
    try:
        _api(bot_token, "setMyCommands", {"commands": commands})
        logger.info("Comandos del bot configurados.")
    except Exception as exc:
        logger.warning("No se pudieron configurar comandos del bot: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telegram bot para refrescar token LaLiga")
    parser.add_argument("--bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument(
        "--allowed-chat-id",
        default=os.getenv("TELEGRAM_ALLOWED_CHAT_ID", ""),
        help="Si se define, solo acepta mensajes de este chat_id",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=int(os.getenv("TOKEN_BOT_POLL_TIMEOUT", "50")),
    )
    return parser


def run_token_bot(
    *,
    bot_token: str,
    allowed_chat_id: str = "",
    poll_timeout: int = 50,
    notify_chat_id: str = "",
    stop_event: Event | None = None,
) -> None:
    if not bot_token:
        raise RuntimeError("Falta bot_token de Telegram.")

    allowed_chat_id = str(allowed_chat_id or "").strip()
    notify_chat_id = str(notify_chat_id or "").strip()
    offset = None

    if notify_chat_id:
        token_max_age = float(os.getenv("TOKEN_MAX_AGE_HOURS", "23"))
        status, age_h = _token_status(max_age_hours=token_max_age)
        if status == "missing":
            msg = "Fantasy Autopilot TOKEN AUSENTE" + GUIDA_RENOVACION_TOKEN
        elif status == "expired":
            msg = (
                f"Fantasy Autopilot TOKEN CADUCADO (edad ~{age_h:.1f}h)"
                + GUIDA_RENOVACION_TOKEN
            )
        elif status == "invalid":
            msg = "Fantasy Autopilot TOKEN INVALIDO" + GUIDA_RENOVACION_TOKEN
        else:
            msg = (
                "Token bot iniciado. Envíame token/URL para refrescar .laliga_token.\n"
                f"{_selected_league_text()}\n"
                "Puedes cambiarla con /ligas y /liga <nombre>."
            )
        send_telegram_message(bot_token, notify_chat_id, msg)

    _set_bot_commands(bot_token)

    logger.info("Telegram token bot iniciado. Token file: %s", TOKEN_FILE)
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Telegram token bot detenido por stop_event.")
            return
        try:
            payload = {"timeout": int(max(1, poll_timeout))}
            if offset is not None:
                payload["offset"] = offset
            updates = _api(bot_token, "getUpdates", payload).get("result", [])
            for upd in updates:
                offset = int(upd.get("update_id", 0)) + 1
                msg = upd.get("message") or upd.get("edited_message") or {}
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))
                text = (msg.get("text") or "").strip()

                if not chat_id or not text:
                    continue

                if allowed_chat_id and chat_id != allowed_chat_id:
                    logger.info("Mensaje ignorado de chat no autorizado: %s", chat_id)
                    continue

                changed, response = _handle_text(
                    text,
                    bot_token=bot_token,
                    chat_id=chat_id,
                )
                _api(
                    bot_token,
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": response,
                        "disable_web_page_preview": True,
                    },
                )
                if changed:
                    logger.info("Token actualizado desde Telegram chat_id=%s", chat_id)
        except KeyboardInterrupt:
            logger.info("Telegram token bot detenido por usuario.")
            return
        except Exception as exc:
            logger.warning("Error en polling Telegram: %s", exc)
            if stop_event:
                stop_event.wait(timeout=5)
            else:
                time.sleep(5)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    if not args.bot_token:
        raise RuntimeError("Falta --bot-token o TELEGRAM_BOT_TOKEN.")

    run_token_bot(
        bot_token=args.bot_token,
        allowed_chat_id=args.allowed_chat_id,
        poll_timeout=args.poll_timeout,
        notify_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        stop_event=None,
    )


if __name__ == "__main__":
    main()
