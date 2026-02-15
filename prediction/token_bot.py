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
from datetime import datetime, timezone
from threading import Event
from urllib.parse import parse_qs, urlparse

import requests

from laliga_fantasy_client import TOKEN_FILE, load_token, save_token
from prediction.telegram_notify import send_telegram_message

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


def _handle_text(text: str) -> tuple[bool, str]:
    t = text.strip()
    if t.lower().startswith("/start") or t.lower().startswith("/help"):
        return (
            False,
            "Enviame un token JWT (eyJ...) o la URL completa de jwt.ms "
            "con id_token/access_token y lo guardare en .laliga_token.",
        )
    if t.lower().startswith("/status"):
        age = _token_age_hours()
        if age is None:
            return False, "No hay token valido guardado."
        return False, f"Token guardado. Edad aproximada: {age:.1f}h."

    token = _extract_token_from_text(t)
    if not token:
        return False, "No detecte un token valido. Usa /help para instrucciones."

    save_token(token)
    return True, "Token guardado correctamente en .laliga_token."


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
        send_telegram_message(
            bot_token,
            notify_chat_id,
            "Token bot iniciado. Enviame token/URL para refrescar .laliga_token.",
        )

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

                changed, response = _handle_text(text)
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
