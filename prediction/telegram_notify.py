from __future__ import annotations

import logging
from typing import Iterable

import requests

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 3900


def _chunk_text(text: str, size: int = TELEGRAM_MAX_LEN) -> Iterable[str]:
    text = (text or "").strip()
    if not text:
        return []
    return (text[i : i + size] for i in range(0, len(text), size))


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    """
    Envía un mensaje por Telegram Bot API.
    No levanta excepción al usuario final si falla: deja warning en logs.
    """
    if not bot_token or not chat_id or not text:
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    for part in _chunk_text(text):
        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": str(chat_id),
                    "text": part,
                    "disable_web_page_preview": True,
                },
                timeout=25,
            )
            if not resp.ok:
                logger.warning(
                    "Telegram sendMessage failed: %s %s",
                    resp.status_code,
                    resp.text[:300],
                )
        except Exception as exc:
            logger.warning("Telegram sendMessage error: %s", exc)
