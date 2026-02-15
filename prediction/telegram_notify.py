from __future__ import annotations

import logging
from typing import Iterable

import requests

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 3900

# URL de login LaLiga Fantasy (Google) que redirige a jwt.ms
LALIGA_LOGIN_URL = (
    "https://login.laliga.es/laligadspprob2c.onmicrosoft.com/oauth2/v2.0/authorize"
    "?p=b2c_1a_5ulaip_parametrized_signin"
    "&client_id=cf110827-e4a9-4d20-affb-8ea0c6f15f94"
    "&redirect_uri=https://jwt.ms"
    "&response_type=id_token"
    "&scope=openid%20cf110827-e4a9-4d20-affb-8ea0c6f15f94"
    "&nonce=laligafantasy"
    "&response_mode=fragment"
)

GUIDA_RENOVACION_TOKEN = f"""
📋 CÓMO RENOVAR EL TOKEN

1️⃣ Abre este enlace en el navegador:
{LALIGA_LOGIN_URL}

2️⃣ Inicia sesión con tu cuenta de LaLiga Fantasy (Google).

3️⃣ Tras el login te redirigirá a jwt.ms.
   Copia la URL COMPLETA de la barra de direcciones
   (empieza con https://jwt.ms/#id_token=eyJ...)

4️⃣ Envía esa URL a este bot de Telegram.
   También vale enviar solo el JWT (eyJ...).
"""


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
