from __future__ import annotations

import logging
import os
import re
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 3900

# URL de login LaLiga Fantasy (Google) que redirige a jwt.ms.
# Se construye por partes y se valida al importar este módulo. La URL final
# tiene ~306 caracteres y NO se debe truncar bajo ningún concepto, porque al
# usuario se le envía por Telegram para iniciar el flujo OAuth de renovación.
_LOGIN_BASE = (
    "https://login.laliga.es/laligadspprob2c.onmicrosoft.com"
    "/oauth2/v2.0/authorize"
)
_LOGIN_PARAMS = (
    ("p", "b2c_1a_5ulaip_parametrized_signin"),
    ("client_id", "cf110827-e4a9-4d20-affb-8ea0c6f15f94"),
    ("redirect_uri", "https://jwt.ms"),
    ("response_type", "id_token"),
    ("scope", "openid%20cf110827-e4a9-4d20-affb-8ea0c6f15f94"),
    ("nonce", "laligafantasy"),
    ("response_mode", "fragment"),
)
# Parámetros OAuth obligatorios. Si alguno se pierde, el flujo falla:
#   - sin client_id  -> AAD no sabe a qué app apunta el login
#   - sin redirect_uri -> AAD no sabe a dónde redirigir el id_token
#   - sin response_type/scope -> AAD no emite id_token
#   - sin nonce -> AAD rechaza la petición (B2C lo exige)
#   - sin response_mode=fragment -> el token nunca llega a jwt.ms en el #fragment
LOGIN_REQUIRED_PARAMS = (
    "p",
    "client_id",
    "redirect_uri",
    "response_type",
    "scope",
    "nonce",
    "response_mode",
)


def _build_default_login_url() -> str:
    qs = "&".join(f"{k}={v}" for k, v in _LOGIN_PARAMS)
    return f"{_LOGIN_BASE}?{qs}"


def validate_login_url(url: str) -> tuple[bool, str]:
    """
    Valida que `url` sea una URL OAuth con todos los parámetros obligatorios.
    Devuelve (ok, motivo).
    """
    raw = (url or "").strip()
    if not raw:
        return False, "URL vacía"
    if not raw.startswith("https://"):
        return False, "La URL no usa https"
    try:
        parsed = urlparse(raw)
    except Exception as exc:
        return False, f"URL malformada: {exc}"
    if not parsed.netloc:
        return False, "URL sin host"

    query_params = parse_qs(parsed.query or "")
    missing = [k for k in LOGIN_REQUIRED_PARAMS if k not in query_params]
    if missing:
        return False, f"Faltan parámetros OAuth: {', '.join(missing)}"

    redirect_uris = query_params.get("redirect_uri") or []
    if not redirect_uris or not redirect_uris[0].startswith("https://jwt.ms"):
        return False, "redirect_uri debe apuntar a https://jwt.ms"

    response_modes = query_params.get("response_mode") or []
    if not response_modes or response_modes[0] != "fragment":
        return False, "response_mode debe ser 'fragment'"

    return True, "ok"


def _resolve_login_url() -> str:
    """
    Permite override por env var (`LALIGA_LOGIN_URL`) para que un cambio futuro
    en LaLiga no obligue a reconstruir la imagen Docker. Si la URL de override
    no pasa la validación, se cae a la URL canónica con un warning.
    """
    override = os.getenv("LALIGA_LOGIN_URL", "").strip()
    if override:
        ok, reason = validate_login_url(override)
        if ok:
            return override
        logger.warning(
            "LALIGA_LOGIN_URL inválida (%s). Uso la URL canónica.", reason
        )
    return _build_default_login_url()


LALIGA_LOGIN_URL = _resolve_login_url()

_url_ok, _url_reason = validate_login_url(LALIGA_LOGIN_URL)
if not _url_ok:
    # Si la URL canónica está rota, el bot no puede pedir renovación. Es mejor
    # fallar pronto y de forma ruidosa que enviar un enlace inservible.
    raise RuntimeError(
        "LALIGA_LOGIN_URL no es válida y no se puede pedir renovación de "
        f"token: {_url_reason}"
    )

GUIDA_RENOVACION_TOKEN = f"""
📋 CÓMO RENOVAR EL TOKEN

1️⃣ Abre este enlace en el navegador (cópialo entero, incluyendo todo lo que
   va detrás del último '&'):
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

    def _split_long_line(line: str) -> list[str]:
        if len(line) <= size:
            return [line]

        parts: list[str] = []
        rest = line
        while len(rest) > size:
            cut = rest.rfind(" ", 0, size + 1)
            if cut < int(size * 0.6):
                cut = size
            parts.append(rest[:cut].rstrip())
            rest = rest[cut:].lstrip()
        if rest:
            parts.append(rest)
        return parts

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for raw_line in text.splitlines():
        for line in _split_long_line(raw_line.rstrip()):
            add_len = len(line) + (1 if current_lines else 0)
            if current_lines and (current_len + add_len > size):
                chunk = "\n".join(current_lines).strip()
                if chunk:
                    chunks.append(chunk)
                current_lines = [line]
                current_len = len(line)
            else:
                if current_lines:
                    current_lines.append(line)
                    current_len += 1 + len(line)
                else:
                    current_lines = [line]
                    current_len = len(line)

    if current_lines:
        chunk = "\n".join(current_lines).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


_HTML_TAG_RE = re.compile(r"<\s*/?\s*([a-zA-Z][a-zA-Z0-9]*)[^>]*>")


def _strip_html_tags(text: str) -> str:
    """
    Quita tags HTML simples para enviar como texto plano. NO es un parser
    real, pero basta para el modo fallback cuando Telegram rechaza el HTML.
    """
    if not text:
        return ""
    return _HTML_TAG_RE.sub("", text)


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: str | None = None,
) -> None:
    """
    Envía un mensaje por Telegram Bot API.

    Guardarraíl: si Telegram rechaza el HTML (parse_entities 400), reintenta
    sin parse_mode con el texto sin tags. Es preferible que el usuario reciba
    la URL en plano a perder el mensaje completo.

    No levanta excepción al usuario final si falla: deja warning en logs.
    """
    if not bot_token or not chat_id or not text:
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    for part in _chunk_text(text):
        try:
            payload: dict = {
                "chat_id": str(chat_id),
                "text": part,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode

            resp = requests.post(url, json=payload, timeout=25)
            if resp.ok:
                continue

            body = resp.text or ""
            is_parse_error = (
                resp.status_code == 400
                and parse_mode
                and ("can't parse entities" in body.lower() or "parse_mode" in body.lower())
            )
            logger.warning(
                "Telegram sendMessage failed: %s %s",
                resp.status_code,
                body[:300],
            )

            if is_parse_error:
                # Fallback: reenviamos sin parse_mode con el texto limpio.
                fallback_payload = {
                    "chat_id": str(chat_id),
                    "text": _strip_html_tags(part),
                    "disable_web_page_preview": True,
                }
                try:
                    fb = requests.post(url, json=fallback_payload, timeout=25)
                    if not fb.ok:
                        logger.warning(
                            "Telegram fallback sendMessage failed: %s %s",
                            fb.status_code,
                            (fb.text or "")[:300],
                        )
                    else:
                        logger.info(
                            "Telegram sendMessage recuperado en fallback "
                            "(sin parse_mode)."
                        )
                except Exception as exc:
                    logger.warning("Telegram fallback sendMessage error: %s", exc)
        except Exception as exc:
            logger.warning("Telegram sendMessage error: %s", exc)
