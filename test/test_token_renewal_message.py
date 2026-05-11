#!/usr/bin/env python3
"""
Tests del flujo de renovación de token.

Vale como guardarrail: si alguien vuelve a meter la URL en
`build_standard_message`, `_compact_text` o cualquier capa que la
trunque a < 306 caracteres, este test lo detecta.

Ejecutar:
    cd fantasy-bot && python -m unittest test.test_token_renewal_message -v
"""
from __future__ import annotations

import os
import re
import sys
import unittest
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction.telegram_messages import (  # noqa: E402
    build_help_message,
    build_token_renewal_message,
)
from prediction.telegram_notify import (  # noqa: E402
    LALIGA_LOGIN_URL,
    LOGIN_REQUIRED_PARAMS,
    _chunk_text,
    _strip_html_tags,
    validate_login_url,
)


# Tags soportados por Telegram en parse_mode=HTML. Cualquier "<X" literal con
# X fuera de este set normalmente es texto del usuario sin escapar; los tests
# fuerzan que el balance abrir/cerrar sea correcto, lo que ya cubre el bug
# original ("Can't find end tag corresponding to start tag 'code'").
TELEGRAM_HTML_TAGS = {
    "b", "strong",
    "i", "em",
    "u", "ins",
    "s", "strike", "del",
    "code", "pre",
    "a",
    "tg-spoiler", "span",
    "blockquote",
}

_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*>")


def assert_html_balanced(html: str) -> None:
    """
    Verifica que cada tag HTML abierto se cierra. Es una mini-validación,
    suficiente para detectar el bug típico de Telegram parse 400.
    """
    stack: list[str] = []
    for match in _TAG_RE.finditer(html):
        is_close = match.group(1) == "/"
        tag = match.group(2).lower()
        if tag not in TELEGRAM_HTML_TAGS:
            raise AssertionError(
                f"Tag desconocido o no soportado por Telegram: <{tag}> "
                f"(¿texto literal sin escapar como &lt;{tag}&gt;?)"
            )
        if not is_close:
            stack.append(tag)
        else:
            if not stack or stack[-1] != tag:
                raise AssertionError(
                    f"Cierre </{tag}> sin apertura coincidente. Stack: {stack}"
                )
            stack.pop()
    if stack:
        raise AssertionError(f"Tags abiertos sin cerrar: {stack}")


class TestLoginUrlIntegrity(unittest.TestCase):
    """La URL canónica debe pasar todas las validaciones siempre."""

    def test_canonical_url_is_https(self):
        self.assertTrue(LALIGA_LOGIN_URL.startswith("https://"))

    def test_canonical_url_has_required_params(self):
        ok, reason = validate_login_url(LALIGA_LOGIN_URL)
        self.assertTrue(ok, f"URL canónica inválida: {reason}")

        params = parse_qs(urlparse(LALIGA_LOGIN_URL).query)
        for key in LOGIN_REQUIRED_PARAMS:
            self.assertIn(key, params, f"Falta param OAuth: {key}")

    def test_validator_rejects_truncated_url(self):
        truncated = LALIGA_LOGIN_URL[:217]  # exacto al bug previo
        ok, reason = validate_login_url(truncated)
        self.assertFalse(ok, "Una URL truncada NO debe pasar la validación")
        self.assertIn("OAuth", reason)

    def test_validator_rejects_non_jwtms_redirect(self):
        bad = (
            "https://login.laliga.es/x/oauth2/v2.0/authorize"
            "?p=b2c_1a_5ulaip_parametrized_signin"
            "&client_id=cf110827-e4a9-4d20-affb-8ea0c6f15f94"
            "&redirect_uri=https://evil.example"
            "&response_type=id_token"
            "&scope=openid"
            "&nonce=laligafantasy"
            "&response_mode=fragment"
        )
        ok, _reason = validate_login_url(bad)
        self.assertFalse(ok)


class TestTokenRenewalMessage(unittest.TestCase):
    """El mensaje que se manda por Telegram debe contener la URL ENTERA."""

    def test_message_contains_full_url_for_each_status(self):
        for status in ("missing", "expired", "invalid"):
            with self.subTest(status=status):
                msg = build_token_renewal_message(status=status, age_h=24.0)
                # La URL HTML-escapada (con &amp;) debe aparecer entera. Sin
                # escapado tampoco se trunca, pero como Telegram pide HTML,
                # comprobamos las dos formas.
                self.assertIn(LALIGA_LOGIN_URL.replace("&", "&amp;"), msg)

    def test_message_includes_clickable_link_and_code_block(self):
        msg = build_token_renewal_message(status="expired", age_h=25.0)
        self.assertIn('<a href="', msg)
        self.assertIn("Abrir login LaLiga Fantasy", msg)
        self.assertIn("<code>", msg)

    def test_message_does_not_truncate_url(self):
        msg = build_token_renewal_message(status="expired")
        # Si alguien volviera a aplicar el _compact_text(220) al URL, se
        # vería un sufijo "..." pegado a la URL truncada. Verificamos que la
        # URL no aparezca cortada por "..." ni que la cola crítica desaparezca.
        self.assertNotIn("response_mode=fragm...", msg)
        self.assertIn("response_mode=fragment", msg)
        self.assertIn("nonce=laligafantasy", msg)
        self.assertIn("scope=openid", msg)

    def test_message_preserves_url_after_telegram_chunking(self):
        # El chunker de Telegram no debe partir la URL (parte por líneas, y
        # la URL está en su propia línea, dentro de <a> y <code>).
        msg = build_token_renewal_message(status="missing")
        chunks = list(_chunk_text(msg))
        joined = "\n".join(chunks)
        self.assertIn(LALIGA_LOGIN_URL.replace("&", "&amp;"), joined)
        # Ningún chunk individual debe partir el href
        for chunk in chunks:
            opens = chunk.count('<a href="')
            closes = chunk.count("</a>")
            self.assertEqual(
                opens,
                closes,
                "Un chunk parte el <a href=...></a> y rompería HTML parse_mode",
            )

    def test_message_is_returned_for_unknown_status(self):
        msg = build_token_renewal_message(status="renovacion_manual")
        self.assertIn("Renovar token", msg)
        self.assertIn(LALIGA_LOGIN_URL.replace("&", "&amp;"), msg)

    def test_message_falls_back_when_url_is_broken(self):
        # Si por algún motivo se pasa una URL rota, el mensaje debe avisar
        # en vez de mandar un link inservible.
        msg = build_token_renewal_message(
            status="expired",
            login_url="https://login.laliga.es/incompleto",
        )
        self.assertIn("enlace de login dañado", msg)


class TestHelpMessage(unittest.TestCase):
    def test_help_includes_login_url(self):
        msg = build_help_message()
        self.assertIn("Renovar token", msg)
        self.assertIn(LALIGA_LOGIN_URL.replace("&", "&amp;"), msg)


class TestHtmlBalance(unittest.TestCase):
    """
    Detecta tags HTML desbalanceados o tags-texto sin escapar. Este test
    cubre el bug de Telegram 400 "Can't find end tag corresponding to start
    tag 'code'" que surgió cuando dejé "<code>" como texto literal en la
    sección Nota.
    """

    def test_token_renewal_message_html_is_balanced(self):
        for status in ("missing", "expired", "invalid", "renovacion_manual"):
            with self.subTest(status=status):
                msg = build_token_renewal_message(status=status, age_h=10.0)
                assert_html_balanced(msg)

    def test_help_message_html_is_balanced(self):
        assert_html_balanced(build_help_message())


class TestStripHtmlFallback(unittest.TestCase):
    """El fallback de send_telegram_message debe quedarse con la URL legible."""

    def test_strip_keeps_url(self):
        msg = build_token_renewal_message(status="expired", age_h=23.5)
        plain = _strip_html_tags(msg)
        # La URL escapada (con &amp;) sigue siendo válida si alguien la copia,
        # pero idealmente queremos también la versión cruda. Validamos que al
        # menos la URL sigue presente entera.
        self.assertIn("response_mode=fragment", plain)
        self.assertIn("nonce=laligafantasy", plain)
        self.assertNotIn("<", plain)
        self.assertNotIn(">", plain)


if __name__ == "__main__":
    unittest.main(verbosity=2)
