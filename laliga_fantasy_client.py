"""
LaLiga Fantasy API Client
=========================
Cliente Python para la API de La Liga Fantasy (Relevo).

CÓMO OBTENER EL TOKEN (elige una):
===================================

    OPCIÓN A - Login con Google (recomendado):
        client = LaLigaFantasyClient.from_google_login(league_id="abc123")

    OPCIÓN B - Email/Password (solo cuentas NO Google/Apple):
        Si tu cuenta usa email y contraseña (no Google), funciona directo:
            client = LaLigaFantasyClient.from_email_password(
                username="email@ejemplo.com", password="contraseña",
                league_id="abc123",
            )

    OPCIÓN C - Token manual:
        Si ya tienes el token por cualquier vía:
            client = LaLigaFantasyClient.from_token("eyJ...", league_id="abc123")

Uso rápido:
-----------
    from laliga_fantasy_client import LaLigaFantasyClient

    client = LaLigaFantasyClient.from_token("eyJ...", league_id="abc123")
    jugadores = client.get_players()
"""

import json
import logging
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de la API
# ---------------------------------------------------------------------------
BASE_URL = "https://api-fantasy.llt-services.com"
AUTH_POLICY = "B2C_1A_ResourceOwnerv2"
TOKEN_FILE = pathlib.Path(".laliga_token")

# Headers que usa la app
DEFAULT_HEADERS = {
    "User-Agent": "okhttp/4.12.0",
    "X-App": "Fantasy",
    "X-Lang": "es",
    "Accept": "application/json",
}

POSITION_MAP = {
    "1": "POR",  # Portero
    "2": "DEF",  # Defensa
    "3": "MED",  # Centrocampista
    "4": "DEL",  # Delantero
    "5": "ENT",  # Entrenador
}


# ===========================================================================
# Azure B2C OAuth config (extraída del APK)
# ===========================================================================
B2C_TENANT = "laligadspprob2c.onmicrosoft.com"
B2C_POLICY = "b2c_1a_5ulaip_parametrized_signin"
B2C_CLIENT_ID = "cf110827-e4a9-4d20-affb-8ea0c6f15f94"
B2C_AUTHORIZE_URL = f"https://login.laliga.es/{B2C_TENANT}/oauth2/v2.0/authorize"
B2C_TOKEN_URL = f"https://login.laliga.es/{B2C_TENANT}/oauth2/v2.0/token"
B2C_REDIRECT_URI = "authredirect://com.lfp.laligafantasy"
B2C_SCOPE = f"openid {B2C_CLIENT_ID} offline_access"


def google_login_open_browser() -> None:
    """
    Paso 1 del login con Google: abre el navegador para autenticarse.
    Después el usuario debe copiar la URL y pasarla con --token.
    """
    import urllib.parse
    import webbrowser

    jwt_ms_redirect = "https://jwt.ms"
    auth_params = {
        "p": B2C_POLICY,
        "client_id": B2C_CLIENT_ID,
        "redirect_uri": jwt_ms_redirect,
        "response_type": "id_token",
        "scope": f"openid {B2C_CLIENT_ID}",
        "nonce": "laligafantasy",
        "response_mode": "fragment",
    }
    auth_url = f"{B2C_AUTHORIZE_URL}?{urllib.parse.urlencode(auth_params)}"

    print()
    print("=" * 70)
    print("  LOGIN CON GOOGLE — Paso 1")
    print("=" * 70)
    print()
    print("  Se abrirá tu navegador para iniciar sesión con Google.")
    print("  Después del login, llegarás a una página 'jwt.ms'.")
    print()
    print("  Copia la URL COMPLETA de la barra de direcciones")
    print("  (empieza con https://jwt.ms/#id_token=eyJ...)")
    print()
    print("  Luego ejecuta:")
    print()
    print('    python laliga_fantasy_client.py --token "URL_QUE_COPIASTE"')
    print()
    print("-" * 70)
    print("  Abriendo navegador...")

    webbrowser.open(auth_url)

    print()
    print("  Listo. Haz login y luego usa --token con la URL.")
    print()


def google_login_flow(redirect_url: str | None = None) -> dict:
    """
    Flujo completo de login con Google vía Azure B2C.

    Si se pasa redirect_url, se usa directamente (sin input()).
    Si no se pasa, abre el navegador y pide la URL por stdin.

    Returns:
        {"access_token": "..."}
    """
    import urllib.parse

    if not redirect_url:
        raise ValueError("No se proporcionó la URL del token.")

    # El token viene en el fragment de la URL: https://jwt.ms/#id_token=eyJ...
    parsed = urllib.parse.urlparse(redirect_url)

    # Intentar extraer de fragment
    fragment_params = urllib.parse.parse_qs(parsed.fragment)
    token = None

    if "id_token" in fragment_params:
        token = fragment_params["id_token"][0]
    elif "access_token" in fragment_params:
        token = fragment_params["access_token"][0]

    # Si no está en el fragment, intentar en query params
    if not token:
        query_params = urllib.parse.parse_qs(parsed.query)
        if "id_token" in query_params:
            token = query_params["id_token"][0]
        elif "access_token" in query_params:
            token = query_params["access_token"][0]
        elif "code" in query_params:
            # Si vino un code en vez de token, intercambiarlo
            code = query_params["code"][0]
            logger.info("Código obtenido, intercambiando por token...")
            resp = requests.post(
                B2C_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": B2C_CLIENT_ID,
                    "code": code,
                    "redirect_uri": jwt_ms_redirect,
                    "scope": B2C_SCOPE,
                },
                params={"p": B2C_POLICY},
            )
            if resp.status_code == 200:
                return resp.json()
            # Intentar con la API
            resp2 = requests.post(
                f"{BASE_URL}/login/v3/token",
                data={"code": code, "policy": B2C_POLICY},
                headers=DEFAULT_HEADERS,
            )
            if resp2.status_code == 200:
                return resp2.json()
            raise RuntimeError(
                f"Error intercambiando código:\n"
                f"  B2C: {resp.status_code} - {resp.text[:300]}\n"
                f"  API: {resp2.status_code} - {resp2.text[:300]}"
            )

    # Si el usuario pegó directamente un token (empieza con eyJ)
    if not token and redirect_url.startswith("eyJ"):
        token = redirect_url

    if not token:
        raise ValueError(
            "No se encontró token en la URL.\n"
            "Asegúrate de copiar la URL completa de la barra de direcciones\n"
            f"URL recibida: {redirect_url[:100]}..."
        )

    logger.info("Token obtenido correctamente.")
    return {"access_token": token}


# ===========================================================================
# Endpoints PÚBLICOS (no requieren token)
# ===========================================================================
class LaLigaFantasyPublic:
    """
    Endpoints de la API que NO requieren autenticación.
    Funcionan sin token, sin proxy, sin parchear nada.

    Datos disponibles:
        - Todos los jugadores de LaLiga (675+)
        - Detalle completo de cada jugador (stats por jornada)
        - Historial de valor de mercado de cada jugador

    Datos que SÍ requieren token (usar LaLigaFantasyClient):
        - Ligas privadas, ranking, mercado de tu liga, plantillas
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _get(self, url: str) -> dict | list:
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()

    def get_players_raw(self) -> list[dict]:
        """
        GET /api/v3/players  (PÚBLICO - sin token)

        Devuelve TODOS los jugadores de LaLiga con datos básicos.

        Response por elemento:
            {
                "id": "68",
                "nickname": "Unai Simón",
                "positionId": "1",
                "playerStatus": "ok",
                "marketValue": "9174877",
                "points": 134,
                "averagePoints": 5.82,
                "lastSeasonPoints": "166",
                "team": {
                    "id": "3",
                    "name": "Athletic Club",
                    "slug": "athletic-club",
                    "badgeColor": "https://..."
                },
                "images": {...}
            }
        """
        url = f"{BASE_URL}/api/v3/players"
        logger.info("Obteniendo todos los jugadores (endpoint público)...")
        return self._get(url)

    def get_players(self) -> list[dict]:
        """Versión formateada de todos los jugadores."""
        return [_format_player(p) for p in self.get_players_raw()]

    def get_player_detail(self, player_id: int) -> dict:
        """
        GET /api/v3/player/{player_id}  (PÚBLICO - sin token)

        Detalle completo de un jugador: stats por jornada, equipo, etc.

        Response:
            {
                "id": "68",
                "nickname": "Unai Simón",
                "positionId": "1",
                "position": "Portero",
                "playerStatus": "ok",
                "marketValue": 9174877,
                "points": 134,
                "averagePoints": 5.82,
                "team": {...},
                "playerStats": [
                    {
                        "weekNumber": 1,
                        "totalPoints": 6,
                        "stats": {
                            "mins_played": [90, 2],
                            "goals": [0, 0],
                            "goal_assist": [0, 0],
                            ...
                        }
                    },
                    ...
                ]
            }
        """
        url = f"{BASE_URL}/api/v3/player/{player_id}"
        logger.info("Obteniendo detalle del jugador %s (público)...", player_id)
        return self._get(url)

    def get_price_history(self, player_id: int) -> list[dict]:
        """
        GET /api/v3/player/{player_id}/market-value  (PÚBLICO - sin token)

        Historial completo de valor de mercado del jugador.

        Response:
            [
                {
                    "lfpId": 3000068,
                    "marketValue": 28500000,
                    "date": "2025-07-03T00:00:00+02:00",
                    "bids": 0
                },
                ...
            ]
        """
        url = f"{BASE_URL}/api/v3/player/{player_id}/market-value"
        return self._get(url)

    def get_all_price_histories(self, player_ids: list[int] | None = None, max_workers: int = 5) -> list[dict]:
        """
        Descarga historial de precios de múltiples jugadores en paralelo.
        Si no se pasan player_ids, obtiene todos los jugadores primero.
        """
        if player_ids is None:
            players = self.get_players()
            player_ids = [p["player_id"] for p in players]

        all_h = []
        logger.info("Descargando historial de %d jugadores...", len(player_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.get_price_history, pid): pid for pid in player_ids}
            for f in as_completed(futs):
                pid = futs[f]
                try:
                    for e in f.result():
                        all_h.append({
                            "player_id": pid,
                            "market_value": _to_int(e.get("marketValue")),
                            "date": e.get("date"),
                        })
                except Exception as exc:
                    logger.error("Error jugador %s: %s", pid, exc)
        logger.info("Historial: %d registros.", len(all_h))
        return all_h


# ===========================================================================
# Gestión de token persistente
# ===========================================================================
def save_token(token: str, filepath: pathlib.Path = TOKEN_FILE) -> None:
    """Guarda el token en disco con timestamp para saber cuándo expira."""
    data = {
        "access_token": token,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "note": "Token válido ~24h desde saved_at",
    }
    filepath.write_text(json.dumps(data, indent=2))
    logger.info("Token guardado en %s", filepath)


def load_token(filepath: pathlib.Path = TOKEN_FILE) -> Optional[str]:
    """
    Carga el token de disco. Devuelve None si no existe o ha expirado (>23h).
    """
    if not filepath.exists():
        return None

    try:
        data = json.loads(filepath.read_text())
        token = data["access_token"]
        saved_at = datetime.fromisoformat(data["saved_at"])
        age_hours = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600

        if age_hours > 23:
            logger.warning("Token guardado hace %.1fh — probablemente expirado.", age_hours)
            return None

        logger.info("Token cargado de %s (edad: %.1fh)", filepath, age_hours)
        return token
    except (json.JSONDecodeError, KeyError):
        return None



# ===========================================================================
# Cliente principal
# ===========================================================================
class LaLigaFantasyClient:
    """
    Cliente para la API de La Liga Fantasy.

    Crear con uno de los class methods:
        - LaLigaFantasyClient.from_google_login(...)   ← RECOMENDADO
        - LaLigaFantasyClient.from_token(...)
        - LaLigaFantasyClient.from_email_password(...)
        - LaLigaFantasyClient.from_saved_token()
    """

    def __init__(self, token: str, league_id: str, manager_id: str = ""):
        self.token = token
        self.league_id = league_id
        self.manager_id = manager_id

        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    # -------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------

    @classmethod
    def from_token(
        cls,
        token: str,
        league_id: str = "",
        manager_id: str = "",
        save: bool = True,
    ) -> "LaLigaFantasyClient":
        """
        Crear cliente con un token obtenido de la app (proxy, HTTP Toolkit, etc).

        Si save=True, guarda el token en .laliga_token para reutilizarlo
        sin tener que volver a capturarlo (válido ~24h).
        """
        league_id = league_id or os.getenv("LALIGA_LEAGUE_ID", "")
        manager_id = manager_id or os.getenv("LALIGA_MANAGER_ID", "")

        if not token:
            raise ValueError("Se requiere un token válido.")

        if save:
            save_token(token)

        logger.info("Cliente creado con token proporcionado.")
        return cls(token=token, league_id=league_id, manager_id=manager_id)

    @classmethod
    def from_saved_token(
        cls,
        league_id: str = "",
        manager_id: str = "",
    ) -> "LaLigaFantasyClient":
        """
        Carga el token guardado previamente en .laliga_token.
        Falla si no existe o ha expirado (>23h).
        """
        league_id = league_id or os.getenv("LALIGA_LEAGUE_ID", "")
        manager_id = manager_id or os.getenv("LALIGA_MANAGER_ID", "")

        token = load_token()
        if not token:
            raise RuntimeError(
                "No hay token guardado o ha expirado.\n"
                "Usa --google para login con Google, o proporciona uno con:\n"
                "LaLigaFantasyClient.from_token('eyJ...')"
            )

        return cls(token=token, league_id=league_id, manager_id=manager_id)

    @classmethod
    def from_google_login(
        cls,
        league_id: str = "",
        manager_id: str = "",
    ) -> "LaLigaFantasyClient":
        """
        Login con Google vía Azure B2C.

        Abre tu navegador, haces login con Google, copias la URL del
        redirect y el script intercambia el código por un token.

        No necesita proxy, ni parchear APK, ni Selenium.
        """
        league_id = league_id or os.getenv("LALIGA_LEAGUE_ID", "")
        manager_id = manager_id or os.getenv("LALIGA_MANAGER_ID", "")

        token_data = google_login_flow()
        access_token = token_data.get("access_token") or token_data.get("id_token", "")

        if not access_token:
            raise RuntimeError(f"No se obtuvo access_token. Respuesta: {token_data}")

        save_token(access_token)
        logger.info("Login con Google exitoso. Token guardado.")
        return cls(token=access_token, league_id=league_id, manager_id=manager_id)

    @classmethod
    def from_email_password(
        cls,
        username: str = "",
        password: str = "",
        league_id: str = "",
        manager_id: str = "",
    ) -> "LaLigaFantasyClient":
        """
        Autenticación con email/password (solo cuentas NO Google/Apple).

        Flujo:
            1. POST /login/v3/email/auth
               Body (form-urlencoded):
                   policy=B2C_1A_ResourceOwnerv2
                   username=<email>
                   password=<contraseña>
               Response: {"code": "eyJ..."}

            2. POST /login/v3/email/token
               Body (form-urlencoded):
                   code=<code>
                   policy=B2C_1A_ResourceOwnerv2
               Response: {"access_token": "eyJ...", "token_type": "Bearer", ...}

        NOTA: Este método NO funciona si tu cuenta usa Google/Apple para login.
        """
        username = username or os.getenv("LALIGA_USERNAME", "")
        password = password or os.getenv("LALIGA_PASSWORD", "")
        league_id = league_id or os.getenv("LALIGA_LEAGUE_ID", "")
        manager_id = manager_id or os.getenv("LALIGA_MANAGER_ID", "")

        if not username or not password:
            raise ValueError(
                "Se requieren username y password.\n"
                "Pásalos como parámetros o define LALIGA_USERNAME y LALIGA_PASSWORD."
            )

        logger.info("Autenticando con email/password...")

        # Paso 1: obtener código
        auth_url = f"{BASE_URL}/login/v3/email/auth"
        auth_payload = {
            "policy": AUTH_POLICY,
            "username": username,
            "password": password,
        }
        resp = requests.post(auth_url, data=auth_payload, headers=DEFAULT_HEADERS)
        resp.raise_for_status()
        code = resp.json()["code"]

        # Paso 2: intercambiar código por token
        token_url = f"{BASE_URL}/login/v3/email/token"
        token_payload = {"code": code, "policy": AUTH_POLICY}
        resp = requests.post(token_url, data=token_payload, headers=DEFAULT_HEADERS)
        resp.raise_for_status()
        access_token = resp.json()["access_token"]

        save_token(access_token)
        logger.info("Token obtenido y guardado (válido ~24h).")
        return cls(token=access_token, league_id=league_id, manager_id=manager_id)

    # -------------------------------------------------------------------
    # Métodos HTTP internos
    # -------------------------------------------------------------------

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def _get(self, url: str) -> dict | list:
        """
        GET autenticado. Maneja redirects HTTPS→HTTP sin perder el token
        (requests por seguridad elimina el header Authorization en esos casos).
        """
        resp = self.session.get(url, headers=self._auth_headers(), allow_redirects=False)
        # Si hay redirect, seguirlo manualmente con el header auth
        while resp.status_code in (301, 302, 303, 307, 308):
            redirect_url = resp.headers.get("Location", "")
            if redirect_url:
                # Forzar HTTPS si la API redirige a HTTP
                if redirect_url.startswith("http://"):
                    redirect_url = "https://" + redirect_url[7:]
                resp = self.session.get(redirect_url, headers=self._auth_headers(), allow_redirects=False)
            else:
                break
        resp.raise_for_status()
        return resp.json()

    def _post(self, url: str, json_body: dict) -> dict | list:
        """POST autenticado. Usado para operaciones de mercado (vender, comprar)."""
        headers = {
            **self._auth_headers(),
            "Content-Type": "application/json",
        }
        resp = self.session.post(url, json=json_body, headers=headers)
        if not resp.ok:
            try:
                err_body = resp.json() if resp.content else resp.text[:500]
            except Exception:
                err_body = resp.text[:500] if resp.text else ""
            logger.warning("POST %s → %s | body: %s | response: %s", url, resp.status_code, json_body, err_body)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _put(self, url: str, json_body: dict) -> dict | list:
        """PUT autenticado. Usado para update de recursos (p. ej., alineación)."""
        headers = {
            **self._auth_headers(),
            "Content-Type": "application/json",
        }
        resp = self.session.put(url, json=json_body, headers=headers)
        if not resp.ok:
            try:
                err_body = resp.json() if resp.content else resp.text[:500]
            except Exception:
                err_body = resp.text[:500] if resp.text else ""
            logger.warning("PUT %s → %s | body: %s | response: %s", url, resp.status_code, json_body, err_body)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # -------------------------------------------------------------------
    # Operaciones de mercado (POST - endpoints extraídos del APK)
    # -------------------------------------------------------------------
    def sell_player_phase1(self, player_team_id: str, price: int) -> dict:
        """
        Fase 1 de venta: publicar jugador en el mercado.

        POST /api/v3/league/{leagueId}/market/sell (o /leagues/... según API real)
        Body: {"salePrice": int, "playerTeamId": str}

        player_team_id: ID del jugador en tu plantilla (PlayerTeam), no player_id.
        Body JSON: {"playerId": "<playerTeamId>", "salePrice": int} — la API usa "playerId" no "playerTeamId".
        """
        # APK SellRequest: @SerializedName("playerId") sobre playerTeamId — la API espera "playerId"
        body = {"playerId": str(player_team_id), "salePrice": int(price)}
        logger.info("Publicando jugador %s en mercado (precio %s)...", player_team_id, price)
        url = f"{BASE_URL}/api/v3/league/{self.league_id}/market/sell"
        return self._post(url, body)

    def sell_player_phase2_accept_league_offer(
        self, market_player_id: str, offer_id: str
    ) -> dict:
        """
        Fase 2 de venta: aceptar la oferta de la liga tras el cierre del mercado.

        POST /api/v4/league/{leagueId}/market/{marketPlayerId}/offer/{offerId}/accept

        market_player_id: ID del ítem en el mercado (MarketPlayer).
        offer_id: ID de la oferta que hace la liga tras el cierre.
        """
        url = f"{BASE_URL}/api/v4/league/{self.league_id}/market/{market_player_id}/offer/{offer_id}/accept"
        logger.info("Aceptando oferta %s por item mercado %s...", offer_id, market_player_id)
        return self._post(url, {})

    def buy_player_bid(
        self,
        market_player_id: str,
        amount: int,
        player_id: int | None = None,
    ) -> dict:
        """
        Pujar por un jugador del mercado de la liga (tipo libre).

        POST /api/v3/league/{leagueId}/market/{marketPlayerId}/bid
        Body: {"money": int}
        """
        item_id = str(market_player_id)
        logger.info("Pujando %s por item %s...", amount, item_id)

        def _resolve_market_item(preferred_id: str, pid: int | None) -> dict:
            """
            Revalida el item en mercado y devuelve metadata útil para pujar.
            """
            fallback = {
                "item_id": str(preferred_id),
                "discr": "",
                "bid_id": "",
                "sale_price": None,
                "market_value": None,
                "bid_money": None,
            }

            try:
                raw = self.get_daily_market_raw()
            except Exception:
                return fallback

            def _build(it: dict) -> dict:
                bid = it.get("bid") or {}
                pm = it.get("playerMaster", {})
                bid_money = None
                for key in ("money", "amount", "bidMoney", "value"):
                    bid_money = _to_int(bid.get(key))
                    if bid_money is not None:
                        break

                return {
                    "item_id": str(it.get("id", preferred_id)),
                    "discr": str(it.get("discr", "")),
                    "bid_id": str(bid.get("id") or ""),
                    "sale_price": _to_int(it.get("salePrice")),
                    "market_value": _to_int(pm.get("marketValue")),
                    "bid_money": bid_money,
                }

            # 1) Intentar por ID exacto.
            for it in raw:
                if str(it.get("id", "")) == str(preferred_id):
                    return _build(it)

            # 2) Fallback por player_id (si cambió el market item id).
            if pid is not None:
                for it in raw:
                    pm = it.get("playerMaster", {})
                    if _to_int(pm.get("id")) == _to_int(pid):
                        return _build(it)

            return fallback

        def _compute_bid_money(requested: int, market_info: dict) -> int:
            money = int(requested)
            sale_price = _to_int(market_info.get("sale_price")) or 0
            market_value = _to_int(market_info.get("market_value")) or 0
            current_bid = _to_int(market_info.get("bid_money")) or 0

            minimum = max(sale_price, market_value)
            if current_bid > 0:
                minimum = max(minimum, current_bid + 1)
            return max(money, minimum)

        def _place_bid(market_id: str, money: int) -> dict:
            url = f"{BASE_URL}/api/v3/league/{self.league_id}/market/{market_id}/bid"
            return self._post(url, {"money": int(money)})

        def _http_error_info(exc: requests.exceptions.HTTPError) -> tuple[int | None, str]:
            status = None
            err_code = ""
            resp = getattr(exc, "response", None)
            if resp is None:
                return status, err_code
            status = resp.status_code
            try:
                data = resp.json() if resp.content else {}
                err_code_raw = data.get("errorCode", "") or data.get("code", "")
                err_code = str(err_code_raw)
            except Exception:
                err_code = ""
            return status, err_code

        market_info = _resolve_market_item(item_id, player_id)

        # Si el item ahora es de manager (no liga), la operación correcta es /offer.
        if market_info.get("discr") and market_info.get("discr") != "marketPlayerLeague":
            offer_money = _compute_bid_money(int(amount), market_info)
            offer_url = (
                f"{BASE_URL}/api/v3/league/{self.league_id}/market/"
                f"{market_info['item_id']}/offer"
            )
            logger.info(
                "Item %s no es de puja (discr=%s). Enviando /offer...",
                market_info["item_id"], market_info["discr"],
            )
            try:
                return self._post(offer_url, {"money": int(offer_money)})
            except requests.exceptions.HTTPError:
                # Si falla, continuamos con flujo de puja por compatibilidad.
                pass

        bid_money = _compute_bid_money(int(amount), market_info)
        if bid_money != int(amount):
            logger.info(
                "Ajustando puja de %s a %s (mínimo legal detectado).",
                amount, bid_money,
            )

        last_exc: requests.exceptions.HTTPError | None = None
        try:
            return _place_bid(market_info["item_id"], bid_money)
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            status, err_code = _http_error_info(exc)

            # 030.01.01 / 030.01.31: cantidad por debajo del mínimo requerido.
            if status == 400 and err_code in {
                "030.01.01",
                "030.01.31",
                "030_01_01",
                "030_01_31",
            }:
                refreshed = _resolve_market_item(market_info["item_id"], player_id)
                refreshed_money = _compute_bid_money(bid_money, refreshed)
                if (
                    refreshed_money != bid_money
                    or refreshed.get("item_id") != market_info.get("item_id")
                ):
                    logger.info(
                        "Reintentando puja con mercado actualizado: item %s, %s.",
                        refreshed.get("item_id"), refreshed_money,
                    )
                    try:
                        return _place_bid(refreshed["item_id"], refreshed_money)
                    except requests.exceptions.HTTPError as exc_retry:
                        last_exc = exc_retry
                market_info = refreshed
                bid_money = refreshed_money

            # Si ya existe puja previa, editarla.
            bid_id = str(market_info.get("bid_id") or "")
            if bid_id:
                edit_money = _compute_bid_money(bid_money, market_info)
                edit_url = (
                    f"{BASE_URL}/api/v3/league/{self.league_id}/market/"
                    f"{market_info['item_id']}/bid/{bid_id}"
                )
                logger.info(
                    "Fallback puja: editando bid existente %s para item %s...",
                    bid_id, market_info["item_id"],
                )
                try:
                    headers = {
                        **self._auth_headers(),
                        "Content-Type": "application/json",
                    }
                    resp = self.session.put(
                        edit_url,
                        json={"money": int(edit_money)},
                        headers=headers,
                    )
                    if not resp.ok:
                        try:
                            err_body = resp.json() if resp.content else resp.text[:500]
                        except Exception:
                            err_body = resp.text[:500] if resp.text else ""
                        logger.warning(
                            "PUT %s → %s | body: %s | response: %s",
                            edit_url,
                            resp.status_code,
                            {"money": int(edit_money)},
                            err_body,
                        )
                    resp.raise_for_status()
                    return resp.json() if resp.content else {}
                except requests.exceptions.HTTPError as exc_edit:
                    last_exc = exc_edit

            # Si el item pasó a ser de manager, usar oferta en lugar de puja.
            if market_info.get("discr") and market_info.get("discr") != "marketPlayerLeague":
                offer_url = (
                    f"{BASE_URL}/api/v3/league/{self.league_id}/market/"
                    f"{market_info['item_id']}/offer"
                )
                offer_money = _compute_bid_money(bid_money, market_info)
                logger.info(
                    "Fallback puja: item %s no es de liga (discr=%s). Probando /offer...",
                    market_info["item_id"], market_info["discr"],
                )
                try:
                    return self._post(offer_url, {"money": int(offer_money)})
                except requests.exceptions.HTTPError as exc_offer:
                    last_exc = exc_offer

            # Fallback legado solo si el endpoint principal no existe en backend.
            status, _ = _http_error_info(last_exc) if last_exc else (None, "")
            if status in (404, 405):
                alt_url = f"{BASE_URL}/api/v3/league/{self.league_id}/market/bid"
                alt_body = {
                    "marketItemId": str(market_info["item_id"]),
                    "money": int(bid_money),
                }
                logger.info(
                    "Fallback legado puja: POST /market/bid para item %s...",
                    market_info["item_id"],
                )
                return self._post(alt_url, alt_body)

            if last_exc:
                raise last_exc
            raise

    def update_team_lineup(
        self,
        team_id: str,
        tactical_formation: list[int],
        goalkeeper_id: str,
        defenders_ids: list[str],
        midfielders_ids: list[str],
        strikers_ids: list[str],
        captain_team_id: str | None = None,
        coach_id: str | None = None,
    ) -> dict:
        """
        Actualiza la alineación del equipo.

        PUT /api/v3/teams/{teamId}/lineup

        Body (según APK UpdateLineupRequest):
          - tactical_formation: [DEF, MED, DEL]
          - goalkeeper: "playerTeamId"
          - defender: ["playerTeamId", ...]
          - midfield: ["playerTeamId", ...]
          - striker: ["playerTeamId", ...]
          - captain: "playerTeamId" (opcional)
          - coach: "playerTeamId" (opcional)
        """
        tid = str(team_id)
        body = {
            "tactical_formation": [int(x) for x in tactical_formation],
            "goalkeeper": str(goalkeeper_id),
            "defender": [str(x) for x in defenders_ids],
            "midfield": [str(x) for x in midfielders_ids],
            "striker": [str(x) for x in strikers_ids],
        }
        if captain_team_id:
            body["captain"] = str(captain_team_id)
        if coach_id:
            body["coach"] = str(coach_id)

        url = f"{BASE_URL}/api/v3/teams/{tid}/lineup"
        logger.info("Actualizando alineación del team %s...", tid)
        return self._put(url, body)

    def get_team_lineup(
        self,
        team_id: str,
        week_number: int | None = None,
    ) -> dict:
        """
        Obtiene la alineación de un equipo.

        GET /api/v3/teams/{teamId}/lineup
        GET /api/v4/teams/{teamId}/lineup/week/{weekNumber} (si week_number)
        """
        tid = str(team_id)
        if week_number is not None:
            url = f"{BASE_URL}/api/v4/teams/{tid}/lineup/week/{int(week_number)}"
        else:
            url = f"{BASE_URL}/api/v3/teams/{tid}/lineup"
        return self._get(url)

    def buy_player_clausulazo(
        self,
        player_team_id: str,
        buyout_clause_to_pay: int | None = None,
    ) -> dict:
        """
        Ejecutar clausulazo sobre un jugador de un rival.

        POST /api/v4/league/{leagueId}/buyout/{playerTeamId}/pay

        player_team_id: ID del jugador en la plantilla del rival (PlayerTeam).
        """
        url = f"{BASE_URL}/api/v4/league/{self.league_id}/buyout/{player_team_id}/pay"

        # Algunas respuestas del backend requieren el campo
        # "buyoutClauseToPay" en el body del pay.
        clause = _to_int(buyout_clause_to_pay)
        if not clause:
            try:
                info = self._get(
                    f"{BASE_URL}/api/v4/league/{self.league_id}/buyout/{player_team_id}"
                )
                if isinstance(info, dict):
                    for key in (
                        "buyoutClauseToPay",
                        "buyoutClause",
                        "clauseToPay",
                        "amount",
                        "price",
                    ):
                        clause = _to_int(info.get(key))
                        if clause:
                            break
            except Exception:
                clause = None

        body = {"buyoutClauseToPay": int(clause)} if clause else {}
        logger.info("Clausulazo a jugador %s...", player_team_id)
        try:
            return self._post(url, body)
        except requests.exceptions.HTTPError:
            # Fallback v5 detectado en el APK.
            alt_url = f"{BASE_URL}/api/v5/league/{self.league_id}/buyout/player"
            alt_body = {"playerTeamId": str(player_team_id)}
            if clause:
                alt_body["buyoutClauseToPay"] = int(clause)
            logger.info("Fallback clausulazo: POST /api/v5/league/.../buyout/player")
            return self._post(alt_url, alt_body)

    # ===================================================================
    # ENDPOINTS DE LA API (GET)
    # ===================================================================

    # -------------------------------------------------------------------
    # Ligas
    # -------------------------------------------------------------------
    def get_leagues(self) -> list[dict]:
        """
        GET /api/v3/leagues
        Ligas del usuario autenticado. Útil para descubrir tu league_id.
        """
        url = f"{BASE_URL}/api/v3/leagues"
        logger.info("Obteniendo ligas...")
        return self._get(url)

    # -------------------------------------------------------------------
    # Jugadores
    # -------------------------------------------------------------------
    def get_players_raw(self) -> list[dict]:
        """
        GET /api/v3/players/league/{league_id}
        Todos los jugadores de la liga (datos crudos).

        Response por elemento:
            {
                "id": "1234",
                "nickname": "Vinicius",
                "positionId": "4",
                "playerStatus": "ok",
                "points": 120,
                "marketValue": "15000000",
                "lastSeasonPoints": "95",
                "averagePoints": "8",
                "images": {...},
                "team": {"id": "...", "shortName": "RMA", ...},
                ...
            }
        """
        url = f"{BASE_URL}/api/v3/players/league/{self.league_id}"
        logger.info("Obteniendo jugadores de la liga %s...", self.league_id)
        return self._get(url)

    def get_players(self) -> list[dict]:
        """
        Jugadores formateados:
            {"player_id", "name", "position", "status", "points",
             "market_value", "points_last_season", "avg_points"}
        """
        return [_format_player(p) for p in self.get_players_raw()]

    # -------------------------------------------------------------------
    # Jugador individual
    # -------------------------------------------------------------------
    def get_player_detail(self, player_id: int) -> dict:
        """
        GET /api/v3/player/{player_id}
        Info detallada de un jugador.
        """
        url = f"{BASE_URL}/api/v3/player/{player_id}"
        return self._get(url)

    # -------------------------------------------------------------------
    # Mercado diario
    # -------------------------------------------------------------------
    def get_daily_market_raw(self) -> list[dict]:
        """
        GET /api/v3/league/{league_id}/market
        Jugadores actualmente en el mercado.

        Response por elemento:
            {
                "playerMaster": {"id": "1234", "nickname": "Vinicius", ...},
                "sellerTeam": {"manager": {"managerName": "...", ...}, ...} | null,
                "discr": "marketPlayerLeague" | "marketPlayerTeam",
                "salePrice": "15000000",
                "numberOfOffers": "3",
                "numberOfBids": "1",
                "expirationDate": "2026-02-15T12:00:00+01:00",
                ...
            }
        """
        url = f"{BASE_URL}/api/v3/league/{self.league_id}/market"
        logger.info("Obteniendo mercado diario...")
        return self._get(url)

    def get_daily_market(self) -> list[dict]:
        return [_format_market_item(i) for i in self.get_daily_market_raw()]

    # -------------------------------------------------------------------
    # Ranking / Managers
    # -------------------------------------------------------------------
    def get_ranking_raw(self) -> list[dict]:
        """
        GET /api/v3/leagues/{league_id}/ranking/
        Ranking de la liga.

        Response por elemento:
            {
                "team": {"id": "abc123", "manager": {"managerName": "...", ...}},
                "points": 350,
                "rank": 1,
                ...
            }
        """
        url = f"{BASE_URL}/api/v3/leagues/{self.league_id}/ranking/"
        logger.info("Obteniendo ranking...")
        return self._get(url)

    def get_manager_ids(self) -> list[str]:
        ranking = self.get_ranking_raw()
        return [e["team"]["id"] for e in ranking if "team" in e]

    # -------------------------------------------------------------------
    # Plantilla de un manager
    # -------------------------------------------------------------------
    def get_team_raw(self, manager_id: str) -> dict:
        """
        GET /api/v3/leagues/{league_id}/teams/{manager_id}
        Plantilla completa de un mánager.

        Response:
            {
                "players": [
                    {
                        "playerMaster": {"id": "1234", ...},
                        "manager": {"id": "...", "managerName": "...", ...},
                        "buyoutClause": "20000000",
                        "buyoutClauseLockedEndTime": "2026-02-20T...",
                        ...
                    },
                    ...
                ]
            }
        """
        url = f"{BASE_URL}/api/v3/leagues/{self.league_id}/teams/{manager_id}"
        logger.info("Obteniendo plantilla del mánager %s...", manager_id)
        return self._get(url)

    def get_team_raw_v4(self, team_id: str) -> dict | None:
        """
        GET /api/v4/leagues/{league_id}/teams/{team_id}
        Versión v4 que puede incluir playerTeamId en los jugadores.
        Devuelve None si el endpoint falla (404, etc).
        """
        try:
            url = f"{BASE_URL}/api/v4/leagues/{self.league_id}/teams/{team_id}"
            return self._get(url)
        except Exception:
            return None

    def get_league_me_raw(self) -> dict | None:
        """
        GET /api/v3/leagues/{leagueId}/me
        Mi equipo en liga. Puede incluir playerTeamId en jugadores.
        Devuelve None si el endpoint falla.
        """
        try:
            url = f"{BASE_URL}/api/v3/leagues/{self.league_id}/me"
            return self._get(url)
        except Exception:
            return None

    def get_team(self, manager_id: str) -> list[dict]:
        raw = self.get_team_raw(manager_id)
        return [_format_owned_player(manager_id, p) for p in raw.get("players", [])]

    def get_my_team(self) -> list[dict]:
        if not self.manager_id:
            raise ValueError("manager_id no configurado.")
        return self.get_team(self.manager_id)

    # -------------------------------------------------------------------
    # Historial de precios
    # -------------------------------------------------------------------
    def get_price_history(self, player_id: int) -> list[dict]:
        """
        GET /api/v3/player/{player_id}/market-value
        Historial de valor de mercado.

        Response:
            [
                {"marketValue": "12000000", "date": "2026-01-15T00:00:00+01:00"},
                ...
            ]
        """
        url = f"{BASE_URL}/api/v3/player/{player_id}/market-value"
        return self._get(url)

    def get_all_price_histories(self, player_ids: list[int], max_workers: int = 5) -> list[dict]:
        """Descarga historial de precios en paralelo para múltiples jugadores."""
        all_h = []
        logger.info("Descargando historial de %d jugadores...", len(player_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self.get_price_history, pid): pid for pid in player_ids}
            for f in as_completed(futs):
                pid = futs[f]
                try:
                    for e in f.result():
                        all_h.append({
                            "player_id": pid,
                            "market_value": _to_int(e.get("marketValue")),
                            "date": e.get("date"),
                        })
                except Exception as exc:
                    logger.error("Error jugador %s: %s", pid, exc)
        logger.info("Historial: %d registros.", len(all_h))
        return all_h

    # -------------------------------------------------------------------
    # Info del usuario autenticado
    # -------------------------------------------------------------------
    def get_user_info(self) -> dict:
        """
        GET /api/v3/user/me
        Info del usuario autenticado (manager_id, nombre, etc).
        """
        url = f"{BASE_URL}/api/v3/user/me"
        return self._get(url)

    def find_my_team_id(self) -> str:
        """
        Busca el team_id del usuario en el ranking de la liga.
        Compara el manager_id del token con los del ranking.
        """
        user = self.get_user_info()
        my_manager_id = str(user.get("id", ""))
        ranking = self.get_ranking_raw()
        for entry in ranking:
            team = entry.get("team", {})
            manager = team.get("manager", {})
            if str(manager.get("id", "")) == my_manager_id:
                return str(team.get("id", ""))
        raise RuntimeError(
            f"No se encontró tu equipo (manager_id={my_manager_id}) "
            f"en la liga {self.league_id}."
        )

    # -------------------------------------------------------------------
    # Actividad del mercado
    # -------------------------------------------------------------------
    def get_activity_page_raw(self, page: int) -> list[dict]:
        """
        GET /api/v3/leagues/{league_id}/news/{page}
        Noticias/actividad (paginado).

        Response:
            [
                {
                    "id": "99999",
                    "title": "Operación de mercado",
                    "msg": "Manager ha comprado a Jugador a LaLiga por 15.000.000 €",
                    "publicationDate": "2026-02-10T10:30:00+01:00",
                    ...
                },
                ...
            ]
        """
        url = f"{BASE_URL}/api/v3/leagues/{self.league_id}/news/{page}"
        return self._get(url)

    def get_activity_page(self, page: int) -> list[dict]:
        raw = self.get_activity_page_raw(page)
        ops = [i for i in raw if i.get("title") == "Operación de mercado"]
        results = []
        for item in ops:
            parsed = _parse_activity_message(item.get("msg", ""))
            if parsed:
                parsed["transaction_id"] = _to_int(item.get("id"))
                parsed["publication_date"] = item.get("publicationDate")
                results.append(parsed)
        return results

    def get_full_activity(self, max_pages: int = 1000) -> list[dict]:
        all_act = []
        page = 1
        while page <= max_pages:
            logger.info("Actividad - página %d...", page)
            act = self.get_activity_page(page)
            if not act:
                break
            all_act.extend(act)
            page += 1
        logger.info("Total operaciones: %d", len(all_act))
        return all_act


# ===========================================================================
# Funciones auxiliares
# ===========================================================================
def _to_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _format_player(p: dict) -> dict:
    return {
        "player_id": _to_int(p.get("id")),
        "name": p.get("nickname"),
        "position": POSITION_MAP.get(str(p.get("positionId", "")), "???"),
        "status": p.get("playerStatus"),
        "points": p.get("points"),
        "market_value": _to_int(p.get("marketValue")),
        "points_last_season": _to_int(p.get("lastSeasonPoints")),
        "avg_points": _to_int(p.get("averagePoints")),
    }


def _format_market_item(item: dict) -> dict:
    pm = item.get("playerMaster", {})
    st = item.get("sellerTeam")
    d = item.get("discr", "")
    if d == "marketPlayerLeague":
        owner = "LaLiga"
    elif st and st.get("manager"):
        owner = st["manager"].get("managerName", "Desconocido")
    else:
        owner = "Desconocido"
    return {
        "player_id": _to_int(pm.get("id")),
        "player_name": pm.get("nickname") or pm.get("name", "?"),
        "position": POSITION_MAP.get(str(pm.get("positionId", "")), "?"),
        "market_value": _to_int(pm.get("marketValue")),
        "owner": owner,
        "direct_offer": item.get("directOffer"),
        "sale_price": _to_int(item.get("salePrice")),
        "offers": _to_int(item.get("numberOfOffers")) or 0,
        "bids": _to_int(item.get("numberOfBids")) or 0,
        "expiration": item.get("expirationDate"),
    }


def _format_owned_player(manager_id: str, p: dict) -> dict:
    pm = p.get("playerMaster", {})
    mi = p.get("manager", {})
    return {
        "player_id": _to_int(pm.get("id")),
        "player_name": pm.get("nickname") or pm.get("name", "?"),
        "position": POSITION_MAP.get(str(pm.get("positionId", "")), "?"),
        "status": pm.get("playerStatus", "?"),
        "manager_name": mi.get("managerName"),
        "league_manager_id": mi.get("id"),
        "manager_id": manager_id,
        "buyout_clause": _to_int(p.get("buyoutClause")),
        "buyout_lock_expiration": p.get("buyoutClauseLockedEndTime"),
    }


def _parse_activity_message(msg: str) -> dict | None:
    pattern = (
        r"^([\w\s]+)\sha\s(comprado|vendido)\sa\s"
        r"([\w\s]+)\sa\s([\w\s]+)\spor\s([\d\.]+)\s€"
    )
    m = re.match(pattern, msg, re.UNICODE)
    if not m:
        return None
    amount_str = m.group(5).replace(".", "")
    return {
        "origin": m.group(1).strip(),
        "tx_type": "buy" if m.group(2) == "comprado" else "sell",
        "player_name": m.group(3).strip(),
        "destination": m.group(4).strip(),
        "amount": int(amount_str) if amount_str else None,
    }


# ===========================================================================
# Snapshot del estado de la liga
# ===========================================================================
SNAPSHOTS_DIR = pathlib.Path("snapshots")


def get_league_snapshot(client: LaLigaFantasyClient) -> dict:
    """
    Genera un snapshot completo del estado de una liga:

    1. mi_equipo  — saldo, plantilla, ofertas recibidas sobre mis jugadores.
    2. mercado    — jugadores listados hoy (libres y puestos en venta por managers).
    3. rivales    — plantilla de cada rival con cláusulas y bloqueos.

    Devuelve un dict listo para serializar a JSON.
    """
    logger.info("Generando snapshot de la liga %s...", client.league_id)

    # ── Datos base ────────────────────────────────────────────────────
    user_info = client.get_user_info()
    my_manager_id = str(user_info.get("id", ""))

    ranking = client.get_ranking_raw()
    leagues = client.get_leagues()
    league_name = ""
    for lg in leagues:
        if str(lg.get("id", "")) == str(client.league_id):
            league_name = lg.get("name", "")
            break

    # Encontrar mi team_id en el ranking
    my_team_id = ""
    my_ranking_entry = {}
    rival_entries = []
    for entry in ranking:
        team = entry.get("team", {})
        manager = team.get("manager", {})
        if str(manager.get("id", "")) == my_manager_id:
            my_team_id = str(team.get("id", ""))
            my_ranking_entry = entry
        else:
            rival_entries.append(entry)

    if not my_team_id:
        raise RuntimeError(
            f"No se encontró tu equipo (manager_id={my_manager_id}) "
            f"en la liga {client.league_id}."
        )

    # ── 1. Mi equipo ─────────────────────────────────────────────────
    logger.info("Obteniendo mi equipo...")
    my_team_raw = client.get_team_raw(my_team_id)
    my_team = my_ranking_entry.get("team", {})

    # Obtener player_team_id desde v4, league/me o mercado (v3 teams no lo incluye)
    v4_player_ids: dict[int, str] = {}  # player_id -> player_team_id
    for src in [client.get_team_raw_v4(my_team_id), client.get_league_me_raw()]:
        if not src:
            continue
        players = src.get("players", src.get("squad", []))
        for p in players:
            pm = p.get("playerMaster", {})
            pid = _to_int(pm.get("id"))
            ptid = (
                p.get("id")
                or p.get("playerTeamId")
                or (p.get("playerTeam") or {}).get("id")
                or (p.get("playerTeam") or {}).get("playerTeamId")
            )
            if pid and ptid:
                v4_player_ids[pid] = str(ptid)

    plantilla = []
    ofertas_recibidas = []
    for p in my_team_raw.get("players", []):
        pm = p.get("playerMaster", {})
        # player_team_id: múltiples posibles claves según estructura API (id, playerTeamId, playerTeam.id)
        ptid = (
            p.get("id")
            or p.get("playerTeamId")
            or (p.get("playerTeam") or {}).get("id")
            or (p.get("playerTeam") or {}).get("playerTeamId")
        )
        if not ptid and pm.get("id"):
            ptid = v4_player_ids.get(_to_int(pm.get("id")))
        player_data = {
            "player_id": _to_int(pm.get("id")),
            "player_team_id": str(ptid or ""),  # Para venta y clausulazo (API v4)
            "nombre": pm.get("nickname") or pm.get("name", "?"),
            "posicion": POSITION_MAP.get(str(pm.get("positionId", "")), "?"),
            "equipo_real": (pm.get("team") or {}).get("name", "?"),
            "estado": pm.get("playerStatus", "?"),
            "valor_mercado": _to_int(pm.get("marketValue")),
            "puntos": pm.get("points"),
            "media_puntos": pm.get("averagePoints"),
            "clausula": _to_int(p.get("buyoutClause")),
            "clausula_bloqueada_hasta": p.get("buyoutClauseLockedEndTime"),
        }

        # ¿Está en venta? (playerMarket presente)
        market_info = p.get("playerMarket")
        if market_info:
            player_data["en_venta"] = True
            player_data["precio_venta"] = _to_int(market_info.get("salePrice"))
            player_data["venta_expiracion"] = market_info.get("expirationDate")
            player_data["num_ofertas"] = _to_int(market_info.get("numberOfOffers")) or 0
            player_data["oferta_directa"] = market_info.get("directOffer", False)
            player_data["market_player_id"] = str(market_info.get("id", ""))
            # offer_id: oferta de la liga tras cierre (leagueOfferId, offers[0].id, etc.)
            offer_id = market_info.get("leagueOfferId") or market_info.get("leagueOffer", {}).get("id")
            if not offer_id and market_info.get("offers"):
                offer_id = market_info["offers"][0].get("id") if market_info["offers"] else None
            player_data["offer_id"] = str(offer_id or "")

            if player_data["num_ofertas"] and player_data["num_ofertas"] > 0:
                # Para fase 2: la liga hace oferta; necesitamos market_player_id y offer_id
                market_info_raw = market_info or {}
                offer_id_recibida = (
                    market_info_raw.get("leagueOfferId")
                    or (market_info_raw.get("leagueOffer") or {}).get("id")
                )
                if not offer_id_recibida and market_info_raw.get("offers"):
                    offer_id_recibida = market_info_raw["offers"][0].get("id")
                ofertas_recibidas.append({
                    "player_id": player_data["player_id"],
                    "player_team_id": player_data.get("player_team_id"),
                    "market_player_id": str(market_info_raw.get("id", "")),
                    "offer_id": str(offer_id_recibida or ""),
                    "nombre": player_data["nombre"],
                    "precio_venta": player_data["precio_venta"],
                    "num_ofertas": player_data["num_ofertas"],
                    "oferta_directa": player_data["oferta_directa"],
                    "expiracion": player_data["venta_expiracion"],
                })
        else:
            player_data["en_venta"] = False

        plantilla.append(player_data)

    mi_equipo = {
        "manager_id": my_manager_id,
        "manager_name": user_info.get("managerName", ""),
        "team_id": my_team_id,
        "saldo_disponible": _to_int(my_team_raw.get("teamMoney")),
        "valor_equipo": _to_int(my_team.get("teamValue")),
        "puntos": my_team.get("teamPoints"),
        "posicion": my_ranking_entry.get("position"),
        "num_jugadores": my_team_raw.get("playersNumber"),
        "plantilla": plantilla,
        "ofertas_recibidas": ofertas_recibidas,
    }

    # ── 2. Mercado ───────────────────────────────────────────────────
    logger.info("Obteniendo mercado...")
    market_raw = client.get_daily_market_raw()

    # Extraer player_team_id del mercado para jugadores nuestros listados
    market_ptid_by_player: dict[int, str] = {}
    for item in market_raw:
        seller = item.get("sellerTeam", {})
        if str(seller.get("id", "")) == my_team_id:
            pm = item.get("playerMaster", {})
            player_team_info = item.get("playerTeam", {})
            pid = _to_int(pm.get("id"))
            ptid = (
                (player_team_info or {}).get("id")
                or (player_team_info or {}).get("playerTeamId")
                or item.get("playerTeamId")
            )
            if pid and ptid:
                market_ptid_by_player[pid] = str(ptid)

    # Completar player_team_id en plantilla desde mercado si faltaba
    for pl in plantilla:
        if not pl.get("player_team_id") and pl.get("player_id"):
            ptid = market_ptid_by_player.get(pl["player_id"])
            if ptid:
                pl["player_team_id"] = ptid

    mercado = []
    for item in market_raw:
        pm = item.get("playerMaster", {})
        discr = item.get("discr", "")
        seller = item.get("sellerTeam")
        player_team_info = item.get("playerTeam", {})

        market_entry = {
            "market_item_id": str(item.get("id", "")),  # = market_player_id para pujas
            "player_id": _to_int(pm.get("id")),
            "nombre": pm.get("nickname") or pm.get("name", "?"),
            "posicion": POSITION_MAP.get(str(pm.get("positionId", "")), "?"),
            "equipo_real": (pm.get("team") or {}).get("name", "?"),
            "valor_mercado": _to_int(pm.get("marketValue")),
            "puntos": pm.get("points"),
            "media_puntos": pm.get("averagePoints"),
            "precio_venta": _to_int(item.get("salePrice")),
            "expiracion": item.get("expirationDate"),
            "estado": item.get("status"),
        }

        if discr == "marketPlayerLeague":
            market_entry["tipo"] = "libre"
            market_entry["vendedor"] = None
            market_entry["pujas"] = _to_int(item.get("numberOfBids")) or 0
        else:
            market_entry["tipo"] = "manager"
            market_entry["vendedor"] = (seller or {}).get("manager", {}).get("managerName", "?")
            market_entry["vendedor_team_id"] = str((seller or {}).get("id", ""))
            market_entry["ofertas"] = _to_int(item.get("numberOfOffers")) or 0
            market_entry["oferta_directa"] = item.get("directOffer", False)
            if player_team_info:
                market_entry["clausula_original"] = _to_int(player_team_info.get("buyoutClause"))

        mercado.append(market_entry)

    # ── 3. Rivales ───────────────────────────────────────────────────
    logger.info("Obteniendo plantillas de rivales...")
    rivales = []
    for entry in rival_entries:
        team = entry.get("team", {})
        manager = team.get("manager", {})
        rival_team_id = str(team.get("id", ""))

        rival_team_raw = client.get_team_raw(rival_team_id)
        rival_v4 = client.get_team_raw_v4(rival_team_id)
        rival_v4_ids: dict[int, str] = {}
        if rival_v4:
            for p in rival_v4.get("players", rival_v4.get("squad", [])):
                pm = p.get("playerMaster", {})
                pid = _to_int(pm.get("id"))
                ptid = (
                    p.get("id")
                    or p.get("playerTeamId")
                    or (p.get("playerTeam") or {}).get("id")
                    or (p.get("playerTeam") or {}).get("playerTeamId")
                )
                if pid and ptid:
                    rival_v4_ids[pid] = str(ptid)

        rival_plantilla = []
        for p in rival_team_raw.get("players", []):
            pm = p.get("playerMaster", {})
            ptid = (
                p.get("id")
                or p.get("playerTeamId")
                or (p.get("playerTeam") or {}).get("id")
                or (p.get("playerTeam") or {}).get("playerTeamId")
            )
            if not ptid and pm.get("id"):
                ptid = rival_v4_ids.get(_to_int(pm.get("id")))
            player_data = {
                "player_id": _to_int(pm.get("id")),
                "player_team_id": str(ptid or ""),  # Para clausulazo (API v4)
                "nombre": pm.get("nickname") or pm.get("name", "?"),
                "posicion": POSITION_MAP.get(str(pm.get("positionId", "")), "?"),
                "equipo_real": (pm.get("team") or {}).get("name", "?"),
                "estado": pm.get("playerStatus", "?"),
                "valor_mercado": _to_int(pm.get("marketValue")),
                "puntos": pm.get("points"),
                "media_puntos": pm.get("averagePoints"),
                "clausula": _to_int(p.get("buyoutClause")),
                "clausula_bloqueada_hasta": p.get("buyoutClauseLockedEndTime"),
            }

            market_info = p.get("playerMarket")
            if market_info:
                player_data["en_venta"] = True
                player_data["precio_venta"] = _to_int(market_info.get("salePrice"))
                player_data["venta_expiracion"] = market_info.get("expirationDate")
            else:
                player_data["en_venta"] = False

            rival_plantilla.append(player_data)

        rivales.append({
            "manager_id": str(manager.get("id", "")),
            "manager_name": manager.get("managerName", "?"),
            "team_id": rival_team_id,
            "puntos": team.get("teamPoints"),
            "valor_equipo": _to_int(team.get("teamValue")),
            "posicion": entry.get("position"),
            "num_jugadores": rival_team_raw.get("playersNumber"),
            "plantilla": rival_plantilla,
        })

    # ── Snapshot final ───────────────────────────────────────────────
    snapshot = {
        "league_id": str(client.league_id),
        "league_name": league_name.strip(),
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "mi_equipo": mi_equipo,
        "mercado": mercado,
        "rivales": rivales,
    }

    logger.info(
        "Snapshot generado: %d jugadores propios, %d en mercado, %d rivales.",
        len(plantilla), len(mercado), len(rivales),
    )
    return snapshot


def save_league_snapshot(snapshot: dict) -> pathlib.Path:
    """
    Guarda un snapshot de liga en snapshots/{league_id}/{timestamp}.json.
    Devuelve el path del archivo creado.
    """
    league_id = snapshot.get("league_id", "unknown")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    league_dir = SNAPSHOTS_DIR / league_id
    league_dir.mkdir(parents=True, exist_ok=True)

    filepath = league_dir / f"{ts}.json"
    filepath.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    logger.info("Snapshot guardado en %s", filepath)
    return filepath


def save_all_league_snapshots(client: LaLigaFantasyClient) -> list[pathlib.Path]:
    """
    Genera y guarda un snapshot para CADA liga del usuario.
    Devuelve la lista de paths creados.
    """
    leagues = client.get_leagues()
    paths = []
    for league in leagues:
        lid = str(league.get("id", ""))
        if not lid:
            continue
        logger.info("Procesando liga: %s (%s)...", league.get("name", "?"), lid)
        client.league_id = lid
        snapshot = get_league_snapshot(client)
        path = save_league_snapshot(snapshot)
        paths.append(path)
    return paths


# ===========================================================================
# CLI
# ===========================================================================
def _pick_league(client: LaLigaFantasyClient) -> str:
    """Obtiene las ligas del usuario y selecciona automáticamente.

    - Si solo hay 1 liga, la usa directamente.
    - Si hay varias, muestra la lista con sus IDs y usa la primera.
      El usuario puede re-ejecutar con --league ID para elegir otra.
    """
    print()
    print("  Obteniendo tus ligas...")
    try:
        leagues = client.get_leagues()
    except requests.exceptions.HTTPError as e:
        print(f"  Error obteniendo ligas: {e}")
        print("  Usa --league ID para especificar tu liga manualmente.")
        sys.exit(1)

    if not leagues:
        print("  No se encontraron ligas para esta cuenta.")
        print("  Usa --league ID para especificar tu liga manualmente.")
        sys.exit(1)

    if len(leagues) == 1:
        lid = str(leagues[0].get("id", ""))
        name = leagues[0].get("name", "Sin nombre")
        print(f"  Liga encontrada: {name} (ID: {lid})")
        print()
        return lid

    # Varias ligas: mostrar todas y usar la primera
    print()
    print(f"  Se encontraron {len(leagues)} ligas:")
    print()
    print(f"  {'#':<4} {'Nombre':<35} {'ID':<20}")
    print("  " + "-" * 60)
    for i, league in enumerate(leagues, 1):
        lid = league.get("id", "?")
        name = league.get("name", "Sin nombre")
        print(f"  {i:<4} {name:<35} {lid:<20}")
    print()

    # Usar la primera por defecto
    first = leagues[0]
    lid = str(first.get("id", ""))
    name = first.get("name", "Sin nombre")
    print(f"  Usando: {name} (ID: {lid})")
    print(f"  Para otra liga, ejecuta: python laliga_fantasy_client.py --league OTRO_ID")
    print()
    return lid


def _run_public_demo() -> None:
    """Demo con datos públicos — sin token."""
    api = LaLigaFantasyPublic()

    print()
    print("=" * 65)
    print("  DATOS PÚBLICOS (sin token)")
    print("=" * 65)

    players = api.get_players()
    print(f"\n  Total jugadores: {len(players)}\n")
    print(f"  {'Nombre':<25} {'Pos':<5} {'Valor':>12} {'Pts':>6} {'Media':>6}")
    print("  " + "-" * 58)
    for p in sorted(players, key=lambda x: x.get("market_value") or 0, reverse=True)[:25]:
        val = p["market_value"] or 0
        avg = p["avg_points"] or 0
        print(
            f"  {p['name']:<25} {p['position']:<5} {val:>12,} € "
            f"{p.get('points') or 0:>5} {avg:>6}"
        )
    print(f"\n  ... mostrando 25 más caros de {len(players)}.")

    # Mostrar historial de precios del jugador más caro
    top = sorted(players, key=lambda x: x.get("market_value") or 0, reverse=True)[0]
    print(f"\n  Historial de precios de {top['name']}:")
    history = api.get_price_history(top["player_id"])
    for h in history[-5:]:
        val = _to_int(h.get("marketValue")) or 0
        print(f"    {h.get('date', '?')[:10]}  →  {val:>12,} €")
    print(f"    ... ({len(history)} registros en total)")

    print()
    print("  NOTA: Para datos de tu liga privada (mercado, ranking,")
    print("  plantillas) necesitas el token. Ver opción 5.")
    print()


def _run_demo(client: LaLigaFantasyClient) -> None:
    """Ejecuta una demo rápida: obtiene jugadores y los muestra."""
    print()
    print("=" * 65)
    print("  Obteniendo jugadores...")
    print("=" * 65)

    try:
        players = client.get_players()
        print(f"\n  Total jugadores: {len(players)}\n")
        print(f"  {'Nombre':<25} {'Pos':<5} {'Valor':>12} {'Pts':>6} {'Estado':<10}")
        print("  " + "-" * 61)
        for p in sorted(players, key=lambda x: x.get("market_value") or 0, reverse=True)[:20]:
            val = p["market_value"] or 0
            print(
                f"  {p['name']:<25} {p['position']:<5} {val:>12,} € "
                f"{p.get('points') or 0:>5} {p['status']:<10}"
            )
        print(f"\n  ... mostrando 20 más caros de {len(players)}.\n")
    except requests.exceptions.HTTPError as e:
        print(f"\n  Error HTTP: {e}")
        if e.response is not None and e.response.status_code == 401:
            print("  → Token expirado o inválido. Captura uno nuevo.")
        elif e.response is not None and e.response.status_code == 404:
            print("  → league_id incorrecto o liga no encontrada.")
        else:
            print("  → Revisa tu token y league_id.")
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LaLiga Fantasy API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Login con Google (abre navegador)
  python laliga_fantasy_client.py --google

  # Usar token ya guardado
  python laliga_fantasy_client.py

  # Guardar un token (p.ej. la URL de jwt.ms)
  python laliga_fantasy_client.py --token "https://jwt.ms/#id_token=eyJ..."

  # Guardar un token JWT directamente
  python laliga_fantasy_client.py --token "eyJhbGciOi..."

  # Especificar league_id
  python laliga_fantasy_client.py --league 016615640

  # Datos públicos (sin token)
  python laliga_fantasy_client.py --public

  # Guardar snapshot del estado de la liga
  python laliga_fantasy_client.py --snapshot
  python laliga_fantasy_client.py --snapshot --league 016615640

  # Scraping de datos externos (lesionados, stats, mercado)
  python laliga_fantasy_client.py --scrape
  python laliga_fantasy_client.py --scrape --output datos.json
        """,
    )
    parser.add_argument("--google", action="store_true", help="Login con Google (abre navegador)")
    parser.add_argument("--token", type=str, help="Token JWT o URL de jwt.ms con el token")
    parser.add_argument("--league", type=str, help="League ID")
    parser.add_argument("--public", action="store_true", help="Solo datos públicos (sin token)")
    parser.add_argument("--snapshot", action="store_true", help="Guardar snapshot del estado de la liga en JSON")
    parser.add_argument("--scrape", action="store_true", help="Scraping de datos externos (lesionados, stats, mercado)")
    parser.add_argument("--output", "-o", type=str, help="Ruta del JSON de salida (para --scrape)")

    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  LaLiga Fantasy API Client")
    print("=" * 65)

    # --- Modo scraping (no requiere token) ---
    if args.scrape:
        from scrapers.scrape_all import run_all_scrapers, save_scrape_data
        data = run_all_scrapers()
        output_path = pathlib.Path(args.output) if args.output else None
        path = save_scrape_data(data, output_path)
        ff = data.get("futbolfantasy", {})
        ss = data.get("sofascore", {})
        mv = data.get("market_values", {})
        print()
        print("=" * 60)
        print("  RESUMEN DEL SCRAPING")
        print("=" * 60)
        if "error" not in ff:
            print(f"  FútbolFantasy:  {len(ff.get('lesionados',[]))} lesionados, "
                  f"{len(ff.get('sancionados',[]))} sancionados, "
                  f"{len(ff.get('apercibidos',[]))} apercibidos")
        if "error" not in ss:
            print(f"  Sofascore:      {len(ss.get('jugadores',[]))} jugadores, "
                  f"{len(ss.get('clasificacion',[]))} equipos")
        if "error" not in mv:
            mercado = mv.get("mercado", {})
            print(f"  Mercado:        {len(mercado.get('subidas',[]))} subidas, "
                  f"{len(mercado.get('bajadas',[]))} bajadas")
        print(f"\n  Guardado en: {path}")
        print()
        return

    # --- Modo público ---
    if args.public:
        _run_public_demo()
        return

    # --- Modo snapshot ---
    if args.snapshot:
        saved = load_token()
        if not saved:
            print("\n  No hay token guardado. Usa --google o --token primero.")
            sys.exit(1)
        league_id = args.league or os.getenv("LALIGA_LEAGUE_ID", "")
        client = LaLigaFantasyClient(token=saved, league_id=league_id)

        if league_id:
            # Snapshot de una liga específica
            snapshot = get_league_snapshot(client)
            path = save_league_snapshot(snapshot)
            print(f"\n  Snapshot guardado en: {path}")
        else:
            # Snapshot de todas las ligas
            paths = save_all_league_snapshots(client)
            print(f"\n  {len(paths)} snapshot(s) guardado(s):")
            for p in paths:
                print(f"    {p}")
        print()
        return

    # --- Guardar/usar un token pasado por argumento ---
    if args.token:
        token = args.token.strip()
        # Si pegaron la URL completa de jwt.ms, extraer el token
        if "jwt.ms" in token and "id_token=" in token:
            import urllib.parse
            parsed = urllib.parse.urlparse(token)
            fragment_params = urllib.parse.parse_qs(parsed.fragment)
            if "id_token" in fragment_params:
                token = fragment_params["id_token"][0]
            elif "access_token" in fragment_params:
                token = fragment_params["access_token"][0]
        # Si pegaron "Bearer eyJ..."
        if token.lower().startswith("bearer "):
            token = token[7:]
        # Si pegaron "authredirect://...?code=..."
        if token.startswith("authredirect://"):
            import urllib.parse
            parsed = urllib.parse.urlparse(token)
            params = urllib.parse.parse_qs(parsed.query)
            if "code" in params:
                token = params["code"][0]

        save_token(token)
        print(f"\n  Token guardado. Primeros chars: {token[:40]}...")
        league_id = args.league or os.getenv("LALIGA_LEAGUE_ID", "")
        client = LaLigaFantasyClient(token=token, league_id=league_id)
        if not client.league_id:
            client.league_id = _pick_league(client)
        _run_demo(client)
        return

    # --- Login con Google: solo abre el navegador ---
    if args.google:
        google_login_open_browser()
        return

    # --- Por defecto: usar token guardado ---
    saved_token = load_token()
    if saved_token:
        league_id = args.league or os.getenv("LALIGA_LEAGUE_ID", "")
        client = LaLigaFantasyClient(token=saved_token, league_id=league_id)
        if not client.league_id:
            client.league_id = _pick_league(client)
        _run_demo(client)
        return

    # --- No hay token: mostrar ayuda ---
    print()
    print("  No hay token guardado. Opciones:")
    print()
    print("  1) Login con Google:")
    print("     python laliga_fantasy_client.py --google")
    print()
    print("  2) Pegar token directamente:")
    print('     python laliga_fantasy_client.py --token "https://jwt.ms/#id_token=eyJ..."')
    print()
    print("  3) Datos públicos (sin token):")
    print("     python laliga_fantasy_client.py --public")
    print()


if __name__ == "__main__":
    main()
