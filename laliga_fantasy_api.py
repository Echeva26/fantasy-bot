"""
LaLiga Fantasy API Client
=========================
Cliente Python para la API de La Liga Fantasy (https://api-fantasy.llt-services.com).

Uso:
    from laliga_fantasy_api import LaLigaFantasyAPI

    api = LaLigaFantasyAPI(
        username="tu-email@ejemplo.com",
        password="tu-contraseña",
        league_id="tu-league-id",
        manager_id="tu-manager-id",
    )

    # Autenticación (se hace automáticamente en cada llamada si no hay token)
    api.authenticate()

    # Obtener ligas
    ligas = api.get_leagues()

    # Obtener jugadores
    jugadores = api.get_players()

    # Obtener mercado diario
    mercado = api.get_daily_market()

    # Obtener ranking de managers
    ranking = api.get_managers_ranking()

    # Obtener plantilla de un manager
    plantilla = api.get_manager_team("manager-id-aqui")

    # Obtener historial de precio de un jugador
    historial = api.get_player_price_history(player_id=1234)

    # Obtener actividad del mercado (todas las páginas)
    actividad = api.get_full_activity()
"""

import re
import os
import time
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BASE_URL = "https://api-fantasy.llt-services.com"
AUTH_POLICY = "B2C_1A_ResourceOwnerv2"

POSITION_MAP = {
    "1": "GKE",  # Portero
    "2": "DEF",  # Defensa
    "3": "MID",  # Centrocampista
    "4": "ATA",  # Delantero
    "5": "COA",  # Entrenador
}


class LaLigaFantasyAPI:
    """Cliente para la API de La Liga Fantasy."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        league_id: Optional[str] = None,
        manager_id: Optional[str] = None,
    ):
        """
        Inicializa el cliente.

        Los parámetros se pueden pasar directamente o se leen de variables
        de entorno: LALIGA_USERNAME, LALIGA_PASSWORD, LALIGA_LEAGUE_ID,
        LALIGA_MANAGER_ID.
        """
        self.username = username or os.getenv("LALIGA_USERNAME", "")
        self.password = password or os.getenv("LALIGA_PASSWORD", "")
        self.league_id = league_id or os.getenv("LALIGA_LEAGUE_ID", "")
        self.manager_id = manager_id or os.getenv("LALIGA_MANAGER_ID", "")

        self._access_token: Optional[str] = None
        self._token_ts: Optional[float] = None  # timestamp de cuando se obtuvo el token
        self._token_ttl: float = 23 * 3600  # Refrescar antes de las 24h

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        })

    # -----------------------------------------------------------------------
    # Autenticación
    # -----------------------------------------------------------------------
    def _get_auth_code(self) -> str:
        """
        POST /login/v3/email/auth
        Envía credenciales y obtiene un código de autenticación.

        Request (form-urlencoded):
            policy:   "B2C_1A_ResourceOwnerv2"
            username: <email>
            password: <contraseña>

        Response JSON:
            {
                "code": "eyJ0eXAiOi..."
            }
        """
        url = f"{BASE_URL}/login/v3/email/auth"
        payload = {
            "policy": AUTH_POLICY,
            "username": self.username,
            "password": self.password,
        }
        logger.info("Solicitando código de autenticación...")
        resp = self.session.post(url, data=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["code"]

    def _get_access_token(self, code: str) -> str:
        """
        POST /login/v3/email/token
        Intercambia el código por un token de acceso (válido ~24h).

        Request (form-urlencoded):
            code:   <código obtenido en /auth>
            policy: "B2C_1A_ResourceOwnerv2"

        Response JSON:
            {
                "access_token": "eyJ0eXAiOi...",
                "token_type": "Bearer",
                ...
            }
        """
        url = f"{BASE_URL}/login/v3/email/token"
        payload = {
            "code": code,
            "policy": AUTH_POLICY,
        }
        logger.info("Intercambiando código por token de acceso...")
        resp = self.session.post(url, data=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"]

    def authenticate(self) -> str:
        """
        Flujo completo de autenticación en 2 pasos:
            1. POST /login/v3/email/auth  → code
            2. POST /login/v3/email/token → access_token

        El token se cachea y se renueva automáticamente antes de expirar.
        """
        if self._access_token and self._token_ts:
            elapsed = time.time() - self._token_ts
            if elapsed < self._token_ttl:
                return self._access_token

        code = self._get_auth_code()
        self._access_token = self._get_access_token(code)
        self._token_ts = time.time()
        logger.info("Autenticación exitosa. Token válido ~24 horas.")
        return self._access_token

    def _auth_headers(self) -> dict:
        """Devuelve headers con el token Bearer para peticiones autenticadas."""
        token = self.authenticate()
        return {"Authorization": f"Bearer {token}"}

    def _get(self, url: str) -> dict | list:
        """GET autenticado que devuelve JSON."""
        resp = self.session.get(url, headers=self._auth_headers())
        resp.raise_for_status()
        return resp.json()

    def _post(self, url: str, json_body: dict) -> dict | list:
        """POST autenticado para operaciones de mercado (vender, comprar)."""
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
        """PUT autenticado para updates de recursos (p. ej. subir cláusula)."""
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

    # -----------------------------------------------------------------------
    # Operaciones de mercado (POST - endpoints extraídos del APK)
    # -----------------------------------------------------------------------
    def sell_player_phase1(self, player_team_id: str, price: int) -> dict:
        """
        Fase 1 de venta: publicar jugador en el mercado.

        POST /api/v3/league/{leagueId}/market/sell
        Body: {"salePrice": int, "playerTeamId": str}

        player_team_id: ID del jugador en tu plantilla (PlayerTeam), no player_id.
        """
        url = f"{BASE_URL}/api/v3/league/{self.league_id}/market/sell"
        # APK SellRequest: @SerializedName("playerId") — la API espera "playerId" no "playerTeamId"
        body = {"playerId": str(player_team_id), "salePrice": int(price)}
        logger.info("Publicando jugador %s en mercado (precio %s)...", player_team_id, price)
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

    def buy_player_bid(self, market_player_id: str, amount: int) -> dict:
        """
        Pujar por un jugador del mercado de la liga (tipo libre).

        POST /api/v3/league/{leagueId}/market/{marketPlayerId}/bid
        Body: {"amount": int}
        """
        url = f"{BASE_URL}/api/v3/league/{self.league_id}/market/{market_player_id}/bid"
        body = {"amount": amount}
        logger.info("Pujando %s por item %s...", amount, market_player_id)
        return self._post(url, body)

    def buy_player_clausulazo(self, player_team_id: str) -> dict:
        """
        Ejecutar clausulazo sobre un jugador de un rival.

        POST /api/v4/league/{leagueId}/buyout/{playerTeamId}/pay

        player_team_id: ID del jugador en la plantilla del rival (PlayerTeam),
                        no player_id ni target_team_id por separado.
        """
        url = f"{BASE_URL}/api/v4/league/{self.league_id}/buyout/{player_team_id}/pay"
        logger.info("Clausulazo a jugador %s...", player_team_id)
        return self._post(url, {})

    def increase_player_clause(
        self,
        player_team_id: str,
        value_to_increase: int,
        factor: float = 2.0,
    ) -> dict:
        """
        Aumentar cláusula de un jugador propio.

        PUT /api/v5/league/{leagueId}/buyout/player
        Body: {"playerId": "<playerTeamId>", "valueToIncrease": int, "factor": float}

        Regla habitual: invertir 1M => +2M en cláusula (factor 2.0).
        """
        ptid = str(player_team_id or "").strip()
        if not ptid:
            raise ValueError("player_team_id vacío")
        if int(value_to_increase) <= 0:
            raise ValueError("value_to_increase debe ser > 0")
        try:
            factor_value = float(factor)
        except Exception:
            factor_value = 2.0
        if factor_value <= 0:
            factor_value = 2.0

        url = f"{BASE_URL}/api/v5/league/{self.league_id}/buyout/player"
        body = {
            "playerId": ptid,
            "valueToIncrease": int(value_to_increase),
            "factor": factor_value,
        }
        logger.info(
            "Subiendo cláusula de jugador %s con inversión %s (factor %.2f)...",
            ptid,
            value_to_increase,
            factor_value,
        )
        return self._put(url, body)

    # -----------------------------------------------------------------------
    # Ligas
    # -----------------------------------------------------------------------
    def get_leagues(self) -> list[dict]:
        """
        GET /api/v3/leagues
        Devuelve las ligas del usuario autenticado.
        Útil para obtener el league-id.

        Response JSON (ejemplo):
            [
                {
                    "id": "abc123",
                    "name": "Mi Liga",
                    "type": "classic",
                    ...
                },
                ...
            ]
        """
        url = f"{BASE_URL}/api/v3/leagues"
        logger.info("Obteniendo ligas del usuario...")
        return self._get(url)

    # -----------------------------------------------------------------------
    # Jugadores
    # -----------------------------------------------------------------------
    def get_players_raw(self) -> list[dict]:
        """
        GET /api/v3/players/league/{league_id}
        Devuelve la lista completa de jugadores de la liga (datos crudos).

        Response JSON (ejemplo de un elemento):
            {
                "id": "1234",
                "nickname": "Vinicius",
                "positionId": "4",
                "playerStatus": "ok",
                "points": 120,
                "marketValue": "15000000",
                "lastSeasonPoints": "95",
                "averagePoints": "8",
                "images": { ... },
                "team": { ... },
                ...
            }
        """
        url = f"{BASE_URL}/api/v3/players/league/{self.league_id}"
        logger.info("Obteniendo jugadores de la liga %s...", self.league_id)
        return self._get(url)

    def get_players(self) -> list[dict]:
        """
        Versión formateada de get_players_raw().
        Devuelve una lista de dicts con campos normalizados:
            {
                "player_id": int,
                "name": str,
                "position": str ("GKE"|"DEF"|"MID"|"ATA"|"COA"),
                "status": str,
                "points": int,
                "market_value": int,
                "points_last_season": int | None,
                "avg_points": int | None,
            }
        """
        raw = self.get_players_raw()
        return [self._format_player(p) for p in raw]

    @staticmethod
    def _format_player(player: dict) -> dict:
        return {
            "player_id": _to_int(player.get("id")),
            "name": player.get("nickname"),
            "position": POSITION_MAP.get(str(player.get("positionId", "")), "???"),
            "status": player.get("playerStatus"),
            "points": player.get("points"),
            "market_value": _to_int(player.get("marketValue")),
            "points_last_season": _to_int(player.get("lastSeasonPoints")),
            "avg_points": _to_int(player.get("averagePoints")),
        }

    # -----------------------------------------------------------------------
    # Mercado diario
    # -----------------------------------------------------------------------
    def get_daily_market_raw(self) -> list[dict]:
        """
        GET /api/v3/league/{league_id}/market
        Devuelve los jugadores actualmente en el mercado de la liga.

        Response JSON (ejemplo de un elemento):
            {
                "playerMaster": {
                    "id": "1234",
                    "nickname": "Vinicius",
                    ...
                },
                "sellerTeam": {
                    "manager": {
                        "managerName": "JuanManager",
                        ...
                    },
                    ...
                } | null,
                "discr": "marketPlayerLeague" | "marketPlayerTeam",
                "directOffer": true | false | null,
                "salePrice": "15000000",
                "numberOfOffers": "3",
                "numberOfBids": "1",
                "expirationDate": "2026-02-15T12:00:00+01:00",
                ...
            }
        """
        url = f"{BASE_URL}/api/v3/league/{self.league_id}/market"
        logger.info("Obteniendo mercado diario de la liga %s...", self.league_id)
        return self._get(url)

    def get_daily_market(self) -> list[dict]:
        """
        Versión formateada de get_daily_market_raw().
        Devuelve:
            {
                "player_id": int,
                "owner": str ("LaLiga" o nombre del mánager vendedor),
                "direct_offer": bool | None,
                "sale_price": int,
                "offers": int,
                "bids": int,
                "expiration": str (ISO datetime),
            }
        """
        raw = self.get_daily_market_raw()
        return [self._format_market_item(item) for item in raw]

    @staticmethod
    def _format_market_item(item: dict) -> dict:
        player_master = item.get("playerMaster", {})
        seller_team = item.get("sellerTeam")
        discr = item.get("discr", "")

        if discr == "marketPlayerLeague":
            owner = "LaLiga"
        elif seller_team and seller_team.get("manager"):
            owner = seller_team["manager"].get("managerName", "Desconocido")
        else:
            owner = "Desconocido"

        return {
            "player_id": _to_int(player_master.get("id")),
            "market_player_id": str(item.get("id", "")),  # Para pujar y aceptar oferta
            "owner": owner,
            "direct_offer": item.get("directOffer"),
            "sale_price": _to_int(item.get("salePrice")),
            "offers": _to_int(item.get("numberOfOffers")) or 0,
            "bids": _to_int(item.get("numberOfBids")) or 0,
            "expiration": item.get("expirationDate"),
        }

    # -----------------------------------------------------------------------
    # Ranking / Managers
    # -----------------------------------------------------------------------
    def get_managers_ranking_raw(self) -> list[dict]:
        """
        GET /api/v3/leagues/{league_id}/ranking/
        Devuelve el ranking de la liga con info de cada mánager.

        Response JSON (ejemplo de un elemento):
            {
                "team": {
                    "id": "abc123",
                    "manager": {
                        "managerName": "JuanManager",
                        ...
                    },
                    ...
                },
                "points": 350,
                "rank": 1,
                ...
            }
        """
        url = f"{BASE_URL}/api/v3/leagues/{self.league_id}/ranking/"
        logger.info("Obteniendo ranking de la liga %s...", self.league_id)
        return self._get(url)

    def get_manager_ids(self) -> list[str]:
        """Extrae los IDs de todos los managers de la liga."""
        ranking = self.get_managers_ranking_raw()
        return [entry["team"]["id"] for entry in ranking if "team" in entry]

    # -----------------------------------------------------------------------
    # Plantilla de un manager (owned players)
    # -----------------------------------------------------------------------
    def get_manager_team_raw(self, manager_id: str) -> dict:
        """
        GET /api/v3/leagues/{league_id}/teams/{manager_id}
        Devuelve la plantilla completa de un mánager.

        Response JSON (ejemplo):
            {
                "players": [
                    {
                        "playerMaster": {
                            "id": "1234",
                            "nickname": "Vinicius",
                            ...
                        },
                        "manager": {
                            "id": "mgr-league-id",
                            "managerName": "JuanManager",
                            ...
                        },
                        "buyoutClause": "20000000",
                        "buyoutClauseLockedEndTime": "2026-02-20T12:00:00+01:00",
                        ...
                    },
                    ...
                ],
                ...
            }
        """
        url = f"{BASE_URL}/api/v3/leagues/{self.league_id}/teams/{manager_id}"
        logger.info("Obteniendo plantilla del mánager %s...", manager_id)
        return self._get(url)

    def get_manager_team(self, manager_id: str) -> list[dict]:
        """
        Versión formateada de get_manager_team_raw().
        Devuelve lista de jugadores con:
            {
                "player_id": int,
                "manager_name": str,
                "league_manager_id": str,
                "manager_id": str,
                "buyout": int | None,
                "buyout_lock_expiration": str (ISO datetime),
            }
        """
        raw = self.get_manager_team_raw(manager_id)
        players = raw.get("players", [])
        return [self._format_owned_player(manager_id, p) for p in players]

    @staticmethod
    def _format_owned_player(manager_id: str, player: dict) -> dict:
        player_master = player.get("playerMaster", {})
        manager_info = player.get("manager", {})
        return {
            "player_id": _to_int(player_master.get("id")),
            "player_team_id": str(player.get("id", "")),  # Para clausulazo / venta
            "manager_name": manager_info.get("managerName"),
            "league_manager_id": manager_info.get("id"),
            "manager_id": manager_id,
            "buyout": _to_int(player.get("buyoutClause")),
            "buyout_lock_expiration": player.get("buyoutClauseLockedEndTime"),
        }

    def get_all_teams(self) -> list[dict]:
        """
        Obtiene las plantillas de TODOS los managers de la liga
        (incluido el propio usuario).
        """
        manager_ids = set(self.get_manager_ids())
        if self.manager_id:
            manager_ids.add(self.manager_id)

        all_players = []
        for mid in manager_ids:
            all_players.extend(self.get_manager_team(mid))
        return all_players

    # -----------------------------------------------------------------------
    # Historial de precios de un jugador
    # -----------------------------------------------------------------------
    def get_player_price_history_raw(self, player_id: int) -> list[dict]:
        """
        GET /api/v3/player/{player_id}/market-value
        Devuelve el historial de valor de mercado de un jugador.

        Response JSON (ejemplo):
            [
                {
                    "marketValue": "12000000",
                    "date": "2026-01-15T00:00:00+01:00"
                },
                {
                    "marketValue": "12500000",
                    "date": "2026-01-16T00:00:00+01:00"
                },
                ...
            ]
        """
        url = f"{BASE_URL}/api/v3/player/{player_id}/market-value"
        logger.info("Obteniendo historial de precios del jugador %s...", player_id)
        return self._get(url)

    def get_player_price_history(self, player_id: int) -> list[dict]:
        """
        Versión formateada.
        Devuelve:
            [
                {"player_id": int, "market_value": int, "date": str},
                ...
            ]
        """
        raw = self.get_player_price_history_raw(player_id)
        return [
            {
                "player_id": player_id,
                "market_value": _to_int(entry.get("marketValue")),
                "date": entry.get("date"),
            }
            for entry in raw
        ]

    def get_all_price_histories(self, max_workers: int = 5) -> list[dict]:
        """
        Obtiene el historial de precios de TODOS los jugadores de la liga
        en paralelo (usando ThreadPoolExecutor).
        """
        players = self.get_players()
        player_ids = [p["player_id"] for p in players]
        all_histories = []

        logger.info(
            "Descargando historial de precios para %d jugadores (workers=%d)...",
            len(player_ids),
            max_workers,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_player_price_history, pid): pid
                for pid in player_ids
            }
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    history = future.result()
                    all_histories.extend(history)
                except Exception as e:
                    logger.error("Error obteniendo historial del jugador %s: %s", pid, e)

        return all_histories

    # -----------------------------------------------------------------------
    # Tendencias de precio (calculadas localmente)
    # -----------------------------------------------------------------------
    @staticmethod
    def compute_price_trends(price_histories: list[dict]) -> list[dict]:
        """
        Calcula tendencias de precio a 3, 7, 14, 30 días y total
        a partir de los datos de historial de precios.

        Devuelve:
            [
                {
                    "player_id": int,
                    "change_3d": float,
                    "change_7d": float,
                    "change_14d": float,
                    "change_30d": float,
                    "change_all": float,
                },
                ...
            ]
        """
        # Agrupar por player_id
        grouped: dict[int, list[dict]] = {}
        for entry in price_histories:
            pid = entry["player_id"]
            grouped.setdefault(pid, []).append(entry)

        trends = []
        for player_id, entries in grouped.items():
            sorted_entries = sorted(entries, key=lambda e: e["date"], reverse=True)
            values = [e["market_value"] for e in sorted_entries if e["market_value"] is not None]

            if len(values) < 2:
                continue

            current = values[0]

            def pct_change(interval: int | None = None) -> float:
                if interval is None or interval >= len(values):
                    before = values[-1]
                else:
                    before = values[interval]
                if before == 0:
                    return 0.0
                return round(((current - before) / before) * 100, 2)

            trends.append({
                "player_id": player_id,
                "change_3d": pct_change(3),
                "change_7d": pct_change(7),
                "change_14d": pct_change(14),
                "change_30d": pct_change(30),
                "change_all": pct_change(None),
            })

        return trends

    # -----------------------------------------------------------------------
    # Actividad del mercado
    # -----------------------------------------------------------------------
    def get_activity_page_raw(self, page: int) -> list[dict]:
        """
        GET /api/v3/leagues/{league_id}/news/{page}
        Devuelve una página de noticias/actividad de la liga.

        Response JSON (ejemplo):
            [
                {
                    "id": "99999",
                    "title": "Operación de mercado",
                    "msg": "JuanManager ha comprado a Vinicius a LaLiga por 15.000.000 €",
                    "publicationDate": "2026-02-10T10:30:00+01:00",
                    ...
                },
                {
                    "id": "99998",
                    "title": "Noticia",
                    "msg": "...",
                    ...
                },
                ...
            ]
        """
        url = f"{BASE_URL}/api/v3/leagues/{self.league_id}/news/{page}"
        return self._get(url)

    def get_activity_page(self, page: int) -> list[dict]:
        """
        Obtiene una página de actividad y filtra solo operaciones de mercado.
        Parsea el mensaje para extraer: origen, destino, jugador, tipo (buy/sell)
        y cantidad.

        Devuelve:
            [
                {
                    "transaction_id": int,
                    "publication_date": str,
                    "origin": str,
                    "destination": str,
                    "player_name": str,
                    "tx_type": "buy" | "sell",
                    "amount": int,
                },
                ...
            ]
        """
        raw = self.get_activity_page_raw(page)
        # Solo nos interesan las operaciones de mercado
        market_ops = [item for item in raw if item.get("title") == "Operación de mercado"]
        results = []
        for item in market_ops:
            parsed = self._parse_activity_message(item.get("msg", ""))
            if parsed:
                parsed["transaction_id"] = _to_int(item.get("id"))
                parsed["publication_date"] = item.get("publicationDate")
                results.append(parsed)
        return results

    @staticmethod
    def _parse_activity_message(msg: str) -> dict | None:
        """
        Parsea mensajes de actividad del mercado.

        Formato esperado:
            "JuanManager ha comprado a Vinicius a LaLiga por 15.000.000 €"
            "PedroManager ha vendido a Benzema a JuanManager por 20.000.000 €"

        Regex utilizada (del código Clojure original):
            ^([\\p{L}0-9_\\s]+)\\sha\\s(?:comprado|vendido)\\sa\\s
            ([\\p{L}\\s]+)\\sa\\s([\\p{L}0-9_\\s]+)\\spor\\s([\\d\\.]+)\\s€
        """
        pattern = (
            r"^([\w\s]+)\sha\s(comprado|vendido)\sa\s"
            r"([\w\s]+)\sa\s([\w\s]+)\spor\s([\d\.]+)\s€"
        )
        match = re.match(pattern, msg, re.UNICODE)
        if not match:
            return None

        action = match.group(2)
        amount_str = match.group(5).replace(".", "")

        return {
            "origin": match.group(1).strip(),
            "tx_type": "buy" if action == "comprado" else "sell",
            "player_name": match.group(3).strip(),
            "destination": match.group(4).strip(),
            "amount": int(amount_str) if amount_str else None,
        }

    def get_full_activity(self, max_pages: int = 1000) -> list[dict]:
        """
        Itera todas las páginas de actividad hasta encontrar una vacía.
        Devuelve la lista completa de operaciones de mercado parseadas.
        """
        all_activity = []
        page = 1

        while page <= max_pages:
            logger.info("Obteniendo actividad - página %d...", page)
            activity = self.get_activity_page(page)
            if not activity:
                break
            all_activity.extend(activity)
            page += 1

        logger.info("Total de operaciones de mercado: %d", len(all_activity))
        return all_activity



# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _to_int(value) -> int | None:
    """Convierte un valor a entero de forma segura."""
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


# ---------------------------------------------------------------------------
# Ejemplo de uso (ejecutar directamente: python laliga_fantasy_api.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Configurar credenciales (via env vars o directamente aquí)
    api = LaLigaFantasyAPI(
        username=os.getenv("LALIGA_USERNAME", ""),
        password=os.getenv("LALIGA_PASSWORD", ""),
        league_id=os.getenv("LALIGA_LEAGUE_ID", ""),
        manager_id=os.getenv("LALIGA_MANAGER_ID", ""),
    )

    # --- 1. Autenticación ---
    print("=" * 60)
    print("1. Autenticando...")
    api.authenticate()
    print("   Token obtenido correctamente.\n")

    # --- 2. Ligas ---
    print("=" * 60)
    print("2. Ligas del usuario:")
    leagues = api.get_leagues()
    for league in leagues:
        print(f"   - {league.get('name', 'N/A')} (ID: {league.get('id', 'N/A')})")
    print()

    # --- 3. Jugadores ---
    print("=" * 60)
    print("3. Jugadores de la liga (primeros 10):")
    players = api.get_players()
    for p in players[:10]:
        print(f"   [{p['position']}] {p['name']} - {p['market_value']:,} € - {p['points']} pts")
    print(f"   ... Total: {len(players)} jugadores\n")

    # --- 4. Mercado diario ---
    print("=" * 60)
    print("4. Mercado diario:")
    market = api.get_daily_market()
    for item in market[:10]:
        print(f"   Jugador {item['player_id']} | Precio: {item['sale_price']:,} € | "
              f"Ofertas: {item['offers']} | Dueño: {item['owner']}")
    print(f"   ... Total en mercado: {len(market)}\n")

    # --- 5. Ranking ---
    print("=" * 60)
    print("5. Ranking de la liga:")
    ranking = api.get_managers_ranking_raw()
    for entry in ranking:
        team = entry.get("team", {})
        mgr = team.get("manager", {})
        print(f"   #{entry.get('rank', '?')} - {mgr.get('managerName', 'N/A')} "
              f"({entry.get('points', 0)} pts)")
    print()

    # --- 6. Actividad ---
    print("=" * 60)
    print("6. Últimas operaciones de mercado (página 1):")
    activity_page = api.get_activity_page(1)
    for tx in activity_page[:5]:
        print(f"   {tx['origin']} → {tx['tx_type']} {tx['player_name']} "
              f"({tx['amount']:,} €) → {tx['destination']}")
    print()

    print("Hecho.")
