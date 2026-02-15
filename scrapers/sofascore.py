"""
Scraper de Sofascore (api.sofascore.com)
=========================================
Extrae estadísticas detalladas de LaLiga vía la API pública de Sofascore:
    - Rating, goles, asistencias, xG, xA por jugador
    - Tiros, pases clave, regates, tackles, intercepciones
    - Tabla de clasificación de equipos

La API devuelve JSON paginado limpio, sin necesidad de parsear HTML.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.sofascore.com/api/v1"
TOURNAMENT_ID = 8  # LaLiga
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}
# Campos estadísticos que queremos extraer
STAT_FIELDS = (
    "rating,appearances,goals,assists,"
    "expectedGoals,expectedAssists,"
    "minutesPlayed,goalsAssistsSum,penaltyGoals,"
    "totalShots,shotsOnTarget,accuratePasses,"
    "keyPasses,successfulDribbles,tackles,interceptions,"
    "bigChancesCreated,bigChancesMissed,"
    "yellowCards,redCards,cleanSheet,saves"
)
# Límite por página de la API
PAGE_SIZE = 100


def _get_json(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ───────────────────────────────────────────────────────────────
# Season ID
# ───────────────────────────────────────────────────────────────
def get_current_season_id() -> tuple[int, str]:
    """
    Devuelve (season_id, season_name) de la temporada actual de LaLiga.
    """
    data = _get_json(f"{API_BASE}/unique-tournament/{TOURNAMENT_ID}/seasons")
    seasons = data.get("seasons", [])
    if not seasons:
        raise RuntimeError("No se encontraron temporadas en Sofascore.")
    current = seasons[0]
    return current["id"], current.get("name", "?")


# ───────────────────────────────────────────────────────────────
# Estadísticas de jugadores
# ───────────────────────────────────────────────────────────────
def scrape_player_stats(
    season_id: int | None = None,
    order: str = "-rating",
    max_players: int = 600,
) -> list[dict]:
    """
    Extrae estadísticas de todos los jugadores de LaLiga.

    Devuelve lista de dicts:
        {
            "id": int,
            "nombre": str,
            "equipo": str,
            "equipo_id": int,
            "rating": float,
            "partidos": int,
            "minutos": int,
            "goles": int,
            "goles_penalti": int,
            "asistencias": int,
            "goles_asistencias": int,
            "xG": float,
            "xA": float,
            "xG_por_90": float,
            "xA_por_90": float,
            "tiros_totales": int,
            "tiros_a_puerta": int,
            "pases_precisos": int,
            "pases_clave": int,
            "regates_exitosos": int,
            "tackles": int,
            "intercepciones": int,
            "grandes_ocasiones_creadas": int,
            "grandes_ocasiones_falladas": int,
            "amarillas": int,
            "rojas": int,
            "porteria_cero": int,
            "paradas": int,
        }
    """
    if season_id is None:
        season_id, _ = get_current_season_id()

    logger.info("Scrapeando estadísticas de Sofascore (season=%d)...", season_id)

    all_players = []
    offset = 0

    while offset < max_players:
        url = f"{API_BASE}/unique-tournament/{TOURNAMENT_ID}/season/{season_id}/statistics"
        params = {
            "limit": PAGE_SIZE,
            "offset": offset,
            "order": order,
            "accumulation": "total",
            "fields": STAT_FIELDS,
        }

        try:
            data = _get_json(url, params)
        except requests.HTTPError as e:
            logger.warning("Error en offset %d: %s", offset, e)
            break

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            player_info = item.get("player", {})
            team_info = item.get("team", {})

            minutes = item.get("minutesPlayed", 0)
            min_90 = minutes / 90 if minutes > 0 else 0
            xg = item.get("expectedGoals", 0) or 0
            xa = item.get("expectedAssists", 0) or 0

            all_players.append({
                "id": player_info.get("id"),
                "nombre": player_info.get("name", "?"),
                "equipo": team_info.get("name", "?"),
                "equipo_id": team_info.get("id"),
                "rating": round(item.get("rating", 0), 2),
                "partidos": item.get("appearances", 0),
                "minutos": minutes,
                "goles": item.get("goals", 0),
                "goles_penalti": item.get("penaltyGoals", 0),
                "asistencias": item.get("assists", 0),
                "goles_asistencias": item.get("goalsAssistsSum", 0),
                "xG": round(xg, 2),
                "xA": round(xa, 2),
                "xG_por_90": round(xg / min_90, 2) if min_90 > 0 else 0,
                "xA_por_90": round(xa / min_90, 2) if min_90 > 0 else 0,
                "tiros_totales": item.get("totalShots", 0),
                "tiros_a_puerta": item.get("shotsOnTarget", 0),
                "pases_precisos": item.get("accuratePasses", 0),
                "pases_clave": item.get("keyPasses", 0),
                "regates_exitosos": item.get("successfulDribbles", 0),
                "tackles": item.get("tackles", 0),
                "intercepciones": item.get("interceptions", 0),
                "grandes_ocasiones_creadas": item.get("bigChancesCreated", 0),
                "grandes_ocasiones_falladas": item.get("bigChancesMissed", 0),
                "amarillas": item.get("yellowCards", 0),
                "rojas": item.get("redCards", 0),
                "porteria_cero": item.get("cleanSheet", 0),
                "paradas": item.get("saves", 0),
            })

        pages = data.get("pages", 0)
        current_page = (offset // PAGE_SIZE) + 1
        logger.info(
            "  Página %d/%d — %d jugadores acumulados",
            current_page, pages, len(all_players),
        )

        offset += PAGE_SIZE
        if current_page >= pages:
            break

        # Pausa para no saturar la API
        time.sleep(0.3)

    logger.info("Jugadores de Sofascore: %d", len(all_players))
    return all_players


# ───────────────────────────────────────────────────────────────
# Clasificación
# ───────────────────────────────────────────────────────────────
def scrape_standings(season_id: int | None = None) -> list[dict]:
    """
    Extrae la clasificación de LaLiga.

    Devuelve:
        {
            "posicion": int,
            "equipo": str,
            "equipo_id": int,
            "partidos": int,
            "victorias": int, "empates": int, "derrotas": int,
            "goles_favor": int, "goles_contra": int,
            "diferencia_goles": int,
            "puntos": int,
        }
    """
    if season_id is None:
        season_id, _ = get_current_season_id()

    logger.info("Scrapeando clasificación de Sofascore (season=%d)...", season_id)

    data = _get_json(
        f"{API_BASE}/unique-tournament/{TOURNAMENT_ID}/season/{season_id}/standings/total"
    )

    standings = []
    for group in data.get("standings", []):
        for row in group.get("rows", []):
            team = row.get("team", {})
            standings.append({
                "posicion": row.get("position", 0),
                "equipo": team.get("name", "?"),
                "equipo_id": team.get("id"),
                "partidos": row.get("matches", 0),
                "victorias": row.get("wins", 0),
                "empates": row.get("draws", 0),
                "derrotas": row.get("losses", 0),
                "goles_favor": row.get("scoresFor", 0),
                "goles_contra": row.get("scoresAgainst", 0),
                "diferencia_goles": row.get("scoresFor", 0) - row.get("scoresAgainst", 0),
                "puntos": row.get("points", 0),
            })

    standings.sort(key=lambda x: x["posicion"])
    logger.info("Equipos en clasificación: %d", len(standings))
    return standings


# ───────────────────────────────────────────────────────────────
# Función combinada
# ───────────────────────────────────────────────────────────────
def scrape_all() -> dict:
    """
    Ejecuta todos los scrapers de Sofascore.

    Devuelve:
        {
            "fuente": "sofascore.com",
            "temporada": str,
            "season_id": int,
            "jugadores": [...],
            "clasificacion": [...],
        }
    """
    season_id, season_name = get_current_season_id()
    return {
        "fuente": "sofascore.com",
        "temporada": season_name,
        "season_id": season_id,
        "jugadores": scrape_player_stats(season_id),
        "clasificacion": scrape_standings(season_id),
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    data = scrape_all()
    print(f"\nJugadores: {len(data['jugadores'])}")
    print(f"Equipos: {len(data['clasificacion'])}")
    # Top 10 por rating
    top = sorted(data["jugadores"], key=lambda x: x["rating"], reverse=True)[:10]
    print("\nTop 10 por rating:")
    for p in top:
        print(
            f"  {p['nombre']:<25} rating={p['rating']:.2f}  "
            f"xG={p['xG']:.2f}  xA={p['xA']:.2f}  "
            f"goles={p['goles']}  asist={p['asistencias']}"
        )
