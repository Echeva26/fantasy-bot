"""
Recolector de datos históricos para el modelo xP
==================================================
Recopila de 4 fuentes y genera un dataset por jugador-jornada:

1. LaLiga Fantasy API — Puntos y stats por jornada de cada jugador
2. Sofascore API      — Fixtures (quién juega contra quién, local/visitante)
3. Sofascore API      — Standings (fuerza defensiva de cada equipo)
4. Sofascore API      — Odds de apuestas por partido

Uso:
    python -m prediction.collect_data
    python -m prediction.collect_data --max-players 50  # Para pruebas rápidas
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# ─── Constantes ───────────────────────────────────────────────
FANTASY_API = "https://api-fantasy.llt-services.com"
SOFASCORE_API = "https://api.sofascore.com/api/v1"
TOURNAMENT_ID = 8  # LaLiga
DATA_DIR = Path(__file__).parent / "data"

FANTASY_HEADERS = {
    "User-Agent": "okhttp/4.12.0",
    "X-App": "Fantasy",
    "X-Lang": "es",
}
SOFASCORE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Mapeo posición ID → nombre
POS_MAP = {1: "POR", 2: "DEF", 3: "MED", 4: "DEL", 5: "ENT"}


def _get(url: str, headers: dict, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited. Esperando %ds...", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    return {}


# ─── 1. Jugadores de LaLiga Fantasy ──────────────────────────
def collect_fantasy_players() -> list[dict]:
    """
    Obtiene la lista de todos los jugadores de LaLiga Fantasy.
    Solo devuelve los activos (con puntos > 0).
    """
    logger.info("Obteniendo lista de jugadores de LaLiga Fantasy...")
    players = _get(f"{FANTASY_API}/api/v3/players", FANTASY_HEADERS)
    active = [p for p in players if p.get("points", 0) > 0]
    logger.info("Jugadores activos: %d / %d", len(active), len(players))
    return active


# ─── 2. Detalle por jornada de cada jugador ───────────────────
def collect_player_detail(player_id: int) -> dict | None:
    """
    Obtiene stats por jornada de un jugador.
    GET /api/v3/player/{id}
    """
    try:
        data = _get(f"{FANTASY_API}/api/v3/player/{player_id}", FANTASY_HEADERS)
        return data
    except Exception as e:
        logger.warning("Error player %d: %s", player_id, e)
        return None


def collect_all_player_details(
    player_ids: list[int],
    max_workers: int = 10,
) -> dict[int, dict]:
    """
    Descarga en paralelo los detalles de todos los jugadores.
    Devuelve {player_id: detail_dict}.
    """
    logger.info("Descargando detalles de %d jugadores...", len(player_ids))
    results = {}
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(collect_player_detail, pid): pid
            for pid in player_ids
        }
        for future in as_completed(futures):
            pid = futures[future]
            detail = future.result()
            if detail and "playerStats" in detail:
                results[pid] = detail
            done += 1
            if done % 50 == 0:
                logger.info("  Progreso: %d/%d", done, len(player_ids))
                time.sleep(0.3)  # Pausa anti rate-limit

    logger.info("Detalles obtenidos: %d/%d", len(results), len(player_ids))
    return results


# ─── 3. Sofascore: Season ID ─────────────────────────────────
def get_sofascore_season_id() -> int:
    data = _get(
        f"{SOFASCORE_API}/unique-tournament/{TOURNAMENT_ID}/seasons",
        SOFASCORE_HEADERS,
    )
    return data["seasons"][0]["id"]


# ─── 4. Sofascore: Fixtures por jornada ──────────────────────
def collect_fixtures(season_id: int, max_round: int = 38) -> dict[int, list[dict]]:
    """
    Obtiene los partidos de cada jornada.
    Devuelve {round_number: [match_dicts]}.

    Cada match_dict:
        {
            "event_id": int,
            "home_team": str,
            "home_team_id": int,
            "away_team": str,
            "away_team_id": int,
            "home_score": int | None,
            "away_score": int | None,
            "status": str,
        }
    """
    logger.info("Obteniendo fixtures de Sofascore (hasta jornada %d)...", max_round)
    fixtures = {}

    for rnd in range(1, max_round + 1):
        try:
            data = _get(
                f"{SOFASCORE_API}/unique-tournament/{TOURNAMENT_ID}"
                f"/season/{season_id}/events/round/{rnd}",
                SOFASCORE_HEADERS,
            )
            events = data.get("events", [])
            if not events:
                break

            matches = []
            for ev in events:
                home = ev.get("homeTeam", {})
                away = ev.get("awayTeam", {})
                hs = ev.get("homeScore", {})
                aws = ev.get("awayScore", {})
                matches.append({
                    "event_id": ev.get("id"),
                    "home_team": home.get("name", "?"),
                    "home_team_id": home.get("id"),
                    "away_team": away.get("name", "?"),
                    "away_team_id": away.get("id"),
                    "home_score": hs.get("current"),
                    "away_score": aws.get("current"),
                    "status": ev.get("status", {}).get("type", "?"),
                })
            fixtures[rnd] = matches
            time.sleep(0.2)
        except Exception as e:
            logger.warning("Error round %d: %s", rnd, e)
            break

    logger.info("Fixtures: %d jornadas con datos", len(fixtures))
    return fixtures


# ─── 5. Sofascore: Standings (fuerza defensiva) ──────────────
def collect_standings(season_id: int) -> list[dict]:
    """
    Obtiene la clasificación para calcular fuerza defensiva.
    """
    logger.info("Obteniendo clasificación de Sofascore...")
    data = _get(
        f"{SOFASCORE_API}/unique-tournament/{TOURNAMENT_ID}"
        f"/season/{season_id}/standings/total",
        SOFASCORE_HEADERS,
    )

    rows = []
    for group in data.get("standings", []):
        for row in group.get("rows", []):
            team = row.get("team", {})
            matches = row.get("matches", 1)
            ga = row.get("scoresAgainst", 0)
            gf = row.get("scoresFor", 0)
            rows.append({
                "team_id": team.get("id"),
                "team": team.get("name", "?"),
                "matches": matches,
                "wins": row.get("wins", 0),
                "draws": row.get("draws", 0),
                "losses": row.get("losses", 0),
                "goals_for": gf,
                "goals_against": ga,
                "goals_for_per_match": round(gf / matches, 2) if matches else 0,
                "goals_against_per_match": round(ga / matches, 2) if matches else 0,
                "points": row.get("points", 0),
                "position": row.get("position", 0),
            })

    rows.sort(key=lambda x: x["position"])
    logger.info("Equipos en clasificación: %d", len(rows))
    return rows


# ─── 6. Sofascore: Odds por partido ──────────────────────────
def collect_odds_for_round(
    fixtures_round: list[dict],
) -> dict[int, dict]:
    """
    Obtiene las odds de apuestas para todos los partidos de una jornada.
    Devuelve {event_id: odds_dict}.

    odds_dict:
        {
            "prob_home": float,    # probabilidad implícita victoria local
            "prob_draw": float,
            "prob_away": float,
            "btts_yes": float,     # probabilidad de que ambos marquen
            "btts_no": float,      # probabilidad de clean sheet
        }
    """
    results = {}
    for match in fixtures_round:
        ev_id = match["event_id"]
        if match["status"] == "notstarted" or match["status"] == "finished":
            try:
                data = _get(
                    f"{SOFASCORE_API}/event/{ev_id}/odds/1/all",
                    SOFASCORE_HEADERS,
                )
                odds = _parse_odds(data)
                results[ev_id] = odds
                time.sleep(0.15)
            except Exception:
                pass
    return results


def _parse_odds(data: dict) -> dict:
    """Parsea las odds de Sofascore a probabilidades implícitas."""
    odds = {
        "prob_home": 0.33,
        "prob_draw": 0.33,
        "prob_away": 0.33,
        "btts_yes": 0.5,
        "btts_no": 0.5,
    }

    for market in data.get("markets", []):
        market_id = market.get("marketId")
        choices = market.get("choices", [])

        if market_id == 1:  # Full time result (1X2)
            for c in choices:
                frac = c.get("fractionalValue", "")
                decimal_odds = _frac_to_decimal(frac)
                if decimal_odds <= 0:
                    continue
                prob = 1.0 / decimal_odds
                name = c.get("name", "")
                if name == "1":
                    odds["prob_home"] = round(prob, 4)
                elif name == "X":
                    odds["prob_draw"] = round(prob, 4)
                elif name == "2":
                    odds["prob_away"] = round(prob, 4)

        elif market_id == 5:  # Both teams to score
            for c in choices:
                frac = c.get("fractionalValue", "")
                decimal_odds = _frac_to_decimal(frac)
                if decimal_odds <= 0:
                    continue
                prob = 1.0 / decimal_odds
                name = c.get("name", "")
                if name == "Yes":
                    odds["btts_yes"] = round(prob, 4)
                elif name == "No":
                    odds["btts_no"] = round(prob, 4)

    return odds


def _frac_to_decimal(frac: str) -> float:
    """Convierte '3/1' a 4.0 (decimal odds)."""
    try:
        parts = frac.split("/")
        if len(parts) == 2:
            return float(parts[0]) / float(parts[1]) + 1
        return float(frac)
    except (ValueError, ZeroDivisionError):
        return 0


# ─── 7. Mapeo Fantasy Team → Sofascore Team ──────────────────
def build_team_mapping(
    fantasy_players: list[dict],
    sofascore_standings: list[dict],
) -> dict[str, int]:
    """
    Mapea nombres de equipo de Fantasy a team_id de Sofascore.
    Usa fuzzy matching por similitud de nombres.
    """
    # Extraer nombres únicos de equipos en Fantasy
    fantasy_teams = set()
    for p in fantasy_players:
        team = p.get("team", {})
        name = team.get("name", "")
        tid = team.get("id", "")
        if name:
            fantasy_teams.add((name, tid))

    # Sofascore teams
    ss_teams = {t["team"]: t["team_id"] for t in sofascore_standings}

    # Matching manual + fuzzy
    mapping = {}
    MANUAL_MAP = {
        "Club Atlético Osasuna": "Osasuna",
        "C.A. Osasuna": "Osasuna",
        "Real Club Celta de Vigo": "Celta Vigo",
        "R.C. Celta de Vigo": "Celta Vigo",
        "Real Betis Balompié": "Real Betis",
        "R. Betis Balompié": "Real Betis",
        "R.C.D. Mallorca": "Mallorca",
        "Real Valladolid C.F.": "Real Valladolid",
        "Villarreal C.F.": "Villarreal",
        "Getafe C.F.": "Getafe",
        "Girona F.C.": "Girona",
        "Girona FC": "Girona",
        "Valencia C.F.": "Valencia",
        "Sevilla F.C.": "Sevilla",
        "Sevilla FC": "Sevilla",
        "Athletic Club": "Athletic Club",
        "F.C. Barcelona": "Barcelona",
        "FC Barcelona": "Barcelona",
        "Real Madrid C.F.": "Real Madrid",
        "Real Madrid CF": "Real Madrid",
        "Club Atlético de Madrid": "Atlético Madrid",
        "Atlético de Madrid": "Atlético Madrid",
        "Rayo Vallecano de Madrid": "Rayo Vallecano",
        "Rayo Vallecano": "Rayo Vallecano",
        "R.C.D. Espanyol de Barcelona": "Espanyol",
        "R.C.D. Espanyol": "Espanyol",
        "Espanyol": "Espanyol",
        "Real Sociedad de Fútbol": "Real Sociedad",
        "Real Sociedad": "Real Sociedad",
        "Deportivo Alavés": "Deportivo Alavés",
        "D. Alavés": "Deportivo Alavés",
        "Levante U.D.": "Levante",
        "Levante UD": "Levante",
        "Elche C.F.": "Elche",
        "Elche CF": "Elche",
        "C.D. Leganés": "Leganés",
        "U.D. Las Palmas": "Las Palmas",
        "U.D. Almería": "Almería",
        "Real Oviedo": "Real Oviedo",
        "Cádiz C.F.": "Cádiz",
        "Cádiz CF": "Cádiz",
        "Granada CF": "Granada CF",
    }

    for f_name, f_id in fantasy_teams:
        # Intenta mapeo manual primero
        ss_name = MANUAL_MAP.get(f_name)
        if ss_name and ss_name in ss_teams:
            mapping[f_name] = ss_teams[ss_name]
            continue

        # Fuzzy: busca substring match
        f_lower = f_name.lower()
        for ss_name, ss_id in ss_teams.items():
            ss_lower = ss_name.lower()
            if (
                ss_lower in f_lower
                or f_lower in ss_lower
                or any(
                    w in ss_lower
                    for w in f_lower.split()
                    if len(w) > 3
                )
            ):
                mapping[f_name] = ss_id
                break

    logger.info(
        "Team mapping: %d/%d equipos mapeados",
        len(mapping), len(fantasy_teams),
    )
    # Mostrar no mapeados
    for f_name, _ in fantasy_teams:
        if f_name not in mapping:
            logger.warning("  Sin mapear: %s", f_name)

    return mapping


# ─── 8. Construir dataset combinado ──────────────────────────
def build_raw_dataset(
    player_details: dict[int, dict],
    fantasy_players: list[dict],
    fixtures: dict[int, list[dict]],
    standings: list[dict],
    team_mapping: dict[str, int],
) -> list[dict]:
    """
    Combina todas las fuentes en un dataset plano: una fila por
    (jugador, jornada).

    Columnas:
        player_id, nombre, posicion, equipo, equipo_sofascore_id,
        jornada, puntos,
        minutos, goles, asistencias, goles_encajados, amarillas, rojas,
        saves, effective_clearance, total_scoring_att, won_contest,
        ball_recovery, poss_lost_all,
        es_local, rival, rival_sofascore_id,
        valor_mercado,
    """
    logger.info("Construyendo dataset combinado...")

    # Indexar: sofascore_team_id → standings info
    team_stats = {t["team_id"]: t for t in standings}

    # Indexar: (round, sofascore_team_id) → {es_local, rival_team_id, event_id}
    fixture_index = {}
    for rnd, matches in fixtures.items():
        for m in matches:
            fixture_index[(rnd, m["home_team_id"])] = {
                "es_local": 1,
                "rival_team_id": m["away_team_id"],
                "rival_team": m["away_team"],
                "event_id": m["event_id"],
                "status": m["status"],
            }
            fixture_index[(rnd, m["away_team_id"])] = {
                "es_local": 0,
                "rival_team_id": m["home_team_id"],
                "rival_team": m["home_team"],
                "event_id": m["event_id"],
                "status": m["status"],
            }

    # Indexar fantasy players
    fp_by_id = {p["id"]: p for p in fantasy_players}

    rows = []
    for pid, detail in player_details.items():
        fp = fp_by_id.get(pid, {})
        team_name = fp.get("team", {}).get("name", "")
        ss_team_id = team_mapping.get(team_name)

        player_stats = detail.get("playerStats", [])
        for week_data in player_stats:
            week = week_data.get("weekNumber")
            if week is None:
                continue

            points = week_data.get("totalPoints", 0)
            stats = week_data.get("stats", {})

            # Extraer stats (formato: [valor, puntos_fantasy])
            def _stat(key):
                val = stats.get(key, [0, 0])
                return val[0] if isinstance(val, list) and val else 0

            # Buscar fixture
            fixture = fixture_index.get((week, ss_team_id), {})

            row = {
                "player_id": pid,
                "nombre": detail.get("nickname") or detail.get("name", "?"),
                "posicion_id": detail.get("positionId", 0),
                "posicion": POS_MAP.get(detail.get("positionId", 0), "?"),
                "equipo": team_name,
                "equipo_ss_id": ss_team_id,
                "jornada": week,
                "puntos": points,
                # Stats del partido
                "minutos": _stat("mins_played"),
                "goles": _stat("goals"),
                "asistencias": _stat("goal_assist"),
                "goles_encajados": _stat("goals_conceded"),
                "amarillas": _stat("yellow_card"),
                "doble_amarilla": _stat("second_yellow_card"),
                "rojas": _stat("red_card"),
                "paradas": _stat("saves"),
                "despejes": _stat("effective_clearance"),
                "tiros": _stat("total_scoring_att"),
                "regates_ganados": _stat("won_contest"),
                "recuperaciones": _stat("ball_recovery"),
                "posesiones_perdidas": _stat("poss_lost_all"),
                "penaltis_provocados": _stat("penalty_won"),
                "penaltis_cometidos": _stat("penalty_conceded"),
                "penaltis_parados": _stat("penalty_save"),
                "penaltis_fallados": _stat("penalty_failed"),
                "goles_propia": _stat("own_goals"),
                "pases_gol_fallados": _stat("offtarget_att_assist"),
                "entradas_area": _stat("pen_area_entries"),
                "marca_points": _stat("marca_points"),
                # Fixture
                "es_local": fixture.get("es_local"),
                "rival": fixture.get("rival_team", ""),
                "rival_ss_id": fixture.get("rival_team_id"),
                "event_id": fixture.get("event_id"),
                # Mercado
                "valor_mercado": fp.get("marketValue", 0),
            }
            rows.append(row)

    logger.info("Dataset: %d filas (jugador-jornada)", len(rows))
    return rows


# ─── 9. Guardar ──────────────────────────────────────────────
def save_dataset(data: dict, filename: str = "raw_dataset.json"):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Guardado: %s", path)
    return path


# ─── Main ─────────────────────────────────────────────────────
def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Recolecta datos históricos para el modelo xP"
    )
    parser.add_argument(
        "--max-players", type=int, default=0,
        help="Limitar jugadores (0 = todos, útil para testing)",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  xP Data Collector — LaLiga Fantasy")
    print("=" * 60)

    # 1. Jugadores de Fantasy
    fantasy_players = collect_fantasy_players()

    player_ids = [p["id"] for p in fantasy_players]
    if args.max_players > 0:
        player_ids = player_ids[: args.max_players]

    # 2. Detalle por jornada
    player_details = collect_all_player_details(player_ids, max_workers=10)

    # 3. Sofascore data
    season_id = get_sofascore_season_id()
    fixtures = collect_fixtures(season_id)
    standings = collect_standings(season_id)

    # 4. Team mapping
    team_mapping = build_team_mapping(fantasy_players, standings)

    # 5. Odds (solo jornadas ya jugadas para entrenamiento)
    all_odds = {}
    for rnd, matches in fixtures.items():
        has_finished = any(m["status"] == "finished" for m in matches)
        if has_finished:
            round_odds = collect_odds_for_round(matches)
            all_odds[rnd] = round_odds

    logger.info("Odds recopiladas para %d jornadas", len(all_odds))

    # 6. Construir dataset
    rows = build_raw_dataset(
        player_details, fantasy_players,
        fixtures, standings, team_mapping,
    )

    # 7. Guardar todo
    dataset = {
        "rows": rows,
        "fixtures": {str(k): v for k, v in fixtures.items()},
        "standings": standings,
        "odds": {str(k): {str(eid): o for eid, o in v.items()} for k, v in all_odds.items()},
        "team_mapping": team_mapping,
        "num_players": len(player_details),
        "num_rounds": len(fixtures),
    }

    path = save_dataset(dataset)

    print()
    print("=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Jugadores:  {len(player_details)}")
    print(f"  Jornadas:   {len(fixtures)}")
    print(f"  Filas:      {len(rows)}")
    print(f"  Odds:       {sum(len(v) for v in all_odds.values())} partidos")
    print(f"  Guardado:   {path}")
    print()


if __name__ == "__main__":
    main()
