"""
Predicción xP para la próxima jornada
=======================================
Carga el modelo entrenado y genera predicciones para cada jugador
en la próxima jornada.

Uso:
    python -m prediction.predict
    python -m prediction.predict --model lightgbm
    python -m prediction.predict --top 30
    python -m prediction.predict --position DEL
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction.collect_data import (
    FANTASY_API,
    FANTASY_HEADERS,
    SOFASCORE_API,
    SOFASCORE_HEADERS,
    TOURNAMENT_ID,
    POS_MAP,
    _get,
    get_sofascore_season_id,
    collect_odds_for_round,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"


def load_model(model_type: str = "xgboost"):
    """Carga el modelo entrenado y sus metadatos."""
    model_path = MODELS_DIR / f"{model_type}_model.pkl"
    meta_path = MODELS_DIR / f"{model_type}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            f"Ejecuta primero: python -m prediction.train"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta


def get_next_round(season_id: int) -> tuple[int, list[dict], int]:
    """
    Encuentra la próxima jornada no jugada y devuelve sus partidos.
    Ignora jornadas que tienen solo 1-2 partidos aplazados (postponed)
    entre jornadas ya jugadas.

    Returns:
        (round_number, matches, first_match_timestamp)
        first_match_timestamp es el unix timestamp del primer partido de la jornada.
    """
    best_round = None
    best_matches = []

    for rnd in range(1, 39):
        data = _get(
            f"{SOFASCORE_API}/unique-tournament/{TOURNAMENT_ID}"
            f"/season/{season_id}/events/round/{rnd}",
            SOFASCORE_HEADERS,
        )
        events = data.get("events", [])
        if not events:
            break

        not_started = [e for e in events if e["status"]["type"] in ("notstarted", "postponed")]
        finished = [e for e in events if e["status"]["type"] == "finished"]

        # Si la mayoría de partidos aún no se han jugado, esta es la próxima jornada real
        if len(not_started) > len(finished) and len(not_started) >= 5:
            matches = []
            timestamps = []
            for ev in events:
                home = ev.get("homeTeam", {})
                away = ev.get("awayTeam", {})
                ts = ev.get("startTimestamp", 0)
                timestamps.append(ts)
                matches.append({
                    "event_id": ev.get("id"),
                    "home_team": home.get("name", "?"),
                    "home_team_id": home.get("id"),
                    "away_team": away.get("name", "?"),
                    "away_team_id": away.get("id"),
                    "status": ev.get("status", {}).get("type", "?"),
                    "start_timestamp": ts,
                })
            first_ts = min(ts for ts in timestamps if ts > 0) if timestamps else 0
            return rnd, matches, first_ts

        time.sleep(0.1)

    return 38, [], 0


def build_prediction_features(
    next_round: int,
    next_matches: list[dict],
    season_id: int,
) -> pd.DataFrame:
    """
    Construye features para la próxima jornada.
    Necesita:
    - Datos históricos del jugador (últimas jornadas)
    - Info del fixture (rival, local/visitante)
    - Odds
    """
    # 1. Cargar el dataset raw existente
    raw_path = DATA_DIR / "raw_dataset.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {raw_path}\n"
            f"Ejecuta primero: python -m prediction.collect_data"
        )

    with open(raw_path) as f:
        dataset = json.load(f)

    rows = pd.DataFrame(dataset["rows"])
    standings = dataset["standings"]
    team_mapping = dataset["team_mapping"]

    # 2. Calcular forma de cada jugador
    rows = rows.sort_values(["player_id", "jornada"])

    # Últimas stats por jugador
    latest = rows.groupby("player_id").tail(5)

    player_form = {}
    for pid, grp in latest.groupby("player_id"):
        grp = grp.sort_values("jornada")
        pts = grp["puntos"].values
        mins = grp["minutos"].values

        player_form[pid] = {
            "media_pts_3j": float(np.mean(pts[-3:])) if len(pts) >= 1 else 0,
            "media_pts_5j": float(np.mean(pts)) if len(pts) >= 1 else 0,
            "media_min_3j": float(np.mean(mins[-3:])) if len(mins) >= 1 else 0,
            "media_min_5j": float(np.mean(mins)) if len(mins) >= 1 else 0,
            "tendencia_pts": (
                float(np.mean(pts[-3:])) - float(np.mean(pts))
                if len(pts) >= 3 else 0
            ),
            "titular_pct_5j": float(np.mean(mins > 60)) if len(mins) >= 1 else 0,
            "puntos_max_5j": float(np.max(pts)) if len(pts) >= 1 else 0,
            "puntos_min_5j": float(np.min(pts)) if len(pts) >= 1 else 0,
            "media_goles_5j": float(grp["goles"].mean()),
            "media_asistencias_5j": float(grp["asistencias"].mean()),
            "media_recuperaciones_5j": float(grp["recuperaciones"].mean()),
            "media_tiros_5j": float(grp["tiros"].mean()),
            "media_paradas_5j": float(grp["paradas"].mean()),
        }

    # 3. Team stats
    team_stats = {}
    for t in standings:
        team_stats[t["team_id"]] = {
            "position": t["position"],
            "ga_per_match": t["goals_against_per_match"],
            "gf_per_match": t["goals_for_per_match"],
        }

    # 4. Fixture index: sofascore_team_id → fixture info
    fixture_index = {}
    for m in next_matches:
        fixture_index[m["home_team_id"]] = {
            "es_local": 1,
            "rival_team_id": m["away_team_id"],
            "rival_team": m["away_team"],
            "event_id": m["event_id"],
        }
        fixture_index[m["away_team_id"]] = {
            "es_local": 0,
            "rival_team_id": m["home_team_id"],
            "rival_team": m["home_team"],
            "event_id": m["event_id"],
        }

    # 5. Odds
    odds_by_event = collect_odds_for_round(next_matches)

    # 6. Get player list with market value
    players = _get(f"{FANTASY_API}/api/v3/players", FANTASY_HEADERS)
    active = [p for p in players if p.get("points", 0) > 0]

    # Max market value for normalization
    def _to_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    max_val = max(_to_int(p.get("marketValue", 0)) for p in active) or 1

    # 7. Build prediction rows
    pred_rows = []
    for p in active:
        pid = p["id"]
        team_name = p.get("team", {}).get("name", "")
        ss_team_id = team_mapping.get(team_name)

        if ss_team_id is None or ss_team_id not in fixture_index:
            continue

        fixture = fixture_index[ss_team_id]
        rival_id = fixture["rival_team_id"]
        event_id = fixture["event_id"]

        # Form
        form = player_form.get(pid, {})

        # FDR
        rival_stats = team_stats.get(rival_id, {})
        fdr = 21 - rival_stats.get("position", 10)

        # Odds
        event_odds = odds_by_event.get(event_id, {})
        if fixture["es_local"] == 1:
            prob_victoria = event_odds.get("prob_home", 0.33)
            prob_derrota = event_odds.get("prob_away", 0.33)
        else:
            prob_victoria = event_odds.get("prob_away", 0.33)
            prob_derrota = event_odds.get("prob_home", 0.33)

        pos_id = int(p.get("positionId", 0)) if p.get("positionId") else 0

        row = {
            "player_id": pid,
            "nombre": p.get("nickname") or p.get("name", "?"),
            "posicion_id": pos_id,
            "posicion": POS_MAP.get(pos_id, "?"),
            "equipo": team_name,
            "jornada": next_round,
            "rival": fixture["rival_team"],
            # Features
            "media_pts_3j": form.get("media_pts_3j", 0),
            "media_pts_5j": form.get("media_pts_5j", 0),
            "media_min_3j": form.get("media_min_3j", 0),
            "media_min_5j": form.get("media_min_5j", 0),
            "tendencia_pts": form.get("tendencia_pts", 0),
            "titular_pct_5j": form.get("titular_pct_5j", 0),
            "puntos_max_5j": form.get("puntos_max_5j", 0),
            "puntos_min_5j": form.get("puntos_min_5j", 0),
            "media_goles_5j": form.get("media_goles_5j", 0),
            "media_asistencias_5j": form.get("media_asistencias_5j", 0),
            "media_recuperaciones_5j": form.get("media_recuperaciones_5j", 0),
            "media_tiros_5j": form.get("media_tiros_5j", 0),
            "media_paradas_5j": form.get("media_paradas_5j", 0),
            "fdr_rival": fdr,
            "goles_contra_rival": rival_stats.get("ga_per_match", 1.2),
            "goles_favor_rival": rival_stats.get("gf_per_match", 1.2),
            "posicion_rival": rival_stats.get("position", 10),
            "es_local": fixture["es_local"],
            "valor_mercado_norm": _to_int(p.get("marketValue", 0)) / max_val,
            "prob_victoria": prob_victoria,
            "prob_derrota": prob_derrota,
            "prob_clean_sheet": event_odds.get("btts_no", 0.5),
            "prob_btts": event_odds.get("btts_yes", 0.5),
        }
        pred_rows.append(row)

    return pd.DataFrame(pred_rows)


def predict(model_type: str = "xgboost") -> tuple[pd.DataFrame, int]:
    """
    Pipeline completo de predicción para la próxima jornada.

    Returns:
        (pred_df, first_match_timestamp)
        first_match_timestamp: unix timestamp del primer partido de la jornada.
    """
    logger.info("Cargando modelo %s...", model_type)
    model, meta = load_model(model_type)
    feature_cols = meta["feature_cols"]

    logger.info("Buscando próxima jornada...")
    season_id = get_sofascore_season_id()
    next_round, next_matches, first_match_ts = get_next_round(season_id)
    logger.info("Próxima jornada: %d (%d partidos)", next_round, len(next_matches))

    if not next_matches:
        logger.warning("No hay partidos pendientes")
        return pd.DataFrame(), first_match_ts

    logger.info("Construyendo features de predicción...")
    pred_df = build_prediction_features(next_round, next_matches, season_id)

    if pred_df.empty:
        logger.warning("No hay jugadores para predecir")
        return pd.DataFrame(), first_match_ts

    # Asegurar que tenemos todas las features
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0

    # Convertir tipos numéricos
    X = pred_df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    pred_df["xP"] = model.predict(X).round(2)

    # Ordenar por xP descendente
    pred_df = pred_df.sort_values("xP", ascending=False).reset_index(drop=True)

    return pred_df, first_match_ts


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Predice xP para la próxima jornada")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--top", type=int, default=30, help="Top N jugadores a mostrar")
    parser.add_argument("--position", type=str, help="Filtrar por posición (POR/DEF/MED/DEL/ENT)")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  xP Predictions — LaLiga Fantasy")
    print("=" * 70)

    pred_df, _ = predict(args.model)

    if pred_df.empty:
        print("\n  No hay predicciones disponibles.")
        return

    jornada = int(pred_df["jornada"].iloc[0])
    print(f"\n  Jornada {jornada} — Modelo: {args.model}")
    print(f"  Jugadores: {len(pred_df)}")

    # Filtrar por posición
    if args.position:
        pred_df = pred_df[pred_df["posicion"] == args.position.upper()]

    # Mostrar top N
    top = pred_df.head(args.top)

    print()
    print(f"  {'#':<3} {'Jugador':<22} {'Pos':>4} {'Equipo':<18} "
          f"{'Rival':<18} {'Local':>5} {'xP':>6} {'Media5':>7}")
    print(f"  {'─'*3} {'─'*22} {'─'*4} {'─'*18} {'─'*18} {'─'*5} {'─'*6} {'─'*7}")

    for i, (_, row) in enumerate(top.iterrows(), 1):
        local = "🏠" if row["es_local"] == 1 else "✈️ "
        print(
            f"  {i:<3} {row['nombre']:<22} {row['posicion']:>4} "
            f"{row['equipo']:<18} {row['rival']:<18} "
            f"{local:>5} {row['xP']:>6.1f} {row['media_pts_5j']:>7.1f}"
        )

    # Resumen por posición
    print()
    print("  Top 3 por posición:")
    for pos in ["POR", "DEF", "MED", "DEL"]:
        pos_df = pred_df[pred_df["posicion"] == pos].head(3)
        names = ", ".join(
            f"{r['nombre']} ({r['xP']:.1f})"
            for _, r in pos_df.iterrows()
        )
        print(f"    {pos}: {names}")

    # Guardar predicciones
    output_path = DATA_DIR / f"predictions_j{jornada}.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"\n  Guardado: {output_path}")
    print()


if __name__ == "__main__":
    main()
