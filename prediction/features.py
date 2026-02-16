"""
Feature Engineering para el modelo xP
======================================
Transforma el dataset bruto en features listos para entrenar.

Features generados:
    ── FORMA (Recency Bias) ──
    media_pts_3j          Media puntos últimas 3 jornadas
    media_pts_5j          Media puntos últimas 5 jornadas
    media_min_3j          Media minutos últimas 3 jornadas (detector de titularidad)
    media_min_5j          Media minutos últimas 5 jornadas
    tendencia_pts         Diferencia entre media_3j y media_5j (forma reciente)
    titular_pct_5j        % de jornadas titular (>60min) en últimas 5
    puntos_max_5j         Máximo de puntos en últimas 5 jornadas
    puntos_min_5j         Mínimo de puntos en últimas 5 jornadas

    ── DIFICULTAD DEL RIVAL (FDR) ──
    fdr_rival             Fixture Difficulty Rating (1-20, 20=más difícil)
    goles_contra_rival    Goles encajados por el rival por partido
    goles_favor_rival     Goles a favor del rival por partido
    posicion_rival        Posición en la tabla del rival

    ── CONTEXTO ──
    es_local              1=casa, 0=fuera
    posicion_id           Posición del jugador (1-5)

    ── MERCADO ──
    valor_mercado_norm    Valor de mercado normalizado (0-1)

    ── ODDS (Apuestas) ──
    prob_victoria         Probabilidad implícita de victoria del equipo
    prob_clean_sheet      Probabilidad de portería a cero (BTTS No)
    prob_btts             Probabilidad de que ambos marquen

    ── TARGET ──
    puntos                Puntos Fantasy de esa jornada (lo que predecimos)
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def load_raw_dataset(path: str | Path | None = None) -> dict:
    """Carga el dataset bruto generado por collect_data."""
    if path is None:
        path = DATA_DIR / "raw_dataset.json"
    with open(path) as f:
        return json.load(f)


def build_features(dataset: dict) -> pd.DataFrame:
    """
    Transforma el dataset bruto en un DataFrame con features.

    Input: dict con claves 'rows', 'standings', 'odds', 'fixtures'
    Output: DataFrame listo para entrenar, con columna 'puntos' como target
    """
    logger.info("Construyendo features...")

    rows = dataset["rows"]
    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("Dataset vacío")
        return df

    # Ordenar por jugador y jornada
    df = df.sort_values(["player_id", "jornada"]).reset_index(drop=True)

    # ─── FORMA (rolling averages) ─────────────────────────────
    logger.info("  Calculando forma (rolling averages)...")

    # Agrupar por jugador
    grp = df.groupby("player_id")

    # Media de puntos últimos 3 y 5 partidos (shift para no usar el actual)
    df["media_pts_3j"] = grp["puntos"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["media_pts_5j"] = grp["puntos"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Media de minutos (detector de titularidad)
    df["media_min_3j"] = grp["minutos"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["media_min_5j"] = grp["minutos"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Tendencia: si media_3j > media_5j → tendencia positiva
    df["tendencia_pts"] = df["media_pts_3j"] - df["media_pts_5j"]

    # % titularidad últimas 5 jornadas (>60 min = titular)
    df["es_titular"] = (df["minutos"] > 60).astype(int)
    df["titular_pct_5j"] = grp["es_titular"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Max/Min puntos últimas 5 jornadas
    df["puntos_max_5j"] = grp["puntos"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).max()
    )
    df["puntos_min_5j"] = grp["puntos"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).min()
    )

    # Stats rolling: media goles, asistencias, recuperaciones
    for col in ["goles", "asistencias", "recuperaciones", "tiros", "paradas"]:
        if col in df.columns:
            df[f"media_{col}_5j"] = grp[col].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).mean()
            )

    # ─── DIFICULTAD DEL RIVAL (FDR) ──────────────────────────
    logger.info("  Calculando FDR...")

    standings = dataset.get("standings", [])
    team_stats = {}
    for t in standings:
        tid = t["team_id"]
        team_stats[tid] = {
            "position": t["position"],
            "ga_per_match": t["goals_against_per_match"],
            "gf_per_match": t["goals_for_per_match"],
            "points": t["points"],
        }

    # FDR = posición del rival (1=líder=más difícil, 20=colista=más fácil)
    # Lo invertimos: 20=muy difícil, 1=fácil
    def _get_fdr(rival_id):
        if rival_id and rival_id in team_stats:
            return 21 - team_stats[rival_id]["position"]  # 20=líder, 1=colista
        return 10  # valor neutral

    df["fdr_rival"] = df["rival_ss_id"].apply(_get_fdr)

    # Goles que encaja el rival por partido (oportunidad de marcar)
    df["goles_contra_rival"] = df["rival_ss_id"].apply(
        lambda x: team_stats.get(x, {}).get("ga_per_match", 1.2)
    )

    # Goles a favor del rival (amenaza para el portero)
    df["goles_favor_rival"] = df["rival_ss_id"].apply(
        lambda x: team_stats.get(x, {}).get("gf_per_match", 1.2)
    )

    # Posición del rival en la tabla
    df["posicion_rival"] = df["rival_ss_id"].apply(
        lambda x: team_stats.get(x, {}).get("position", 10)
    )

    # ─── ODDS (Apuestas) ─────────────────────────────────────
    logger.info("  Añadiendo odds...")

    odds_data = dataset.get("odds", {})

    # Crear índice: (jornada, event_id) → odds
    odds_index = {}
    for rnd_str, events in odds_data.items():
        for ev_id_str, odds in events.items():
            odds_index[(int(rnd_str), int(ev_id_str))] = odds

    def _get_odds(row, field):
        key = (row.get("jornada"), row.get("event_id"))
        if key in odds_index:
            o = odds_index[key]
            # Si es local, prob_victoria = prob_home; si visitante = prob_away
            if field == "prob_victoria":
                return o["prob_home"] if row.get("es_local") == 1 else o["prob_away"]
            elif field == "prob_derrota":
                return o["prob_away"] if row.get("es_local") == 1 else o["prob_home"]
            else:
                return o.get(field, 0.5)
        return None

    df["prob_victoria"] = df.apply(lambda r: _get_odds(r, "prob_victoria"), axis=1)
    df["prob_derrota"] = df.apply(lambda r: _get_odds(r, "prob_derrota"), axis=1)
    df["prob_clean_sheet"] = df.apply(lambda r: _get_odds(r, "btts_no"), axis=1)
    df["prob_btts"] = df.apply(lambda r: _get_odds(r, "btts_yes"), axis=1)

    # ─── NORMALIZACIÓN ────────────────────────────────────────
    logger.info("  Normalizando...")

    # Valor de mercado normalizado (0-1)
    df["valor_mercado"] = pd.to_numeric(df["valor_mercado"], errors="coerce").fillna(0)
    max_val = df["valor_mercado"].max()
    df["valor_mercado_norm"] = df["valor_mercado"] / max_val if max_val > 0 else 0

    # ─── FEATURES FINALES ─────────────────────────────────────
    # Seleccionar columnas para el modelo
    FEATURE_COLS = [
        # Forma
        "media_pts_3j", "media_pts_5j",
        "media_min_3j", "media_min_5j",
        "tendencia_pts",
        "titular_pct_5j",
        "puntos_max_5j", "puntos_min_5j",
        # Stats rolling
        "media_goles_5j", "media_asistencias_5j",
        "media_recuperaciones_5j", "media_tiros_5j", "media_paradas_5j",
        # Rival
        "fdr_rival",
        "goles_contra_rival", "goles_favor_rival",
        "posicion_rival",
        # Contexto
        "es_local",
        "posicion_id",
        # Mercado
        "valor_mercado_norm",
        # Odds
        "prob_victoria", "prob_derrota",
        "prob_clean_sheet", "prob_btts",
    ]

    META_COLS = [
        "player_id", "nombre", "posicion", "equipo",
        "jornada", "puntos",
    ]

    # Verificar que existen
    existing_features = [c for c in FEATURE_COLS if c in df.columns]
    existing_meta = [c for c in META_COLS if c in df.columns]

    result = df[existing_meta + existing_features].copy()

    # Rellenar NaN con 0 para features (las primeras jornadas no tienen rolling)
    for col in existing_features:
        result[col] = result[col].fillna(0)

    # Eliminar filas donde es_local es None (sin fixture match)
    result = result[result["es_local"].notna()].copy()
    result["es_local"] = result["es_local"].astype(int)

    logger.info(
        "Dataset final: %d filas, %d features, %d meta",
        len(result), len(existing_features), len(existing_meta),
    )

    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = load_raw_dataset()
    df = build_features(dataset)
    print(f"\nDataset: {df.shape}")
    print(f"\nColumnas:\n{df.columns.tolist()}")
    print(f"\nMuestra:\n{df.head()}")
    print(f"\nEstadísticas target (puntos):")
    print(df["puntos"].describe())

    # Guardar CSV
    csv_path = DATA_DIR / "features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nGuardado: {csv_path}")
