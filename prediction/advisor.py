"""
Fantasy Advisor — Recomendaciones inteligentes para la próxima jornada
======================================================================
Conecta el modelo xP con el estado real de tu liga para generar
un informe .md con recomendaciones personalizadas:

    1. xP de tu plantilla actual (quién juega, quién no)
    2. Jugadores alcanzables mejores que los tuyos (mercado + clausulazos)
    3. Recomendaciones concretas: a quién fichar, a quién vender, qué once poner

Uso:
    python -m prediction.advisor
    python -m prediction.advisor --league 016615640
    python -m prediction.advisor --output informe.md
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction.predict import predict

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent
MODEL_TYPE = "xgboost"


# ─── 1. Cargar estado de la liga ──────────────────────────────
def load_snapshot(league_id: str = "") -> dict:
    """
    Genera un snapshot fresco del estado de la liga.
    Requiere token guardado en .laliga_token.
    """
    from laliga_fantasy_client import (
        LaLigaFantasyClient,
        get_league_snapshot,
    )

    client = LaLigaFantasyClient.from_saved_token(league_id=league_id)

    # Si no hay league_id, buscar la primera liga
    if not client.league_id:
        leagues = client.get_leagues()
        if not leagues:
            raise RuntimeError("No se encontraron ligas para este usuario.")
        client.league_id = str(leagues[0]["id"])
        logger.info("Liga seleccionada: %s (%s)", leagues[0].get("name"), client.league_id)

    snapshot = get_league_snapshot(client)
    return snapshot


def load_snapshot_from_file(path: str | Path) -> dict:
    """Carga un snapshot desde un JSON existente."""
    with open(path) as f:
        return json.load(f)


# ─── Utilidad: ventana de clausulazos ─────────────────────────
CLAUSULAZO_LOCKOUT_HOURS = 24  # Regla: prohibidos desde 24h antes del primer partido
BID_COMPETITION_MIN_RAISE_EUR = 2_000_000
BID_COMPETITION_MAX_RAISE_EUR = 4_000_000
BID_COMPETITION_ROUND_STEP_EUR = 10_000
BID_COMPETITION_REFERENCE_XP_LOW = 4.0
BID_COMPETITION_REFERENCE_XP_HIGH = 9.0
BID_COMPETITION_REFERENCE_VALUE_LOW = 8_000_000
BID_COMPETITION_REFERENCE_VALUE_HIGH = 25_000_000
BID_COMPETITION_BID_PRESSURE_STEP_EUR = 250_000
BID_COMPETITION_BID_PRESSURE_CAP_EUR = 750_000


def _round_up_to_step(value: int, step: int) -> int:
    if step <= 0:
        return int(value)
    n = int(value)
    if n <= 0:
        return 0
    return ((n + step - 1) // step) * step


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_competitive_bid_amount(
    *,
    base_amount: int,
    market_value: int = 0,
    external_bids: int = 0,
    expected_points: float = 0.0,
) -> int:
    """
    Ajusta la puja si ya hay competencia en el mercado.

    Regla:
    - Sin pujas de terceros: mantener base (mínimo legal).
    - Con pujas de terceros: sumar un incremento absoluto entre 2M y 4M,
      adaptado al xP y valor del jugador.
    """
    amount = max(int(base_amount or 0), int(market_value or 0))
    competitors = max(0, int(external_bids or 0))
    if competitors <= 0:
        return amount

    xp = float(expected_points or 0.0)
    xp_norm = _clamp01(
        (xp - BID_COMPETITION_REFERENCE_XP_LOW)
        / max(0.1, BID_COMPETITION_REFERENCE_XP_HIGH - BID_COMPETITION_REFERENCE_XP_LOW)
    )

    value_ref = max(amount, int(market_value or 0))
    value_norm = _clamp01(
        (value_ref - BID_COMPETITION_REFERENCE_VALUE_LOW)
        / max(
            1.0,
            BID_COMPETITION_REFERENCE_VALUE_HIGH - BID_COMPETITION_REFERENCE_VALUE_LOW,
        )
    )

    # Bonus principal en rango [2M, 4M] según contexto del jugador.
    premium_abs = (
        BID_COMPETITION_MIN_RAISE_EUR
        + int(round(1_250_000 * xp_norm))
        + int(round(750_000 * value_norm))
    )

    # Más presión de mercado => acercar al tope.
    if competitors > 1:
        pressure_bonus = min(
            BID_COMPETITION_BID_PRESSURE_CAP_EUR,
            (competitors - 1) * BID_COMPETITION_BID_PRESSURE_STEP_EUR,
        )
        premium_abs += int(pressure_bonus)

    premium_abs = max(
        BID_COMPETITION_MIN_RAISE_EUR,
        min(BID_COMPETITION_MAX_RAISE_EUR, premium_abs),
    )
    return _round_up_to_step(amount + premium_abs, BID_COMPETITION_ROUND_STEP_EUR)


def clausulazos_available(first_match_ts: int) -> tuple[bool, float]:
    """
    Determina si los clausulazos están disponibles según la fecha del
    primer partido de la jornada.

    En LaLiga Fantasy, los clausulazos están prohibidos desde 24h antes
    del inicio de la jornada. Si quedan 24h o menos para el primer
    partido, solo quedan disponibles ventas, subidas de cláusula y
    compras de mercado mediante pujas.

    Returns:
        (disponible: bool, horas_restantes: float)
        horas_restantes = horas hasta el primer partido (negativo si ya empezó)
    """
    if first_match_ts <= 0:
        return True, 999.0  # Sin datos, asumimos disponible

    now = datetime.now(timezone.utc)
    first_match = datetime.fromtimestamp(first_match_ts, tz=timezone.utc)
    delta = first_match - now
    horas = delta.total_seconds() / 3600

    disponible = horas > CLAUSULAZO_LOCKOUT_HOURS
    return disponible, round(horas, 1)


def _parse_api_datetime(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def get_laliga_calendar_first_match_ts() -> tuple[int, str]:
    """
    Obtiene el próximo primer partido futuro desde la API de LaLiga Fantasy.

    Esta es la fuente operativa para la regla de clausulazos, porque es la misma
    competición que aplica el bloqueo 24h. No depende de Sofascore ni de que se
    regeneren predicciones.
    """
    from laliga_fantasy_client import LaLigaFantasyPublic

    api = LaLigaFantasyPublic()
    current_week = api.get_current_week() or {}
    calendar = api.get_calendar() or []

    dates: list[datetime] = []
    for item in calendar:
        if not isinstance(item, dict):
            continue
        dt = _parse_api_datetime(
            item.get("matchDate")
            or item.get("date")
            or item.get("time")
            or item.get("startDate")
        )
        if dt:
            dates.append(dt)

    if not dates:
        raise RuntimeError("No se encontraron fechas de partido en /api/v3/calendar.")

    now = datetime.now(timezone.utc)
    # Si la jornada actual ya ha empezado, el mínimo global queda en el pasado y
    # provoca urgencias falsas. Para alineación y clausulazos usamos el próximo
    # inicio de jornada/partido que aún no haya ocurrido.
    future_dates = sorted(dt for dt in dates if dt >= now - timedelta(minutes=5))
    target_dates = future_dates or sorted(dates)
    week = current_week.get("weekNumber") or current_week.get("nextWeek") or "?"
    source = f"laliga_calendar_week_{week}"
    if not future_dates:
        source += "_past_only"
    return int(target_dates[0].timestamp()), source


def current_week_clausulazos_available(
    fallback_first_match_ts: int = 0,
    *,
    fail_closed: bool = False,
) -> tuple[bool, float, str]:
    """
    Determina la ventana de clausulazos para la jornada actual.

    Prioridad:
    1. Calendario oficial de LaLiga Fantasy.
    2. Si `fail_closed=True`, cerrado por seguridad si el calendario falla.
    3. Timestamp de predicciones como fallback solo para simulación/informes.
    4. Si no hay ninguna fuente: abierto por compatibilidad.
    """
    try:
        first_match_ts, source = get_laliga_calendar_first_match_ts()
        ok, hours = clausulazos_available(first_match_ts)
        return ok, hours, source
    except Exception as exc:
        calendar_error = f"{type(exc).__name__}: {exc}"

    if fail_closed:
        return False, 0.0, f"unverified_fail_closed: {calendar_error}"

    if int(fallback_first_match_ts or 0) > 0:
        ok, hours = clausulazos_available(int(fallback_first_match_ts))
        return ok, hours, f"predictions_fallback_after_calendar_error: {calendar_error}"

    return True, 999.0, f"unverified_open_compat: {calendar_error}"


# ─── 2. Generar predicciones xP ──────────────────────────────
def get_predictions(model_type: str = MODEL_TYPE) -> tuple[pd.DataFrame, int]:
    """
    Genera predicciones xP para la próxima jornada.
    Returns:
        (pred_df, first_match_timestamp)
    """
    mt = (model_type or MODEL_TYPE).strip().lower()
    if mt != MODEL_TYPE:
        raise ValueError(f"Modelo no soportado: {model_type}. Solo se permite '{MODEL_TYPE}'.")
    return predict(model_type=MODEL_TYPE)


# ─── 3. Analizar la plantilla del usuario ─────────────────────
def analyze_my_team(
    snapshot: dict,
    pred_df: pd.DataFrame,
) -> dict:
    """
    Cruza la plantilla del usuario con las predicciones xP.
    Devuelve un análisis completo.
    """
    mi_equipo = snapshot["mi_equipo"]
    plantilla = mi_equipo["plantilla"]
    saldo = mi_equipo["saldo_disponible"]

    # Crear índice de predicciones por player_id (normalizar a int)
    xp_index = {}
    if not pred_df.empty:
        pred_copy = pred_df.copy()
        pred_copy["player_id"] = pred_copy["player_id"].astype(int)
        xp_index = pred_copy.set_index("player_id").to_dict("index")

    # Estados que implican que el jugador NO va a jugar
    ESTADOS_NO_DISPONIBLE = {"injured", "doubtful", "suspended", "doubt"}

    jugadores_con_xp = []
    for p in plantilla:
        pid = p["player_id"]
        xp_data = xp_index.get(pid, {})
        estado = (p.get("estado") or "ok").lower()

        # Si está lesionado/sancionado, su xP real es 0 — no va a jugar
        no_disponible = estado in ESTADOS_NO_DISPONIBLE
        xp_raw = xp_data.get("xP", 0)

        jugador = {
            **p,
            "xP": 0 if no_disponible else xp_raw,
            "xP_si_jugara": xp_raw,  # Para referencia
            "no_disponible": no_disponible,
            "rival": xp_data.get("rival", "?"),
            "es_local": xp_data.get("es_local", 0),
            "media_pts_3j": xp_data.get("media_pts_3j", 0),
            "media_pts_5j": xp_data.get("media_pts_5j", 0),
            "media_min_5j": xp_data.get("media_min_5j", 0),
            "titular_pct_5j": xp_data.get("titular_pct_5j", 0),
            "tendencia_pts": xp_data.get("tendencia_pts", 0),
            "prob_victoria": xp_data.get("prob_victoria", 0),
            "fdr_rival": xp_data.get("fdr_rival", 10),
        }
        jugadores_con_xp.append(jugador)

    # Ordenar por xP
    jugadores_con_xp.sort(key=lambda x: x["xP"], reverse=True)

    # Calcular once ideal evaluando TODAS las formaciones y eligiendo la de mayor xP
    once_ideal = _calcular_once(jugadores_con_xp)
    xp_total_once = sum(j["xP"] for j in once_ideal)
    xp_total_plantilla = sum(j["xP"] for j in jugadores_con_xp)
    formacion_once = once_ideal[0].pop("_formacion", "4-3-3") if once_ideal else "4-3-3"
    xp_por_formacion = once_ideal[0].pop("_xp_por_formacion", {}) if once_ideal else {}

    # Detectar jugadores con problemas
    problemas = []
    for j in jugadores_con_xp:
        if j.get("estado") and j["estado"] not in ("ok", "?"):
            problemas.append({"jugador": j["nombre"], "problema": j["estado"]})
        elif j.get("media_min_5j", 0) < 30:
            problemas.append({"jugador": j["nombre"], "problema": "No juega (media <30 min)"})
        elif j.get("titular_pct_5j", 0) < 0.2:
            problemas.append({"jugador": j["nombre"], "problema": "Suplente habitual"})

    return {
        "saldo": saldo,
        "manager_name": mi_equipo.get("manager_name", ""),
        "posicion_liga": mi_equipo.get("posicion", 0),
        "puntos_liga": mi_equipo.get("puntos", 0),
        "jugadores": jugadores_con_xp,
        "once_ideal": once_ideal,
        "formacion_once": formacion_once,
        "xp_por_formacion": xp_por_formacion,
        "xp_total_once": round(xp_total_once, 1),
        "xp_total_plantilla": round(xp_total_plantilla, 1),
        "problemas": problemas,
    }


# Todas las formaciones permitidas en LaLiga Fantasy (1 POR + 10 de campo).
# (DEF, MED, DEL) con DEF 2-5, MED 2-5, DEL 1-3, DEF+MED+DEL=10.
# Generamos todas las combinaciones válidas para no dejar ninguna fuera.
def _generar_formaciones_validas() -> list[tuple[int, int, int]]:
    out = []
    for n_def in range(2, 6):   # 2 a 5 defensas
        for n_med in range(2, 6):  # 2 a 5 medios
            n_del = 10 - n_def - n_med
            if 1 <= n_del <= 3:  # 1 a 3 delanteros
                out.append((n_def, n_med, n_del))
    return out


FORMACIONES_VALIDAS = _generar_formaciones_validas()


def _once_para_formacion(
    disponibles_por_pos: dict[str, list[dict]],
    n_def: int,
    n_med: int,
    n_del: int,
) -> tuple[list[dict], float]:
    """
    Devuelve el mejor once para una formación (n_def, n_med, n_del):
    los N jugadores con más xP en cada posición. xP total = suma de sus xP.
    Si no hay suficientes jugadores, devuelve ([], 0).
    """
    por = disponibles_por_pos.get("POR", [])
    defs = disponibles_por_pos.get("DEF", [])
    meds = disponibles_por_pos.get("MED", [])
    dels = disponibles_por_pos.get("DEL", [])

    if len(por) < 1 or len(defs) < n_def or len(meds) < n_med or len(dels) < n_del:
        return [], 0.0

    por_sorted = sorted(por, key=lambda x: x["xP"], reverse=True)
    def_sorted = sorted(defs, key=lambda x: x["xP"], reverse=True)
    med_sorted = sorted(meds, key=lambda x: x["xP"], reverse=True)
    del_sorted = sorted(dels, key=lambda x: x["xP"], reverse=True)

    once = (
        [por_sorted[0]]
        + def_sorted[:n_def]
        + med_sorted[:n_med]
        + del_sorted[:n_del]
    )
    xp_total = sum(j["xP"] for j in once)
    return once, round(xp_total, 2)


def _calcular_once(jugadores: list[dict]) -> list[dict]:
    """
    Evalúa TODAS las formaciones permitidas (todas las combinaciones
    DEF+MED+DEL que sumen 10 con límites por posición) y elige el once
    con MAYOR xP total. Sin trampas: para cada formación se toman
    los mejores N por posición (por xP) y se suma; se elige la formación
    cuya suma es máxima.

    Excluye lesionados/sancionados (no_disponible).

    Returns:
        Lista de 11 jugadores (mejor once). En el primer jugador se añade
        "_formacion" (ej. "3-5-2") y "_xp_por_formacion" (dict formacion -> xP)
        para el informe.
    """
    disponibles = [j for j in jugadores if not j.get("no_disponible", False)]

    por_pos: dict[str, list[dict]] = {}
    for j in disponibles:
        pos = j.get("posicion", "?")
        if pos not in por_pos:
            por_pos[pos] = []
        por_pos[pos].append(j)

    # Evaluar cada formación y guardar (xp, once, nombre, n_del, n_med) para desempates
    resultados: list[tuple[float, list[dict], str, int, int]] = []

    for (n_def, n_med, n_del) in FORMACIONES_VALIDAS:
        once, xp = _once_para_formacion(por_pos, n_def, n_med, n_del)
        if len(once) == 11:
            resultados.append((xp, once, f"{n_def}-{n_med}-{n_del}", n_del, n_med))

    # Ordenar: 1) mayor xP, 2) empate → más delanteros, 3) empate → más medios
    resultados.sort(key=lambda r: (r[0], r[3], r[4]), reverse=True)

    if not resultados:
        # Fallback si no hay ninguna formación válida
        por_pos_all = {"POR": 1, "DEF": 4, "MED": 3, "DEL": 3}
        once = []
        usados = set()
        for pos, n in por_pos_all.items():
            cand = sorted(por_pos.get(pos, []), key=lambda x: x["xP"], reverse=True)
            once.extend(cand[:n])
            usados.update(j["player_id"] for j in cand[:n])
        restantes = [j for j in disponibles if j["player_id"] not in usados]
        restantes.sort(key=lambda x: x["xP"], reverse=True)
        for j in restantes:
            if len(once) >= 11:
                break
            once.append(j)
        mejor_once = once
        mejor_formacion = "4-3-3 (fallback)"
        xp_por_formacion = {}
    else:
        mejor_xp, mejor_once, mejor_formacion, _, _ = resultados[0]
        # Guardar xP de todas las formaciones para mostrarlas en el informe
        xp_por_formacion = {nombre: xp for xp, _, nombre, _, _ in resultados}

    if mejor_once:
        mejor_once[0]["_formacion"] = mejor_formacion
        mejor_once[0]["_xp_por_formacion"] = xp_por_formacion

    return mejor_once


def can_field_valid_lineup(jugadores: list[dict]) -> bool:
    """
    Indica si la plantilla puede formar un once válido de LaLiga Fantasy.

    A diferencia de `_calcular_once`, aquí no hay fallback: debe existir una
    formación reglamentaria con 1 POR, 2-5 DEF, 2-5 MED y 1-3 DEL.
    """
    disponibles = [j for j in jugadores if not j.get("no_disponible", False)]
    counts = {
        "POR": 0,
        "DEF": 0,
        "MED": 0,
        "DEL": 0,
    }
    for j in disponibles:
        pos = str(j.get("posicion", "")).strip().upper()
        if pos in counts:
            counts[pos] += 1

    if counts["POR"] < 1:
        return False
    return any(
        counts["DEF"] >= n_def
        and counts["MED"] >= n_med
        and counts["DEL"] >= n_del
        for n_def, n_med, n_del in FORMACIONES_VALIDAS
    )


def sale_keeps_lineup_viable(jugadores: list[dict], player_id: int) -> bool:
    """
    Valida que vender un jugador no deje la plantilla peor para alinear.

    Si ahora mismo hay un once válido, exige conservarlo. Si la plantilla ya no
    puede alinear once completo, al menos no permite empeorar el número de
    jugadores alineables.
    """
    pid = int(player_id)
    remaining = [dict(j) for j in jugadores if int(j.get("player_id", 0) or 0) != pid]
    if can_field_valid_lineup(jugadores):
        return can_field_valid_lineup(remaining)

    sold = next(
        (j for j in jugadores if int(j.get("player_id", 0) or 0) == pid),
        {},
    )
    sold_pos = str(sold.get("posicion", "")).strip().upper()
    minimum_by_pos = {"POR": 1, "DEF": 2, "MED": 2, "DEL": 1}
    if sold_pos in minimum_by_pos:
        available_same_pos = [
            j for j in jugadores
            if str(j.get("posicion", "")).strip().upper() == sold_pos
            and not j.get("no_disponible", False)
        ]
        if len(available_same_pos) <= minimum_by_pos[sold_pos]:
            return False

    current_once = _calcular_once([dict(j) for j in jugadores])
    remaining_once = _calcular_once(remaining)
    return len(remaining_once) >= len(current_once)


# ─── 4. Obtener estados de jugadores ──────────────────────────
ESTADOS_NO_DISPONIBLE = {"injured", "doubtful", "suspended"}


def _build_player_status_index() -> dict[int, str]:
    """
    Construye un índice de estados combinando:
    1. API pública de LaLiga Fantasy (playerStatus)
    2. Scraper de FútbolFantasy (lesionados + sancionados)

    La fuente 2 suele estar más actualizada para lesiones recientes.
    Devuelve {player_id: status}.
    """
    import requests

    index = {}

    # --- Fuente 1: API pública ---
    logger.info("Consultando estados de jugadores (API pública)...")
    try:
        r = requests.get(
            "https://api-fantasy.llt-services.com/api/v3/players",
            headers={"User-Agent": "okhttp/4.12.0", "X-App": "Fantasy", "X-Lang": "es"},
            timeout=15,
        )
        r.raise_for_status()
        players = r.json()
        name_to_id = {}
        for p in players:
            pid = int(p.get("id", 0)) if p.get("id") else 0
            status = (p.get("playerStatus") or "ok").lower()
            index[pid] = status
            # Guardar mapeo nombre → id para cruzar con scraper
            name = (p.get("nickname") or p.get("name") or "").strip().lower()
            if name and pid:
                name_to_id[name] = pid
    except Exception as e:
        logger.warning("No se pudieron obtener estados de API: %s", e)
        name_to_id = {}

    # --- Fuente 2: Scraper FútbolFantasy ---
    scrapes_dir = Path(__file__).parent.parent / "scrapes"
    if scrapes_dir.exists():
        try:
            scrape_files = sorted(scrapes_dir.iterdir(), reverse=True)
            if scrape_files:
                with open(scrape_files[0]) as f:
                    scrape_data = json.load(f)

                ff = scrape_data.get("futbolfantasy", {})
                extra_injured = 0

                # Lesionados
                for les in ff.get("lesionados", []):
                    nombre = les.get("nombre", "").strip().lower()
                    prob = les.get("probabilidad_titular", 100)
                    estado = les.get("estado", "")

                    if prob == 0 or estado == "baja":
                        pid = name_to_id.get(nombre)
                        if pid and index.get(pid) == "ok":
                            index[pid] = "injured"
                            extra_injured += 1

                # Sancionados
                for san in ff.get("sancionados", []):
                    # Nombre puede ser "José Manuel Copete" → buscar parcial
                    nombre_full = san.get("nombre", "").strip().lower()
                    # Intentar match directo primero
                    pid = name_to_id.get(nombre_full)
                    if not pid:
                        # Match parcial: buscar por apellido
                        parts = nombre_full.split()
                        for part in reversed(parts):
                            if len(part) > 3:
                                for nk, nid in name_to_id.items():
                                    if part in nk:
                                        pid = nid
                                        break
                            if pid:
                                break
                    if pid and index.get(pid) not in ("suspended",):
                        index[pid] = "suspended"
                        extra_injured += 1

                if extra_injured:
                    logger.info(
                        "  FútbolFantasy: %d bajas adicionales detectadas",
                        extra_injured,
                    )
        except Exception as e:
            logger.warning("No se pudieron leer datos de scraper: %s", e)

    no_disp = sum(1 for s in index.values() if s in ESTADOS_NO_DISPONIBLE)
    logger.info("  Estados totales: %d jugadores (%d no disponibles)", len(index), no_disp)

    return index


# ─── 5. Analizar jugadores alcanzables ────────────────────────
def analyze_available_players(
    snapshot: dict,
    pred_df: pd.DataFrame,
    saldo: int,
    player_status: dict[int, str] | None = None,
) -> dict:
    """
    Analiza jugadores ALCANZABLES (los que se pueden adquirir antes de la jornada):
    1. Mercado de pujas de la liga (tipo "libre"): subastas donde pujas y el máximo se lo lleva.
       No se incluyen jugadores publicados por otros managers (tipo "manager"; requieren aceptación).
    2. Clausulazos de rivales (si la ventana está abierta): cláusula desbloqueada y asequible.

    Filtra lesionados, sancionados, dudosos; y en mercado solo entradas con subasta aún abierta.
    """
    # Índice de estados: combina snapshot + API pública
    if player_status is None:
        player_status = _build_player_status_index()

    xp_index = {}
    if not pred_df.empty:
        pred_copy = pred_df.copy()
        pred_copy["player_id"] = pred_copy["player_id"].astype(int)
        xp_index = pred_copy.set_index("player_id").to_dict("index")

    def _is_available(pid: int, estado_snapshot: str = "ok") -> bool:
        """Verifica si un jugador está disponible (no lesionado/sancionado)."""
        # Priorizar estado de la API pública (más actualizado)
        api_status = player_status.get(pid, "ok")
        if api_status in ESTADOS_NO_DISPONIBLE:
            return False
        # Fallback al estado del snapshot
        if estado_snapshot.lower() in ESTADOS_NO_DISPONIBLE:
            return False
        return True

    # --- Mercado ALCANZABLE: solo "libre" (mercado de pujas de la liga) ---
    # Los jugadores publicados por otros managers (tipo "manager") NO son alcanzables:
    # requieren que el otro usuario acepte el trato. Solo son alcanzables los que
    # se pueden clausular o los del mercado de pujas (tipo "libre") antes del cierre.
    mercado_raw = snapshot.get("mercado", [])
    now = datetime.now(timezone.utc)
    mercado_con_xp = []
    for p in mercado_raw:
        if p.get("tipo") != "libre":
            continue  # Excluir tipo "manager" (venta por otro usuario)

        # Opcional: solo si la subasta sigue abierta (expiracion en el futuro)
        expiracion = p.get("expiracion")
        if expiracion:
            try:
                exp_dt = datetime.fromisoformat(expiracion.replace("Z", "+00:00"))
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                if exp_dt <= now:
                    continue  # Puja ya cerrada
            except (ValueError, TypeError):
                pass

        pid = p["player_id"]
        if not _is_available(pid):
            continue

        xp_data = xp_index.get(pid, {})
        precio = p.get("precio_venta", 0) or 0

        entry = {
            **p,
            "xP": xp_data.get("xP", 0),
            "rival_next": xp_data.get("rival", "?"),
            "es_local": xp_data.get("es_local", 0),
            "media_pts_5j": xp_data.get("media_pts_5j", 0),
            "titular_pct_5j": xp_data.get("titular_pct_5j", 0),
            "asequible": precio <= saldo,
        }
        mercado_con_xp.append(entry)

    mercado_con_xp.sort(key=lambda x: x["xP"], reverse=True)

    # --- Clausulazos de rivales (solo si la ventana está abierta; ver main) ---
    clausulazos = []
    for rival in snapshot.get("rivales", []):
        rival_name = rival.get("manager_name", "?")
        for p in rival.get("plantilla", []):
            pid = p["player_id"]
            clausula = p.get("clausula", 0)
            bloqueada_hasta = p.get("clausula_bloqueada_hasta")
            estado = p.get("estado", "ok")

            # Verificar si la cláusula está desbloqueada
            bloqueada = False
            if bloqueada_hasta:
                try:
                    block_dt = datetime.fromisoformat(
                        bloqueada_hasta.replace("Z", "+00:00")
                    )
                    if block_dt > now:
                        bloqueada = True
                except (ValueError, TypeError):
                    pass

            if bloqueada:
                continue  # Skip cláusula bloqueada

            # Filtrar no disponibles (lesionados, sancionados, dudosos)
            if not _is_available(pid, estado):
                continue

            xp_data = xp_index.get(pid, {})

            entry = {
                **p,
                "team_id": rival.get("team_id", ""),  # Para buy_player_clausulazo
                "propietario": rival_name,
                "xP": xp_data.get("xP", 0),
                "rival_next": xp_data.get("rival", "?"),
                "es_local": xp_data.get("es_local", 0),
                "media_pts_5j": xp_data.get("media_pts_5j", 0),
                "titular_pct_5j": xp_data.get("titular_pct_5j", 0),
                "asequible": clausula <= saldo,
            }
            clausulazos.append(entry)

    clausulazos.sort(key=lambda x: x["xP"], reverse=True)

    return {
        "mercado": mercado_con_xp,
        "clausulazos": clausulazos,
    }


# ─── 5. Simular plan de movimientos ───────────────────────────
def simulate_transfer_plan(
    team_analysis: dict,
    available: dict,
    snapshot: dict,
    allow_clausulazos: bool = True,
) -> dict:
    """
    Simula un plan de movimientos con ventas y compras INDEPENDIENTES.

    Importante:
    - Vender NO obliga a comprar.
    - Comprar NO obliga a vender.
    - El saldo para compras es el saldo actual (ventas fase 1 no abonan
      dinero hasta fase 2 de aceptación de oferta).
    """
    VENTA_PCT = 0.90
    MAX_SOBREPRECIO = 1.15
    UMBRAL_GANANCIA_COMPRA = 0.5
    MAX_VENTAS = 4
    MAX_COMPRAS = 4

    saldo = team_analysis["saldo"]
    plantilla = [dict(j) for j in team_analysis["jugadores"]]
    movimientos = []
    alertas: list[str] = []
    priority_needs: list[dict] = []
    non_executable_recommendations: list[dict] = []
    jugadores_vendidos_ids: set[int] = set()
    jugadores_comprados_ids: set[int] = set()
    once_actual_ids = {j["player_id"] for j in team_analysis.get("once_ideal", [])}

    def _sale_income(jugador: dict) -> int:
        return int((jugador.get("valor_mercado", 0) or 0) * VENTA_PCT)

    def _sale_publication_price(jugador: dict) -> int:
        valor_mercado = int(jugador.get("valor_mercado", 0) or 0)
        return max(valor_mercado, int(valor_mercado * VENTA_PCT))

    def _es_buen_valor(compra: dict) -> bool:
        vm = compra.get("valor_mercado") or 0
        coste = compra.get("coste", 0)
        if vm <= 0:
            return True
        if coste <= vm * MAX_SOBREPRECIO:
            return True

        # Excepción controlada en subastas competidas:
        # permitimos sobreprecio moderado cuando hay competencia y buen xP.
        pujas = int(compra.get("pujas_detectadas", 0) or 0)
        incremento = int(compra.get("incremento_competitivo", 0) or 0)
        xp = float(compra.get("xP", 0) or 0.0)
        return (
            pujas > 0
            and incremento >= BID_COMPETITION_MIN_RAISE_EUR
            and incremento <= BID_COMPETITION_MAX_RAISE_EUR
            and xp >= 4.5
            and coste <= vm * 1.45
        )

    def _crear_fichaje(compra: dict) -> dict:
        return {
            "player_id": compra["player_id"],
            "nombre": compra.get("nombre", "?"),
            "posicion": compra.get("posicion", "?"),
            "equipo_real": compra.get("equipo_real", "?"),
            "estado": "ok",
            "valor_mercado": compra.get("valor_mercado", 0),
            "xP": compra.get("xP", 0),
            "xP_si_jugara": compra.get("xP", 0),
            "no_disponible": False,
            "rival": compra.get("rival_next", "?"),
            "es_local": compra.get("es_local", 0),
            "media_pts_5j": compra.get("media_pts_5j", 0),
            "media_min_5j": 70,
            "titular_pct_5j": 0.8,
            "tendencia_pts": 0,
            "media_pts_3j": compra.get("media_pts_5j", 0),
            "nuevo_fichaje": True,
        }

    def _position_key(raw: object) -> str:
        txt = str(raw or "").strip().upper()
        aliases = {
            "PT": "POR",
            "POR": "POR",
            "PORTERO": "POR",
            "GK": "POR",
            "DF": "DEF",
            "DEF": "DEF",
            "DEFENSA": "DEF",
            "MC": "MED",
            "MED": "MED",
            "CENTROCAMPISTA": "MED",
            "DL": "DEL",
            "DC": "DEL",
            "DEL": "DEL",
            "DELANTERO": "DEL",
        }
        return aliases.get(txt, txt)

    def _player_available(jugador: dict) -> bool:
        estado = str(jugador.get("estado", "") or "").strip().lower()
        if bool(jugador.get("no_disponible")):
            return False
        if estado and estado not in {"ok", "doubt", "warn", "apercibido"}:
            return False
        return True

    def _coverage_needs(jugadores: list[dict]) -> list[dict]:
        minimums = {"POR": 1, "DEF": 3, "MED": 3, "DEL": 1}
        counts = {pos: 0 for pos in minimums}
        for jugador in jugadores:
            if not _player_available(jugador):
                continue
            pos = _position_key(jugador.get("posicion"))
            if pos in counts:
                counts[pos] += 1
        needs = []
        for pos, minimum in minimums.items():
            missing = max(0, minimum - counts.get(pos, 0))
            if missing:
                needs.append(
                    {
                        "position": pos,
                        "missing": missing,
                        "available": counts.get(pos, 0),
                        "required_min": minimum,
                    }
                )
        return needs

    def _xp_once(jugadores: list[dict]) -> float:
        once = _calcular_once(jugadores)
        return float(sum(j["xP"] for j in once))

    def _lineup_sale_ok(jugadores: list[dict], player_id: int) -> bool:
        return sale_keeps_lineup_viable([dict(j) for j in jugadores], player_id)

    def _debt_sale_candidate(jugador: dict, jugadores_base: list[dict]) -> dict | None:
        pid = int(jugador.get("player_id", 0) or 0)
        if not pid:
            return None
        if not jugador.get("player_team_id") and not jugador.get("en_venta"):
            return None
        if not _lineup_sale_ok(jugadores_base, pid):
            return None

        before_xp = _xp_once([dict(j) for j in jugadores_base])
        remaining = [dict(j) for j in jugadores_base if int(j.get("player_id", 0) or 0) != pid]
        after_xp = _xp_once(remaining)
        marginal_loss = max(0.0, round(before_xp - after_xp, 2))
        player_xp = float(jugador.get("xP", 0) or 0.0)
        market_value = int(jugador.get("valor_mercado", 0) or 0)
        clause_value = int(jugador.get("clausula", 0) or 0)
        clause_ratio = (clause_value / market_value) if market_value > 0 and clause_value > 0 else 0.0
        income = _sale_income(jugador)
        if income <= 0:
            return None

        if jugador.get("no_disponible"):
            impact_bucket = 0
            reason = f"No disponible ({jugador.get('estado', 'baja')})"
        elif marginal_loss <= 0.05:
            impact_bucket = 1
            reason = "Impacto nulo en el once"
        elif pid not in once_actual_ids and player_xp < 2:
            impact_bucket = 2
            reason = "Fuera del once y xP bajo"
        elif player_xp < 1.5:
            impact_bucket = 3
            reason = "xP muy bajo"
        else:
            impact_bucket = 4
            reason = "Venta necesaria para saldar deuda"

        # Una cláusula alta indica que el jugador estaba protegido; no lo blinda
        # si tiene impacto bajo, pero lo relega frente a descartes equivalentes.
        clause_penalty = min(3.0, clause_ratio)
        coverage_bonus = min(3.0, income / max(1, abs(int(saldo or 0))))
        sale_priority = (
            impact_bucket,
            round(marginal_loss, 2),
            round(player_xp, 2),
            round(clause_penalty, 2),
            -round(coverage_bonus, 2),
            -income,
        )
        return {
            **jugador,
            "motivo_venta": reason,
            "impacto_xp_once": marginal_loss,
            "ingresos_estimados": income,
            "ratio_clausula_valor": round(clause_ratio, 2) if clause_ratio else None,
            "prioridad_deuda": sale_priority,
        }

    # ── Modo deuda: saldo negativo, prioridad absoluta a volver a saldo >= 0 ──
    if saldo < 0:
        deuda_objetivo = abs(int(saldo))
        saldo_proyectado = int(saldo)
        deuda_cubierta = 0
        debt_movements: list[dict] = []
        remaining_squad = [dict(j) for j in plantilla]

        while saldo_proyectado < 0:
            candidates = []
            for jugador in remaining_squad:
                pid = int(jugador.get("player_id", 0) or 0)
                if not pid or pid in jugadores_vendidos_ids:
                    continue
                candidate = _debt_sale_candidate(jugador, remaining_squad)
                if candidate:
                    candidates.append(candidate)

            if not candidates:
                break

            candidates.sort(key=lambda x: x["prioridad_deuda"])
            venta = candidates[0]
            pid = int(venta["player_id"])
            jugadores_vendidos_ids.add(pid)
            income = int(venta["ingresos_estimados"])
            saldo_antes = saldo_proyectado
            saldo_proyectado += income
            deuda_cubierta += income
            remaining_squad = [
                j for j in remaining_squad
                if int(j.get("player_id", 0) or 0) != pid
            ]

            debt_movements.append({
                "paso": len(debt_movements) + 1,
                "venta": {
                    "player_id": venta["player_id"],
                    "player_team_id": venta.get("player_team_id"),
                    "nombre": venta["nombre"],
                    "posicion": venta["posicion"],
                    "xP": venta["xP"],
                    "impacto_xp_once": venta["impacto_xp_once"],
                    "valor_mercado": venta.get("valor_mercado", 0),
                    "clausula": venta.get("clausula", 0),
                    "ratio_clausula_valor": venta.get("ratio_clausula_valor"),
                    "ingresos": income,
                    "motivo": venta.get("motivo_venta", "?"),
                    "precio_publicacion": _sale_publication_price(venta),
                    "ya_en_venta": bool(venta.get("en_venta")),
                },
                "compra": None,
                "saldo_antes": saldo_antes,
                "saldo_despues": saldo_proyectado,
                "ganancia_xp": -float(venta["impacto_xp_once"]),
            })

        once_post = _calcular_once(remaining_squad)
        xp_total_post = sum(j["xP"] for j in once_post)
        formacion_post = once_post[0].pop("_formacion", "4-3-3") if once_post else "4-3-3"
        if once_post:
            once_post[0].pop("_xp_por_formacion", None)

        return {
            "modo_deuda": True,
            "deuda_objetivo": deuda_objetivo,
            "deuda_cubierta_estimada": deuda_cubierta,
            "deuda_pendiente_estimada": max(0, -saldo_proyectado),
            "alertas": (
                []
                if saldo_proyectado >= 0
                else [
                    "No hay ventas ejecutables suficientes que cubran la deuda sin comprometer la alineación."
                ]
            ),
            "movimientos": debt_movements,
            "plantilla_final": remaining_squad,
            "once_post": once_post,
            "formacion_post": formacion_post,
            "xp_total_post": round(xp_total_post, 1),
            "saldo_final": saldo_proyectado,
            "jugadores_vendidos": len(jugadores_vendidos_ids),
            "jugadores_comprados": 0,
        }

    # ── Pool de compras ───────────────────────────────────────
    pool_compras = []
    for p in available["mercado"]:
        precio_venta = p.get("precio_venta", 0) or 0
        valor_mercado = p.get("valor_mercado", 0) or 0
        pujas_detectadas = int(p.get("pujas", 0) or 0)
        coste_base = max(precio_venta, valor_mercado)
        xp_compra = float(p.get("xP", 0) or 0.0)
        coste_competitivo = compute_competitive_bid_amount(
            base_amount=coste_base,
            market_value=valor_mercado,
            external_bids=pujas_detectadas,
            expected_points=xp_compra,
        )
        pool_compras.append({
            **p,
            "tipo_op": "mercado",
            # Si hay pujas de terceros, subimos la oferta para competir.
            "coste_base": coste_base,
            "pujas_detectadas": pujas_detectadas,
            "incremento_competitivo": max(0, int(coste_competitivo - coste_base)),
            "coste": coste_competitivo,
        })
    if allow_clausulazos:
        for p in available["clausulazos"]:
            pool_compras.append({
                **p,
                "tipo_op": "clausulazo",
                "coste": p.get("clausula", 0) or 0,
            })
    pool_compras.sort(key=lambda x: x.get("xP", 0), reverse=True)

    # ── 0) Cobertura crítica de alineación ────────────────────
    # Antes de optimizar xP o valor, el motor debe asegurar que existe una
    # plantilla capaz de formar once válido. Esto evita gastar ciclos en mejoras
    # marginales cuando falta, por ejemplo, un portero disponible.
    for need in _coverage_needs(plantilla):
        pos = need["position"]
        priority_needs.append(
            {
                **need,
                "reason": f"Faltan {need['missing']} jugador(es) disponible(s) en {pos} para una alineación válida.",
            }
        )
        candidates = [
            c
            for c in pool_compras
            if _position_key(c.get("posicion")) == pos
            and c.get("tipo_op") == "mercado"
            and int(c.get("player_id", 0) or 0) not in jugadores_comprados_ids
        ]
        candidates.sort(
            key=lambda c: (
                int(c.get("coste", 0) or 0) > int(saldo or 0),
                -float(c.get("xP", 0) or 0.0),
                int(c.get("coste", 0) or 0),
            )
        )
        affordable = [c for c in candidates if int(c.get("coste", 0) or 0) <= int(saldo or 0)]
        if not affordable:
            cheapest = min((int(c.get("coste", 0) or 0) for c in candidates), default=0)
            non_executable_recommendations.append(
                {
                    "type": "coverage_need",
                    "position": pos,
                    "reason": (
                        f"Necesidad de {pos}, pero no hay candidato de mercado financiable "
                        "con el saldo actual."
                    ),
                    "min_market_cost": cheapest or None,
                    "saldo_actual": int(saldo or 0),
                }
            )
            alertas.append(
                f"Necesidad de {pos}: no hay compra financiable con saldo actual; revisar ventas/ofertas manuales."
            )
            continue

        for compra in affordable[: int(need["missing"])]:
            if int(compra.get("player_id", 0) or 0) in jugadores_comprados_ids:
                continue
            jugadores_comprados_ids.add(compra["player_id"])
            saldo_antes = saldo
            saldo -= int(compra["coste"])
            plantilla.append(_crear_fichaje(compra))
            movimientos.append(
                {
                    "paso": len(movimientos) + 1,
                    "venta": None,
                    "compra": {
                        "player_id": compra["player_id"],
                        "tipo": compra.get("tipo_op", "?"),
                        "market_item_id": compra.get("market_item_id"),
                        "player_team_id": compra.get("player_team_id"),
                        "nombre": compra["nombre"],
                        "posicion": compra.get("posicion", "?"),
                        "equipo_real": compra.get("equipo_real", "?"),
                        "xP": compra.get("xP", 0),
                        "coste": compra["coste"],
                        "coste_base": compra.get("coste_base", compra["coste"]),
                        "pujas_detectadas": compra.get("pujas_detectadas", 0),
                        "incremento_competitivo": compra.get("incremento_competitivo", 0),
                        "propietario": compra.get("propietario", compra.get("vendedor", "")),
                        "motivo": f"Cubrir necesidad de alineación en {pos}",
                    },
                    "saldo_antes": saldo_antes,
                    "saldo_despues": saldo,
                    "ganancia_xp": 0.0,
                    "priority": "lineup_coverage",
                }
            )

    # ── 1) Ventas INDEPENDIENTES ──────────────────────────────
    candidatos_venta = []
    for j in plantilla:
        pid = j["player_id"]
        motivo = None
        prioridad = 99

        if j.get("no_disponible"):
            motivo = j.get("estado", "injured")
            prioridad = 0
        elif pid not in once_actual_ids and j.get("media_min_5j", 0) < 30 and j["xP"] < 2:
            motivo = "No juega"
            prioridad = 1
        elif pid not in once_actual_ids and j.get("titular_pct_5j", 0) < 0.15 and j["xP"] < 2:
            motivo = "Suplente habitual"
            prioridad = 2
        elif pid not in once_actual_ids and j["xP"] < 1.5:
            motivo = "Mejora por xP"
            prioridad = 3

        if motivo:
            if not _lineup_sale_ok(plantilla, pid):
                continue
            ingresos = _sale_income(j)
            candidatos_venta.append({
                **j,
                "motivo_venta": motivo,
                "ingresos_estimados": ingresos,
                "prioridad": prioridad,
            })

    candidatos_venta.sort(key=lambda x: (x["prioridad"], x["xP"]))

    for venta in candidatos_venta[:MAX_VENTAS]:
        pid = venta["player_id"]
        if pid in jugadores_vendidos_ids:
            continue
        if not _lineup_sale_ok(plantilla, pid):
            continue
        jugadores_vendidos_ids.add(pid)
        plantilla = [j for j in plantilla if j["player_id"] != pid]

        movimientos.append({
            "paso": len(movimientos) + 1,
            "venta": {
                "player_id": venta["player_id"],
                "player_team_id": venta.get("player_team_id"),
                "nombre": venta["nombre"],
                "posicion": venta["posicion"],
                "xP": venta["xP"],
                "valor_mercado": venta.get("valor_mercado", 0),
                "clausula": venta.get("clausula", 0),
                "ingresos": venta["ingresos_estimados"],
                "motivo": venta.get("motivo_venta", "?"),
                "precio_publicacion": _sale_publication_price(venta),
            },
            "compra": None,
            "saldo_antes": saldo,
            "saldo_despues": saldo,
            "ganancia_xp": 0.0,
        })

    # ── 2) Compras INDEPENDIENTES ─────────────────────────────
    # Si sigue faltando cobertura básica, no gastamos saldo en mejoras de otras
    # posiciones: se informa como recomendación no ejecutable.
    unresolved_coverage_needs = _coverage_needs(plantilla)
    for _ in range(0 if unresolved_coverage_needs else MAX_COMPRAS):
        xp_once_antes = _xp_once(plantilla)
        ids_plantilla = {j["player_id"] for j in plantilla}
        mejor_compra = None

        for compra in pool_compras:
            pid = compra["player_id"]
            coste = compra.get("coste", 0) or 0

            if pid in jugadores_comprados_ids or pid in ids_plantilla:
                continue
            if coste > saldo:
                continue
            if not _es_buen_valor(compra):
                continue

            candidato = _crear_fichaje(compra)
            xp_once_despues = _xp_once(plantilla + [candidato])
            ganancia = round(xp_once_despues - xp_once_antes, 1)
            if ganancia < UMBRAL_GANANCIA_COMPRA:
                continue

            if (
                not mejor_compra
                or ganancia > mejor_compra["ganancia_xp"]
                or (
                    ganancia == mejor_compra["ganancia_xp"]
                    and compra.get("xP", 0) > mejor_compra["compra"].get("xP", 0)
                )
            ):
                mejor_compra = {
                    "compra": compra,
                    "ganancia_xp": ganancia,
                }

        if not mejor_compra:
            break

        compra = mejor_compra["compra"]
        jugadores_comprados_ids.add(compra["player_id"])
        saldo_antes = saldo
        saldo -= compra["coste"]
        plantilla.append(_crear_fichaje(compra))

        movimientos.append({
            "paso": len(movimientos) + 1,
            "venta": None,
            "compra": {
                "player_id": compra["player_id"],
                "tipo": compra.get("tipo_op", "?"),
                "market_item_id": compra.get("market_item_id"),
                "player_team_id": compra.get("player_team_id"),
                "nombre": compra["nombre"],
                "posicion": compra.get("posicion", "?"),
                "equipo_real": compra.get("equipo_real", "?"),
                "xP": compra.get("xP", 0),
                "coste": compra["coste"],
                "coste_base": compra.get("coste_base", compra["coste"]),
                "pujas_detectadas": compra.get("pujas_detectadas", 0),
                "incremento_competitivo": compra.get("incremento_competitivo", 0),
                "propietario": compra.get("propietario", compra.get("vendedor", "")),
            },
            "saldo_antes": saldo_antes,
            "saldo_despues": saldo,
            "ganancia_xp": mejor_compra["ganancia_xp"],
        })

    # ── Once ideal post-movimientos ───────────────────────────
    once_post = _calcular_once(plantilla)
    xp_total_post = sum(j["xP"] for j in once_post)
    formacion_post = once_post[0].pop("_formacion", "4-3-3") if once_post else "4-3-3"
    if once_post:
        once_post[0].pop("_xp_por_formacion", None)

    return {
        "movimientos": movimientos,
        "priority_needs": priority_needs,
        "non_executable_recommendations": non_executable_recommendations,
        "alertas": alertas,
        "plantilla_final": plantilla,
        "once_post": once_post,
        "formacion_post": formacion_post,
        "xp_total_post": round(xp_total_post, 1),
        "saldo_final": saldo,
        "jugadores_vendidos": len(jugadores_vendidos_ids),
        "jugadores_comprados": len(jugadores_comprados_ids),
    }


# ─── 6. Generar informe Markdown ──────────────────────────────
def generate_report(
    team_analysis: dict,
    transfer_plan: dict,
    snapshot: dict,
    pred_df: pd.DataFrame,
    model_type: str,
    clausulazos_ok: bool = True,
    horas_al_partido: float = 999.0,
) -> str:
    """Genera un informe completo en formato Markdown."""

    jornada = int(pred_df["jornada"].iloc[0]) if not pred_df.empty else "?"
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    fm = _format_money

    L = []  # lines
    L.append(f"# Fantasy Advisor — Jornada {jornada}")
    L.append(f"")
    L.append(f"> Informe generado el {now} | Modelo: {model_type}")
    L.append(f"> Liga: **{snapshot.get('league_name', '?')}** | "
             f"Manager: **{team_analysis['manager_name']}**")
    L.append(f"")

    # ── Ventana de mercado ────────────────────────────────────
    if horas_al_partido < 48:
        if clausulazos_ok:
            L.append(f"> Primer partido en **{horas_al_partido:.0f}h**. "
                     f"Clausulazos y mercado de pujas disponibles.")
        else:
            L.append(f"> **Jornada inminente** (primer partido en "
                     f"**{horas_al_partido:.0f}h**). "
                     f"Clausulazos **BLOQUEADOS**. Solo mercado de pujas (hasta cierre).")
        L.append(f"")
    L.append(
        f"> **Jugadores alcanzables:** solo clausulazos (si abiertos) y mercado de pujas. "
        f"En pujas competidas se puede aplicar un incremento estratégico de **2M a 4M** "
        f"según xP y contexto del jugador."
    )
    L.append(f"")
    if transfer_plan.get("modo_deuda"):
        L.append(
            f"> **Modo deuda activado:** saldo negativo de "
            f"**{fm(team_analysis['saldo'])}**. Se bloquean compras y subidas de cláusula "
            f"hasta estimar saldo >= 0 con ventas que mantengan una alineación viable."
        )
        L.append(f"")

    movs = transfer_plan["movimientos"]
    xp_antes = team_analysis["xp_total_once"]
    xp_despues = transfer_plan["xp_total_post"]
    diff_xp = round(xp_despues - xp_antes, 1)
    diff_str = f"+{diff_xp}" if diff_xp > 0 else str(diff_xp)

    # ── Resumen ────────────────────────────────────────────────
    L.append(f"---")
    L.append(f"## Resumen")
    L.append(f"")
    L.append(f"| | Ahora | Despues de movimientos |")
    L.append(f"|---|------:|----------------------:|")
    L.append(f"| Posicion liga | {team_analysis['posicion_liga']} | — |")
    L.append(f"| Puntos totales | {team_analysis['puntos_liga']} | — |")
    L.append(f"| Saldo | {fm(team_analysis['saldo'])} | "
             f"**{fm(transfer_plan['saldo_final'])}** |")
    L.append(f"| xP del once | **{xp_antes} pts** | "
             f"**{xp_despues} pts** ({diff_str}) |")
    L.append(f"| Jugadores | {len(team_analysis['jugadores'])} | "
             f"{len(transfer_plan['plantilla_final'])} |")
    if movs:
        L.append(f"| Movimientos | — | "
                 f"{transfer_plan['jugadores_vendidos']} ventas, "
                 f"{transfer_plan['jugadores_comprados']} compras |")
    L.append(f"")

    # ── Alertas ────────────────────────────────────────────────
    if team_analysis["problemas"]:
        L.append(f"---")
        L.append(f"## Alertas")
        L.append(f"")
        for p in team_analysis["problemas"]:
            L.append(f"- **{p['jugador']}**: {p['problema']}")
        for alerta in transfer_plan.get("alertas", []) or []:
            L.append(f"- **Modo deuda**: {alerta}")
        L.append(f"")
    elif transfer_plan.get("alertas"):
        L.append(f"---")
        L.append(f"## Alertas")
        L.append(f"")
        for alerta in transfer_plan.get("alertas", []) or []:
            L.append(f"- **Modo deuda**: {alerta}")
        L.append(f"")

    # ── ONCE ACTUAL ───────────────────────────────────────────
    L.append(f"---")
    formacion_actual = team_analysis.get("formacion_once", "4-3-3")
    L.append(f"## Once actual (sin movimientos) — {formacion_actual} — {xp_antes} xP")
    L.append(f"")
    L.append(f"*Se evaluaron todas las combinaciones válidas (DEF 2-5, MED 2-5, DEL 1-3). Se elige la de **mayor xP**; en empate, la que tenga más delanteros y, si sigue el empate, la que tenga más medios.*")
    L.append(f"")

    xp_por_form = team_analysis.get("xp_por_formacion") or {}
    if xp_por_form:
        # Ordenar por xP descendente para que se vea claro cuál ganó
        items = sorted(xp_por_form.items(), key=lambda x: -x[1])
        L.append(f"**xP por formación (elegida = {formacion_actual}):**")
        L.append(f"")
        L.append(f"| Formación | xP total |")
        L.append(f"|-----------|----------|")
        for nombre, xp in items:
            marca = " **← elegida**" if nombre == formacion_actual else ""
            L.append(f"| {nombre} | {xp:.1f}{marca} |")
        L.append(f"")

    L.append(f"| Pos | Jugador | Equipo | Rival | Local | xP |")
    L.append(f"|-----|---------|--------|-------|-------|---:|")

    for j in team_analysis["once_ideal"]:
        local = "Si" if j.get("es_local") == 1 else "No"
        L.append(
            f"| {j['posicion']} | **{j['nombre']}** | "
            f"{j.get('equipo_real', '?')} | {j.get('rival', '?')} | "
            f"{local} | {j['xP']:.1f} |"
        )

    # Banquillo compacto
    once_ids = {j["player_id"] for j in team_analysis["once_ideal"]}
    banquillo = [j for j in team_analysis["jugadores"] if j["player_id"] not in once_ids]
    if banquillo:
        L.append(f"")
        L.append(f"**Banquillo:** ", )
        parts = []
        for j in banquillo:
            tag = ""
            if j.get("no_disponible"):
                tag = " BAJA"
            elif j.get("media_min_5j", 0) < 30:
                tag = " (no juega)"
            parts.append(f"{j['nombre']} {j['posicion']} {j['xP']:.1f}xP{tag}")
        L[-1] = f"**Banquillo:** {', '.join(parts)}"
    L.append(f"")

    # ── PLAN DE MOVIMIENTOS ───────────────────────────────────
    L.append(f"---")
    L.append(f"## Plan de movimientos")
    L.append(f"")

    if not movs:
        L.append(f"No hay movimientos recomendados. Tu equipo ya esta optimizado "
                 f"para la jornada {jornada}.")
    else:
        L.append(f"Ejecutar en este orden:")
        L.append(f"")

        for mov in movs:
            paso = mov["paso"]
            venta = mov.get("venta")
            compra = mov.get("compra")

            L.append(f"### Paso {paso}")
            L.append(f"")

            if venta:
                L.append(
                    f"1. **Vender** a {venta['nombre']} ({venta['posicion']}) "
                    f"— {venta['motivo']}"
                )
                L.append(
                    f"   - Valor de mercado: {fm(venta['valor_mercado'])} "
                    f"→ ingresos estimados: **{fm(venta['ingresos'])}**"
                )
                if venta.get("clausula"):
                    L.append(f"   - Cláusula actual: {fm(venta['clausula'])}")
                if venta.get("impacto_xp_once") is not None:
                    L.append(
                        f"   - Impacto estimado en el once: "
                        f"{venta['impacto_xp_once']:.1f} xP"
                    )
                else:
                    L.append(f"   - xP del jugador: {venta['xP']:.1f}")

            if compra:
                tipo_compra = compra["tipo"]
                if tipo_compra == "clausulazo":
                    L.append(
                        f"{'2' if venta else '1'}. **Clausulazo** a "
                        f"{compra['nombre']} ({compra['posicion']}, "
                        f"{compra['equipo_real']})"
                    )
                    L.append(
                        f"   - Clausula: **{fm(compra['coste'])}** "
                        f"(de {compra.get('propietario', '?')})"
                    )
                else:
                    L.append(
                        f"{'2' if venta else '1'}. **Fichar** a "
                        f"{compra['nombre']} ({compra['posicion']}, "
                        f"{compra['equipo_real']})"
                    )
                    L.append(
                        f"   - Precio: **{fm(compra['coste'])}** "
                        f"(mercado, {compra.get('propietario') or compra.get('tipo', '?')})"
                    )
                    pujas_detectadas = int(compra.get("pujas_detectadas", 0) or 0)
                    coste_base = int(compra.get("coste_base", compra["coste"]) or compra["coste"])
                    incremento = int(compra.get("incremento_competitivo", 0) or 0)
                    if pujas_detectadas > 0 and compra["coste"] > coste_base:
                        L.append(
                            f"   - Ajuste competitivo: {pujas_detectadas} puja(s) detectada(s). "
                            f"Base {fm(coste_base)} -> oferta {fm(compra['coste'])} "
                            f"(+{fm(incremento)})"
                        )

                L.append(f"   - xP que ganas: **{compra['xP']:.1f}**")
                L.append(f"   - Saldo: {fm(mov['saldo_antes'])} → **{fm(mov['saldo_despues'])}**")
                L.append(f"   - Ganancia neta: **{mov['ganancia_xp']:+.1f} xP**")
            elif venta:
                L.append(f"   - Este paso es **solo venta** (sin compra asociada).")
                L.append(f"   - Saldo: {fm(mov['saldo_antes'])} → **{fm(mov['saldo_despues'])}**")
                L.append(f"   - Ganancia neta: **{mov['ganancia_xp']:+.1f} xP**")
            else:
                L.append(f"   - Sin acciones ejecutables en este paso.")
            L.append(f"")

        # Resumen de movimientos
        total_ganancia = sum(m["ganancia_xp"] for m in movs)
        L.append(f"**Total: {len(movs)} movimientos, {total_ganancia:+.1f} xP en el once**")
        L.append(f"")

    # ── ONCE IDEAL POST-MOVIMIENTOS ───────────────────────────
    L.append(f"---")
    formacion_post = transfer_plan.get("formacion_post", "4-3-3")
    L.append(f"## Once ideal post-movimientos — {formacion_post} — {xp_despues} xP")
    L.append(f"")
    L.append(f"| Pos | Jugador | Equipo | Rival | Local | xP | Nuevo |")
    L.append(f"|-----|---------|--------|-------|-------|---:|:-----:|")

    for j in transfer_plan["once_post"]:
        local = "Si" if j.get("es_local") == 1 else "No"
        nuevo = "✓" if j.get("nuevo_fichaje") else ""
        L.append(
            f"| {j['posicion']} | **{j['nombre']}** | "
            f"{j.get('equipo_real', '?')} | {j.get('rival', '?')} | "
            f"{local} | {j['xP']:.1f} | {nuevo} |"
        )
    L.append(f"")

    # Comparativa
    L.append(f"### Comparativa")
    L.append(f"")
    L.append(f"| | Antes | Despues | Diferencia |")
    L.append(f"|---|------:|-------:|-----------:|")
    L.append(f"| xP once | {xp_antes} | {xp_despues} | **{diff_str}** |")
    L.append(f"| Saldo | {fm(team_analysis['saldo'])} | "
             f"{fm(transfer_plan['saldo_final'])} | "
             f"{fm(transfer_plan['saldo_final'] - team_analysis['saldo'])} |")
    L.append(f"")

    # ── Top xP global (referencia) ─────────────────────────────
    L.append(f"---")
    L.append(f"## Top 20 xP global (Jornada {jornada})")
    L.append(f"")
    L.append(f"| # | Jugador | Pos | Equipo | Rival | xP |")
    L.append(f"|---|---------|-----|--------|-------|----|")

    # IDs de plantilla final
    final_ids = {j["player_id"] for j in transfer_plan["plantilla_final"]}

    top_global = pred_df.head(20) if not pred_df.empty else pd.DataFrame()
    for i, (_, row) in enumerate(top_global.iterrows(), 1):
        row_pid = int(row["player_id"]) if row.get("player_id") else 0
        marker = " ⭐" if row_pid in final_ids else ""
        L.append(
            f"| {i} | {row['nombre']}{marker} | {row['posicion']} | "
            f"{row['equipo']} | {row.get('rival', '?')} | {row['xP']:.1f} |"
        )
    L.append(f"")
    L.append(f"> ⭐ = En tu plantilla")

    # ── Footer ─────────────────────────────────────────────────
    L.append(f"")
    L.append(f"---")
    L.append(f"")
    L.append(f"*Generado por Fantasy Advisor | Modelo {model_type} | "
             f"MAE ~2.6 pts*")

    return "\n".join(L)


def _format_money(amount: int) -> str:
    """Formatea un número como dinero: 3.809.659 → 3.8M"""
    sign = "-" if amount < 0 else ""
    n = abs(int(amount or 0))
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{sign}{n / 1_000:.0f}K"
    return f"{sign}{n}"


# ─── Main ─────────────────────────────────────────────────────
def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fantasy Advisor: recomendaciones para la próxima jornada"
    )
    parser.add_argument("--league", type=str, default="", help="League ID")
    parser.add_argument("--output", type=str, help="Ruta del informe .md")
    parser.add_argument(
        "--snapshot", type=str,
        help="Cargar snapshot desde JSON (en vez de generar uno nuevo)",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Fantasy Advisor — LaLiga Fantasy")
    print("=" * 60)

    # 1. Cargar/generar snapshot
    if args.snapshot:
        print(f"\n  Cargando snapshot desde {args.snapshot}...")
        snapshot = load_snapshot_from_file(args.snapshot)
    else:
        print("\n  Generando snapshot fresco de la liga...")
        snapshot = load_snapshot(args.league)

    mi_equipo = snapshot["mi_equipo"]
    print(f"  Manager: {mi_equipo.get('manager_name')}")
    print(f"  Liga: {snapshot.get('league_name')}")
    print(f"  Saldo: {_format_money(mi_equipo['saldo_disponible'])}")
    print(f"  Jugadores: {len(mi_equipo['plantilla'])}")

    # 2. Predicciones xP
    print(f"\n  Generando predicciones xP ({MODEL_TYPE})...")
    pred_df, first_match_ts = get_predictions(MODEL_TYPE)
    jornada = int(pred_df["jornada"].iloc[0]) if not pred_df.empty else "?"
    print(f"  Jornada: {jornada}")
    print(f"  Jugadores con predicción: {len(pred_df)}")

    # 2b. Comprobar ventana de clausulazos
    clausulazos_ok, horas_al_partido, claus_source = current_week_clausulazos_available(
        first_match_ts,
        fail_closed=False,
    )
    if first_match_ts > 0:
        from datetime import datetime as _dt, timezone as _tz
        first_dt = _dt.fromtimestamp(first_match_ts, tz=_tz.utc)
        print(f"  Primer partido: {first_dt.strftime('%d/%m %H:%M UTC')} "
              f"(en {horas_al_partido:.0f}h)")
        print(f"  Fuente ventana clausulazos: {claus_source}")
    if clausulazos_ok:
        print(f"  Clausulazos: DISPONIBLES")
    else:
        print(f"  Clausulazos: BLOQUEADOS (jornada en <24h)")

    # 3. Análisis del equipo
    print(f"\n  Analizando tu plantilla...")
    team_analysis = analyze_my_team(snapshot, pred_df)
    print(f"  xP once ideal: {team_analysis['xp_total_once']}")

    # 4. Jugadores disponibles
    print(f"\n  Analizando mercado y clausulazos...")
    available = analyze_available_players(snapshot, pred_df, mi_equipo["saldo_disponible"])
    print(f"  Mercado: {len(available['mercado'])} jugadores disponibles")
    if clausulazos_ok:
        print(f"  Clausulazos: {len(available['clausulazos'])} jugadores desbloqueados")
    else:
        print(f"  Clausulazos: BLOQUEADOS — no se incluiran en las recomendaciones")

    # 5. Simular plan de movimientos
    print(f"\n  Simulando plan de movimientos...")
    transfer_plan = simulate_transfer_plan(
        team_analysis, available, snapshot,
        allow_clausulazos=clausulazos_ok,
    )
    n_movs = len(transfer_plan["movimientos"])
    print(f"  Movimientos: {n_movs}")
    print(f"  xP once actual:  {team_analysis['xp_total_once']}")
    print(f"  xP once despues: {transfer_plan['xp_total_post']}")

    # 6. Informe
    report = generate_report(
        team_analysis, transfer_plan,
        snapshot, pred_df, MODEL_TYPE,
        clausulazos_ok=clausulazos_ok,
        horas_al_partido=horas_al_partido,
    )

    # Guardar
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"informe_j{jornada}.md"

    output_path.write_text(report, encoding="utf-8")
    print(f"\n  Informe guardado: {output_path}")

    # Resumen rápido en consola
    fm = _format_money
    print()
    print("=" * 60)
    print("  RESUMEN RAPIDO")
    print("=" * 60)

    print(f"\n  Once ACTUAL ({team_analysis.get('formacion_once', '4-3-3')}) — {team_analysis['xp_total_once']} xP:")
    for j in team_analysis["once_ideal"]:
        print(f"    {j['posicion']:<4} {j['nombre']:<22} xP={j['xP']:.1f}")

    if transfer_plan["movimientos"]:
        print(f"\n  MOVIMIENTOS:")
        for mov in transfer_plan["movimientos"]:
            v = mov.get("venta")
            c = mov.get("compra")
            if v and c:
                print(f"    Paso {mov['paso']}: Vender {v['nombre']} (+{fm(v['ingresos'])}) "
                      f"→ Fichar {c['nombre']} (-{fm(c['coste'])})")
            elif v:
                print(f"    Paso {mov['paso']}: Vender {v['nombre']} (+{fm(v['ingresos'])})")
            elif c:
                print(f"    Paso {mov['paso']}: Fichar {c['nombre']} (-{fm(c['coste'])})")

        print(f"\n  Once POST-MOVIMIENTOS ({transfer_plan.get('formacion_post', '4-3-3')}) — {transfer_plan['xp_total_post']} xP:")
        for j in transfer_plan["once_post"]:
            nuevo = " ★NEW" if j.get("nuevo_fichaje") else ""
            print(f"    {j['posicion']:<4} {j['nombre']:<22} xP={j['xP']:.1f}{nuevo}")

        diff = transfer_plan['xp_total_post'] - team_analysis['xp_total_once']
        print(f"\n  MEJORA: +{diff:.1f} xP en el once")
        print(f"  SALDO: {fm(team_analysis['saldo'])} → {fm(transfer_plan['saldo_final'])}")

    if team_analysis["problemas"]:
        print(f"\n  Alertas ({len(team_analysis['problemas'])}):")
        for p in team_analysis["problemas"][:5]:
            print(f"    {p['jugador']}: {p['problema']}")

    print()


if __name__ == "__main__":
    main()
