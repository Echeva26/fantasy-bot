"""
Auto-set de alineación (D-1)
============================
Calcula el mejor once por xP y lo guarda en la API de LaLiga Fantasy.

Pensado para ejecutarse automáticamente un día antes del inicio de jornada,
después del mercado.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from laliga_fantasy_client import LaLigaFantasyClient
from prediction.advisor import analyze_my_team, get_predictions, load_snapshot

logger = logging.getLogger(__name__)
MODEL_TYPE = "xgboost"


def _parse_hhmm(raw: str) -> tuple[int, int]:
    s = (raw or "").strip()
    hh, mm = s.split(":", 1)
    h = int(hh)
    m = int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Hora inválida: {raw}")
    return h, m


def _should_run_day_before(
    first_match_ts: int,
    tz_name: str,
    after_market_time: str,
) -> tuple[bool, str]:
    if first_match_ts <= 0:
        return False, "Sin timestamp del primer partido"

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")

    now_local = datetime.now(tz)
    first_local = datetime.fromtimestamp(first_match_ts, tz=tz)
    days_to_first = (first_local.date() - now_local.date()).days
    limit_h, limit_m = _parse_hhmm(after_market_time)
    after_market_ok = (now_local.hour, now_local.minute) >= (limit_h, limit_m)

    if days_to_first != 1:
        return (
            False,
            f"No es D-1 (faltan {days_to_first} días para el primer partido)",
        )
    if not after_market_ok:
        return (
            False,
            f"Antes de la hora post-mercado ({after_market_time})",
        )
    return True, "OK D-1 + post-mercado"


def _formation_parts(formation_str: str) -> tuple[int, int, int]:
    try:
        d, m, f = formation_str.split("-")
        return int(d), int(m), int(f)
    except Exception as exc:
        raise ValueError(f"Formación inválida: {formation_str}") from exc


def _build_lineup_payload(team_analysis: dict, snapshot: dict) -> dict:
    once = team_analysis.get("once_ideal", [])
    if len(once) < 11:
        raise RuntimeError(f"Once inválido: {len(once)} jugadores")

    form = team_analysis.get("formacion_once", "")
    n_def, n_med, n_del = _formation_parts(form)

    plantilla = snapshot.get("mi_equipo", {}).get("plantilla", [])
    pid_to_ptid = {
        int(p.get("player_id")): str(p.get("player_team_id", ""))
        for p in plantilla
        if p.get("player_id") is not None
    }

    def _ptid(player: dict) -> str:
        direct = str(player.get("player_team_id") or "").strip()
        if direct:
            return direct
        pid = int(player.get("player_id"))
        fallback = str(pid_to_ptid.get(pid, "")).strip()
        if fallback:
            return fallback
        raise RuntimeError(f"Falta player_team_id para {player.get('nombre', pid)}")

    by_pos: dict[str, list[dict]] = {"POR": [], "DEF": [], "MED": [], "DEL": []}
    for p in once:
        pos = p.get("posicion")
        if pos in by_pos:
            by_pos[pos].append(p)

    if len(by_pos["POR"]) != 1:
        raise RuntimeError("No hay exactamente 1 portero en el once")
    if len(by_pos["DEF"]) != n_def or len(by_pos["MED"]) != n_med or len(by_pos["DEL"]) != n_del:
        raise RuntimeError(
            f"Conteo posición/forma no cuadra: forma={n_def}-{n_med}-{n_del}, "
            f"def={len(by_pos['DEF'])}, med={len(by_pos['MED'])}, del={len(by_pos['DEL'])}"
        )

    starters_outfield = by_pos["DEF"] + by_pos["MED"] + by_pos["DEL"]
    captain = max(starters_outfield, key=lambda x: x.get("xP", 0)) if starters_outfield else by_pos["POR"][0]

    return {
        "formation": [n_def, n_med, n_del],
        "goalkeeper_id": _ptid(by_pos["POR"][0]),
        "defenders_ids": [_ptid(p) for p in by_pos["DEF"]],
        "midfielders_ids": [_ptid(p) for p in by_pos["MED"]],
        "strikers_ids": [_ptid(p) for p in by_pos["DEL"]],
        "captain_id": _ptid(captain),
    }


def autoset_best_lineup(
    *,
    league_id: str,
    model: str = MODEL_TYPE,
    day_before_only: bool = True,
    after_market_time: str = "08:10",
    timezone_name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    mt = (model or MODEL_TYPE).strip().lower()
    if mt != MODEL_TYPE:
        raise ValueError(f"Modelo no soportado: {model}. Solo se permite '{MODEL_TYPE}'.")

    tz_name = timezone_name or os.getenv("TZ", "Europe/Madrid")
    snapshot = load_snapshot(league_id)
    pred_df, first_match_ts = get_predictions(MODEL_TYPE)
    if pred_df.empty:
        raise RuntimeError("Sin predicciones disponibles para calcular alineación")

    jornada = int(pred_df["jornada"].iloc[0])
    if day_before_only and not force:
        ok, reason = _should_run_day_before(first_match_ts, tz_name, after_market_time)
        if not ok:
            return {
                "applied": False,
                "skipped": True,
                "reason": reason,
                "jornada": jornada,
                "first_match_ts": first_match_ts,
            }

    team_analysis = analyze_my_team(snapshot, pred_df)
    payload = _build_lineup_payload(team_analysis, snapshot)
    team_id = str(snapshot.get("mi_equipo", {}).get("team_id", ""))
    if not team_id:
        raise RuntimeError("No se encontró team_id en snapshot")

    if dry_run:
        return {
            "applied": False,
            "skipped": False,
            "dry_run": True,
            "jornada": jornada,
            "team_id": team_id,
            "formation": payload["formation"],
        }

    client = LaLigaFantasyClient.from_saved_token(league_id=league_id)
    client.update_team_lineup(
        team_id=team_id,
        tactical_formation=payload["formation"],
        goalkeeper_id=payload["goalkeeper_id"],
        defenders_ids=payload["defenders_ids"],
        midfielders_ids=payload["midfielders_ids"],
        strikers_ids=payload["strikers_ids"],
        captain_team_id=payload["captain_id"],
    )

    return {
        "applied": True,
        "skipped": False,
        "jornada": jornada,
        "team_id": team_id,
        "formation": payload["formation"],
        "captain_id": payload["captain_id"],
        "xp_once": team_analysis.get("xp_total_once"),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Auto-set alineación por xP (D-1)")
    parser.add_argument("--league", default=os.getenv("LALIGA_LEAGUE_ID", ""))
    parser.add_argument(
        "--after-market-time",
        default=os.getenv("LINEUP_AUTO_AFTER_TIME", "08:10"),
        help="Hora local mínima para ejecutar (HH:MM)",
    )
    parser.add_argument(
        "--timezone",
        default=os.getenv("TZ", "Europe/Madrid"),
        help="Zona horaria (IANA), ej. Europe/Madrid",
    )
    parser.add_argument("--force", action="store_true", help="Ignorar check D-1")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin guardar")
    parser.add_argument(
        "--no-day-before-check",
        action="store_true",
        help="No exigir que sea D-1",
    )
    args = parser.parse_args()

    if not args.league:
        raise RuntimeError("--league o LALIGA_LEAGUE_ID requerido")

    result = autoset_best_lineup(
        league_id=args.league,
        model=MODEL_TYPE,
        day_before_only=not args.no_day_before_check,
        after_market_time=args.after_market_time,
        timezone_name=args.timezone,
        force=args.force,
        dry_run=args.dry_run,
    )

    if result.get("skipped"):
        print(f"[SKIP] {result.get('reason')}")
        return
    if result.get("dry_run"):
        print(
            f"[DRY-RUN] Jornada {result.get('jornada')} | "
            f"Forma {'-'.join(str(x) for x in result.get('formation', []))}"
        )
        return

    print(
        f"[OK] Alineación guardada | jornada={result.get('jornada')} "
        f"forma={'-'.join(str(x) for x in result.get('formation', []))} "
        f"xP={result.get('xp_once')}"
    )


if __name__ == "__main__":
    main()
