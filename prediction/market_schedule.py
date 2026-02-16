"""
Utilidades para calcular horarios operativos en función del cierre real del mercado.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from laliga_fantasy_client import LaLigaFantasyClient

PRE_OFFSET_MINUTES = 10
POST_OFFSET_MINUTES = 10


def _is_league_market_item(item: dict) -> bool:
    """
    Detecta si un item de mercado pertenece a la liga (no a manager).
    """
    discr = str(item.get("discr", "")).strip().lower()
    if "marketplayerleague" in discr:
        return True

    # Fallback defensivo: en respuestas antiguas/variantes, sellerTeam nulo
    # suele indicar jugador publicado por la propia liga.
    if "marketplayer" in discr:
        seller = item.get("sellerTeam")
        if seller in (None, "", {}):
            return True
    return False


def _parse_iso_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _pick_close_dt(candidates: list[datetime]) -> datetime | None:
    if not candidates:
        return None

    # Agrupar por minuto para obtener el cierre dominante.
    per_minute = [dt.replace(second=0, microsecond=0) for dt in candidates]
    counts = Counter(per_minute)
    top_count = counts.most_common(1)[0][1]
    top = [dt for dt, c in counts.items() if c == top_count]
    if len(top) == 1:
        return top[0]

    now = datetime.now(timezone.utc)
    future = sorted(dt for dt in top if dt >= now - timedelta(minutes=30))
    if future:
        return future[0]
    return sorted(top)[-1]


def _normalize_close_dt(close_dt: datetime) -> datetime:
    """
    Normaliza al bloque de 5 minutos anterior para evitar ruido de minutos.
    Ejemplo: 20:18 -> 20:15.
    """
    dt = close_dt.replace(second=0, microsecond=0)
    minute = dt.minute - (dt.minute % 5)
    return dt.replace(minute=minute)


def get_market_close_datetime_utc(league_id: str) -> tuple[datetime | None, str]:
    """
    Devuelve (market_close_utc, error_msg).
    """
    lid = str(league_id or "").strip()
    if not lid:
        return None, "league_id vacío"
    try:
        client = LaLigaFantasyClient.from_saved_token(league_id=lid)
        raw = client.get_daily_market_raw() or []
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if not raw:
        return None, "mercado vacío"

    league_items = [i for i in raw if _is_league_market_item(i)]
    if not league_items:
        return None, "no se detectaron jugadores publicados por la liga en el mercado"
    source = league_items

    expirations = []
    for item in source:
        dt = _parse_iso_dt(item.get("expirationDate"))
        if dt:
            expirations.append(dt.astimezone(timezone.utc))

    close_dt = _pick_close_dt(expirations)
    if not close_dt:
        return None, "no se encontró expirationDate en el mercado"
    return _normalize_close_dt(close_dt), ""


def build_market_schedule(
    league_id: str,
    *,
    timezone_name: str = "Europe/Madrid",
    pre_offset_minutes: int = PRE_OFFSET_MINUTES,
    post_offset_minutes: int = POST_OFFSET_MINUTES,
) -> tuple[dict | None, str]:
    close_utc, err = get_market_close_datetime_utc(league_id)
    if not close_utc:
        return None, err

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = timezone.utc

    close_local = close_utc.astimezone(tz)
    pre_local = close_local - timedelta(minutes=int(pre_offset_minutes))
    post_local = close_local + timedelta(minutes=int(post_offset_minutes))
    market_key = close_local.strftime("%Y-%m-%dT%H:%M")

    return (
        {
            "market_key": market_key,
            "close_utc": close_utc,
            "close_local": close_local,
            "pre_local": pre_local,
            "post_local": post_local,
            "timezone_name": timezone_name,
        },
        "",
    )


def schedule_message(schedule: dict) -> str:
    close_local = schedule["close_local"]
    pre_local = schedule["pre_local"]
    post_local = schedule["post_local"]
    tz_name = schedule.get("timezone_name", "")
    return (
        f"Mercado: {close_local.strftime('%H:%M')}\n"
        f"PRE automático: {pre_local.strftime('%H:%M')} (10 min antes)\n"
        f"POST automático: {post_local.strftime('%H:%M')} (10 min después)\n"
        f"Zona horaria: {tz_name}"
    )
