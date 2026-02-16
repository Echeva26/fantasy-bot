"""
Persistencia y resolución de liga activa para el bot.

Permite seleccionar liga vía Telegram sin depender de LALIGA_LEAGUE_ID fijo.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LEAGUE_SELECTION_FILE = Path(".league_selection.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_placeholder(value: str) -> bool:
    v = str(value or "").strip().upper()
    return v in {"", "TU_LEAGUE_ID", "YOUR_LEAGUE_ID", "CHANGE_ME"}


def load_selected_league(path: Path = LEAGUE_SELECTION_FILE) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        lid = str(data.get("league_id", "")).strip()
        if not lid:
            return None
        return {
            "league_id": lid,
            "league_name": str(data.get("league_name", "")).strip(),
            "selected_at": str(data.get("selected_at", "")).strip(),
            "source": str(data.get("source", "")).strip(),
        }
    except Exception:
        return None


def save_selected_league(
    league_id: str,
    league_name: str = "",
    *,
    source: str = "manual",
    path: Path = LEAGUE_SELECTION_FILE,
) -> dict:
    payload = {
        "league_id": str(league_id).strip(),
        "league_name": str(league_name or "").strip(),
        "selected_at": _now_iso(),
        "source": str(source).strip(),
    }
    if not payload["league_id"]:
        raise ValueError("league_id vacío")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def get_selected_league_id(path: Path = LEAGUE_SELECTION_FILE) -> str:
    data = load_selected_league(path=path)
    return str((data or {}).get("league_id", "")).strip()


def resolve_league_id(
    explicit_league_id: str = "",
    *,
    env_var: str = "LALIGA_LEAGUE_ID",
    path: Path = LEAGUE_SELECTION_FILE,
) -> str:
    """
    Prioridad:
      1) league_id explícita (CLI/argumento)
      2) liga seleccionada en archivo local (Telegram)
      3) variable de entorno LALIGA_LEAGUE_ID
    """
    explicit = str(explicit_league_id or "").strip()
    if explicit and not _is_placeholder(explicit):
        return explicit

    selected = get_selected_league_id(path=path)
    if selected and not _is_placeholder(selected):
        return selected

    env_value = str(os.getenv(env_var, "")).strip()
    if env_value and not _is_placeholder(env_value):
        return env_value
    return ""
