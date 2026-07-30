"""
LangChain tools para gestionar LaLiga Fantasy.

Este módulo envuelve la lógica existente del repositorio en herramientas
estructuradas para un agente LLM.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from laliga_fantasy_client import LaLigaFantasyClient
from prediction.advisor import (
    analyze_available_players,
    analyze_my_team,
    sale_keeps_lineup_viable,
    compute_competitive_bid_amount,
    current_week_clausulazos_available,
    get_predictions,
    load_snapshot,
    simulate_transfer_plan,
)
from prediction.advisor_execute import execute_movements, run_aceptar_ofertas
from prediction.langsmith_config import finish_langsmith_span, langsmith_run_span
from prediction.lineup_autoset import autoset_best_lineup
from prediction.scrape_freshness import (
    latest_scrape_file,
    max_age_minutes,
    scrape_age_minutes,
    scrapes_dir,
)

logger = logging.getLogger(__name__)
MODEL_TYPE = "xgboost"
CLAUSE_INCREASE_FACTOR = 2.0
CLAUSE_EXPOSURE_THRESHOLD = 0.88


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _as_json(payload: Any) -> str:
    return json.dumps(_to_builtin(payload), ensure_ascii=False)


def _tool_run_name(tool_name: str) -> str:
    return f"fantasy-bot.tool.{tool_name}"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _norm_text(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = "".join(
        ch for ch in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(ch)
    )
    return " ".join(raw.split())


def _clause_exposure_ratio(market_value: Any, clause_value: Any) -> float | None:
    market = _safe_int(market_value, 0)
    clause = _safe_int(clause_value, 0)
    if market <= 0 or clause <= 0:
        return None
    return float(market) / float(clause)


@dataclass
class FantasyAgentRuntime:
    league_id: str
    model_type: str = MODEL_TYPE
    dry_run: bool = False
    phase: str = "full"
    _snapshot: dict | None = field(default=None, init=False, repr=False)
    _pred_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _first_match_ts: int = field(default=0, init=False, repr=False)
    _team_analysis: dict | None = field(default=None, init=False, repr=False)
    _available: dict | None = field(default=None, init=False, repr=False)
    _transfer_plan: dict | None = field(default=None, init=False, repr=False)
    _clausulazos_ok: bool | None = field(default=None, init=False, repr=False)
    _hours_to_first_match: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        mt = (self.model_type or MODEL_TYPE).strip().lower()
        if mt != MODEL_TYPE:
            logger.warning(
                "Modelo no soportado '%s' en runtime LangChain; usando '%s'.",
                self.model_type,
                MODEL_TYPE,
            )
        self.model_type = MODEL_TYPE
        phase = (self.phase or "full").strip().lower()
        if phase not in {"pre", "post", "full"}:
            logger.warning(
                "Fase no soportada '%s' en runtime LangChain; usando 'full'.",
                self.phase,
            )
            phase = "full"
        self.phase = phase

    def invalidate(self) -> None:
        self._snapshot = None
        self._pred_df = None
        self._team_analysis = None
        self._available = None
        self._transfer_plan = None
        self._clausulazos_ok = None
        self._hours_to_first_match = None

    def get_snapshot(self, force_refresh: bool = False) -> dict:
        if self._snapshot is None or force_refresh:
            self._snapshot = load_snapshot(self.league_id)
            self._team_analysis = None
            self._available = None
            self._transfer_plan = None
        return self._snapshot

    def get_predictions(self, force_refresh: bool = False) -> tuple[pd.DataFrame, int]:
        if self._pred_df is None or force_refresh:
            self._pred_df, self._first_match_ts = get_predictions(self.model_type)
            self._team_analysis = None
            self._available = None
            self._transfer_plan = None
        return self._pred_df, int(self._first_match_ts)

    def get_team_analysis(self, force_refresh: bool = False) -> dict:
        if self._team_analysis is None or force_refresh:
            snapshot = self.get_snapshot(force_refresh=force_refresh)
            pred_df, _ = self.get_predictions(force_refresh=force_refresh)
            self._team_analysis = analyze_my_team(snapshot, pred_df)
            self._transfer_plan = None
        return self._team_analysis

    def get_available(self, force_refresh: bool = False) -> dict:
        if self._available is None or force_refresh:
            snapshot = self.get_snapshot(force_refresh=force_refresh)
            pred_df, _ = self.get_predictions(force_refresh=force_refresh)
            saldo = snapshot.get("mi_equipo", {}).get("saldo_disponible", 0)
            self._available = analyze_available_players(snapshot, pred_df, saldo)
            self._transfer_plan = None
        return self._available

    def get_transfer_plan(self, force_refresh: bool = False) -> tuple[dict, bool, float]:
        if self._transfer_plan is None or force_refresh:
            snapshot = self.get_snapshot(force_refresh=force_refresh)
            team_analysis = self.get_team_analysis(force_refresh=force_refresh)
            available = self.get_available(force_refresh=force_refresh)
            _, first_match_ts = self.get_predictions(force_refresh=force_refresh)
            claus_ok, hours_to_match, _ = current_week_clausulazos_available(
                first_match_ts,
                fail_closed=False,
            )
            self._transfer_plan = simulate_transfer_plan(
                team_analysis=team_analysis,
                available=available,
                snapshot=snapshot,
                allow_clausulazos=claus_ok,
            )
            self._clausulazos_ok = claus_ok
            self._hours_to_first_match = hours_to_match
        return (
            self._transfer_plan,
            bool(self._clausulazos_ok),
            float(self._hours_to_first_match or 0.0),
        )

    def get_client(self) -> LaLigaFantasyClient:
        return LaLigaFantasyClient.from_saved_token(league_id=self.league_id)


def _compress_transfer_plan(plan: dict) -> dict:
    out = {
        "modo_deuda": plan.get("modo_deuda", False),
        "deuda_objetivo": plan.get("deuda_objetivo"),
        "deuda_cubierta_estimada": plan.get("deuda_cubierta_estimada"),
        "deuda_pendiente_estimada": plan.get("deuda_pendiente_estimada"),
        "alertas": plan.get("alertas", []),
        "priority_needs": plan.get("priority_needs", []),
        "non_executable_recommendations": plan.get("non_executable_recommendations", []),
        "saldo_final": plan.get("saldo_final"),
        "xp_total_post": plan.get("xp_total_post"),
        "formacion_post": plan.get("formacion_post"),
        "jugadores_vendidos": plan.get("jugadores_vendidos"),
        "jugadores_comprados": plan.get("jugadores_comprados"),
        "movimientos": [],
    }
    for mov in plan.get("movimientos", []):
        row = {
            "paso": mov.get("paso"),
            "saldo_antes": mov.get("saldo_antes"),
            "saldo_despues": mov.get("saldo_despues"),
            "ganancia_xp": mov.get("ganancia_xp"),
        }
        venta = mov.get("venta")
        compra = mov.get("compra")
        if venta:
            row["venta"] = {
                "nombre": venta.get("nombre"),
                "player_id": venta.get("player_id"),
                "player_team_id": venta.get("player_team_id"),
                "valor_mercado": venta.get("valor_mercado"),
                "clausula": venta.get("clausula"),
                "ratio_clausula_valor": venta.get("ratio_clausula_valor"),
                "precio_publicacion": venta.get("precio_publicacion"),
                "motivo": venta.get("motivo"),
                "xP": venta.get("xP"),
                "impacto_xp_once": venta.get("impacto_xp_once"),
                "ya_en_venta": venta.get("ya_en_venta"),
            }
        if compra:
            row["compra"] = {
                "nombre": compra.get("nombre"),
                "player_id": compra.get("player_id"),
                "player_team_id": compra.get("player_team_id"),
                "market_item_id": compra.get("market_item_id"),
                "tipo": compra.get("tipo"),
                "coste": compra.get("coste"),
                "xP": compra.get("xP"),
                "propietario": compra.get("propietario"),
                "motivo": compra.get("motivo"),
            }
        out["movimientos"].append(row)
    return out


def build_langchain_tools(runtime: FantasyAgentRuntime) -> list:
    try:
        from langchain_core.tools import tool
    except Exception as exc:
        raise RuntimeError(
            "LangChain no está instalado. Instala dependencias con: pip install -r requirements.txt"
        ) from exc

    def _block_if_post(action: str) -> str | None:
        if runtime.phase != "post":
            return None
        return _as_json(
            {
                "ok": False,
                "blocked": True,
                "phase": runtime.phase,
                "action": action,
                "reason": (
                    "Accion bloqueada en fase POST. "
                    "En POST solo se permiten accept_closed_offers y autoset_best_lineup_tool."
                ),
            }
        )

    def _block_if_buyout_locked(action: str = "buyout_player_tool") -> str | None:
        try:
            claus_ok, hours_to_match, source = current_week_clausulazos_available(
                fail_closed=True,
            )
        except Exception as exc:
            return _as_json(
                {
                    "ok": False,
                    "blocked": True,
                    "action": action,
                    "reason": (
                        "No se pudo validar la ventana de clausulazos; por seguridad "
                        "no se ejecuta ningún clausulazo real."
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if claus_ok:
            return None
        return _as_json(
            {
                "ok": False,
                "blocked": True,
                "action": action,
                "hours_to_first_match": round(hours_to_match, 1),
                "source": source,
                "reason": (
                    "Clausulazos bloqueados: LaLiga Fantasy no permite comprar "
                    "jugadores mediante clausulazo desde 24h antes del inicio de "
                    "la jornada. Usa mercado de pujas, ventas o subidas de cláusula."
                ),
            }
        )

    @tool
    def snapshot_summary(force_refresh: bool = False) -> str:
        """Obtiene el contexto global actual: liga, equipo, saldo, posición en la clasificación
        y tamaño del mercado. LLAMA ESTO PRIMERO para saber el estado antes de tomar decisiones.
        CRÍTICO: comprueba que saldo_disponible sea positivo (si es negativo antes de jornada = 0 puntos)."""
        snapshot = runtime.get_snapshot(force_refresh=force_refresh)
        mi = snapshot.get("mi_equipo", {})
        payload = {
            "league_id": snapshot.get("league_id"),
            "league_name": snapshot.get("league_name"),
            "snapshot_at": snapshot.get("snapshot_at"),
            "manager_name": mi.get("manager_name"),
            "team_id": mi.get("team_id"),
            "posicion": mi.get("posicion"),
            "puntos": mi.get("puntos"),
            "saldo_disponible": mi.get("saldo_disponible"),
            "num_jugadores": mi.get("num_jugadores"),
            "mercado_items": len(snapshot.get("mercado", [])),
            "num_rivales": len(snapshot.get("rivales", [])),
        }
        return _as_json(payload)

    @tool
    def my_squad(force_refresh: bool = False) -> str:
        """Lista TODOS los jugadores de tu plantilla con xP, estado, cláusula y valor de mercado.
        Ordenados por xP descendente. Usa esto para identificar:
        - Jugadores con xP alta (candidatos a capitán y titulares fijos)
        - Jugadores lesionados/sancionados (candidatos a venta)
        - Jugadores con cláusula expuesta (ratio_valor_vs_clausula ≥ 0.85 → subir cláusula)
        - Huecos en la alineación por posición"""
        team_analysis = runtime.get_team_analysis(force_refresh=force_refresh)
        players = sorted(
            team_analysis.get("jugadores", []),
            key=lambda x: (x.get("xP", 0), x.get("valor_mercado", 0)),
            reverse=True,
        )
        payload = {
            "xp_total_once": team_analysis.get("xp_total_once"),
            "formacion_once": team_analysis.get("formacion_once"),
            "players": [
                {
                    "player_id": p.get("player_id"),
                    "player_team_id": p.get("player_team_id"),
                    "nombre": p.get("nombre"),
                    "posicion": p.get("posicion"),
                    "estado": p.get("estado"),
                    "en_venta": p.get("en_venta"),
                    "valor_mercado": p.get("valor_mercado"),
                    "clausula": p.get("clausula"),
                    "clausula_bloqueada_hasta": p.get("clausula_bloqueada_hasta"),
                    "ratio_valor_vs_clausula": _clause_exposure_ratio(
                        p.get("valor_mercado"),
                        p.get("clausula"),
                    ),
                    "xP": p.get("xP", 0),
                    "rival": p.get("rival"),
                    "es_local": p.get("es_local"),
                }
                for p in players
            ],
        }
        return _as_json(payload)

    @tool
    def predictions_top(limit: int = 25, position: str = "", force_refresh: bool = False) -> str:
        """Muestra los jugadores con mayor xP predicha (modelo XGBoost) para la próxima jornada.
        Filtra por posición con POR/DEF/MED/DEL. Usa esto para:
        - Identificar candidatos a capitán (top 1-3 del ranking general)
        - Buscar fichajes potenciales por posición
        - Comparar si tus titulares están entre los top de su posición"""
        pred_df, first_match_ts = runtime.get_predictions(force_refresh=force_refresh)
        if pred_df.empty:
            return _as_json({"count": 0, "items": [], "first_match_ts": first_match_ts})
        n = max(1, min(int(limit), 200))
        df = pred_df
        if position:
            df = df[df["posicion"] == position.upper()]
        top = df.head(n)
        return _as_json(
            {
                "count": len(top),
                "jornada": int(top["jornada"].iloc[0]) if len(top) else None,
                "first_match_ts": int(first_match_ts),
                "items": top[
                    ["player_id", "nombre", "posicion", "equipo", "rival", "es_local", "xP"]
                ].to_dict("records"),
            }
        )

    @tool
    def player_outlook(player_query: str, force_refresh: bool = False) -> str:
        """Busca información detallada de un jugador por nombre o player_id.
        Devuelve: rival próximo, si juega local/visitante, xP predicha y estado.
        Útil para evaluar un fichaje concreto o verificar la situación de un jugador
        antes de venderlo, comprarlo o ponerlo de capitán."""
        query = (player_query or "").strip().lower()
        if not query:
            return _as_json({"count": 0, "items": [], "error": "player_query vacío"})

        pred_df, _ = runtime.get_predictions(force_refresh=force_refresh)
        team_analysis = runtime.get_team_analysis(force_refresh=force_refresh)
        team_index = {
            int(p.get("player_id")): p for p in team_analysis.get("jugadores", []) if p.get("player_id") is not None
        }

        matches = []
        if query.isdigit():
            pid = int(query)
            rows = pred_df[pred_df["player_id"].astype(int) == pid]
        else:
            rows = pred_df[pred_df["nombre"].str.lower().str.contains(query, na=False)]

        for _, row in rows.head(20).iterrows():
            pid = int(row.get("player_id"))
            team_row = team_index.get(pid, {})
            matches.append(
                {
                    "player_id": pid,
                    "nombre": row.get("nombre"),
                    "equipo": row.get("equipo"),
                    "posicion": row.get("posicion"),
                    "rival": row.get("rival"),
                    "es_local": row.get("es_local"),
                    "xP": row.get("xP"),
                    "estado": team_row.get("estado", "ok"),
                    "player_team_id": team_row.get("player_team_id", ""),
                }
            )

        return _as_json({"count": len(matches), "items": matches})

    @tool
    def market_opportunities(limit: int = 30, force_refresh: bool = False) -> str:
        """Devuelve las mejores oportunidades de fichaje: pujas en mercado libre Y clausulazos.
        Cada item incluye coste estimado (con incremento competitivo si hay otras pujas), xP,
        rival próximo y si juega de local. Los clausulazos solo aparecen si están disponibles
        (ventana temporal adecuada). Recuerda: las pujas son secretas y gana la más alta."""
        available = runtime.get_available(force_refresh=force_refresh)
        _, first_match_ts = runtime.get_predictions(force_refresh=force_refresh)
        claus_ok, hours_to_match, claus_source = current_week_clausulazos_available(
            first_match_ts,
            fail_closed=False,
        )

        items = []
        for p in available.get("mercado", []):
            coste_base = max(p.get("precio_venta", 0) or 0, p.get("valor_mercado", 0) or 0)
            pujas_detectadas = _safe_int(p.get("pujas"), 0)
            coste = compute_competitive_bid_amount(
                base_amount=coste_base,
                market_value=_safe_int(p.get("valor_mercado"), 0),
                external_bids=pujas_detectadas,
                expected_points=_safe_float(p.get("xP"), 0.0),
            )
            items.append(
                {
                    "tipo": "mercado",
                    "player_id": p.get("player_id"),
                    "market_item_id": p.get("market_item_id"),
                    "nombre": p.get("nombre"),
                    "posicion": p.get("posicion"),
                    "equipo_real": p.get("equipo_real"),
                    "coste_base": coste_base,
                    "pujas_detectadas": pujas_detectadas,
                    "incremento_competitivo": max(0, _safe_int(coste) - _safe_int(coste_base)),
                    "coste": coste,
                    "xP": p.get("xP"),
                    "rival_next": p.get("rival_next"),
                    "es_local": p.get("es_local"),
                    "valor_mercado": p.get("valor_mercado"),
                }
            )

        if claus_ok:
            for p in available.get("clausulazos", []):
                items.append(
                    {
                        "tipo": "clausulazo",
                        "player_id": p.get("player_id"),
                        "player_team_id": p.get("player_team_id"),
                        "nombre": p.get("nombre"),
                        "posicion": p.get("posicion"),
                        "equipo_real": p.get("equipo_real"),
                        "coste": p.get("clausula", 0),
                        "xP": p.get("xP"),
                        "rival_next": p.get("rival_next"),
                        "es_local": p.get("es_local"),
                        "valor_mercado": p.get("valor_mercado"),
                        "propietario": p.get("propietario"),
                    }
                )

        items.sort(key=lambda x: (x.get("xP", 0), -(x.get("coste", 0) or 0)), reverse=True)
        n = max(1, min(int(limit), 200))

        return _as_json(
            {
                "count": min(len(items), n),
                "hours_to_first_match": round(hours_to_match, 1),
                "hours_to_lineup_deadline": round(hours_to_match, 1),
                "clausulazos_available": claus_ok,
                "clausulazos_window_source": claus_source,
                "items": items[:n],
            }
        )

    @tool
    def news_reader_tool(limit: int = 30) -> str:
        """
        Lector RAG local de noticias y scrapes recientes.

        Lee el JSON más nuevo de `scrapes/` y devuelve lesiones, sanciones,
        apercibidos y movimientos de mercado relevantes para mi plantilla y
        jugadores disponibles.
        """
        scrape_dir = scrapes_dir()
        if not scrape_dir.exists():
            return _as_json(
                {
                    "ok": False,
                    "fresh": False,
                    "stale": True,
                    "source": "scrapes",
                    "latest_file": "",
                    "age_minutes": None,
                    "reason": "No existe el directorio scrapes/. Ejecuta python -m scrapers.scrape_all.",
                    "items": [],
                }
            )

        latest = latest_scrape_file(scrape_dir)
        if latest is None:
            return _as_json(
                {
                    "ok": False,
                    "fresh": False,
                    "stale": True,
                    "source": "scrapes",
                    "latest_file": "",
                    "age_minutes": None,
                    "reason": "No hay snapshots JSON en scrapes/. Ejecuta python -m scrapers.scrape_all.",
                    "items": [],
                }
            )

        age = scrape_age_minutes(latest)
        max_age = max_age_minutes()
        fresh = age <= max_age
        stale_reason = (
            ""
            if fresh
            else f"Último scrape obsoleto ({age:.1f} min > {max_age} min)."
        )
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except Exception as exc:
            return _as_json(
                {
                    "ok": False,
                    "fresh": False,
                    "stale": True,
                    "source": "scrapes",
                    "latest_file": str(latest),
                    "age_minutes": round(age, 1),
                    "source_file": str(latest),
                    "reason": f"No se pudo leer el scrape: {type(exc).__name__}: {exc}",
                    "items": [],
                }
            )

        snapshot = runtime.get_snapshot(force_refresh=False)
        my_players = snapshot.get("mi_equipo", {}).get("plantilla", []) or []
        market_players = snapshot.get("mercado", []) or []
        names: set[str] = set()
        for p in my_players:
            names.add(_norm_text(p.get("nombre") or p.get("player_name") or p.get("name")))
        for p in market_players:
            names.add(_norm_text(p.get("nombre") or p.get("player_name") or p.get("name")))
        names.discard("")

        def _is_relevant(row: dict) -> bool:
            row_name = _norm_text(row.get("nombre") or row.get("name"))
            if not row_name:
                return False
            if row_name in names:
                return True
            return any(row_name in n or n in row_name for n in names if len(n) >= 5)

        items: list[dict] = []
        ff = data.get("futbolfantasy", {}) if isinstance(data, dict) else {}
        if isinstance(ff, dict):
            for row in ff.get("lesionados", []) or []:
                if isinstance(row, dict) and _is_relevant(row):
                    items.append(
                        {
                            "tipo": "lesion",
                            "nombre": row.get("nombre"),
                            "equipo": row.get("equipo"),
                            "estado": row.get("estado"),
                            "probabilidad_titular": row.get("probabilidad_titular"),
                            "detalle": row.get("lesion") or row.get("info_retorno"),
                            "info_retorno": row.get("info_retorno"),
                            "url": row.get("url_parte"),
                        }
                    )
            for row in ff.get("sancionados", []) or []:
                if isinstance(row, dict) and _is_relevant(row):
                    items.append(
                        {
                            "tipo": "sancion",
                            "nombre": row.get("nombre"),
                            "motivo": row.get("motivo"),
                            "ciclo": row.get("ciclo"),
                            "jornadas_tarjetas": row.get("jornadas_tarjetas"),
                        }
                    )
            for row in ff.get("apercibidos", []) or []:
                if isinstance(row, dict) and _is_relevant(row):
                    items.append(
                        {
                            "tipo": "apercibido",
                            "nombre": row.get("nombre"),
                            "motivo": row.get("motivo") or row.get("ciclo"),
                            "jornadas_tarjetas": row.get("jornadas_tarjetas"),
                        }
                    )

        mv = data.get("market_values", {}) if isinstance(data, dict) else {}
        mercado = mv.get("mercado", {}) if isinstance(mv, dict) else {}
        if isinstance(mercado, dict):
            for bucket in ("subidas", "bajadas"):
                for row in mercado.get(bucket, []) or []:
                    if isinstance(row, dict) and _is_relevant(row):
                        items.append(
                            {
                                "tipo": f"valor_{bucket[:-1]}",
                                "nombre": row.get("nombre") or row.get("name"),
                                "equipo": row.get("equipo") or row.get("team"),
                                "valor": row.get("valor") or row.get("market_value"),
                                "variacion": row.get("variacion") or row.get("change"),
                            }
                        )

        n = max(1, min(int(limit), 200))
        return _as_json(
            {
                "ok": True,
                "fresh": fresh,
                "stale": not fresh,
                "source": "scrapes",
                "latest_file": str(latest),
                "age_minutes": round(age, 1),
                "source_file": str(latest),
                "scraped_at": data.get("scraped_at") if isinstance(data, dict) else None,
                "reason": stale_reason,
                "counts": {
                    "lesionados": len(ff.get("lesionados", []) or []) if isinstance(ff, dict) else 0,
                    "sancionados": len(ff.get("sancionados", []) or []) if isinstance(ff, dict) else 0,
                    "apercibidos": len(ff.get("apercibidos", []) or []) if isinstance(ff, dict) else 0,
                    "relevantes": len(items),
                },
                "items": items[:n],
            }
        )

    @tool
    def simulate_transfer_plan(force_refresh: bool = False) -> str:
        """Simula plan de ventas/compras recomendado por el motor actual del repositorio."""
        with langsmith_run_span(
            _tool_run_name("simulate_transfer_plan"),
            run_type="tool",
            phase=runtime.phase,
            command="agent-tool",
            league_id=runtime.league_id,
            dry_run=runtime.dry_run,
            extra_metadata={"tool_name": "simulate_transfer_plan"},
            inputs={"force_refresh": bool(force_refresh)},
        ) as span:
            team_analysis = runtime.get_team_analysis(force_refresh=force_refresh)
            plan, claus_ok, hours_to_match = runtime.get_transfer_plan(force_refresh=force_refresh)
            payload = {
                "clausulazos_available": claus_ok,
                "hours_to_first_match": round(hours_to_match, 1),
                "hours_to_lineup_deadline": round(hours_to_match, 1),
                "clausulazos_window_source": "laliga_calendar_or_fallback",
                "summary": {
                    "modo_deuda": bool(plan.get("modo_deuda", False)),
                    "deuda_objetivo": plan.get("deuda_objetivo"),
                    "deuda_cubierta_estimada": plan.get("deuda_cubierta_estimada"),
                    "deuda_pendiente_estimada": plan.get("deuda_pendiente_estimada"),
                    "xp_once_actual": team_analysis.get("xp_total_once"),
                    "xp_once_post": plan.get("xp_total_post"),
                    "xp_delta": round((plan.get("xp_total_post", 0) or 0) - (team_analysis.get("xp_total_once", 0) or 0), 1),
                    "saldo_actual": team_analysis.get("saldo"),
                    "saldo_final": plan.get("saldo_final"),
                    "movimientos": len(plan.get("movimientos", [])),
                },
                "plan": _compress_transfer_plan(plan),
                "priority_needs": plan.get("priority_needs", []),
                "non_executable_recommendations": plan.get("non_executable_recommendations", []),
            }
            finish_langsmith_span(
                span,
                {
                    "ok": True,
                    "movimientos": len(plan.get("movimientos", [])),
                    "saldo_actual": team_analysis.get("saldo"),
                    "saldo_final": plan.get("saldo_final"),
                    "modo_deuda": bool(plan.get("modo_deuda", False)),
                },
            )
            return _as_json(payload)

    @tool
    def execute_simulated_plan(force_refresh: bool = False) -> str:
        """Ejecuta el plan simulado (ventas fase1 + compras). Respeta `dry_run` del runtime."""
        with langsmith_run_span(
            _tool_run_name("execute_simulated_plan"),
            run_type="tool",
            phase=runtime.phase,
            command="agent-tool",
            league_id=runtime.league_id,
            dry_run=runtime.dry_run,
            extra_metadata={"tool_name": "execute_simulated_plan"},
            inputs={"force_refresh": bool(force_refresh)},
        ) as span:
            blocked = _block_if_post("execute_simulated_plan")
            if blocked:
                finish_langsmith_span(span, {"ok": False, "blocked": True})
                return blocked
            try:
                snapshot = runtime.get_snapshot(force_refresh=force_refresh)
                plan, _, _ = runtime.get_transfer_plan(force_refresh=force_refresh)
                movimientos = plan.get("movimientos", [])
                ventas, compras, errores = execute_movements(
                    snapshot=snapshot,
                    transfer_plan=plan,
                    dry_run=runtime.dry_run,
                )
                runtime.invalidate()
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "dry_run": runtime.dry_run,
                        "movimientos_planificados": len(movimientos),
                        "ventas": len(ventas or []),
                        "compras": len(compras or []),
                        "errores": len(errores or []),
                    },
                )
                return _as_json(
                    {
                        "dry_run": runtime.dry_run,
                        "movimientos_planificados": len(movimientos),
                        "ventas_fase1": ventas,
                        "compras": compras,
                        "errores": errores,
                    }
                )
            except Exception as exc:
                finish_langsmith_span(
                    span,
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def accept_closed_offers() -> str:
        """Ejecuta fase2 de ventas: aceptar ofertas de liga ya cerradas."""
        with langsmith_run_span(
            _tool_run_name("accept_closed_offers"),
            run_type="tool",
            phase=runtime.phase,
            command="agent-tool",
            league_id=runtime.league_id,
            dry_run=runtime.dry_run,
            extra_metadata={"tool_name": "accept_closed_offers"},
            inputs={},
        ) as span:
            if runtime.dry_run:
                finish_langsmith_span(span, {"ok": True, "dry_run": True, "accepted_offers": 0})
                return _as_json(
                    {
                        "dry_run": True,
                        "action": "accept_closed_offers",
                        "note": "No se ejecuta fase2 real en dry-run.",
                    }
                )
            try:
                args = SimpleNamespace(league=runtime.league_id)
                accepted = run_aceptar_ofertas(args)
                runtime.invalidate()
                finish_langsmith_span(
                    span,
                    {"ok": True, "accepted_offers": int(accepted), "dry_run": runtime.dry_run},
                )
                return _as_json({"accepted_offers": int(accepted), "dry_run": runtime.dry_run})
            except Exception as exc:
                finish_langsmith_span(
                    span,
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                )
                return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def autoset_best_lineup_tool(
        day_before_only: bool = True,
        force: bool = False,
        after_market_time: str = "08:10",
    ) -> str:
        """Calcula y guarda automáticamente la mejor alineación por xP."""
        with langsmith_run_span(
            _tool_run_name("autoset_best_lineup"),
            run_type="tool",
            phase=runtime.phase,
            command="agent-tool",
            league_id=runtime.league_id,
            dry_run=runtime.dry_run,
            extra_metadata={"tool_name": "autoset_best_lineup_tool"},
            inputs={
                "day_before_only": bool(day_before_only),
                "force": bool(force),
                "after_market_time": str(after_market_time),
            },
        ) as span:
            try:
                result = autoset_best_lineup(
                    league_id=runtime.league_id,
                    model=runtime.model_type,
                    day_before_only=day_before_only,
                    after_market_time=after_market_time,
                    timezone_name=None,
                    force=force,
                    dry_run=runtime.dry_run,
                )
                if result.get("applied"):
                    runtime.invalidate()
                result["dry_run_runtime"] = runtime.dry_run
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "applied": bool(result.get("applied")),
                        "dry_run": bool(result.get("dry_run")),
                        "formation": result.get("formation"),
                        "xp_once": result.get("xp_once"),
                    },
                )
                return _as_json(result)
            except Exception as exc:
                finish_langsmith_span(
                    span,
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                )
                return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def sell_player_phase1_tool(player_team_id: str, sale_price: int) -> str:
        """Publica un jugador propio en mercado (fase1). Requiere `player_team_id`."""
        with langsmith_run_span(
            _tool_run_name("sell_player_phase1"),
            run_type="tool",
            phase=runtime.phase,
            command="agent-tool",
            league_id=runtime.league_id,
            dry_run=runtime.dry_run,
            extra_metadata={"tool_name": "sell_player_phase1_tool"},
            inputs={"player_team_id": str(player_team_id or "").strip(), "sale_price": _safe_int(sale_price, 0)},
        ) as span:
            blocked = _block_if_post("sell_player_phase1_tool")
            if blocked:
                finish_langsmith_span(span, {"ok": False, "blocked": True})
                return blocked
            ptid = str(player_team_id or "").strip()
            price = _safe_int(sale_price, 0)
            if not ptid:
                finish_langsmith_span(span, {"ok": False, "error": "player_team_id vacío"})
                return _as_json({"ok": False, "error": "player_team_id vacío"})
            if price <= 0:
                finish_langsmith_span(span, {"ok": False, "error": "sale_price debe ser > 0"})
                return _as_json({"ok": False, "error": "sale_price debe ser > 0"})

            team_analysis = runtime.get_team_analysis(force_refresh=False)
            players = [
                p for p in team_analysis.get("jugadores", [])
                if isinstance(p, dict)
            ]
            selected = None
            for p in players:
                if str(p.get("player_team_id", "")).strip() == ptid:
                    selected = p
                    break
            if not selected:
                finish_langsmith_span(span, {"ok": False, "player_team_id": ptid, "found": False})
                return _as_json(
                    {
                        "ok": False,
                        "error": "No se encontró ese player_team_id en tu plantilla actual.",
                        "player_team_id": ptid,
                    }
                )

            if bool(selected.get("en_venta")):
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "skipped": True,
                        "already_on_market": True,
                        "player_team_id": ptid,
                        "nombre": selected.get("nombre"),
                    },
                )
                return _as_json(
                    {
                        "ok": True,
                        "skipped": True,
                        "action": "sell_player_phase1",
                        "already_on_market": True,
                        "player_team_id": ptid,
                        "nombre": selected.get("nombre"),
                        "precio_venta": selected.get("precio_venta"),
                        "market_player_id": selected.get("market_player_id"),
                        "reason": "Venta omitida: el jugador ya esta publicado en el mercado.",
                    }
                )

            if not sale_keeps_lineup_viable(players, int(selected.get("player_id", 0) or 0)):
                finish_langsmith_span(
                    span,
                    {
                        "ok": False,
                        "blocked": True,
                        "player_team_id": ptid,
                        "nombre": selected.get("nombre"),
                    },
                )
                return _as_json(
                    {
                        "ok": False,
                        "blocked": True,
                        "action": "sell_player_phase1",
                        "player_team_id": ptid,
                        "nombre": selected.get("nombre"),
                        "posicion": selected.get("posicion"),
                        "xP": selected.get("xP"),
                        "valor_mercado": selected.get("valor_mercado"),
                        "clausula": selected.get("clausula"),
                        "reason": (
                            "Venta bloqueada: quitar este jugador dejaría la plantilla "
                            "sin una alineación viable según las formaciones permitidas."
                        ),
                    }
                )
            if runtime.dry_run:
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "dry_run": True,
                        "player_team_id": ptid,
                        "nombre": selected.get("nombre"),
                        "sale_price": price,
                    },
                )
                return _as_json(
                    {
                        "dry_run": True,
                        "action": "sell_player_phase1",
                        "player_team_id": ptid,
                        "sale_price": price,
                        "nombre": selected.get("nombre"),
                    }
                )
            try:
                client = runtime.get_client()
                client.sell_player_phase1(player_team_id=ptid, price=price)
                runtime.invalidate()
                finish_langsmith_span(
                    span,
                    {"ok": True, "player_team_id": ptid, "nombre": selected.get("nombre"), "sale_price": price},
                )
                return _as_json({"ok": True, "player_team_id": ptid, "sale_price": price})
            except Exception as exc:
                finish_langsmith_span(
                    span,
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "player_team_id": ptid},
                )
                return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def place_bid_tool(market_item_id: str, amount: int, player_id: int = 0) -> str:
        """Hace/actualiza una puja en mercado libre."""
        with langsmith_run_span(
            _tool_run_name("place_bid"),
            run_type="tool",
            phase=runtime.phase,
            command="agent-tool",
            league_id=runtime.league_id,
            dry_run=runtime.dry_run,
            extra_metadata={"tool_name": "place_bid_tool"},
            inputs={
                "market_item_id": str(market_item_id or "").strip(),
                "amount": int(amount),
                "player_id": int(player_id) if player_id else None,
            },
        ) as span:
            blocked = _block_if_post("place_bid_tool")
            if blocked:
                finish_langsmith_span(span, {"ok": False, "blocked": True})
                return blocked
            if runtime.dry_run:
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "dry_run": True,
                        "market_item_id": market_item_id,
                        "amount": int(amount),
                    },
                )
                return _as_json(
                    {
                        "dry_run": True,
                        "action": "bid",
                        "market_item_id": market_item_id,
                        "amount": int(amount),
                        "player_id": int(player_id) if player_id else None,
                    }
                )
            try:
                client = runtime.get_client()
                client.buy_player_bid(
                    market_player_id=market_item_id,
                    amount=int(amount),
                    player_id=int(player_id) if player_id else None,
                )
                runtime.invalidate()
                finish_langsmith_span(
                    span,
                    {"ok": True, "market_item_id": market_item_id, "amount": int(amount)},
                )
                return _as_json({"ok": True, "market_item_id": market_item_id, "amount": int(amount)})
            except Exception as exc:
                finish_langsmith_span(
                    span,
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "market_item_id": market_item_id},
                )
                return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def buyout_player_tool(player_team_id: str, clause_to_pay: int = 0) -> str:
        """Ejecuta un clausulazo sobre un `player_team_id` rival. Prohibido desde 24h antes de jornada."""
        with langsmith_run_span(
            _tool_run_name("buyout_player"),
            run_type="tool",
            phase=runtime.phase,
            command="agent-tool",
            league_id=runtime.league_id,
            dry_run=runtime.dry_run,
            extra_metadata={"tool_name": "buyout_player_tool"},
            inputs={
                "player_team_id": str(player_team_id or "").strip(),
                "clause_to_pay": int(clause_to_pay) if clause_to_pay else None,
            },
        ) as span:
            blocked = _block_if_post("buyout_player_tool")
            if blocked:
                finish_langsmith_span(span, {"ok": False, "blocked": True})
                return blocked
            blocked = _block_if_buyout_locked("buyout_player_tool")
            if blocked:
                finish_langsmith_span(span, {"ok": False, "blocked": True, "window_locked": True})
                return blocked
            if runtime.dry_run:
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "dry_run": True,
                        "player_team_id": player_team_id,
                        "clause_to_pay": int(clause_to_pay) if clause_to_pay else None,
                    },
                )
                return _as_json(
                    {
                        "dry_run": True,
                        "action": "buyout",
                        "player_team_id": player_team_id,
                        "clause_to_pay": int(clause_to_pay) if clause_to_pay else None,
                    }
                )
            try:
                client = runtime.get_client()
                client.buy_player_clausulazo(
                    player_team_id=player_team_id,
                    buyout_clause_to_pay=int(clause_to_pay) if clause_to_pay else None,
                )
                runtime.invalidate()
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "player_team_id": player_team_id,
                        "clause_to_pay": int(clause_to_pay) if clause_to_pay else None,
                    },
                )
                return _as_json(
                    {
                        "ok": True,
                        "player_team_id": player_team_id,
                        "clause_to_pay": int(clause_to_pay) if clause_to_pay else None,
                    }
                )
            except Exception as exc:
                finish_langsmith_span(
                    span,
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "player_team_id": player_team_id},
                )
                return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def increase_clause_tool(
        player_team_id: str,
        value_to_increase: int,
        force: bool = False,
    ) -> str:
        """
        Aumenta la cláusula de un jugador propio.

        Regla económica: 1M invertido sube 2M de cláusula (factor fijo 2.0).
        Debe usarse de forma moderada: prioriza jugadores clave y expuestos
        (valor de mercado cercano a la cláusula).
        """
        with langsmith_run_span(
            _tool_run_name("increase_clause"),
            run_type="tool",
            phase=runtime.phase,
            command="agent-tool",
            league_id=runtime.league_id,
            dry_run=runtime.dry_run,
            extra_metadata={"tool_name": "increase_clause_tool"},
            inputs={
                "player_team_id": str(player_team_id or "").strip(),
                "value_to_increase": _safe_int(value_to_increase, 0),
                "force": bool(force),
            },
        ) as span:
            blocked = _block_if_post("increase_clause_tool")
            if blocked:
                finish_langsmith_span(span, {"ok": False, "blocked": True})
                return blocked
            ptid = str(player_team_id or "").strip()
            amount = _safe_int(value_to_increase, 0)
            if not ptid:
                finish_langsmith_span(span, {"ok": False, "error": "player_team_id vacío"})
                return _as_json({"ok": False, "error": "player_team_id vacío"})
            if amount <= 0:
                finish_langsmith_span(span, {"ok": False, "error": "value_to_increase debe ser > 0"})
                return _as_json({"ok": False, "error": "value_to_increase debe ser > 0"})

            team_analysis = runtime.get_team_analysis(force_refresh=False)
            players = team_analysis.get("jugadores", [])
            if not isinstance(players, list):
                players = []

            selected = None
            sorted_players = sorted(
                [p for p in players if isinstance(p, dict)],
                key=lambda x: _safe_float(x.get("xP"), 0.0),
                reverse=True,
            )
            for p in sorted_players:
                if str(p.get("player_team_id", "")).strip() == ptid:
                    selected = p
                    break

            if not selected:
                finish_langsmith_span(span, {"ok": False, "player_team_id": ptid, "found": False})
                return _as_json(
                    {
                        "ok": False,
                        "error": "No se encontró ese player_team_id en tu plantilla actual.",
                        "player_team_id": ptid,
                    }
                )

            key_player_team_ids = {
                str(p.get("player_team_id", "")).strip()
                for p in sorted_players[:7]
                if str(p.get("player_team_id", "")).strip()
            }
            xp_value = _safe_float(selected.get("xP"), 0.0)
            market_value = _safe_int(selected.get("valor_mercado"), 0)
            clause_value = _safe_int(selected.get("clausula"), 0)
            ratio = _clause_exposure_ratio(market_value, clause_value)
            exposed = ratio is not None and ratio >= CLAUSE_EXPOSURE_THRESHOLD
            is_key = ptid in key_player_team_ids
            increase_delta = int(round(amount * CLAUSE_INCREASE_FACTOR))
            estimated_new_clause = clause_value + increase_delta if clause_value > 0 else None

            if not force and (not is_key or not exposed):
                finish_langsmith_span(
                    span,
                    {
                        "ok": False,
                        "blocked": True,
                        "player_team_id": ptid,
                        "jugador_clave": is_key,
                        "expuesto_clausulazo": exposed,
                    },
                )
                return _as_json(
                    {
                        "ok": False,
                        "blocked": True,
                        "player_team_id": ptid,
                        "nombre": selected.get("nombre"),
                        "xP": xp_value,
                        "valor_mercado": market_value,
                        "clausula_actual": clause_value,
                        "ratio_valor_vs_clausula": ratio,
                        "regla_moderacion": {
                            "jugador_clave": is_key,
                            "expuesto_clausulazo": exposed,
                            "umbral_exposicion": CLAUSE_EXPOSURE_THRESHOLD,
                        },
                        "error": (
                            "Bloqueado por moderación: solo subir cláusula en jugadores clave "
                            "y con valor de mercado cercano a cláusula. Usa force=true solo si hay justificación."
                        ),
                    }
                )

            if runtime.dry_run:
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "dry_run": True,
                        "player_team_id": ptid,
                        "jugador_clave": is_key,
                        "expuesto_clausulazo": exposed,
                        "value_to_increase": amount,
                    },
                )
                return _as_json(
                    {
                        "dry_run": True,
                        "action": "increase_clause",
                        "player_team_id": ptid,
                        "nombre": selected.get("nombre"),
                        "value_to_increase": amount,
                        "factor": CLAUSE_INCREASE_FACTOR,
                        "estimated_clause_increase": increase_delta,
                        "clausula_actual": clause_value,
                        "clausula_estimada_nueva": estimated_new_clause,
                        "ratio_valor_vs_clausula": ratio,
                        "jugador_clave": is_key,
                        "expuesto_clausulazo": exposed,
                    }
                )

            try:
                client = runtime.get_client()
                client.increase_player_clause(
                    player_team_id=ptid,
                    value_to_increase=amount,
                    factor=CLAUSE_INCREASE_FACTOR,
                )
                runtime.invalidate()
                finish_langsmith_span(
                    span,
                    {
                        "ok": True,
                        "player_team_id": ptid,
                        "value_to_increase": amount,
                        "estimated_clause_increase": increase_delta,
                        "clausula_estimada_nueva": estimated_new_clause,
                    },
                )
                return _as_json(
                    {
                        "ok": True,
                        "player_team_id": ptid,
                        "value_to_increase": amount,
                        "factor": CLAUSE_INCREASE_FACTOR,
                        "estimated_clause_increase": increase_delta,
                        "clausula_estimada_nueva": estimated_new_clause,
                    }
                )
            except Exception as exc:
                finish_langsmith_span(
                    span,
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "player_team_id": ptid},
                )
                return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def current_lineup() -> str:
        """Obtiene la alineación actualmente guardada en la API.
        Úsala para verificar: formación, titulares, capitán y si hay posiciones vacías.
        Recuerda: -4 pt por posición vacía y el capitán duplica sus puntos."""
        try:
            snapshot = runtime.get_snapshot(force_refresh=False)
            team_id = str(snapshot.get("mi_equipo", {}).get("team_id", ""))
            if not team_id:
                return _as_json({"ok": False, "error": "No se pudo obtener team_id"})
            client = runtime.get_client()
            lineup = client.get_team_lineup(team_id=team_id)
            return _as_json({"ok": True, "team_id": team_id, "lineup": lineup})
        except Exception as exc:
            return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return [
        snapshot_summary,
        my_squad,
        predictions_top,
        player_outlook,
        market_opportunities,
        news_reader_tool,
        simulate_transfer_plan,
        execute_simulated_plan,
        accept_closed_offers,
        autoset_best_lineup_tool,
        sell_player_phase1_tool,
        place_bid_tool,
        buyout_player_tool,
        increase_clause_tool,
        current_lineup,
    ]
