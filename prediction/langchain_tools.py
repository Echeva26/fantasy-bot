"""
LangChain tools para gestionar LaLiga Fantasy.

Este módulo envuelve la lógica existente del repositorio en herramientas
estructuradas para un agente LLM.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd

from laliga_fantasy_client import LaLigaFantasyClient
from prediction.advisor import (
    analyze_available_players,
    analyze_my_team,
    clausulazos_available,
    get_predictions,
    load_snapshot,
    simulate_transfer_plan,
)
from prediction.advisor_execute import execute_movements, run_aceptar_ofertas
from prediction.lineup_autoset import autoset_best_lineup

logger = logging.getLogger(__name__)
MODEL_TYPE = "xgboost"


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


@dataclass
class FantasyAgentRuntime:
    league_id: str
    model_type: str = MODEL_TYPE
    dry_run: bool = False
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
            claus_ok, hours_to_match = clausulazos_available(first_match_ts)
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
                "precio_publicacion": venta.get("precio_publicacion"),
                "motivo": venta.get("motivo"),
                "xP": venta.get("xP"),
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

    @tool
    def snapshot_summary(force_refresh: bool = False) -> str:
        """Obtiene resumen del estado actual de la liga y del equipo del manager."""
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
        """Lista jugadores de mi plantilla con ids útiles para mercado y alineación."""
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
        """Muestra predicciones xP top para la próxima jornada. `position` opcional: POR/DEF/MED/DEL."""
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
        """
        Busca información de un jugador por nombre o player_id:
        rival próximo, local/visitante, xP y estado.
        """
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
        """Devuelve oportunidades de compra del mercado (pujas) y clausulazos disponibles."""
        available = runtime.get_available(force_refresh=force_refresh)
        _, first_match_ts = runtime.get_predictions(force_refresh=force_refresh)
        claus_ok, hours_to_match = clausulazos_available(first_match_ts)

        items = []
        for p in available.get("mercado", []):
            coste = max(p.get("precio_venta", 0) or 0, p.get("valor_mercado", 0) or 0)
            items.append(
                {
                    "tipo": "mercado",
                    "player_id": p.get("player_id"),
                    "market_item_id": p.get("market_item_id"),
                    "nombre": p.get("nombre"),
                    "posicion": p.get("posicion"),
                    "equipo_real": p.get("equipo_real"),
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
                "clausulazos_available": claus_ok,
                "items": items[:n],
            }
        )

    @tool
    def simulate_transfer_plan(force_refresh: bool = False) -> str:
        """Simula plan de ventas/compras recomendado por el motor actual del repositorio."""
        team_analysis = runtime.get_team_analysis(force_refresh=force_refresh)
        plan, claus_ok, hours_to_match = runtime.get_transfer_plan(force_refresh=force_refresh)
        payload = {
            "clausulazos_available": claus_ok,
            "hours_to_first_match": round(hours_to_match, 1),
            "summary": {
                "xp_once_actual": team_analysis.get("xp_total_once"),
                "xp_once_post": plan.get("xp_total_post"),
                "xp_delta": round((plan.get("xp_total_post", 0) or 0) - (team_analysis.get("xp_total_once", 0) or 0), 1),
                "saldo_actual": team_analysis.get("saldo"),
                "saldo_final": plan.get("saldo_final"),
                "movimientos": len(plan.get("movimientos", [])),
            },
            "plan": _compress_transfer_plan(plan),
        }
        return _as_json(payload)

    @tool
    def execute_simulated_plan(force_refresh: bool = False) -> str:
        """Ejecuta el plan simulado (ventas fase1 + compras). Respeta `dry_run` del runtime."""
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
            return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def accept_closed_offers() -> str:
        """Ejecuta fase2 de ventas: aceptar ofertas de liga ya cerradas."""
        if runtime.dry_run:
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
            return _as_json({"accepted_offers": int(accepted), "dry_run": runtime.dry_run})
        except Exception as exc:
            return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def autoset_best_lineup_tool(
        day_before_only: bool = True,
        force: bool = False,
        after_market_time: str = "08:10",
    ) -> str:
        """Calcula y guarda automáticamente la mejor alineación por xP."""
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
            return _as_json(result)
        except Exception as exc:
            return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def sell_player_phase1_tool(player_team_id: str, sale_price: int) -> str:
        """Publica un jugador propio en mercado (fase1). Requiere `player_team_id`."""
        if runtime.dry_run:
            return _as_json(
                {
                    "dry_run": True,
                    "action": "sell_player_phase1",
                    "player_team_id": player_team_id,
                    "sale_price": int(sale_price),
                }
            )
        try:
            client = runtime.get_client()
            res = client.sell_player_phase1(player_team_id=player_team_id, price=int(sale_price))
            runtime.invalidate()
            return _as_json({"ok": True, "response": res})
        except Exception as exc:
            return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def place_bid_tool(market_item_id: str, amount: int, player_id: int = 0) -> str:
        """Hace/actualiza una puja en mercado libre."""
        if runtime.dry_run:
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
            res = client.buy_player_bid(
                market_player_id=market_item_id,
                amount=int(amount),
                player_id=int(player_id) if player_id else None,
            )
            runtime.invalidate()
            return _as_json({"ok": True, "response": res})
        except Exception as exc:
            return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def buyout_player_tool(player_team_id: str, clause_to_pay: int = 0) -> str:
        """Ejecuta un clausulazo sobre un `player_team_id` rival."""
        if runtime.dry_run:
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
            res = client.buy_player_clausulazo(
                player_team_id=player_team_id,
                buyout_clause_to_pay=int(clause_to_pay) if clause_to_pay else None,
            )
            runtime.invalidate()
            return _as_json({"ok": True, "response": res})
        except Exception as exc:
            return _as_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    @tool
    def current_lineup() -> str:
        """Obtiene la alineación actual guardada en la API para mi equipo."""
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
        simulate_transfer_plan,
        execute_simulated_plan,
        accept_closed_offers,
        autoset_best_lineup_tool,
        sell_player_phase1_tool,
        place_bid_tool,
        buyout_player_tool,
        current_lineup,
    ]
