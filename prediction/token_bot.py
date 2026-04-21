"""
Telegram Token Bot
==================
Recibe tokens por Telegram y actualiza .laliga_token.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from urllib.parse import parse_qs, urlparse

import requests

from laliga_fantasy_client import LaLigaFantasyClient, TOKEN_FILE, load_token, save_token
from prediction.advisor import (
    CLAUSULAZO_LOCKOUT_HOURS,
    current_week_clausulazos_available,
)
from prediction.advisor_execute import run_aceptar_ofertas
from prediction.langchain_agent import run_agent_objective
from prediction.league_selection import (
    load_selected_league,
    resolve_league_id,
    save_selected_league,
)
from prediction.lineup_autoset import autoset_best_lineup
from prediction.market_schedule import build_market_schedule, schedule_message
from prediction.telegram_messages import (
    TELEGRAM_PARSE_MODE,
    build_error_message,
    build_help_message,
    build_progress_message,
    build_standard_message,
)
from prediction.telegram_notify import GUIDA_RENOVACION_TOKEN, send_telegram_message

logger = logging.getLogger(__name__)

JWT_RE = re.compile(r"(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)")
REPORT_PLAN_CACHE_FILE = Path(".langchain_last_report_plan.json")
EXECUTABLE_TOOLS = {
    "sell_player_phase1_tool",
    "place_bid_tool",
    "buyout_player_tool",
    "increase_clause_tool",
}
REPORT_PLAN_OBJECTIVE = (
    "Genera el informe del ciclo actual y un plan EJECUTABLE de compraventa.\n"
    "Reglas obligatorias:\n"
    "1) Usa snapshot_summary, my_squad, market_opportunities, news_reader_tool y simulate_transfer_plan.\n"
    "2) Simula EXACTAMENTE las operaciones finales llamando las herramientas:\n"
    "   sell_player_phase1_tool, place_bid_tool, buyout_player_tool, increase_clause_tool.\n"
    "3) Respeta el orden real de ejecución y no uses accept_closed_offers.\n"
    "4) Como dry_run está activo, no habrá cambios reales ahora.\n"
    "5) Si propones subir cláusula, sé moderado: solo jugadores clave y expuestos.\n"
    "   Regla fija: cada 1M invertido sube 2M la cláusula.\n"
    "6) Si recomiendas proteger jugadores, DEBES materializarlo llamando increase_clause_tool.\n"
    "7) REGLA CRÍTICA: no se puede comprar ningún jugador mediante clausulazo "
    "desde 24h antes del inicio de la jornada. Si faltan 24h o menos para el "
    "primer partido, NO llames buyout_player_tool y descarta todos los "
    "clausulazos; usa solo pujas de mercado, ventas o subidas de cláusula.\n"
    "8) Devuelve resumen breve y claro en español."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return int(default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _tool_input_dict(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
            if isinstance(val, dict):
                return val
        except Exception:
            return {}
    return {}


def _money_short(value: object) -> str:
    amount = _safe_int(value, 0)
    sign = "-" if amount < 0 else ""
    n = abs(amount)
    if n >= 1_000_000_000:
        return f"{sign}{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{sign}{n / 1_000:.1f}K"
    return f"{sign}{n}"


def _compact_text(text: object, limit: int = 280) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _json_dict_from_text(text: object) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {}

    direct = _tool_input_dict(raw)
    if direct:
        return direct

    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
        except Exception:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)

    if not candidates:
        return {}

    for obj in reversed(candidates):
        if "decision_general" in obj:
            return obj
    for obj in reversed(candidates):
        if "plan" in obj and "summary" in obj:
            return obj
    return candidates[-1]


def _format_action_label(tool_name: str, payload: dict) -> str:
    tool_name = str(tool_name or "").strip()
    name = str(payload.get("nombre", "")).strip()
    if tool_name == "sell_player_phase1_tool":
        player_ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        return f"Vender {player_ref} por {_money_short(payload.get('sale_price', 0))}"
    if tool_name == "place_bid_tool":
        player_ref = name or f"market_item_id={payload.get('market_item_id', '?')}"
        return f"Pujar por {player_ref} con {_money_short(payload.get('amount', 0))}"
    if tool_name == "buyout_player_tool":
        player_ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        clause = payload.get("clause_to_pay")
        if clause in (None, "", 0, "0"):
            return f"Clausulazo a {player_ref}"
        return f"Clausulazo a {player_ref} pagando {_money_short(clause)}"
    if tool_name == "increase_clause_tool":
        player_ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        invest = _safe_int(payload.get("value_to_increase"), 0)
        if invest <= 0:
            return f"Subir cláusula de {player_ref}"
        delta = invest * 2
        return (
            f"Subir cláusula de {player_ref} invirtiendo {_money_short(invest)} "
            f"(+{_money_short(delta)} de cláusula)"
        )
    return tool_name


def _extract_direct_tool_actions(steps: list[dict]) -> list[dict]:
    actions: list[dict] = []
    for step in steps:
        tool_name = str(step.get("tool", "")).strip()
        if tool_name not in EXECUTABLE_TOOLS:
            continue
        obs = _json_dict_from_text(step.get("observation"))
        if isinstance(obs, dict):
            if bool(obs.get("blocked")):
                continue
            if obs.get("ok") is False and not bool(obs.get("dry_run")):
                continue
        payload = _tool_input_dict(step.get("tool_input"))
        action = {
            "tool": tool_name,
            "tool_input": payload,
        }
        label = _format_action_label(tool_name, payload)
        if label:
            action["label"] = label
        actions.append(action)
    return actions


def _extract_latest_simulation_payload(steps: list[dict]) -> dict:
    for step in reversed(steps):
        if str(step.get("tool", "")).strip() != "simulate_transfer_plan":
            continue
        payload = _json_dict_from_text(step.get("observation"))
        if isinstance(payload.get("plan"), dict):
            return payload
    return {}


def _extract_actions_from_simulation_payload(payload: dict) -> list[dict]:
    actions: list[dict] = []
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        return actions
    movimientos = plan.get("movimientos")
    if not isinstance(movimientos, list):
        return actions

    for mov in movimientos:
        if not isinstance(mov, dict):
            continue

        venta = mov.get("venta")
        if isinstance(venta, dict):
            player_team_id = str(venta.get("player_team_id", "")).strip()
            sale_price = max(
                _safe_int(venta.get("precio_publicacion"), 0),
                _safe_int(venta.get("valor_mercado"), 0),
            )
            if player_team_id and sale_price > 0:
                payload_sell = {
                    "player_team_id": player_team_id,
                    "sale_price": sale_price,
                    "nombre": str(venta.get("nombre", "")).strip(),
                }
                actions.append(
                    {
                        "tool": "sell_player_phase1_tool",
                        "tool_input": payload_sell,
                        "label": _format_action_label("sell_player_phase1_tool", payload_sell),
                    }
                )

        compra = mov.get("compra")
        if not isinstance(compra, dict):
            continue
        tipo = str(compra.get("tipo", "")).strip().lower()
        nombre = str(compra.get("nombre", "")).strip()
        if tipo == "clausulazo":
            player_team_id = str(compra.get("player_team_id", "")).strip()
            if not player_team_id:
                continue
            payload_buyout = {
                "player_team_id": player_team_id,
                "nombre": nombre,
            }
            clause = _safe_int(compra.get("coste"), 0)
            if clause > 0:
                payload_buyout["clause_to_pay"] = clause
            actions.append(
                {
                    "tool": "buyout_player_tool",
                    "tool_input": payload_buyout,
                    "label": _format_action_label("buyout_player_tool", payload_buyout),
                }
            )
            continue

        market_item_id = str(compra.get("market_item_id", "")).strip()
        amount = _safe_int(compra.get("coste"), 0)
        if not market_item_id or amount <= 0:
            continue
        payload_bid = {
            "market_item_id": market_item_id,
            "amount": amount,
            "nombre": nombre,
        }
        player_id = _safe_int(compra.get("player_id"), 0)
        if player_id > 0:
            payload_bid["player_id"] = player_id
        actions.append(
            {
                "tool": "place_bid_tool",
                "tool_input": payload_bid,
                "label": _format_action_label("place_bid_tool", payload_bid),
            }
        )

    return actions


def _extract_executable_actions(steps: list[dict]) -> tuple[list[dict], str]:
    direct_actions = _extract_direct_tool_actions(steps)
    if direct_actions:
        return direct_actions, "tool_calls"

    simulation_payload = _extract_latest_simulation_payload(steps)
    simulated_actions = _extract_actions_from_simulation_payload(simulation_payload)
    if simulated_actions:
        return simulated_actions, "simulate_transfer_plan"

    return [], "none"


def _extract_agent_report_payload(output: str) -> dict:
    payload = _json_dict_from_text(output)
    if not payload:
        return {}
    if "decision_general" in payload:
        return payload
    return {}


def _report_recommends_clause_protection(report_payload: dict) -> bool:
    decision = str(report_payload.get("decision_general", "")).lower()
    siguiente = str(report_payload.get("siguiente_revision_recomendada", "")).lower()
    riesgos = report_payload.get("riesgos_detectados", [])
    riesgos_txt = " ".join(str(r or "") for r in riesgos) if isinstance(riesgos, list) else ""
    riesgos_txt = riesgos_txt.lower()

    all_text = " ".join([decision, siguiente, riesgos_txt]).strip()
    if not all_text:
        return False

    has_clause_word = any(w in all_text for w in ("clausula", "cláusula", "clausul", "buyout"))
    has_action_word = any(w in all_text for w in ("aument", "sub", "proteg", "blind", "evaluar"))
    has_exposure_signal = any(
        w in all_text
        for w in ("exposicion a clausulazo", "exposición a clausulazo", "ratio valor_vs_clausula", "muy vulnerable")
    )

    return (has_clause_word and has_action_word) or (has_clause_word and has_exposure_signal)


def _extract_clause_names_from_report(report_payload: dict) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    decision_names: list[str] = []

    def _push(name_raw: object) -> None:
        name = str(name_raw or "").strip()
        if not name:
            return
        name = re.sub(r"^[\-\d\.\)\s]+", "", name).strip()
        if "(" in name:
            name = name.split("(", 1)[0].strip()
        if not name:
            return
        key = _norm(name)
        if not key or key in seen:
            return
        seen.add(key)
        names.append(name)

    decision = str(report_payload.get("decision_general", "") or "")
    m = re.search(r"proteger\s+a\s+([^.;]+)", decision, flags=re.IGNORECASE)
    if m:
        chunk = m.group(1).replace(" y ", ",")
        for raw in chunk.split(","):
            _push(raw)
        decision_names = list(names)

    if decision_names:
        return decision_names

    riesgos = report_payload.get("riesgos_detectados", [])
    if isinstance(riesgos, list):
        for row in riesgos:
            txt = str(row or "").strip()
            if not txt:
                continue
            if "clausul" not in txt.lower():
                continue
            chunk = txt.split(":", 1)[1] if ":" in txt else txt
            for raw in chunk.split(","):
                _push(raw)

    return names


def _extract_my_squad_players_from_steps(steps: list[dict]) -> list[dict]:
    for step in reversed(steps):
        if str(step.get("tool", "")).strip() != "my_squad":
            continue
        payload = _json_dict_from_text(step.get("observation"))
        players = payload.get("players")
        if isinstance(players, list):
            return [p for p in players if isinstance(p, dict)]
    return []


def _recommended_clause_increase_value(player: dict) -> int:
    market_value = _safe_int(player.get("valor_mercado"), 0)
    clause_value = _safe_int(player.get("clausula"), 0)
    if market_value <= 0:
        return 1_000_000

    target_clause = int(market_value * 1.08)
    delta = max(0, target_clause - max(0, clause_value))
    invest = max(1_000_000, min(3_000_000, (delta + 1) // 2))
    # Redondeo a 100k para importes limpios.
    invest = ((invest + 99_999) // 100_000) * 100_000
    return int(invest)


def _build_clause_actions_from_report(report_payload: dict, steps: list[dict]) -> list[dict]:
    if not _report_recommends_clause_protection(report_payload):
        return []

    players = _extract_my_squad_players_from_steps(steps)
    if not players:
        return []

    by_name: dict[str, dict] = {}
    for p in players:
        ptid = str(p.get("player_team_id", "")).strip()
        name = str(p.get("nombre", "")).strip()
        if name:
            by_name[_norm(name)] = p

    names = _extract_clause_names_from_report(report_payload)
    selected: list[dict] = []
    selected_ptids: set[str] = set()

    for name in names:
        p = by_name.get(_norm(name))
        if not p:
            continue
        ptid = str(p.get("player_team_id", "")).strip()
        if not ptid or ptid in selected_ptids:
            continue
        selected.append(p)
        selected_ptids.add(ptid)
        if len(selected) >= 3:
            break

    target_recommended = 2
    if not selected:
        exposed = sorted(
            players,
            key=lambda x: _safe_float(x.get("ratio_valor_vs_clausula"), 0.0),
            reverse=True,
        )
        for p in exposed:
            ratio = _safe_float(p.get("ratio_valor_vs_clausula"), 0.0)
            if ratio < 0.88:
                continue
            ptid = str(p.get("player_team_id", "")).strip()
            if not ptid or ptid in selected_ptids:
                continue
            selected.append(p)
            selected_ptids.add(ptid)
            if len(selected) >= target_recommended:
                break
    elif len(selected) < target_recommended:
        exposed = sorted(
            players,
            key=lambda x: _safe_float(x.get("ratio_valor_vs_clausula"), 0.0),
            reverse=True,
        )
        for p in exposed:
            ratio = _safe_float(p.get("ratio_valor_vs_clausula"), 0.0)
            if ratio < 0.88:
                continue
            ptid = str(p.get("player_team_id", "")).strip()
            if not ptid or ptid in selected_ptids:
                continue
            selected.append(p)
            selected_ptids.add(ptid)
            if len(selected) >= target_recommended:
                break

    actions: list[dict] = []
    for p in selected:
        ptid = str(p.get("player_team_id", "")).strip()
        if not ptid:
            continue
        payload = {
            "player_team_id": ptid,
            "value_to_increase": _recommended_clause_increase_value(p),
            "nombre": str(p.get("nombre", "")).strip(),
        }
        actions.append(
            {
                "tool": "increase_clause_tool",
                "tool_input": payload,
                "label": _format_action_label("increase_clause_tool", payload),
            }
        )
    return actions


def _format_tools_used(steps: list[dict], limit: int = 8) -> str:
    tool_names: list[str] = []
    for s in steps:
        name = str(s.get("tool", "")).strip()
        if name and name not in tool_names:
            tool_names.append(name)
    if not tool_names:
        return "sin tools registradas"
    out = ", ".join(tool_names[:limit])
    if len(tool_names) > limit:
        out += ", ..."
    return out


def _build_informe_message(
    *,
    league_name: str,
    market_key: str,
    steps: list[dict],
    output: str,
    actions: list[dict],
    action_source: str,
    simulation_payload: dict,
) -> str:
    report_payload = _extract_agent_report_payload(output)
    decision = _compact_text(report_payload.get("decision_general", ""), 420)
    if not decision:
        decision = _compact_text(output, 420)

    riesgos = report_payload.get("riesgos_detectados", [])
    if not isinstance(riesgos, list):
        riesgos = []
    riesgos_txt = [_compact_text(r, 180) for r in riesgos[:3] if str(r).strip()]

    siguiente = _compact_text(report_payload.get("siguiente_revision_recomendada", ""), 220)
    summary = simulation_payload.get("summary", {}) if isinstance(simulation_payload, dict) else {}
    if not isinstance(summary, dict):
        summary = {}

    source_txt = {
        "tool_calls": "tools ejecutables del agente",
        "simulate_transfer_plan": "plan de simulacion del motor",
        "simulate_transfer_plan+report_clause_fallback": "plan de simulacion + clausulas sugeridas",
        "tool_calls+report_clause_fallback": "tools ejecutables + clausulas sugeridas",
        "none": "sin plan ejecutable detectado",
    }.get(action_source, action_source)

    summary_rows = [
        "Modo: simulación (dry-run)",
        f"Plan cacheado: {len(actions)} acción(es) ejecutable(s)",
        f"Fuente del plan: {source_txt}",
    ]

    xp_now = summary.get("xp_once_actual")
    xp_post = summary.get("xp_once_post")
    xp_delta = summary.get("xp_delta")
    saldo_now = summary.get("saldo_actual")
    saldo_final = summary.get("saldo_final")
    movs = summary.get("movimientos")
    has_summary = any(v not in (None, "") for v in (xp_now, xp_post, saldo_now, saldo_final, movs))
    sim_rows: list[str] = []
    if has_summary:
        if xp_now not in (None, "") and xp_post not in (None, ""):
            delta = _safe_float(xp_delta, _safe_float(xp_post) - _safe_float(xp_now))
            delta_txt = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            sim_rows.append(f"xP once: {_safe_float(xp_now):.1f} → {_safe_float(xp_post):.1f} ({delta_txt})")
        if saldo_now not in (None, "") and saldo_final not in (None, ""):
            sim_rows.append(f"Saldo: {_money_short(saldo_now)} → {_money_short(saldo_final)}")
        if movs not in (None, ""):
            sim_rows.append(f"Movimientos simulados: {_safe_int(movs, 0)}")

    action_rows: list[str] = []
    if actions:
        for idx, action in enumerate(actions[:8], 1):
            payload = _tool_input_dict(action.get("tool_input"))
            label = str(action.get("label", "")).strip() or _format_action_label(
                str(action.get("tool", "")).strip(), payload
            )
            action_rows.append(f"{idx}. {label}")
        if len(actions) > 8:
            action_rows.append(f"+{len(actions) - 8} acción(es) más")
    else:
        action_rows = [
            "No se detectaron acciones ejecutables en este informe.",
            "Si el texto recomienda operar, vuelve a lanzar /informe para regenerar el plan del ciclo.",
        ]

    sections: list[tuple[str, list[str] | str]] = [
        ("Resumen", summary_rows),
    ]
    if sim_rows:
        sections.append(("Simulación", sim_rows))
    sections.append(("Acciones propuestas", action_rows))
    if decision:
        sections.append(("Decisión IA", decision))
    if riesgos_txt:
        sections.append(("Riesgos clave", riesgos_txt))
    if siguiente:
        sections.append(("Siguiente revisión", siguiente))
    sections.append(("Técnico", [f"Tools usadas: {len(steps)}", f"Tools: {_format_tools_used(steps)}"]))

    return build_standard_message(
        title="📊 Informe IA",
        status="Plan preparado",
        league_name=league_name,
        market_key=market_key,
        sections=sections,
        footer="Usa /compraventa para ejecutar este plan en el mismo ciclo.",
    )


def _build_compraventa_message(
    *,
    league_name: str,
    market_key: str,
    action_source: str,
    summary: dict,
) -> str:
    source_txt = {
        "tool_calls": "tools ejecutables del agente",
        "simulate_transfer_plan": "plan de simulacion cacheado",
        "simulate_transfer_plan+report_clause_fallback": "plan cacheado + clausulas sugeridas",
        "tool_calls+report_clause_fallback": "tools ejecutables + clausulas sugeridas",
        "none": "origen no informado",
    }.get(str(action_source), str(action_source or "none"))

    result_rows = [
        f"Fuente del plan: {source_txt}",
        f"Acciones planificadas: {summary.get('actions_total', 0)}",
        f"Acciones OK: {summary.get('actions_ok', 0)}",
        f"Acciones saltadas: {summary.get('actions_skipped', 0)}",
        f"Ventas fase 1: {summary.get('ventas', 0)}",
        f"Pujas: {summary.get('pujas', 0)}",
        f"Clausulazos: {summary.get('clausulazos', 0)}",
        f"Subidas de cláusula: {summary.get('clausulas_subidas', 0)}",
    ]

    saldo_actual = summary.get("saldo_actual")
    saldo_restante = summary.get("saldo_restante_estimado")
    balance_rows: list[str] = []
    if saldo_actual is not None:
        source = str(summary.get("saldo_source", "")).strip()
        source_txt_balance = "API" if source == "api" else "cache del informe" if source == "cache" else source
        balance_rows.extend(
            [
                f"Saldo validado: {_money_short(saldo_actual)}"
                + (f" ({source_txt_balance})" if source_txt_balance else ""),
                f"Saldo restante estimado: {_money_short(saldo_restante)}",
            ]
        )
    if summary.get("saldo_warning"):
        balance_rows.append(f"Aviso saldo: {_compact_text(summary.get('saldo_warning'), 220)}")

    buyout_window = summary.get("clausulazos_window", {})
    rule_rows: list[str] = []
    if isinstance(buyout_window, dict) and not bool(buyout_window.get("available", True)):
        hours = buyout_window.get("hours_to_first_match")
        hours_txt = (
            f" Quedan {_safe_float(hours):.1f}h para el primer partido."
            if hours not in (None, "")
            else ""
        )
        skipped = _safe_int(summary.get("clausulazos_bloqueados_24h"), 0)
        rule_rows.append(
            "Regla clausulazos: BLOQUEADOS desde 24h antes de jornada."
            f"{hours_txt}"
        )
        if skipped:
            rule_rows.append(f"Clausulazos saltados por regla 24h: {skipped}")

    details = summary.get("details", [])
    detail_rows: list[str] = []
    if isinstance(details, list) and details:
        for row in details[:10]:
            detail_rows.append(str(row))
        if len(details) > 10:
            detail_rows.append(f"+{len(details) - 10} línea(s) más")

    errors = summary.get("errors", [])
    error_rows: list[str] = []
    if isinstance(errors, list) and errors:
        for err in errors[:8]:
            error_rows.append(_compact_text(err, 220))
        if len(errors) > 8:
            error_rows.append(f"+{len(errors) - 8} error(es) más")

    sections: list[tuple[str, list[str] | str]] = [("Resultado", result_rows)]
    if balance_rows:
        sections.append(("Saldo", balance_rows))
    if rule_rows:
        sections.append(("Reglas aplicadas", rule_rows))
    if detail_rows:
        sections.append(("Detalle", detail_rows))
    if error_rows:
        sections.append((f"Errores ({len(errors)})", error_rows))

    status = "Ejecutada con errores" if error_rows else "Plan ejecutado"
    return build_standard_message(
        title="💸 Compraventa IA",
        status=status,
        league_name=league_name,
        market_key=market_key,
        sections=sections,
    )

def _market_key_for_league(league_id: str) -> tuple[str, str]:
    sched, err = build_market_schedule(
        league_id,
        timezone_name=os.getenv("TZ", "Europe/Madrid"),
    )
    if not sched:
        return "", err
    return str(sched.get("market_key", "")).strip(), ""


def _load_report_plan_cache(path: Path = REPORT_PLAN_CACHE_FILE) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _save_report_plan_cache(payload: dict, path: Path = REPORT_PLAN_CACHE_FILE) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _action_cash_cost(tool: str, payload: dict) -> int:
    if tool == "place_bid_tool":
        return _safe_int(payload.get("amount"), 0)
    if tool == "buyout_player_tool":
        return _safe_int(payload.get("clause_to_pay"), 0)
    if tool == "increase_clause_tool":
        return _safe_int(payload.get("value_to_increase"), 0)
    return 0


def _resolve_current_balance(
    client: LaLigaFantasyClient,
    fallback_balance: object = None,
) -> tuple[int | None, str, str]:
    try:
        return int(client.get_my_team_money()), "api", ""
    except Exception as exc:
        fallback = _safe_int(fallback_balance, -1)
        if fallback >= 0:
            return fallback, "cache", f"{type(exc).__name__}: {exc}"
        return None, "unknown", f"{type(exc).__name__}: {exc}"


def _buyout_window_for_execution() -> dict:
    available, hours_to_match, source = current_week_clausulazos_available(
        fail_closed=True,
    )

    if available:
        return {
            "available": True,
            "hours_to_first_match": round(float(hours_to_match), 1),
            "source": source,
            "reason": "",
        }

    return {
        "available": False,
        "hours_to_first_match": round(float(hours_to_match), 1),
        "source": source,
        "reason": (
            f"Regla 24h: LaLiga Fantasy no permite comprar mediante clausulazo "
            f"desde {CLAUSULAZO_LOCKOUT_HOURS}h antes del primer partido de la jornada."
        ),
    }


def _execute_cached_actions(
    league_id: str,
    actions: list[dict],
    *,
    fallback_balance: object = None,
) -> dict:
    client = LaLigaFantasyClient.from_saved_token(league_id=league_id)
    current_balance, balance_source, balance_warning = _resolve_current_balance(
        client,
        fallback_balance=fallback_balance,
    )
    summary = {
        "actions_total": len(actions),
        "actions_ok": 0,
        "actions_skipped": 0,
        "ventas": 0,
        "pujas": 0,
        "clausulazos": 0,
        "clausulas_subidas": 0,
        "clausulazos_bloqueados_24h": 0,
        "clausulazos_window": {},
        "saldo_actual": current_balance,
        "saldo_source": balance_source,
        "saldo_warning": balance_warning,
        "details": [],
        "errors": [],
    }
    remaining_balance = current_balance
    buyout_window = (
        _buyout_window_for_execution()
        if any(str(a.get("tool", "")).strip() == "buyout_player_tool" for a in actions if isinstance(a, dict))
        else {"available": True, "hours_to_first_match": None, "reason": ""}
    )
    summary["clausulazos_window"] = buyout_window

    for idx, action in enumerate(actions, 1):
        tool = str(action.get("tool", "")).strip()
        payload = _tool_input_dict(action.get("tool_input"))
        label = str(action.get("label", "")).strip() or _format_action_label(tool, payload)
        cash_cost = _action_cash_cost(tool, payload)
        if tool == "buyout_player_tool" and not bool(buyout_window.get("available")):
            summary["actions_skipped"] += 1
            summary["clausulazos_bloqueados_24h"] += 1
            hours = buyout_window.get("hours_to_first_match")
            hours_txt = (
                f" Quedan {float(hours):.1f}h para el primer partido."
                if isinstance(hours, (int, float))
                else ""
            )
            summary["details"].append(
                f"[SKIP] {idx}. {label} -> clausulazos bloqueados por regla 24h."
                f"{hours_txt}"
            )
            continue
        if remaining_balance is not None and cash_cost > remaining_balance:
            summary["actions_skipped"] += 1
            summary["details"].append(
                f"[SKIP] {idx}. {label} -> saldo insuficiente "
                f"({_money_short(remaining_balance)} disponible, {_money_short(cash_cost)} requerido)"
            )
            continue
        try:
            if tool == "sell_player_phase1_tool":
                player_team_id = str(payload.get("player_team_id", "")).strip()
                sale_price = _safe_int(payload.get("sale_price"), 0)
                if not player_team_id or sale_price <= 0:
                    raise ValueError("faltan player_team_id o sale_price")
                client.sell_player_phase1(player_team_id=player_team_id, price=sale_price)
                summary["actions_ok"] += 1
                summary["ventas"] += 1
                summary["details"].append(f"[OK] {idx}. {label}")
                continue

            if tool == "place_bid_tool":
                market_item_id = str(payload.get("market_item_id", "")).strip()
                amount = _safe_int(payload.get("amount"), 0)
                player_id = payload.get("player_id")
                player_id_int = _safe_int(player_id, 0) if player_id not in (None, "") else 0
                if not market_item_id or amount <= 0:
                    raise ValueError("faltan market_item_id o amount")
                client.buy_player_bid(
                    market_player_id=market_item_id,
                    amount=amount,
                    player_id=player_id_int or None,
                )
                summary["actions_ok"] += 1
                summary["pujas"] += 1
                if remaining_balance is not None:
                    remaining_balance -= cash_cost
                summary["details"].append(f"[OK] {idx}. {label}")
                continue

            if tool == "buyout_player_tool":
                player_team_id = str(payload.get("player_team_id", "")).strip()
                clause = payload.get("clause_to_pay")
                clause_int = _safe_int(clause, 0) if clause not in (None, "") else 0
                if not player_team_id:
                    raise ValueError("falta player_team_id")
                client.buy_player_clausulazo(
                    player_team_id=player_team_id,
                    buyout_clause_to_pay=clause_int or None,
                )
                summary["actions_ok"] += 1
                summary["clausulazos"] += 1
                if remaining_balance is not None:
                    remaining_balance -= cash_cost
                summary["details"].append(f"[OK] {idx}. {label}")
                continue

            if tool == "increase_clause_tool":
                player_team_id = str(payload.get("player_team_id", "")).strip()
                value_to_increase = _safe_int(payload.get("value_to_increase"), 0)
                if not player_team_id or value_to_increase <= 0:
                    raise ValueError("faltan player_team_id o value_to_increase")
                client.increase_player_clause(
                    player_team_id=player_team_id,
                    value_to_increase=value_to_increase,
                    factor=2.0,
                )
                summary["actions_ok"] += 1
                summary["clausulas_subidas"] += 1
                if remaining_balance is not None:
                    remaining_balance -= cash_cost
                summary["details"].append(f"[OK] {idx}. {label}")
                continue

            msg = f"Paso {idx}: tool no soportada ({tool})"
            summary["errors"].append(msg)
            summary["details"].append(f"[ERROR] {idx}. {label} -> {msg}")
        except Exception as exc:
            msg = f"Paso {idx} ({tool}): {type(exc).__name__}: {exc}"
            summary["errors"].append(msg)
            summary["details"].append(f"[ERROR] {idx}. {label} -> {type(exc).__name__}: {exc}")

    summary["saldo_restante_estimado"] = remaining_balance
    return summary


def _api(bot_token: str, method: str, payload: dict | None = None) -> dict:
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    resp = requests.post(url, json=payload or {}, timeout=65)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data


def _extract_token_from_text(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None

    # 1) JWT directo
    match = JWT_RE.search(text)
    if match:
        return match.group(1)

    # 2) URL con fragment/query id_token/access_token
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        frag = parse_qs(parsed.fragment or "")
        qry = parse_qs(parsed.query or "")
        for src in (frag, qry):
            for key in ("id_token", "access_token"):
                values = src.get(key, [])
                if values and values[0].startswith("eyJ"):
                    return values[0]
    return None


def _token_age_hours() -> float | None:
    try:
        if not TOKEN_FILE.exists():
            return None
        data = TOKEN_FILE.read_text(encoding="utf-8")
        token = load_token()
        if not token:
            return None
        raw = json.loads(data)
        saved_at = datetime.fromisoformat(raw["saved_at"])
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600.0
        return age
    except Exception:
        return None


def _token_status(max_age_hours: float = 23.0) -> tuple[str, float | None]:
    """
    Devuelve (status, age_h) con status: "ok" | "missing" | "expired" | "invalid"
    """
    if not TOKEN_FILE.exists():
        return "missing", None
    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        saved_at_raw = data.get("saved_at")
        if not saved_at_raw:
            return "invalid", None
        saved_at = datetime.fromisoformat(saved_at_raw)
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600.0
        if age_h >= max_age_hours:
            return "expired", age_h
        return "ok", age_h
    except Exception:
        return "invalid", None


def _list_user_leagues() -> tuple[list[dict], str]:
    """
    Devuelve (ligas, error).
    ligas: [{"id": "...", "name": "..."}, ...]
    """
    status, _ = _token_status(max_age_hours=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")))
    if status != "ok":
        return [], f"token no válido ({status})"
    try:
        client = LaLigaFantasyClient.from_saved_token(league_id="")
        raw = client.get_leagues() or []
        leagues = []
        for lg in raw:
            lid = str(lg.get("id", "")).strip()
            if not lid:
                continue
            leagues.append(
                {
                    "id": lid,
                    "name": str(lg.get("name", "")).strip() or "(Sin nombre)",
                }
            )
        return leagues, ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _resolve_operational_league_id() -> tuple[str, str]:
    """
    Resuelve league_id para ejecutar comandos operativos.
    Si hay una sola liga en la cuenta, la selecciona automáticamente.
    """
    league_id = resolve_league_id("")
    if league_id:
        return league_id, ""

    leagues, err = _list_user_leagues()
    if err:
        return "", f"No se pudo resolver la liga: {err}"
    if not leagues:
        return "", "No se encontraron ligas para este usuario."
    if len(leagues) == 1:
        lg = leagues[0]
        save_selected_league(lg["id"], lg["name"], source="auto_single_league")
        return lg["id"], ""

    return (
        "",
        "No hay liga seleccionada. Usa /ligas para verlas y /liga <nombre> para elegir.",
    )


def _selected_league_text() -> str:
    selected = load_selected_league()
    if not selected:
        fallback = resolve_league_id("")
        if fallback:
            return "Liga activa: definida por configuracion avanzada."
        return "Liga seleccionada: (ninguna)"
    name = selected.get("league_name", "")
    if name:
        return f"Liga activa: {name}"
    return "Liga activa: seleccionada"


def _norm(text: str) -> str:
    s = (text or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return " ".join(s.split())


def _cmd_list_leagues() -> str:
    leagues, err = _list_user_leagues()
    if err:
        return build_error_message("No se pudieron listar ligas", err)
    if not leagues:
        return build_standard_message(
            title="🏆 Ligas",
            status="Sin resultados",
            sections=[("Resultado", ["No se encontraron ligas para este usuario."])],
        )

    selected_id = resolve_league_id("")
    rows = []
    for i, lg in enumerate(leagues, 1):
        mark = " · activa" if str(lg["id"]) == str(selected_id) else ""
        rows.append(f"{i}. {lg['name']}{mark}")
    return build_standard_message(
        title="🏆 Ligas disponibles",
        status=f"{len(leagues)} liga(s) encontradas",
        sections=[("Opciones", rows)],
        footer="Usa /liga <nombre> o /liga <número> para seleccionarla.",
    )


def _cmd_select_league(raw_text: str) -> str:
    parts = raw_text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        current = _selected_league_text()
        return build_standard_message(
            title="🏆 Liga activa",
            status="Sin cambios",
            sections=[("Estado", [current])],
            footer="Para seleccionar liga: /liga <nombre>. Para ver opciones: /ligas.",
        )
    target = parts[1].strip()

    leagues, err = _list_user_leagues()
    if err:
        return build_error_message("No se pudo seleccionar liga", err)
    if not leagues:
        return build_standard_message(
            title="🏆 Liga activa",
            status="Sin resultados",
            sections=[("Resultado", ["No se encontraron ligas para este usuario."])],
        )

    chosen = None
    if target.isdigit():
        idx = int(target)
        if 1 <= idx <= len(leagues):
            chosen = leagues[idx - 1]
    else:
        norm_target = _norm(target)
        exact = [lg for lg in leagues if _norm(lg.get("name", "")) == norm_target]
        if len(exact) == 1:
            chosen = exact[0]
        elif len(exact) > 1:
            return build_standard_message(
                title="🏆 Liga activa",
                status="Coincidencia múltiple",
                sections=[("Opciones", [lg["name"] for lg in exact[:10]])],
                footer="Escribe el nombre completo tal cual o usa el número de /ligas.",
            )
        else:
            partial = [lg for lg in leagues if norm_target in _norm(lg.get("name", ""))]
            if len(partial) == 1:
                chosen = partial[0]
            elif len(partial) > 1:
                return build_standard_message(
                    title="🏆 Liga activa",
                    status="Coincidencia múltiple",
                    sections=[("Opciones", [lg["name"] for lg in partial[:10]])],
                    footer="Escribe un nombre más específico o usa el número de /ligas.",
                )

    if not chosen:
        return build_standard_message(
            title="🏆 Liga activa",
            status="No encontrada",
            sections=[("Búsqueda", [f"No encontré la liga '{target}'."])],
            footer="Usa /ligas para ver opciones y luego /liga <nombre>.",
        )

    save_selected_league(chosen["id"], chosen["name"], source="telegram")
    sched, sched_err = build_market_schedule(
        chosen["id"],
        timezone_name=os.getenv("TZ", "Europe/Madrid"),
    )
    schedule_rows = (
        schedule_message(sched).splitlines()
        if sched
        else [f"No pude calcular la hora de mercado ahora: {sched_err}"]
    )
    return build_standard_message(
        title="🏆 Liga activa",
        status="Seleccionada correctamente",
        league_name=chosen["name"],
        sections=[("Horario", schedule_rows)],
    )


def _run_langchain_agent_cmd(
    *,
    bot_token: str,
    chat_id: str,
    dry_run: bool,
) -> str:
    """
    Ejecuta LangChain para informe (dry-run) o compraventa real (no dry-run).
    """
    send_telegram_message(
        bot_token,
        chat_id,
        build_progress_message(
            "📊 Informe IA" if dry_run else "💸 Compraventa IA",
            "Generando informe y preparando plan cacheado."
            if dry_run
            else "Ejecutando el plan cacheado del ciclo actual.",
        ),
        parse_mode=TELEGRAM_PARSE_MODE,
    )

    league_id, league_err = _resolve_operational_league_id()
    if not league_id:
        return build_error_message("Liga no disponible", league_err, footer="Usa /ligas y /liga <nombre> para seleccionar una liga.")

    status, _ = _token_status(max_age_hours=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")))
    if status != "ok":
        return build_error_message("Token no válido", f"Estado: {status}", footer="Renueva el token siguiendo /help.")

    try:
        market_key, market_err = _market_key_for_league(league_id)
        if not market_key:
            return build_error_message("Ciclo de mercado no disponible", market_err)

        selected = load_selected_league() or {}
        league_name = str(selected.get("league_name", "")).strip() or league_id

        if dry_run:
            res = run_agent_objective(
                league_id=league_id,
                objective=REPORT_PLAN_OBJECTIVE,
                phase="pre",
                llm_model=os.getenv("LANGCHAIN_LLM_MODEL", "gpt-5-mini"),
                temperature=float(os.getenv("LANGCHAIN_TEMPERATURE", "0.1")),
                max_iterations=max(1, int(os.getenv("LANGCHAIN_MAX_ITERATIONS", "20"))),
                dry_run=True,
                verbose=False,
            )

            output = str(res.get("output", "") or "").strip() or "Sin salida textual del agente."
            steps = res.get("steps", []) or []
            simulation_payload = _extract_latest_simulation_payload(steps)
            actions, action_source = _extract_executable_actions(steps)
            report_payload = _extract_agent_report_payload(output)
            if not any(str(a.get("tool", "")).strip() == "increase_clause_tool" for a in actions):
                inferred_clause_actions = _build_clause_actions_from_report(report_payload, steps)
                if inferred_clause_actions:
                    actions.extend(inferred_clause_actions)
                    action_source = (
                        "tool_calls+report_clause_fallback"
                        if action_source == "tool_calls"
                        else "simulate_transfer_plan+report_clause_fallback"
                    )
            sim_summary = (
                simulation_payload.get("summary")
                if isinstance(simulation_payload.get("summary"), dict)
                else {}
            )

            cache_payload = {
                "version": 1,
                "created_at": _now_iso(),
                "league_id": league_id,
                "league_name": league_name,
                "market_key": market_key,
                "llm_model": str(res.get("llm_model", "")),
                "objective": REPORT_PLAN_OBJECTIVE,
                "output": output,
                "action_source": action_source,
                "simulation_summary": sim_summary,
                "actions": actions,
                "actions_count": len(actions),
                "executed_at": "",
            }
            _save_report_plan_cache(cache_payload)

            report_text = _build_informe_message(
                league_name=league_name,
                market_key=market_key,
                steps=steps,
                output=output,
                actions=actions,
                action_source=action_source,
                simulation_payload=simulation_payload,
            )
            send_telegram_message(bot_token, chat_id, report_text, parse_mode=TELEGRAM_PARSE_MODE)
            return build_standard_message(
                title="📊 Informe IA",
                status="Completado",
                league_name=league_name,
                market_key=market_key,
                sections=[("Resultado", ["Informe generado", "Plan cacheado para este ciclo"])],
                footer="Revisa el informe anterior y usa /compraventa si quieres ejecutar el plan.",
            )

        # /compraventa: ejecutar EXACTAMENTE el último plan de /informe del mismo ciclo.
        cache = _load_report_plan_cache()
        if not cache:
            return build_standard_message(
                title="💸 Compraventa IA",
                status="Sin plan cacheado",
                league_name=league_name,
                market_key=market_key,
                sections=[("Resultado", ["No hay plan de /informe disponible para este ciclo."])],
                footer="Primero ejecuta /informe.",
            )

        cached_league = str(cache.get("league_id", "")).strip()
        cached_market_key = str(cache.get("market_key", "")).strip()
        if cached_league != league_id or cached_market_key != market_key:
            return build_standard_message(
                title="💸 Compraventa IA",
                status="Plan desactualizado",
                league_name=league_name,
                market_key=market_key,
                sections=[
                    (
                        "Ciclos",
                        [
                            f"Actual: {market_key}",
                            f"Informe cacheado: {cached_market_key or '(sin ciclo)'}",
                        ],
                    )
                ],
                footer="Vuelve a ejecutar /informe y después /compraventa.",
            )

        if str(cache.get("executed_at", "")).strip():
            return build_standard_message(
                title="💸 Compraventa IA",
                status="Plan ya ejecutado",
                league_name=league_name,
                market_key=market_key,
                sections=[("Resultado", ["Este plan ya fue ejecutado en este ciclo."])],
                footer="Ejecuta /informe para generar uno nuevo.",
            )

        actions = cache.get("actions", [])
        if not isinstance(actions, list) or not actions:
            return build_standard_message(
                title="💸 Compraventa IA",
                status="Sin acciones ejecutables",
                league_name=league_name,
                market_key=market_key,
                sections=[("Resultado", ["El /informe de este ciclo no dejó acciones ejecutables."])],
                footer="Genera un nuevo /informe y revisa el plan propuesto.",
            )

        sim_summary = cache.get("simulation_summary", {})
        sim_summary = sim_summary if isinstance(sim_summary, dict) else {}
        summary = _execute_cached_actions(
            league_id=league_id,
            actions=actions,
            fallback_balance=sim_summary.get("saldo_actual"),
        )
        cache["executed_at"] = _now_iso()
        cache["execution_summary"] = summary
        _save_report_plan_cache(cache)

        errors = summary.get("errors", []) or []
        resume = _build_compraventa_message(
            league_name=league_name,
            market_key=market_key,
            action_source=str(cache.get("action_source", "")).strip() or "none",
            summary=summary,
        )

        send_telegram_message(bot_token, chat_id, resume, parse_mode=TELEGRAM_PARSE_MODE)
        if errors:
            return build_standard_message(
                title="💸 Compraventa IA",
                status="Ejecutada con errores",
                league_name=league_name,
                market_key=market_key,
                sections=[("Resultado", [f"Errores detectados: {len(errors)}"])],
                footer="Revisa el resumen anterior antes de lanzar otra acción.",
            )
        return build_standard_message(
            title="💸 Compraventa IA",
            status="Completada",
            league_name=league_name,
            market_key=market_key,
            sections=[("Resultado", ["Plan cacheado ejecutado correctamente."])],
        )
    except Exception as exc:
        cmd = "/informe" if dry_run else "/compraventa"
        logger.exception("Error en %s (LangChain): %s", cmd, exc)
        return build_error_message(
            f"Error ejecutando {cmd}",
            f"{type(exc).__name__}: {exc}",
            footer="Revisa logs si el error se repite.",
        )


def _run_optimize_lineup_cmd(
    *,
    bot_token: str,
    chat_id: str,
) -> str:
    """
    Recalcula y guarda la mejor alineación actual por xP (xgboost).
    """
    send_telegram_message(
        bot_token,
        chat_id,
        build_progress_message("🧩 Optimizar alineación", "Calculando el mejor once por xP."),
        parse_mode=TELEGRAM_PARSE_MODE,
    )

    league_id, league_err = _resolve_operational_league_id()
    if not league_id:
        return build_error_message("Liga no disponible", league_err, footer="Usa /ligas y /liga <nombre> para seleccionar una liga.")

    status, _ = _token_status(max_age_hours=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")))
    if status != "ok":
        return build_error_message("Token no válido", f"Estado: {status}", footer="Renueva el token siguiendo /help.")

    selected = load_selected_league() or {}
    league_name = str(selected.get("league_name", "")).strip() or league_id

    try:
        result = autoset_best_lineup(
            league_id=league_id,
            model="xgboost",
            day_before_only=False,
            after_market_time="00:00",
            timezone_name=os.getenv("TZ", "Europe/Madrid"),
            force=True,
            dry_run=False,
        )
    except Exception as exc:
        logger.exception("Error en /optimizar: %s", exc)
        return build_error_message("Error ejecutando /optimizar", f"{type(exc).__name__}: {exc}")

    if result.get("applied"):
        formation = result.get("formation") or []
        form_txt = "-".join(str(x) for x in formation) if formation else "?"
        xp_once = result.get("xp_once")
        xp_txt = (
            f"{_safe_float(xp_once):.1f}"
            if xp_once not in (None, "")
            else "?"
        )
        jornada = result.get("jornada", "?")
        return build_standard_message(
            title="🧩 Alineación optimizada",
            status="Guardada en LaLiga Fantasy",
            league_name=league_name,
            sections=[
                (
                    "Once",
                    [
                        f"Jornada: {jornada}",
                        f"Formación: {form_txt}",
                        f"xP del once: {xp_txt}",
                    ],
                )
            ],
        )

    if result.get("skipped"):
        reason = str(result.get("reason", "sin detalle"))
        return build_standard_message(
            title="🧩 Alineación optimizada",
            status="No aplicada",
            league_name=league_name,
            sections=[("Motivo", [reason])],
        )

    if result.get("dry_run"):
        return build_standard_message(
            title="🧩 Alineación optimizada",
            status="Dry-run",
            league_name=league_name,
            sections=[("Resultado", ["Simulación completada sin cambios reales."])],
        )

    reason = str(result.get("reason", "resultado no esperado"))
    return build_standard_message(
        title="🧩 Alineación optimizada",
        status="No aplicada",
        league_name=league_name,
        sections=[("Motivo", [reason])],
    )


def _run_pending_sales_cmd(
    *,
    bot_token: str,
    chat_id: str,
) -> str:
    """
    Acepta ofertas de liga pendientes (fase 2 de ventas) una vez cerrado mercado.
    Equivale a `python -m prediction.advisor_execute --aceptar-ofertas`.
    """
    send_telegram_message(
        bot_token,
        chat_id,
        build_progress_message("🤝 Ventas pendientes", "Revisando ofertas cerradas de la liga."),
        parse_mode=TELEGRAM_PARSE_MODE,
    )

    league_id, league_err = _resolve_operational_league_id()
    if not league_id:
        return build_error_message("Liga no disponible", league_err, footer="Usa /ligas y /liga <nombre> para seleccionar una liga.")

    status, _ = _token_status(max_age_hours=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")))
    if status != "ok":
        return build_error_message("Token no válido", f"Estado: {status}", footer="Renueva el token siguiendo /help.")

    selected = load_selected_league() or {}
    league_name = str(selected.get("league_name", "")).strip() or league_id

    try:
        accepted = int(run_aceptar_ofertas(argparse.Namespace(league=league_id)))
    except Exception as exc:
        logger.exception("Error en /ventas: %s", exc)
        return build_error_message("Error ejecutando /ventas", f"{type(exc).__name__}: {exc}")

    if accepted > 0:
        return build_standard_message(
            title="🤝 Ventas pendientes",
            status="Fase 2 completada",
            league_name=league_name,
            sections=[("Resultado", [f"Ofertas de liga aceptadas: {accepted}"])],
        )

    return build_standard_message(
        title="🤝 Ventas pendientes",
        status="Sin acciones",
        league_name=league_name,
        sections=[("Resultado", ["No hay ventas pendientes para aceptar ahora."])],
        footer="Si acabas de publicar ventas, espera al cierre de mercado y vuelve a usar /ventas.",
    )


def _handle_text(
    text: str,
    *,
    bot_token: str = "",
    chat_id: str = "",
) -> tuple[bool, str]:
    t = text.strip().lower()
    if t.startswith("/start") or t.startswith("/help"):
        return False, build_help_message()
    if t.startswith("/status"):
        age = _token_age_hours()
        if age is None:
            return False, build_standard_message(
                title="🩺 Estado del bot",
                status="Token no disponible",
                sections=[("Token", ["No hay token válido guardado."])],
                footer="Renueva el token siguiendo /help.",
            )
        return False, build_standard_message(
            title="🩺 Estado del bot",
            status="Operativo",
            sections=[
                ("Token", [f"Edad aproximada: {age:.1f}h"]),
                ("Liga", [_selected_league_text()]),
            ],
        )

    if t.startswith("/ligas"):
        return False, _cmd_list_leagues()

    if t.startswith("/liga"):
        return False, _cmd_select_league(text)

    if t.startswith("/informe"):
        if bot_token and chat_id:
            msg = _run_langchain_agent_cmd(
                bot_token=bot_token,
                chat_id=chat_id,
                dry_run=True,
            )
        else:
            msg = build_error_message("Comando no disponible", "/informe requiere contexto de bot.")
        return False, msg

    if t.startswith("/compraventa"):
        if bot_token and chat_id:
            msg = _run_langchain_agent_cmd(
                bot_token=bot_token,
                chat_id=chat_id,
                dry_run=False,
            )
        else:
            msg = build_error_message("Comando no disponible", "/compraventa requiere contexto de bot.")
        return False, msg

    if t.startswith("/ventas"):
        if bot_token and chat_id:
            msg = _run_pending_sales_cmd(
                bot_token=bot_token,
                chat_id=chat_id,
            )
        else:
            msg = build_error_message("Comando no disponible", "/ventas requiere contexto de bot.")
        return False, msg

    if t.startswith("/optimizar"):
        if bot_token and chat_id:
            msg = _run_optimize_lineup_cmd(
                bot_token=bot_token,
                chat_id=chat_id,
            )
        else:
            msg = build_error_message("Comando no disponible", "/optimizar requiere contexto de bot.")
        return False, msg

    token = _extract_token_from_text(text.strip())
    if not token:
        return False, build_standard_message(
            title="🤖 Fantasy Bot",
            status="Mensaje no reconocido",
            sections=[("Detalle", ["No detecté un comando ni un token válido."])],
            footer="Usa /help para ver instrucciones.",
        )

    save_token(token)
    leagues, err = _list_user_leagues()
    if not err and len(leagues) == 1:
        lg = leagues[0]
        save_selected_league(lg["id"], lg["name"], source="auto_single_league")
        sched, sched_err = build_market_schedule(
            lg["id"],
            timezone_name=os.getenv("TZ", "Europe/Madrid"),
        )
        schedule_rows = (
            schedule_message(sched).splitlines()
            if sched
            else [f"No pude calcular la hora de mercado ahora: {sched_err}"]
        )
        return True, build_standard_message(
            title="🔐 Token actualizado",
            status="Guardado correctamente",
            league_name=lg["name"],
            sections=[
                ("Liga", ["Liga auto-seleccionada al ser la única disponible."]),
                ("Horario", schedule_rows),
            ],
        )
    return True, build_standard_message(
        title="🔐 Token actualizado",
        status="Guardado correctamente",
        sections=[("Liga", ["Hay varias ligas disponibles."])],
        footer="Usa /ligas y /liga para elegir la activa.",
    )


def _set_bot_commands(bot_token: str) -> None:
    """Configura el menú de comandos predefinidos en Telegram."""
    commands = [
        {"command": "informe", "description": "Informe IA y plan cacheado (ciclo actual)"},
        {"command": "compraventa", "description": "Ejecutar plan cacheado del ultimo /informe"},
        {"command": "ventas", "description": "Aceptar ofertas de liga pendientes (fase 2)"},
        {"command": "optimizar", "description": "Optimizar y guardar ahora la alineacion"},
        {"command": "ligas", "description": "Listar ligas disponibles"},
        {"command": "liga", "description": "Seleccionar liga por nombre"},
        {"command": "status", "description": "Estado del token"},
        {"command": "help", "description": "Ayuda y lista de comandos"},
    ]
    try:
        _api(bot_token, "setMyCommands", {"commands": commands})
        logger.info("Comandos del bot configurados.")
    except Exception as exc:
        logger.warning("No se pudieron configurar comandos del bot: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telegram bot para refrescar token LaLiga")
    parser.add_argument("--bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument(
        "--allowed-chat-id",
        default=os.getenv("TELEGRAM_ALLOWED_CHAT_ID", ""),
        help="Si se define, solo acepta mensajes de este chat_id",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=int(os.getenv("TOKEN_BOT_POLL_TIMEOUT", "50")),
    )
    return parser


def run_token_bot(
    *,
    bot_token: str,
    allowed_chat_id: str = "",
    poll_timeout: int = 50,
    notify_chat_id: str = "",
    stop_event: Event | None = None,
) -> None:
    if not bot_token:
        raise RuntimeError("Falta bot_token de Telegram.")

    allowed_chat_id = str(allowed_chat_id or "").strip()
    notify_chat_id = str(notify_chat_id or "").strip()
    offset = None

    if notify_chat_id:
        token_max_age = float(os.getenv("TOKEN_MAX_AGE_HOURS", "23"))
        status, age_h = _token_status(max_age_hours=token_max_age)
        if status == "missing":
            msg = build_standard_message(
                title="🔐 Token LaLiga",
                status="Ausente",
                sections=[("Renovación", GUIDA_RENOVACION_TOKEN.splitlines())],
            )
        elif status == "expired":
            msg = build_standard_message(
                title="🔐 Token LaLiga",
                status=f"Caducado · edad ~{age_h:.1f}h",
                sections=[("Renovación", GUIDA_RENOVACION_TOKEN.splitlines())],
            )
        elif status == "invalid":
            msg = build_standard_message(
                title="🔐 Token LaLiga",
                status="Inválido",
                sections=[("Renovación", GUIDA_RENOVACION_TOKEN.splitlines())],
            )
        else:
            msg = build_standard_message(
                title="🤖 Fantasy Bot iniciado",
                status="Escuchando Telegram",
                sections=[
                    ("Token", ["Token actual válido"]),
                    ("Liga", [_selected_league_text()]),
                ],
                footer="Puedes cambiar la liga con /ligas y /liga <nombre>.",
            )
        send_telegram_message(bot_token, notify_chat_id, msg, parse_mode=TELEGRAM_PARSE_MODE)

    _set_bot_commands(bot_token)

    logger.info("Telegram token bot iniciado. Token file: %s", TOKEN_FILE)
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Telegram token bot detenido por stop_event.")
            return
        try:
            payload = {"timeout": int(max(1, poll_timeout))}
            if offset is not None:
                payload["offset"] = offset
            updates = _api(bot_token, "getUpdates", payload).get("result", [])
            for upd in updates:
                offset = int(upd.get("update_id", 0)) + 1
                msg = upd.get("message") or upd.get("edited_message") or {}
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))
                text = (msg.get("text") or "").strip()

                if not chat_id or not text:
                    continue

                if allowed_chat_id and chat_id != allowed_chat_id:
                    logger.info("Mensaje ignorado de chat no autorizado: %s", chat_id)
                    continue

                changed, response = _handle_text(
                    text,
                    bot_token=bot_token,
                    chat_id=chat_id,
                )
                _api(
                    bot_token,
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": response,
                        "disable_web_page_preview": True,
                        "parse_mode": TELEGRAM_PARSE_MODE,
                    },
                )
                if changed:
                    logger.info("Token actualizado desde Telegram chat_id=%s", chat_id)
        except KeyboardInterrupt:
            logger.info("Telegram token bot detenido por usuario.")
            return
        except Exception as exc:
            logger.warning("Error en polling Telegram: %s", exc)
            if stop_event:
                stop_event.wait(timeout=5)
            else:
                time.sleep(5)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    if not args.bot_token:
        raise RuntimeError("Falta --bot-token o TELEGRAM_BOT_TOKEN.")

    run_token_bot(
        bot_token=args.bot_token,
        allowed_chat_id=args.allowed_chat_id,
        poll_timeout=args.poll_timeout,
        notify_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        stop_event=None,
    )


if __name__ == "__main__":
    main()
