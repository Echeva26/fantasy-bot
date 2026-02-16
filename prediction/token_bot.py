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
from prediction.langchain_agent import run_agent_objective
from prediction.league_selection import (
    load_selected_league,
    resolve_league_id,
    save_selected_league,
)
from prediction.lineup_autoset import autoset_best_lineup
from prediction.market_schedule import build_market_schedule, schedule_message
from prediction.telegram_notify import GUIDA_RENOVACION_TOKEN, send_telegram_message

logger = logging.getLogger(__name__)

JWT_RE = re.compile(r"(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)")
REPORT_PLAN_CACHE_FILE = Path(".langchain_last_report_plan.json")
EXECUTABLE_TOOLS = {"sell_player_phase1_tool", "place_bid_tool", "buyout_player_tool"}
REPORT_PLAN_OBJECTIVE = (
    "Genera el informe del ciclo actual y un plan EJECUTABLE de compraventa.\n"
    "Reglas obligatorias:\n"
    "1) Usa snapshot_summary, my_squad, market_opportunities y simulate_transfer_plan.\n"
    "2) Simula EXACTAMENTE las operaciones finales llamando las herramientas:\n"
    "   sell_player_phase1_tool, place_bid_tool, buyout_player_tool.\n"
    "3) Respeta el orden real de ejecución y no uses accept_closed_offers.\n"
    "4) Como dry_run está activo, no habrá cambios reales ahora.\n"
    "5) Devuelve resumen breve y claro en español."
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
    return tool_name


def _extract_direct_tool_actions(steps: list[dict]) -> list[dict]:
    actions: list[dict] = []
    for step in steps:
        tool_name = str(step.get("tool", "")).strip()
        if tool_name not in EXECUTABLE_TOOLS:
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
        "none": "sin plan ejecutable detectado",
    }.get(action_source, action_source)

    lines = [
        "INFORME IA (LangChain)",
        f"Liga: {league_name}",
        "Modo: simulacion (dry-run)",
        f"Ciclo mercado: {market_key}",
        "",
        f"Plan cacheado: {len(actions)} accion(es) ejecutables",
        f"Fuente del plan: {source_txt}",
        "Usa /compraventa para ejecutar ESTE plan en este mismo ciclo.",
    ]

    xp_now = summary.get("xp_once_actual")
    xp_post = summary.get("xp_once_post")
    xp_delta = summary.get("xp_delta")
    saldo_now = summary.get("saldo_actual")
    saldo_final = summary.get("saldo_final")
    movs = summary.get("movimientos")
    has_summary = any(v not in (None, "") for v in (xp_now, xp_post, saldo_now, saldo_final, movs))
    if has_summary:
        lines.extend(["", "Resumen simulacion:"])
        if xp_now not in (None, "") and xp_post not in (None, ""):
            delta = _safe_float(xp_delta, _safe_float(xp_post) - _safe_float(xp_now))
            delta_txt = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            lines.append(f"- xP once: {_safe_float(xp_now):.1f} -> {_safe_float(xp_post):.1f} ({delta_txt})")
        if saldo_now not in (None, "") and saldo_final not in (None, ""):
            lines.append(f"- Saldo: {_money_short(saldo_now)} -> {_money_short(saldo_final)}")
        if movs not in (None, ""):
            lines.append(f"- Movimientos simulados: {_safe_int(movs, 0)}")

    if actions:
        lines.extend(["", "Acciones propuestas para /compraventa:"])
        for idx, action in enumerate(actions[:8], 1):
            payload = _tool_input_dict(action.get("tool_input"))
            label = str(action.get("label", "")).strip() or _format_action_label(
                str(action.get("tool", "")).strip(), payload
            )
            lines.append(f"{idx}. {label}")
        if len(actions) > 8:
            lines.append(f"... y {len(actions) - 8} accion(es) mas.")
    else:
        lines.extend(
            [
                "",
                "No se detectaron acciones ejecutables en este informe.",
                "Si el texto recomienda operar, vuelve a lanzar /informe para regenerar el plan del ciclo.",
            ]
        )

    if decision:
        lines.extend(["", "Decision IA:", decision])

    if riesgos_txt:
        lines.append("")
        lines.append("Riesgos clave:")
        for r in riesgos_txt:
            lines.append(f"- {r}")

    if siguiente:
        lines.extend(["", f"Siguiente revision recomendada: {siguiente}"])

    lines.extend(
        [
            "",
            f"Tools usadas: {len(steps)}",
            f"Tools: {_format_tools_used(steps)}",
        ]
    )
    return "\n".join(lines).strip()


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
        "none": "origen no informado",
    }.get(str(action_source), str(action_source or "none"))

    lines = [
        "COMPRAVENTA IA (LangChain)",
        f"Liga: {league_name}",
        f"Ciclo mercado: {market_key}",
        f"Fuente del plan: {source_txt}",
        "",
        f"Acciones planificadas: {summary.get('actions_total', 0)}",
        f"Acciones OK: {summary.get('actions_ok', 0)}",
        f"Ventas fase1: {summary.get('ventas', 0)}",
        f"Pujas: {summary.get('pujas', 0)}",
        f"Clausulazos: {summary.get('clausulazos', 0)}",
    ]

    details = summary.get("details", [])
    if isinstance(details, list) and details:
        lines.append("")
        lines.append("Detalle de ejecucion:")
        for row in details[:10]:
            lines.append(f"- {row}")
        if len(details) > 10:
            lines.append(f"- ... {len(details) - 10} linea(s) mas.")

    errors = summary.get("errors", [])
    if isinstance(errors, list) and errors:
        lines.append("")
        lines.append(f"Errores ({len(errors)}):")
        for err in errors[:8]:
            lines.append(f"- {_compact_text(err, 220)}")
        if len(errors) > 8:
            lines.append(f"- ... {len(errors) - 8} error(es) mas.")

    return "\n".join(lines).strip()

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


def _execute_cached_actions(league_id: str, actions: list[dict]) -> dict:
    client = LaLigaFantasyClient.from_saved_token(league_id=league_id)
    summary = {
        "actions_total": len(actions),
        "actions_ok": 0,
        "ventas": 0,
        "pujas": 0,
        "clausulazos": 0,
        "details": [],
        "errors": [],
    }

    for idx, action in enumerate(actions, 1):
        tool = str(action.get("tool", "")).strip()
        payload = _tool_input_dict(action.get("tool_input"))
        label = str(action.get("label", "")).strip() or _format_action_label(tool, payload)
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
                summary["details"].append(f"[OK] {idx}. {label}")
                continue

            msg = f"Paso {idx}: tool no soportada ({tool})"
            summary["errors"].append(msg)
            summary["details"].append(f"[ERROR] {idx}. {label} -> {msg}")
        except Exception as exc:
            msg = f"Paso {idx} ({tool}): {type(exc).__name__}: {exc}"
            summary["errors"].append(msg)
            summary["details"].append(f"[ERROR] {idx}. {label} -> {type(exc).__name__}: {exc}")

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
        return f"No se pudieron listar ligas: {err}"
    if not leagues:
        return "No se encontraron ligas para este usuario."

    selected_id = resolve_league_id("")
    lines = ["Ligas disponibles:"]
    for i, lg in enumerate(leagues, 1):
        mark = " ✅" if str(lg["id"]) == str(selected_id) else ""
        lines.append(f"{i}. {lg['name']}{mark}")
    lines.append("")
    lines.append("Usa /liga <nombre> para seleccionarla.")
    return "\n".join(lines)


def _cmd_select_league(raw_text: str) -> str:
    parts = raw_text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        current = _selected_league_text()
        return (
            f"{current}\n"
            "Para seleccionar liga: /liga <nombre>\n"
            "Para ver ligas: /ligas"
        )
    target = parts[1].strip()

    leagues, err = _list_user_leagues()
    if err:
        return f"No se pudo seleccionar liga: {err}"
    if not leagues:
        return "No se encontraron ligas para este usuario."

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
            names = "\n".join(f"- {lg['name']}" for lg in exact[:10])
            return (
                "Hay varias ligas con ese nombre exacto. "
                "Escribe el nombre completo tal cual o usa el número:\n"
                f"{names}"
            )
        else:
            partial = [lg for lg in leagues if norm_target in _norm(lg.get("name", ""))]
            if len(partial) == 1:
                chosen = partial[0]
            elif len(partial) > 1:
                names = "\n".join(f"- {lg['name']}" for lg in partial[:10])
                return (
                    "Coinciden varias ligas con ese texto. "
                    "Escribe un nombre más específico o usa el número:\n"
                    f"{names}"
                )

    if not chosen:
        return (
            f"No encontré la liga '{target}'.\n"
            "Usa /ligas para ver opciones y luego /liga <nombre>."
        )

    save_selected_league(chosen["id"], chosen["name"], source="telegram")
    sched, sched_err = build_market_schedule(
        chosen["id"],
        timezone_name=os.getenv("TZ", "Europe/Madrid"),
    )
    schedule_txt = (
        "\n\n" + schedule_message(sched)
        if sched
        else f"\n\nNo pude calcular la hora de mercado ahora: {sched_err}"
    )
    return (
        "Liga seleccionada correctamente.\n"
        f"{chosen['name']}"
        f"{schedule_txt}"
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
        "Generando informe IA (LangChain)..."
        if dry_run
        else "Ejecutando plan IA de compraventa (LangChain)...",
    )

    league_id, league_err = _resolve_operational_league_id()
    if not league_id:
        return f"Error de liga: {league_err}"

    status, _ = _token_status(max_age_hours=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")))
    if status != "ok":
        return f"Error: token no valido ({status}). Renuevalo con /help."

    try:
        market_key, market_err = _market_key_for_league(league_id)
        if not market_key:
            return f"No se pudo resolver ciclo de mercado actual: {market_err}"

        selected = load_selected_league() or {}
        league_name = str(selected.get("league_name", "")).strip() or league_id

        if dry_run:
            res = run_agent_objective(
                league_id=league_id,
                objective=REPORT_PLAN_OBJECTIVE,
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
            send_telegram_message(bot_token, chat_id, report_text)
            return "Informe IA generado y plan cacheado para este ciclo."

        # /compraventa: ejecutar EXACTAMENTE el último plan de /informe del mismo ciclo.
        cache = _load_report_plan_cache()
        if not cache:
            return (
                "No hay plan cacheado de /informe para ejecutar.\n"
                "Primero ejecuta /informe en este ciclo de mercado."
            )

        cached_league = str(cache.get("league_id", "")).strip()
        cached_market_key = str(cache.get("market_key", "")).strip()
        if cached_league != league_id or cached_market_key != market_key:
            return (
                "El ultimo /informe no pertenece al ciclo de mercado actual.\n"
                f"Actual: {market_key} | Informe cacheado: {cached_market_key or '(sin ciclo)'}\n"
                "Vuelve a ejecutar /informe y despues /compraventa."
            )

        if str(cache.get("executed_at", "")).strip():
            return "Este plan ya fue ejecutado en este ciclo. Ejecuta /informe para generar uno nuevo."

        actions = cache.get("actions", [])
        if not isinstance(actions, list) or not actions:
            return (
                "El /informe de este ciclo no dejó acciones ejecutables.\n"
                "Genera un nuevo /informe y revisa que incluya plan de compraventa."
            )

        summary = _execute_cached_actions(league_id=league_id, actions=actions)
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

        send_telegram_message(bot_token, chat_id, resume)
        if errors:
            return f"Compraventa ejecutada con errores ({len(errors)})."
        return "Compraventa ejecutada con el plan cacheado del informe."
    except Exception as exc:
        cmd = "/informe" if dry_run else "/compraventa"
        logger.exception("Error en %s (LangChain): %s", cmd, exc)
        return f"Error ejecutando {cmd} (LangChain): {type(exc).__name__}: {exc}"


def _run_optimize_lineup_cmd(
    *,
    bot_token: str,
    chat_id: str,
) -> str:
    """
    Recalcula y guarda la mejor alineación actual por xP (xgboost).
    """
    send_telegram_message(bot_token, chat_id, "Optimizando alineacion actual por xP...")

    league_id, league_err = _resolve_operational_league_id()
    if not league_id:
        return f"Error de liga: {league_err}"

    status, _ = _token_status(max_age_hours=float(os.getenv("TOKEN_MAX_AGE_HOURS", "23")))
    if status != "ok":
        return f"Error: token no valido ({status}). Renuevalo con /help."

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
        return f"Error ejecutando /optimizar: {type(exc).__name__}: {exc}"

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
        text = (
            "ALINEACION OPTIMIZADA\n"
            f"Liga: {league_name}\n"
            f"Jornada: {jornada}\n"
            f"Formacion: {form_txt}\n"
            f"xP del once: {xp_txt}\n"
            "Estado: guardada en LaLiga Fantasy."
        )
        return text

    if result.get("skipped"):
        reason = str(result.get("reason", "sin detalle"))
        return f"No se pudo optimizar ahora: {reason}"

    if result.get("dry_run"):
        return "Optimizacion en dry-run (sin cambios)."

    reason = str(result.get("reason", "resultado no esperado"))
    return f"No se pudo optimizar la alineacion: {reason}"


def _handle_text(
    text: str,
    *,
    bot_token: str = "",
    chat_id: str = "",
) -> tuple[bool, str]:
    t = text.strip().lower()
    if t.startswith("/start") or t.startswith("/help"):
        return (
            False,
            "Comandos:\n"
            "• /help - Esta ayuda\n"
            "• /status - Estado del token\n"
            "• /ligas - Listar ligas disponibles\n"
            "• /liga <nombre> - Seleccionar liga activa\n"
            "• /informe - Generar informe IA y cachear plan del ciclo\n"
            "• /compraventa - Ejecutar en real el plan del ultimo /informe (mismo ciclo)\n\n"
            "• /optimizar - Guardar ahora la mejor alineacion por xP\n\n"
            "Para renovar token: envia JWT (eyJ...) o URL de jwt.ms con id_token.",
        )
    if t.startswith("/status"):
        age = _token_age_hours()
        if age is None:
            return False, "No hay token valido guardado."
        return False, f"Token guardado. Edad aproximada: {age:.1f}h.\n{_selected_league_text()}"

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
            msg = "Comando /informe requiere contexto de bot."
        return False, msg

    if t.startswith("/compraventa"):
        if bot_token and chat_id:
            msg = _run_langchain_agent_cmd(
                bot_token=bot_token,
                chat_id=chat_id,
                dry_run=False,
            )
        else:
            msg = "Comando /compraventa requiere contexto de bot."
        return False, msg

    if t.startswith("/optimizar"):
        if bot_token and chat_id:
            msg = _run_optimize_lineup_cmd(
                bot_token=bot_token,
                chat_id=chat_id,
            )
        else:
            msg = "Comando /optimizar requiere contexto de bot."
        return False, msg

    token = _extract_token_from_text(text.strip())
    if not token:
        return False, "No detecte un token valido. Usa /help para instrucciones."

    save_token(token)
    leagues, err = _list_user_leagues()
    if not err and len(leagues) == 1:
        lg = leagues[0]
        save_selected_league(lg["id"], lg["name"], source="auto_single_league")
        sched, sched_err = build_market_schedule(
            lg["id"],
            timezone_name=os.getenv("TZ", "Europe/Madrid"),
        )
        schedule_txt = (
            "\n\n" + schedule_message(sched)
            if sched
            else f"\n\nNo pude calcular la hora de mercado ahora: {sched_err}"
        )
        return (
            True,
            "Token guardado correctamente en .laliga_token.\n"
            f"Liga auto-seleccionada: {lg['name']}"
            f"{schedule_txt}",
        )
    return (
        True,
        "Token guardado correctamente en .laliga_token.\n"
        "Si tienes varias ligas, usa /ligas y /liga para elegir la activa.",
    )


def _set_bot_commands(bot_token: str) -> None:
    """Configura el menú de comandos predefinidos en Telegram."""
    commands = [
        {"command": "informe", "description": "Informe IA y plan cacheado (ciclo actual)"},
        {"command": "compraventa", "description": "Ejecutar plan cacheado del ultimo /informe"},
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
            msg = "Fantasy Autopilot TOKEN AUSENTE" + GUIDA_RENOVACION_TOKEN
        elif status == "expired":
            msg = (
                f"Fantasy Autopilot TOKEN CADUCADO (edad ~{age_h:.1f}h)"
                + GUIDA_RENOVACION_TOKEN
            )
        elif status == "invalid":
            msg = "Fantasy Autopilot TOKEN INVALIDO" + GUIDA_RENOVACION_TOKEN
        else:
            msg = (
                "Token bot iniciado. Envíame token/URL para refrescar .laliga_token.\n"
                f"{_selected_league_text()}\n"
                "Puedes cambiarla con /ligas y /liga <nombre>."
            )
        send_telegram_message(bot_token, notify_chat_id, msg)

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
