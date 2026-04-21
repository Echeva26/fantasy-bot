from __future__ import annotations

import html
import json
import re
from typing import Any

_ACTION_TOOLS = {
    "accept_closed_offers",
    "autoset_best_lineup_tool",
    "sell_player_phase1_tool",
    "place_bid_tool",
    "buyout_player_tool",
    "increase_clause_tool",
}

TELEGRAM_PARSE_MODE = "HTML"


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


def _compact_text(text: Any, limit: int = 360) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _strip_json_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _json_dict_from_text(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}

    for candidate in (raw, _strip_json_comments(raw)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for source in (raw, _strip_json_comments(raw)):
        for idx, ch in enumerate(source):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(source[idx:])
            except Exception:
                continue
            if isinstance(obj, dict):
                candidates.append(obj)

    for obj in reversed(candidates):
        if "decision_general" in obj:
            return obj
    return candidates[-1] if candidates else {}


def _normalize_agent_payload(output: Any) -> dict[str, Any]:
    payload = _json_dict_from_text(output)
    if not payload:
        return {}

    # Algunos modelos devuelven {"decision_general": "{...json...}"}.
    # Lo desplegamos para no enviar el JSON crudo al usuario.
    for _ in range(3):
        nested = _json_dict_from_text(payload.get("decision_general"))
        if not nested or "decision_general" not in nested:
            break
        merged = dict(payload)
        merged.update(nested)
        payload = merged

    return payload


def _h(text: Any) -> str:
    return html.escape(str(text or ""), quote=False)


def _money_short(value: Any) -> str:
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


def _formation_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "-".join(parts) if parts else "?"
    return str(value or "?").strip() or "?"


def _action_label(tool: str, payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    name = str(payload.get("nombre", "")).strip()

    if tool == "accept_closed_offers":
        return "Aceptar ofertas cerradas de la liga"
    if tool == "autoset_best_lineup_tool":
        return "Guardar la mejor alineación por xP"
    if tool == "sell_player_phase1_tool":
        ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        return f"Vender {ref} por {_money_short(payload.get('sale_price', 0))}"
    if tool == "place_bid_tool":
        ref = name or f"market_item_id={payload.get('market_item_id', '?')}"
        return f"Pujar por {ref} con {_money_short(payload.get('amount', 0))}"
    if tool == "buyout_player_tool":
        ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        clause = payload.get("clause_to_pay")
        if clause in (None, "", 0, "0"):
            return f"Clausulazo a {ref}"
        return f"Clausulazo a {ref} pagando {_money_short(clause)}"
    if tool == "increase_clause_tool":
        ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        invest = _safe_int(payload.get("value_to_increase"), 0)
        if invest <= 0:
            return f"Subir cláusula de {ref}"
        return (
            f"Subir cláusula de {ref} invirtiendo {_money_short(invest)} "
            f"(+{_money_short(invest * 2)} de cláusula)"
        )
    return tool or "Acción sin nombre"


def _actions_from_payload(payload: dict[str, Any]) -> list[str]:
    raw_actions = payload.get("acciones_ejecutadas")
    if not raw_actions:
        raw_actions = payload.get("acciones_ejecutables")
    if not isinstance(raw_actions, list):
        return []

    labels: list[str] = []
    for action in raw_actions:
        if isinstance(action, str):
            labels.append(_compact_text(action, 150))
            continue
        if not isinstance(action, dict):
            continue
        label = str(action.get("label", "")).strip()
        if not label:
            tool = str(action.get("tool", "")).strip()
            tool_input = action.get("tool_input")
            label = _action_label(tool, tool_input if isinstance(tool_input, dict) else {})
        labels.append(_compact_text(label, 150))
    return [x for x in labels if x]


def _lineup_from_payload(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("alineacion")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for player in raw:
        if isinstance(player, dict):
            name = (
                player.get("nombre")
                or player.get("name")
                or player.get("player_team_id")
                or player.get("player_id")
            )
            pos = player.get("posicion") or player.get("position")
            label = f"{name} ({pos})" if name and pos else str(name or "")
        else:
            label = str(player or "")
        label = _compact_text(label, 90)
        if label:
            out.append(label)
    return out


def _list_from_payload(payload: dict[str, Any], key: str, limit: int) -> list[str]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        return []
    return [_compact_text(x, limit) for x in raw if str(x or "").strip()]


def _execution_highlights(steps: list[dict[str, Any]]) -> list[str]:
    highlights: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool", "")).strip()
        if tool not in _ACTION_TOOLS:
            continue

        obs = _json_dict_from_text(step.get("observation"))
        payload = step.get("tool_input")
        payload = payload if isinstance(payload, dict) else {}
        label = _action_label(tool, payload)

        if obs.get("ok") is False:
            highlights.append(f"Error en {label}: {_compact_text(obs.get('error'), 120)}")
            continue
        if obs.get("blocked"):
            reason = obs.get("reason") or obs.get("error") or "bloqueada por reglas de fase"
            highlights.append(f"Bloqueada: {label} ({_compact_text(reason, 120)})")
            continue

        if tool == "accept_closed_offers":
            if obs.get("dry_run"):
                highlights.append("Ventas cerradas revisadas (dry-run)")
                continue
            accepted = _safe_int(obs.get("accepted_offers"), 0)
            if accepted > 0:
                highlights.append(f"Ofertas cerradas aceptadas: {accepted}")
            else:
                highlights.append("Sin ofertas cerradas que aceptar")
            continue

        if tool == "autoset_best_lineup_tool":
            if obs.get("applied"):
                form = _formation_text(obs.get("formation"))
                xp = obs.get("xp_once")
                xp_txt = f" · xP {_safe_float(xp):.1f}" if xp not in (None, "") else ""
                highlights.append(f"Alineación guardada ({form}){xp_txt}")
            elif obs.get("dry_run"):
                form = _formation_text(obs.get("formation"))
                highlights.append(f"Alineación calculada en dry-run ({form})")
            elif obs.get("skipped"):
                highlights.append(f"Alineación no cambiada: {_compact_text(obs.get('reason'), 120)}")
            else:
                highlights.append(label)
            continue

        if tool:
            highlights.append(label)

    seen: set[str] = set()
    out: list[str] = []
    for item in highlights:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _section(title: str, rows: list[str]) -> str:
    clean_rows = [row for row in rows if str(row or "").strip()]
    if not clean_rows:
        clean_rows = ["• Sin datos relevantes."]
    return f"<b>{_h(title)}</b>\n" + "\n".join(clean_rows)


def _bullet_rows(items: list[str], limit: int) -> list[str]:
    rows = [f"• {_h(item)}" for item in items[:limit]]
    if len(items) > limit:
        rows.append(f"• +{len(items) - limit} más")
    return rows


def _plain_rows(items: list[Any], limit: int = 8) -> list[str]:
    rows: list[str] = []
    for item in items[:limit]:
        text = _compact_text(item, 220)
        if text:
            rows.append(f"• {_h(text)}")
    if len(items) > limit:
        rows.append(f"• +{len(items) - limit} más")
    return rows


def build_standard_message(
    *,
    title: str,
    status: str = "",
    league_name: str = "",
    market_key: str = "",
    sections: list[tuple[str, list[Any] | Any]] | None = None,
    footer: str = "",
) -> str:
    header = [f"<b>{_h(title)}</b>", "━━━━━━━━━━━━━━━━━━━━"]
    if league_name:
        header.append(f"🏆 <b>Liga:</b> {_h(league_name)}")
    if market_key:
        header.append(f"🕒 <b>Ciclo:</b> <code>{_h(market_key)}</code>")
    if status:
        header.append(f"📌 <b>Estado:</b> {_h(status)}")

    blocks = ["\n".join(header)]
    for name, content in sections or []:
        if isinstance(content, list):
            rows = _plain_rows(content)
        else:
            text = _compact_text(content, 800)
            rows = [f"• {_h(text)}"] if text else []
        blocks.append(_section(name, rows))

    if footer:
        blocks.append(_section("Siguiente paso", [f"• {_h(_compact_text(footer, 260))}"]))
    return "\n\n".join(blocks).strip()


def build_progress_message(title: str, detail: str = "") -> str:
    sections = [("Proceso", [detail])] if detail else []
    return build_standard_message(title=title, status="En curso", sections=sections)


def build_error_message(title: str, detail: Any, *, footer: str = "") -> str:
    return build_standard_message(
        title=f"⚠️ {title}",
        status="Requiere atención",
        sections=[("Detalle", [_compact_text(detail, 500)])],
        footer=footer,
    )


def build_help_message() -> str:
    return build_standard_message(
        title="🤖 Fantasy Bot · Ayuda",
        status="Comandos disponibles",
        sections=[
            (
                "Mercado",
                [
                    "/informe · Genera informe IA y cachea el plan del ciclo",
                    "/compraventa · Ejecuta el último plan cacheado del ciclo",
                    "/ventas · Acepta ofertas de liga pendientes tras el cierre",
                ],
            ),
            (
                "Equipo",
                [
                    "/optimizar · Guarda ahora la mejor alineación por xP",
                    "/ligas · Lista tus ligas disponibles",
                    "/liga <nombre> · Selecciona la liga activa",
                    "/status · Revisa token y liga activa",
                ],
            ),
            (
                "Token",
                [
                    "Para renovar token, envía el JWT o la URL completa de jwt.ms con id_token.",
                ],
            ),
        ],
    )


def build_agent_cycle_message(
    *,
    phase: str,
    market_key: str,
    tools_count: int,
    output: Any,
    steps: list[dict[str, Any]] | None = None,
    league_name: str = "",
) -> str:
    payload = _normalize_agent_payload(output)
    steps = steps or []

    phase_txt = str(phase or "").strip().upper() or "CICLO"
    decision = _compact_text(payload.get("decision_general") or output, 520)
    actions = _execution_highlights(steps) or _actions_from_payload(payload)
    discarded = _list_from_payload(payload, "acciones_descartadas", 150)
    risks = _list_from_payload(payload, "riesgos_detectados", 150)
    lineup = _lineup_from_payload(payload)
    next_review = _compact_text(payload.get("siguiente_revision_recomendada"), 180)

    status_bits = []
    if actions:
        status_bits.append(f"✅ {len(actions)} acción(es) registradas")
    else:
        status_bits.append("✅ Ciclo completado sin acciones ejecutables")
    if discarded:
        status_bits.append(f"🚫 {len(discarded)} descarte(s)")
    if risks:
        status_bits.append(f"⚠️ {len(risks)} riesgo(s)")

    header_rows = [
        f"<b>⚽ Fantasy Bot · {phase_txt} completado</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    if league_name:
        header_rows.append(f"🏆 <b>Liga:</b> {_h(league_name)}")
    if market_key:
        header_rows.append(f"🕒 <b>Ciclo:</b> <code>{_h(market_key)}</code>")
    header_rows.append(f"🧰 <b>Tools:</b> {_h(tools_count)}")

    sections = [
        "\n".join(header_rows),
        _section("Resultado", [f"• {_h(bit)}" for bit in status_bits]),
        _section("Decisión IA", [f"• {_h(decision or 'Sin explicación del agente.')}"]),
        _section("Acciones", _bullet_rows(actions, 5)),
        _section("Descartado", _bullet_rows(discarded, 4)),
        _section("Alineación", _bullet_rows(lineup, 11)),
        _section("Riesgos", _bullet_rows(risks, 3)),
        _section("Siguiente revisión", [f"• {_h(next_review or 'En el próximo ciclo programado.')}"]),
    ]
    return "\n\n".join(sections).strip()
