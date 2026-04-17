"""
Agente LangGraph para gestión autónoma de LaLiga Fantasy.

Arquitectura:
  contexto -> analista -> ojeador -> manager -> ejecutor -> final

El grafo reutiliza las tools reales del repo para leer estado, mercado,
noticias locales y ejecutar acciones. Los nodos LLM producen informes y
decisiones; el nodo ejecutor solo acepta herramientas validadas.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TypedDict

from prediction.langchain_tools import FantasyAgentRuntime, build_langchain_tools

logger = logging.getLogger(__name__)

MODEL_TYPE = "xgboost"
ALLOWED_EXECUTION_TOOLS = {
    "sell_player_phase1_tool",
    "place_bid_tool",
    "buyout_player_tool",
    "increase_clause_tool",
}


class FantasyGraphState(TypedDict, total=False):
    league_id: str
    objective: str
    phase: str
    dry_run: bool
    model_type: str
    llm_model: str
    runtime: FantasyAgentRuntime
    tools: dict[str, Any]
    context: dict[str, Any]
    analyst_report: dict[str, Any]
    scout_report: dict[str, Any]
    manager_decision: dict[str, Any]
    execution: dict[str, Any]
    output: str
    steps: list[dict[str, Any]]
    errors: list[str]


ANALYST_PROMPT = """
Eres el Agente Analista de un mánager de LaLiga Fantasy.
Tu trabajo es revisar SOLO la plantilla propia: titulares, suplentes,
jugadores en venta, riesgo de lesión/sanción, exposición a clausulazo y
posibles ventas o banquillazos.

Devuelve JSON válido con:
{
  "resumen": "...",
  "jugadores_a_alinear": ["..."],
  "jugadores_a_sentarse": ["..."],
  "jugadores_a_vender": [{"nombre": "...", "motivo": "..."}],
  "proteccion_clausulas": [{"nombre": "...", "motivo": "..."}],
  "riesgos": ["..."]
}
No inventes ids. Si falta un dato, dilo como riesgo.
"""

SCOUT_PROMPT = """
Eres el Agente Ojeador de un mánager de LaLiga Fantasy.
Tu trabajo es revisar SOLO mercado, clausulazos y noticias locales.
Busca chollos, subidas/bajadas de valor, jugadores lesionados o sancionados
y oportunidades que mejoren la plantilla sin romper presupuesto.

Devuelve JSON válido con:
{
  "resumen": "...",
  "chollos": [{"nombre": "...", "tipo": "mercado|clausulazo", "motivo": "..."}],
  "evitar": [{"nombre": "...", "motivo": "..."}],
  "riesgos_mercado": ["..."]
}
No inventes ids. Si falta un dato, dilo como riesgo.
"""

MANAGER_PROMPT = """
Eres el Mánager principal de LaLiga Fantasy.
Recibes el estado global, el informe del Analista y el informe del Ojeador.
Debes tomar la decisión final validando saldo, límites de plantilla,
fase operativa y coherencia de ids.

Reglas:
1. Usa las acciones propuestas por el motor cuando quieras operar. No inventes ids.
2. En fase post no hagas compras ni ventas; solo post-mercado y alineación.
3. Si dry_run está activo, prepara acciones simuladas y explica riesgos.
4. La protección de cláusula debe ser moderada: solo jugadores clave y expuestos.
5. Regla fija: por cada 1M invertido, la cláusula sube 2M.

Devuelve JSON válido con:
{
  "decision_general": "...",
  "acciones_ejecutables": [
    {"tool": "sell_player_phase1_tool", "tool_input": {"player_team_id": "...", "sale_price": 0}},
    {"tool": "place_bid_tool", "tool_input": {"market_item_id": "...", "amount": 0, "player_id": 0}},
    {"tool": "buyout_player_tool", "tool_input": {"player_team_id": "...", "clause_to_pay": 0}},
    {"tool": "increase_clause_tool", "tool_input": {"player_team_id": "...", "value_to_increase": 0}}
  ],
  "acciones_descartadas": ["..."],
  "alineacion": ["..."],
  "riesgos_detectados": ["..."],
  "siguiente_revision_recomendada": "..."
}
Incluye solo acciones que realmente quieras ejecutar o simular.
"""


def _load_stack() -> dict[str, Any]:
    try:
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "dependencia desconocida")
        raise RuntimeError(
            "No se pudo cargar LangGraph/LangChain-OpenAI.\n"
            f"Falta el módulo: {missing}\n"
            "Instalación local: .venv/bin/pip install -r requirements.txt\n"
            "Docker: docker compose build --no-cache autonomous-bot && "
            "docker compose up -d --force-recreate autonomous-bot"
        ) from exc

    return {
        "ChatOpenAI": ChatOpenAI,
        "StateGraph": StateGraph,
        "START": START,
        "END": END,
    }


def _message_content(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _json_dict_from_text(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
        except Exception:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
    return candidates[-1] if candidates else {}


def _compact_json(payload: Any, limit: int = 14000) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _invoke_llm(llm: Any, system_prompt: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    msg = llm.invoke(
        [
            ("system", system_prompt),
            ("human", _compact_json(payload)),
        ]
    )
    text = _message_content(msg).strip()
    parsed = _json_dict_from_text(text)
    if parsed:
        return text, parsed
    return text, {"resumen": text}


def _append_steps(state: FantasyGraphState, new_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(state.get("steps", []) or []) + new_steps


def _invoke_tool(
    tools: dict[str, Any],
    name: str,
    tool_input: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload = tool_input or {}
    observation = tools[name].invoke(payload)
    observation_text = str(observation)
    step = {
        "tool": name,
        "tool_input": payload,
        "observation": observation_text[:20000],
    }
    return observation_text, step


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _money_short(value: Any) -> str:
    amount = _safe_int(value, 0)
    sign = "-" if amount < 0 else ""
    n = abs(amount)
    if n >= 1_000_000:
        return f"{sign}{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{sign}{n / 1_000:.1f}K"
    return f"{sign}{n}"


def _format_action_label(tool_name: str, payload: dict[str, Any]) -> str:
    name = str(payload.get("nombre", "")).strip()
    if tool_name == "sell_player_phase1_tool":
        ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        return f"Vender {ref} por {_money_short(payload.get('sale_price', 0))}"
    if tool_name == "place_bid_tool":
        ref = name or f"market_item_id={payload.get('market_item_id', '?')}"
        return f"Pujar por {ref} con {_money_short(payload.get('amount', 0))}"
    if tool_name == "buyout_player_tool":
        ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        clause = _safe_int(payload.get("clause_to_pay"), 0)
        if clause > 0:
            return f"Clausulazo a {ref} pagando {_money_short(clause)}"
        return f"Clausulazo a {ref}"
    if tool_name == "increase_clause_tool":
        ref = name or f"player_team_id={payload.get('player_team_id', '?')}"
        invest = _safe_int(payload.get("value_to_increase"), 0)
        if invest > 0:
            return (
                f"Subir cláusula de {ref} invirtiendo {_money_short(invest)} "
                f"(+{_money_short(invest * 2)} de cláusula)"
            )
        return f"Subir cláusula de {ref}"
    return tool_name


def _actions_from_simulation(simulation_payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan = simulation_payload.get("plan")
    if not isinstance(plan, dict):
        return []
    movimientos = plan.get("movimientos")
    if not isinstance(movimientos, list):
        return []

    actions: list[dict[str, Any]] = []
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
                payload = {
                    "player_team_id": player_team_id,
                    "sale_price": sale_price,
                    "nombre": str(venta.get("nombre", "")).strip(),
                }
                actions.append(
                    {
                        "tool": "sell_player_phase1_tool",
                        "tool_input": payload,
                        "label": _format_action_label("sell_player_phase1_tool", payload),
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
            payload = {
                "player_team_id": player_team_id,
                "nombre": nombre,
            }
            clause = _safe_int(compra.get("coste"), 0)
            if clause > 0:
                payload["clause_to_pay"] = clause
            actions.append(
                {
                    "tool": "buyout_player_tool",
                    "tool_input": payload,
                    "label": _format_action_label("buyout_player_tool", payload),
                }
            )
            continue

        market_item_id = str(compra.get("market_item_id", "")).strip()
        amount = _safe_int(compra.get("coste"), 0)
        if market_item_id and amount > 0:
            payload = {
                "market_item_id": market_item_id,
                "amount": amount,
                "nombre": nombre,
            }
            player_id = _safe_int(compra.get("player_id"), 0)
            if player_id > 0:
                payload["player_id"] = player_id
            actions.append(
                {
                    "tool": "place_bid_tool",
                    "tool_input": payload,
                    "label": _format_action_label("place_bid_tool", payload),
                }
            )

    return actions


def _action_key(action: dict[str, Any]) -> tuple[str, str]:
    tool = str(action.get("tool", "")).strip()
    payload = action.get("tool_input")
    payload = payload if isinstance(payload, dict) else {}
    if tool == "sell_player_phase1_tool":
        return tool, str(payload.get("player_team_id", "")).strip()
    if tool == "place_bid_tool":
        return tool, str(payload.get("market_item_id", "")).strip()
    if tool in {"buyout_player_tool", "increase_clause_tool"}:
        return tool, str(payload.get("player_team_id", "")).strip()
    return tool, json.dumps(payload, sort_keys=True, default=str)


def _validated_manager_actions(
    manager_decision: dict[str, Any],
    proposed_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    proposed_by_key = {_action_key(a): a for a in proposed_actions}
    raw_actions = manager_decision.get("acciones_ejecutables", [])
    if not isinstance(raw_actions, list):
        return proposed_actions

    selected: list[dict[str, Any]] = []
    for raw in raw_actions:
        if not isinstance(raw, dict):
            continue
        tool = str(raw.get("tool", "")).strip()
        payload = raw.get("tool_input")
        payload = payload if isinstance(payload, dict) else {}
        if tool not in ALLOWED_EXECUTION_TOOLS:
            continue
        candidate = {"tool": tool, "tool_input": payload}
        key = _action_key(candidate)
        if tool == "increase_clause_tool":
            ptid = str(payload.get("player_team_id", "")).strip()
            amount = _safe_int(payload.get("value_to_increase"), 0)
            if ptid and amount > 0:
                candidate["label"] = _format_action_label(tool, payload)
                selected.append(candidate)
            continue
        if key in proposed_by_key:
            selected.append(proposed_by_key[key])

    return selected if selected else proposed_actions


def _build_graph(llm: Any, *, verbose: bool = False) -> Any:
    stack = _load_stack()
    StateGraph = stack["StateGraph"]
    START = stack["START"]
    END = stack["END"]

    def context_node(state: FantasyGraphState) -> dict[str, Any]:
        tools = state["tools"]
        context: dict[str, Any] = {}
        steps: list[dict[str, Any]] = []
        for name, payload in (
            ("snapshot_summary", {"force_refresh": True}),
            ("my_squad", {"force_refresh": False}),
            ("market_opportunities", {"limit": 60, "force_refresh": False}),
            ("news_reader_tool", {"limit": 60}),
            ("simulate_transfer_plan", {"force_refresh": False}),
            ("current_lineup", {}),
        ):
            observation, step = _invoke_tool(tools, name, payload)
            steps.append(step)
            context[name] = _json_dict_from_text(observation)

        proposed_actions = _actions_from_simulation(context.get("simulate_transfer_plan", {}))
        context["acciones_propuestas_motor"] = proposed_actions
        return {
            "context": context,
            "steps": _append_steps(state, steps),
        }

    def analyst_node(state: FantasyGraphState) -> dict[str, Any]:
        payload = {
            "objective": state.get("objective", ""),
            "phase": state.get("phase", ""),
            "dry_run": state.get("dry_run", False),
            "snapshot": state.get("context", {}).get("snapshot_summary", {}),
            "plantilla": state.get("context", {}).get("my_squad", {}),
            "noticias": state.get("context", {}).get("news_reader_tool", {}),
            "alineacion_actual": state.get("context", {}).get("current_lineup", {}),
        }
        text, parsed = _invoke_llm(llm, ANALYST_PROMPT, payload)
        step = {
            "tool": "analyst_agent",
            "tool_input": {"node": "analyst"},
            "observation": text[:20000],
        }
        return {
            "analyst_report": parsed,
            "steps": _append_steps(state, [step]),
        }

    def scout_node(state: FantasyGraphState) -> dict[str, Any]:
        payload = {
            "objective": state.get("objective", ""),
            "phase": state.get("phase", ""),
            "dry_run": state.get("dry_run", False),
            "snapshot": state.get("context", {}).get("snapshot_summary", {}),
            "mercado": state.get("context", {}).get("market_opportunities", {}),
            "noticias": state.get("context", {}).get("news_reader_tool", {}),
            "acciones_propuestas_motor": state.get("context", {}).get("acciones_propuestas_motor", []),
        }
        text, parsed = _invoke_llm(llm, SCOUT_PROMPT, payload)
        step = {
            "tool": "scout_agent",
            "tool_input": {"node": "scout"},
            "observation": text[:20000],
        }
        return {
            "scout_report": parsed,
            "steps": _append_steps(state, [step]),
        }

    def manager_node(state: FantasyGraphState) -> dict[str, Any]:
        payload = {
            "objective": state.get("objective", ""),
            "phase": state.get("phase", ""),
            "dry_run": state.get("dry_run", False),
            "contexto_global": state.get("context", {}),
            "informe_analista": state.get("analyst_report", {}),
            "informe_ojeador": state.get("scout_report", {}),
            "acciones_propuestas_motor": state.get("context", {}).get("acciones_propuestas_motor", []),
        }
        text, parsed = _invoke_llm(llm, MANAGER_PROMPT, payload)
        if "decision_general" not in parsed:
            parsed = {
                "decision_general": text,
                "acciones_ejecutables": [],
                "acciones_descartadas": [],
                "alineacion": [],
                "riesgos_detectados": ["La respuesta del mánager no venía en JSON estructurado."],
                "siguiente_revision_recomendada": "Revisar en el próximo ciclo de mercado.",
            }
        step = {
            "tool": "manager_agent",
            "tool_input": {"node": "manager"},
            "observation": text[:20000],
        }
        return {
            "manager_decision": parsed,
            "steps": _append_steps(state, [step]),
        }

    def executor_node(state: FantasyGraphState) -> dict[str, Any]:
        tools = state["tools"]
        phase = str(state.get("phase", "full")).strip().lower()
        execution_steps: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        if phase == "post":
            actions = [
                {"tool": "accept_closed_offers", "tool_input": {}},
                {
                    "tool": "autoset_best_lineup_tool",
                    "tool_input": {
                        "day_before_only": True,
                        "force": False,
                        "after_market_time": "08:10",
                    },
                },
            ]
        else:
            proposed = state.get("context", {}).get("acciones_propuestas_motor", [])
            proposed = proposed if isinstance(proposed, list) else []
            actions = _validated_manager_actions(state.get("manager_decision", {}), proposed)

        for action in actions:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool", "")).strip()
            payload = action.get("tool_input")
            payload = payload if isinstance(payload, dict) else {}
            if tool_name not in tools:
                results.append(
                    {
                        "tool": tool_name,
                        "ok": False,
                        "error": "Herramienta no registrada.",
                    }
                )
                continue
            observation, step = _invoke_tool(tools, tool_name, payload)
            execution_steps.append(step)
            parsed_obs = _json_dict_from_text(observation)
            results.append(
                {
                    "tool": tool_name,
                    "tool_input": payload,
                    "label": action.get("label") or _format_action_label(tool_name, payload),
                    "observation": parsed_obs or observation[:1200],
                }
            )

        return {
            "execution": {
                "dry_run": bool(state.get("dry_run", False)),
                "phase": phase,
                "actions_count": len(results),
                "results": results,
            },
            "steps": _append_steps(state, execution_steps),
        }

    def final_node(state: FantasyGraphState) -> dict[str, Any]:
        decision = state.get("manager_decision", {}) or {}
        execution = state.get("execution", {}) or {}
        results = execution.get("results", []) if isinstance(execution, dict) else []
        labels: list[str] = []
        if isinstance(results, list):
            for row in results:
                if isinstance(row, dict):
                    label = str(row.get("label", "")).strip()
                    if label:
                        labels.append(label)

        output_payload = {
            "decision_general": str(decision.get("decision_general", "")).strip(),
            "acciones_ejecutadas": labels,
            "acciones_descartadas": decision.get("acciones_descartadas", []),
            "riesgos_detectados": decision.get("riesgos_detectados", []),
            "siguiente_revision_recomendada": decision.get(
                "siguiente_revision_recomendada",
                "Revisar en el siguiente ciclo.",
            ),
            "arquitectura": {
                "engine": "langgraph",
                "nodos": [
                    "contexto",
                    "analista",
                    "ojeador",
                    "manager",
                    "ejecutor",
                ],
                "dry_run": bool(state.get("dry_run", False)),
                "phase": state.get("phase", ""),
            },
        }
        output = json.dumps(output_payload, indent=2, ensure_ascii=False, default=str)
        return {"output": output}

    graph = StateGraph(FantasyGraphState)
    graph.add_node("contexto", context_node)
    graph.add_node("analista", analyst_node)
    graph.add_node("ojeador", scout_node)
    graph.add_node("manager", manager_node)
    graph.add_node("ejecutor", executor_node)
    graph.add_node("final", final_node)
    graph.add_edge(START, "contexto")
    graph.add_edge("contexto", "analista")
    graph.add_edge("analista", "ojeador")
    graph.add_edge("ojeador", "manager")
    graph.add_edge("manager", "ejecutor")
    graph.add_edge("ejecutor", "final")
    graph.add_edge("final", END)
    return graph.compile(debug=bool(verbose), name="fantasy_langgraph_agent")


def run_graph_objective(
    *,
    league_id: str,
    objective: str,
    phase: str = "full",
    model_type: str = MODEL_TYPE,
    llm_model: str = "gpt-5-mini",
    temperature: float = 0.1,
    max_iterations: int = 20,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "Falta OPENAI_API_KEY para ejecutar el agente LangGraph.\n"
            "Configúralo en .env y reinicia el servicio."
        )

    stack = _load_stack()
    ChatOpenAI = stack["ChatOpenAI"]
    runtime = FantasyAgentRuntime(
        league_id=league_id,
        model_type=model_type,
        dry_run=dry_run,
        phase=phase,
    )
    tools = {tool.name: tool for tool in build_langchain_tools(runtime)}
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    graph = _build_graph(llm, verbose=verbose)
    result = graph.invoke(
        {
            "league_id": league_id,
            "objective": objective,
            "phase": phase,
            "dry_run": dry_run,
            "model_type": model_type,
            "llm_model": llm_model,
            "runtime": runtime,
            "tools": tools,
            "context": {},
            "steps": [],
            "errors": [],
        },
        config={
            "recursion_limit": max(10, int(max_iterations or 20) + 10),
        },
    )

    return {
        "league_id": league_id,
        "objective": objective,
        "phase": phase,
        "dry_run": dry_run,
        "model_type": model_type,
        "llm_model": llm_model,
        "engine": "langgraph",
        "output": str(result.get("output", "") or "").strip(),
        "steps": result.get("steps", []) or [],
        "context": result.get("context", {}) or {},
        "analyst_report": result.get("analyst_report", {}) or {},
        "scout_report": result.get("scout_report", {}) or {},
        "manager_decision": result.get("manager_decision", {}) or {},
        "execution": result.get("execution", {}) or {},
    }
