"""
Agente LangGraph/LangChain para gestión autónoma de LaLiga Fantasy.

Ejemplos:
  python -m prediction.langchain_agent --phase pre
  python -m prediction.langchain_agent --phase full --dry-run
  python -m prediction.langchain_agent --objective "Analiza y optimiza mi once para la jornada."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from prediction.league_selection import resolve_league_id
from prediction.langsmith_config import (
    build_langsmith_config,
    langsmith_trace_context,
)
from prediction.langchain_tools import FantasyAgentRuntime, build_langchain_tools

logger = logging.getLogger(__name__)

MODEL_TYPE = "xgboost"


SYSTEM_PROMPT = """
Eres un agente autónomo de LaLiga Fantasy.
Tu misión es gestionar el equipo al 100%: análisis diario, compras/ventas, aceptación de ofertas y alineación.

Reglas operativas:
1. Empieza por obtener contexto real del equipo y del mercado usando herramientas.
2. Usa predicciones xP y estado de jugadores para justificar decisiones.
3. Antes de ejecutar movimientos, consulta al menos una vez la simulación del plan.
4. Respeta presupuesto, ventanas de mercado y limitaciones de datos.
5. Si dry_run está activo, actúa como simulación (sin cambios reales).
6. Después de acciones ejecutadas, verifica estado de nuevo con herramientas.
7. La subida de cláusula debe ser moderada: solo jugadores clave y expuestos a clausulazo.
8. Regla fija de cláusulas: por cada 1M invertido, la cláusula sube 2M (factor 2.0).
9. Regla de jornada: no confundas cierre de mercado con deadline de alineación.
   Una carencia como "falta portero" es planificada si quedan más de 48h para
   la jornada, alta entre 24-48h, crítica en <=24h y emergencia en <=1h o si el
   once sigue inválido al deadline. La prioridad es llegar con saldo positivo y
   11 alineados.
10. Las ventas fase 1 no financian compras inmediatas: el dinero solo cuenta
    cuando la oferta se acepta/procesa.
11. Puedes vender jugadores con ratio valor/xP ineficiente: si un jugador vale
    mucho para los puntos esperados, y la venta no compromete la alineación,
    venderlo puede liberar dinero para conseguir más puntos por menos coste en
    ciclos posteriores. El ratio sigue una curva logarítmica valor -> xP
    esperado; no uses una división lineal euros/xP. Antes de confirmar, revisa
    su cláusula: si la cláusula está muy por encima del valor de mercado, evita
    venderlo porque hay inversión en cláusula pendiente de amortizar.
12. Regla crítica de clausulazos: no se puede comprar ningún jugador mediante
   clausulazo desde 24h antes del inicio de la jornada. Si faltan 24h o menos
   para el primer partido, descarta todos los buyout_player_tool/clausulazos y
   usa solo mercado de pujas, ventas o subidas de cláusula.
13. Regla crítica de saldo negativo: si `saldo_disponible` o el saldo del plan
    es negativo, cambia a modo deuda. La prioridad absoluta es volver a saldo
    >= 0 antes de la jornada: no compres ni subas cláusulas mientras siga la
    deuda, vende primero jugadores de impacto bajo o nulo. Para elegir ventas,
    valida expected points, impacto marginal en el once, valor de mercado,
    cláusula y que tras vender todavía se pueda alinear una formación válida
    (por ejemplo, no vendas el único portero).

Formato de salida final:
- Responde en español.
- Incluye un bloque JSON válido con:
  {
    "decision_general": "...",
    "acciones_ejecutadas": ["..."],
    "acciones_ajustadas": ["..."],
    "acciones_rechazadas": ["..."],
    "acciones_descartadas": ["..."],
    "riesgos_detectados": ["..."],
    "siguiente_revision_recomendada": "..."
  }
"""


PHASE_OBJECTIVES = {
    "pre": (
        "Fase PRE mercado. "
        "Analiza el estado actual, evalúa oportunidades, decide y ejecuta la mejor estrategia "
        "de ventas fase1 y compras para maximizar xP de la próxima jornada."
    ),
    "post": (
        "Fase POST mercado. "
        "Acepta ofertas cerradas si existen y ajusta/guarda la mejor alineación posible."
    ),
    "full": (
        "Gestión completa diaria. "
        "Realiza secuencia completa: análisis, movimientos de mercado oportunos, "
        "aceptación de ofertas cerradas y alineación final."
    ),
}


def _load_langchain_stack() -> dict[str, Any]:
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "dependencia desconocida")
        raise RuntimeError(
            "No se pudo cargar LangChain/LangChain-OpenAI.\n"
            f"Falta el módulo: {missing}\n"
            "Instalación local: .venv/bin/pip install -r requirements.txt\n"
            "Docker: docker compose build --no-cache autonomous-bot && "
            "docker compose up -d --force-recreate autonomous-bot"
        ) from exc

    # Compatibilidad con API legacy (<1.0)
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        return {
            "api": "legacy",
            "ChatOpenAI": ChatOpenAI,
            "AgentExecutor": AgentExecutor,
            "create_tool_calling_agent": create_tool_calling_agent,
            "ChatPromptTemplate": ChatPromptTemplate,
            "MessagesPlaceholder": MessagesPlaceholder,
        }
    except Exception:
        pass

    # Compatibilidad con API actual (>=1.0)
    try:
        from langchain.agents import create_agent

        return {
            "api": "modern",
            "ChatOpenAI": ChatOpenAI,
            "create_agent": create_agent,
        }
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "dependencia desconocida")
        raise RuntimeError(
            "No se pudo cargar LangChain/LangChain-OpenAI.\n"
            f"Falta el módulo: {missing}\n"
            "Instalación local: .venv/bin/pip install -r requirements.txt\n"
            "Docker: docker compose build --no-cache autonomous-bot && "
            "docker compose up -d --force-recreate autonomous-bot"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "No se pudo inicializar LangChain.\n"
            "Parece una incompatibilidad de versión (API antigua vs nueva).\n"
            "Prueba: .venv/bin/pip install -r requirements.txt\n"
            "o fija versión compatible en requirements."
        ) from exc


def build_agent_executor(
    runtime: FantasyAgentRuntime,
    *,
    llm_model: str,
    temperature: float = 0.1,
    max_iterations: int = 20,
    verbose: bool = False,
):
    stack = _load_langchain_stack()
    ChatOpenAI = stack["ChatOpenAI"]

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "Falta OPENAI_API_KEY para ejecutar el agente LangChain.\n"
            "Configúralo en .env y reinicia el servicio."
        )

    tools = build_langchain_tools(runtime)
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    if stack["api"] == "legacy":
        AgentExecutor = stack["AgentExecutor"]
        create_tool_calling_agent = stack["create_tool_calling_agent"]
        ChatPromptTemplate = stack["ChatPromptTemplate"]
        MessagesPlaceholder = stack["MessagesPlaceholder"]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            max_iterations=max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    # API moderna de LangChain (v1+)
    create_agent = stack["create_agent"]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        debug=bool(verbose),
    )


def _message_content(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    parts.append(str(txt))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_steps_from_messages(messages: list[Any]) -> list[dict]:
    steps: list[dict] = []
    pending_calls: dict[str, dict[str, Any]] = {}
    for msg in messages:
        cls_name = msg.__class__.__name__
        if cls_name == "AIMessage":
            for call in getattr(msg, "tool_calls", []) or []:
                call_id = str(call.get("id", "")).strip()
                name = str(call.get("name", "")).strip() or "tool"
                args = call.get("args", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                if not isinstance(args, dict):
                    args = {}
                if call_id:
                    pending_calls[call_id] = {
                        "tool": name,
                        "tool_input": args,
                    }
        if cls_name == "ToolMessage":
            call_id = str(getattr(msg, "tool_call_id", "")).strip()
            call_meta = pending_calls.get(call_id, {})
            tool_name = str(call_meta.get("tool", "")).strip()
            if not tool_name:
                tool_name = str(getattr(msg, "name", "")).strip() or "tool"
            steps.append(
                {
                    "tool": tool_name,
                    "tool_input": call_meta.get("tool_input", {}) or {},
                    "observation": _message_content(msg)[:20000],
                }
            )
    return steps


def _extract_output(response: dict) -> str:
    output = str(response.get("output", "") or "").strip()
    if output:
        return output

    messages = response.get("messages") or []
    for msg in reversed(messages):
        cls_name = msg.__class__.__name__
        if cls_name == "AIMessage":
            text = _message_content(msg).strip()
            if text:
                return text

    if messages:
        return _message_content(messages[-1]).strip()
    return ""


def _agent_run_name(
    phase: str,
    objective: str,
    *,
    command: str | None = None,
    explicit_run_name: str | None = None,
) -> str:
    if explicit_run_name:
        return explicit_run_name
    phase_key = str(phase or "").strip().lower()
    command_key = str(command or "").strip().lower()
    if command_key == "informe":
        return "fantasy-bot.manual-report"
    if command_key == "compraventa":
        return "fantasy-bot.manual-compraventa"
    if command_key == "daemon":
        return "fantasy-bot.daemon-cycle"
    if phase_key == "pre":
        return "fantasy-bot.pre-market-agent"
    if phase_key == "post":
        return "fantasy-bot.post-market-agent"
    if phase_key == "full":
        return "fantasy-bot.full-agent"
    if objective:
        return "fantasy-bot.objective-agent"
    return "fantasy-bot.langchain-agent"


def _invoke_executor_with_config(
    executor: Any,
    objective: str,
    invoke_config: dict[str, Any],
) -> dict[str, Any]:
    try:
        return executor.invoke({"input": objective}, config=invoke_config)
    except TypeError:
        return executor.invoke({"input": objective})
    except Exception:
        try:
            return executor.invoke(
                {"messages": [{"role": "user", "content": objective}]},
                config=invoke_config,
            )
        except TypeError:
            return executor.invoke({"messages": [{"role": "user", "content": objective}]})


def run_agent_objective(
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
    engine: str | None = None,
    trace_run_name: str | None = None,
    trace_command: str | None = None,
    trace_market_cycle_id: str | None = None,
    trace_extra_metadata: dict | None = None,
) -> dict:
    engine_key = (
        engine
        or os.getenv("FANTASY_AGENT_ENGINE")
        or os.getenv("LANGCHAIN_AGENT_ENGINE")
        or "langgraph"
    ).strip().lower()
    run_name = _agent_run_name(
        phase,
        objective,
        command=trace_command,
        explicit_run_name=trace_run_name,
    )
    trace_metadata = dict(trace_extra_metadata or {})
    trace_metadata.setdefault("engine", engine_key)
    trace_metadata.setdefault("llm_model", llm_model)
    trace_metadata.setdefault("model_provider", "openai")
    if engine_key in {"langgraph", "graph"}:
        from prediction.langgraph_agent import run_graph_objective

        return run_graph_objective(
            league_id=league_id,
            objective=objective,
            phase=phase,
            model_type=model_type,
            llm_model=llm_model,
            temperature=temperature,
            max_iterations=max_iterations,
            dry_run=dry_run,
            verbose=verbose,
            trace_run_name=run_name,
            trace_command=trace_command,
            trace_market_cycle_id=trace_market_cycle_id,
            trace_extra_metadata=trace_metadata,
        )
    if engine_key not in {"legacy", "langchain"}:
        raise ValueError(
            f"Motor de agente inválido: {engine_key}. Usa langgraph o legacy."
        )

    runtime = FantasyAgentRuntime(
        league_id=league_id,
        model_type=model_type,
        dry_run=dry_run,
        phase=phase,
    )
    executor = build_agent_executor(
        runtime,
        llm_model=llm_model,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    invoke_config = build_langsmith_config(
        run_name=run_name,
        phase=phase,
        command=trace_command,
        league_id=league_id,
        market_cycle_id=trace_market_cycle_id,
        dry_run=dry_run,
        extra_metadata=trace_metadata,
    )

    with langsmith_trace_context(
        run_name=run_name,
        phase=phase,
        command=trace_command,
        league_id=league_id,
        market_cycle_id=trace_market_cycle_id,
        dry_run=dry_run,
        extra_metadata=trace_metadata,
    ):
        response = _invoke_executor_with_config(
            executor,
            objective,
            invoke_config,
        )

    steps = []
    if response.get("intermediate_steps"):
        for step in response.get("intermediate_steps", []):
            action, observation = step
            steps.append(
                {
                    "tool": getattr(action, "tool", ""),
                    "tool_input": getattr(action, "tool_input", {}),
                    "observation": str(observation)[:20000],
                }
            )
    else:
        steps = _extract_steps_from_messages(response.get("messages") or [])

    return {
        "league_id": league_id,
        "objective": objective,
        "phase": phase,
        "dry_run": dry_run,
        "model_type": model_type,
        "llm_model": llm_model,
        "engine": "legacy",
        "trace_run_name": run_name,
        "output": _extract_output(response),
        "steps": steps,
    }


def run_agent_phase(
    *,
    league_id: str,
    phase: str,
    model_type: str = MODEL_TYPE,
    llm_model: str = "gpt-5-mini",
    temperature: float = 0.1,
    max_iterations: int = 20,
    dry_run: bool = False,
    verbose: bool = False,
    engine: str | None = None,
    trace_run_name: str | None = None,
    trace_command: str | None = None,
    trace_market_cycle_id: str | None = None,
    trace_extra_metadata: dict | None = None,
) -> dict:
    phase_key = (phase or "pre").strip().lower()
    if phase_key not in PHASE_OBJECTIVES:
        raise ValueError(f"Fase inválida: {phase}. Usa pre/post/full.")
    return run_agent_objective(
        league_id=league_id,
        objective=PHASE_OBJECTIVES[phase_key],
        phase=phase_key,
        model_type=model_type,
        llm_model=llm_model,
        temperature=temperature,
        max_iterations=max_iterations,
        dry_run=dry_run,
        verbose=verbose,
        engine=engine,
        trace_run_name=trace_run_name,
        trace_command=trace_command,
        trace_market_cycle_id=trace_market_cycle_id,
        trace_extra_metadata=trace_extra_metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente LangGraph para LaLiga Fantasy")
    parser.add_argument(
        "--league",
        default="",
        help="Liga fija opcional (modo avanzado). Si se omite, usa la selección de Telegram.",
    )
    parser.add_argument("--phase", choices=["pre", "post", "full"], default="pre")
    parser.add_argument(
        "--objective",
        default="",
        help="Si se define, reemplaza el objetivo por fase.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LANGCHAIN_LLM_MODEL", "gpt-5-mini"),
        help="Modelo LLM usado por el agente.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LANGCHAIN_TEMPERATURE", "0.1")),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.getenv("LANGCHAIN_MAX_ITERATIONS", "20")),
    )
    parser.add_argument(
        "--engine",
        choices=["langgraph", "legacy"],
        default=(
            os.getenv("FANTASY_AGENT_ENGINE")
            or os.getenv("LANGCHAIN_AGENT_ENGINE")
            or "langgraph"
        ),
        help="Motor del agente. Por defecto usa LangGraph; legacy conserva el agent executor anterior.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, help="Guardar resultado JSON en archivo.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    league_id = resolve_league_id(args.league)
    if not league_id:
        raise RuntimeError(
            "No se pudo resolver league_id.\n"
            "Selecciona liga en Telegram con /ligas y /liga <nombre> "
            "o usa --league en modo avanzado."
        )

    objective = args.objective.strip() if args.objective else PHASE_OBJECTIVES[args.phase]
    result = run_agent_objective(
        league_id=league_id,
        objective=objective,
        phase=args.phase,
        model_type=MODEL_TYPE,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_iterations=max(1, args.max_iterations),
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
        engine=args.engine,
    )

    print()
    print("=" * 72)
    print("FANTASY AGENT")
    print("=" * 72)
    print(f"Liga: {result['league_id']}")
    print(f"Objetivo: {result['objective']}")
    print(f"Dry run: {result['dry_run']}")
    print(f"Motor: {result.get('engine', args.engine)}")
    print("-" * 72)
    print(result["output"])
    print("-" * 72)
    print(f"Tools usadas: {len(result['steps'])}")
    print()

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resultado guardado en: {out}")


if __name__ == "__main__":
    main()
