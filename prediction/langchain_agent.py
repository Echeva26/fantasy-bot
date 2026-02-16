"""
Agente LangChain para gestión autónoma de LaLiga Fantasy.

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

Formato de salida final:
- Responde en español.
- Incluye un bloque JSON válido con:
  {
    "decision_general": "...",
    "acciones_ejecutadas": ["..."],
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
                    "observation": _message_content(msg)[:4000],
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


def run_agent_objective(
    *,
    league_id: str,
    objective: str,
    model_type: str = MODEL_TYPE,
    llm_model: str = "gpt-5-mini",
    temperature: float = 0.1,
    max_iterations: int = 20,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    runtime = FantasyAgentRuntime(
        league_id=league_id,
        model_type=model_type,
        dry_run=dry_run,
    )
    executor = build_agent_executor(
        runtime,
        llm_model=llm_model,
        temperature=temperature,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    # API legacy: {"input": ...}
    # API moderna: {"messages": [{"role":"user","content": ...}]}
    try:
        response = executor.invoke({"input": objective})
    except Exception:
        response = executor.invoke(
            {"messages": [{"role": "user", "content": objective}]}
        )

    steps = []
    if response.get("intermediate_steps"):
        for step in response.get("intermediate_steps", []):
            action, observation = step
            steps.append(
                {
                    "tool": getattr(action, "tool", ""),
                    "tool_input": getattr(action, "tool_input", {}),
                    "observation": str(observation)[:4000],
                }
            )
    else:
        steps = _extract_steps_from_messages(response.get("messages") or [])

    return {
        "league_id": league_id,
        "objective": objective,
        "dry_run": dry_run,
        "model_type": model_type,
        "llm_model": llm_model,
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
) -> dict:
    phase_key = (phase or "pre").strip().lower()
    if phase_key not in PHASE_OBJECTIVES:
        raise ValueError(f"Fase inválida: {phase}. Usa pre/post/full.")
    return run_agent_objective(
        league_id=league_id,
        objective=PHASE_OBJECTIVES[phase_key],
        model_type=model_type,
        llm_model=llm_model,
        temperature=temperature,
        max_iterations=max_iterations,
        dry_run=dry_run,
        verbose=verbose,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente LangChain para LaLiga Fantasy")
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
        help="Modelo LLM usado por LangChain.",
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
        model_type=MODEL_TYPE,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_iterations=max(1, args.max_iterations),
        dry_run=bool(args.dry_run),
        verbose=bool(args.verbose),
    )

    print()
    print("=" * 72)
    print("LANGCHAIN FANTASY AGENT")
    print("=" * 72)
    print(f"Liga: {result['league_id']}")
    print(f"Objetivo: {result['objective']}")
    print(f"Dry run: {result['dry_run']}")
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
