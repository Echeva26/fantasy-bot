"""
Agente LangChain para gestión autónoma de LaLiga Fantasy.

Ejemplos:
  python -m prediction.langchain_agent --phase pre --league 016615640
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

from prediction.langchain_tools import FantasyAgentRuntime, build_langchain_tools

logger = logging.getLogger(__name__)


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


def _load_langchain_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise RuntimeError(
            "No se pudo cargar LangChain/LangChain-OpenAI. "
            "Instala dependencias con: pip install -r requirements.txt"
        ) from exc
    return AgentExecutor, create_tool_calling_agent, ChatPromptTemplate, MessagesPlaceholder, ChatOpenAI


def build_agent_executor(
    runtime: FantasyAgentRuntime,
    *,
    llm_model: str,
    temperature: float = 0.1,
    max_iterations: int = 20,
    verbose: bool = False,
):
    (
        AgentExecutor,
        create_tool_calling_agent,
        ChatPromptTemplate,
        MessagesPlaceholder,
        ChatOpenAI,
    ) = _load_langchain_stack()

    tools = build_langchain_tools(runtime)
    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    return executor


def run_agent_objective(
    *,
    league_id: str,
    objective: str,
    model_type: str = "xgboost",
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
    response = executor.invoke({"input": objective})

    steps = []
    for step in response.get("intermediate_steps", []):
        action, observation = step
        steps.append(
            {
                "tool": getattr(action, "tool", ""),
                "tool_input": getattr(action, "tool_input", {}),
                "observation": str(observation)[:4000],
            }
        )

    return {
        "league_id": league_id,
        "objective": objective,
        "dry_run": dry_run,
        "model_type": model_type,
        "llm_model": llm_model,
        "output": response.get("output", ""),
        "steps": steps,
    }


def run_agent_phase(
    *,
    league_id: str,
    phase: str,
    model_type: str = "xgboost",
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
    parser.add_argument("--league", default=os.getenv("LALIGA_LEAGUE_ID", ""))
    parser.add_argument("--phase", choices=["pre", "post", "full"], default="pre")
    parser.add_argument(
        "--objective",
        default="",
        help="Si se define, reemplaza el objetivo por fase.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AUTOPILOT_MODEL", "xgboost"),
        choices=["xgboost", "lightgbm"],
        help="Modelo de predicción xP.",
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

    if not args.league:
        raise RuntimeError("Falta league_id: usa --league o LALIGA_LEAGUE_ID.")

    objective = args.objective.strip() if args.objective else PHASE_OBJECTIVES[args.phase]
    result = run_agent_objective(
        league_id=args.league,
        objective=objective,
        model_type=args.model,
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
