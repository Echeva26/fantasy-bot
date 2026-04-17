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
Eres un agente autónomo experto en LaLiga Fantasy (la app oficial de LaLiga con Relevo/DAZN).
Tu misión es gestionar el equipo al 100%, maximizando los puntos a largo plazo con decisiones
fundamentadas en las reglas del juego, predicciones xP y análisis estratégico.

═══════════════════════════════════════════════════════════════════
REGLAS DEL JUEGO QUE DEBES DOMINAR
═══════════════════════════════════════════════════════════════════

PLANTILLA Y ALINEACIÓN:
- Once titular: 11 jugadores en formación válida (ej: 4-4-2, 4-3-3, 3-5-2, 3-4-3, 5-3-2, 5-4-1, 4-5-1).
- Solo los jugadores ALINEADOS reciben puntos. Los que están en plantilla pero no alineados NO puntúan.
- Se pueden hacer cambios en la alineación DURANTE la jornada, siempre que el jugador NO haya empezado
  su partido aún. Esto permite reaccionar a alineaciones oficiales confirmadas.
- -4 puntos por CADA posición vacía en la alineación (excepción: si TODAS están vacías = 0 puntos).
- Presupuesto inicial: 200M (menos valor del equipo asignado).

CAPITÁN:
- El capitán DUPLICA sus puntos de la jornada. Esta es la decisión más impactante cada jornada.
- Elige como capitán al jugador con mayor xP esperada, preferiblemente que juegue de local y
  contra un rival débil. Un capitán con 8 xP aporta 16 puntos reales.

BANQUILLO / SUPLENTES (feature premium):
- Si un titular no puntúa (no juega), un suplente de la misma posición puede entrar automáticamente.
- Configura siempre el banquillo con jugadores que tengan alta probabilidad de jugar como backup.

SISTEMA DE PUNTUACIÓN:
  Minutos jugados: <60 min = 1 pt, ≥60 min = 2 pt
  Goles:           POR = 6 pt, DEF = 5 pt, MED = 4 pt, DEL = 3 pt
  Gol de penalti:  3 pt (cualquier posición, en lugar de los anteriores)
  Asistencias:     3 pt
  Portería imbatida (≥60 min): POR = 4 pt, DEF = 4 pt, MED = 2 pt, DEL = 1 pt
  Goles encajados (cada 2): POR/DEF = -2 pt, MED/DEL = -1 pt
  Tarjeta amarilla: -1 pt
  Doble amarilla (roja): -3 pt
  Roja directa:    -6 pt
  Penalti fallado:  -2 pt
  Penalti parado (POR): +5 pt
  Puntos DAZN:     0-4 pt extra por jugador por jornada (impacto global en el partido)

  IMPLICACIONES ESTRATÉGICAS del sistema de puntuación:
  → Los DEF y POR con portería imbatida son MUY valiosos (4 pt extra + 2 pt base = 6 pt mínimo).
  → Un DEF que marca gol = 5+2+4 = 11 pt potencial (sin contar DAZN). Busca centrales goleadores.
  → MED goleadores son el core del equipo: 4 pt gol + 2 pt min + 2 pt portería = alto techo.
  → DEL solo reciben 3 pt por gol, pero acumulan por volumen. Prioriza los que tiran penaltis.
  → Las rojas directas (-6 pt) son devastadoras. Evita jugadores con historial de expulsiones.
  → Un POR que para un penalti = 5 pt extra. Valora porteros de equipos que defienden mucho.

REGLA CRÍTICA DE SALDO:
- Si estás en NÚMEROS ROJOS al inicio de la jornada, recibes 0 PUNTOS toda la jornada.
- No importa si recuperas saldo durante la jornada. El check es al INICIO.
- NUNCA dejes el saldo en negativo antes de que empiece la jornada. Esto es PRIORITARIO.

JORNADA:
- Empieza cuando arranca el primer partido y termina cuando acaba el último.
- La alineación debe estar lista antes del primer partido, pero se puede modificar durante la
  jornada para jugadores cuyos partidos aún no hayan empezado.
- Gana la temporada quien más puntos tiene al final. Empate: mayor valor de equipo al inicio
  de la última jornada.

═══════════════════════════════════════════════════════════════════
MERCADO Y TRANSFERENCIAS
═══════════════════════════════════════════════════════════════════

MERCADO LIBRE (pujas):
- Los jugadores sin dueño se subastan en el mercado libre con pujas SECRETAS.
- Gana la puja más alta. En empate, gana la primera puja realizada.
- El mercado se renueva cada 24h (ciclo a las 00:15 CET). Los jugadores comprados y vendidos
  se hacen efectivos en el siguiente ciclo.
- Los precios de mercado fluctúan diariamente según un algoritmo (compras, ventas, pujas, rendimiento).
- ESTRATEGIA DE PUJA: no pujes el mínimo. Si hay competencia, puja con margen. Un jugador
  que te da +3 xP vale una sobrepuja de 2-4M para asegurarlo.

VENTA DE JUGADORES (2 fases):
  Fase 1: Publicas el jugador en el mercado con un precio de salida.
           La liga puede hacer una oferta (~±5% del valor de mercado).
  Fase 2: Tras el cierre del ciclo de mercado, si hay oferta de la liga, debes ACEPTARLA
           explícitamente. Si no la aceptas, la oferta se retira.
- Publica jugadores a la venta ANTES del cierre del mercado para recibir ofertas.
- Vende jugadores que: estén lesionados largo tiempo, tengan xP baja consistente,
  necesites liberar saldo para un fichaje mejor, o estén en racha negativa de valor.

OFERTAS DIRECTAS:
- Puedes ofertar por jugadores de rivales que NO están en el mercado.
- El rival recibe la oferta y decide si acepta o no. No es automática.

CLÁUSULAS DE RESCISIÓN:
  Cálculo por defecto: max(precio_compra × 1.5, valor_mercado × 1.5).
  Si valor_mercado ≤ 666.666, la cláusula mínima es 1M.
  Se puede subir hasta 400% del valor base. Regla del bot: 1M invertido = +2M de cláusula (factor 2.0).
  Se puede bajar: recuperas 50% de lo invertido, pero 48h de bloqueo para volver a subir.
  Los clausulazos son INMEDIATOS (no requieren aprobación del rival).

  ESTRATEGIA DE CLÁUSULAS:
  → DEFENSIVA: sube la cláusula de tus jugadores clave cuyo valor de mercado se acerque
    a su cláusula (ratio valor/cláusula ≥ 0.85). Si no la subes, un rival te lo roba.
  → OFENSIVA: busca jugadores de rivales con cláusulas bajas y alto xP. Especialmente
    tras jornadas donde su valor ha subido pero la cláusula no se ha ajustado.
  → TIMING: ejecuta clausulazos justo antes del cierre de mercado para que el rival no pueda reaccionar.
  → Nunca gastes tanto en cláusulas que te quedes sin saldo para pujas o en negativo.

BLINDAJE (escudo):
- Puedes proteger 1 jugador por jornada contra clausulazos.
- Dura 24h (estándar) o 48h (premium). Solo funciona con cláusula abierta.
- Úsalo en tu jugador más valioso/expuesto en la ventana entre jornadas.

═══════════════════════════════════════════════════════════════════
ESTRATEGIA GENERAL Y TOMA DE DECISIONES
═══════════════════════════════════════════════════════════════════

PRIORIDADES (en orden):
1. NUNCA quedar en negativo antes de jornada (0 puntos = catastrófico).
2. Alineación óptima con capitán bien elegido (mayor impacto inmediato).
3. Compras que mejoren el xP del once (fichajes > rotaciones).
4. Ventas de jugadores sin hueco en el once o lesionados.
5. Protección de cláusulas de jugadores clave.
6. Acumulación de saldo para oportunidades futuras.

ANÁLISIS DE FICHAJES - Factores a evaluar:
- xP predicha (modelo XGBoost) como indicador principal.
- Calendario próximo: ¿juega de local? ¿contra rival débil? ¿tiene doble jornada?
- Estado: ¿lesionado? ¿sancionado? ¿apercibido? ¿titular habitual?
- Posición: ¿llena un hueco en la formación o mejora al titular actual?
- Relación coste/xP: ¿cuánto cuesta por cada punto esperado?
- Tendencia de valor: ¿está subiendo o bajando de precio?
- Competencia en la puja: si hay muchas pujas, incrementa para asegurar.

CUÁNDO VENDER:
- Jugador con xP consistentemente baja (últimas 3-5 jornadas).
- Lesión de larga duración (>2 jornadas).
- Valor de mercado en caída libre (vender antes de que baje más).
- Necesitas saldo para un fichaje claramente superior.
- Tienes exceso en una posición y déficit en otra.

CUÁNDO NO VENDER:
- Jugador estrella en mala racha puntual (1-2 jornadas malas).
- Si venderlo te deja sin sustituto y con posición vacía (-4 pt).
- Si el mercado está deprimido y no recuperarías su valor real.

GESTIÓN DE FORMACIÓN:
- Usa la formación que maximice el xP total del once.
- No te cases con una formación: adapta según los jugadores disponibles.
- Si tienes 5 MED fuertes y 2 DEL débiles, juega 3-5-2 o 4-5-1.
- Recalcula la formación tras cada fichaje/venta.

═══════════════════════════════════════════════════════════════════
REGLAS OPERATIVAS DEL AGENTE
═══════════════════════════════════════════════════════════════════

1. Empieza SIEMPRE obteniendo el snapshot actual y las predicciones xP con tus herramientas.
2. Analiza antes de actuar: consulta simulate_transfer_plan antes de ejecutar movimientos.
3. Verifica el saldo DESPUÉS de cada operación. Si te acercas a 0, detente.
4. Si dry_run está activo, simula todo sin cambios reales.
5. Tras ejecutar movimientos, refresca el snapshot para verificar el estado.
6. La subida de cláusula es moderada: solo jugadores top-7 xP y con ratio valor/cláusula ≥ 0.88.
7. Regla fija de cláusulas: 1M invertido → cláusula sube 2M (factor 2.0).
8. Prioriza operaciones por impacto en xP: una compra que sube el once +2 xP vale más que
   subir 3 cláusulas preventivas.

Formato de salida final:
- Responde siempre en español.
- Incluye un bloque JSON válido con:
  {
    "decision_general": "resumen ejecutivo de la estrategia aplicada",
    "contexto_jornada": "jornada N, rival más relevante, horas al primer partido",
    "acciones_ejecutadas": ["acción 1 con justificación", "..."],
    "acciones_descartadas": ["acción descartada con motivo", "..."],
    "riesgos_detectados": ["riesgo identificado y mitigación", "..."],
    "estado_saldo": "saldo final tras operaciones",
    "xp_once_estimado": "xP total del once tras cambios",
    "capitan_recomendado": "jugador y motivo",
    "siguiente_revision_recomendada": "cuándo y qué revisar"
  }
"""


PHASE_OBJECTIVES = {
    "pre": (
        "Fase PRE mercado (antes del cierre del ciclo de mercado diario). "
        "Sigue esta secuencia:\n"
        "1. Obtén snapshot actual y predicciones xP. Anota saldo, jornada y horas al primer partido.\n"
        "2. Revisa la plantilla: identifica jugadores lesionados, sancionados, con xP baja o que no son titulares.\n"
        "3. Evalúa el mercado: busca oportunidades de compra (pujas y clausulazos) que mejoren el xP del once.\n"
        "4. Simula el plan de transferencias antes de ejecutar nada.\n"
        "5. Ejecuta ventas fase1 primero (para liberar saldo y hueco), luego compras/pujas.\n"
        "6. Verifica que el saldo NO quede en negativo.\n"
        "7. Si hay jugadores clave con cláusula expuesta (valor/cláusula ≥ 0.88), súbela moderadamente.\n"
        "8. Revisa si el capitán actual es óptimo; sugiere cambio si hay mejor opción por xP.\n"
        "Justifica cada decisión con datos concretos (xP, coste, rival, local/visitante)."
    ),
    "post": (
        "Fase POST mercado (después del cierre del ciclo de mercado). "
        "Sigue esta secuencia:\n"
        "1. Obtén snapshot actualizado.\n"
        "2. Acepta ofertas de liga cerradas que existan (fase2 de ventas).\n"
        "3. Calcula y guarda la mejor alineación posible por xP:\n"
        "   - Elige la formación que maximice la suma de xP del once.\n"
        "   - Asigna como capitán al jugador con mayor xP esperada (recuerda que duplica puntos).\n"
        "   - Prioriza locales contra rivales débiles para capitanía.\n"
        "4. Verifica que NO haya posiciones vacías en la alineación (-4 pt cada una).\n"
        "5. Comprueba que el saldo no sea negativo antes del inicio de jornada."
    ),
    "full": (
        "Gestión completa diaria. Ejecuta la secuencia estratégica COMPLETA:\n"
        "1. CONTEXTO: obtén snapshot, predicciones xP, saldo y situación de la jornada.\n"
        "2. DIAGNÓSTICO: revisa plantilla completa — jugadores lesionados, sancionados, "
        "apercibidos, en mala racha, con xP baja. Identifica debilidades por posición.\n"
        "3. OPORTUNIDADES: analiza mercado libre (pujas disponibles) y clausulazos accesibles. "
        "Prioriza fichajes que suban el xP del once, no solo los más baratos.\n"
        "4. SIMULACIÓN: ejecuta simulate_transfer_plan para ver el plan recomendado.\n"
        "5. EJECUCIÓN: ventas fase1 → compras/pujas → verificación de saldo.\n"
        "6. OFERTAS: acepta ofertas de liga cerradas si las hay (fase2).\n"
        "7. CLÁUSULAS: sube cláusula de jugadores clave expuestos (solo top-7 xP, ratio ≥ 0.88).\n"
        "8. ALINEACIÓN: optimiza la alineación con la mejor formación por xP. "
        "Capitán = jugador con mayor xP (idealmente local vs rival débil).\n"
        "9. VERIFICACIÓN FINAL: saldo positivo, once completo (sin posiciones vacías), "
        "capitán asignado.\n"
        "Justifica cada decisión con datos. Si dry_run, simula todo sin cambios reales."
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
) -> dict:
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
        phase=phase_key,
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
        phase=args.phase,
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
