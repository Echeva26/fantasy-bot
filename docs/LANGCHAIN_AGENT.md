# LangChain Fantasy Agent

## Objetivo

Gestionar el equipo de LaLiga Fantasy de forma autónoma usando:
- herramientas analíticas del repo (`prediction/*`),
- operaciones de mercado/alineación de la API (`laliga_fantasy_client.py`),
- y un LLM orquestado por LangChain.

## Módulos añadidos

- `prediction/langchain_tools.py`
  - expone tools para snapshot, predicciones, mercado, simulación y ejecución.
- `prediction/langchain_agent.py`
  - ejecuta una misión puntual (`pre`, `post`, `full` o `objective` custom).
- `prediction/langchain_autonomous.py`
  - daemon diario PRE/POST.
- `prediction/docker_langchain_autonomous.py`
  - runner único (daemon LangChain + token bot Telegram).

## Variables de entorno

```env
OPENAI_API_KEY=...
LANGCHAIN_LLM_MODEL=gpt-5-mini
LANGCHAIN_TEMPERATURE=0.1
LANGCHAIN_MAX_ITERATIONS=20
LALIGA_LEAGUE_ID=...
```

También reutiliza:
- `AUTOPILOT_PRE_TIME`
- `AUTOPILOT_POST_TIME`
- `AUTOPILOT_POLL_SECONDS`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_ALLOWED_CHAT_ID`

## Ejecución manual

```bash
python -m prediction.langchain_agent --phase pre --league TU_LEAGUE_ID
python -m prediction.langchain_agent --phase post --league TU_LEAGUE_ID
python -m prediction.langchain_agent --phase full --league TU_LEAGUE_ID
```

Dry run:

```bash
python -m prediction.langchain_agent --phase full --league TU_LEAGUE_ID --dry-run
```

Objetivo custom:

```bash
python -m prediction.langchain_agent --league TU_LEAGUE_ID --objective "Analiza mi plantilla y ejecuta decisiones óptimas."
```

## Daemon autónomo

```bash
python -m prediction.langchain_autonomous --league TU_LEAGUE_ID
```

Estado en disco:
- `.langchain_agent_state.json`

## Runner Docker-like (un proceso)

```bash
python -m prediction.docker_langchain_autonomous --league TU_LEAGUE_ID
```

Incluye:
- scheduler LangChain PRE/POST,
- token bot de Telegram (si está habilitado).

## Notas

- El agente usa tanto herramientas de alto nivel (simulación/ejecución) como
  herramientas directas de API (`sell`, `bid`, `buyout`).
- Para evitar operaciones reales mientras validas comportamiento, usa `--dry-run`.
