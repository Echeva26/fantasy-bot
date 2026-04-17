# LangGraph Fantasy Agent

## 1. Que hace el bot

El agente IA gestiona tu equipo de LaLiga Fantasy de forma autonoma con un
grafo LangGraph. El runner conserva el nombre historico `langchain_agent.py`,
pero el motor por defecto es `langgraph`.

- Usa tools del repositorio + API oficial para analizar equipo, mercado y expected points.
- Usa `news_reader_tool` como lector RAG local sobre `scrapes/` para lesiones, sanciones, apercibidos y movimientos de valor.
- Divide el ciclo en nodos: contexto, analista, ojeador, manager, ejecutor y salida final.
- Modelo de prediccion fijo: `xgboost`.
- La liga se selecciona en Telegram por nombre (`/ligas` y `/liga <nombre>`), sin configurar `LALIGA_LEAGUE_ID`.
- Detecta automaticamente la hora de mercado desde la expiracion de jugadores publicados en la liga.
- Ejecuta PRE siempre 10 minutos antes del cierre real de mercado (flujo conjunto de `/informe` + `/compraventa` del ciclo actual).
- Ejecuta POST siempre 10 minutos despues del cierre real de mercado.
- Incluye comando manual `/ventas` para forzar fase 2 (aceptar ofertas de liga pendientes tras cierre).
- Guarda alineacion exactamente 23h55 antes del inicio de jornada.
- Permite optimizacion manual inmediata de alineacion con `/optimizar`.
- Puede subir clausulas de jugadores propios con criterio de moderacion (solo clave + expuestos).
- Regla fija de clausula: por cada 1M invertido, sube 2M la clausula.
- Si el token no esta valido, avisa por Telegram para renovarlo.

## 2. Como ponerlo a funcionar

1. Crea `.env`:

```bash
cp .env.example .env
```

2. Configura `.env` con LLM y Telegram:

```env
OPENAI_API_KEY=...
FANTASY_AGENT_ENGINE=langgraph
LANGCHAIN_LLM_MODEL=gpt-5-mini
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_ALLOWED_CHAT_ID=...
```

### Obtener variables de Telegram

1. Crea el bot con `@BotFather`:
- Abre `@BotFather` en Telegram.
- Ejecuta `/newbot`.
- Sigue el asistente y copia el token.
- Ese valor es `TELEGRAM_BOT_TOKEN`.

2. Obtén tu `chat_id`:
- Abre chat con tu bot y envía `/start`.
- Ejecuta:

```bash
curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates"
```

- Busca `message.chat.id` en la respuesta.
- Ese valor es `TELEGRAM_CHAT_ID`.

3. Define `TELEGRAM_ALLOWED_CHAT_ID`:
- Para chat privado, usa el mismo valor que `TELEGRAM_CHAT_ID`.
- Para grupo, añade el bot al grupo, envía un mensaje y vuelve a ejecutar `getUpdates` para obtener el `chat.id` del grupo.

4. Arranca el modo autonomo 24/7:

```bash
docker compose up -d
docker compose logs -f autonomous-bot
```

5. En Telegram:
- Envia token (URL `jwt.ms` o JWT `eyJ...`).
- `/ligas`
- `/liga <nombre>`

El bot confirmara la hora detectada de mercado y los horarios automaticos PRE/POST (-10m/+10m).

## 3. Flujo de nodos

1. `contexto`
   - Llama a herramientas reales para cargar estado global: plantilla, saldo, puntos, mercado disponible, predicciones xP, noticias locales y alineacion actual.
2. `analista`
   - Sub-agente centrado en tu equipo: ventas, banquillo, riesgos y proteccion de clausulas.
3. `ojeador`
   - Sub-agente centrado en mercado: chollos, clausulazos, subidas/bajadas y jugadores a evitar.
4. `manager`
   - LLM principal que cruza informes, valida presupuesto/fase y decide acciones.
5. `ejecutor`
   - Ejecuta o simula ventas, pujas, clausulazos, subidas de clausula, ofertas cerradas y alineacion.

Para ejecutar manualmente:

```bash
python -m prediction.langchain_agent --phase pre --dry-run
python -m prediction.langchain_agent --phase full --dry-run --engine langgraph
python -m prediction.langchain_agent --phase pre --dry-run --engine legacy
```
