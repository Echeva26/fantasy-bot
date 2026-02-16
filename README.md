# Fantasy Bot IA (Produccion)

Bot autonomo para LaLiga Fantasy basado en LangChain.

Por defecto (via `docker-compose.yml`) arranca:
- daemon IA PRE/POST mercado (`prediction.docker_langchain_autonomous`),
- bot de Telegram para renovar token (si `TOKEN_BOT_ENABLED=1`).

## 1. Requisitos

- Docker + Docker Compose
- `OPENAI_API_KEY`
- Token de bot Telegram (recomendado)
- `.laliga_token` valido (o renovarlo desde Telegram)

## 2. Configuracion minima

```bash
cp .env.example .env
```

Configura al menos esto en `.env`:

```env
LALIGA_LEAGUE_ID=TU_LEAGUE_ID
TZ=Europe/Madrid

AUTOPILOT_MODEL=xgboost
AUTOPILOT_PRE_TIME=07:50
AUTOPILOT_POST_TIME=08:10
AUTOPILOT_POLL_SECONDS=30

OPENAI_API_KEY=...
LANGCHAIN_LLM_MODEL=gpt-5-mini
LANGCHAIN_TEMPERATURE=0.1
LANGCHAIN_MAX_ITERATIONS=20

TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_ALLOWED_CHAT_ID=...
TOKEN_BOT_ENABLED=1
TOKEN_BOT_POLL_TIMEOUT=50

TOKEN_MAX_AGE_HOURS=23
TOKEN_ALERT_COOLDOWN_MINUTES=360
```

## 3. Arranque en produccion (24/7)

```bash
docker compose build
docker compose up -d
docker compose logs -f autonomous-bot
```

## 4. Operacion diaria

El agente ejecuta automaticamente:
- PRE (`AUTOPILOT_PRE_TIME`): analiza estado del equipo/mercado y decide acciones.
- POST (`AUTOPILOT_POST_TIME`): gestiona cierre (aceptacion de ofertas/alineacion segun decision del agente).

Persistencia local:
- `.laliga_token`
- `.langchain_agent_state.json`

## 5. Comandos utiles

Iniciar:
```bash
docker compose up -d
```

Logs en vivo:
```bash
docker compose logs -f autonomous-bot
```

Reiniciar:
```bash
docker compose restart autonomous-bot
```

Parar:
```bash
docker compose down
```

## 6. Validacion segura (sin tocar mercado real)

Prueba manual del agente en dry-run:

```bash
docker compose run --rm autonomous-bot \
  python -m prediction.langchain_agent --phase full --league TU_LEAGUE_ID --dry-run
```

## 7. Renovacion de token

Si el token falta/caduca, el bot avisa por Telegram.

Para renovarlo, envia al bot:
- URL completa de `jwt.ms` con token, o
- JWT (`eyJ...`).

## 8. Troubleshooting rapido

No ejecuta ciclos:
- Revisa `LALIGA_LEAGUE_ID`, `TZ`, horarios PRE/POST y logs.

Errores del LLM:
- Revisa `OPENAI_API_KEY` y `LANGCHAIN_LLM_MODEL`.

No llegan mensajes Telegram:
- Revisa `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` y que hayas escrito al bot al menos una vez.

## 9. Seguridad

- No subir `.env` ni `.laliga_token` al repositorio.
- Mantener `TELEGRAM_ALLOWED_CHAT_ID` configurado.
- No compartir claves (`OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`).

## 10. Documentacion completa

Detalles del agente y herramientas:
- `docs/LANGCHAIN_AGENT.md`
