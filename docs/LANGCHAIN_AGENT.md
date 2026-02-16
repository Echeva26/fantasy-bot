# LangChain Fantasy Agent

## 1. Que hace el bot

El agente IA gestiona tu equipo de LaLiga Fantasy de forma autonoma:

- Usa tools del repositorio + API oficial para analizar equipo, mercado y expected points.
- Modelo de prediccion fijo: `xgboost`.
- La liga se selecciona en Telegram por nombre (`/ligas` y `/liga <nombre>`), sin configurar `LALIGA_LEAGUE_ID`.
- Detecta automaticamente la hora de mercado desde la expiracion de jugadores publicados en la liga.
- Ejecuta PRE siempre 5 minutos antes del cierre real de mercado.
- Ejecuta POST siempre 5 minutos despues del cierre real de mercado.
- Guarda alineacion exactamente 23h55 antes del inicio de jornada.
- Si el token no esta valido, avisa por Telegram para renovarlo.

## 2. Como ponerlo a funcionar

1. Crea `.env`:

```bash
cp .env.example .env
```

2. Configura `.env` con LLM y Telegram:

```env
OPENAI_API_KEY=...
LANGCHAIN_LLM_MODEL=gpt-5-mini
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_ALLOWED_CHAT_ID=...
```

3. Arranca el modo autonomo 24/7:

```bash
docker compose up -d
docker compose logs -f autonomous-bot
```

4. En Telegram:
- Envia token (URL `jwt.ms` o JWT `eyJ...`).
- `/ligas`
- `/liga <nombre>`

El bot confirmara la hora detectada de mercado y los horarios automaticos PRE/POST (-5m/+5m).
