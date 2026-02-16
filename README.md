# Fantasy Bot IA (LangChain)

## 1. Que hace el bot

El bot gestiona tu equipo de LaLiga Fantasy de forma autonoma 24/7.

- Usa herramientas del repo + API real de LaLiga Fantasy para analizar plantilla, mercado y expected points.
- El modelo de prediccion es fijo: `xgboost`.
- La liga se elige en Telegram por nombre: `/ligas` y `/liga <nombre>`.
- Al elegir la liga, el bot detecta automaticamente la hora real de cierre del mercado leyendo la expiracion de jugadores publicados.
- PRE mercado: se ejecuta siempre 5 minutos antes del cierre real.
- POST mercado: se ejecuta siempre 5 minutos despues del cierre real.
- Guarda la alineacion exactamente 23 horas y 55 minutos antes del inicio de la jornada.
- Si el token falta o caduca, te avisa por Telegram para renovarlo.

## 2. Como ponerlo a funcionar

1. Crea el entorno:

```bash
cp .env.example .env
```

2. Rellena en `.env` solo lo necesario:

```env
OPENAI_API_KEY=...
LANGCHAIN_LLM_MODEL=gpt-5-mini
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_ALLOWED_CHAT_ID=...
```

### Obtener variables de Telegram

1. Crea el bot con `@BotFather`:
- En Telegram abre `@BotFather`.
- Ejecuta `/newbot`.
- Sigue los pasos y copia el token que te da.
- Ese valor es `TELEGRAM_BOT_TOKEN`.

2. Obtén tu `chat_id`:
- Abre chat con tu bot y envía `/start`.
- Ejecuta:

```bash
curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates"
```

- Busca el campo `message.chat.id` en la respuesta.
- Ese valor es `TELEGRAM_CHAT_ID`.

3. Define `TELEGRAM_ALLOWED_CHAT_ID`:
- Si solo quieres permitir tu chat, usa el mismo valor que `TELEGRAM_CHAT_ID`.
- Si usas grupo, añade el bot al grupo, envía un mensaje y vuelve a ejecutar `getUpdates` para tomar el `chat.id` del grupo.

4. Arranca el bot:

```bash
docker compose build
docker compose up -d
docker compose logs -f autonomous-bot
```

5. En Telegram:
- Envia el token (JWT `eyJ...` o URL de `jwt.ms`).
- Ejecuta `/ligas`.
- Ejecuta `/liga <nombre>`.

Al seleccionar la liga, el bot te respondera con la hora de mercado detectada y confirmara que PRE y POST se ejecutan 5 minutos antes/despues.
