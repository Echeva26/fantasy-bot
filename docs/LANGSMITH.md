# LangSmith en Fantasy Bot

## Qué se traza

La instrumentación añadida cubre:

- Ejecuciones del agente con `langgraph` y con el ejecutor `legacy`.
- Fase `PRE` mercado.
- Fase `POST` mercado.
- Comando `/informe`.
- Comando `/compraventa`.
- Comando `/ventas`.
- Comando `/optimizar`.
- Ejecución automática del daemon.
- Guardado automático de alineación.
- Herramientas críticas y ejecución del plan cacheado.
- Errores y fallos operativos relevantes.

Las trazas incluyen `run_name`, `tags` y `metadata` segura, por ejemplo:

- `fantasy-bot.pre-market-agent`
- `fantasy-bot.post-market-agent`
- `fantasy-bot.manual-report`
- `fantasy-bot.manual-compraventa`
- `fantasy-bot.manual-ventas`
- `fantasy-bot.manual-optimizar`
- `fantasy-bot.daemon-cycle`

## Variables de entorno

Añade estas variables a tu `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=fantasy-bot-prod
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_CALLBACKS_BACKGROUND=true
```

Recomendación:

- desarrollo: `LANGSMITH_PROJECT=fantasy-bot-dev`
- producción: `LANGSMITH_PROJECT=fantasy-bot-prod`

No subas claves reales al repositorio.

## Activarlo en local

1. Instala dependencias:

```bash
python3 -m pip install -r requirements.txt
```

2. Configura `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=tu_api_key
LANGSMITH_PROJECT=fantasy-bot-dev
```

3. Valida la configuración sin tocar LaLiga:

```bash
python3 scripts/check_langsmith_config.py
```

4. Para validar trazas sin operaciones reales, usa `dry-run`:

```bash
python3 -m prediction.langchain_agent --phase pre --dry-run
```

## Activarlo en Docker

`docker-compose.yml` ya propaga estas variables desde `.env` al contenedor:

- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_ENDPOINT`
- `LANGCHAIN_CALLBACKS_BACKGROUND`

Flujo recomendado:

```bash
docker compose build
docker compose up -d
docker compose logs -f autonomous-bot
```

Si quieres ejecutar solo con `docker run`, pasa las variables explícitamente:

```bash
docker run --rm \
  --env-file .env \
  -e LANGSMITH_TRACING=true \
  -e LANGSMITH_API_KEY=tu_api_key \
  -e LANGSMITH_PROJECT=fantasy-bot-dev \
  fantasy-bot:latest
```

## Seguridad

La capa de observabilidad sanitiza metadata antes de enviarla a LangSmith.
Se enmascaran claves sensibles como:

- `token`
- `access_token`
- `authorization`
- `password`
- `cookie`
- `set-cookie`
- `x-api-key`
- `api_key`
- `secret`
- `bearer`
- `refresh_token`

Además, la integración evita enviar:

- tokens de LaLiga
- headers de autorización
- cookies
- contraseñas
- cuerpos completos de respuestas privadas de la API

## Comportamiento esperado

- Si `LANGSMITH_TRACING=false`, no se envían trazas.
- Si falta `LANGSMITH_API_KEY`, el bot sigue arrancando y funcionando.
- Si `langsmith` no está instalado, la observabilidad se desactiva sin romper nada.
- La lógica funcional del bot no cambia.

## Recomendación de validación

Primero valida en un proyecto separado de LangSmith y con `dry-run`:

```env
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=fantasy-bot-dev
```

Cuando confirmes que las trazas se ven bien y no exponen datos sensibles, replica la configuración en producción con otro proyecto.
