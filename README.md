# Fantasy Bot Autonomo (Docker Unico)

Bot para LaLiga Fantasy orientado a ejecucion **autonoma** dentro de **un solo contenedor Docker**.

Incluye en el mismo proceso:
- Scheduler diario PRE mercado (analizar y ejecutar pujas/ventas).
- Scheduler diario POST mercado (aceptar ofertas de liga).
- Auto-set de alineación D-1 (un día antes de jornada, post-mercado).
- Bot de Telegram para renovar token cuando caduque.
- Alertas de token caducado/faltante por Telegram.

---

## 1. Como funciona

El contenedor levanta `python -m prediction.docker_autonomous`, que arranca dos bucles internos:

1. **Autopilot daemon**
- Comprueba estado de `.laliga_token`.
- Si el token es valido, ejecuta:
  - PRE a la hora `AUTOPILOT_PRE_TIME`.
  - POST a la hora `AUTOPILOT_POST_TIME`.
  - Auto-set de alineación por xP en D-1 (si está habilitado).
- Si el token no es valido, no opera y te avisa por Telegram.

2. **Token bot (Telegram)**
- Escucha mensajes en tu bot de Telegram.
- Si le envias:
  - URL completa de `jwt.ms` con `id_token`/`access_token`, o
  - JWT `eyJ...`
  guarda/actualiza `.laliga_token`.

Todo queda persistido en tu repo (volumen bind):
- `.laliga_token`
- `.autopilot_state.json`
- informes `informe_jXX.md`

---

## 2. Requisitos

- Docker + Docker Compose instalados.
- Token de bot de Telegram (`@BotFather`).
- Tu `chat_id` de Telegram (para alertas).

---

## 3. Configuracion rapida

### 3.1 Crear `.env`

```bash
cp .env.example .env
```

Edita `.env` con tus valores:

```env
LALIGA_LEAGUE_ID=TU_LEAGUE_ID
TZ=Europe/Madrid

AUTOPILOT_MODEL=xgboost
AUTOPILOT_PRE_TIME=07:50
AUTOPILOT_POST_TIME=08:10
AUTOPILOT_POLL_SECONDS=30

TOKEN_MAX_AGE_HOURS=23
TOKEN_ALERT_COOLDOWN_MINUTES=360

# Alineación automática (D-1, post-mercado)
LINEUP_AUTO_ENABLED=1
LINEUP_AUTO_AFTER_TIME=08:10
LINEUP_AUTO_DAY_BEFORE_ONLY=1

TELEGRAM_BOT_TOKEN=123456:ABC...
TELEGRAM_CHAT_ID=123456789
TELEGRAM_ALLOWED_CHAT_ID=123456789

TOKEN_BOT_ENABLED=1
TOKEN_BOT_POLL_TIMEOUT=50
```

### 3.2 Obtener `TELEGRAM_CHAT_ID`

1. Abre chat con tu bot y manda cualquier mensaje (`/start`).
2. Ejecuta:

```bash
curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates"
```

3. Copia `message.chat.id` como `TELEGRAM_CHAT_ID`.
4. Usa ese mismo valor en `TELEGRAM_ALLOWED_CHAT_ID` para restringir quien puede renovar token.

---

## 4. Arranque del sistema

```bash
docker compose build
docker compose up -d
docker compose logs -f autonomous-bot
```

El servicio queda 24/7 con `restart: unless-stopped`.

---

## 5. Primer token y renovacion diaria

Cuando el bot arranca:
- si no hay token, te avisara por Telegram.

Para renovarlo:
1. Consigue token de LaLiga (JWT o URL de jwt.ms).
2. Envialo al bot de Telegram.
3. El bot responde: `Token guardado correctamente en .laliga_token.`
4. El daemon retomara operaciones automaticamente.

Comandos utiles por Telegram:
- `/help` - Ayuda y lista de comandos
- `/status` - Estado del token
- `/informe` - Generar informe de predicciones inmediatamente
- `/compraventa` - Ejecutar recomendaciones (ventas + compras) del informe

---

## 6. Flujo diario automatico

1. **Antes del cierre** (`AUTOPILOT_PRE_TIME`)
- Genera snapshot/predicciones.
- Evalua ventas/compras.
- Ejecuta ventas fase 1 y pujas/clausulazos.
- Envia resumen a Telegram.

2. **Despues del cierre** (`AUTOPILOT_POST_TIME`)
- Acepta ofertas de la liga de ventas cerradas (fase 2).
- Si la próxima jornada empieza mañana (D-1), calcula el mejor once por xP y lo guarda vía API.
- Envia resumen a Telegram.

3. **Control de token**
- Si token falta/caduca/esta invalido:
  - manda alerta Telegram.
  - no ejecuta operaciones hasta que lo renueves.

---

## 7. Comandos de operacion

### Iniciar
```bash
docker compose up -d
```

### Ver logs
```bash
docker compose logs -f autonomous-bot
```

### Reiniciar
```bash
docker compose restart autonomous-bot
```

### Parar
```bash
docker compose down
```

### Ejecutar en primer plano (debug)
```bash
docker compose up autonomous-bot
```

---

## 8. Ajuste de horarios

Se configuran en `.env`:
- `AUTOPILOT_PRE_TIME=HH:MM`
- `AUTOPILOT_POST_TIME=HH:MM`
- `TZ=Europe/Madrid` (o tu zona)

Ejemplo:
- Cierre mercado 08:00
- PRE: `07:50`
- POST: `08:10`

---

## 9. Troubleshooting

### No llega mensaje de Telegram
- Verifica `TELEGRAM_BOT_TOKEN`.
- Verifica `TELEGRAM_CHAT_ID`.
- Asegura que hablaste al bot al menos una vez.
- Revisa logs: `docker compose logs -f autonomous-bot`.

### Token no se actualiza al enviarlo
- Comprueba `TELEGRAM_ALLOWED_CHAT_ID`.
- Prueba enviar JWT `eyJ...` puro.
- O envia URL completa de `jwt.ms`.

### No ejecuta PRE/POST
- Verifica `LALIGA_LEAGUE_ID`.
- Verifica hora del contenedor (`TZ`).
- Verifica token valido (`/status` por Telegram).
- Revisa `.autopilot_state.json` para ver ultima ejecucion.

### No guarda alineación automática
- Verifica `LINEUP_AUTO_ENABLED=1`.
- Verifica `LINEUP_AUTO_AFTER_TIME` (hora local post-mercado).
- Verifica `LINEUP_AUTO_DAY_BEFORE_ONLY=1` (solo guarda en D-1).
- Revisa logs para mensajes `LINEUP` y `LINEUP ERROR`.

### Quiero probar sin tocar mercado real
- Arranca con dry-run manual:
```bash
docker compose run --rm autonomous-bot python -m prediction.autopilot --mode pre --dry-run --league TU_LEAGUE_ID
```

### Quiero probar solo el auto-set de alineación
```bash
docker compose run --rm autonomous-bot python -m prediction.lineup_autoset --league TU_LEAGUE_ID --dry-run --force
```

---

## 10. Archivos clave

- `docker-compose.yml`: servicio unico `autonomous-bot`.
- `Dockerfile`: imagen base del proyecto.
- `.env.example`: plantilla de configuracion.
- `prediction/docker_autonomous.py`: runner unico (daemon + token bot).
- `prediction/autopilot_daemon.py`: scheduler y control de token.
- `prediction/token_bot.py`: bot de Telegram para renovar token.
- `prediction/lineup_autoset.py`: cálculo y guardado de alineación por xP.

---

## 11. Seguridad recomendada

- No subas `.env` ni `.laliga_token` al repositorio.
- Usa `TELEGRAM_ALLOWED_CHAT_ID` siempre.
- No compartas `TELEGRAM_BOT_TOKEN`.
