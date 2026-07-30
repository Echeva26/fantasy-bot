# LaLiga Fantasy API — Endpoints (actualizado Mar 2026)

**Fuente:** Decompilación APK v5.3.1.1 + probing activo de la API (Mar 2026).

**Base URL:** `https://api-fantasy.llt-services.com`

---

## Estado de los endpoints (Mar 2026)

### Endpoints PÚBLICOS (sin token)

| Endpoint | Método | Estado | Notas |
|----------|--------|--------|-------|
| `/api/v5/players` | GET | ✅ 200 | **NUEVO** — reemplaza `/api/v3/players` (404). Estructura cambiada. |
| `/api/v3/player/{playerId}` | GET | ✅ 200 | Detalle de jugador. Funciona. |
| `/api/v3/player/{playerId}/market-value` | GET | ✅ 200 | Historial de precios. Funciona. |
| `/api/v3/calendar` | GET | ✅ 200 | Calendario de jornadas. |
| `/api/v3/week/current` | GET | ✅ 200 | Jornada actual (weekNumber, isLive, etc). |
| `/api/v3/players` | GET | ❌ 404 | **DEPRECADO** — migrado a `/api/v5/players`. |

### Cambio de estructura: `/api/v5/players` vs `/api/v3/players`

**v3 (antiguo):**
```json
{
    "id": "68",
    "nickname": "Unai Simón",
    "positionId": "1",
    "team": {"id": "3", "name": "Athletic Club", "slug": "athletic-club"},
    "images": {"transparent": {"256x256": "https://..."}}
}
```

**v5 (actual):**
```json
{
    "id": "68",
    "positionId": "1",
    "nickname": "Unai Simón",
    "lastSeasonPoints": "166",
    "playerStatus": "ok",
    "marketValue": "6535733",
    "points": 164,
    "averagePoints": 5.65,
    "image": "https://assets-fantasy.llt-services.com/players/...",
    "teamId": "3"
}
```

Diferencias clave:
- `team` (objeto) → `teamId` (string)
- `images` (objeto anidado) → `image` (URL directa)

---

### Endpoints con AUTENTICACIÓN (Bearer token)

| Endpoint | Método | Estado | Uso |
|----------|--------|--------|-----|
| `/api/v3/leagues` | GET, POST | ✅ | Ligas del usuario |
| `/api/v4/leagues` | GET, POST | ✅ | Alternativa v4 |
| `/api/v3/user/me` | GET, PUT | ✅ | Info del usuario autenticado |
| `/api/v4/user/me` | GET, POST, PUT | ✅ | v4 añade POST |
| `/api/v3/players/league/{leagueId}` | GET | ✅ | Jugadores de la liga (autenticado) |
| `/api/v3/league/{leagueId}/market` | GET | ✅ | Mercado diario |
| `/api/v3/league/{leagueId}/market/sell` | POST | ✅ | Publicar jugador en venta |
| `/api/v3/league/{leagueId}/market/direct-offer` | POST | ✅ | Oferta directa |
| `/api/v3/league/{leagueId}/market/immediate-sale` | POST | ✅ | Venta inmediata |
| `/api/v3/league/{leagueId}/market/history` | GET | ✅ | Histórico del mercado |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/bid` | POST | ✅ | Pujar por jugador |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/offer` | POST | ✅ | Hacer oferta |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/delete` | DELETE | ✅ | Retirar del mercado |
| `/api/v3/league/{leagueId}/market/{mpId}/bid/{bidId}/cancel` | DELETE | ✅ | Cancelar puja |
| `/api/v3/league/{leagueId}/market/{mpId}/offer/{offerId}/cancel` | DELETE | ✅ | Cancelar oferta |
| `/api/v3/league/{leagueId}/market/{mpId}/offer/{offerId}/reject` | POST | ✅ | Rechazar oferta |
| `/api/v4/league/{leagueId}/market/{mpId}/offer/{offerId}/accept` | POST | ✅ | Aceptar oferta (fase 2 venta) |
| `/api/v3/leagues/{leagueId}/ranking` | GET | ✅ | Ranking v3 |
| `/api/v4/leagues/{leagueId}/ranking` | GET | ✅ | Ranking v4 |
| `/api/v5/leagues/{leagueId}/ranking` | GET | ✅ | Ranking v5 |
| `/api/v3/leagues/{leagueId}/teams/{teamId}` | GET | ✅ | Plantilla de equipo |
| `/api/v4/leagues/{leagueId}/teams/{teamId}` | GET | ✅ | Plantilla v4 (incluye playerTeamId) |
| `/api/v3/teams/{teamId}/lineup` | GET, PUT | ✅ | Alineación actual (GET) + actualizar (PUT) |
| `/api/v4/teams/{teamId}/lineup/week/{weekNumber}` | GET | ✅ | Alineación de jornada específica |
| `/api/v3/teams/{teamId}` | GET | ✅ | Info del equipo |
| `/api/v3/teams/{teamId}/money` | GET | ✅ | Saldo del equipo |
| `/api/v5/league/{leagueId}/buyout/player` | PUT | ✅ | Subir cláusula propia |
| `/api/v4/league/{leagueId}/buyout/{playerTeamId}/pay` | POST | ✅ | Clausulazo |
| `/api/v4/league/{leagueId}/buyout/{playerTeamId}` | GET | ✅ | Info de clausulazo |
| `/api/v4/league/{leagueId}/playerTeam/{ptId}/loan` | GET | ✅ | Préstamos |
| `/api/v4/league/{leagueId}/playerTeam/{ptId}/offer` | GET | ✅ | Ofertas sobre jugador |
| `/api/v4/league/{leagueId}/playerTeam/{ptId}/reject-all-offers` | POST | ✅ | Rechazar todas ofertas |
| `/api/v4/league/{leagueId}/playerTeam/{ptId}/return-loan` | PUT | ✅ | Devolver préstamo |
| `/api/v4/league/{leagueId}/loan/{loanOfferId}/accept` | PUT | ✅ | Aceptar préstamo |
| `/api/v4/league/{leagueId}/team/daily-reward` | POST | ✅ | Recompensa diaria |
| `/api/v4/league/{leagueId}/team/{teamId}/check-daily-reward` | GET | ✅ | Comprobar recompensa |

| Endpoint | Uso probable |
|----------|--------------|
| `/api/v3/league/{leagueId}/market` | GET mercado |
| `/api/v3/league/{leagueId}/market/sell` | POST publicar venta |
| `/api/v3/league/{leagueId}/market/history` | Histórico mercado |
| `/api/v3/league/{leagueId}/market/direct-offer` | Oferta directa |
| `/api/v3/league/{leagueId}/market/immediate-sale` | Venta inmediata |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/bid` | POST pujar |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/offer` | POST hacer oferta |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/delete` | Eliminar del mercado |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/bid/{bidId}` | Gestionar puja |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/offer/{offerId}` | Gestionar oferta |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/bid/{bidId}/cancel` | Cancelar puja |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/offer/{offerId}/cancel` | Cancelar oferta |
| `/api/v3/league/{leagueId}/market/{marketPlayerId}/offer/{offerId}/reject` | Rechazar oferta |
| `/api/v3/player/{playerId}/market-value` | Historial precios (requiere autenticación en la API actual) |
| `/api/v3/players` | Lista jugadores legacy; comprobado 2026-04-17: devuelve 404 sin token |
| `/api/v3/players/league/{leagueId}` | Jugadores de la liga (fallback autenticado actualizado) |
| `/api/v3/leagues/{leagueId}` | Info liga |
| `/api/v3/leagues/{leagueId}/me` | Mi equipo en liga |
| `/api/v3/teams/{teamId}` | Plantilla de un equipo |
| `/api/v3/teams/{teamId}/lineup` | Alineación |
| `/api/v3/teams/{teamId}/money` | Saldo |
| `/api/v3/calendar` | Calendario público |
| `/api/v3/week/current` | Jornada actual pública |
| `/api/v3/ranking/...` | Rankings |

| Endpoint | Estado anterior | Notas |
|----------|----------------|-------|
| `/api/v3/players` | GET público | → Migrado a `/api/v5/players` |
| `/api/v3/leagues/{leagueId}/news/{page}` | GET actividad | 404 en v3/v4/v5. Sin reemplazo conocido. |
| `/api/v4/league/{leagueId}/market` | GET mercado v4 | Solo funciona v3 |
| `/api/v4/players/league/{leagueId}` | GET jugadores v4 | Solo funciona v3 |
| `/api/v5/players/league/{leagueId}` | GET jugadores v5 | Solo funciona v3 |
| `/api/v4/teams/{teamId}/lineup` | GET/PUT lineup v4 | Solo funciona v3 (sin /week) |
| `/api/v5/teams/{teamId}/lineup` | GET/PUT lineup v5 | Solo funciona v3 |
| `/api/v3/teams/{teamId}/lineup/week/{wn}` | GET lineup v3 | Solo funciona v4 |

| Endpoint | Uso probable |
|----------|--------------|
| `/api/v4/league/{leagueId}/buyout/{playerTeamId}` | GET info clausulazo |
| `/api/v4/league/{leagueId}/buyout/{playerTeamId}/pay` | POST pagar clausulazo |
| `/api/v4/league/{leagueId}/market/{marketPlayerId}/offer/{offerId}/accept` | POST aceptar oferta (fase 2) |
| `/api/v4/league/{leagueId}/playerTeam/{playerTeamId}/loan` | Préstamos |
| `/api/v4/league/{leagueId}/playerTeam/{playerTeamId}/offer` | Ofertas sobre jugador |
| `/api/v4/league/{leagueId}/playerTeam/{playerTeamId}/reject-all-offers` | Rechazar todas ofertas |
| `/api/v4/league/{leagueId}/playerTeam/{playerTeamId}/return-loan` | Devolver préstamo |
| `/api/v4/league/{leagueId}/loan/{loanOfferId}/accept` | Aceptar préstamo |
| `/api/v4/leagues/{leagueId}/teams` | Equipos de la liga; fallback actualizado para ranking (comprobado 2026-04-17) |
| `/api/v4/leagues/{leagueId}/teams/{teamId}` | Equipo (v4) |
| `/api/v4/league/{leagueId}/team/daily-reward` | Recompensa diaria |
| `/api/v4/league/{leagueId}/team/{teamId}/check-daily-reward` | Comprobar recompensa |

| Endpoint | Antes | Ahora | Notas |
|----------|-------|-------|-------|
| `/api/v3/leagues/{leagueId}/me` | GET | DELETE solamente | Ya no sirve para obtener info del equipo propio. Usar `find_my_team_id()` + `get_team_raw_v4()`. |

---

## Operaciones de mercado (resumen)

| Operación | Método | Endpoint | Body |
|-----------|--------|----------|------|
| **Publicar en venta** | POST | `/api/v3/league/{leagueId}/market/sell` | `{"playerId": "<playerTeamId>", "salePrice": int}` |
| **Aceptar oferta liga** | POST | `/api/v4/league/{leagueId}/market/{mpId}/offer/{offerId}/accept` | `{}` |
| **Pujar (libre)** | POST | `/api/v3/league/{leagueId}/market/{mpId}/bid` | `{"amount": int}` |
| **Editar puja** | PUT | `/api/v3/league/{leagueId}/market/{mpId}/bid/{bidId}` | `{"amount": int}` |
| **Oferta (manager)** | POST | `/api/v3/league/{leagueId}/market/{mpId}/offer` | `{"amount": int}` |
| **Clausulazo** | POST | `/api/v4/league/{leagueId}/buyout/{playerTeamId}/pay` | `{"buyoutClauseToPay": int}` (o `{}`) |
| **Subir cláusula** | PUT | `/api/v5/league/{leagueId}/buyout/player` | `{"playerId": "<ptId>", "valueToIncrease": int, "factor": float}` |
| **Venta directa** | POST | `/api/v3/league/{leagueId}/market/direct-offer` | (sin documentar) |
| **Venta inmediata** | POST | `/api/v3/league/{leagueId}/market/immediate-sale` | (sin documentar) |
| **Retirar del mercado** | DELETE | `/api/v3/league/{leagueId}/market/{mpId}/delete` | — |
| **Cancelar puja** | DELETE | `/api/v3/league/{leagueId}/market/{mpId}/bid/{bidId}/cancel` | — |

### Body del PUT lineup (`/api/v3/teams/{teamId}/lineup`)

El body puede requerir formato camelCase (APK v5.3.1.6+):
```json
{
    "tacticalFormation": [4, 3, 3],
    "goalkeeper": "<playerTeamId>",
    "defenders": ["<ptId>", "..."],
    "midfielders": ["<ptId>", "..."],
    "strikers": ["<ptId>", "..."],
    "captain": "<playerTeamId>",
    "coach": "<playerTeamId>"
}
```

Formato alternativo snake_case (APK v5.3.1.1):
```json
{
    "tactical_formation": [4, 3, 3],
    "goalkeeper": "<playerTeamId>",
    "defender": ["<ptId>", "..."],
    "midfield": ["<ptId>", "..."],
    "striker": ["<ptId>", "..."],
    "captain": "<playerTeamId>",
    "coach": "<playerTeamId>"
}
```

El cliente implementa fallback: intenta camelCase primero, si 400 reintenta snake_case.

### Body del POST bid

El campo del importe es `"amount"` (no `"money"`):
```json
{"amount": 15000000}
```

El cliente implementa fallback: intenta `"amount"` primero, si 400 reintenta `"money"`.

---

## IDs relevantes

- **player_id** / **playerMasterId**: ID del jugador en el catálogo global (PlayerMaster)
- **playerTeamId**: ID del jugador en una plantilla concreta (PlayerTeam)
- **marketPlayerId**: ID del ítem en el mercado (PlayerMarket / MarketPlayer)
- **offerId**: ID de una oferta de compra
- **bidId**: ID de una puja en una subasta

---

## Metodología del probing (Mar 2026)

Se verificó cada endpoint con:
1. **GET sin auth** → distinguir público (200) de autenticado (401) de muerto (404)
2. **GET con token expirado** → 401 = existe, 404 = no existe
3. **OPTIONS** → descubrir métodos permitidos (Allow header)
4. **POST/PUT** → probar cambios de método

Última verificación: 22 de marzo de 2026.
APK analizado: v5.3.1.1 (Dic 2025). Última versión disponible: v5.3.1.6 (Mar 2026).
