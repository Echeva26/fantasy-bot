# LaLiga Fantasy API — Endpoints extraídos del APK

**Fuente:** Decompilación de `com.lfp.laligafantasy.apk` (v5.3.1.1) — strings extraídos de los DEX.

**Base URL:** `https://api-fantasy.llt-services.com`

---

## Resumen: Operaciones de mercado (nuestras necesidades)

| Operación | Método | Endpoint | Parámetros |
|-----------|--------|----------|------------|
| **Fase 1 venta** (publicar) | POST | `POST /api/v3/league/{leagueId}/market/sell` | Body: `salePrice`, `playerTeamId` |
| **Fase 2 venta** (aceptar oferta liga) | POST | `POST /api/v4/league/{leagueId}/market/{marketPlayerId}/offer/{offerId}/accept` | `marketPlayerId`, `offerId` en path |
| **Puja** (mercado libre) | POST | `POST /api/v3/league/{leagueId}/market/{marketPlayerId}/bid` | Body: `amount` |
| **Clausulazo** | POST | `POST /api/v4/league/{leagueId}/buyout/{playerTeamId}/pay` | `playerTeamId` en path |
| **Subir cláusula propia** | PUT | `PUT /api/v5/league/{leagueId}/buyout/player` | Body: `playerId`(playerTeamId), `valueToIncrease`, `factor` |

---

## Diferencias respecto a lo inferido anteriormente

### 1. Publicar jugador en venta (fase 1)

**Antes (inferido):**
```
POST /api/v3/leagues/{leagueId}/teams/{teamId}/players/{playerId}/market
Body: {"salePrice": "..."}
```

**Real (APK - SellRequest.smali):**
```
POST /api/v3/league/{leagueId}/market/sell
Body: {"playerId": "...", "salePrice": int}  // playerId = playerTeamId (ID del jugador EN TU EQUIPO). @SerializedName("playerId")
```

> **Importante:** Se usa `playerTeamId` (ID del jugador en tu plantilla), no `player_id` (PlayerMaster). El endpoint es `/market/sell`, no `/teams/.../players/.../market`.

### 2. Aceptar oferta de la liga (fase 2)

**Antes (inferido):**
```
POST /api/v3/leagues/{leagueId}/teams/{teamId}/players/{playerId}/market/accept
```

**Real (APK):**
```
POST /api/v4/league/{leagueId}/market/{marketPlayerId}/offer/{offerId}/accept
```

> **Importante:** Requiere `marketPlayerId` (ID del item en el mercado) y `offerId` (ID de la oferta que hace la liga). Tras publicar, el jugador pasa a estar en el mercado; cuando cierra, la liga hace una oferta y debes aceptarla con este endpoint.

### 3. Puja por jugador del mercado

**Antes (inferido):**
```
POST /api/v3/league/{leagueId}/market/bid
Body: {"marketItemId": "...", "amount": "..."}
```

**Real (APK):**
```
POST /api/v3/league/{leagueId}/market/{marketPlayerId}/bid
Body: {"amount": ...}  // marketPlayerId en el path
```

### 4. Clausulazo

**Antes (inferido):**
```
POST /api/v3/leagues/{leagueId}/teams/{targetTeamId}/players/{playerId}/buyout
```

**Real (APK):**
```
POST /api/v4/league/{leagueId}/buyout/{playerTeamId}/pay
```

> **Importante:** Se usa `playerTeamId` (ID del jugador en el equipo del rival, en su plantilla), no `player_id` + `target_team_id` por separado.

### 5. Subir cláusula de un jugador propio

**Real (APK - IncreaseClauseRequest + Retrofit):**
```
PUT /api/v5/league/{leagueId}/buyout/player
Body: {
  "playerId": "<playerTeamId>",
  "valueToIncrease": <int>,
  "factor": <float>
}
```

> **Importante:** `playerId` es realmente el `playerTeamId` del jugador en tu plantilla.  
> Regla operativa usada por el bot: 1M invertido -> +2M de cláusula (`factor=2.0`).

---

## Listado completo de endpoints extraídos

### api/v3

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
| `/api/v3/player/{playerId}/market-value` | Historial precios |
| `/api/v3/players` | Lista jugadores |
| `/api/v3/players/league/{leagueId}` | Jugadores de la liga |
| `/api/v3/leagues/{leagueId}` | Info liga |
| `/api/v3/leagues/{leagueId}/me` | Mi equipo en liga |
| `/api/v3/teams/{teamId}` | Plantilla de un equipo |
| `/api/v3/teams/{teamId}/lineup` | Alineación |
| `/api/v3/teams/{teamId}/money` | Saldo |
| `/api/v3/calendar` | Calendario |
| `/api/v3/week/current` | Jornada actual |
| `/api/v3/ranking/...` | Rankings |

### api/v4

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
| `/api/v4/leagues/{leagueId}/teams/{teamId}` | Equipo (v4) |
| `/api/v4/league/{leagueId}/team/daily-reward` | Recompensa diaria |
| `/api/v4/league/{leagueId}/team/{teamId}/check-daily-reward` | Comprobar recompensa |

### api/v5

| Endpoint | Uso probable |
|----------|--------------|
| `/api/v5/league/{leagueId}/buyout/player` | PUT subir cláusula propia / POST clausulazo (alternativa) |

---

## IDs relevantes

- **player_id** / **playerMasterId**: ID del jugador en el catálogo global (PlayerMaster)
- **playerTeamId**: ID del jugador en una plantilla concreta (PlayerTeam). Cada manager tiene su propio `playerTeamId` para el mismo jugador.
- **marketPlayerId**: ID del ítem cuando el jugador está en el mercado (PlayerMarket / MarketPlayer)
- **offerId**: ID de una oferta de compra sobre un jugador en venta
- **bidId**: ID de una puja en una subasta

---

## Cómo obtener los IDs en nuestro snapshot

1. **playerTeamId** (para vender y clausulazo): En `mi_equipo.plantilla` cada jugador debería tener un ID que es el `playerTeamId`. Actualmente usamos `player_id` (PlayerMaster). Hay que verificar si el snapshot incluye el ID de PlayerTeam — puede venir en la respuesta de `GET /api/v3/teams/{teamId}` o `GET /api/v4/leagues/{leagueId}/teams/{teamId}`.

2. **marketPlayerId** (para pujar y aceptar oferta): En el mercado, cada ítem tiene su ID. Ya añadimos `market_item_id` al snapshot; probablemente sea el `marketPlayerId`.

3. **offerId** (para fase 2): Cuando un jugador está en venta y la liga hace oferta, esa oferta tiene un ID. Probablemente venga en `ofertas_recibidas` o en la estructura del jugador en venta. Hay que inspeccionar la respuesta de la API al cargar el equipo o el mercado.
