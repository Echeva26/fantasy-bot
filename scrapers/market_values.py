"""
Scraper de valores de mercado (futbolfantasy.com/analytics)
============================================================
Extrae las subidas y bajadas diarias de valor de mercado
de los jugadores de LaLiga Fantasy desde FútbolFantasy.

Fuente: https://www.futbolfantasy.com/analytics/laliga-fantasy/mercado

Incluye:
    - Variación de precio (1d, 3d, 7d, 15d, 30d, total)
    - Porcentaje de variación
    - Valor actual y valores históricos
    - Equipo y nombre
    - Resumen por equipo
"""

import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

URL = "https://www.futbolfantasy.com/analytics/laliga-fantasy/mercado"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _parse_int(text: str) -> int:
    """Convierte '+1.845.152' o '-96.535' a int."""
    cleaned = text.strip().replace(".", "").replace("+", "")
    try:
        return int(cleaned)
    except ValueError:
        return 0


def _parse_pct(text: str) -> float:
    """Convierte '-0,89%' a float."""
    cleaned = text.strip().replace("%", "").replace(",", ".").replace("+", "")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def scrape_market_values() -> dict:
    """
    Extrae las variaciones de valor de mercado de LaLiga Fantasy.

    La página tiene divs con class "elemento". Primero vienen los equipos
    (resumen), luego un header "Jugador | ...", y después los jugadores.

    Cada fila de jugador tiene la estructura (separada por |):
        Nombre | Apodo | Equipo |
        Dif1d | Dif3d | Dif7d | Dif15d | Dif30d | DifTotal |
        %1d | %3d | %7d | %15d | %30d | %Total |
        Dias | 'd' | 'días' |
        ValorActual | Valor-1d | Valor-3d | Valor-7d | ...

    Devuelve:
        {
            "subidas": [...],
            "bajadas": [...],
            "sin_cambio": int,
            "resumen_equipos": [{"equipo": str, "variacion_total": int}],
        }
    """
    logger.info("Scrapeando valores de mercado de futbolfantasy.com/analytics...")

    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    elements = soup.find_all("div", class_="elemento")
    logger.info("Elementos encontrados: %d", len(elements))

    subidas = []
    bajadas = []
    sin_cambio = 0
    resumen_equipos = []
    in_players = False

    for el in elements:
        text = el.get_text(" | ", strip=True)

        # Detectar header de jugadores para cambiar de fase
        if text.startswith("Jugador"):
            in_players = True
            continue

        if not in_players:
            # Fase equipos: "Real Madrid | +2.567.946"
            if text.startswith("Equipo"):
                continue
            parts = text.split(" | ")
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                team_name = parts[0]
                val_str = parts[-1]
                if re.match(r"^[+-][\d\.]+$", val_str):
                    resumen_equipos.append({
                        "equipo": team_name,
                        "variacion_total": _parse_int(val_str),
                    })
            continue

        # --- Fase jugadores ---
        # Separar por " | "
        parts = text.split(" | ")
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 15:
            continue

        # Encontrar el equipo via img alt
        imgs = el.find_all("img")
        equipo = ""
        for img in imgs:
            alt = img.get("alt", "").strip()
            if alt and alt not in ("", "Más info"):
                equipo = alt
                break

        # Nombre: primer campo que no sea un número ni el equipo
        nombre = parts[0] if parts else "?"

        # Encontrar dónde empiezan los números (variaciones)
        # Los primeros campos son: Nombre | Apodo | Equipo
        # Después vienen las 6 variaciones (con signo +/-)
        num_start = -1
        for i, p in enumerate(parts):
            if re.match(r"^[+-]\d", p):
                num_start = i
                break

        if num_start < 0 or num_start + 6 > len(parts):
            continue

        # Extraer 6 variaciones absolutas
        diffs = []
        for i in range(num_start, min(num_start + 6, len(parts))):
            if re.match(r"^[+-][\d\.]+$", parts[i]):
                diffs.append(_parse_int(parts[i]))
            else:
                break

        if len(diffs) < 1:
            continue

        # Extraer porcentajes (después de las variaciones)
        pct_start = num_start + len(diffs)
        pcts = []
        for i in range(pct_start, min(pct_start + 6, len(parts))):
            if re.match(r"^[+-]?[\d,]+%$", parts[i]):
                pcts.append(_parse_pct(parts[i]))
            else:
                break

        # Extraer días de datos
        dias = 0
        dias_start = pct_start + len(pcts)
        for i in range(dias_start, min(dias_start + 3, len(parts))):
            if re.match(r"^\d+$", parts[i]):
                dias = int(parts[i])
                break

        # Extraer valores (números sin signo, grandes)
        # Están después de "dias | d | días"
        valores = []
        for p in parts:
            if re.match(r"^[\d\.]{4,}$", p) and not re.match(r"^[+-]", p):
                valores.append(_parse_int(p))

        variacion_1d = diffs[0] if len(diffs) > 0 else 0
        variacion_3d = diffs[1] if len(diffs) > 1 else 0
        variacion_7d = diffs[2] if len(diffs) > 2 else 0
        variacion_15d = diffs[3] if len(diffs) > 3 else 0
        variacion_30d = diffs[4] if len(diffs) > 4 else 0

        player = {
            "nombre": nombre,
            "equipo": equipo,
            "variacion_1d": variacion_1d,
            "variacion_3d": variacion_3d,
            "variacion_7d": variacion_7d,
            "variacion_15d": variacion_15d,
            "variacion_30d": variacion_30d,
            "pct_1d": pcts[0] if pcts else 0.0,
            "pct_30d": pcts[4] if len(pcts) > 4 else 0.0,
            "dias_datos": dias,
            "valor_actual": valores[0] if valores else 0,
            "valor_anterior": valores[1] if len(valores) > 1 else 0,
        }

        if variacion_1d > 0:
            subidas.append(player)
        elif variacion_1d < 0:
            bajadas.append(player)
        else:
            sin_cambio += 1

    # Ordenar por variación absoluta
    subidas.sort(key=lambda x: x["variacion_1d"], reverse=True)
    bajadas.sort(key=lambda x: x["variacion_1d"])

    logger.info(
        "Mercado: %d subidas, %d bajadas, %d sin cambio",
        len(subidas), len(bajadas), sin_cambio,
    )

    return {
        "subidas": subidas,
        "bajadas": bajadas,
        "sin_cambio": sin_cambio,
        "resumen_equipos": resumen_equipos,
    }


def scrape_all() -> dict:
    """
    Devuelve:
        {
            "fuente": "futbolfantasy.com/analytics",
            "mercado": {...},
        }
    """
    return {
        "fuente": "futbolfantasy.com/analytics",
        "mercado": scrape_market_values(),
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    data = scrape_all()
    mercado = data["mercado"]
    print(f"\nSubidas: {len(mercado['subidas'])}")
    print(f"Bajadas: {len(mercado['bajadas'])}")
    print(f"Sin cambio: {mercado['sin_cambio']}")
    print(f"\nEquipos:")
    for eq in mercado["resumen_equipos"]:
        print(f"  {eq['equipo']:<20} {eq['variacion_total']:>+12,}")
    print(f"\nTop 10 subidas:")
    for p in mercado["subidas"][:10]:
        print(f"  {p['nombre']:<25} {p['variacion_1d']:>+10,}  valor={p['valor_actual']:>12,} ({p['equipo']})")
    print(f"\nTop 10 bajadas:")
    for p in mercado["bajadas"][:10]:
        print(f"  {p['nombre']:<25} {p['variacion_1d']:>+10,}  valor={p['valor_actual']:>12,} ({p['equipo']})")
