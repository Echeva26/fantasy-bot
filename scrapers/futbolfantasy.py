"""
Scraper de FútbolFantasy (futbolfantasy.com)
=============================================
Extrae:
    - Lesionados (estado, parte médico, % titularidad, fechas de baja)
    - Sancionados (motivo, jornadas de tarjetas)
    - Apercibidos (ciclo, jornadas de tarjetas)
"""

import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://www.futbolfantasy.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _fetch(url: str) -> BeautifulSoup:
    """Descarga una página y devuelve el soup."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def _extract_team_name(header) -> str:
    """Extrae el nombre del equipo de un header de sección."""
    img = header.find("img")
    if img and img.get("alt"):
        return img["alt"].strip()
    text = header.get_text(strip=True)
    return text if text else "?"


# ───────────────────────────────────────────────────────────────
# Lesionados
# ───────────────────────────────────────────────────────────────
def scrape_lesionados() -> list[dict]:
    """
    Scrapes https://www.futbolfantasy.com/laliga/lesionados

    Devuelve lista de dicts:
        {
            "nombre": str,
            "equipo": str,
            "estado": "baja" | "duda" | "disponible",
            "probabilidad_titular": int (0-100),
            "lesion": str,
            "desde": str,
            "dias_baja": int,
            "info_retorno": str,
            "url_parte": str | None,
        }
    """
    logger.info("Scrapeando lesionados de futbolfantasy.com...")
    soup = _fetch(f"{BASE_URL}/laliga/lesionados")

    results = []
    current_team = "?"

    # La estructura: headers de equipo (img + nombre) seguidos de bloques de jugadores
    # Cada equipo tiene un header con la imagen del escudo
    content = soup.find("div", class_="col-md-9") or soup.find("main") or soup
    children = list(content.children) if hasattr(content, "children") else []

    # Estrategia: iterar todos los elementos y detectar patrones
    for el in soup.select("[class*='equipo'], [class*='team']"):
        pass  # fallback si la estructura cambia

    # Enfoque robusto: buscar todos los links a jugadores y su contexto
    player_links = soup.select("a[href*='/jugadores/']")
    seen = set()

    for link in player_links:
        href = link.get("href", "")
        name = link.get_text(strip=True)
        if not name or href in seen or "/jugadores/" not in href:
            continue
        seen.add(href)

        # Buscar el contenedor padre que tenga la info
        parent = link.parent
        # Recorrer hacia arriba buscando un contenedor con info
        for _ in range(5):
            if parent is None:
                break
            text = parent.get_text(" ", strip=True)
            if "días)" in text or "día)" in text or "Baja" in text or "Duda" in text:
                break
            parent = parent.parent

        if parent is None:
            continue

        block_text = parent.get_text(" ", strip=True)

        # Detectar estado
        estado = "baja"
        prob = 0
        if "lesionado_box_min" in str(parent):
            estado = "baja"
        elif "duda_box_min" in str(parent):
            estado = "duda"
        elif "disponible_box_min" in str(parent):
            estado = "disponible"

        # Extraer probabilidad: "0%", "30%", etc.
        prob_match = re.search(r"(\d+)\s*%", block_text)
        if prob_match:
            prob = int(prob_match.group(1))

        # Extraer lesión (texto descriptivo antes de "Desde")
        lesion = ""
        desde = ""
        dias = 0
        info_retorno = ""

        # Formato: "Nombre Lesión **Desde DD/MM (X días)Info retorno"
        desde_match = re.search(r"Desde\s+(\d{2}/\d{2})\s*\((\d+)\s*días?\)", block_text)
        if desde_match:
            desde = desde_match.group(1)
            dias = int(desde_match.group(2))

        # Texto de lesión: entre el nombre y "Desde" o "%"
        parts = block_text.split(name, 1)
        if len(parts) > 1:
            after_name = parts[1]
            # Limpiar
            lesion_match = re.search(r"(\d+%\s*)(.*?)(?:\*|Desde)", after_name)
            if lesion_match:
                lesion = lesion_match.group(2).strip()
            else:
                # Intentar sin %
                lesion_match2 = re.search(r"(.*?)(?:\*|Desde)", after_name)
                if lesion_match2:
                    lesion = lesion_match2.group(1).strip()
                    # Quitar porcentaje del inicio
                    lesion = re.sub(r"^\d+%\s*", "", lesion).strip()

        # Info de retorno: "Baja hasta...", "Duda para...", "Disponible para..."
        retorno_match = re.search(
            r"(Baja (?:hasta|confirmada|indefinida)[^\.]*|"
            r"Duda para[^\.]*|"
            r"Disponible para[^\.]*)",
            block_text,
        )
        if retorno_match:
            info_retorno = retorno_match.group(1).strip()

        # Detectar equipo: buscar el header de equipo más cercano antes
        team_header = None
        prev = parent.find_previous(string=re.compile(
            r"(Alavés|Athletic|Atlético|Barcelona|Betis|Celta|"
            r"Cádiz|Elche|Espanyol|Getafe|Girona|Levante|Mallorca|"
            r"Osasuna|Rayo|Real Madrid|Real Sociedad|Sevilla|"
            r"Valencia|Valladolid|Villarreal|Almería|Las Palmas|"
            r"Granada|Leganés|Real Oviedo)"
        ))
        equipo = prev.strip() if prev else "?"

        # URL del parte médico
        parte_link = parent.find("a", href=re.compile(r"/noticias/"))
        url_parte = None
        if parte_link:
            url_parte = parte_link.get("href", "")
            if url_parte and not url_parte.startswith("http"):
                url_parte = BASE_URL + url_parte

        if name:
            results.append({
                "nombre": name,
                "equipo": equipo,
                "estado": estado,
                "probabilidad_titular": prob,
                "lesion": lesion,
                "desde": desde,
                "dias_baja": dias,
                "info_retorno": info_retorno,
                "url_parte": url_parte,
            })

    logger.info("Lesionados encontrados: %d", len(results))
    return results


# ───────────────────────────────────────────────────────────────
# Sancionados
# ───────────────────────────────────────────────────────────────
def scrape_sancionados() -> list[dict]:
    """
    Scrapes https://www.futbolfantasy.com/laliga/sancionados

    Devuelve:
        {
            "nombre": str,
            "motivo": str ("Acumulación de tarjetas" | "Roja directa" | "Doble amarilla"),
            "ciclo": str | None,
            "jornadas_tarjetas": list[str],
        }
    """
    logger.info("Scrapeando sancionados de futbolfantasy.com...")
    soup = _fetch(f"{BASE_URL}/laliga/sancionados")

    results = []
    player_links = soup.select("a[href*='/jugadores/']")
    seen = set()

    for link in player_links:
        href = link.get("href", "")
        name = link.get_text(strip=True)
        if not name or href in seen:
            continue
        seen.add(href)

        # Buscar el contexto alrededor
        parent = link.parent
        for _ in range(4):
            if parent is None:
                break
            text = parent.get_text(" ", strip=True)
            if "tarjetas" in text.lower() or "roja" in text.lower() or "amarilla" in text.lower():
                break
            parent = parent.parent

        if parent is None:
            continue

        block_text = parent.get_text(" ", strip=True)

        motivo = "Desconocido"
        if "Acumulación de tarjetas" in block_text:
            motivo = "Acumulación de tarjetas"
        elif "Roja directa" in block_text:
            motivo = "Roja directa"
        elif "Doble amarilla" in block_text:
            motivo = "Doble amarilla"

        # Ciclo
        ciclo = None
        ciclo_match = re.search(r"(\d+[roe]+ ciclo)", block_text)
        if ciclo_match:
            ciclo = ciclo_match.group(1)

        # Jornadas
        jornadas = re.findall(r"J(\d+)", block_text)

        results.append({
            "nombre": name,
            "motivo": motivo,
            "ciclo": ciclo,
            "jornadas_tarjetas": [f"J{j}" for j in jornadas],
        })

    logger.info("Sancionados encontrados: %d", len(results))
    return results


# ───────────────────────────────────────────────────────────────
# Apercibidos
# ───────────────────────────────────────────────────────────────
def scrape_apercibidos() -> list[dict]:
    """
    Scrapes https://www.futbolfantasy.com/laliga/apercibidos

    Devuelve:
        {
            "nombre": str,
            "ciclo": str,
            "jornadas_tarjetas": list[str],
        }
    """
    logger.info("Scrapeando apercibidos de futbolfantasy.com...")
    soup = _fetch(f"{BASE_URL}/laliga/apercibidos")

    results = []
    player_links = soup.select("a[href*='/jugadores/']")
    seen = set()

    for link in player_links:
        href = link.get("href", "")
        name = link.get_text(strip=True)
        if not name or href in seen:
            continue
        seen.add(href)

        parent = link.parent
        for _ in range(4):
            if parent is None:
                break
            text = parent.get_text(" ", strip=True)
            if "ciclo" in text.lower() or "J" in text:
                break
            parent = parent.parent

        if parent is None:
            continue

        block_text = parent.get_text(" ", strip=True)

        ciclo = None
        ciclo_match = re.search(r"(\d+[roe]+ ciclo)", block_text)
        if ciclo_match:
            ciclo = ciclo_match.group(1)

        jornadas = re.findall(r"J(\d+)", block_text)

        results.append({
            "nombre": name,
            "ciclo": ciclo,
            "jornadas_tarjetas": [f"J{j}" for j in jornadas],
        })

    logger.info("Apercibidos encontrados: %d", len(results))
    return results


# ───────────────────────────────────────────────────────────────
# Función combinada
# ───────────────────────────────────────────────────────────────
def scrape_all() -> dict:
    """
    Ejecuta todos los scrapers de FútbolFantasy.

    Devuelve:
        {
            "fuente": "futbolfantasy.com",
            "lesionados": [...],
            "sancionados": [...],
            "apercibidos": [...],
        }
    """
    return {
        "fuente": "futbolfantasy.com",
        "lesionados": scrape_lesionados(),
        "sancionados": scrape_sancionados(),
        "apercibidos": scrape_apercibidos(),
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    data = scrape_all()
    print(json.dumps(data, indent=2, ensure_ascii=False))
