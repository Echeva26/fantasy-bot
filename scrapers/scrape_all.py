"""
Orquestador de scrapers
========================
Ejecuta todos los scrapers y genera un JSON combinado con datos externos
útiles para tomar decisiones en LaLiga Fantasy.

Uso:
    python -m scrapers.scrape_all
    python -m scrapers.scrape_all --output datos_externos.json

Desde código:
    from scrapers.scrape_all import run_all_scrapers, save_scrape_data
    data = run_all_scrapers()
    save_scrape_data(data)
"""

import json
import logging
import pathlib
import sys
import os
from datetime import datetime, timezone

# Asegurar que el directorio raíz está en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scrapers import futbolfantasy, sofascore, market_values

logger = logging.getLogger(__name__)

SCRAPES_DIR = pathlib.Path("scrapes")


def run_all_scrapers() -> dict:
    """
    Ejecuta todos los scrapers disponibles y devuelve un dict combinado.

    Cada scraper se ejecuta de forma independiente; si uno falla,
    los demás siguen funcionando.

    Devuelve:
        {
            "scraped_at": str (ISO datetime),
            "futbolfantasy": {
                "fuente": "futbolfantasy.com",
                "lesionados": [...],
                "sancionados": [...],
                "apercibidos": [...],
            },
            "understat": {
                "fuente": "understat.com",
                "temporada": "2025/2026",
                "jugadores": [...],
                "equipos": [...],
            },
            "analiticafantasy": {
                "fuente": "analiticafantasy.com",
                "mercado": {"subidas": [...], "bajadas": [...]},
            },
        }
    """
    logger.info("Ejecutando todos los scrapers...")
    result = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── FútbolFantasy ─────────────────────────────────────────
    try:
        logger.info("─" * 50)
        logger.info("Scraper: FútbolFantasy")
        data = futbolfantasy.scrape_all()
        result["futbolfantasy"] = data
        logger.info(
            "  OK: %d lesionados, %d sancionados, %d apercibidos",
            len(data.get("lesionados", [])),
            len(data.get("sancionados", [])),
            len(data.get("apercibidos", [])),
        )
    except Exception as e:
        logger.error("  FALLO: futbolfantasy — %s", e)
        result["futbolfantasy"] = {"error": str(e)}

    # ── Sofascore ──────────────────────────────────────────────
    try:
        logger.info("─" * 50)
        logger.info("Scraper: Sofascore")
        data = sofascore.scrape_all()
        result["sofascore"] = data
        logger.info(
            "  OK: %d jugadores, %d equipos",
            len(data.get("jugadores", [])),
            len(data.get("clasificacion", [])),
        )
    except Exception as e:
        logger.error("  FALLO: sofascore — %s", e)
        result["sofascore"] = {"error": str(e)}

    # ── Valores de Mercado (FútbolFantasy Analytics) ─────────
    try:
        logger.info("─" * 50)
        logger.info("Scraper: Valores de mercado (subidas/bajadas)")
        data = market_values.scrape_all()
        result["market_values"] = data
        mercado = data.get("mercado", {})
        logger.info(
            "  OK: %d subidas, %d bajadas, %d sin cambio",
            len(mercado.get("subidas", [])),
            len(mercado.get("bajadas", [])),
            mercado.get("sin_cambio", 0),
        )
    except Exception as e:
        logger.error("  FALLO: market_values — %s", e)
        result["market_values"] = {"error": str(e)}

    logger.info("─" * 50)
    logger.info("Scraping completado.")
    return result


def save_scrape_data(data: dict, output: pathlib.Path | None = None) -> pathlib.Path:
    """
    Guarda el resultado del scraping en un JSON.

    Si no se pasa output, se guarda en scrapes/{timestamp}.json.
    """
    if output is None:
        SCRAPES_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output = SCRAPES_DIR / f"{ts}.json"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Datos guardados en %s", output)
    return output


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Scraper de datos externos para LaLiga Fantasy")
    parser.add_argument("--output", "-o", type=str, help="Ruta del archivo JSON de salida")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  LaLiga Fantasy — Scraper de datos externos")
    print("=" * 60)

    data = run_all_scrapers()

    output_path = pathlib.Path(args.output) if args.output else None
    path = save_scrape_data(data, output_path)

    # Resumen
    print()
    print("=" * 60)
    print("  RESUMEN")
    print("=" * 60)

    ff = data.get("futbolfantasy", {})
    if "error" not in ff:
        print(f"  FútbolFantasy:")
        print(f"    Lesionados:  {len(ff.get('lesionados', []))}")
        print(f"    Sancionados: {len(ff.get('sancionados', []))}")
        print(f"    Apercibidos: {len(ff.get('apercibidos', []))}")
    else:
        print(f"  FútbolFantasy: ERROR — {ff['error']}")

    ss = data.get("sofascore", {})
    if "error" not in ss:
        print(f"  Sofascore:")
        print(f"    Jugadores:   {len(ss.get('jugadores', []))}")
        print(f"    Clasificación: {len(ss.get('clasificacion', []))}")
    else:
        print(f"  Sofascore: ERROR — {ss['error']}")

    mv = data.get("market_values", {})
    if "error" not in mv:
        mercado = mv.get("mercado", {})
        print(f"  Valores de mercado:")
        print(f"    Subidas:     {len(mercado.get('subidas', []))}")
        print(f"    Bajadas:     {len(mercado.get('bajadas', []))}")
        print(f"    Sin cambio:  {mercado.get('sin_cambio', 0)}")
    else:
        print(f"  Valores de mercado: ERROR — {mv['error']}")

    print()
    print(f"  Guardado en: {path}")
    print()


if __name__ == "__main__":
    main()
