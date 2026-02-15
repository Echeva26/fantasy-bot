"""
Fantasy Agent Completo
======================
Ejecuta el flujo end-to-end en una sola orden:
  1) Obtener snapshot + predicciones + análisis
  2) Generar informe y mostrar plan
  3) Ejecutar movimientos (ventas fase 1 + compras)
  4) Intentar fase 2 (aceptar ofertas cerradas), opcional

Uso:
    python -m prediction.full_agent --league 016615640
    python -m prediction.full_agent --league 016615640 --yes
    python -m prediction.full_agent --league 016615640 --yes --no-fase2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from prediction.advisor_execute import (
    execute_movements,
    print_plan_summary,
    run_aceptar_ofertas,
    run_advisor_pipeline,
)


def _save_report(report: str, jornada: str | int, output: str | None) -> Path:
    if output:
        out_path = Path(output)
    else:
        out_path = Path(__file__).parent.parent / f"informe_j{jornada}.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def _confirm_execution(auto_yes: bool) -> bool:
    if auto_yes:
        return True
    print(
        "\n  Si confirmas, se ejecutarán: ventas (fase 1) y compras "
        "(pujas/clausulazos) que tengas saldo para pagar."
    )
    respuesta = input("  ¿Ejecutar estos movimientos? (s/n): ").strip().lower()
    return respuesta in ("s", "si", "y", "yes")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Fantasy Agent completo: analizar + ejecutar en un único flujo"
    )
    parser.add_argument("--league", type=str, default="", help="League ID")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--snapshot", type=str, help="Cargar snapshot desde JSON")
    parser.add_argument("--output", type=str, help="Ruta del informe .md")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin ejecutar")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="No pedir confirmación y ejecutar automáticamente",
    )
    parser.add_argument(
        "--no-fase2",
        action="store_true",
        help="No intentar la fase 2 (aceptar ofertas de la liga) al final",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Fantasy Agent — Flujo Completo")
    print("=" * 60)

    result = run_advisor_pipeline(args)
    snapshot = result["snapshot"]
    transfer_plan = result["transfer_plan"]
    report = result["report"]
    jornada = result["jornada"]

    # Completar league en args para fase 2 aunque no se pasara por CLI.
    if not args.league:
        args.league = str(snapshot.get("league_id", ""))

    out_path = _save_report(report, jornada, args.output)
    print(f"\n  Informe guardado: {out_path}")

    print("\n" + "-" * 60)
    print(report[:3500])
    if len(report) > 3500:
        print("...")
    print("-" * 60)

    print_plan_summary(transfer_plan)
    movs = transfer_plan.get("movimientos", [])
    ventas = 0
    compras = 0
    errores: list[str] = []

    if not movs:
        print("\n  Sin movimientos que ejecutar.")
    elif not _confirm_execution(args.yes):
        print("\n  Ejecución cancelada.")
    else:
        print("\n  Ejecutando movimientos...")
        ventas, compras, errores = execute_movements(
            snapshot, transfer_plan, dry_run=args.dry_run
        )
        print(f"\n  Fase 1 ventas (publicados): {ventas}")
        print(f"  Compras: {compras}")
        if errores:
            print("  Errores:", errores)

    if args.no_fase2:
        return

    # Intentar fase 2 si hay ventas recién publicadas o ventas ya abiertas.
    hay_ventas_abiertas = any(
        p.get("en_venta") for p in snapshot.get("mi_equipo", {}).get("plantilla", [])
    )
    if ventas > 0 or hay_ventas_abiertas:
        print("\n  Revisando fase 2 (ofertas cerradas listas para aceptar)...")
        aceptadas = run_aceptar_ofertas(args)
        print(f"  Ofertas aceptadas ahora: {aceptadas}")
        if aceptadas == 0:
            print("  ℹ No hay ofertas cerradas para aceptar en este momento.")


if __name__ == "__main__":
    main()
