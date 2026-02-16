"""
Fantasy Advisor — Ejecución automática de movimientos
======================================================
Genera el informe de predicción, muestra los movimientos recomendados
y permite ejecutarlos en la API de LaLiga Fantasy tras confirmación.

Flujo:
  1. Genera predicciones y plan de movimientos (igual que advisor)
  2. Muestra el informe y los movimientos recomendados
  3. Pregunta: ¿Ejecutar estos movimientos? (s/n)
  4. Si sí: ejecuta vía API
     - Fase 1 venta: publicar en el mercado
     - Compras: puja (mercado) o clausulazo
     - Fase 2 venta: tras cierre del mercado (--aceptar-ofertas)

IMPORTANTE: La venta tiene dos fases:
  - Fase 1: publicar jugador en el mercado (ahora)
  - Fase 2: tras el cierre del mercado, aceptar la oferta de la liga
            → python -m prediction.advisor_execute --aceptar-ofertas

Uso:
    python -m prediction.advisor_execute
    python -m prediction.advisor_execute --league 016615640
    python -m prediction.advisor_execute --aceptar-ofertas
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction.advisor import (
    analyze_available_players,
    analyze_my_team,
    clausulazos_available,
    generate_report,
    get_predictions,
    load_snapshot,
    load_snapshot_from_file,
    simulate_transfer_plan,
)
from prediction.advisor import _format_money as fm

logger = logging.getLogger(__name__)
MODEL_TYPE = "xgboost"


def _format_money(amount: int) -> str:
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"{amount / 1_000:.0f}K"
    return str(amount)


def run_advisor_pipeline(args: argparse.Namespace) -> dict:
    """Ejecuta el pipeline del advisor (snapshot, predicciones, análisis, plan)."""
    if args.snapshot:
        snapshot = load_snapshot_from_file(args.snapshot)
    else:
        snapshot = load_snapshot(args.league)
    pred_df, first_match_ts = get_predictions(MODEL_TYPE)
    jornada = int(pred_df["jornada"].iloc[0]) if not pred_df.empty else "?"
    clausulazos_ok, horas_al_partido = clausulazos_available(first_match_ts)
    team_analysis = analyze_my_team(snapshot, pred_df)
    mi_equipo = snapshot["mi_equipo"]
    available = analyze_available_players(
        snapshot, pred_df, mi_equipo["saldo_disponible"]
    )
    transfer_plan = simulate_transfer_plan(
        team_analysis, available, snapshot, allow_clausulazos=clausulazos_ok
    )
    report = generate_report(
        team_analysis, transfer_plan, snapshot, pred_df, MODEL_TYPE,
        clausulazos_ok=clausulazos_ok, horas_al_partido=horas_al_partido,
    )
    return {
        "snapshot": snapshot,
        "pred_df": pred_df,
        "jornada": jornada,
        "team_analysis": team_analysis,
        "transfer_plan": transfer_plan,
        "report": report,
    }


def print_plan_summary(transfer_plan: dict) -> None:
    """Imprime resumen del plan en consola."""
    movs = transfer_plan.get("movimientos", [])
    if not movs:
        print("\n  No hay movimientos recomendados.")
        return
    print("\n  ─── PLAN DE MOVIMIENTOS ───")
    for mov in movs:
        v = mov.get("venta")
        c = mov.get("compra")
        if v:
            print(f"  {mov['paso']}. VENDER: {v['nombre']} ({v['posicion']})")
            print(f"       Ingresos: {fm(v['ingresos'])} | motivo: {v['motivo']}")
        if c:
            print(f"      COMPRAR: {c['nombre']} ({c['posicion']}, {c['equipo_real']})")
            print(f"       Coste: {fm(c['coste'])} ({c['tipo']}) | xP: {c['xP']:.1f}")
        print(f"       Saldo: {fm(mov['saldo_antes'])} → {fm(mov['saldo_despues'])}")
        print()


def execute_movements(
    snapshot: dict,
    transfer_plan: dict,
    dry_run: bool = False,
) -> tuple[int, int, list[str]]:
    """
    Ejecuta los movimientos vía API.
    Returns:
        (ventas_fase1, compras, errores)
    """
    from laliga_fantasy_client import LaLigaFantasyClient

    league_id = snapshot.get("league_id", "") or snapshot.get("mi_equipo", {}).get("league_id")
    mi_equipo = snapshot["mi_equipo"]

    if not league_id:
        league_id = os.getenv("LALIGA_LEAGUE_ID", "")
    if not league_id:
        raise RuntimeError("league_id no encontrado en snapshot ni en LALIGA_LEAGUE_ID.")

    client = LaLigaFantasyClient.from_saved_token(league_id=league_id)
    client.league_id = str(league_id)

    saldo_disponible = mi_equipo.get("saldo_disponible", 0)
    ventas_fase1 = 0
    compras = 0
    errores: list[str] = []
    compras_saltadas_saldo = 0  # Para informar al usuario
    compras_saltadas_faltan_datos = 0

    # Jugadores que ya están en venta: plantilla + mercado (no volver a publicar)
    team_id = str(mi_equipo.get("team_id", ""))
    ya_en_venta = {
        p["player_id"]
        for p in mi_equipo.get("plantilla", [])
        if p.get("en_venta")
    }
    # También los que aparecen en el mercado como vendidos por nosotros
    for item in snapshot.get("mercado", []):
        if str(item.get("vendedor_team_id", "")) == team_id:
            pid = item.get("player_id")
            if pid:
                ya_en_venta.add(pid)

    for mov in transfer_plan.get("movimientos", []):
        v = mov.get("venta")
        c = mov.get("compra")
        coste = c.get("coste", 0) if c else 0

        # Fase 1 venta: publicar en el mercado (requiere player_team_id)
        if v and not dry_run:
            if v.get("player_id") in ya_en_venta:
                print(f"    [SKIP] Ya en venta: {v['nombre']}")
                # Ya está publicada; seguimos para intentar la compra de este paso.
            ptid = v.get("player_team_id")
            if v.get("player_id") not in ya_en_venta and not ptid:
                errores.append(f"Venta de {v['nombre']}: falta player_team_id en snapshot")
                continue
            try:
                if v.get("player_id") not in ya_en_venta:
                    precio_base = v.get("precio_publicacion") or int(v.get("valor_mercado", 0) * 0.9)
                    # La API puede rechazar precios < valor_mercado; usar al menos valor_mercado
                    valor_mercado = v.get("valor_mercado", 0) or 0
                    precio = max(precio_base, valor_mercado) if valor_mercado else precio_base
                    client.sell_player_phase1(ptid, precio)
                    ventas_fase1 += 1
                    print(f"    [OK] Publicado en mercado: {v['nombre']} a {_format_money(precio)}")
            except Exception as e:
                msg = f"Error vendiendo {v['nombre']}: {e}"
                errores.append(msg)
                print(f"    [ERROR] {msg}")

        # Comprar: si tenemos saldo suficiente (el dinero de ventas llega tras fase 2, no lo sumamos)
        if not c or dry_run:
            continue

        # Verificar datos necesarios
        if c.get("tipo") == "clausulazo" and not c.get("player_team_id"):
            compras_saltadas_faltan_datos += 1
            continue
        if c.get("tipo") != "clausulazo" and not c.get("market_item_id"):
            compras_saltadas_faltan_datos += 1
            continue

        if coste > saldo_disponible:
            compras_saltadas_saldo += 1
            print(f"    [SKIP] Comprar {c.get('nombre', '?')}: saldo insuficiente (tienes {_format_money(saldo_disponible)}, cuesta {_format_money(coste)})")
            continue

        try:
            if c.get("tipo") == "clausulazo":
                ptid = c.get("player_team_id")
                client.buy_player_clausulazo(
                    ptid,
                    buyout_clause_to_pay=coste,
                )
                compras += 1
                saldo_disponible -= coste
                print(f"    [OK] Clausulazo: {c['nombre']} de {c.get('propietario', '?')}")
            else:
                mid = c.get("market_item_id")
                client.buy_player_bid(
                    mid,
                    coste,
                    player_id=c.get("player_id"),
                )
                compras += 1
                saldo_disponible -= coste
                print(f"    [OK] Puja: {c['nombre']} por {_format_money(coste)}")
        except Exception as e:
            msg = f"Error comprando {c['nombre']}: {e}"
            errores.append(msg)
            print(f"    [ERROR] {msg}")

    # Explicar por qué no se ejecutaron compras
    if compras == 0 and (compras_saltadas_saldo or compras_saltadas_faltan_datos):
        print()
        if compras_saltadas_saldo:
            print("  ℹ Compras no ejecutadas: saldo insuficiente.")
            print("    El dinero de las ventas llegará tras ejecutar --aceptar-ofertas")
            print("    cuando cierre el mercado. Hasta entonces no puedes comprar con ese dinero.")
        if compras_saltadas_faltan_datos:
            print("  ℹ Compras bloqueadas: faltan market_item_id o player_team_id en el plan.")

    return ventas_fase1, compras, errores


def run_aceptar_ofertas(args: argparse.Namespace) -> int:
    """
    Fase 2 de ventas: aceptar la oferta de la liga para jugadores
    que publicamos y cuya subasta ya ha cerrado.
    """
    from laliga_fantasy_client import LaLigaFantasyClient, get_league_snapshot

    league_id = args.league or os.getenv("LALIGA_LEAGUE_ID", "")
    if not league_id:
        raise RuntimeError("--league o LALIGA_LEAGUE_ID requerido.")

    client = LaLigaFantasyClient.from_saved_token(league_id=league_id)
    client.league_id = str(league_id)

    snapshot = get_league_snapshot(client)
    mi_equipo = snapshot["mi_equipo"]

    now = datetime.now(timezone.utc)
    aceptados = 0

    for p in mi_equipo.get("plantilla", []):
        if not p.get("en_venta"):
            continue
        expiracion = p.get("venta_expiracion")
        if not expiracion:
            continue
        try:
            exp_dt = datetime.fromisoformat(expiracion.replace("Z", "+00:00"))
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        if exp_dt > now:
            continue  # Aún no ha cerrado la subasta

        mpid = p.get("market_player_id")
        oid = p.get("offer_id")
        # La oferta de la liga: necesitamos market_player_id y offer_id
        # offer_id puede venir de ofertas_recibidas al hacer match por player
        if not mpid:
            for of in mi_equipo.get("ofertas_recibidas", []):
                if of.get("player_id") == p.get("player_id"):
                    mpid = of.get("market_player_id")
                    oid = of.get("offer_id")
                    break
        if not mpid or not oid:
            print(f"  [SKIP] {p['nombre']}: falta market_player_id u offer_id para aceptar")
            continue

        try:
            client.sell_player_phase2_accept_league_offer(mpid, oid)
            aceptados += 1
            print(f"  [OK] Oferta aceptada: {p['nombre']}")
        except Exception as e:
            print(f"  [ERROR] {p['nombre']}: {e}")

    return aceptados


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Fantasy Advisor: predicción + ejecución de movimientos"
    )
    parser.add_argument("--league", type=str, default="", help="League ID")
    parser.add_argument("--snapshot", type=str, help="Cargar snapshot desde JSON")
    parser.add_argument("--output", type=str, help="Ruta del informe .md")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin ejecutar")
    parser.add_argument(
        "--aceptar-ofertas",
        action="store_true",
        help="Fase 2: aceptar ofertas de la liga tras cierre del mercado",
    )
    args = parser.parse_args()

    if args.aceptar_ofertas:
        print("\n  Fase 2: Aceptar ofertas de la liga (ventas publicadas)...")
        n = run_aceptar_ofertas(args)
        print(f"\n  Aceptadas: {n}")
        return

    print()
    print("=" * 60)
    print("  Fantasy Advisor — Ejecución")
    print("=" * 60)

    result = run_advisor_pipeline(args)
    snapshot = result["snapshot"]
    transfer_plan = result["transfer_plan"]
    report = result["report"]
    jornada = result["jornada"]

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n  Informe guardado: {args.output}")
    else:
        output_dir = Path(__file__).parent.parent
        out_path = output_dir / f"informe_j{jornada}.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"\n  Informe guardado: {out_path}")

    print("\n" + "-" * 60)
    print(report[:3500])
    if len(report) > 3500:
        print("...")
    print("-" * 60)

    print_plan_summary(transfer_plan)

    movs = transfer_plan.get("movimientos", [])
    if not movs:
        print("\n  Sin movimientos que ejecutar.")
        return

    print("\n  Si confirmas, se ejecutarán: ventas (fase 1) y compras (pujas/clausulazos) que tengas saldo para pagar.")
    respuesta = input("  ¿Ejecutar estos movimientos? (s/n): ").strip().lower()
    if respuesta not in ("s", "si", "y", "yes"):
        print("\n  Ejecución cancelada.")
        return

    print("\n  Ejecutando movimientos...")
    ventas, compras, errores = execute_movements(snapshot, transfer_plan, dry_run=args.dry_run)
    print(f"\n  Fase 1 ventas (publicados): {ventas}")
    print(f"  Compras: {compras}")
    if errores:
        print("  Errores:", errores)

    if ventas > 0:
        print(
            "\n  IMPORTANTE: Tras el cierre del mercado, ejecuta:"
        )
        league = snapshot.get("league_id", args.league or "")
        print(
            f"    python -m prediction.advisor_execute --aceptar-ofertas --league {league}"
        )
        print("  para aceptar las ofertas de la liga (fase 2 de venta).")


if __name__ == "__main__":
    main()
