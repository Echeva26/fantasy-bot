"""
Autopilot del Fantasy Bot
=========================
Orquesta el flujo de forma autónoma para ejecución por cron:

- Modo pre: análisis + ejecución de movimientos (fase 1 ventas + compras)
- Modo post: aceptar ofertas cerradas (fase 2 ventas)
- Modo both: ejecuta pre y luego post
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from prediction.advisor_execute import (
    execute_movements,
    run_aceptar_ofertas,
    run_advisor_pipeline,
)
from prediction.telegram_notify import send_telegram_message

logger = logging.getLogger(__name__)


@dataclass
class PreRunResult:
    report_path: Path
    movimientos: int
    ventas_fase1: int
    compras: int
    errores: list[str]
    message: str = ""
    report_content: str = ""
    report_telegram: str = ""


def _money(value: int | float | None) -> str:
    v = int(value or 0)
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return str(v)


def _notify_if_configured(text: str) -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if bot_token and chat_id:
        send_telegram_message(bot_token, chat_id, text)


def _save_report(report: str, jornada: str | int, output: str | None) -> Path:
    out_path = Path(output) if output else Path(__file__).parent.parent / f"informe_j{jornada}.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def _build_telegram_report(
    *,
    result: dict,
    movimientos: int,
    ventas_fase1: int,
    compras: int,
    errores: list[str],
    analysis_only: bool,
) -> str:
    snapshot = result.get("snapshot", {})
    team = result.get("team_analysis", {})
    plan = result.get("transfer_plan", {})
    jornada = result.get("jornada", "?")

    xp_now = float(team.get("xp_total_once", 0) or 0)
    xp_post = float(plan.get("xp_total_post", xp_now) or 0)
    diff = xp_post - xp_now
    saldo_now = int(team.get("saldo", 0) or 0)
    saldo_post = int(plan.get("saldo_final", saldo_now) or 0)

    lines: list[str] = [
        f"INFORME FANTASY | Jornada {jornada}",
        f"Liga: {snapshot.get('league_name', '?')}",
        f"Manager: {team.get('manager_name', '?')}",
        "",
        "RESUMEN",
        f"- xP once: {xp_now:.1f} -> {xp_post:.1f} ({diff:+.1f})",
        f"- Saldo: {_money(saldo_now)} -> {_money(saldo_post)}",
        f"- Movimientos recomendados: {movimientos}",
    ]

    if not analysis_only:
        lines.append(f"- Ejecutado: ventas {ventas_fase1} | compras {compras}")
        if errores:
            lines.append(f"- Errores de ejecucion: {len(errores)}")

    problemas = team.get("problemas") or []
    if problemas:
        lines.append("")
        lines.append(f"ALERTAS ({min(len(problemas), 5)})")
        for p in problemas[:5]:
            jugador = str(p.get("jugador", "?"))
            problema = str(p.get("problema", "?"))
            lines.append(f"- {jugador}: {problema}")

    movs = plan.get("movimientos") or []
    lines.append("")
    lines.append("PLAN RECOMENDADO")
    if not movs:
        lines.append("- No hay movimientos. Plantilla ya optimizada para la jornada.")
    else:
        for mov in movs[:8]:
            paso = mov.get("paso", "?")
            venta = mov.get("venta") or {}
            compra = mov.get("compra") or {}
            saldo_antes = _money(mov.get("saldo_antes", 0))
            saldo_despues = _money(mov.get("saldo_despues", 0))
            ganancia = float(mov.get("ganancia_xp", 0) or 0)

            if venta and compra:
                lines.append(
                    f"- Paso {paso}: VENDER {venta.get('nombre', '?')} -> "
                    f"COMPRAR {compra.get('nombre', '?')} ({ganancia:+.1f} xP)"
                )
            elif venta:
                lines.append(
                    f"- Paso {paso}: VENDER {venta.get('nombre', '?')} ({ganancia:+.1f} xP)"
                )
            elif compra:
                lines.append(
                    f"- Paso {paso}: COMPRAR {compra.get('nombre', '?')} ({ganancia:+.1f} xP)"
                )
            else:
                lines.append(f"- Paso {paso}: sin acciones")
            lines.append(f"  Saldo: {saldo_antes} -> {saldo_despues}")

        if len(movs) > 8:
            lines.append(f"- ... y {len(movs) - 8} pasos mas")

    once_post = plan.get("once_post") or []
    if once_post:
        lines.append("")
        lines.append(f"ONCE IDEAL ({plan.get('formacion_post', '4-3-3')})")
        for j in once_post[:11]:
            pos = str(j.get("posicion", "?"))
            nombre = str(j.get("nombre", "?"))
            xp = float(j.get("xP", 0) or 0)
            nuevo = " [nuevo]" if j.get("nuevo_fichaje") else ""
            lines.append(f"- {pos} {nombre}: {xp:.1f}xP{nuevo}")

    return "\n".join(lines)


def run_pre_market(
    args: argparse.Namespace,
    *,
    skip_notify: bool = False,
) -> PreRunResult:
    result = run_advisor_pipeline(args)
    snapshot = result["snapshot"]
    transfer_plan = result["transfer_plan"]
    report = result["report"]
    jornada = result["jornada"]

    report_path = _save_report(report, jornada, args.output)
    movimientos = len(transfer_plan.get("movimientos", []))
    ventas_fase1 = 0
    compras = 0
    errores: list[str] = []

    if movimientos and not args.analysis_only:
        ventas_fase1, compras, errores = execute_movements(
            snapshot=snapshot,
            transfer_plan=transfer_plan,
            dry_run=args.dry_run,
        )

    saldo_ini = result["team_analysis"]["saldo"]
    saldo_fin = transfer_plan.get("saldo_final", saldo_ini)
    msg = (
        "Fantasy Autopilot (PRE)\n"
        f"Liga: {snapshot.get('league_name', args.league or '?')}\n"
        f"Movimientos recomendados: {movimientos}\n"
        f"Ventas fase1: {ventas_fase1} | Compras: {compras}\n"
        f"Saldo: {_money(saldo_ini)} -> {_money(saldo_fin)}\n"
        f"Informe: {report_path.name}"
    )
    if errores:
        msg += f"\nErrores: {len(errores)}"
    if not skip_notify:
        _notify_if_configured(msg)

    report_telegram = _build_telegram_report(
        result=result,
        movimientos=movimientos,
        ventas_fase1=ventas_fase1,
        compras=compras,
        errores=errores,
        analysis_only=bool(args.analysis_only),
    )

    return PreRunResult(
        report_path=report_path,
        movimientos=movimientos,
        ventas_fase1=ventas_fase1,
        compras=compras,
        errores=errores,
        message=msg,
        report_content=report,
        report_telegram=report_telegram,
    )


def run_post_market(args: argparse.Namespace) -> int:
    accept_args = SimpleNamespace(league=args.league)
    accepted = run_aceptar_ofertas(accept_args)
    _notify_if_configured(
        "Fantasy Autopilot (POST)\n"
        f"Liga: {args.league}\n"
        f"Ofertas aceptadas: {accepted}"
    )
    return accepted


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Autopilot para cron (pre/post mercado)")
    parser.add_argument(
        "--mode",
        choices=["pre", "post", "both"],
        default="pre",
        help="pre: ejecutar recomendaciones; post: aceptar ofertas; both: ambos",
    )
    parser.add_argument("--league", default=os.getenv("LALIGA_LEAGUE_ID", ""), help="League ID")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm"])
    parser.add_argument("--snapshot", type=str, help="Cargar snapshot desde JSON")
    parser.add_argument("--output", type=str, help="Ruta del informe .md")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin ejecutar API")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Solo analizar/generar informe, sin ejecutar movimientos",
    )
    args = parser.parse_args()

    try:
        if not args.league and not args.snapshot:
            raise RuntimeError("--league o --snapshot es obligatorio para autopilot.")

        if args.mode in ("pre", "both"):
            pre = run_pre_market(args)
            print(
                f"[PRE] informe={pre.report_path} movs={pre.movimientos} "
                f"ventas={pre.ventas_fase1} compras={pre.compras} errores={len(pre.errores)}"
            )
            if pre.errores:
                print("[PRE] Errores:", pre.errores)

        if args.mode in ("post", "both"):
            accepted = run_post_market(args)
            print(f"[POST] Ofertas aceptadas: {accepted}")
    except Exception as exc:
        logger.exception("Autopilot failed: %s", exc)
        _notify_if_configured(f"Fantasy Autopilot ERROR\n{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
