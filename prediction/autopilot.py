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

    return PreRunResult(
        report_path=report_path,
        movimientos=movimientos,
        ventas_fase1=ventas_fase1,
        compras=compras,
        errores=errores,
        message=msg,
        report_content=report,
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
