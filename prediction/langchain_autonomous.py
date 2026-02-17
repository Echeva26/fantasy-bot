"""
Daemon autónomo para el agente LangChain.

Ejemplo:
  python -m prediction.langchain_autonomous
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from zoneinfo import ZoneInfo

from laliga_fantasy_client import TOKEN_FILE, load_token
from prediction.league_selection import load_selected_league, resolve_league_id
from prediction.langchain_agent import run_agent_objective, run_agent_phase
from prediction.lineup_autoset import autoset_best_lineup
from prediction.market_schedule import build_market_schedule, schedule_message
from prediction.predict import get_next_round, get_sofascore_season_id
from prediction.telegram_notify import GUIDA_RENOVACION_TOKEN, send_telegram_message
from prediction.token_bot import (
    REPORT_PLAN_OBJECTIVE,
    _build_clause_actions_from_report,
    _build_compraventa_message,
    _build_informe_message,
    _execute_cached_actions,
    _extract_agent_report_payload,
    _extract_executable_actions,
    _extract_latest_simulation_payload,
    _now_iso,
    _save_report_plan_cache,
)

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = Path(".langchain_agent_state.json")
MODEL_TYPE = "xgboost"
POLL_SECONDS = 30
MARKET_REFRESH_SECONDS = 300
LINEUP_REFRESH_SECONDS = 900
LINEUP_MINUTES_BEFORE_FIRST_MATCH = 23 * 60 + 55


def _parse_iso_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _parse_market_key_local(raw: str | None, tz: ZoneInfo | timezone) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)
        return dt
    except Exception:
        return None


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def _notify(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        send_telegram_message(token, chat_id, text)


def _token_ok() -> bool:
    token_max_age = float(os.getenv("TOKEN_MAX_AGE_HOURS", "23"))
    status, _ = _token_status(max_age_hours=token_max_age)
    return status == "ok"


def _token_status(max_age_hours: float = 23.0) -> tuple[str, float | None]:
    """
    Devuelve (status, age_h) con status: "ok" | "missing" | "expired" | "invalid".
    """
    if not TOKEN_FILE.exists():
        return "missing", None
    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        saved_at_raw = data.get("saved_at")
        if not saved_at_raw:
            return "invalid", None
        saved_at = datetime.fromisoformat(saved_at_raw)
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600.0
        if age_h >= max_age_hours:
            return "expired", age_h
        return "ok", age_h
    except Exception:
        return "invalid", None


def _maybe_alert_token_issue(state: dict, cooldown_minutes: int) -> dict:
    now_utc = datetime.now(timezone.utc)
    new_state = dict(state)

    token_max_age = float(os.getenv("TOKEN_MAX_AGE_HOURS", "23"))
    status, age_h = _token_status(max_age_hours=token_max_age)
    if status == "ok":
        new_state.pop("token_alert_at", None)
        return new_state

    cooldown = timedelta(minutes=max(1, int(cooldown_minutes)))
    last_alert = _parse_iso_dt(new_state.get("token_alert_at"))
    should_alert = (
        last_alert is None
        or (now_utc - last_alert) >= cooldown
    )

    if should_alert:
        if status == "expired":
            msg = (
                f"Fantasy LangChain Agent: TOKEN CADUCADO (edad ~{(age_h or 0.0):.1f}h)"
                + GUIDA_RENOVACION_TOKEN
            )
        elif status == "missing":
            msg = "Fantasy LangChain Agent: TOKEN AUSENTE" + GUIDA_RENOVACION_TOKEN
        else:
            msg = "Fantasy LangChain Agent: TOKEN INVALIDO" + GUIDA_RENOVACION_TOKEN
        logger.warning(msg.replace("\n", " | "))
        _notify(msg)
        new_state["token_alert_at"] = now_utc.isoformat()

    return new_state


def _maybe_alert_missing_league(state: dict, cooldown_minutes: int) -> dict:
    now_utc = datetime.now(timezone.utc)
    new_state = dict(state)

    cooldown = timedelta(minutes=max(1, int(cooldown_minutes)))
    last_alert = _parse_iso_dt(new_state.get("league_alert_at"))
    should_alert = (
        last_alert is None
        or (now_utc - last_alert) >= cooldown
    )

    if should_alert:
        _notify(
            "Fantasy LangChain Agent: no hay liga seleccionada.\n"
            "Usa el bot de Telegram: /ligas y /liga <nombre>."
        )
        new_state["league_alert_at"] = now_utc.isoformat()

    return new_state


def _maybe_alert_market_unknown(state: dict, cooldown_minutes: int, err: str) -> dict:
    now_utc = datetime.now(timezone.utc)
    new_state = dict(state)

    cooldown = timedelta(minutes=max(1, int(cooldown_minutes)))
    last_alert = _parse_iso_dt(new_state.get("market_alert_at"))
    should_alert = (
        last_alert is None
        or (now_utc - last_alert) >= cooldown
    )

    if should_alert:
        _notify(
            "Fantasy LangChain Agent: no pude calcular la hora del mercado ahora.\n"
            f"Detalle: {err}"
        )
        new_state["market_alert_at"] = now_utc.isoformat()

    return new_state


def _build_objective(phase: str, custom_pre: str, custom_post: str) -> str | None:
    if phase == "pre" and custom_pre.strip():
        return custom_pre.strip()
    if phase == "post" and custom_post.strip():
        return custom_post.strip()
    return None


def _run_phase(phase: str, args: argparse.Namespace, league_id: str) -> dict:
    custom_objective = _build_objective(phase, args.pre_objective, args.post_objective)
    if custom_objective:
        return run_agent_objective(
            league_id=league_id,
            objective=custom_objective,
            model_type=MODEL_TYPE,
            llm_model=args.llm_model,
            temperature=args.temperature,
            max_iterations=args.max_iterations,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    return run_agent_phase(
        league_id=league_id,
        phase=phase,
        model_type=MODEL_TYPE,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def _run_pre_informe_plus_compraventa(
    *,
    args: argparse.Namespace,
    league_id: str,
    league_name: str,
    market_key: str,
) -> dict:
    """
    PRE operativo equivalente a ejecutar /informe y /compraventa en secuencia.
    - Siempre genera plan en dry-run (como /informe).
    - Si el daemon NO está en dry-run, ejecuta el plan resultante (como /compraventa).
    """
    res = run_agent_objective(
        league_id=league_id,
        objective=REPORT_PLAN_OBJECTIVE,
        llm_model=args.llm_model,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        dry_run=True,
        verbose=args.verbose,
    )

    output = str(res.get("output", "") or "").strip() or "Sin salida textual del agente."
    steps = res.get("steps", []) or []
    simulation_payload = _extract_latest_simulation_payload(steps)
    actions, action_source = _extract_executable_actions(steps)
    report_payload = _extract_agent_report_payload(output)
    if not any(str(a.get("tool", "")).strip() == "increase_clause_tool" for a in actions):
        inferred_clause_actions = _build_clause_actions_from_report(report_payload, steps)
        if inferred_clause_actions:
            actions.extend(inferred_clause_actions)
            action_source = (
                "tool_calls+report_clause_fallback"
                if action_source == "tool_calls"
                else "simulate_transfer_plan+report_clause_fallback"
            )

    sim_summary = (
        simulation_payload.get("summary")
        if isinstance(simulation_payload, dict) and isinstance(simulation_payload.get("summary"), dict)
        else {}
    )

    cache_payload = {
        "version": 1,
        "created_at": _now_iso(),
        "league_id": league_id,
        "league_name": league_name,
        "market_key": market_key,
        "llm_model": str(res.get("llm_model", "")),
        "objective": REPORT_PLAN_OBJECTIVE,
        "output": output,
        "action_source": action_source,
        "simulation_summary": sim_summary,
        "actions": actions,
        "actions_count": len(actions),
        "executed_at": "",
    }

    report_text = _build_informe_message(
        league_name=league_name,
        market_key=market_key,
        steps=steps,
        output=output,
        actions=actions,
        action_source=action_source,
        simulation_payload=simulation_payload,
    )

    out = {
        "output": output,
        "steps": steps,
        "actions": actions,
        "action_source": action_source,
        "report_text": report_text,
        "compraventa_text": "",
        "execution_summary": {},
        "executed": False,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        _save_report_plan_cache(cache_payload)
        return out

    if not actions:
        _save_report_plan_cache(cache_payload)
        return out

    summary = _execute_cached_actions(league_id=league_id, actions=actions)
    cache_payload["executed_at"] = _now_iso()
    cache_payload["execution_summary"] = summary
    _save_report_plan_cache(cache_payload)

    out["execution_summary"] = summary
    out["executed"] = True
    out["compraventa_text"] = _build_compraventa_message(
        league_name=league_name,
        market_key=market_key,
        action_source=action_source,
        summary=summary,
    )
    return out


def _market_schedule_from_state(state: dict) -> dict | None:
    close = _parse_iso_dt(state.get("market_close_local"))
    pre = _parse_iso_dt(state.get("market_pre_local"))
    post = _parse_iso_dt(state.get("market_post_local"))
    key = str(state.get("market_key", "")).strip()
    if not (close and pre and post and key):
        return None
    return {
        "market_key": key,
        "close_local": close,
        "pre_local": pre,
        "post_local": post,
        "timezone_name": state.get("market_timezone", ""),
    }


def _refresh_market_schedule(
    state: dict,
    *,
    league_id: str,
    tz_name: str,
    cooldown_seconds: int = MARKET_REFRESH_SECONDS,
) -> tuple[dict, dict | None, str, bool]:
    now_utc = datetime.now(timezone.utc)
    new_state = dict(state)

    last_refresh = _parse_iso_dt(new_state.get("market_refreshed_at"))
    cached = _market_schedule_from_state(new_state)
    if cached and last_refresh and (now_utc - last_refresh).total_seconds() < cooldown_seconds:
        return new_state, cached, "", False

    schedule, err = build_market_schedule(league_id, timezone_name=tz_name)
    new_state["market_refreshed_at"] = now_utc.isoformat()
    if not schedule:
        return new_state, None, err, False

    old_key = str(new_state.get("market_key", ""))
    changed = old_key != schedule["market_key"]

    new_state["market_key"] = schedule["market_key"]
    new_state["market_close_local"] = schedule["close_local"].isoformat()
    new_state["market_pre_local"] = schedule["pre_local"].isoformat()
    new_state["market_post_local"] = schedule["post_local"].isoformat()
    new_state["market_timezone"] = schedule.get("timezone_name", tz_name)
    new_state.pop("market_alert_at", None)

    return new_state, schedule, "", changed


def _refresh_lineup_target(
    state: dict,
    *,
    cooldown_seconds: int = LINEUP_REFRESH_SECONDS,
) -> tuple[dict, str]:
    now_utc = datetime.now(timezone.utc)
    new_state = dict(state)

    last_refresh = _parse_iso_dt(new_state.get("lineup_refreshed_at"))
    if last_refresh and (now_utc - last_refresh).total_seconds() < cooldown_seconds:
        return new_state, ""

    try:
        season_id = get_sofascore_season_id()
        jornada, _, first_match_ts = get_next_round(season_id)
        new_state["lineup_refreshed_at"] = now_utc.isoformat()
        if first_match_ts <= 0:
            new_state.pop("lineup_jornada", None)
            new_state.pop("lineup_first_match_ts", None)
            new_state.pop("lineup_target_ts", None)
            return new_state, "sin timestamp del primer partido"

        target_ts = int(first_match_ts) - (LINEUP_MINUTES_BEFORE_FIRST_MATCH * 60)
        new_state["lineup_jornada"] = int(jornada)
        new_state["lineup_first_match_ts"] = int(first_match_ts)
        new_state["lineup_target_ts"] = int(target_ts)
        new_state["lineup_target_at"] = datetime.fromtimestamp(target_ts, tz=timezone.utc).isoformat()
        return new_state, ""
    except Exception as exc:
        new_state["lineup_refreshed_at"] = now_utc.isoformat()
        return new_state, f"{type(exc).__name__}: {exc}"


def _maybe_run_lineup(
    state: dict,
    *,
    league_id: str,
    dry_run: bool,
) -> tuple[dict, str]:
    new_state = dict(state)
    jornada = new_state.get("lineup_jornada")
    target_ts = int(new_state.get("lineup_target_ts", 0) or 0)
    first_ts = int(new_state.get("lineup_first_match_ts", 0) or 0)

    if not jornada or target_ts <= 0 or first_ts <= 0:
        return new_state, ""

    applied_jornada = int(new_state.get("last_lineup_applied_jornada", 0) or 0)
    if applied_jornada == int(jornada):
        return new_state, ""

    now_ts = int(datetime.now(timezone.utc).timestamp())
    if now_ts < target_ts:
        return new_state, ""
    if now_ts >= first_ts:
        new_state["last_lineup_applied_jornada"] = int(jornada)
        new_state["last_lineup_applied_at"] = datetime.now(timezone.utc).isoformat()
        return new_state, "lineup saltado: jornada ya iniciada"

    result = autoset_best_lineup(
        league_id=league_id,
        model=MODEL_TYPE,
        day_before_only=False,
        after_market_time="00:00",
        timezone_name=os.getenv("TZ", "Europe/Madrid"),
        force=True,
        dry_run=dry_run,
    )

    if result.get("applied") or result.get("dry_run"):
        new_state["last_lineup_applied_jornada"] = int(jornada)
        new_state["last_lineup_applied_at"] = datetime.now(timezone.utc).isoformat()
        msg = (
            "Fantasy LangChain LINEUP\n"
            f"Jornada: {jornada}\n"
            f"Objetivo: 23h55 antes del inicio"
        )
        return new_state, msg

    reason = str(result.get("reason", "no aplicada"))
    return new_state, f"lineup no aplicada: {reason}"


def run_daemon(args: argparse.Namespace, stop_event: Event | None = None) -> None:
    tz_name = os.getenv("TZ", "Europe/Madrid")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning("TZ inválida (%s), usando UTC.", tz_name)
        tz = timezone.utc

    state_path = Path(args.state_file)
    state = _load_state(state_path)

    startup_league = resolve_league_id(args.league)
    startup_sel = load_selected_league() or {}
    startup_name = startup_sel.get("league_name", "")
    if startup_name:
        startup_league_desc = startup_name
    elif startup_league:
        startup_league_desc = "definida por configuracion avanzada"
    else:
        startup_league_desc = "pendiente (usa /ligas y /liga en Telegram)"

    _notify(
        "Fantasy LangChain daemon iniciado\n"
        f"Liga: {startup_league_desc}\n"
        "PRE/POST: automáticos a 10 min antes/después del cierre real de mercado\n"
        "Alineación: 23h55 antes del inicio de jornada"
    )
    logger.info(
        "LangChain daemon iniciado liga_inicial=%s llm=%s model_fijo=%s",
        startup_league,
        args.llm_model,
        MODEL_TYPE,
    )

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Daemon LangChain detenido por stop_event.")
            return
        try:
            now_local = datetime.now(tz)
            now_utc = datetime.now(timezone.utc)
            state = _maybe_alert_token_issue(state, args.token_alert_cooldown_minutes)

            league_id = resolve_league_id(args.league)
            has_token = _token_ok()

            prev_league = str(state.get("active_league_id", ""))
            if league_id and league_id != prev_league:
                sel = load_selected_league() or {}
                league_name = sel.get("league_name", "")
                if league_name:
                    _notify(f"Fantasy LangChain Agent: liga activa\n{league_name}")

                # Reset de ciclo por cambio de liga
                for key in (
                    "last_pre_market_key",
                    "last_post_market_key",
                    "pending_post_market_key",
                    "pending_post_close_local",
                    "pending_post_local",
                    "pending_post_set_at",
                    "market_key",
                    "market_close_local",
                    "market_pre_local",
                    "market_post_local",
                    "market_refreshed_at",
                    "lineup_jornada",
                    "lineup_first_match_ts",
                    "lineup_target_ts",
                    "lineup_target_at",
                    "lineup_refreshed_at",
                    "last_lineup_applied_jornada",
                ):
                    state.pop(key, None)

                state["active_league_id"] = league_id
                state["active_league_name"] = league_name

            if not league_id:
                state = _maybe_alert_missing_league(state, args.token_alert_cooldown_minutes)
                _save_state(state_path, state)
                if stop_event:
                    stop_event.wait(timeout=POLL_SECONDS)
                else:
                    time.sleep(POLL_SECONDS)
                continue

            state.pop("league_alert_at", None)

            if not has_token:
                _save_state(state_path, state)
                if stop_event:
                    stop_event.wait(timeout=POLL_SECONDS)
                else:
                    time.sleep(POLL_SECONDS)
                continue

            # 1) Refrescar horario real de mercado (liga)
            state, schedule, market_err, market_changed = _refresh_market_schedule(
                state,
                league_id=league_id,
                tz_name=tz_name,
            )

            if not schedule:
                logger.warning("No se pudo calcular horario de mercado: %s", market_err)
                state = _maybe_alert_market_unknown(
                    state,
                    args.token_alert_cooldown_minutes,
                    market_err,
                )
            else:
                market_key = schedule["market_key"]
                close_local = schedule["close_local"]
                pre_local = schedule["pre_local"]
                post_local = schedule["post_local"]

                if market_changed:
                    _notify("Horario de mercado detectado\n" + schedule_message(schedule))

                # PRE: 10 min antes del cierre y antes del cierre
                if (
                    now_local >= pre_local
                    and now_local < close_local
                    and str(state.get("last_pre_market_key", "")) != market_key
                ):
                    logger.info("Lanzando fase PRE (market_key=%s)...", market_key)
                    selected = load_selected_league() or {}
                    league_name = str(selected.get("league_name", "")).strip() or league_id

                    if args.pre_objective.strip():
                        # Modo avanzado: objetivo PRE custom.
                        res = _run_phase("pre", args, league_id)
                        output = str(res.get("output", "") or "")
                        _notify(
                            "Fantasy LangChain (PRE) completado\n"
                            f"Tools: {len(res.get('steps', []))}\n"
                            f"Resumen:\n{output[:1200]}"
                        )
                    else:
                        # Flujo estándar solicitado: PRE = /informe + /compraventa.
                        pre_run = _run_pre_informe_plus_compraventa(
                            args=args,
                            league_id=league_id,
                            league_name=league_name,
                            market_key=market_key,
                        )
                        output = str(pre_run.get("output", "") or "")
                        report_text = str(pre_run.get("report_text", "") or "").strip()
                        if report_text:
                            _notify(report_text)
                        if pre_run.get("executed"):
                            resume_text = str(pre_run.get("compraventa_text", "") or "").strip()
                            if resume_text:
                                _notify(resume_text)
                        elif args.dry_run:
                            _notify(
                                "Fantasy LangChain (PRE)\n"
                                "Dry-run activo: plan generado y cacheado sin ejecución real."
                            )
                        else:
                            _notify(
                                "Fantasy LangChain (PRE)\n"
                                "El informe no dejó acciones ejecutables para este ciclo."
                            )

                    state["last_pre_market_key"] = market_key
                    state["last_pre_run_at"] = now_utc.isoformat()
                    state["last_pre_output"] = output[:2000]
                    # POST debe ejecutarse para el MISMO ciclo que lanzó PRE,
                    # aunque el mercado refresque y cambie de market_key.
                    state["pending_post_market_key"] = market_key
                    state["pending_post_close_local"] = close_local.isoformat()
                    state["pending_post_local"] = post_local.isoformat()
                    state["pending_post_set_at"] = now_utc.isoformat()

            # POST: ejecutar sobre el ciclo pendiente generado por PRE.
            pending_post_key = str(state.get("pending_post_market_key", "")).strip()
            if not pending_post_key:
                # Recuperación defensiva: si PRE sí corrió pero no quedó pendiente
                # (estado generado por versiones anteriores), rearmar POST.
                last_pre_key = str(state.get("last_pre_market_key", "")).strip()
                last_post_key = str(state.get("last_post_market_key", "")).strip()
                if last_pre_key and last_pre_key != last_post_key:
                    recovered_close_local = _parse_market_key_local(last_pre_key, tz)
                    if recovered_close_local:
                        recovered_post_local = recovered_close_local + timedelta(minutes=10)
                        state["pending_post_market_key"] = last_pre_key
                        state["pending_post_close_local"] = recovered_close_local.isoformat()
                        state["pending_post_local"] = recovered_post_local.isoformat()
                        state["pending_post_set_at"] = now_utc.isoformat()
                        pending_post_key = last_pre_key
                        logger.info(
                            "Recuperado POST pendiente desde last_pre_market_key=%s",
                            last_pre_key,
                        )

            pending_post_key = str(state.get("pending_post_market_key", "")).strip()
            pending_post_close_local = _parse_iso_dt(state.get("pending_post_close_local"))
            pending_post_local = _parse_iso_dt(state.get("pending_post_local"))
            # Compatibilidad defensiva con estados antiguos sin close guardado.
            if pending_post_local and not pending_post_close_local:
                pending_post_close_local = pending_post_local - timedelta(minutes=10)
            if (
                pending_post_key
                and pending_post_close_local
                and pending_post_local
                and now_local >= pending_post_close_local
                and now_local >= pending_post_local
                and str(state.get("last_post_market_key", "")) != pending_post_key
            ):
                logger.info(
                    "Lanzando fase POST pendiente (pending_market_key=%s)...",
                    pending_post_key,
                )
                res = _run_phase("post", args, league_id)
                output = res.get("output", "")
                state["last_post_market_key"] = pending_post_key
                state["last_post_run_at"] = now_utc.isoformat()
                state["last_post_output"] = output[:2000]
                state.pop("pending_post_market_key", None)
                state.pop("pending_post_close_local", None)
                state.pop("pending_post_local", None)
                state.pop("pending_post_set_at", None)
                _notify(
                    "Fantasy LangChain (POST) completado\n"
                    f"Ciclo: {pending_post_key}\n"
                    f"Tools: {len(res.get('steps', []))}\n"
                    f"Resumen:\n{output[:1200]}"
                )

            # 2) Programación de alineación exacta: 23h55 antes
            state, lineup_err = _refresh_lineup_target(state)
            if lineup_err:
                logger.info("Lineup target info: %s", lineup_err)
            state, lineup_msg = _maybe_run_lineup(
                state,
                league_id=league_id,
                dry_run=args.dry_run,
            )
            if lineup_msg.startswith("Fantasy LangChain LINEUP"):
                _notify(lineup_msg)
            elif lineup_msg:
                logger.info(lineup_msg)

            _save_state(state_path, state)
            if stop_event:
                stop_event.wait(timeout=POLL_SECONDS)
            else:
                time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Daemon LangChain detenido por usuario.")
            return
        except Exception as exc:
            logger.exception("Error en bucle LangChain daemon: %s", exc)
            _notify(f"Fantasy LangChain daemon ERROR\n{type(exc).__name__}: {exc}")
            if stop_event:
                stop_event.wait(timeout=POLL_SECONDS)
            else:
                time.sleep(POLL_SECONDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daemon autónomo con agente LangChain")
    parser.add_argument(
        "--league",
        default="",
        help="Liga fija opcional (modo avanzado). Si se omite, usa la selección de Telegram.",
    )
    parser.add_argument(
        "--state-file",
        default=os.getenv("LANGCHAIN_STATE_FILE", str(DEFAULT_STATE_FILE)),
    )
    parser.add_argument(
        "--token-alert-cooldown-minutes",
        type=int,
        default=int(os.getenv("TOKEN_ALERT_COOLDOWN_MINUTES", "360")),
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("LANGCHAIN_LLM_MODEL", "gpt-5-mini"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LANGCHAIN_TEMPERATURE", "0.1")),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=int(os.getenv("LANGCHAIN_MAX_ITERATIONS", "20")),
    )
    parser.add_argument(
        "--pre-objective",
        default=os.getenv("LANGCHAIN_PRE_OBJECTIVE", ""),
        help="Objetivo custom para fase PRE.",
    )
    parser.add_argument(
        "--post-objective",
        default=os.getenv("LANGCHAIN_POST_OBJECTIVE", ""),
        help="Objetivo custom para fase POST.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_daemon(args=args, stop_event=None)


if __name__ == "__main__":
    main()
