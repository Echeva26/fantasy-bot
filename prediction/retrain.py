"""
Job de reentrenamiento periódico del modelo xP.

Mantiene frescos los artefactos del pipeline ML (raw_dataset.json,
features.csv, xgboost_model.pkl) reejecutando en subprocesos:

    python -m prediction.collect_data
    python -m prediction.features
    python -m prediction.train

Disparo (una vez al día, tras RETRAIN_TIME):
- Falta algún artefacto.
- Hay una jornada completada nueva respecto al dataset (weekNumber de la API).
- Si la API no responde, fallback por edad: dataset más viejo que
  RETRAIN_MAX_AGE_DAYS.

El estado (última ejecución, último error, jornada cubierta) se persiste en
``.retrain_state.json``. Un fallo se notifica por Telegram y se reintenta tras
RETRAIN_ERROR_RETRY_SECONDS sin tumbar el daemon.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from prediction.scrape_freshness import repo_root

logger = logging.getLogger(__name__)

STATE_FILE = repo_root() / ".retrain_state.json"
DATA_DIR = repo_root() / "prediction" / "data"
MODELS_DIR = repo_root() / "prediction" / "models"

DEFAULT_RETRAIN_TIME = "05:30"
DEFAULT_MAX_AGE_DAYS = 7
DEFAULT_STEP_TIMEOUT_SECONDS = 1800
DEFAULT_ERROR_RETRY_SECONDS = 21600
DEFAULT_POLL_SECONDS = 300

PIPELINE_STEPS: list[list[str]] = [
    [sys.executable, "-m", "prediction.collect_data"],
    [sys.executable, "-m", "prediction.features"],
    [sys.executable, "-m", "prediction.train"],
]

Runner = Callable[[list[str], Path, int], "subprocess.CompletedProcess[str]"]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def retrain_enabled() -> bool:
    return _env_bool("RETRAIN_ENABLED", True)


def retrain_time() -> tuple[int, int]:
    raw = os.getenv("RETRAIN_TIME", DEFAULT_RETRAIN_TIME).strip()
    try:
        hour, minute = raw.split(":")
        return max(0, min(23, int(hour))), max(0, min(59, int(minute)))
    except Exception:
        hour, minute = DEFAULT_RETRAIN_TIME.split(":")
        return int(hour), int(minute)


def max_age_days() -> int:
    return _env_int("RETRAIN_MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS)


def step_timeout_seconds() -> int:
    return _env_int("RETRAIN_STEP_TIMEOUT_SECONDS", DEFAULT_STEP_TIMEOUT_SECONDS)


def error_retry_seconds() -> int:
    return _env_int("RETRAIN_ERROR_RETRY_SECONDS", DEFAULT_ERROR_RETRY_SECONDS)


def artifact_paths() -> dict[str, Path]:
    return {
        "raw_dataset": DATA_DIR / "raw_dataset.json",
        "features": DATA_DIR / "features.csv",
        "model": MODELS_DIR / "xgboost_model.pkl",
        "meta": MODELS_DIR / "xgboost_meta.json",
    }


def load_state(path: Path | None = None) -> dict:
    state_path = path or STATE_FILE
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict, path: Path | None = None) -> None:
    state_path = path or STATE_FILE
    try:
        state_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("No se pudo guardar %s: %s", state_path, exc)


def dataset_max_jornada(path: Path | None = None) -> int | None:
    """Máxima jornada presente en raw_dataset.json (None si no se puede leer)."""
    dataset_path = path or artifact_paths()["raw_dataset"]
    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        jornadas = [
            int(row["jornada"])
            for row in data.get("rows", [])
            if str(row.get("jornada", "")).strip().isdigit()
        ]
        return max(jornadas) if jornadas else None
    except Exception:
        return None


def dataset_age_days(path: Path | None = None, *, now: datetime | None = None) -> float | None:
    dataset_path = path or artifact_paths()["raw_dataset"]
    try:
        modified = datetime.fromtimestamp(dataset_path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (current - modified).total_seconds() / 86400.0)


def fetch_completed_week() -> int | None:
    """Última jornada completada según la API (weekNumber actual - 1)."""
    try:
        from laliga_fantasy_client import LaLigaFantasyPublic

        week = LaLigaFantasyPublic().get_current_week() or {}
        number = int(week.get("weekNumber") or 0)
        return number - 1 if number > 1 else None
    except Exception as exc:
        logger.info("No se pudo consultar la jornada actual: %s", exc)
        return None


_UNSET = object()


def retrain_due(
    state: dict,
    *,
    now: datetime | None = None,
    completed_week: object = _UNSET,
    covered_jornada: object = _UNSET,
    age_days: object = _UNSET,
) -> tuple[bool, str]:
    """
    Decide si toca reentrenar. Devuelve (due, motivo).

    `completed_week`, `covered_jornada` y `age_days` son inyectables para tests
    (None simula "no disponible"); si no se pasan, se resuelven contra la API y
    el filesystem.
    """
    current = now or datetime.now()
    hour, minute = retrain_time()
    if (current.hour, current.minute) < (hour, minute):
        return False, "before_window"

    today = current.strftime("%Y-%m-%d")
    if state.get("last_success_date") == today:
        return False, "already_ran_today"

    last_error_at = str(state.get("last_error_at") or "")
    if state.get("last_error_date") == today and last_error_at:
        try:
            error_dt = datetime.fromisoformat(last_error_at)
            elapsed = (current - error_dt).total_seconds()
            if elapsed < error_retry_seconds():
                return False, "error_backoff"
        except Exception:
            pass

    paths = artifact_paths()
    missing = [name for name in ("raw_dataset", "features", "model") if not paths[name].exists()]
    if missing:
        return True, f"missing_artifacts:{','.join(missing)}"

    covered = (
        state.get("dataset_max_jornada") or dataset_max_jornada()
        if covered_jornada is _UNSET
        else covered_jornada
    )
    completed = fetch_completed_week() if completed_week is _UNSET else completed_week
    if completed is not None and covered is not None and int(completed) > int(covered):
        return True, f"new_jornada:{completed}>{covered}"

    if completed is None:
        age = dataset_age_days() if age_days is _UNSET else age_days
        if age is not None and age >= max_age_days():
            return True, f"stale_age:{age:.1f}d"

    return False, "up_to_date"


def _default_runner(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def run_retrain_pipeline(
    *,
    timeout_seconds: int | None = None,
    runner: Runner | None = None,
) -> dict:
    """
    Ejecuta collect_data → features → train en subprocesos.

    Devuelve {"ok", "steps", "error", "metrics", "dataset_max_jornada"}.
    Aborta en el primer paso fallido; nunca lanza excepción.
    """
    timeout = timeout_seconds if timeout_seconds is not None else step_timeout_seconds()
    run = runner or _default_runner
    root = repo_root()
    steps: list[dict] = []

    for cmd in PIPELINE_STEPS:
        step_name = cmd[-1]
        started = time.monotonic()
        try:
            completed = run(cmd, root, timeout)
        except subprocess.TimeoutExpired:
            error = f"Timeout en {step_name} tras {timeout}s."
            steps.append({"step": step_name, "ok": False, "error": error})
            return {"ok": False, "steps": steps, "error": error}
        except Exception as exc:
            error = f"{step_name}: {type(exc).__name__}: {exc}"
            steps.append({"step": step_name, "ok": False, "error": error})
            return {"ok": False, "steps": steps, "error": error}

        elapsed = round(time.monotonic() - started, 1)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-500:]
            error = f"{step_name} falló (rc={completed.returncode}): {detail}"
            steps.append({"step": step_name, "ok": False, "seconds": elapsed, "error": error})
            return {"ok": False, "steps": steps, "error": error}

        steps.append({"step": step_name, "ok": True, "seconds": elapsed})
        logger.info("Reentrenamiento: %s completado en %.1fs", step_name, elapsed)

    metrics: dict = {}
    try:
        meta = json.loads(artifact_paths()["meta"].read_text(encoding="utf-8"))
        metrics = (meta.get("metrics") or {}).get("walk_forward_oof") or {}
        metrics["folds"] = (meta.get("metrics") or {}).get("folds")
    except Exception:
        pass

    return {
        "ok": True,
        "steps": steps,
        "error": "",
        "metrics": metrics,
        "dataset_max_jornada": dataset_max_jornada(),
    }


def _build_notification(result: dict, reason: str) -> str:
    if result.get("ok"):
        metrics = result.get("metrics") or {}
        mae = metrics.get("mae")
        r2 = metrics.get("r2")
        folds = metrics.get("folds")
        jornada = result.get("dataset_max_jornada")
        lines = ["🧠 Modelo xP reentrenado."]
        if jornada:
            lines.append(f"Datos hasta la jornada {jornada}.")
        if mae is not None:
            lines.append(f"MAE {mae} · R² {r2} ({folds} folds walk-forward).")
        lines.append(f"Motivo: {reason}")
        return "\n".join(lines)
    return (
        "⚠️ Falló el reentrenamiento del modelo xP.\n"
        f"Motivo del disparo: {reason}\n"
        f"Error: {result.get('error', 'desconocido')}\n"
        "Se reintentará automáticamente."
    )


def maybe_run_retrain(
    *,
    notify: Callable[[str], None] | None = None,
    force: bool = False,
    timeout_seconds: int | None = None,
    runner: Runner | None = None,
    state_path: Path | None = None,
) -> dict:
    """
    Chequea si toca reentrenar y, si procede, ejecuta el pipeline y persiste estado.

    Devuelve el resultado del pipeline, o {"ok": True, "skipped": reason} si no tocaba.
    """
    state = load_state(state_path)
    now = datetime.now()

    if force:
        due, reason = True, "forced"
    else:
        due, reason = retrain_due(state, now=now)
    if not due:
        return {"ok": True, "skipped": reason}

    logger.info("Reentrenamiento del modelo xP (motivo: %s)...", reason)
    result = run_retrain_pipeline(timeout_seconds=timeout_seconds, runner=runner)

    today = now.strftime("%Y-%m-%d")
    state["last_run_at"] = now.isoformat()
    state["last_trigger_reason"] = reason
    if result.get("ok"):
        state["last_success_at"] = now.isoformat()
        state["last_success_date"] = today
        state["dataset_max_jornada"] = result.get("dataset_max_jornada")
        state["last_metrics"] = result.get("metrics") or {}
        state.pop("last_error", None)
        state.pop("last_error_at", None)
        state.pop("last_error_date", None)
    else:
        state["last_error"] = result.get("error", "")
        state["last_error_at"] = now.isoformat()
        state["last_error_date"] = today
    save_state(state, state_path)

    if notify is not None:
        try:
            notify(_build_notification(result, reason))
        except Exception as exc:
            logger.warning("No se pudo notificar el reentrenamiento: %s", exc)
    return result


def run_retrain_daemon(
    *,
    stop_event: threading.Event,
    bot_token: str = "",
    notify_chat_id: str = "",
    poll_seconds: int | None = None,
) -> None:
    """Bucle del hilo de reentrenamiento (robusto: nunca deja morir el hilo por un ciclo)."""
    from prediction.telegram_notify import send_telegram_message

    poll = poll_seconds if poll_seconds is not None else _env_int(
        "RETRAIN_POLL_SECONDS", DEFAULT_POLL_SECONDS
    )

    def _notify(text: str) -> None:
        send_telegram_message(bot_token, notify_chat_id, text)

    notify = _notify if bot_token and notify_chat_id else None
    logger.info(
        "Retrain daemon activo (ventana diaria %s, chequeo cada %ss).",
        os.getenv("RETRAIN_TIME", DEFAULT_RETRAIN_TIME),
        poll,
    )
    while not stop_event.is_set():
        try:
            maybe_run_retrain(notify=notify)
        except Exception as exc:
            logger.exception("Ciclo de reentrenamiento falló: %s", exc)
        stop_event.wait(poll)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Reentrena el modelo xP (collect_data → features → train)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reentrena ya, ignorando ventana horaria y chequeo de jornada.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Solo informa si tocaría reentrenar, sin ejecutar nada.",
    )
    parser.add_argument(
        "--step-timeout",
        type=int,
        default=None,
        help=f"Timeout por paso en segundos (default {DEFAULT_STEP_TIMEOUT_SECONDS}).",
    )
    args = parser.parse_args()

    if args.check:
        due, reason = retrain_due(load_state())
        print(json.dumps({"due": due, "reason": reason}, indent=2))
        return

    result = maybe_run_retrain(force=args.force, timeout_seconds=args.step_timeout)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
