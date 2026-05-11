"""
Stale-aware freshness checks for local scraper snapshots.

The agent reads news from ``scrapes/*.json``. This helper keeps that directory
fresh enough for reports without making every run depend on a successful scrape.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


DEFAULT_MAX_AGE_MINUTES = 120
DEFAULT_TIMEOUT_SECONDS = 90


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def scrapes_dir() -> Path:
    return repo_root() / "scrapes"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except Exception:
        return default


def max_age_minutes() -> int:
    return _env_int("SCRAPES_MAX_AGE_MINUTES", DEFAULT_MAX_AGE_MINUTES)


def scraper_timeout_seconds() -> int:
    return _env_int("SCRAPER_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)


def latest_scrape_file(base_dir: Path | None = None) -> Path | None:
    base = base_dir or scrapes_dir()
    if not base.exists() or not base.is_dir():
        return None
    files = [p for p in base.iterdir() if p.is_file() and p.suffix == ".json"]
    if not files:
        return None
    return max(files, key=lambda p: (p.stat().st_mtime, p.name))


def scrape_age_minutes(path: Path, *, now: datetime | None = None) -> float:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return max(0.0, (current - modified).total_seconds() / 60.0)


@dataclass(frozen=True)
class ScrapeFreshnessResult:
    ok: bool
    fresh: bool
    ran_scraper: bool
    latest_file: str
    age_minutes: float | None
    error: str
    reason: str

    @property
    def stale(self) -> bool:
        return not self.fresh

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "fresh": self.fresh,
            "stale": self.stale,
            "ran_scraper": self.ran_scraper,
            "latest_file": self.latest_file,
            "age_minutes": (
                round(float(self.age_minutes), 1)
                if self.age_minutes is not None
                else None
            ),
            "error": self.error,
            "reason": self.reason,
        }


Runner = Callable[[list[str], Path, int], subprocess.CompletedProcess[str]]


def _default_runner(
    cmd: list[str],
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
        check=False,
    )


def _fresh_result(
    *,
    latest: Path | None,
    age: float | None,
    fresh: bool,
    ran_scraper: bool,
    error: str = "",
    reason: str = "",
) -> ScrapeFreshnessResult:
    return ScrapeFreshnessResult(
        ok=bool(fresh and latest),
        fresh=bool(fresh and latest),
        ran_scraper=ran_scraper,
        latest_file=str(latest) if latest else "",
        age_minutes=age,
        error=error,
        reason=reason,
    )


def ensure_fresh_scrapes(
    *,
    max_age: int | None = None,
    timeout_seconds: int | None = None,
    base_dir: Path | None = None,
    runner: Runner | None = None,
    now: datetime | None = None,
) -> ScrapeFreshnessResult:
    """
    Ensure ``scrapes/`` has a recent JSON snapshot.

    If the directory is missing, empty, or stale, this runs the scraper once and
    returns a structured status. A failed scraper is reported but does not raise:
    the agent can continue with a degraded news context.
    """
    root = repo_root()
    base = base_dir or scrapes_dir()
    limit = max_age if max_age is not None else max_age_minutes()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else scraper_timeout_seconds()
    )
    run = runner or _default_runner

    latest = latest_scrape_file(base)
    age = scrape_age_minutes(latest, now=now) if latest else None
    if latest and age is not None and age <= limit:
        return _fresh_result(
            latest=latest,
            age=age,
            fresh=True,
            ran_scraper=False,
            reason="Último scrape fresco.",
        )

    reason = (
        "No existe el directorio scrapes/."
        if not base.exists()
        else "No hay snapshots JSON en scrapes/."
        if latest is None
        else f"Último scrape obsoleto ({age:.1f} min > {limit} min)."
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = base / f"{timestamp}.json"
    cmd = [
        sys.executable,
        "-m",
        "scrapers.scrape_all",
        "--output",
        str(output),
    ]

    try:
        completed = run(cmd, root, timeout)
    except subprocess.TimeoutExpired:
        latest_after = latest_scrape_file(base)
        age_after = (
            scrape_age_minutes(latest_after, now=now)
            if latest_after
            else None
        )
        return _fresh_result(
            latest=latest_after,
            age=age_after,
            fresh=bool(latest_after and age_after is not None and age_after <= limit),
            ran_scraper=True,
            error=f"Timeout ejecutando scraper tras {timeout}s.",
            reason=reason,
        )
    except Exception as exc:
        latest_after = latest_scrape_file(base)
        age_after = (
            scrape_age_minutes(latest_after, now=now)
            if latest_after
            else None
        )
        return _fresh_result(
            latest=latest_after,
            age=age_after,
            fresh=bool(latest_after and age_after is not None and age_after <= limit),
            ran_scraper=True,
            error=f"{type(exc).__name__}: {exc}",
            reason=reason,
        )

    latest_after = latest_scrape_file(base)
    age_after = (
        scrape_age_minutes(latest_after, now=now)
        if latest_after
        else None
    )
    fresh_after = bool(latest_after and age_after is not None and age_after <= limit)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"returncode={completed.returncode}"
        return _fresh_result(
            latest=latest_after,
            age=age_after,
            fresh=fresh_after,
            ran_scraper=True,
            error=f"Scraper falló: {detail[:500]}",
            reason=reason,
        )

    return _fresh_result(
        latest=latest_after,
        age=age_after,
        fresh=fresh_after,
        ran_scraper=True,
        error="" if fresh_after else "El scraper terminó pero no dejó un JSON fresco.",
        reason=reason,
    )
