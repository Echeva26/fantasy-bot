#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

LEAGUE_ID="${LALIGA_LEAGUE_ID:-${1:-}}"
if [[ -z "$LEAGUE_ID" ]]; then
  echo "Error: define LALIGA_LEAGUE_ID o pásalo como primer argumento."
  exit 1
fi

exec "$PYTHON_BIN" -m prediction.autopilot --mode post --league "$LEAGUE_ID"
