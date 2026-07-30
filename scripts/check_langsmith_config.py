#!/usr/bin/env python3
"""
Chequeo local de la configuración LangSmith sin tocar LaLiga Fantasy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction.langsmith_config import (  # noqa: E402
    build_langsmith_config,
    get_langsmith_project_name,
    is_langsmith_enabled,
    sanitize_langsmith_metadata,
)


def main() -> None:
    print("LangSmith enabled:", is_langsmith_enabled())
    print("LangSmith project:", get_langsmith_project_name())
    print("LangSmith endpoint:", os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))

    sample = build_langsmith_config(
        run_name="fantasy-bot.check-langsmith",
        phase="pre",
        command="check",
        league_id="league-demo",
        market_cycle_id="market-demo",
        dry_run=True,
        extra_metadata={
            "engine": "langgraph",
            "token": "should-not-appear",
            "nested": {"authorization": "Bearer secret"},
        },
    )

    print("\nSample config:")
    print(json.dumps(sample, indent=2, ensure_ascii=False))

    sanitized = sanitize_langsmith_metadata(
        {
            "api_key": "secret",
            "payload": [{"password": "pw"}, {"ok": True}],
        }
    )
    print("\nSanitized metadata:")
    print(json.dumps(sanitized, indent=2, ensure_ascii=False))

    serialized = json.dumps(sample, ensure_ascii=False)
    if "should-not-appear" in serialized or "Bearer secret" in serialized:
        raise SystemExit("Fallo: se detectaron secretos sin sanitizar en la config de ejemplo.")

    print("\nChequeo OK: no se detectaron secretos en la config de ejemplo.")


if __name__ == "__main__":
    main()
