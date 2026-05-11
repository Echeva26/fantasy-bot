from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction import langchain_tools
from prediction.langchain_tools import build_langchain_tools
from prediction.scrape_freshness import ensure_fresh_scrapes


class ScrapeFreshnessTests(unittest.TestCase):
    def test_missing_dir_runs_scraper_and_creates_fresh_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "scrapes"

            def runner(cmd: list[str], _cwd: Path, _timeout: int):
                output = Path(cmd[-1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"scraped_at": "now"}), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            result = ensure_fresh_scrapes(base_dir=base, runner=runner)

        self.assertTrue(result.ok)
        self.assertTrue(result.fresh)
        self.assertTrue(result.ran_scraper)
        self.assertTrue(result.latest_file.endswith(".json"))

    def test_fresh_file_does_not_run_scraper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            latest = base / "fresh.json"
            latest.write_text("{}", encoding="utf-8")
            calls = []

            def runner(cmd: list[str], _cwd: Path, _timeout: int):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            result = ensure_fresh_scrapes(base_dir=base, runner=runner, max_age=120)

        self.assertTrue(result.ok)
        self.assertFalse(result.ran_scraper)
        self.assertEqual(calls, [])

    def test_stale_file_runs_scraper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old = base / "old.json"
            old.write_text("{}", encoding="utf-8")
            old_time = time.time() - 4 * 3600
            os.utime(old, (old_time, old_time))

            def runner(cmd: list[str], _cwd: Path, _timeout: int):
                output = Path(cmd[-1])
                output.write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")

            result = ensure_fresh_scrapes(base_dir=base, runner=runner, max_age=30)

        self.assertTrue(result.ok)
        self.assertTrue(result.ran_scraper)
        self.assertTrue(result.fresh)

    def test_scraper_failure_reports_degraded_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            def runner(cmd: list[str], _cwd: Path, _timeout: int):
                return subprocess.CompletedProcess(cmd, 2, "", "boom")

            result = ensure_fresh_scrapes(base_dir=base, runner=runner)

        self.assertFalse(result.ok)
        self.assertFalse(result.fresh)
        self.assertTrue(result.ran_scraper)
        self.assertIn("boom", result.error)


class NewsReaderStatusTests(unittest.TestCase):
    def _news_tool(self, base: Path):
        runtime = SimpleNamespace(
            phase="pre",
            dry_run=True,
            league_id="league",
            get_snapshot=lambda force_refresh=False: {"mi_equipo": {"plantilla": []}, "mercado": []},
        )
        tools = {tool.name: tool for tool in build_langchain_tools(runtime)}
        return tools["news_reader_tool"]

    def test_news_reader_reports_missing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "missing"
            with mock.patch.object(langchain_tools, "scrapes_dir", return_value=base):
                payload = json.loads(self._news_tool(base).invoke({"limit": 5}))

        self.assertFalse(payload["ok"])
        self.assertTrue(payload["stale"])
        self.assertIsNone(payload["age_minutes"])

    def test_news_reader_reports_invalid_json_with_age(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "bad.json").write_text("{bad", encoding="utf-8")
            with mock.patch.object(langchain_tools, "scrapes_dir", return_value=base):
                payload = json.loads(self._news_tool(base).invoke({"limit": 5}))

        self.assertFalse(payload["ok"])
        self.assertTrue(payload["stale"])
        self.assertIn("No se pudo leer", payload["reason"])
        self.assertIsNotNone(payload["age_minutes"])

    def test_news_reader_reports_stale_but_readable_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            latest = base / "old.json"
            latest.write_text(json.dumps({"scraped_at": "old"}), encoding="utf-8")
            old_time = time.time() - 4 * 3600
            os.utime(latest, (old_time, old_time))
            with mock.patch.object(langchain_tools, "scrapes_dir", return_value=base):
                with mock.patch.object(langchain_tools, "max_age_minutes", return_value=30):
                    payload = json.loads(self._news_tool(base).invoke({"limit": 5}))

        self.assertTrue(payload["ok"])
        self.assertFalse(payload["fresh"])
        self.assertTrue(payload["stale"])
        self.assertIn("obsoleto", payload["reason"])


if __name__ == "__main__":
    unittest.main()
