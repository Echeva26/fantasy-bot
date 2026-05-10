import os
import unittest
from unittest import mock

from prediction.langsmith_config import (
    build_langsmith_config,
    is_langsmith_enabled,
    sanitize_langsmith_metadata,
)


class LangSmithConfigTests(unittest.TestCase):
    def test_is_langsmith_enabled_false_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_langsmith_enabled())

    def test_is_langsmith_enabled_false_without_api_key(self) -> None:
        env = {"LANGSMITH_TRACING": "true"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(is_langsmith_enabled())

    def test_is_langsmith_enabled_false_without_langsmith_module(self) -> None:
        env = {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "test-key",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("prediction.langsmith_config._load_langsmith_runtime", return_value=(None, None)):
                self.assertFalse(is_langsmith_enabled())

    def test_build_langsmith_config_adds_tags_and_metadata(self) -> None:
        env = {
            "ENVIRONMENT": "test",
            "LANGCHAIN_LLM_MODEL": "gpt-5-mini",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = build_langsmith_config(
                run_name="fantasy-bot.manual-report",
                phase="pre",
                command="informe",
                league_id="league-123",
                market_cycle_id="2026-05-10T09:00:00+02:00",
                dry_run=True,
                extra_metadata={"engine": "langgraph", "authorization": "secret"},
            )

        self.assertEqual(config["run_name"], "fantasy-bot.manual-report")
        self.assertIn("fantasy-bot", config["tags"])
        self.assertIn("langchain-agent", config["tags"])
        self.assertIn("pre", config["tags"])
        self.assertIn("informe", config["tags"])
        self.assertIn("dry-run", config["tags"])
        self.assertEqual(config["metadata"]["app"], "fantasy-bot")
        self.assertEqual(config["metadata"]["component"], "langchain-agent")
        self.assertEqual(config["metadata"]["league_id"], "league-123")
        self.assertEqual(config["metadata"]["market_cycle_id"], "2026-05-10T09:00:00+02:00")
        self.assertEqual(config["metadata"]["environment"], "test")
        self.assertEqual(config["metadata"]["model"], "gpt-5-mini")
        self.assertEqual(config["metadata"]["authorization"], "[REDACTED]")

    def test_sanitize_langsmith_metadata_redacts_recursively(self) -> None:
        payload = {
            "token": "abc",
            "nested": {
                "authorization": "Bearer secret",
                "items": [
                    {"password": "pw"},
                    {"cookie": "cookie-value"},
                    {"ok": True},
                ],
            },
        }
        sanitized = sanitize_langsmith_metadata(payload)
        self.assertEqual(sanitized["token"], "[REDACTED]")
        self.assertEqual(sanitized["nested"]["authorization"], "[REDACTED]")
        self.assertEqual(sanitized["nested"]["items"][0]["password"], "[REDACTED]")
        self.assertEqual(sanitized["nested"]["items"][1]["cookie"], "[REDACTED]")
        self.assertTrue(sanitized["nested"]["items"][2]["ok"])


if __name__ == "__main__":
    unittest.main()
