"""
Tests de la lógica de disparo del reentrenamiento periódico.

Ejecutar desde la raíz del repo:
    python -m unittest test.test_retrain_schedule -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prediction import retrain


IN_WINDOW = datetime(2026, 7, 30, 6, 0)
BEFORE_WINDOW = datetime(2026, 7, 30, 4, 0)


class RetrainDueTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("RETRAIN_TIME", None)

        # Los artefactos existen por defecto en estos tests.
        paths_patch = mock.patch.object(
            retrain,
            "artifact_paths",
            return_value={
                "raw_dataset": Path(__file__),
                "features": Path(__file__),
                "model": Path(__file__),
                "meta": Path(__file__),
            },
        )
        paths_patch.start()
        self.addCleanup(paths_patch.stop)

    def test_before_window_skips(self):
        due, reason = retrain.retrain_due({}, now=BEFORE_WINDOW, completed_week=25, covered_jornada=24)
        self.assertFalse(due)
        self.assertEqual(reason, "before_window")

    def test_new_jornada_triggers(self):
        due, reason = retrain.retrain_due({}, now=IN_WINDOW, completed_week=25, covered_jornada=24)
        self.assertTrue(due)
        self.assertTrue(reason.startswith("new_jornada"))

    def test_up_to_date_skips(self):
        due, reason = retrain.retrain_due({}, now=IN_WINDOW, completed_week=24, covered_jornada=24)
        self.assertFalse(due)
        self.assertEqual(reason, "up_to_date")

    def test_already_ran_today_skips(self):
        state = {"last_success_date": IN_WINDOW.strftime("%Y-%m-%d")}
        due, reason = retrain.retrain_due(state, now=IN_WINDOW, completed_week=25, covered_jornada=24)
        self.assertFalse(due)
        self.assertEqual(reason, "already_ran_today")

    def test_error_backoff_skips_until_retry(self):
        state = {
            "last_error_date": IN_WINDOW.strftime("%Y-%m-%d"),
            "last_error_at": IN_WINDOW.replace(hour=5, minute=45).isoformat(),
        }
        due, reason = retrain.retrain_due(state, now=IN_WINDOW, completed_week=25, covered_jornada=24)
        self.assertFalse(due)
        self.assertEqual(reason, "error_backoff")

    def test_missing_artifacts_trigger(self):
        with mock.patch.object(
            retrain,
            "artifact_paths",
            return_value={
                "raw_dataset": Path("/nonexistent/raw.json"),
                "features": Path(__file__),
                "model": Path(__file__),
                "meta": Path(__file__),
            },
        ):
            due, reason = retrain.retrain_due({}, now=IN_WINDOW, completed_week=None, covered_jornada=None)
        self.assertTrue(due)
        self.assertTrue(reason.startswith("missing_artifacts"))

    def test_api_down_falls_back_to_age(self):
        due, reason = retrain.retrain_due(
            {}, now=IN_WINDOW, completed_week=None, covered_jornada=24, age_days=10.0
        )
        self.assertTrue(due)
        self.assertTrue(reason.startswith("stale_age"))

        due, reason = retrain.retrain_due(
            {}, now=IN_WINDOW, completed_week=None, covered_jornada=24, age_days=2.0
        )
        self.assertFalse(due)
        self.assertEqual(reason, "up_to_date")


class RetrainPipelineTest(unittest.TestCase):
    def test_stops_on_first_failed_step(self):
        calls = []

        def fake_runner(cmd, cwd, timeout):
            calls.append(cmd[-1])
            rc = 1 if cmd[-1] == "prediction.features" else 0
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="boom")

        result = retrain.run_retrain_pipeline(runner=fake_runner)
        self.assertFalse(result["ok"])
        self.assertIn("prediction.features", result["error"])
        self.assertEqual(calls, ["prediction.collect_data", "prediction.features"])

    def test_maybe_run_persists_state_and_notifies(self):
        messages = []

        def fake_runner(cmd, cwd, timeout):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            with mock.patch.object(retrain, "dataset_max_jornada", return_value=25):
                result = retrain.maybe_run_retrain(
                    notify=messages.append,
                    force=True,
                    runner=fake_runner,
                    state_path=state_path,
                )
            self.assertTrue(result["ok"])
            state = retrain.load_state(state_path)
            self.assertEqual(state.get("dataset_max_jornada"), 25)
            self.assertEqual(state.get("last_trigger_reason"), "forced")
            self.assertTrue(state.get("last_success_at"))
        self.assertEqual(len(messages), 1)
        self.assertIn("reentrenado", messages[0])


if __name__ == "__main__":
    unittest.main()
