from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction import lineup_autoset


class LineupAutosetTests(unittest.TestCase):
    def test_formation_parts_accepts_fallback_suffix(self) -> None:
        self.assertEqual(lineup_autoset._formation_parts("4-3-3 (fallback)"), (4, 3, 3))
        self.assertEqual(lineup_autoset._formation_parts(" 3 - 5 - 2 "), (3, 5, 2))

    def test_formation_parts_rejects_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            lineup_autoset._formation_parts("4-4-4")

    def test_autoset_skips_when_predictions_are_unavailable(self) -> None:
        with mock.patch.object(lineup_autoset, "load_snapshot", return_value={"mi_equipo": {}}):
            with mock.patch.object(
                lineup_autoset,
                "get_predictions",
                side_effect=RuntimeError("SofaScore 403"),
            ):
                result = lineup_autoset.autoset_best_lineup(
                    league_id="league",
                    force=True,
                    dry_run=True,
                )

        self.assertFalse(result["applied"])
        self.assertTrue(result["skipped"])
        self.assertIn("Predicciones no disponibles", result["reason"])


if __name__ == "__main__":
    unittest.main()
