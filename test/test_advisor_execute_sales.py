from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction.advisor_execute import _extract_offer_id, _parse_api_datetime


class AdvisorExecuteSalesTests(unittest.TestCase):
    def test_extract_offer_id_from_known_api_shapes(self) -> None:
        self.assertEqual(_extract_offer_id({"leagueOfferId": "lo1"}), "lo1")
        self.assertEqual(_extract_offer_id({"offerId": "of1"}), "of1")
        self.assertEqual(_extract_offer_id({"leagueOffer": {"id": "lo2"}}), "lo2")
        self.assertEqual(_extract_offer_id({"offer": {"id": "of2"}}), "of2")
        self.assertEqual(_extract_offer_id({"offers": [{"id": "list1"}]}), "list1")

    def test_extract_offer_id_returns_empty_for_unaccepted_shapes(self) -> None:
        self.assertEqual(_extract_offer_id(None), "")
        self.assertEqual(_extract_offer_id({}), "")
        self.assertEqual(_extract_offer_id({"numberOfOffers": 1}), "")

    def test_parse_api_datetime_accepts_z_suffix(self) -> None:
        parsed = _parse_api_datetime("2026-05-15T20:18:08Z")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.isoformat(), "2026-05-15T20:18:08+00:00")


if __name__ == "__main__":
    unittest.main()
