from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction.advisor_execute import (
    _extract_offer_id,
    _parse_api_datetime,
    _select_offer_id,
)


class AdvisorExecuteSalesTests(unittest.TestCase):
    def test_extract_offer_id_from_known_api_shapes(self) -> None:
        self.assertEqual(_extract_offer_id({"leagueOfferId": "lo1"}), "lo1")
        self.assertEqual(_extract_offer_id({"offerId": "of1"}), "of1")
        self.assertEqual(_extract_offer_id({"leagueOffer": {"id": "lo2"}}), "lo2")
        self.assertEqual(_extract_offer_id({"offer": {"id": "of2"}}), "of2")
        self.assertEqual(_extract_offer_id({"offers": [{"id": "list1"}]}), "list1")
        self.assertEqual(_extract_offer_id({"marketOffers": [{"id": "mo1"}]}), "mo1")
        self.assertEqual(_extract_offer_id({"acceptedOffer": {"id": "ao1"}}), "ao1")

    def test_extract_offer_id_returns_empty_for_unaccepted_shapes(self) -> None:
        self.assertEqual(_extract_offer_id(None), "")
        self.assertEqual(_extract_offer_id({}), "")
        self.assertEqual(_extract_offer_id({"numberOfOffers": 1}), "")

    def test_parse_api_datetime_accepts_z_suffix(self) -> None:
        parsed = _parse_api_datetime("2026-05-15T20:18:08Z")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.isoformat(), "2026-05-15T20:18:08+00:00")

    def test_select_offer_id_prefers_league_offer(self) -> None:
        payload = {
            "offers": [
                {"id": "manager1", "type": "manager"},
                {"id": "league1", "type": "league"},
            ]
        }

        self.assertEqual(_select_offer_id(payload), "league1")

    def test_select_offer_id_accepts_list_payloads(self) -> None:
        self.assertEqual(_select_offer_id([{"id": "list1"}]), "list1")


if __name__ == "__main__":
    unittest.main()
