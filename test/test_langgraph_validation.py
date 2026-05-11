from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction.langgraph_agent import (
    _financially_validated_actions,
    _validated_manager_actions_with_context,
)


class LangGraphActionValidationTests(unittest.TestCase):
    def test_rejects_manager_actions_not_generated_by_motor(self) -> None:
        manager = {
            "acciones_ejecutables": [
                {
                    "tool": "sell_player_phase1_tool",
                    "tool_input": {"player_team_id": "p1", "sale_price": 1000},
                },
                {
                    "tool": "place_bid_tool",
                    "tool_input": {"market_item_id": "m1", "amount": 1000},
                },
            ]
        }

        selected, rejected = _validated_manager_actions_with_context(manager, [], {})

        self.assertEqual(selected, [])
        self.assertEqual(len(rejected), 2)
        self.assertIn("no generada", rejected[0]["reason"])

    def test_allows_actions_present_in_simulation(self) -> None:
        action = {
            "tool": "sell_player_phase1_tool",
            "tool_input": {"player_team_id": "p1", "sale_price": 1000},
            "label": "Vender p1",
        }
        manager = {"acciones_ejecutables": [action]}

        selected, rejected = _validated_manager_actions_with_context(manager, [action], {})

        self.assertEqual(selected, [action])
        self.assertEqual(rejected, [])

    def test_allows_deterministic_clause_protection_only_for_exposed_key_players(self) -> None:
        manager = {
            "acciones_ejecutables": [
                {
                    "tool": "increase_clause_tool",
                    "tool_input": {"player_team_id": "key", "value_to_increase": 999999},
                },
                {
                    "tool": "increase_clause_tool",
                    "tool_input": {"player_team_id": "bench", "value_to_increase": 50000},
                },
            ]
        }
        context = {
            "my_squad": {
                "players": [
                    {
                        "player_team_id": "key",
                        "nombre": "Clave",
                        "estado": "ok",
                        "en_venta": False,
                        "ratio_valor_vs_clausula": 0.95,
                        "xP": 6.0,
                    },
                    {
                        "player_team_id": "bench",
                        "nombre": "Banco",
                        "estado": "ok",
                        "en_venta": False,
                        "ratio_valor_vs_clausula": 0.95,
                        "xP": 3.0,
                    },
                ]
            }
        }

        selected, rejected = _validated_manager_actions_with_context(manager, [], context)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["tool"], "increase_clause_tool")
        self.assertEqual(selected[0]["tool_input"]["player_team_id"], "key")
        self.assertEqual(selected[0]["tool_input"]["value_to_increase"], 50000)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["tool_input"]["player_team_id"], "bench")

    def test_financial_validation_applies_sales_before_bids(self) -> None:
        actions = [
            {
                "tool": "place_bid_tool",
                "tool_input": {"market_item_id": "m1", "amount": 1000},
            },
            {
                "tool": "sell_player_phase1_tool",
                "tool_input": {"player_team_id": "p1", "sale_price": 1000},
            },
        ]
        context = {"snapshot_summary": {"saldo_disponible": 100}}

        accepted, rejected = _financially_validated_actions(actions, context)

        self.assertEqual(rejected, [])
        self.assertEqual([a["tool"] for a in accepted], ["sell_player_phase1_tool", "place_bid_tool"])

    def test_financial_validation_rejects_bid_without_sale_cover(self) -> None:
        actions = [
            {
                "tool": "place_bid_tool",
                "tool_input": {"market_item_id": "m1", "amount": 1000},
            }
        ]
        context = {"snapshot_summary": {"saldo_disponible": 100}}

        accepted, rejected = _financially_validated_actions(actions, context)

        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("Saldo insuficiente", rejected[0]["reason"])


if __name__ == "__main__":
    unittest.main()
