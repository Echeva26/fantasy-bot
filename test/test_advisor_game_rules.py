from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction.advisor import simulate_transfer_plan


def _player(
    player_id: int,
    position: str,
    *,
    unavailable: bool = False,
    xp: float = 3.0,
    market_value: int = 1_000_000,
    clause: int = 1_000_000,
) -> dict:
    return {
        "player_id": player_id,
        "player_team_id": f"pt{player_id}",
        "nombre": f"Jugador {player_id}",
        "posicion": position,
        "estado": "sancionado" if unavailable else "ok",
        "no_disponible": unavailable,
        "valor_mercado": market_value,
        "clausula": clause,
        "xP": xp,
        "media_min_5j": 90,
        "titular_pct_5j": 1.0,
    }


class AdvisorGameRulesTests(unittest.TestCase):
    def test_simulation_prioritizes_affordable_goalkeeper_coverage(self) -> None:
        players = [
            _player(1, "POR", unavailable=True, xp=4.0),
            _player(2, "DEF"),
            _player(3, "DEF"),
            _player(4, "DEF"),
            _player(5, "MED"),
            _player(6, "MED"),
            _player(7, "MED"),
            _player(8, "DEL"),
            _player(9, "DEL"),
            _player(10, "DEF"),
        ]
        available = {
            "mercado": [
                {
                    "player_id": 99,
                    "market_item_id": "m99",
                    "nombre": "Portero Mercado",
                    "posicion": "POR",
                    "precio_venta": 1_000_000,
                    "valor_mercado": 1_000_000,
                    "xP": 3.0,
                    "pujas": 0,
                }
            ],
            "clausulazos": [],
        }

        plan = simulate_transfer_plan(
            {"saldo": 2_000_000, "jugadores": players, "once_ideal": players},
            available,
            {},
            allow_clausulazos=False,
        )

        self.assertEqual(plan["priority_needs"][0]["position"], "POR")
        self.assertEqual(plan["movimientos"][0]["priority"], "lineup_coverage")
        self.assertEqual(plan["movimientos"][0]["compra"]["market_item_id"], "m99")

    def test_simulation_sells_unavailable_goalkeeper_when_replacement_not_affordable(self) -> None:
        players = [
            _player(1, "POR", unavailable=True, xp=4.0),
            _player(2, "DEF"),
            _player(3, "DEF"),
            _player(4, "DEF"),
            _player(5, "MED"),
            _player(6, "MED"),
            _player(7, "MED"),
            _player(8, "DEL"),
            _player(9, "DEL"),
            _player(10, "DEF"),
        ]
        available = {
            "mercado": [
                {
                    "player_id": 99,
                    "market_item_id": "m99",
                    "nombre": "Portero Caro",
                    "posicion": "POR",
                    "precio_venta": 1_000_000,
                    "valor_mercado": 1_000_000,
                    "xP": 3.0,
                    "pujas": 0,
                }
            ],
            "clausulazos": [],
        }

        plan = simulate_transfer_plan(
            {"saldo": 72_505, "jugadores": players, "once_ideal": players},
            available,
            {},
            allow_clausulazos=False,
        )

        self.assertEqual(plan["priority_needs"][0]["position"], "POR")
        self.assertEqual(plan["movimientos"][0]["venta"]["player_id"], 1)
        self.assertIn("sancionado", plan["movimientos"][0]["venta"]["motivo"])
        self.assertEqual(plan["non_executable_recommendations"][0]["position"], "POR")

    def test_simulation_sells_bad_value_xp_player_when_clause_not_protected(self) -> None:
        players = [
            _player(1, "POR", xp=4.0),
            _player(2, "DEF", xp=4.0),
            _player(3, "DEF", xp=4.0),
            _player(4, "DEF", xp=4.0),
            _player(5, "DEF", xp=4.0),
            _player(6, "MED", xp=4.0),
            _player(7, "MED", xp=4.0),
            _player(8, "MED", xp=4.0),
            _player(9, "DEL", xp=4.0),
            _player(10, "DEL", xp=4.0),
            _player(11, "DEL", xp=4.0),
            _player(12, "MED", xp=2.4, market_value=8_000_000, clause=8_000_000),
        ]

        plan = simulate_transfer_plan(
            {"saldo": 500_000, "jugadores": players, "once_ideal": players[:11]},
            {"mercado": [], "clausulazos": []},
            {},
            allow_clausulazos=False,
        )

        sale = next(m["venta"] for m in plan["movimientos"] if m.get("venta"))
        self.assertEqual(sale["player_id"], 12)
        self.assertIn("curva logarítmica", sale["motivo"])
        self.assertGreater(sale["ratio_valor_xp"], 2_500_000)
        self.assertGreater(sale["brecha_xp_valor_log"], 1.25)
        self.assertGreater(sale["xp_esperado_por_valor_log"], sale["xP"])

    def test_simulation_does_not_sell_bad_value_xp_player_with_high_clause(self) -> None:
        players = [
            _player(1, "POR", xp=4.0),
            _player(2, "DEF", xp=4.0),
            _player(3, "DEF", xp=4.0),
            _player(4, "DEF", xp=4.0),
            _player(5, "DEF", xp=4.0),
            _player(6, "MED", xp=4.0),
            _player(7, "MED", xp=4.0),
            _player(8, "MED", xp=4.0),
            _player(9, "DEL", xp=4.0),
            _player(10, "DEL", xp=4.0),
            _player(11, "DEL", xp=4.0),
            _player(12, "MED", xp=2.4, market_value=8_000_000, clause=11_000_000),
        ]

        plan = simulate_transfer_plan(
            {"saldo": 500_000, "jugadores": players, "once_ideal": players[:11]},
            {"mercado": [], "clausulazos": []},
            {},
            allow_clausulazos=False,
        )

        self.assertEqual(plan["movimientos"], [])

    def test_simulation_does_not_sell_premium_high_xp_player_on_log_curve(self) -> None:
        players = [
            _player(1, "POR", xp=4.0),
            _player(2, "DEF", xp=4.0),
            _player(3, "DEF", xp=4.0),
            _player(4, "DEF", xp=4.0),
            _player(5, "DEF", xp=4.0),
            _player(6, "MED", xp=4.0),
            _player(7, "MED", xp=4.0),
            _player(8, "MED", xp=4.0),
            _player(9, "DEL", xp=4.0),
            _player(10, "DEL", xp=4.0),
            _player(11, "DEL", xp=4.0),
            _player(12, "MED", xp=9.0, market_value=45_000_000, clause=45_000_000),
        ]

        plan = simulate_transfer_plan(
            {"saldo": 500_000, "jugadores": players, "once_ideal": players[:11]},
            {"mercado": [], "clausulazos": []},
            {},
            allow_clausulazos=False,
        )

        self.assertEqual(plan["movimientos"], [])


if __name__ == "__main__":
    unittest.main()
