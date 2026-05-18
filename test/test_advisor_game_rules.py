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
) -> dict:
    return {
        "player_id": player_id,
        "player_team_id": f"pt{player_id}",
        "nombre": f"Jugador {player_id}",
        "posicion": position,
        "estado": "sancionado" if unavailable else "ok",
        "no_disponible": unavailable,
        "valor_mercado": 1_000_000,
        "clausula": 1_000_000,
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

    def test_simulation_recommends_manual_plan_when_goalkeeper_not_affordable(self) -> None:
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
        self.assertEqual(plan["movimientos"], [])
        self.assertEqual(plan["non_executable_recommendations"][0]["position"], "POR")


if __name__ == "__main__":
    unittest.main()
