"""
Resolución de equipo para jugadores de LaLiga Fantasy.

La API pública v5 devuelve solo `teamId` (string), mientras que v3 incluía
`team: { id, name, ... }`. El pipeline de predicción y el dataset necesitan el
nombre canónico que coincide con las claves de `team_mapping` en
`raw_dataset.json`.
"""

from __future__ import annotations

# teamId (API Fantasy) -> clave exacta en team_mapping del dataset histórico
# (debe coincidir con prediction/data/raw_dataset.json → team_mapping)
FANTASY_TEAM_ID_TO_MAPPING_KEY: dict[str, str] = {
    "2": "Atlético de Madrid",
    "3": "Athletic Club",
    "4": "FC Barcelona",
    "5": "Real Betis",
    "6": "Celta",
    "7": "Elche CF",
    "8": "RCD Espanyol de Barcelona",
    "9": "Getafe CF",
    "11": "Levante UD",
    "13": "C.A. Osasuna",
    "14": "Rayo Vallecano",
    "15": "Real Madrid",
    "16": "Real Sociedad",
    "17": "Sevilla FC",
    "18": "Valencia CF",
    "20": "Villarreal CF",
    "21": "Deportivo Alavés",
    # En el dataset la clave usa espacio no separador (U+00A0) entre "Girona" y "FC"
    "28": "Girona\u00a0FC",
    "33": "RCD Mallorca",
    "157": "Real Oviedo",
}


def fantasy_team_name_for_mapping(player: dict) -> str:
    """
    Devuelve el nombre de equipo usable con team_mapping (raw_dataset).

    Compatible con:
    - v3 / liga autenticada: `team.name`
    - v5 público: solo `teamId`
    """
    if not isinstance(player, dict):
        return ""

    team = player.get("team")
    if isinstance(team, dict):
        name = str(team.get("name", "") or "").strip()
        if name:
            return name

    tid = str(player.get("teamId", "") or "").strip()
    if tid:
        return FANTASY_TEAM_ID_TO_MAPPING_KEY.get(tid, "")
    return ""
