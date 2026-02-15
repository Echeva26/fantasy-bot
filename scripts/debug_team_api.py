#!/usr/bin/env python3
"""
Script de depuración: inspecciona la respuesta v3/v4 de teams para ver
qué IDs devuelve la API (player_id vs player_team_id).

Sirve para diagnosticar el 500 en POST /market/sell cuando enviamos
playerTeamId incorrecto (ej. playerMaster.id en lugar de PlayerTeam.id).

Uso:
    python scripts/debug_team_api.py
    python scripts/debug_team_api.py --league 016615640
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.getLogger().setLevel(logging.WARNING)

from laliga_fantasy_client import LaLigaFantasyClient, load_token

BASE_URL = "https://api-fantasy.llt-services.com"


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Inspecciona respuesta API teams v3/v4")
    p.add_argument("--league", default="", help="League ID (o usa LALIGA_LEAGUE_ID)")
    args = p.parse_args()

    token = load_token()
    if not token:
        print("ERROR: No hay token. Ejecuta python laliga_fantasy_client.py --token ...")
        sys.exit(1)

    league_id = args.league or os.getenv("LALIGA_LEAGUE_ID", "")
    client = LaLigaFantasyClient(token=token, league_id=league_id)

    if not client.league_id:
        leagues = client.get_leagues()
        if not leagues:
            print("No se encontraron ligas.")
            sys.exit(1)
        client.league_id = str(leagues[0]["id"])
        print(f"Liga: {leagues[0].get('name')} ({client.league_id})\n")

    # Obtener my_team_id
    user_info = client.get_user_info()
    my_manager_id = str(user_info.get("id", ""))
    ranking = client.get_ranking_raw()
    my_team_id = ""
    for entry in ranking:
        team = entry.get("team", {})
        mgr = team.get("manager", {})
        if str(mgr.get("id", "")) == my_manager_id:
            my_team_id = str(team.get("id", ""))
            break

    if not my_team_id:
        print("No se encontró tu equipo en el ranking.")
        sys.exit(1)

    print(f"my_team_id: {my_team_id}")
    print()

    # v3
    print("=== V3: GET /api/v3/leagues/{id}/teams/{teamId} ===")
    v3_url = f"{BASE_URL}/api/v3/leagues/{client.league_id}/teams/{my_team_id}"
    print(f"URL: {v3_url}\n")
    v3_raw = client.get_team_raw(my_team_id)
    players_v3 = v3_raw.get("players", [])
    print(f"Total jugadores: {len(players_v3)}\n")

    # Primeros 3 jugadores completos (para ver estructura)
    for i, p in enumerate(players_v3[:3]):
        pm = p.get("playerMaster", {})
        nickname = pm.get("nickname", pm.get("name", "?"))
        print(f"--- Jugador {i+1}: {nickname} (playerMaster.id={pm.get('id')}) ---")
        # Claves relevantes: id, playerTeamId, playerTeam
        print(f"  p.get('id'):              {repr(p.get('id'))}")
        print(f"  p.get('playerTeamId'):    {repr(p.get('playerTeamId'))}")
        pt = p.get("playerTeam") or {}
        print(f"  p.get('playerTeam'):      {dict(pt) if pt else '{}'}")
        if pt:
            print(f"    playerTeam.id:          {repr(pt.get('id'))}")
            print(f"    playerTeam.playerTeamId:{repr(pt.get('playerTeamId'))}")
        print()

    # league/me
    print("=== league/me: GET /api/v3/leagues/{id}/me ===")
    me_url = f"{BASE_URL}/api/v3/leagues/{client.league_id}/me"
    print(f"URL: {me_url}\n")
    try:
        me_raw = client.get_league_me_raw()
        if me_raw:
            players_me = me_raw.get("players", me_raw.get("squad", []))
            print(f"Total jugadores: {len(players_me)}\n")
            for i, p in enumerate(players_me[:3]):
                pm = p.get("playerMaster", {})
                nickname = pm.get("nickname", pm.get("name", "?"))
                print(f"--- Jugador {i+1}: {nickname} (playerMaster.id={pm.get('id')}) ---")
                print(f"  p.get('id'):              {repr(p.get('id'))}")
                print(f"  p.get('playerTeamId'):    {repr(p.get('playerTeamId'))}")
                pt = p.get("playerTeam") or {}
                if pt:
                    print(f"  playerTeam.id:           {repr(pt.get('id'))}")
                print()
        else:
            print("(league/me devolvió None)\n")
    except Exception as e:
        print(f"ERROR league/me: {e}\n")

    # v4
    print("=== V4: GET /api/v4/leagues/{id}/teams/{teamId} ===")
    v4_url = f"{BASE_URL}/api/v4/leagues/{client.league_id}/teams/{my_team_id}"
    print(f"URL: {v4_url}\n")
    try:
        v4_raw = client.get_team_raw_v4(my_team_id)
        if v4_raw:
            players_v4 = v4_raw.get("players", v4_raw.get("squad", []))
            print(f"Total jugadores: {len(players_v4)}\n")
            for i, p in enumerate(players_v4[:3]):
                pm = p.get("playerMaster", {})
                nickname = pm.get("nickname", pm.get("name", "?"))
                print(f"--- Jugador {i+1}: {nickname} (playerMaster.id={pm.get('id')}) ---")
                print(f"  p.get('id'):              {repr(p.get('id'))}")
                print(f"  p.get('playerTeamId'):    {repr(p.get('playerTeamId'))}")
                pt = p.get("playerTeam") or {}
                print(f"  p.get('playerTeam'):      {dict(pt) if pt else '{}'}")
                if pt:
                    print(f"    playerTeam.id:          {repr(pt.get('id'))}")
                    print(f"    playerTeam.playerTeamId:{repr(pt.get('playerTeamId'))}")
                print()
        else:
            print("(v4 devolvió None)\n")
    except Exception as e:
        print(f"ERROR v4: {e}\n")

    # Buscar Ugrinic específicamente
    target_nick = "Ugrinic"
    for src, players, label in [
        (v3_raw, players_v3, "v3"),
    ]:
        for p in players:
            pm = p.get("playerMaster", {})
            if target_nick.lower() in (pm.get("nickname") or "").lower():
                print(f"=== {target_nick} en {label} ===")
                print(json.dumps(p, indent=2, default=str, ensure_ascii=False)[:2000])
                print()


if __name__ == "__main__":
    main()
