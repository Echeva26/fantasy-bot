#!/usr/bin/env python3
"""
Test de todos los endpoints de la API de LaLiga Fantasy.
Ejecutar: python -m test.test_all_apis [--league LEAGUE_ID]
"""
import sys
import os
import time
import argparse
import logging

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from laliga_fantasy_client import (
    LaLigaFantasyClient,
    LaLigaFantasyPublic,
    load_token,
)

logging.basicConfig(level=logging.WARNING)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0
results: list[tuple[str, str, str]] = []  # (nombre, estado, detalle)


def test(name: str):
    """Decorador-wrapper para registrar resultado de cada test."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global passed, failed, skipped
            print(f"\n  {BOLD}TEST:{RESET} {name} ... ", end="", flush=True)
            t0 = time.time()
            try:
                detail = fn(*args, **kwargs)
                elapsed = time.time() - t0
                print(f"{GREEN}OK{RESET} ({elapsed:.1f}s)")
                if detail:
                    print(f"         {detail}")
                passed += 1
                results.append((name, "OK", detail or ""))
            except Exception as e:
                elapsed = time.time() - t0
                print(f"{RED}FAIL{RESET} ({elapsed:.1f}s)")
                print(f"         {RED}{e}{RESET}")
                failed += 1
                results.append((name, "FAIL", str(e)))
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────
# TESTS PÚBLICOS (sin token)
# ─────────────────────────────────────────────────────────────────
def run_public_tests():
    print()
    print(f"  {BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}  ENDPOINTS PÚBLICOS (sin token){RESET}")
    print(f"  {BOLD}{'='*60}{RESET}")

    api = LaLigaFantasyPublic()

    player_ids: list[int] = []

    @test("Público: get_current_week")
    def t1():
        data = api.get_current_week()
        assert isinstance(data, dict), f"Esperaba dict, recibí {type(data)}"
        assert "weekNumber" in data or "nextWeek" in data, f"Campos inesperados: {list(data.keys())[:5]}"
        return f"jornada={data.get('weekNumber', '?')}, live={data.get('isLive', '?')}"
    t1()

    @test("Público: get_calendar")
    def t2():
        data = api.get_calendar()
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        return f"{len(data)} partidos en calendario"
    t2()

    @test("Legacy jugadores: get_players_raw con fallback autenticado")
    def t3():
        global skipped
        try:
            data = api.get_players_raw()
            assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
            assert len(data) > 0, "Lista vacía"
            for p in data[:5]:
                pid = p.get("id")
                if pid:
                    player_ids.append(int(pid))
            return f"{len(data)} jugadores"
        except Exception as e:
            if "ya no es público" in str(e) or "token válido" in str(e):
                skipped += 1
                return f"{YELLOW}REQUIERE TOKEN (endpoint público deprecado){RESET}"
            raise
    t3()

    @test("Legacy jugadores: get_player_detail con fallback autenticado")
    def t4():
        global skipped
        if not player_ids:
            skipped += 1
            return f"{YELLOW}SKIP: sin player_ids; requiere token para fallback{RESET}"
        pid = player_ids[0]
        try:
            data = api.get_player_detail(pid)
            assert isinstance(data, dict), f"Esperaba dict, recibí {type(data)}"
            return f"Jugador ID {pid}: {data.get('nickname', '?')}"
        except Exception as e:
            if "ya no es público" in str(e) or "token válido" in str(e):
                skipped += 1
                return f"{YELLOW}REQUIERE TOKEN (endpoint público deprecado){RESET}"
            raise
    t4()

    return player_ids


# ─────────────────────────────────────────────────────────────────
# TESTS PRIVADOS (con token)
# ─────────────────────────────────────────────────────────────────
def run_private_tests(client: LaLigaFantasyClient, player_ids: list[int]):
    print()
    print(f"  {BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}  ENDPOINTS PRIVADOS (con token, liga: {client.league_id}){RESET}")
    print(f"  {BOLD}{'='*60}{RESET}")

    manager_ids: list[str] = []

    @test("Privado: get_leagues")
    def t1():
        data = client.get_leagues()
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        assert len(data) > 0, "No se encontraron ligas"
        names = [l.get("name", "?") for l in data]
        return f"{len(data)} liga(s): {', '.join(names)}"
    t1()

    @test("Privado: get_players_raw")
    def t2():
        data = client.get_players_raw()
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        assert len(data) > 0, "Lista vacía"
        # Guardar player_ids si no los teníamos
        if not player_ids:
            for p in data[:5]:
                pid = p.get("id")
                if pid:
                    player_ids.append(int(pid))
        return f"{len(data)} jugadores"
    t2()

    @test("Privado: get_players (formateado)")
    def t3():
        data = client.get_players()
        assert isinstance(data, list) and len(data) > 0, "Lista vacía"
        first = data[0]
        return f"{len(data)} jugadores, ejemplo: {first.get('name', '?')} ({first.get('market_value', 0):,} €)"
    t3()

    @test("Privado: get_player_detail")
    def t4():
        if not player_ids:
            raise Exception("No hay player_ids")
        pid = player_ids[0]
        data = client.get_player_detail(pid)
        assert isinstance(data, dict), f"Esperaba dict, recibí {type(data)}"
        return f"ID {pid}: {data.get('nickname', '?')}"
    t4()

    @test("Privado: get_daily_market_raw")
    def t5():
        data = client.get_daily_market_raw()
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        return f"{len(data)} jugadores en el mercado"
    t5()

    @test("Privado: get_daily_market (formateado)")
    def t6():
        data = client.get_daily_market()
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        if data:
            first = data[0]
            return f"{len(data)} jugadores, ejemplo: {first.get('player_name', '?')} ({first.get('position', '?')}) - {first.get('sale_price', 0):,} €"
        return f"0 jugadores en el mercado (puede estar vacío)"
    t6()

    @test("Privado: get_ranking_raw")
    def t7():
        data = client.get_ranking_raw()
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        assert len(data) > 0, "Ranking vacío"
        # Guardar manager IDs
        for entry in data:
            tid = entry.get("team", {}).get("id")
            if tid:
                manager_ids.append(str(tid))
        top3 = []
        for entry in data[:3]:
            mgr = entry.get("team", {}).get("manager", {}).get("managerName", "?")
            pts = entry.get("points", 0)
            top3.append(f"{mgr} ({pts} pts)")
        return f"{len(data)} managers. Top 3: {', '.join(top3)}"
    t7()

    @test("Privado: get_manager_ids")
    def t8():
        ids = client.get_manager_ids()
        assert isinstance(ids, list) and len(ids) > 0, "Sin managers"
        return f"{len(ids)} managers: {ids[:3]}..."
    t8()

    @test("Privado: get_my_team_money")
    def t_money():
        money = client.get_my_team_money()
        assert isinstance(money, int), f"Esperaba int, recibí {type(money)}"
        assert money >= 0, f"Saldo inválido: {money}"
        return f"saldo={money:,} €"
    t_money()

    @test("Privado: get_team_raw (primer manager)")
    def t9():
        if not manager_ids:
            raise Exception("No hay manager_ids del ranking")
        mid = manager_ids[0]
        data = client.get_team_raw(mid)
        assert isinstance(data, dict), f"Esperaba dict, recibí {type(data)}"
        players = data.get("players", [])
        return f"Manager {mid}: {len(players)} jugadores en plantilla"
    t9()

    @test("Privado: get_team (formateado, primer manager)")
    def t10():
        if not manager_ids:
            raise Exception("No hay manager_ids del ranking")
        mid = manager_ids[0]
        data = client.get_team(mid)
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        if data:
            first = data[0]
            clause = first.get('buyout_clause') or 0
            return f"{len(data)} jugadores. Ejemplo: {first.get('player_name', '?')} ({first.get('position', '?')}) - cláusula: {clause:,} €"
        return f"0 jugadores"
    t10()

    @test("Privado: get_price_history")
    def t11():
        if not player_ids:
            raise Exception("No hay player_ids")
        pid = player_ids[0]
        data = client.get_price_history(pid)
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        return f"Jugador {pid}: {len(data)} registros"
    t11()

    @test("Privado: get_all_price_histories (3 jugadores)")
    def t12():
        test_ids = player_ids[:3]
        if not test_ids:
            raise Exception("No hay player_ids")
        data = client.get_all_price_histories(test_ids)
        assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
        return f"{len(data)} registros totales para {len(test_ids)} jugadores"
    t12()

    @test("Privado: get_activity_page_raw (página 1)")
    def t13():
        global skipped
        try:
            data = client.get_activity_page_raw(1)
            assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
            return f"{len(data)} noticias en página 1"
        except Exception as e:
            if "404" in str(e):
                skipped += 1
                return f"{YELLOW}ENDPOINT DEPRECADO (404){RESET}"
            raise
    t13()

    @test("Privado: get_activity_page (formateado, página 1)")
    def t14():
        global skipped
        try:
            data = client.get_activity_page(1)
            assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
            if data:
                first = data[0]
                return f"{len(data)} operaciones. Ejemplo: {first.get('origin', '?')} -> {first.get('player_name', '?')}"
            return f"0 operaciones de mercado en pág 1"
        except Exception as e:
            if "404" in str(e):
                skipped += 1
                return f"{YELLOW}ENDPOINT DEPRECADO (404){RESET}"
            raise
    t14()

    @test("Privado: get_full_activity (máx 3 páginas)")
    def t15():
        global skipped
        try:
            data = client.get_full_activity(max_pages=3)
            assert isinstance(data, list), f"Esperaba lista, recibí {type(data)}"
            return f"{len(data)} operaciones totales (hasta 3 páginas)"
        except Exception as e:
            if "404" in str(e):
                skipped += 1
                return f"{YELLOW}ENDPOINT DEPRECADO (404){RESET}"
            raise
    t15()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test de todos los endpoints")
    parser.add_argument("--league", type=str, default="", help="League ID")
    args = parser.parse_args()

    print()
    print(f"  {BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}  TEST DE TODOS LOS ENDPOINTS - LaLiga Fantasy API{RESET}")
    print(f"  {BOLD}{'='*60}{RESET}")

    t_start = time.time()

    # --- Tests públicos ---
    player_ids = run_public_tests()

    # --- Tests privados ---
    token = load_token()
    if token:
        league_id = args.league or ""
        client = LaLigaFantasyClient(token=token, league_id=league_id)
        if not client.league_id:
            # Intentar obtener liga automáticamente
            try:
                leagues = client.get_leagues()
                if leagues:
                    client.league_id = str(leagues[0].get("id", ""))
            except Exception:
                pass

        if client.league_id:
            run_private_tests(client, player_ids)
        else:
            print(f"\n  {YELLOW}SKIP: Tests privados — no se pudo determinar league_id{RESET}")
            print(f"  Usa: python test_all_apis.py --league TU_LEAGUE_ID")
    else:
        print(f"\n  {YELLOW}SKIP: Tests privados — no hay token guardado{RESET}")
        print(f"  Usa: python laliga_fantasy_client.py --google para obtener uno")

    # --- Resumen ---
    elapsed = time.time() - t_start
    print()
    print(f"  {BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}  RESUMEN{RESET}")
    print(f"  {BOLD}{'='*60}{RESET}")
    print()
    for name, status, detail in results:
        icon = f"{GREEN}✓{RESET}" if status == "OK" else f"{RED}✗{RESET}"
        print(f"  {icon} {name}")
    print()
    total = passed + failed
    color = GREEN if failed == 0 else RED
    print(f"  {color}{BOLD}{passed}/{total} tests pasaron{RESET} en {elapsed:.1f}s")
    if skipped > 0:
        print(f"  {YELLOW}{skipped} endpoint(s) deprecados (404){RESET}")
    if failed > 0:
        print(f"  {RED}{failed} test(s) fallaron{RESET}")
    print()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
