"""
Predicción de puntos Fantasy (xP — Expected Points)
=====================================================
Sistema de ML para predecir cuántos puntos hará cada jugador
en la próxima jornada de LaLiga Fantasy.

Módulos:
    collect_data  — Recolecta datos históricos de múltiples fuentes
    features      — Feature Engineering: transforma datos brutos en features
    train         — Entrena el modelo xgboost
    predict       — Genera predicciones para la próxima jornada
    advisor       — Conecta xP con tu liga y genera informe de recomendaciones
    langchain_tools       — Herramientas para agente LLM (snapshot/xP/mercado/API)
    langchain_agent       — Ejecutor de agente LangChain por objetivo/fase
    langchain_autonomous  — Daemon autónomo diario PRE/POST usando el agente
"""
