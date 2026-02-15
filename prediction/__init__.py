"""
Predicción de puntos Fantasy (xP — Expected Points)
=====================================================
Sistema de ML para predecir cuántos puntos hará cada jugador
en la próxima jornada de LaLiga Fantasy.

Módulos:
    collect_data  — Recolecta datos históricos de múltiples fuentes
    features      — Feature Engineering: transforma datos brutos en features
    train         — Entrena el modelo XGBoost/LightGBM
    predict       — Genera predicciones para la próxima jornada
    advisor       — Conecta xP con tu liga y genera informe de recomendaciones
"""
