"""
Entrenamiento del modelo xP (Expected Points)
===============================================
Entrena un regresor XGBoost y LightGBM para predecir puntos Fantasy.

Estrategia: Walk-Forward Validation (expanding window)
    Para cada jornada k (desde MIN_TRAIN_WEEKS en adelante):
        - Train:  jornadas 1 .. k-2
        - Val:    jornada k-1  (para early stopping)
        - Test:   jornada k    (predicción "out-of-sample")

    Todas las predicciones OOF se acumulan y se evalúan juntas.
    El modelo FINAL se entrena con TODAS las jornadas (para máxima información).

Uso:
    python -m prediction.train
    python -m prediction.train --model xgboost
    python -m prediction.train --model lightgbm
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

# Mínimo de jornadas de entrenamiento antes de empezar a validar.
# Necesitamos al menos ~5 jornadas para que los rolling averages tengan sentido
# + 1 de validación, así que empezamos a predecir desde la jornada 7 en adelante.
MIN_TRAIN_WEEKS = 6


def load_features(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el CSV de features."""
    if path is None:
        path = DATA_DIR / "features.csv"
    return pd.read_csv(path)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Obtiene las columnas feature (excluyendo meta y target)."""
    meta = {"player_id", "nombre", "posicion", "equipo", "jornada", "puntos"}
    return [c for c in df.columns if c not in meta]


# ─── Modelos ──────────────────────────────────────────────────

def _make_xgboost(early_stopping: int = 50):
    """Crea una instancia de XGBRegressor."""
    import xgboost as xgb
    return xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping,
    )


def _make_lightgbm(early_stopping: int = 50):
    """Crea una instancia de LGBMRegressor."""
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _fit_model(model, model_type: str,
               X_train, y_train, X_val, y_val, verbose: int = 0):
    """Entrena un modelo con early stopping."""
    if model_type == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )
    elif model_type == "lightgbm":
        import lightgbm as lgb
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    return model


# ─── Walk-Forward Validation ─────────────────────────────────

def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Walk-Forward Validation con expanding window.

    Para cada jornada k (MIN_TRAIN_WEEKS+1 .. max_jornada):
        Train:  j1 .. k-2
        Val:    j k-1  (early stopping)
        Predict: j k   (out-of-sample)

    Returns:
        oof_df:      DataFrame con TODAS las predicciones out-of-sample
        fold_metrics: lista de métricas por fold [{jornada, mae, rmse, r2, n_train, n_pred}, ...]
    """
    jornadas = sorted(df["jornada"].unique())
    max_j = max(jornadas)
    min_predict = MIN_TRAIN_WEEKS + 1  # Primera jornada que predecimos

    oof_parts = []
    fold_metrics = []

    logger.info(
        "Walk-Forward: predecir desde j%d hasta j%d (%d folds)",
        min_predict, max_j, max_j - min_predict + 1,
    )

    for k in range(min_predict, max_j + 1):
        # Train: todas las jornadas hasta k-2
        # Val: jornada k-1 (para early stopping)
        # Test: jornada k (out-of-sample)
        train_mask = df["jornada"] < (k - 1)
        val_mask = df["jornada"] == (k - 1)
        test_mask = df["jornada"] == k

        train_data = df[train_mask]
        val_data = df[val_mask]
        test_data = df[test_mask]

        if len(train_data) < 50 or len(val_data) == 0 or len(test_data) == 0:
            continue

        X_train = train_data[feature_cols]
        y_train = train_data["puntos"]
        X_val = val_data[feature_cols]
        y_val = val_data["puntos"]
        X_test = test_data[feature_cols]
        y_test = test_data["puntos"]

        # Crear y entrenar modelo para este fold
        if model_type == "xgboost":
            model = _make_xgboost()
        else:
            model = _make_lightgbm()

        model = _fit_model(model, model_type, X_train, y_train, X_val, y_val)

        # Predecir la jornada k
        preds = model.predict(X_test)

        # Guardar predicciones OOF
        oof = test_data[["player_id", "nombre", "posicion", "equipo", "jornada", "puntos"]].copy()
        oof["xP"] = preds.round(2)
        oof["error"] = (oof["puntos"] - oof["xP"]).abs()
        oof_parts.append(oof)

        # Métricas de este fold
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds) if len(y_test) > 1 else 0.0

        fold_metrics.append({
            "jornada": k,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
            "n_train": len(train_data),
            "n_pred": len(test_data),
        })

        logger.info(
            "  Fold j%d: train=%d, pred=%d → MAE=%.2f RMSE=%.2f R²=%.3f",
            k, len(train_data), len(test_data), mae, rmse, r2,
        )

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()

    return oof_df, fold_metrics


# ─── Entrenar modelo final ──────────────────────────────────

def train_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
) -> tuple:
    """
    Entrena el modelo FINAL con TODOS los datos disponibles.

    Usa las últimas 2 jornadas como val para early stopping,
    pero el train incluye todo lo anterior.
    """
    max_j = df["jornada"].max()
    val_j = max_j  # Última jornada como val para early stopping
    train_mask = df["jornada"] < val_j
    val_mask = df["jornada"] == val_j

    X_train = df[train_mask][feature_cols]
    y_train = df[train_mask]["puntos"]
    X_val = df[val_mask][feature_cols]
    y_val = df[val_mask]["puntos"]

    logger.info(
        "Modelo FINAL: train=j1-%d (%d filas), val=j%d (%d filas)",
        val_j - 1, len(X_train), val_j, len(X_val),
    )

    if model_type == "xgboost":
        model = _make_xgboost()
    else:
        model = _make_lightgbm()

    model = _fit_model(model, model_type, X_train, y_train, X_val, y_val, verbose=50)

    return model, model_type


# ─── Utilidades ──────────────────────────────────────────────

def evaluate_oof(oof_df: pd.DataFrame) -> dict:
    """Evalúa métricas globales sobre todas las predicciones OOF."""
    if oof_df.empty:
        return {"mae": 0, "rmse": 0, "r2": 0}

    y_true = oof_df["puntos"]
    y_pred = oof_df["xP"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def feature_importance(model, feature_cols: list[str], model_type: str) -> pd.DataFrame:
    """Obtiene la importancia de features."""
    importance = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    return fi


def save_model(model, model_type: str, metrics: dict, feature_cols: list[str]):
    """Guarda el modelo y metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{model_type}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "model_type": model_type,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "training_strategy": "walk_forward_validation",
    }
    meta_path = MODELS_DIR / f"{model_type}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Modelo guardado: %s", model_path)
    return model_path


# ─── Main ────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Entrena el modelo xP")
    parser.add_argument(
        "--model", choices=["xgboost", "lightgbm", "both"],
        default="both", help="Qué modelo entrenar",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  xP Model Training — Walk-Forward Validation")
    print("=" * 60)

    # Cargar features
    df = load_features()
    jornadas = sorted(df["jornada"].unique())
    print(f"\n  Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"  Jornadas: {int(min(jornadas))} - {int(max(jornadas))} ({len(jornadas)} jornadas)")
    print(f"  Jugadores: {df['player_id'].nunique()}")

    feature_cols = get_feature_cols(df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Walk-Forward: predecir desde j{MIN_TRAIN_WEEKS + 1} hasta j{int(max(jornadas))}")

    results = {}
    all_models = []
    if args.model in ("xgboost", "both"):
        all_models.append("xgboost")
    if args.model in ("lightgbm", "both"):
        all_models.append("lightgbm")

    for model_type in all_models:
        print()
        print(f"  {'─' * 50}")
        print(f"  {model_type.upper()}")
        print(f"  {'─' * 50}")

        # ── Walk-Forward Validation ───────────────────────
        print(f"\n  Walk-Forward Validation...")
        oof_df, fold_metrics = walk_forward_validate(df, feature_cols, model_type)

        if oof_df.empty:
            print("  No hay suficientes datos para walk-forward.")
            continue

        # Métricas globales OOF
        global_metrics = evaluate_oof(oof_df)
        print(f"\n  Métricas OOF globales ({len(oof_df)} predicciones, "
              f"{len(fold_metrics)} folds):")
        print(f"    MAE  = {global_metrics['mae']:.3f}")
        print(f"    RMSE = {global_metrics['rmse']:.3f}")
        print(f"    R²   = {global_metrics['r2']:.3f}")

        # Evolución por fold
        print(f"\n  Evolución por jornada:")
        print(f"  {'J':>4} {'Train':>7} {'Pred':>6} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
        print(f"  {'─'*4} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*7}")
        for fm in fold_metrics:
            print(
                f"  {fm['jornada']:>4} {fm['n_train']:>7} {fm['n_pred']:>6} "
                f"{fm['mae']:>7.2f} {fm['rmse']:>7.2f} {fm['r2']:>7.3f}"
            )

        # ── Entrenar modelo FINAL ─────────────────────────
        print(f"\n  Entrenando modelo FINAL con TODOS los datos...")
        final_model, mt = train_final_model(df, feature_cols, model_type)

        # Feature importance
        fi = feature_importance(final_model, feature_cols, model_type)
        print(f"\n  Top 10 features ({model_type}):")
        max_imp = fi["importance"].max() if not fi.empty else 1
        for _, row in fi.head(10).iterrows():
            bar_len = int(row["importance"] / max_imp * 30) if max_imp > 0 else 0
            bar = "█" * bar_len
            print(f"    {row['feature']:<25} {row['importance']:.4f} {bar}")

        # Guardar
        save_metrics = {
            "walk_forward_oof": global_metrics,
            "folds": len(fold_metrics),
            "total_oof_predictions": len(oof_df),
            "fold_details": fold_metrics,
        }
        save_model(final_model, model_type, save_metrics, feature_cols)

        results[model_type] = global_metrics

        # Guardar OOF predictions
        oof_path = DATA_DIR / f"oof_{model_type}.csv"
        oof_df.to_csv(oof_path, index=False)
        print(f"\n  Predicciones OOF guardadas: {oof_path}")

    # ── Resumen comparativo ───────────────────────────────────
    if len(results) >= 2:
        print()
        print("  " + "=" * 50)
        print("  COMPARACIÓN (Walk-Forward OOF)")
        print("  " + "=" * 50)
        print(f"  {'Modelo':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
        print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8}")
        for name, m in results.items():
            print(f"  {name:<15} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f}")

        best = min(results, key=lambda k: results[k]["mae"])
        print(f"\n  Mejor modelo: {best} (menor MAE)")

    # ── Muestra de predicciones OOF ───────────────────────────
    if results:
        # Usar el mejor modelo y sus OOF
        best_type = min(results, key=lambda k: results[k]["mae"])
        oof_path = DATA_DIR / f"oof_{best_type}.csv"
        if oof_path.exists():
            oof = pd.read_csv(oof_path)
            last_j = oof["jornada"].max()
            muestra = oof[oof["jornada"] == last_j].nlargest(15, "xP")

            print(f"\n  Muestra OOF — Jornada {int(last_j)} ({best_type}):")
            print(f"  {'Jugador':<22} {'Pos':>4} {'Real':>5} {'xP':>5} {'Error':>6}")
            print(f"  {'─'*22} {'─'*4} {'─'*5} {'─'*5} {'─'*6}")
            for _, row in muestra.iterrows():
                print(
                    f"  {row['nombre']:<22} {row['posicion']:>4} "
                    f"{row['puntos']:>5.0f} {row['xP']:>5.1f} {row['error']:>6.1f}"
                )

    print()


if __name__ == "__main__":
    main()
