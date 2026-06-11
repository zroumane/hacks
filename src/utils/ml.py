import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

MODELS_DIR = Path(__file__).resolve().parents[2] / "output" / "models"


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Métrique officielle de la compétition."""
    return float(mean_squared_error(y_true, np.clip(y_pred, 0, 1)))


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> XGBRegressor:
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = XGBRegressor(**default_params)

    eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )
    return model


def cross_validate_ts(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
    params: dict | None = None,
) -> list[float]:
    """Validation croisée temporelle — retourne les Brier Scores par fold."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = train_xgb(X_tr, y_tr, X_val, y_val, params)
        preds = model.predict(X_val)
        score = brier_score(y_val.values, preds)
        scores.append(score)
        print(f"  Fold {fold + 1} — Brier Score : {score:.4f}")

    print(f"  Moyenne : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores


def save_model(model, name: str) -> Path:
    """Sauvegarde le modèle dans output/models/<name>.pkl"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé : {path}")
    return path


def load_model(name: str):
    """Charge un modèle depuis output/models/<name>.pkl"""
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
