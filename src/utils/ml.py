import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from xgboost import XGBRegressor

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False

MODELS_DIR = Path(__file__).resolve().parents[2] / "output" / "models"


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Métrique officielle de la compétition."""
    return float(mean_squared_error(y_true, np.clip(y_pred, 0, 1)))


# ── XGBoost (régression, labels continus) ─────────────────────────────────────

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
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model


# ── CatBoost (classification, labels binaires) ────────────────────────────────

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> "CatBoostClassifier":
    if not HAS_CATBOOST:
        raise ImportError("catboost non installé : pip install catboost")
    default_params = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "verbose": False,
    }
    if params:
        default_params.update(params)
    model = CatBoostClassifier(**default_params)
    eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set)
    return model


# ── TabPFN (classification, labels binaires) ──────────────────────────────────

def _resolve_device(device: str | None) -> str:
    """Renvoie le device demandé, sinon 'cuda' si dispo, sinon 'cpu'."""
    if device is not None:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def train_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    device: str | None = None,
) -> "TabPFNClassifier":
    if not HAS_TABPFN:
        raise ImportError("tabpfn non installé : pip install tabpfn")
    clf = TabPFNClassifier(device=_resolve_device(device))
    clf.fit(X_train.values, y_train.values)
    return clf


def predict_proba_tabpfn(model, X: pd.DataFrame) -> np.ndarray:
    """Retourne P(classe=1) par batch de 5 000 pour éviter OOM."""
    preds = []
    for i in range(0, len(X), 5000):
        batch = X.iloc[i : i + 5000].values
        preds.append(model.predict_proba(batch)[:, 1])
    return np.concatenate(preds)


# ── Sous-échantillonnage des négatifs (pour TabPFN, limite ~10k lignes) ───────

def subsample_negatives_kmeans(
    X: pd.DataFrame,
    y: pd.Series,
    n_clusters: int = 3000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Garde tous les positifs + un représentant par cluster K-Means des négatifs.

    Les features sont **standardisées avant le clustering** : sans ça, K-Means est
    dominé par les colonnes de grande magnitude (ex. total_parking_minutes ~1e4)
    et ignore les ratios d'aérosols (~1e-10), ce qui ruine la diversité visée.

    Retourne les indices (triés) à conserver, dans le référentiel de X.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    y_arr   = np.asarray(y)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    if len(neg_idx) <= n_clusters:
        return np.sort(np.concatenate([pos_idx, neg_idx]))

    X_neg  = StandardScaler().fit_transform(X.iloc[neg_idx].values)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    labels = kmeans.fit_predict(X_neg)
    dists  = np.linalg.norm(X_neg - kmeans.cluster_centers_[labels], axis=1)

    selected = [
        neg_idx[labels == c][np.argmin(dists[labels == c])]
        for c in range(n_clusters)
        if (labels == c).any()
    ]
    return np.sort(np.concatenate([pos_idx, np.array(selected)]))


# ── Validation croisée GroupKFold (hold-out par avion) ────────────────────────

def cross_validate_group(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_fn,
    n_splits: int = 5,
    predict_fn=None,
) -> list[float]:
    """
    GroupKFold par aircraft_id : tous les mois d'un avion restent dans le même fold.
    Évalue le Brier Score sur chaque fold et affiche la moyenne.

    model_fn(X_tr, y_tr, X_val, y_val) → modèle entraîné
    predict_fn(model, X_val) → array de probabilités  (optionnel, sinon predict_proba)
    """
    gkf = GroupKFold(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val_f = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn(X_tr, y_tr, X_val_f, y_val_f)

        if predict_fn is not None:
            preds = predict_fn(model, X_val_f)
        elif hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_val_f)[:, 1]
        else:
            preds = np.clip(model.predict(X_val_f), 0, 1)

        score = brier_score(y_val_f.values, preds)
        scores.append(score)
        n_pos = int(y_val_f.sum())
        n_avions = groups.iloc[val_idx].nunique()
        print(f"  Fold {fold + 1} — Brier : {score:.4f}  ({n_avions} avions, {n_pos} positifs)")

    print(f"  Moyenne : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores


# ── Validation croisée temporelle (legacy) ────────────────────────────────────

def cross_validate_ts(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
    params: dict | None = None,
) -> list[float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val_f = y.iloc[train_idx], y.iloc[val_idx]
        model = train_xgb(X_tr, y_tr, X_val_f, y_val_f, params)
        preds = model.predict(X_val_f)
        score = brier_score(y_val_f.values, preds)
        scores.append(score)
        print(f"  Fold {fold + 1} — Brier Score : {score:.4f}")
    print(f"  Moyenne : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_predict(
    models: list,
    X: pd.DataFrame,
    weights: list[float] | None = None,
    clip_range: tuple[float, float] = (0.02, 0.98),
) -> np.ndarray:
    """
    Moyenne pondérée des probabilités de N modèles.
    Chaque modèle doit exposer predict_proba(X)[:, 1] ou predict(X).
    """
    if weights is None:
        weights = [1.0] * len(models)
    w_total = sum(weights)

    result = np.zeros(len(X))
    for model, w in zip(models, weights):
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = np.clip(model.predict(X), 0, 1)
        result += w * preds

    result /= w_total
    return np.clip(result, *clip_range)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé : {path}")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_models() -> list[tuple[str, object]]:
    """Charge tous les .pkl dans output/models/ → liste de (nom, modèle)."""
    if not MODELS_DIR.exists():
        return []
    models = []
    for path in sorted(MODELS_DIR.glob("*.pkl")):
        with open(path, "rb") as f:
            models.append((path.stem, pickle.load(f)))
    return models
