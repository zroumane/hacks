"""
Étape 3 — Features physiques ISO 9223 / FAA AC 43-4B sur CatBoost.

Ajoute les features physiques à CatBoost et compare le gain en CV.
Si TabPFN est disponible, le ré-entraîne aussi avec les nouvelles features.

Features ajoutées :
  - cum_wet     : proxy Time of Wetness cumulé (Σ mois RH > 80 %)
  - salt_active : sel marin × (RH > 75 %) — seuil de déliquescence NaCl
  - log_tow     : log1p(cum_wet) — structure log dose-response ISO 9223
  - iso_cross   : log1p(TOW) × log1p(SO₂) — terme croisé ISO 9223
  - sea_salt_total : somme des 3 fractions granulométriques

Usage : uv run src/algo/step3_physics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from utils.io import load_input
from utils.features import (
    FEATURE_COLS_PHYSICS,
    add_age_feature,
    add_physical_features,
    add_sea_salt_total,
    build_binary_labels,
)
from utils.ml import (
    HAS_CATBOOST,
    HAS_TABPFN,
    brier_score,
    cross_validate_group,
    predict_proba_tabpfn,
    save_model,
    train_catboost,
    train_tabpfn,
)

BUFFER_MONTHS = 12
MAX_TABPFN    = 9000
N_CLUSTERS    = 3000


# ── 1. Chargement & features ───────────────────────────────────────────────────
print("Chargement des données...")
corr      = load_input("corrosions_training.csv")
env_train = load_input("environment_training.csv")

corr["observation_date"] = pd.to_datetime(corr["observation_date"])
merged = env_train.merge(
    corr[["aircraft_id", "observation_date", "aircraft_delivery_year", "aircraft_delivery_month"]],
    on="aircraft_id",
    how="inner",
)
merged["month_dt"] = pd.to_datetime(merged["year_month"])
merged = merged[merged["month_dt"] <= merged["observation_date"]].copy()
merged["months_until"] = (
    (merged["observation_date"].dt.year  - merged["month_dt"].dt.year)  * 12
    + (merged["observation_date"].dt.month - merged["month_dt"].dt.month)
)

# Features physiques — ordre important : sea_salt_total avant physical_features
merged = add_sea_salt_total(merged)
merged = add_age_feature(merged, merged["aircraft_delivery_year"], merged["aircraft_delivery_month"])
merged = merged.sort_values(["aircraft_id", "month_dt"]).reset_index(drop=True)
merged = add_physical_features(merged)   # cum_wet nécessite le tri par avion+date

y      = build_binary_labels(merged, buffer_months=BUFFER_MONTHS)
X      = merged[FEATURE_COLS_PHYSICS].fillna(0)
groups = merged["aircraft_id"]

print(f"  {len(merged)} lignes | {len(FEATURE_COLS_PHYSICS)} features | positifs : {int(y.sum())}")


# ── 2. CatBoost + features physiques ──────────────────────────────────────────
if HAS_CATBOOST:
    print("\nValidation GroupKFold CatBoost + features physiques (5 folds)...")
    cross_validate_group(
        X, y, groups,
        model_fn=lambda Xtr, ytr, Xv, yv: train_catboost(Xtr, ytr, Xv, yv),
        n_splits=5,
    )
    print("\nEntraînement final CatBoost + physics...")
    model_cb = train_catboost(X, y)
    preds_cb = model_cb.predict_proba(X)[:, 1]
    print(f"  Brier Score (train) : {brier_score(y.values, preds_cb):.4f}")
    save_model(model_cb, "catboost_step3")
else:
    print("CatBoost non disponible — skipped.")


# ── 3. TabPFN + features physiques (si disponible) ────────────────────────────
if HAS_TABPFN:
    print("\nSous-échantillonnage K-Means pour TabPFN...")
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    X_neg   = X.iloc[neg_idx].values
    kmeans  = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3)
    kmeans.fit(X_neg)
    dists  = np.linalg.norm(X_neg - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    sel_neg = [neg_idx[kmeans.labels_ == c][np.argmin(dists[kmeans.labels_ == c])]
               for c in range(N_CLUSTERS) if (kmeans.labels_ == c).any()]
    sel_idx = np.sort(np.concatenate([pos_idx, np.array(sel_neg)]))

    X_sub = X.iloc[sel_idx].reset_index(drop=True)
    y_sub = y.iloc[sel_idx].reset_index(drop=True)
    g_sub = groups.iloc[sel_idx].reset_index(drop=True)
    print(f"  {len(X_sub)} lignes après sous-échantillonnage")

    print("\nValidation GroupKFold TabPFN + features physiques (5 folds)...")
    cross_validate_group(
        X_sub, y_sub, g_sub,
        model_fn=lambda Xtr, ytr, Xv, yv: train_tabpfn(Xtr, ytr),
        predict_fn=predict_proba_tabpfn,
        n_splits=5,
    )
    print("\nEntraînement final TabPFN + physics...")
    model_tf = train_tabpfn(X_sub, y_sub)
    preds_tf = predict_proba_tabpfn(model_tf, X_sub)
    print(f"  Brier Score (train sous-éch.) : {brier_score(y_sub.values, preds_tf):.4f}")
    save_model(model_tf, "tabpfn_step3")
else:
    print("TabPFN non disponible — skipped.")

print("\nÉtape 3 terminée. Lancer step4_ensemble.py pour la soumission finale.")
