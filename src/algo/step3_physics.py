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

from utils.io import load_input
from utils.features import FEATURE_COLS_PHYSICS, build_training_frame
from utils.ml import (
    HAS_CATBOOST,
    HAS_TABPFN,
    brier_score,
    cross_validate_group,
    predict_proba_tabpfn,
    save_model,
    subsample_negatives_kmeans,
    train_catboost,
    train_tabpfn,
)

BUFFER_MONTHS = 12
GREY_MONTHS   = 6      # zone grise exclue juste avant la détection (censure)
MAX_TABPFN    = 9000
N_CLUSTERS    = 3000


# ── 1. Chargement & features ───────────────────────────────────────────────────
print("Chargement des données...")
corr      = load_input("corrosions_training.csv")
env_train = load_input("environment_training.csv")

merged, X, y, groups = build_training_frame(
    corr, env_train, buffer_months=BUFFER_MONTHS, grey_months=GREY_MONTHS, physics=True,
)

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
    print("\nSous-échantillonnage K-Means pour TabPFN (features standardisées)...")
    sel_idx = subsample_negatives_kmeans(X, y, n_clusters=N_CLUSTERS)

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
