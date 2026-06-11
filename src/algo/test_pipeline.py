"""
Test rapide du pipeline complet (~30 secondes, pas de GPU).

Mêmes étapes que step1→step4 mais avec :
  - 50 estimateurs au lieu de 500
  - 2 folds au lieu de 5
  - Pas de TabPFN

Usage : uv run src/algo/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import load_input, save_output
from utils.features import (
    FEATURE_COLS_BASE, FEATURE_COLS_PHYSICS,
    add_age_feature, add_physical_features, add_sea_salt_total,
    build_binary_labels, check_inspection_bias,
)
from utils.ml import (
    HAS_CATBOOST, brier_score, cross_validate_group, save_model,
    train_catboost, ensemble_predict,
)

BUFFER_MONTHS = 12
FAST_PARAMS   = {"iterations": 50, "learning_rate": 0.1, "depth": 4, "verbose": False}

if not HAS_CATBOOST:
    print("ERREUR : pip install catboost"); sys.exit(1)

# ── Données ────────────────────────────────────────────────────────────────────
print("Chargement...")
corr      = load_input("corrosions_training.csv")
env_train = load_input("environment_training.csv")
env_test  = load_input("environment_test.csv")
sample    = load_input("sample_submission.csv") if Path("input/sample_submission.csv").exists() else load_input("test.csv")

corr["observation_date"] = pd.to_datetime(corr["observation_date"])
merged = env_train.merge(
    corr[["aircraft_id", "observation_date", "aircraft_delivery_year", "aircraft_delivery_month"]],
    on="aircraft_id", how="inner",
)
merged["month_dt"] = pd.to_datetime(merged["year_month"])
merged = merged[merged["month_dt"] <= merged["observation_date"]].copy()
merged["months_until"] = (
    (merged["observation_date"].dt.year  - merged["month_dt"].dt.year)  * 12
    + (merged["observation_date"].dt.month - merged["month_dt"].dt.month)
)
merged = add_sea_salt_total(merged)
merged = add_age_feature(merged, merged["aircraft_delivery_year"], merged["aircraft_delivery_month"])
merged = merged.sort_values(["aircraft_id", "month_dt"]).reset_index(drop=True)
merged = add_physical_features(merged)

y      = build_binary_labels(merged, buffer_months=BUFFER_MONTHS)
X      = merged[FEATURE_COLS_PHYSICS].fillna(0)
groups = merged["aircraft_id"]
print(f"  {len(merged)} lignes | positifs : {int(y.sum())} ({y.mean():.1%})")

# ── CV rapide ─────────────────────────────────────────────────────────────────
print("\nCV GroupKFold (2 folds)...")
cross_validate_group(
    X, y, groups,
    model_fn=lambda Xtr, ytr, Xv, yv: train_catboost(Xtr, ytr, Xv, yv, FAST_PARAMS),
    n_splits=2,
)

# ── Entraînement final ─────────────────────────────────────────────────────────
print("\nEntraînement final...")
model = train_catboost(X, y, params=FAST_PARAMS)
save_model(model, "catboost_test")

# ── Prédiction test ───────────────────────────────────────────────────────────
print("\nPrédiction...")
first_month = env_test.groupby("aircraft_id")["year_month"].min().reset_index()
first_month.columns = ["aircraft_id", "estimated_delivery"]
first_month["delivery_year"]  = pd.to_datetime(first_month["estimated_delivery"]).dt.year
first_month["delivery_month"] = pd.to_datetime(first_month["estimated_delivery"]).dt.month
env_test = env_test.merge(first_month[["aircraft_id", "delivery_year", "delivery_month"]], on="aircraft_id")
env_test = add_sea_salt_total(env_test)
env_test = add_age_feature(env_test, env_test["delivery_year"], env_test["delivery_month"])
env_test = env_test.sort_values(["aircraft_id", "year_month"]).reset_index(drop=True)
env_test = add_physical_features(env_test)

X_test = env_test[FEATURE_COLS_PHYSICS].fillna(0)
env_test["corrosion_risk"] = np.clip(model.predict_proba(X_test)[:, 1], 0.02, 0.98)
env_test["id"] = env_test["aircraft_id"] + "_" + env_test["year_month"]

submission = sample.copy()
submission["corrosion_risk"] = submission["id"].map(env_test.set_index("id")["corrosion_risk"]).fillna(0.5)

path = save_output(submission, "submission_test.csv")
print(f"\nOK — soumission : {path}")
print(submission["corrosion_risk"].describe().round(3).to_string())
