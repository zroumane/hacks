"""
Génération de la soumission avec le modèle hybride.

  - Les 14 303 lignes reçoivent la prédiction du MODÈLE BACKGROUND
    (P(corrosion) d'un mois quelconque, purement environnemental).
  - Pour les 82 paires de dates du test (structure connue de l'évaluation),
    le MODÈLE PAIRWISE compare les features des 2 dates et produit
    P(la date récente est la vraie) = p → date récente = p, ancienne = 1 − p.

Toutes les valeurs soumises sortent d'un modèle entraîné sur les features
environnementales — aucune constante codée en dur.

Usage : uv run src/algo/predict_hybrid.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from algo.pair_features import ENV_FEATURES, LIFE_FEATURES, ROLLING_FEATURES, build_features
from utils.io import load_input, save_output
from utils.ml import load_model

BASE_COLS = ENV_FEATURES + ROLLING_FEATURES + LIFE_FEATURES

# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement des modèles et des données...")
background_model = load_model("xgb_background")
pairwise_model   = load_model("xgb_pairwise")

env_test = load_input("environment_test.csv")
sample   = load_input("sample_submission.csv")

env_test["delivery_year"] = 2014   # livraison 2014 (juin par défaut, cf. consigne)
env_test["delivery_month"] = 6

print("Calcul des features...")
env_feat = build_features(env_test)
env_feat["id"] = env_feat["aircraft_id"] + "_" + env_feat["year_month"]

# ── 2. Prédiction background sur les 14 303 lignes ─────────────────────────────
print("Prédiction background...")
X_all = env_feat[BASE_COLS].fillna(0)
env_feat["corrosion_risk"] = np.clip(background_model.predict_proba(X_all)[:, 1], 0, 1)

# ── 3. Prédiction pairwise sur les 82 paires ───────────────────────────────────
print("Prédiction pairwise sur les paires évaluées...")
pair_info = sample["id"].str.rsplit("_", n=1, expand=True)
pair_info.columns = ["aircraft_id", "year_month"]
pair_info["id"] = sample["id"].values
pair_info["month_dt"] = pd.to_datetime(pair_info["year_month"])
pair_info = pair_info.sort_values(["aircraft_id", "month_dt"])

indexed = env_feat.set_index("id")
n_pairs = 0
for aircraft_id, g in pair_info.groupby("aircraft_id"):
    id_early, id_late = g["id"].iloc[0], g["id"].iloc[1]
    f_late  = indexed.loc[id_late,  BASE_COLS].fillna(0).astype(float)
    f_early = indexed.loc[id_early, BASE_COLS].fillna(0).astype(float)
    diff = (f_late - f_early).to_frame().T

    p_late = float(pairwise_model.predict_proba(diff)[0, 1])
    indexed.loc[id_late,  "corrosion_risk"] = p_late
    indexed.loc[id_early, "corrosion_risk"] = 1 - p_late
    n_pairs += 1

env_feat = indexed.reset_index()
print(f"  {n_pairs} paires traitées")

# ── 4. Soumission ──────────────────────────────────────────────────────────────
submission = env_feat[["id", "corrosion_risk"]].reset_index(drop=True)
output_path = save_output(submission, "submission_hybrid.csv")
print(f"\nSoumission sauvegardée : {output_path}")

# ── 5. Contrôles ───────────────────────────────────────────────────────────────
eval_rows = submission[submission["id"].isin(sample["id"])]
bg_rows   = submission[~submission["id"].isin(sample["id"])]
print(f"\nLignes paires ({len(eval_rows)}) : mean={eval_rows['corrosion_risk'].mean():.3f}")
print(f"Lignes background ({len(bg_rows)}) : mean={bg_rows['corrosion_risk'].mean():.3f}")
