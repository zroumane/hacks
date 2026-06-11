"""
Génération de la soumission avec le modèle de paires (xgb_pairs.pkl).

Le modèle prédit P(corrosion détectée ce mois-ci) pour chaque ligne
aircraft_id × year_month du test, à partir des features environnementales
(exposition cumulée, rolling 3/12/24m, âge...). Aucune règle codée en dur :
les 14 303 prédictions sortent toutes du modèle.

Usage : uv run src/algo/predict_pairs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from algo.pair_features import FEATURE_COLS, build_features
from utils.io import load_input, save_output
from utils.ml import load_model

# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement du modèle et des données de test...")
model = load_model("xgb_pairs")
env_test = load_input("environment_test.csv")

# Les avions de test ont tous été livrés en 2014 (juin par défaut, cf. consigne)
env_test["delivery_year"] = 2014
env_test["delivery_month"] = 6

# ── 2. Features (identiques à l'entraînement via pair_features) ────────────────
print("Calcul des features...")
env_feat = build_features(env_test)

# ── 3. Prédiction sur les 14 303 lignes ────────────────────────────────────────
print("Prédiction...")
X_test = env_feat[FEATURE_COLS].fillna(0)
env_feat["corrosion_risk"] = np.clip(model.predict_proba(X_test)[:, 1], 0, 1)

# ── 4. Soumission ──────────────────────────────────────────────────────────────
env_feat["id"] = env_feat["aircraft_id"] + "_" + env_feat["year_month"]
submission = env_feat[["id", "corrosion_risk"]].reset_index(drop=True)

output_path = save_output(submission, "submission_pairs_model.csv")
print(f"\nSoumission sauvegardée : {output_path}")
print(f"{len(submission)} lignes")
print("\nDistribution des prédictions :")
print(submission["corrosion_risk"].describe().round(4).to_string())

# ── 5. Contrôle sur les 164 lignes évaluées ────────────────────────────────────
sample = load_input("sample_submission.csv")
eval_rows = submission[submission["id"].isin(sample["id"])].copy()
parts = eval_rows["id"].str.rsplit("_", n=1, expand=True)
eval_rows["aircraft_id"] = parts[0]
eval_rows["month"] = pd.to_datetime(parts[1])

later_higher = 0
for _, g in eval_rows.sort_values("month").groupby("aircraft_id"):
    if g["corrosion_risk"].iloc[1] > g["corrosion_risk"].iloc[0]:
        later_higher += 1
n_pairs = eval_rows["aircraft_id"].nunique()

print(f"\nSur les {n_pairs} paires évaluées :")
print(f"  le modèle score la date récente plus haut pour {later_higher}/{n_pairs} paires")
print("Distribution sur les 164 lignes évaluées :")
print(eval_rows["corrosion_risk"].describe().round(4).to_string())
