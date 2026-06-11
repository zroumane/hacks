"""
Génération de la soumission 100 % features — sans aucune connaissance de la
structure de l'évaluation.

Un seul modèle (xgb_background) appliqué uniformément aux 14 303 lignes
aircraft_id × year_month : chaque ligne reçoit P(corrosion détectée ce mois-ci)
calculée à partir des features environnementales uniquement (exposition au sel,
humidité, parking, SO₂, rolling 3/12/24m, cumul vie entière).

Aucune ligne n'est traitée différemment des autres : le sample_submission
n'est pas utilisé, ni la date de livraison (pas de feature d'âge).

Usage : uv run src/algo/predict_pure.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from algo.pair_features import (
    ENV_FEATURES,
    LIFE_FEATURES,
    RATIO_FEATURES,
    ROLLING_FEATURES,
    build_features,
)
from utils.io import load_input, save_output
from utils.ml import load_model

BASE_COLS = ENV_FEATURES + ROLLING_FEATURES + LIFE_FEATURES + RATIO_FEATURES

# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement du modèle et des données...")
model = load_model("xgb_pure")
env_test = load_input("environment_test.csv")

# delivery_* requis par build_features pour calculer l'âge, mais l'âge n'est
# PAS dans BASE_COLS — il n'entre pas dans le modèle.
env_test["delivery_year"] = 2014
env_test["delivery_month"] = 6

# ── 2. Features et prédiction uniforme ─────────────────────────────────────────
print("Calcul des features...")
env_feat = build_features(env_test)

print("Prédiction sur les 14 303 lignes...")
X = env_feat[BASE_COLS].fillna(0)
env_feat["corrosion_risk"] = np.clip(model.predict_proba(X)[:, 1], 0, 1)

# ── 3. Soumission ──────────────────────────────────────────────────────────────
env_feat["id"] = env_feat["aircraft_id"] + "_" + env_feat["year_month"]
submission = env_feat[["id", "corrosion_risk"]].reset_index(drop=True)

output_path = save_output(submission, "submission_pure.csv")
print(f"\nSoumission sauvegardée : {output_path}")
print("\nDistribution des prédictions :")
print(submission["corrosion_risk"].describe().round(4).to_string())
