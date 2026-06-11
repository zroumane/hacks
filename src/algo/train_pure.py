"""
Entraînement du modèle pur — un seul modèle, 100 % features environnementales,
aucune connaissance de la structure de l'évaluation.

Données d'entraînement (une ligne = un mois d'un avion) :
  - POSITIFS  : le mois d'observation de la corrosion (Y=1), pour TOUS les
    avions dont l'historique couvre ce mois
  - NÉGATIFS  : des mois échantillonnés ALÉATOIREMENT parmi ceux situés à
    ≥24 mois avant l'observation. Justification physique : la corrosion
    détectée à l'observation s'est vraisemblablement initiée dans les 1-2 ans
    précédents ; au-delà de 24 mois en arrière, l'avion est supposé sain.
    Aucun écart fixe copié de la structure de l'évaluation.

Features : valeurs du mois courant + rolling 3/12/24m + cumul vie entière
+ ratios exposition récente / exposition vie entière. Pas d'âge (la date de
livraison des avions test est approximative) ni d'artefact d'historique.

Usage : uv run src/algo/train_pure.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from algo.pair_features import (
    ENV_FEATURES,
    LIFE_FEATURES,
    RATIO_FEATURES,
    ROLLING_FEATURES,
    build_features,
    make_background_negatives,
)
from utils.io import load_input
from utils.ml import brier_score, save_model

PURE_COLS = ENV_FEATURES + ROLLING_FEATURES + LIFE_FEATURES + RATIO_FEATURES

XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


def make_positives(corr: pd.DataFrame, env_feat: pd.DataFrame) -> pd.DataFrame:
    """Positifs : mois d'observation, pour tous les avions couverts par l'historique."""
    corr = corr.copy()
    corr["obs_month"] = pd.to_datetime(corr["observation_date"]).dt.to_period("M")

    env_feat = env_feat.copy()
    env_feat["period"] = env_feat["month_dt"].dt.to_period("M")
    indexed = env_feat.set_index(["aircraft_id", "period"])

    rows = []
    for _, r in corr.iterrows():
        try:
            pos = indexed.loc[(r["aircraft_id"], r["obs_month"])].copy()
        except KeyError:
            continue
        pos["target"] = 1
        pos["aircraft_id"] = r["aircraft_id"]
        rows.append(pos)

    print(f"  {len(rows)} positifs (mois d'observation)")
    return pd.DataFrame(rows).reset_index(drop=True)


# ── 1. Chargement et features ──────────────────────────────────────────────────
print("Chargement des données...")
corr = load_input("corrosions_training.csv")
env  = load_input("environment_training.csv")

deliveries = corr[["aircraft_id", "aircraft_delivery_year", "aircraft_delivery_month"]].drop_duplicates("aircraft_id")
deliveries = deliveries.rename(columns={
    "aircraft_delivery_year": "delivery_year",
    "aircraft_delivery_month": "delivery_month",
})
env = env.merge(deliveries, on="aircraft_id", how="inner")

print("Calcul des features...")
env_feat = build_features(env)

print("Construction des lignes d'entraînement...")
positives = make_positives(corr, env_feat)
# Négatifs : mois aléatoires ≥24 mois avant l'observation (zone supposée saine).
# 4 par avion pour garder un ratio positifs/négatifs comparable aux versions
# précédentes (~1:4).
negatives = make_background_negatives(corr, env_feat, n_per_aircraft=4, min_gap=24)

train_df = pd.concat([positives, negatives], ignore_index=True)
n_pos = int(train_df["target"].sum())
print(f"  Total : {len(train_df)} lignes ({n_pos} positifs, {len(train_df) - n_pos} négatifs)")

X = train_df[PURE_COLS].fillna(0)
y = train_df["target"]
groups = train_df["aircraft_id"]

# ── 2. Validation croisée groupée par avion ────────────────────────────────────
print("\nValidation croisée (5 folds, groupés par avion)...")
cv = GroupKFold(n_splits=5)
briers, pos_means, neg_means = [], [], []
for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X.iloc[tr], y.iloc[tr], verbose=False)
    p = m.predict_proba(X.iloc[va])[:, 1]
    yv = y.iloc[va].values
    bs = brier_score(yv, p)
    briers.append(bs)
    pos_means.append(p[yv == 1].mean())
    neg_means.append(p[yv == 0].mean())
    print(f"  Fold {fold + 1} — Brier : {bs:.4f} | pred(Y=1) : {p[yv==1].mean():.3f} | pred(Y=0) : {p[yv==0].mean():.3f}")
print(f"  Moyenne — Brier : {np.mean(briers):.4f} | pred(Y=1) : {np.mean(pos_means):.3f} | pred(Y=0) : {np.mean(neg_means):.3f}")

# ── 3. Entraînement final ──────────────────────────────────────────────────────
print("\nEntraînement final...")
model = XGBClassifier(**XGB_PARAMS)
model.fit(X, y, verbose=False)

importances = pd.Series(model.feature_importances_, index=PURE_COLS)
print("\nTop 15 features :")
print(importances.sort_values(ascending=False).head(15).round(4).to_string())

save_model(model, "xgb_pure")
print("\nTerminé. Lancer : uv run src/algo/predict_pure.py")
