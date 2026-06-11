"""
Entraînement du modèle hybride à deux têtes.

Constat (cf. Docs/Solution_Legit.md) :
  - avec l'âge en feature : bon ranking des paires mais le modèle prédit haut
    sur tous les mois tardifs du test (background ~0.44 → pénalité Brier)
  - sans l'âge : background propre (~0.18) mais ranking des paires dégradé

Solution : deux modèles spécialisés.

  1. MODÈLE PAIRWISE — pour comparer 2 dates d'un même avion.
     Entrée : différence de features F(date récente) − F(date ancienne).
     La différence d'âge est constante (24 mois) donc s'annule : le modèle
     est structurellement forcé d'utiliser le signal environnemental
     (exposition cumulée au sel, humidité, parking, SO₂...).
     Entraîné symétriquement : (F(obs)−F(leurre), y=1) et (F(leurre)−F(obs), y=0).

  2. MODÈLE BACKGROUND — pour estimer P(corrosion) d'un mois quelconque.
     Sans feature d'âge, avec négatifs background (mois ordinaires) :
     prédit bas par défaut, ce qui colle aux ~30 lignes hors paires de
     l'évaluation (Y=0).

Usage : uv run src/algo/train_hybrid.py
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
    ROLLING_FEATURES,
    build_features,
    make_background_negatives,
    make_pairs,
)
from utils.io import load_input
from utils.ml import brier_score, save_model

# Features purement environnementales — pas d'âge ni d'artefact d'historique
BASE_COLS = ENV_FEATURES + ROLLING_FEATURES + LIFE_FEATURES

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

print("Construction des paires et des négatifs background...")
pairs = make_pairs(corr, env_feat)
background = make_background_negatives(corr, env_feat)

# ── 2. MODÈLE PAIRWISE — différences de features ───────────────────────────────
print("\n=== Modèle pairwise ===")

pos = pairs[pairs["target"] == 1].set_index("aircraft_id")
neg = pairs[pairs["target"] == 0].set_index("aircraft_id")
common = pos.index.intersection(neg.index)

diff = pos.loc[common, BASE_COLS].fillna(0) - neg.loc[common, BASE_COLS].fillna(0)

# Entraînement symétrique : les deux orientations de chaque paire
X_pw = pd.concat([diff, -diff], ignore_index=True)
y_pw = pd.Series([1] * len(diff) + [0] * len(diff))
groups_pw = pd.Series(list(common) * 2)

cv = GroupKFold(n_splits=5)
briers, accs = [], []
for fold, (tr, va) in enumerate(cv.split(X_pw, y_pw, groups_pw)):
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X_pw.iloc[tr], y_pw.iloc[tr], verbose=False)
    p = m.predict_proba(X_pw.iloc[va])[:, 1]
    bs  = brier_score(y_pw.iloc[va].values, p)
    acc = ((p > 0.5) == y_pw.iloc[va].values).mean()
    briers.append(bs); accs.append(acc)
    print(f"  Fold {fold + 1} — Brier : {bs:.4f} | Accuracy : {acc:.1%}")
print(f"  Moyenne — Brier : {np.mean(briers):.4f} | Accuracy : {np.mean(accs):.1%}")

pairwise_model = XGBClassifier(**XGB_PARAMS)
pairwise_model.fit(X_pw, y_pw, verbose=False)
save_model(pairwise_model, "xgb_pairwise")

importances = pd.Series(pairwise_model.feature_importances_, index=BASE_COLS)
print("\nTop 10 features (différences) :")
print(importances.sort_values(ascending=False).head(10).round(4).to_string())

# ── 3. MODÈLE BACKGROUND — P(corrosion) d'un mois quelconque, sans âge ─────────
print("\n=== Modèle background ===")

train_df = pd.concat([pairs, background], ignore_index=True)
X_bg = train_df[BASE_COLS].fillna(0)
y_bg = train_df["target"]
groups_bg = train_df["aircraft_id"]

briers = []
for fold, (tr, va) in enumerate(cv.split(X_bg, y_bg, groups_bg)):
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(X_bg.iloc[tr], y_bg.iloc[tr], verbose=False)
    p = m.predict_proba(X_bg.iloc[va])[:, 1]
    bs = brier_score(y_bg.iloc[va].values, p)
    briers.append(bs)
    print(f"  Fold {fold + 1} — Brier : {bs:.4f}")
print(f"  Moyenne — Brier : {np.mean(briers):.4f}")

background_model = XGBClassifier(**XGB_PARAMS)
background_model.fit(X_bg, y_bg, verbose=False)
save_model(background_model, "xgb_background")

print("\nTerminé. Lancer : uv run src/algo/predict_hybrid.py")
