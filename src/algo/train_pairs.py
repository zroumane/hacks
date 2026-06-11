"""
Entraînement du modèle de paires — version légitime basée sur les features.

Principe :
  L'évaluation Kaggle compare, pour chaque avion, 2 dates espacées de 24 mois :
  le mois de détection réel (Y=1) et un leurre 24 mois avant (Y=0).

  On reproduit EXACTEMENT cette structure dans l'entraînement :
  pour chaque avion de corrosions_training.csv, on génère 2 lignes :
    - observation_date          → Y = 1
    - observation_date - 24 mois → Y = 0

  Le modèle (XGBoost, objectif logistique) apprend à distinguer ces 2 dates
  uniquement à partir des features environnementales : exposition cumulée
  au sel marin, humidité, SO₂, temps de parking, âge de l'avion...

Validation :
  GroupKFold sur aircraft_id (un avion entier est soit en train soit en
  validation, jamais coupé en deux) — on mesure le Brier Score et la
  "pair accuracy" : % de paires où la vraie date reçoit le score le plus haut.

Usage : uv run src/algo/train_pairs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from algo.pair_features import FEATURE_COLS, build_features
from utils.io import load_input
from utils.ml import brier_score, save_model

PAIR_GAP_MONTHS = 24   # même écart que dans sample_submission
N_BACKGROUND = 3       # négatifs supplémentaires par avion (mois ordinaires)
BACKGROUND_MIN_GAP = 36  # un mois background doit être à ≥36 mois de l'observation


def make_pairs(corr: pd.DataFrame, env_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque avion du training, extrait les features aux 2 dates candidates :
    observation_date (Y=1) et observation_date - 24 mois (Y=0).
    Les avions dont l'historique ne couvre pas les 2 dates sont écartés.
    """
    corr = corr.copy()
    corr["obs_month"] = pd.to_datetime(corr["observation_date"]).dt.to_period("M")
    corr["decoy_month"] = corr["obs_month"] - PAIR_GAP_MONTHS

    env_feat = env_feat.copy()
    env_feat["period"] = env_feat["month_dt"].dt.to_period("M")
    indexed = env_feat.set_index(["aircraft_id", "period"])

    rows = []
    skipped = 0
    for _, r in corr.iterrows():
        try:
            pos = indexed.loc[(r["aircraft_id"], r["obs_month"])]
            neg = indexed.loc[(r["aircraft_id"], r["decoy_month"])]
        except KeyError:
            skipped += 1
            continue
        pos = pos.copy(); pos["target"] = 1
        neg = neg.copy(); neg["target"] = 0
        pos["aircraft_id"] = neg["aircraft_id"] = r["aircraft_id"]
        rows.append(pos)
        rows.append(neg)

    print(f"  {len(rows) // 2} paires construites, {skipped} avions écartés (historique incomplet)")
    return pd.DataFrame(rows).reset_index(drop=True)


def make_background_negatives(corr: pd.DataFrame, env_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Échantillonne des mois "ordinaires" (loin de toute observation de corrosion)
    comme négatifs supplémentaires. Sans eux, le modèle n'a vu que des dates
    candidates et ne sait pas prédire bas sur un mois quelconque — or
    l'évaluation Kaggle contient ~30 lignes background en plus des 164 paires.
    """
    rng = np.random.default_rng(42)
    corr = corr.copy()
    corr["obs_month"] = pd.to_datetime(corr["observation_date"]).dt.to_period("M")
    obs_by_aircraft = corr.set_index("aircraft_id")["obs_month"]

    env_feat = env_feat.copy()
    env_feat["period"] = env_feat["month_dt"].dt.to_period("M")

    rows = []
    for aircraft_id, grp in env_feat.groupby("aircraft_id"):
        if aircraft_id not in obs_by_aircraft.index:
            continue
        obs = obs_by_aircraft.loc[aircraft_id]
        # Mois à au moins BACKGROUND_MIN_GAP mois AVANT l'observation
        candidates = grp[(obs - grp["period"]).apply(lambda d: d.n) >= BACKGROUND_MIN_GAP]
        if candidates.empty:
            continue
        take = min(N_BACKGROUND, len(candidates))
        sampled = candidates.iloc[rng.choice(len(candidates), size=take, replace=False)]
        for _, r in sampled.iterrows():
            r = r.copy()
            r["target"] = 0
            rows.append(r)

    print(f"  {len(rows)} négatifs background ajoutés")
    return pd.DataFrame(rows).reset_index(drop=True)


def pair_accuracy(df: pd.DataFrame, preds: np.ndarray) -> float:
    """
    % de paires où la date Y=1 reçoit une prédiction plus haute que la date
    leurre (obs-24m). Mesurée uniquement sur les lignes de paires (is_pair).
    """
    tmp = df[["aircraft_id", "target", "is_pair"]].copy()
    tmp["pred"] = preds
    tmp = tmp[tmp["is_pair"]]
    correct, total = 0, 0
    for _, g in tmp.groupby("aircraft_id"):
        pos = g.loc[g["target"] == 1, "pred"]
        neg = g.loc[g["target"] == 0, "pred"]
        if len(pos) and len(neg):
            total += 1
            if pos.iloc[0] > neg.iloc[0]:
                correct += 1
    return correct / total if total else float("nan")


# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement des données...")
corr = load_input("corrosions_training.csv")
env  = load_input("environment_training.csv")

# Date de livraison nécessaire au calcul de l'âge
deliveries = corr[["aircraft_id", "aircraft_delivery_year", "aircraft_delivery_month"]].drop_duplicates("aircraft_id")
deliveries = deliveries.rename(columns={
    "aircraft_delivery_year": "delivery_year",
    "aircraft_delivery_month": "delivery_month",
})
env = env.merge(deliveries, on="aircraft_id", how="inner")

# ── 2. Features dérivées (rolling, cumul vie entière, âge) ─────────────────────
print("Calcul des features (rolling 3/12/24m, exposition cumulée, âge)...")
env_feat = build_features(env)

# ── 3. Construction des paires + négatifs background ──────────────────────────
print("Construction des paires (observation vs observation - 24 mois)...")
pairs = make_pairs(corr, env_feat)
background = make_background_negatives(corr, env_feat)

pairs["is_pair"] = True
background["is_pair"] = False
train_df = pd.concat([pairs, background], ignore_index=True)
print(f"  Total : {len(train_df)} lignes ({train_df['target'].sum():.0f} positifs)")

X = train_df[FEATURE_COLS].fillna(0)
y = train_df["target"]
groups = train_df["aircraft_id"]

# ── 4. Validation croisée groupée par avion ────────────────────────────────────
print("\nValidation croisée (5 folds, groupés par avion)...")

xgb_params = dict(
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

cv = GroupKFold(n_splits=5)
briers, accs = [], []
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
    model = XGBClassifier(**xgb_params)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
    preds = model.predict_proba(X.iloc[va_idx])[:, 1]

    bs  = brier_score(y.iloc[va_idx].values, preds)
    acc = pair_accuracy(train_df.iloc[va_idx], preds)
    briers.append(bs)
    accs.append(acc)
    print(f"  Fold {fold + 1} — Brier : {bs:.4f} | Pair accuracy : {acc:.1%}")

print(f"  Moyenne — Brier : {np.mean(briers):.4f} ± {np.std(briers):.4f} | "
      f"Pair accuracy : {np.mean(accs):.1%}")

# ── 5. Entraînement final sur toutes les paires ────────────────────────────────
print("\nEntraînement final sur toutes les paires...")
model = XGBClassifier(**xgb_params)
model.fit(X, y, verbose=False)

# Top features pour vérifier que le modèle s'appuie bien sur l'environnement
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
print("\nTop 15 features :")
print(importances.sort_values(ascending=False).head(15).round(4).to_string())

save_model(model, "xgb_pairs")
print("\nTerminé. Lancer : uv run src/algo/predict_pairs.py")
