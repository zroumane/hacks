"""
Étape 2 — TabPFN v2.5 (GPU) + sous-échantillonnage des négatifs.

TabPFN a une limite dure d'environ 10 000 lignes en train. On sous-échantillonne
les négatifs avec K-Means pour conserver la diversité du jeu d'entraînement.

Ce script :
  1. Construit les mêmes labels binaires que step1
  2. Sous-échantillonne les négatifs par K-Means clustering
  3. Valide TabPFN en GroupKFold
  4. Entraîne le modèle final sur le jeu sous-échantillonné
  5. Sauvegarde dans output/models/tabpfn_step2.pkl

Usage : uv run src/algo/step2_tabpfn.py
Note  : nécessite GPU T4 (Colab) et : pip install tabpfn
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from utils.io import load_input
from utils.features import (
    FEATURE_COLS_BASE,
    add_age_feature,
    add_sea_salt_total,
    build_binary_labels,
)
from utils.ml import (
    HAS_TABPFN,
    brier_score,
    cross_validate_group,
    predict_proba_tabpfn,
    save_model,
    train_tabpfn,
)

BUFFER_MONTHS = 12
MAX_TRAIN     = 9000   # limite mémoire TabPFN (~10 000 lignes)
N_CLUSTERS    = 3000   # clusters K-Means sur les négatifs pour sous-échantillonner

if not HAS_TABPFN:
    print("ERREUR : tabpfn non installé → pip install tabpfn")
    sys.exit(1)


# ── 1. Chargement & labels (identique à step1) ────────────────────────────────
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
merged = add_sea_salt_total(merged)
merged = add_age_feature(merged, merged["aircraft_delivery_year"], merged["aircraft_delivery_month"])
merged = merged.sort_values(["aircraft_id", "month_dt"]).reset_index(drop=True)

y_full      = build_binary_labels(merged, buffer_months=BUFFER_MONTHS)
X_full      = merged[FEATURE_COLS_BASE].fillna(0)
groups_full = merged["aircraft_id"]

print(f"  {len(merged)} lignes avant sous-échantillonnage | positifs : {int(y_full.sum())}")


# ── 2. Sous-échantillonnage K-Means des négatifs ──────────────────────────────
print(f"\nSous-échantillonnage K-Means ({N_CLUSTERS} clusters sur les négatifs)...")

pos_idx = np.where(y_full == 1)[0]
neg_idx = np.where(y_full == 0)[0]

X_neg = X_full.iloc[neg_idx].values
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=3)
kmeans.fit(X_neg)

centers = kmeans.cluster_centers_
dists   = np.linalg.norm(X_neg - centers[kmeans.labels_], axis=1)

# Par cluster, on garde le point le plus proche du centre
cluster_labels  = kmeans.labels_
selected_neg    = []
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    if mask.any():
        best = neg_idx[mask][np.argmin(dists[mask])]
        selected_neg.append(best)

selected_idx = np.concatenate([pos_idx, np.array(selected_neg)])
selected_idx = np.sort(selected_idx)

X_sub      = X_full.iloc[selected_idx].reset_index(drop=True)
y_sub      = y_full.iloc[selected_idx].reset_index(drop=True)
groups_sub = groups_full.iloc[selected_idx].reset_index(drop=True)

print(f"  Après sous-échantillonnage : {len(X_sub)} lignes | positifs : {int(y_sub.sum())}")

if len(X_sub) > MAX_TRAIN:
    # Dernier filet de sécurité : tirage aléatoire stratifié
    print(f"  Encore > {MAX_TRAIN} lignes → tirage aléatoire stratifié final")
    from sklearn.utils import resample
    X_sub, y_sub, groups_sub = resample(
        X_sub, y_sub, groups_sub,
        n_samples=MAX_TRAIN,
        stratify=y_sub,
        random_state=42,
    )
    print(f"  Après tirage final : {len(X_sub)} lignes")


# ── 3. Validation croisée GroupKFold ──────────────────────────────────────────
print("\nValidation GroupKFold (5 folds, hold-out par avion)...")
cross_validate_group(
    X_sub, y_sub, groups_sub,
    model_fn=lambda Xtr, ytr, Xv, yv: train_tabpfn(Xtr, ytr),
    predict_fn=predict_proba_tabpfn,
    n_splits=5,
)

# ── 4. Entraînement final ──────────────────────────────────────────────────────
print("\nEntraînement final TabPFN...")
model = train_tabpfn(X_sub, y_sub)

train_preds = predict_proba_tabpfn(model, X_sub)
print(f"  Brier Score (train sous-échantillonné) : {brier_score(y_sub.values, train_preds):.4f}")

save_model(model, "tabpfn_step2")
print("\nÉtape 2 terminée. Lancer step3_physics.py ou step4_ensemble.py.")
