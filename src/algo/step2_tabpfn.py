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

from utils.io import load_input
from utils.features import build_training_frame
from utils.ml import (
    HAS_TABPFN,
    brier_score,
    cross_validate_group,
    predict_proba_tabpfn,
    save_model,
    subsample_negatives_kmeans,
    train_tabpfn,
)

BUFFER_MONTHS = 12
GREY_MONTHS   = 6      # zone grise exclue juste avant la détection (censure)
MAX_TRAIN     = 9000   # limite mémoire TabPFN (~10 000 lignes)
N_CLUSTERS    = 3000   # clusters K-Means sur les négatifs pour sous-échantillonner

if not HAS_TABPFN:
    print("ERREUR : tabpfn non installé → pip install tabpfn")
    sys.exit(1)


# ── 1. Chargement & labels (identique à step1) ────────────────────────────────
print("Chargement des données...")
corr      = load_input("corrosions_training.csv")
env_train = load_input("environment_training.csv")

_, X_full, y_full, groups_full = build_training_frame(
    corr, env_train, buffer_months=BUFFER_MONTHS, grey_months=GREY_MONTHS,
)

print(f"  {len(X_full)} lignes avant sous-échantillonnage | positifs : {int(y_full.sum())}")


# ── 2. Sous-échantillonnage K-Means des négatifs (features standardisées) ──────
print(f"\nSous-échantillonnage K-Means ({N_CLUSTERS} clusters sur les négatifs)...")

selected_idx = subsample_negatives_kmeans(X_full, y_full, n_clusters=N_CLUSTERS)

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
