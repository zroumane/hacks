"""
Étape 1 — CatBoost + GroupKFold + labels binaires.

Ce script :
  1. Vérifie le biais d'inspection (corrélation âge↔détection)
  2. Construit des labels binaires avec buffer configurable
  3. Valide CatBoost en GroupKFold (hold-out par avion)
  4. Entraîne le modèle final et sauvegarde dans output/models/catboost_step1.pkl

Usage : uv run src/algo/step1_catboost.py [--buffer 12]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_input
from utils.features import (
    build_training_frame,
    check_inspection_bias,
)
from utils.ml import (
    HAS_CATBOOST,
    brier_score,
    cross_validate_group,
    save_model,
    train_catboost,
)

BUFFER_MONTHS = 12   # fenêtre de label positif (ajuster si histogramme montre biais)
GREY_MONTHS   = 6    # zone grise exclue juste avant la détection (censure)

if not HAS_CATBOOST:
    print("ERREUR : catboost non installé → pip install catboost")
    sys.exit(1)


# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement des données...")
corr      = load_input("corrosions_training.csv")
env_train = load_input("environment_training.csv")

# ── 2. Diagnostic biais d'inspection ──────────────────────────────────────────
check_inspection_bias(corr)

# ── 3. Construction des labels ────────────────────────────────────────────────
print("\nConstruction des labels binaires (zone grise exclue)...")
merged, X, y, groups = build_training_frame(
    corr, env_train, buffer_months=BUFFER_MONTHS, grey_months=GREY_MONTHS,
)

pos_rate = y.mean()
print(f"  {len(merged)} lignes | positifs : {int(y.sum())} ({pos_rate:.1%}) | "
      f"buffer : {BUFFER_MONTHS} mois | zone grise : {GREY_MONTHS} mois")

# ── 4. Validation croisée GroupKFold ──────────────────────────────────────────
print("\nValidation GroupKFold (5 folds, hold-out par avion)...")
cross_validate_group(
    X, y, groups,
    model_fn=lambda Xtr, ytr, Xv, yv: train_catboost(Xtr, ytr, Xv, yv),
    n_splits=5,
)

# ── 5. Entraînement final ──────────────────────────────────────────────────────
print("\nEntraînement final CatBoost...")
model = train_catboost(X, y)

train_preds = model.predict_proba(X)[:, 1]
print(f"  Brier Score (train, optimiste — voir la CV ci-dessus) : "
      f"{brier_score(y.values, train_preds):.4f}")

save_model(model, "catboost_step1")
print("\nÉtape 1 terminée. Lancer step2_tabpfn.py ou step4_ensemble.py pour la soumission.")
