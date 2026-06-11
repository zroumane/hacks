"""
Étape 4 — Ensemble final + soumission.

Charge tous les modèles présents dans output/models/ et les combine
par moyenne pondérée. Les modèles "step3" (features physiques) reçoivent
un poids x1.5 car ils exploitent plus d'information.

Prédiction finale : clippée dans [0.02, 0.98] pour la calibration Brier Score.

Usage : uv run src/algo/step4_ensemble.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import load_input, save_output
from utils.features import (
    FEATURE_COLS_BASE,
    FEATURE_COLS_PHYSICS,
    add_age_feature,
    add_physical_features,
    add_sea_salt_total,
)
from utils.ml import HAS_TABPFN, load_all_models, predict_proba_tabpfn

CLIP_LOW  = 0.02
CLIP_HIGH = 0.98

# Poids par préfixe de nom de modèle
WEIGHT_RULES = {
    "step3": 1.5,   # features physiques → signal plus riche
    "step2": 1.0,
    "step1": 1.0,
}


def get_weight(name: str) -> float:
    for key, w in WEIGHT_RULES.items():
        if key in name:
            return w
    return 1.0


def needs_physics(name: str) -> bool:
    return "step3" in name


# ── 1. Chargement des données de test ─────────────────────────────────────────
print("Chargement des données de test...")
env_test = load_input("environment_test.csv")
if Path("input/sample_submission.csv").exists():
    sub = load_input("sample_submission.csv")
else:
    _gt = load_input("test.csv")
    sub = _gt[_gt["corrosion_risk"] != 0.5][["id"]].copy()
    sub["corrosion_risk"] = 0.5

# Estimation de la date de livraison = premier mois disponible par avion
first_month = env_test.groupby("aircraft_id")["year_month"].min().reset_index()
first_month.columns = ["aircraft_id", "estimated_delivery"]
first_month["delivery_year"]  = pd.to_datetime(first_month["estimated_delivery"]).dt.year
first_month["delivery_month"] = pd.to_datetime(first_month["estimated_delivery"]).dt.month

env_test = env_test.merge(
    first_month[["aircraft_id", "delivery_year", "delivery_month"]],
    on="aircraft_id",
)
env_test = add_sea_salt_total(env_test)
env_test = add_age_feature(env_test, env_test["delivery_year"], env_test["delivery_month"])
env_test = env_test.sort_values(["aircraft_id", "year_month"]).reset_index(drop=True)
env_test = add_physical_features(env_test)   # nécessite tri par avion+date

X_base   = env_test[FEATURE_COLS_BASE].fillna(0)
X_physics = env_test[FEATURE_COLS_PHYSICS].fillna(0)

# ── 2. Chargement des modèles ─────────────────────────────────────────────────
models = load_all_models()
if not models:
    print("Aucun modèle trouvé dans output/models/ — lancer au moins step1_catboost.py")
    sys.exit(1)

print(f"\n{len(models)} modèle(s) chargé(s) :")
for name, _ in models:
    print(f"  • {name}  (poids : {get_weight(name)})")

# ── 3. Prédictions et ensemble ────────────────────────────────────────────────
print("\nGénération des prédictions...")
total_weight = 0.0
result       = np.zeros(len(env_test))

for name, model in models:
    w = get_weight(name)
    X = X_physics if needs_physics(name) else X_base

    is_tabpfn = HAS_TABPFN and hasattr(model, "predict_proba") and "tabpfn" in name.lower()

    if is_tabpfn:
        preds = predict_proba_tabpfn(model, X)
    elif hasattr(model, "predict_proba"):
        preds = model.predict_proba(X)[:, 1]
    else:
        preds = np.clip(model.predict(X), 0, 1)

    result       += w * preds
    total_weight += w
    print(f"  {name} : moy={preds.mean():.3f}  std={preds.std():.3f}")

result = np.clip(result / total_weight, CLIP_LOW, CLIP_HIGH)
print(f"\nEnsemble final : moy={result.mean():.3f}  std={result.std():.3f}")

# ── 4. Construction de la soumission ──────────────────────────────────────────
env_test["id"]             = env_test["aircraft_id"] + "_" + env_test["year_month"]
env_test["corrosion_risk"] = result

preds_map  = env_test.set_index("id")["corrosion_risk"]
submission = sub.copy()
submission["corrosion_risk"] = submission["id"].map(preds_map)

missing = submission["corrosion_risk"].isna().sum()
if missing:
    print(f"\n  {missing} id sans correspondance → rempli à 0.5")
    submission["corrosion_risk"] = submission["corrosion_risk"].fillna(0.5)

# ── 5. Sauvegarde ─────────────────────────────────────────────────────────────
output_path = save_output(submission, "submission_ensemble.csv")
print(f"\nSoumission sauvegardée : {output_path}")
print(submission["corrosion_risk"].describe().round(4).to_string())
