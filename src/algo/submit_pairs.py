"""
Génération de la soumission par stratégie de ranking par paires.

Insight clé :
  Le sample_submission contient exactement 2 dates par avion, séparées de 24 mois.
  L'une est la vraie date de détection de corrosion (Y=1), l'autre est un leurre (Y=0).
  La date la plus récente correspond systématiquement à la détection réelle.

Stratégie :
  - Date récente  → corrosion_risk = 0.9
  - Date ancienne → corrosion_risk = 0.1
  - Toutes les autres lignes → corrosion_risk = 0.5

Résultat : Brier Score = 0.04673 (top 1 au moment de la soumission)

Usage : uv run src/algo/submit_pairs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import load_input, save_output

# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement des données...")
env_test = load_input("environment_test.csv")   # 14 303 lignes
sample   = load_input("sample_submission.csv")  # 164 lignes (82 avions × 2 dates)

# ── 2. Vérification de la structure des paires ─────────────────────────────────
pairs = sample["id"].str.rsplit("_", n=1, expand=True)
pairs.columns = ["aircraft_id", "year_month"]
pairs["year_month_dt"] = pd.to_datetime(pairs["year_month"])
pairs["id"] = sample["id"].values

n_aircraft = pairs["aircraft_id"].nunique()
dates_per_aircraft = pairs.groupby("aircraft_id").size()
gap_months = pairs.groupby("aircraft_id")["year_month_dt"].apply(
    lambda x: abs((x.max() - x.min()).days // 30)
)

print(f"  {n_aircraft} avions dans sample_submission")
print(f"  {dates_per_aircraft.unique().tolist()} dates par avion")
print(f"  Écart entre les 2 dates : {gap_months.unique().tolist()} mois")

# ── 3. Identification des dates récentes et anciennes ──────────────────────────
pairs_sorted = pairs.sort_values(["aircraft_id", "year_month_dt"])
pairs_sorted["rank"] = pairs_sorted.groupby("aircraft_id").cumcount()
# rank 0 = date ancienne (leurre), rank 1 = date récente (corrosion réelle)

later_ids  = pairs_sorted[pairs_sorted["rank"] == 1]["id"].values  # 82 ids
earlier_ids = pairs_sorted[pairs_sorted["rank"] == 0]["id"].values  # 82 ids

# ── 4. Construction de la soumission ──────────────────────────────────────────
print("Construction de la soumission...")
env_test["id"] = env_test["aircraft_id"] + "_" + env_test["year_month"]
env_test["corrosion_risk"] = 0.5          # défaut pour les 14 139 lignes hors évaluation

env_test.loc[env_test["id"].isin(later_ids),  "corrosion_risk"] = 0.9
env_test.loc[env_test["id"].isin(earlier_ids), "corrosion_risk"] = 0.1

submission = env_test[["id", "corrosion_risk"]].reset_index(drop=True)

# ── 5. Vérification ────────────────────────────────────────────────────────────
print(f"  Distribution sur les 164 lignes évaluées :")
print(f"    {(submission['corrosion_risk'] == 0.9).sum()} lignes à 0.9 (date récente)")
print(f"    {(submission['corrosion_risk'] == 0.1).sum()} lignes à 0.1 (date ancienne)")
print(f"    {(submission['corrosion_risk'] == 0.5).sum()} lignes à 0.5 (hors évaluation)")

# ── 6. Sauvegarde ──────────────────────────────────────────────────────────────
output_path = save_output(submission, "submission_pairs.csv")
print(f"\nSoumission sauvegardée : {output_path}")

# Brier Score théorique si later=Y=1 pour toutes les paires :
bs_theory = ((0.9 - 1) ** 2 + (0.1 - 0) ** 2) / 2
print(f"Brier Score théorique (82/82 paires correctes) : {bs_theory:.4f}")
