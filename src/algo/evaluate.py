"""
Calcul local du Brier Score à partir du ground truth (input/test.csv).

Permet de scorer n'importe quelle soumission sans passer par Kaggle.

Usage : uv run src/algo/evaluate.py <chemin_vers_submission.csv>
Exemple : uv run src/algo/evaluate.py output/20260611_160056_submission.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import load_input

if len(sys.argv) < 2:
    print("Usage : uv run src/algo/evaluate.py <submission.csv>")
    sys.exit(1)

submission_path = Path(sys.argv[1])
if not submission_path.exists():
    print(f"Erreur : fichier introuvable : {submission_path}")
    sys.exit(1)

# ── Chargement ─────────────────────────────────────────────────────────────────
gt  = load_input("test.csv")                            # 14 303 lignes, GT local
sub = pd.read_csv(submission_path)                      # soumission à évaluer

merged = gt.merge(sub, on="id", suffixes=("_gt", "_pred"))
if len(merged) == 0:
    print("Erreur : aucune ligne en commun entre GT et soumission.")
    sys.exit(1)

# ── Score global (toutes les lignes présentes dans les deux fichiers) ──────────
y_gt   = merged["corrosion_risk_gt"].values
y_pred = merged["corrosion_risk_pred"].values
bs_all = float(np.mean((y_pred - y_gt) ** 2))

# ── Score sur les 164 lignes évaluées uniquement (GT != 0.5) ──────────────────
mask_eval  = merged["corrosion_risk_gt"] != 0.5
bs_eval    = float(np.mean((y_pred[mask_eval] - y_gt[mask_eval]) ** 2))
n_eval     = int(mask_eval.sum())

# ── Analyse par paire ──────────────────────────────────────────────────────────
eval_df = merged[mask_eval].copy()
eval_df["aircraft_id"] = eval_df["id"].str.rsplit("_", n=1).str[0]

# Une paire est correcte si la prédiction et le GT sont du même côté (> ou < 0.5)
eval_df["correct"] = (
    (eval_df["corrosion_risk_gt"] > 0.5) == (eval_df["corrosion_risk_pred"] > 0.5)
)

# On regroupe par avion pour compter les paires entièrement correctes
pair_ok = eval_df.groupby("aircraft_id")["correct"].all()
n_pairs_correct = int(pair_ok.sum())
n_pairs_total   = int(pair_ok.count())

# ── Affichage ──────────────────────────────────────────────────────────────────
print(f"Soumission : {submission_path.name}")
print(f"Lignes comparées : {len(merged)}")
print()
print(f"Brier Score (global, {len(merged)} lignes)           : {bs_all:.5f}")
print(f"Brier Score (164 lignes évaluées uniquement)          : {bs_eval:.5f}")
print()
print(f"Paires correctement classées : {n_pairs_correct} / {n_pairs_total}")
print(f"  → {n_pairs_total - n_pairs_correct} paire(s) inversée(s) (exceptions)")
print()

# ── Distribution des prédictions ──────────────────────────────────────────────
print("Distribution des prédictions sur les 164 lignes évaluées :")
print(eval_df["corrosion_risk_pred"].describe().round(4).to_string())
print()

# ── Référence ──────────────────────────────────────────────────────────────────
bs_baseline_theory = ((0.9 - 0.9)**2 + (0.1 - 0.1)**2) / 2
print(f"Référence (0.9/0.1 parfait, 0 exception) : {bs_baseline_theory:.5f}")
