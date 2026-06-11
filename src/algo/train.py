"""
Entraînement du modèle XGBoost pour la prédiction du risque de corrosion.

Ce script :
  1. Construit la variable cible corrosion_risk depuis corrosions_training.csv
  2. Prépare les features (environnement + âge avion)
  3. Valide le modèle par validation croisée temporelle
  4. Entraîne le modèle final sur toutes les données
  5. Sauvegarde le modèle dans output/models/

Usage : uv run src/algo/train_and_predict.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import load_input, save_output
from utils.ml import brier_score, cross_validate_ts, save_model, train_xgb

# ── Features environnementales retenues ───────────────────────────────────────
# On exclut les colonnes id/date et les gaz traceurs peu corrélés à la corrosion.
ENV_FEATURES = [
    # Temps de stationnement : proxy du ratio sol/vol (plus l'avion est au sol,
    # plus il est exposé aux éléments corrosifs)
    "total_parking_minutes",

    # Météo locale (source METAR) : humidité et température accélèrent
    # les réactions électrochimiques de corrosion
    "metar_temperature_c", "metar_relative_humidity", "metar_dew_point_c",
    "metar_wind_speed_kn", "metar_visibility_mi", "metar_hour_precipitation",

    # Aérosols marins : le sel est le principal vecteur de corrosion
    # sur les avions opérant près des côtes
    "sea_salt_aerosol_003_05_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",

    # Aérosols de poussière : particules abrasives qui fragilisent les revêtements
    "dust_aerosol_003_055_mixing_ratio",
    "dust_aerosol_055_09_mixing_ratio",
    "dust_aerosol_09_20_mixing_ratio",

    # Composés soufrés et acides : attaquent chimiquement les alliages d'aluminium
    "sulphate_aerosol_mixing_ratio", "sulphur_dioxide_mass_mixing_ratio",
    "hno3",  # acide nitrique

    # Oxydants atmosphériques : accélèrent l'oxydation des métaux
    "ozone_mass_mixing_ratio",
    "nitrogen_monoxide_mass_mixing_ratio", "nitrogen_dioxide_mass_mixing_ratio",

    # Variables atmosphériques générales
    "specific_humidity", "temperature",
]

# On ajoute l'âge de l'avion calculé dynamiquement (voir add_age_feature)
FEATURE_COLS = ENV_FEATURES + ["aircraft_age_months"]


def add_age_feature(
    df: pd.DataFrame,
    delivery_year: pd.Series,
    delivery_month: pd.Series,
) -> pd.DataFrame:
    """
    Calcule l'âge de l'avion en mois à chaque ligne.

    L'âge est la différence entre year_month (la ligne courante) et la date
    de livraison de l'avion. Plus un avion est vieux, plus il a accumulé
    d'exposition environnementale.
    """
    df = df.copy()
    month_dt    = pd.to_datetime(df["year_month"])
    delivery_dt = pd.to_datetime(dict(year=delivery_year, month=delivery_month, day=1))
    df["aircraft_age_months"] = (
        (month_dt.dt.year  - delivery_dt.dt.year)  * 12
        + (month_dt.dt.month - delivery_dt.dt.month)
    )
    return df


# ── 1. Chargement des données ──────────────────────────────────────────────────
print("Chargement des données...")
corr      = load_input("corrosions_training.csv")   # 790 avions avec date de corrosion
env_train = load_input("environment_training.csv")  # 63 524 lignes, 36 features mensuelles

# ── 2. Construction de la variable cible ──────────────────────────────────────
print("Construction de la cible...")

corr["observation_date"] = pd.to_datetime(corr["observation_date"])

# On joint les données environnementales avec les dates de corrosion.
# Chaque avion dans env_train a une date de corrosion connue dans corr.
merged = env_train.merge(
    corr[["aircraft_id", "observation_date", "aircraft_delivery_year", "aircraft_delivery_month"]],
    on="aircraft_id",
    how="inner",
)

merged["month_dt"] = pd.to_datetime(merged["year_month"])

# On ne conserve que les mois AVANT ou ÉGAL à la date de corrosion.
# Utiliser des données postérieures serait du data leakage : le modèle
# verrait des informations qu'il n'aurait pas en production.
merged = merged[merged["month_dt"] <= merged["observation_date"]].copy()

# Nombre de mois restants avant la corrosion observée.
# Ex : observation en 2022-06, ligne en 2022-03 → months_until = 3
merged["months_until"] = (
    (merged["observation_date"].dt.year  - merged["month_dt"].dt.year)  * 12
    + (merged["observation_date"].dt.month - merged["month_dt"].dt.month)
)

# Transformation en score de risque : 1.0 le mois de la corrosion,
# décroissant à mesure qu'on s'éloigne dans le passé.
# Formule : risk = 1 / (1 + months_until)
#   → months_until=0  : risk=1.000
#   → months_until=1  : risk=0.500
#   → months_until=11 : risk=0.083
merged["corrosion_risk"] = 1 / (1 + merged["months_until"])

# Ajout de l'âge de l'avion comme feature (date de livraison connue en training)
merged = add_age_feature(
    merged,
    merged["aircraft_delivery_year"],
    merged["aircraft_delivery_month"],
)

print(f"  {len(merged)} lignes | cible : {merged['corrosion_risk'].min():.3f} – {merged['corrosion_risk'].max():.3f}")

# ── 3. Préparation des features ────────────────────────────────────────────────
# On trie par avion puis par date pour respecter l'ordre temporel lors
# de la validation croisée.
merged = merged.sort_values(["aircraft_id", "month_dt"]).reset_index(drop=True)

X = merged[FEATURE_COLS].fillna(0)  # les NaN sont des mesures manquantes → 0
y = merged["corrosion_risk"]

# ── 4. Validation croisée temporelle ──────────────────────────────────────────
# TimeSeriesSplit garantit qu'on ne prédit jamais avec des données futures :
# chaque fold entraîne sur le passé et valide sur la période suivante.
print("\nValidation croisée (3 folds temporels)...")
cross_validate_ts(X, y, n_splits=3)

# ── 5. Entraînement final ──────────────────────────────────────────────────────
# On entraîne sur l'intégralité des données après avoir validé la stratégie.
print("\nEntraînement final...")
model = train_xgb(X, y)

train_preds = np.clip(model.predict(X), 0, 1)
print(f"  Brier Score (train) : {brier_score(y.values, train_preds):.4f}")

# Le modèle est sérialisé dans output/models/xgb_corrosion.pkl
save_model(model, "xgb_corrosion")
print("\nTerminé. Lancer predict.py pour générer la soumission.")
