"""
Génération de la soumission Kaggle à partir d'un modèle entraîné.

Ce script :
  1. Charge le modèle .pkl passé en argument
  2. Prépare les features du jeu de test (environment_test.csv)
  3. Génère les prédictions pour toute les combinaisons de aircraft id et year_month
  4. Sauvegarde le fichier de soumission dans output/

Usage : uv run src/algo/predict.py <model.pkl>
Exemple : uv run src/algo/predict.py output/models/xgb_corrosion.pkl
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import load_input, save_output
from utils.ml import load_model

# Doit correspondre exactement aux features de train.py
ENV_FEATURES = [
    "total_parking_minutes",
    "metar_temperature_c", "metar_relative_humidity", "metar_dew_point_c",
    "metar_wind_speed_kn", "metar_visibility_mi", "metar_hour_precipitation",
    "sea_salt_aerosol_003_05_mixing_ratio", "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
    "dust_aerosol_003_055_mixing_ratio", "dust_aerosol_055_09_mixing_ratio",
    "dust_aerosol_09_20_mixing_ratio",
    "hydrophilic_organic_matter_aerosol_mixing_ratio",
    "hydrophobic_organic_matter_aerosol_mixing_ratio",
    "hydrophilic_black_carbon_aerosol_mixing_ratio",
    "hydrophobic_black_carbon_aerosol_mixing_ratio",
    "sulphate_aerosol_mixing_ratio", "sulphur_dioxide_mass_mixing_ratio",
    "hno3", "ozone_mass_mixing_ratio",
    "nitrogen_monoxide_mass_mixing_ratio", "nitrogen_dioxide_mass_mixing_ratio",
    "formaldehyde", "h2o2", "oh", "organic_nitrates",
    "ethane", "c3h8", "isoprene", "carbon_monoxide_mass_mixing_ratio",
    "specific_humidity", "temperature",
]
CUMUL_FEATURES = [
    f"{col}_{window}m"
    for col in [
        "sea_salt_aerosol_05_5_mixing_ratio",
        "metar_relative_humidity",
        "total_parking_minutes",
        "sulphur_dioxide_mass_mixing_ratio",
        "hno3",
    ]
    for window in [3, 6, 12]
]
FEATURE_COLS = ENV_FEATURES + CUMUL_FEATURES + ["aircraft_age_months"]


def add_age_feature(df, delivery_year, delivery_month):
    df = df.copy()
    month_dt    = pd.to_datetime(df["year_month"])
    delivery_dt = pd.to_datetime(dict(year=delivery_year, month=delivery_month, day=1))
    df["aircraft_age_months"] = (
        (month_dt.dt.year  - delivery_dt.dt.year)  * 12
        + (month_dt.dt.month - delivery_dt.dt.month)
    )
    return df


def build_cumulative_features(df):
    df = df.copy()
    for col in [
        "sea_salt_aerosol_05_5_mixing_ratio",
        "metar_relative_humidity",
        "total_parking_minutes",
        "sulphur_dioxide_mass_mixing_ratio",
        "hno3",
    ]:
        for window in [3, 6, 12]:
            df[f"{col}_{window}m"] = (
                df.groupby("aircraft_id")[col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
    return df


# ── Argument ───────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage : uv run src/algo/predict.py <model.pkl>")
    print("Exemple : uv run src/algo/predict.py output/models/xgb_corrosion.pkl")
    sys.exit(1)

model_path = Path(sys.argv[1])
if not model_path.exists():
    print(f"Erreur : fichier introuvable : {model_path}")
    sys.exit(1)

# ── 1. Chargement du modèle et des données ─────────────────────────────────────
print(f"Chargement du modèle : {model_path.name}...")
model = load_model(model_path.stem)

print("Chargement des données de test...")
env_test = load_input("environment_test.csv")  # 14 303 lignes, 142 avions

# ── 2. Feature âge avion ───────────────────────────────────────────────────────
# Les avions de test ont tous été livrés en 2014 (juin par défaut).
# Leur date de livraison n'est pas dans les données → on applique cette constante.
env_test["delivery_year"]  = 2014
env_test["delivery_month"] = 6
env_test = env_test.sort_values(["aircraft_id", "year_month"]).reset_index(drop=True)
env_test = add_age_feature(env_test, env_test["delivery_year"], env_test["delivery_month"])
env_test = build_cumulative_features(env_test)

# ── 3. Prédictions ─────────────────────────────────────────────────────────────
print("Génération des prédictions...")

X_test = env_test[FEATURE_COLS].fillna(0)

# On clipe entre 0 et 1 car le Brier Score attend des probabilités
env_test["corrosion_risk"] = np.clip(model.predict(X_test), 0, 1)

# ── 4. Construction de la soumission ──────────────────────────────────────────
# La soumission contient tous les couples aircraft_id × year_month de environment_test
# soit 14 303 lignes. Le format de l'id est "<aircraft_id>_<year_month>".
env_test["id"] = env_test["aircraft_id"] + "_" + env_test["year_month"]
submission = env_test[["id", "corrosion_risk"]].reset_index(drop=True)

print(f"  {len(submission)} lignes dans la soumission")

# ── 5. Sauvegarde ─────────────────────────────────────────────────────────────
output_path = save_output(submission, "submission.csv")

print(f"\nSoumission sauvegardée : {output_path}")
print(submission["corrosion_risk"].describe().round(4).to_string())
