"""
Entraînement du modèle XGBoost pour la prédiction du risque de corrosion.

Changements v2 vs v1 :
  - Cible binaire (0/1) au lieu du score continu 1/(1+months)
    → aligne l'entraînement avec le Brier Score réel de la compétition
  - Zone grise exclue (mois 1–11 avant observation) pour éviter les labels ambigus
  - Features cumulatives sur 3, 6, 12 mois (sel, humidité, parking, SO₂)
  - Toutes les 36 features environnementales utilisées
  - scale_pos_weight pour gérer le déséquilibre fort (1 positif pour ~80 négatifs)

Usage : uv run src/algo/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import load_input
from utils.ml import brier_score, cross_validate_ts, save_model, train_xgb

# ── Toutes les features environnementales ─────────────────────────────────────
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

# Features cumulatives calculées dans build_cumulative_features()
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

# Zone grise : les mois proches de l'observation mais pas encore l'observation.
# Le label Y=0 ici serait trompeur (l'avion était déjà en train de corroder).
GRAY_ZONE_MONTHS = 12


def add_age_feature(df, delivery_year, delivery_month):
    df = df.copy()
    month_dt    = pd.to_datetime(df["year_month"])
    delivery_dt = pd.to_datetime(dict(year=delivery_year, month=delivery_month, day=1))
    df["aircraft_age_months"] = (
        (month_dt.dt.year  - delivery_dt.dt.year)  * 12
        + (month_dt.dt.month - delivery_dt.dt.month)
    )
    return df


def build_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des rolling mean sur 3, 6, 12 mois pour les features les plus
    corrélées à la corrosion. Triée par avion + date avant appel.
    """
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


# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement des données...")
corr      = load_input("corrosions_training.csv")
env_train = load_input("environment_training.csv")

# ── 2. Merge et calcul de months_until ────────────────────────────────────────
print("Construction de la cible binaire...")

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

# Cible binaire :
#   Y=1  au mois exact de l'observation (corrosion détectée)
#   Y=0  à plus de GRAY_ZONE_MONTHS mois avant (négatifs fiables)
#   exclu : zone grise [1, GRAY_ZONE_MONTHS[ — label ambigu, on ne les entraîne pas dessus
merged["corrosion_risk"] = np.where(merged["months_until"] == 0, 1, 0)
merged = merged[(merged["months_until"] == 0) | (merged["months_until"] > GRAY_ZONE_MONTHS)].copy()

n_pos = merged["corrosion_risk"].sum()
n_neg = len(merged) - n_pos
print(f"  {n_pos:.0f} positifs / {n_neg:.0f} négatifs (ratio 1:{n_neg/n_pos:.0f})")

# ── 3. Features ────────────────────────────────────────────────────────────────
merged = merged.sort_values(["aircraft_id", "month_dt"]).reset_index(drop=True)
merged = add_age_feature(merged, merged["aircraft_delivery_year"], merged["aircraft_delivery_month"])
merged = build_cumulative_features(merged)

X = merged[FEATURE_COLS].fillna(0)
y = merged["corrosion_risk"]

# ── 4. Validation croisée ─────────────────────────────────────────────────────
print("\nValidation croisée (3 folds temporels)...")
cross_validate_ts(X, y, n_splits=3)

# ── 5. Entraînement final ──────────────────────────────────────────────────────
print("\nEntraînement final...")

# scale_pos_weight compense le déséquilibre : sans ça, le modèle prédit presque
# toujours 0 car les négatifs dominent largement
model = train_xgb(X, y, params={"scale_pos_weight": n_neg / n_pos})

train_preds = np.clip(model.predict(X), 0, 1)
print(f"  Brier Score (train) : {brier_score(y.values, train_preds):.4f}")

save_model(model, "xgb_corrosion_v2")
print("\nTerminé. Lancer : uv run src/algo/predict.py output/models/xgb_corrosion_v2.pkl")
