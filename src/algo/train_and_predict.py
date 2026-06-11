import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import load_input, save_output
from utils.ml import brier_score, cross_validate_ts, load_model, save_model, train_xgb

ENV_FEATURES = [
    "total_parking_minutes",
    "metar_temperature_c", "metar_relative_humidity", "metar_dew_point_c",
    "metar_wind_speed_kn", "metar_visibility_mi", "metar_hour_precipitation",
    "sea_salt_aerosol_003_05_mixing_ratio", "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
    "dust_aerosol_003_055_mixing_ratio", "dust_aerosol_055_09_mixing_ratio",
    "dust_aerosol_09_20_mixing_ratio",
    "sulphate_aerosol_mixing_ratio", "sulphur_dioxide_mass_mixing_ratio",
    "hno3", "ozone_mass_mixing_ratio",
    "nitrogen_monoxide_mass_mixing_ratio", "nitrogen_dioxide_mass_mixing_ratio",
    "specific_humidity", "temperature",
]
FEATURE_COLS = ENV_FEATURES + ["aircraft_age_months"]


def add_age_feature(df: pd.DataFrame, delivery_year: pd.Series, delivery_month: pd.Series) -> pd.DataFrame:
    df = df.copy()
    month_dt = pd.to_datetime(df["year_month"])
    delivery_dt = pd.to_datetime(dict(year=delivery_year, month=delivery_month, day=1))
    df["aircraft_age_months"] = (
        (month_dt.dt.year - delivery_dt.dt.year) * 12
        + (month_dt.dt.month - delivery_dt.dt.month)
    )
    return df


# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("Chargement des données...")
corr      = load_input("corrosions_training.csv")
env_train = load_input("environment_training.csv")
env_test  = load_input("environment_test.csv")
sub       = load_input("sample_submission.csv")

# ── 2. Construction de la cible ────────────────────────────────────────────────
print("Construction de la cible...")

corr["observation_date"] = pd.to_datetime(corr["observation_date"])

merged = env_train.merge(
    corr[["aircraft_id", "observation_date", "aircraft_delivery_year", "aircraft_delivery_month"]],
    on="aircraft_id",
    how="inner",
)

merged["month_dt"] = pd.to_datetime(merged["year_month"])

# On garde uniquement les mois avant ou à la date d'observation
merged = merged[merged["month_dt"] <= merged["observation_date"]].copy()

# Mois restants avant corrosion → risque
merged["months_until"] = (
    (merged["observation_date"].dt.year  - merged["month_dt"].dt.year)  * 12
    + (merged["observation_date"].dt.month - merged["month_dt"].dt.month)
)
merged["corrosion_risk"] = 1 / (1 + merged["months_until"])

# Feature âge avion
merged = add_age_feature(merged, merged["aircraft_delivery_year"], merged["aircraft_delivery_month"])

print(f"  {len(merged)} lignes d'entraînement | cible : {merged['corrosion_risk'].min():.3f} – {merged['corrosion_risk'].max():.3f}")

# ── 3. Features / target ───────────────────────────────────────────────────────
merged = merged.sort_values(["aircraft_id", "month_dt"]).reset_index(drop=True)

X = merged[FEATURE_COLS].fillna(0)
y = merged["corrosion_risk"]

# ── 4. Validation croisée temporelle ──────────────────────────────────────────
print("\nValidation croisée (3 folds temporels)...")
cross_validate_ts(X, y, n_splits=3)

# ── 5. Entraînement final sur tout le training ─────────────────────────────────
print("\nEntraînement final...")
model = train_xgb(X, y)
train_preds = np.clip(model.predict(X), 0, 1)
print(f"  Brier Score (train) : {brier_score(y.values, train_preds):.4f}")

save_model(model, "xgb_corrosion")

# ── 6. Prédictions sur le test ─────────────────────────────────────────────────
print("\nPrédictions sur le jeu de test...")

# Âge : on estime la date de livraison = premier mois connu dans env_test
first_month = env_test.groupby("aircraft_id")["year_month"].min().reset_index()
first_month.columns = ["aircraft_id", "estimated_delivery"]
first_month["delivery_year"]  = pd.to_datetime(first_month["estimated_delivery"]).dt.year
first_month["delivery_month"] = pd.to_datetime(first_month["estimated_delivery"]).dt.month

env_test = env_test.merge(first_month[["aircraft_id", "delivery_year", "delivery_month"]], on="aircraft_id")
env_test = add_age_feature(env_test, env_test["delivery_year"], env_test["delivery_month"])

X_test = env_test[FEATURE_COLS].fillna(0)
env_test["corrosion_risk"] = np.clip(model.predict(X_test), 0, 1)

# ── 7. Construction de la soumission ───────────────────────────────────────────
print("Construction de la soumission...")

env_test["id"] = env_test["aircraft_id"] + "_" + env_test["year_month"]
preds = env_test[["id", "corrosion_risk"]].set_index("id")

submission = sub.copy()
submission["corrosion_risk"] = submission["id"].map(preds["corrosion_risk"])

missing = submission["corrosion_risk"].isna().sum()
if missing:
    print(f"  {missing} id sans prédiction → rempli à 0.5")
    submission["corrosion_risk"] = submission["corrosion_risk"].fillna(0.5)

output_path = save_output(submission, "submission.csv")
print(f"\nSoumission sauvegardée : {output_path}")
print(submission.describe())
