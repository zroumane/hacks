"""
Feature engineering physique basé sur la chimie de la corrosion (ISO 9223, FAA AC 43-4B).

Toutes les fonctions sont sans état (stateless) : elles prennent un DataFrame et
retournent un DataFrame augmenté. L'ordre d'appel recommandé :
  df = add_sea_salt_total(df)
  df = add_age_feature(df, year_col, delivery_year, delivery_month)
  df = add_physical_features(df)       # nécessite cum_wet → groupby aircraft_id
"""

import numpy as np
import pandas as pd

# ── Colonnes sel marin (3 fractions granulométriques) ─────────────────────────
SEA_SALT_COLS = [
    "sea_salt_aerosol_003_05_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
]

# ── Features de base ───────────────────────────────────────────────────────────
ENV_FEATURES_BASE = [
    "total_parking_minutes",
    "metar_temperature_c", "metar_relative_humidity", "metar_dew_point_c",
    "metar_wind_speed_kn", "metar_visibility_mi", "metar_hour_precipitation",
    "sea_salt_aerosol_003_05_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
    "dust_aerosol_003_055_mixing_ratio",
    "dust_aerosol_055_09_mixing_ratio",
    "dust_aerosol_09_20_mixing_ratio",
    "sulphate_aerosol_mixing_ratio", "sulphur_dioxide_mass_mixing_ratio",
    "hno3", "ozone_mass_mixing_ratio",
    "nitrogen_monoxide_mass_mixing_ratio", "nitrogen_dioxide_mass_mixing_ratio",
    "specific_humidity", "temperature",
]

# Features physiques ajoutées par add_physical_features()
PHYSICAL_FEATURES = [
    "sea_salt_total",
    "cum_wet",       # proxy Time of Wetness cumulé (Σ mois avec RH > 80 %)
    "salt_active",   # sel marin × (RH > 75 %) — seuil de déliquescence NaCl
    "log_tow",       # log1p(cum_wet) — structure log de la dose-response ISO 9223
    "iso_cross",     # log1p(TOW) × log1p(SO₂) — terme croisé ISO 9223
]

FEATURE_COLS_BASE    = ENV_FEATURES_BASE + ["aircraft_age_months"]
FEATURE_COLS_PHYSICS = FEATURE_COLS_BASE + PHYSICAL_FEATURES


def add_sea_salt_total(df: pd.DataFrame) -> pd.DataFrame:
    """Somme des 3 fractions granulométriques de sel marin."""
    df = df.copy()
    df["sea_salt_total"] = df[SEA_SALT_COLS].fillna(0).sum(axis=1)
    return df


def add_age_feature(
    df: pd.DataFrame,
    delivery_year: pd.Series,
    delivery_month: pd.Series,
) -> pd.DataFrame:
    """Âge de l'avion en mois à chaque ligne (year_month − date de livraison)."""
    df = df.copy()
    month_dt    = pd.to_datetime(df["year_month"])
    delivery_dt = pd.to_datetime(dict(year=delivery_year, month=delivery_month, day=1))
    df["aircraft_age_months"] = (
        (month_dt.dt.year  - delivery_dt.dt.year)  * 12
        + (month_dt.dt.month - delivery_dt.dt.month)
    )
    return df


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features physiques ISO 9223 / FAA AC 43-4B.

    Pré-requis : df trié par (aircraft_id, year_month) et contenant
    metar_relative_humidity, sulphate_aerosol_mixing_ratio, sea_salt_total.
    """
    df = df.copy()

    # Time of Wetness : mois avec RH > 80 % (corrosion active)
    df["wet"]     = (df["metar_relative_humidity"] > 80).astype(int)
    df["cum_wet"] = df.groupby("aircraft_id")["wet"].cumsum()

    # Sel actif : déliquescence du NaCl commence à ~75 % RH
    df["salt_active"] = df["sea_salt_total"] * (df["metar_relative_humidity"] > 75).astype(float)

    # Termes log façon dose-response ISO 9223
    df["log_tow"]   = np.log1p(df["cum_wet"])
    df["iso_cross"] = np.log1p(df["cum_wet"]) * np.log1p(df["sulphate_aerosol_mixing_ratio"].fillna(0))

    return df.drop(columns=["wet"])


def build_binary_labels(merged: pd.DataFrame, buffer_months: int = 12) -> pd.Series:
    """
    Label binaire : 1 si la corrosion est détectée dans les `buffer_months` mois suivants.

    Le buffer de 12 mois est un compromis entre la zone grise réelle (jusqu'à 36 mois
    d'incertitude liée aux C-checks) et la précision du signal pour le modèle.
    """
    return (merged["months_until"] <= buffer_months).astype(int)


def check_inspection_bias(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie si les détections se regroupent autour des intervalles C-check
    (multiples de ~24-36 mois → signal d'inspection, pas de physique).

    Retourne un DataFrame avec la distribution des âges à la détection.
    """
    c = corr.copy()
    c["observation_date"] = pd.to_datetime(c["observation_date"])
    c["age_at_detection_months"] = (
        (c["observation_date"].dt.year  - c["aircraft_delivery_year"])  * 12
        + (c["observation_date"].dt.month - c["aircraft_delivery_month"])
    )
    stats = c["age_at_detection_months"].describe().round(1)
    print("\n── Biais d'inspection (âge avion à la détection) ─────────────────")
    print(stats.to_string())

    # Détection de pics aux multiples de 12 mois
    bins = pd.cut(c["age_at_detection_months"], bins=range(0, int(c["age_at_detection_months"].max()) + 13, 12))
    counts = c.groupby(bins, observed=True)["aircraft_id"].count()
    print("\nDistribution par tranche de 12 mois :")
    print(counts.to_string())
    print("─" * 60)

    return c[["aircraft_id", "age_at_detection_months"]]
