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


def build_binary_labels(
    merged: pd.DataFrame,
    buffer_months: int = 12,
    grey_months: int = 6,
) -> pd.Series:
    """
    Label de corrosion avec exclusion de la zone grise (censure par intervalle).

      - 1   si months_until <= buffer_months          (corrosion détectée / imminente)
      - 0   si months_until >  buffer_months + grey    (avion jeune, sain fiable)
      - NaN dans ]buffer, buffer+grey]                 (zone grise → à exclure du train)

    Le buffer de 12 mois est un compromis entre la zone grise réelle (jusqu'à 36 mois
    d'incertitude liée aux C-checks) et la précision du signal. La bande grise évite
    la frontière nette qui injecte du bruit juste avant la détection (cf. Analyse.md).
    Mettre grey_months=0 pour retrouver l'ancien comportement binaire strict.
    """
    mu = merged["months_until"]
    y = pd.Series(np.nan, index=merged.index, dtype="float64")
    y[mu <= buffer_months] = 1.0
    y[mu > buffer_months + grey_months] = 0.0
    return y


def build_training_frame(
    corr: pd.DataFrame,
    env_train: pd.DataFrame,
    buffer_months: int = 12,
    grey_months: int = 6,
    physics: bool = False,
):
    """
    Construit le jeu d'entraînement complet à partir des deux CSV bruts.

    Étapes (identiques pour step1/2/3, factorisées ici pour éviter la dérive) :
      1. merge corrosions × environnement, filtre les mois <= date de détection
      2. calcule months_until + features (sel marin, âge, physiques si `physics`)
      3. construit les labels avec exclusion de la zone grise
      4. retire les lignes de la zone grise (label NaN)

    Retourne (merged, X, y, groups). `cum_wet` est calculé sur la timeline complète
    AVANT le retrait de la zone grise, donc le cumul reste correct.
    """
    corr = corr.copy()
    corr["observation_date"] = pd.to_datetime(corr["observation_date"])

    merged = env_train.merge(
        corr[["aircraft_id", "observation_date",
              "aircraft_delivery_year", "aircraft_delivery_month"]],
        on="aircraft_id",
        how="inner",
    )
    merged["month_dt"] = pd.to_datetime(merged["year_month"])
    merged = merged[merged["month_dt"] <= merged["observation_date"]].copy()
    merged["months_until"] = (
        (merged["observation_date"].dt.year  - merged["month_dt"].dt.year)  * 12
        + (merged["observation_date"].dt.month - merged["month_dt"].dt.month)
    )

    merged = add_sea_salt_total(merged)
    merged = add_age_feature(
        merged, merged["aircraft_delivery_year"], merged["aircraft_delivery_month"]
    )
    merged = merged.sort_values(["aircraft_id", "month_dt"]).reset_index(drop=True)
    if physics:
        merged = add_physical_features(merged)   # cum_wet : sur la timeline complète

    y = build_binary_labels(merged, buffer_months=buffer_months, grey_months=grey_months)

    keep = y.notna()
    merged = merged[keep].reset_index(drop=True)
    y = y[keep].astype(int).reset_index(drop=True)

    cols = FEATURE_COLS_PHYSICS if physics else FEATURE_COLS_BASE
    X = merged[cols].fillna(0)
    groups = merged["aircraft_id"]
    return merged, X, y, groups


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
