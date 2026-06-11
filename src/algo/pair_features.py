"""
Construction des features pour le problème de paires.

Partagé entre train_pairs.py et predict_pairs.py pour garantir que
l'entraînement et la prédiction utilisent exactement les mêmes features.

Pour chaque ligne (aircraft_id, year_month) on calcule :
  - les 36 valeurs environnementales du mois courant
  - des rolling means 3 / 12 / 24 mois sur les variables corrosives
  - la moyenne cumulée depuis le début de l'historique (exposition vie entière)
  - l'âge de l'avion en mois
  - le nombre de mois d'historique disponibles

Aucune donnée postérieure au mois cible n'est utilisée (pas de leakage) :
les rolling/expanding ne regardent que le passé.
"""

import numpy as np
import pandas as pd

PAIR_GAP_MONTHS = 24     # écart observation/leurre, même structure que le test
N_BACKGROUND = 3         # négatifs background par avion (8 testé : background meilleur mais positifs dégradés, net perdant)
BACKGROUND_MIN_GAP = 36  # un mois background doit être à ≥36 mois de l'observation

# Les 36 features environnementales brutes (mois courant)
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

# Variables physiquement liées à la corrosion → rolling + cumul vie entière
CORROSIVE_COLS = [
    "sea_salt_aerosol_003_05_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
    "metar_relative_humidity",
    "metar_dew_point_c",
    "metar_hour_precipitation",
    "total_parking_minutes",
    "sulphur_dioxide_mass_mixing_ratio",
    "sulphate_aerosol_mixing_ratio",
    "hno3",
    "specific_humidity",
]

ROLLING_WINDOWS = [3, 12, 24]

ROLLING_FEATURES = [
    f"{col}_roll{w}m" for col in CORROSIVE_COLS for w in ROLLING_WINDOWS
]
LIFE_FEATURES = [f"{col}_life" for col in CORROSIVE_COLS]

# Features relatives : exposition récente rapportée à la moyenne vie entière
# de l'avion. Capture "ce mois suit une période anormalement corrosive POUR CET
# AVION", indépendamment du niveau absolu de son site d'exploitation.
RATIO_FEATURES = [
    f"{col}_roll{w}m_vs_life" for col in CORROSIVE_COLS for w in [12, 24]
]

# aircraft_age_months est physiquement valide (corrosion cumulative, ISO 9224)
# et fiable : tous les avions test ont été livrés en 2014 (±6 mois d'erreur max).
# n_months_history est exclue : profondeur d'historique = artefact des données.
FEATURE_COLS = ENV_FEATURES + ROLLING_FEATURES + LIFE_FEATURES + ["aircraft_age_months"]


def build_features(env: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule toutes les features dérivées sur un dataframe environnemental
    (une ligne par aircraft_id × year_month).

    Le dataframe doit contenir aircraft_id, year_month et les colonnes
    delivery_year / delivery_month.
    """
    df = env.copy()
    df["month_dt"] = pd.to_datetime(df["year_month"])
    df = df.sort_values(["aircraft_id", "month_dt"]).reset_index(drop=True)

    grouped = df.groupby("aircraft_id")

    for col in CORROSIVE_COLS:
        for w in ROLLING_WINDOWS:
            df[f"{col}_roll{w}m"] = grouped[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
        # Exposition moyenne depuis le début de l'historique
        df[f"{col}_life"] = grouped[col].transform(
            lambda x: x.expanding(min_periods=1).mean()
        )

    for col in CORROSIVE_COLS:
        for w in [12, 24]:
            df[f"{col}_roll{w}m_vs_life"] = df[f"{col}_roll{w}m"] / (df[f"{col}_life"].abs() + 1e-12)

    df["n_months_history"] = grouped.cumcount() + 1

    df["aircraft_age_months"] = (
        (df["month_dt"].dt.year - df["delivery_year"]) * 12
        + (df["month_dt"].dt.month - df["delivery_month"])
    )

    return df


def make_pairs(corr: pd.DataFrame, env_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque avion du training, extrait les features aux 2 dates candidates :
    observation_date (Y=1) et observation_date - 24 mois (Y=0).
    Les avions dont l'historique ne couvre pas les 2 dates sont écartés.
    """
    corr = corr.copy()
    corr["obs_month"] = pd.to_datetime(corr["observation_date"]).dt.to_period("M")
    corr["decoy_month"] = corr["obs_month"] - PAIR_GAP_MONTHS

    env_feat = env_feat.copy()
    env_feat["period"] = env_feat["month_dt"].dt.to_period("M")
    indexed = env_feat.set_index(["aircraft_id", "period"])

    rows = []
    skipped = 0
    for _, r in corr.iterrows():
        try:
            pos = indexed.loc[(r["aircraft_id"], r["obs_month"])]
            neg = indexed.loc[(r["aircraft_id"], r["decoy_month"])]
        except KeyError:
            skipped += 1
            continue
        pos = pos.copy(); pos["target"] = 1
        neg = neg.copy(); neg["target"] = 0
        pos["aircraft_id"] = neg["aircraft_id"] = r["aircraft_id"]
        rows.append(pos)
        rows.append(neg)

    print(f"  {len(rows) // 2} paires construites, {skipped} avions écartés (historique incomplet)")
    return pd.DataFrame(rows).reset_index(drop=True)


def make_background_negatives(
    corr: pd.DataFrame,
    env_feat: pd.DataFrame,
    n_per_aircraft: int = N_BACKGROUND,
    min_gap: int = BACKGROUND_MIN_GAP,
) -> pd.DataFrame:
    """
    Échantillonne des mois "ordinaires" (loin de toute observation de corrosion)
    comme négatifs supplémentaires. Sans eux, le modèle n'a vu que des dates
    candidates et ne sait pas prédire bas sur un mois quelconque — or
    l'évaluation Kaggle contient ~30 lignes background en plus des 164 paires.
    """
    rng = np.random.default_rng(42)
    corr = corr.copy()
    corr["obs_month"] = pd.to_datetime(corr["observation_date"]).dt.to_period("M")
    obs_by_aircraft = corr.set_index("aircraft_id")["obs_month"]

    env_feat = env_feat.copy()
    env_feat["period"] = env_feat["month_dt"].dt.to_period("M")

    rows = []
    for aircraft_id, grp in env_feat.groupby("aircraft_id"):
        if aircraft_id not in obs_by_aircraft.index:
            continue
        obs = obs_by_aircraft.loc[aircraft_id]
        candidates = grp[(obs - grp["period"]).apply(lambda d: d.n) >= min_gap]
        if candidates.empty:
            continue
        take = min(n_per_aircraft, len(candidates))
        sampled = candidates.iloc[rng.choice(len(candidates), size=take, replace=False)]
        for _, r in sampled.iterrows():
            r = r.copy()
            r["target"] = 0
            rows.append(r)

    print(f"  {len(rows)} négatifs background ajoutés")
    return pd.DataFrame(rows).reset_index(drop=True)
