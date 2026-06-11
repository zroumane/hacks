import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from ibm_watsonx_ai.foundation_models.schema import TSForecastParameters

from utils.io import load_input, save_output
from utils.watsonx import get_ts_model

# --- Config ---
CONTEXT_LENGTH  = 96   # mois d'historique fournis au modèle
PREDICTION_LENGTH = 12  # mois à prédire

TARGET_COLUMNS = [
    "metar_temperature_c",
    "metar_relative_humidity",
    "metar_hour_precipitation",
    "total_parking_minutes",
]

# On prend les avions avec assez d'historique
MIN_HISTORY = CONTEXT_LENGTH

# --- Chargement ---
print("Chargement environment_training.csv...")
df = load_input("environment_training.csv")
df["month_start_date"] = pd.to_datetime(df["month_start_date"], utc=True)
df = df.sort_values(["aircraft_id", "month_start_date"])

# --- Sélection des avions avec assez d'historique ---
counts = df.groupby("aircraft_id").size()
eligible = counts[counts >= MIN_HISTORY].index.tolist()
print(f"{len(eligible)} avions avec >= {MIN_HISTORY} mois d'historique")

# On prend 3 avions pour le test
sample_ids = eligible[:3]

input_data = []
for aircraft_id in sample_ids:
    window = (
        df[df["aircraft_id"] == aircraft_id]
        .tail(CONTEXT_LENGTH)
        .copy()
    )
    input_data.append(window)

input_df = pd.concat(input_data).reset_index(drop=True)
print(f"Données préparées : {input_df.shape}")

# --- Modèle ---
print("Connexion au modèle Granite TTM...")
model = get_ts_model("ibm/granite-ttm-512-96-r2")

params = TSForecastParameters(
    id_columns=["aircraft_id"],
    timestamp_column="month_start_date",
    target_columns=TARGET_COLUMNS,
    freq="MS",
    prediction_length=PREDICTION_LENGTH,
)

# --- Prévision ---
print(f"Prévision sur {PREDICTION_LENGTH} mois...")
results = model.forecast(data=input_df, params=params)

df_out = pd.DataFrame(
    results["results"][0],
    columns=["aircraft_id", "month_start_date"] + TARGET_COLUMNS,
)

print("\n--- Résultats (premières lignes) ---")
print(df_out.head(PREDICTION_LENGTH))

# --- Sauvegarde ---
output_path = save_output(df_out.to_csv(index=False), "forecast.csv")
print(f"\nRésultats sauvegardés : {output_path}")
