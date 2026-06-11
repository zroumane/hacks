import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dateutil.relativedelta import relativedelta
from ibm_watsonx_ai.foundation_models.schema import TSForecastParameters

from utils.io import load_input, save_output
from utils.watsonx import get_ts_model

# --- Config ---
CONTEXT_LENGTH    = 512  # requis par granite-ttm-512-96-r2
PREDICTION_LENGTH = 12

TARGET_COLUMNS = [
    "metar_temperature_c",
    "metar_relative_humidity",
    "metar_hour_precipitation",
    "total_parking_minutes",
]


def pad_to_context(df_aircraft: pd.DataFrame, context_length: int) -> pd.DataFrame:
    """Padde la série en début de fenêtre avec la première valeur connue."""
    n = len(df_aircraft)
    if n >= context_length:
        return df_aircraft.tail(context_length)

    first_date = pd.to_datetime(df_aircraft["month_start_date"].iloc[0])
    pad_rows = []
    for i in range(context_length - n, 0, -1):
        padded_date = first_date - relativedelta(months=i)
        row = df_aircraft.iloc[0].copy()
        row["month_start_date"] = padded_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        pad_rows.append(row)

    return pd.concat([pd.DataFrame(pad_rows), df_aircraft], ignore_index=True)


# --- Chargement ---
print("Chargement environment_training.csv...")
df = load_input("environment_training.csv")
df["month_start_date"] = (
    pd.to_datetime(df["month_start_date"], utc=True)
    .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
)
df = df.sort_values(["aircraft_id", "month_start_date"])

# On prend 3 avions pour le test
sample_ids = df["aircraft_id"].unique()[:3]

input_data = []
for aircraft_id in sample_ids:
    window = df[df["aircraft_id"] == aircraft_id].copy()
    window = pad_to_context(window, CONTEXT_LENGTH)
    input_data.append(window)
    print(f"  {aircraft_id} : {len(df[df['aircraft_id'] == aircraft_id])} mois réels → {CONTEXT_LENGTH} après padding")

input_df = pd.concat(input_data).reset_index(drop=True)

# --- Modèle ---
print("\nConnexion au modèle Granite TTM...")
model = get_ts_model("ibm/granite-ttm-512-96-r2")

params = TSForecastParameters(
    id_columns=["aircraft_id"],
    timestamp_column="month_start_date",
    target_columns=TARGET_COLUMNS,
    freq="1M",
    prediction_length=PREDICTION_LENGTH,
)

# --- Prévision ---
print(f"Prévision sur {PREDICTION_LENGTH} mois...")
results = model.forecast(data=input_df, params=params)

df_out = pd.DataFrame(
    results["results"][0],
    columns=["aircraft_id", "month_start_date"] + TARGET_COLUMNS,
)

print("\n--- Résultats ---")
print(df_out.to_string(index=False))

output_path = save_output(df_out, "forecast.csv")
print(f"\nRésultats sauvegardés : {output_path}")
