from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datasets import load_recorded_crime

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

def load_model_and_encoder():
    model = joblib.load(MODEL_DIR / "crime_model.joblib")
    enc = joblib.load(MODEL_DIR / "crime_encoder.joblib")
    return model, enc

def forecast_future(station, offence, start_year=2024, horizon=3):
    model, enc = load_model_and_encoder()

    years = list(range(start_year, start_year + horizon))

    df_features = pd.DataFrame({
        "year": years,
        "garda_station": station,
        "offence": offence
    })

    X_cat = enc.transform(df_features[["garda_station", "offence"]])
    X = np.hstack([df_features[["year"]].values, X_cat])

    df_features["predicted_incidents"] = model.predict(X)
    return df_features

if __name__ == "__main__":
    # Example usage:
    df = load_recorded_crime(cleaned=True)
    station = df["garda_station"].iloc[0]
    offence = df["offence"].iloc[0]

    print(f"Forecasting for {station} - {offence}")
    result = forecast_future(station, offence)
    print(result)
