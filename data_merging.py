from __future__ import annotations
from pathlib import Path
import pandas as pd
from datasets import load_recorded_crime

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def build_minimal_analytical_dataset() -> pd.DataFrame:
    """
    Recorded Crime is already clean; just sort and save a modelling-ready dataset.
    """

    df = load_recorded_crime(cleaned=True).copy()

    # Final standard columns expected down the pipeline:
    #   year, garda_station, offence, incidents

    df = df.sort_values(["year", "garda_station", "offence"]).reset_index(drop=True)

    return df

def save_minimal_analytical_dataset(filename="final_analytical_crime_data.csv"):
    df = build_minimal_analytical_dataset()
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {filename} (rows={len(df)})")

if __name__ == "__main__":
    save_minimal_analytical_dataset()
