from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

# -------------------------------------------------------------------
# Base paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "cleaned"

RAW_DATA_PATH = DATA_DIR / "recorded_crime.csv"

# -------------------------------------------------------------------
# CSO dataset endpoint (CSV download)
# -------------------------------------------------------------------
CSO_CSV_URL = (
    "https://ws.cso.ie/public/api.restful/"
    "PxStat.Data.Cube_API.ReadDataset/CJA07/CSV/1.0/en"
)

# -------------------------------------------------------------------
# Data acquisition
# -------------------------------------------------------------------
def download_recorded_crime(force: bool = True) -> Path:
    """
    Download the Recorded Crime dataset programmatically from the CSO Open Data API.

    Parameters
    ----------
    force : bool
        If True, forces re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the downloaded raw CSV file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and not force:
        return RAW_DATA_PATH

    response = requests.get(CSO_CSV_URL, timeout=60)
    response.raise_for_status()

    RAW_DATA_PATH.write_bytes(response.content)
    return RAW_DATA_PATH


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_recorded_crime(cleaned: bool = True) -> pd.DataFrame:
    """
    Load the Recorded Crime dataset.

    - cleaned=True  -> data/cleaned/recorded_crime_cleaned.csv
    - cleaned=False -> data/recorded_crime.csv (downloaded programmatically)

    Expected columns after cleaning (ideally):
        year: int
        division: str
        garda_station: str
        offence: str
        incidents: int
    """
    if cleaned:
        path = CLEAN_DIR / "recorded_crime_cleaned.csv"
        if not path.exists():
            raise FileNotFoundError(
                "Cleaned dataset not found. "
                "Run preprocessing before loading cleaned data."
            )
    else:
        # Ensure raw dataset exists (download only if missing)
        download_recorded_crime()
        path = RAW_DATA_PATH

    df = pd.read_csv(path)

    # Ensure data types are consistent
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if "incidents" in df.columns:
        df["incidents"] = (
            pd.to_numeric(df["incidents"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return df


# -------------------------------------------------------------------
# Train / test split
# -------------------------------------------------------------------
def get_train_test_split(
    df: pd.DataFrame,
    train_start: int = 2018,
    train_end: int = 2021,
    test_start: int = 2022,
    test_end: int = 2023,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataframe into train and test sets based on 'year'.

    Implements:
        Train: 2018–2021
        Test:  2022–2023
    """
    if "year" not in df.columns:
        raise ValueError("DataFrame must contain a 'year' column.")

    train_mask = (df["year"] >= train_start) & (df["year"] <= train_end)
    test_mask = (df["year"] >= test_start) & (df["year"] <= test_end)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    return train_df, test_df


# -------------------------------------------------------------------
# Module execution guard
# -------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "dataset.py defines data ingestion and preprocessing utilities.\n"
        "No data is downloaded unless download_recorded_crime() or\n"
        "load_recorded_crime(cleaned=False) is explicitly called."
    )
