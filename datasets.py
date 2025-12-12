from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "cleaned"


def load_recorded_crime(cleaned: bool = True) -> pd.DataFrame:
    """
    Load the Recorded Crime dataset.

    - cleaned=True  -> data/cleaned/recorded_crime_cleaned.csv
    - cleaned=False -> data/recorded_crime.csv

    Expected columns after cleaning (ideally):
        year: int
        division: str
        garda_station: str
        offence: str
        incidents: int
    """
    if cleaned:
        path = CLEAN_DIR / "recorded_crime_cleaned.csv"
    else:
        path = DATA_DIR / "recorded_crime.csv"

    df = pd.read_csv(path)

    # Ensure types are reasonable
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if "incidents" in df.columns:
        df["incidents"] = (
            pd.to_numeric(df["incidents"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return df


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
