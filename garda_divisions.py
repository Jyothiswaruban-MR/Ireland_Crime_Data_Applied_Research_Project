from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MAP_PATH = BASE_DIR / "station_division_mapping.csv"
DIV_CLEAN_PATH = BASE_DIR / "division_clean_map.csv"
DIV_REGION_PATH = BASE_DIR / "division_region_mapping.csv"


def load_division_clean_map() -> pd.DataFrame:
    print("\n[DEBUG] Loading division_clean_map.csv")
    print("→ File exists:", DIV_CLEAN_PATH.exists())
    print("→ Path:", DIV_CLEAN_PATH)

    df = pd.read_csv(DIV_CLEAN_PATH)
    print("→ Loaded rows:", len(df))

    df["raw"] = df["raw"].astype(str).str.strip()
    df["clean"] = df["clean"].astype(str).str.strip()
    return df


def load_division_region_map() -> pd.DataFrame:
    print("\n[DEBUG] Loading division_region_mapping.csv")
    print("→ File exists:", DIV_REGION_PATH.exists())
    print("→ Path:", DIV_REGION_PATH)

    df = pd.read_csv(DIV_REGION_PATH)
    print("→ Loaded rows:", len(df))

    df["division"] = df["division"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    return df


def load_station_mapping() -> pd.DataFrame:
    print("\n[DEBUG] Loading station_division_mapping.csv")
    print("→ File exists:", MAP_PATH.exists())
    print("→ Path:", MAP_PATH)

    df = pd.read_csv(MAP_PATH)
    print("→ Loaded rows:", len(df))

    # Dropping header junk if present
    df = df[df["raw"] != "garda_station"].copy()
    print("→ After removing header-like row:", len(df))

    # Standardizing columns
    df["station_code"] = df["station_code"].astype(str).str.replace(".0", "", regex=False)
    df["station_name"] = df["station_name"].astype(str).str.strip()
    df["division"] = df["division"].astype(str).str.strip()

   
    clean_map = load_division_clean_map()

    df = df.merge(clean_map, left_on="division", right_on="raw", how="left")
    print("→ After merging clean_map:", len(df))

    df["division"] = df["clean"].fillna(df["division"])  # Normalize divisions
    df.drop(columns=["raw_y", "clean"], inplace=True, errors="ignore")
    df = df.rename(columns={"raw_x": "raw"})

    df["division"] = df["division"].astype(str).str.strip()

    div_region = load_division_region_map()

    df = df.merge(div_region, on="division", how="left")
    print("→ After merging region map:", len(df))

    missing_regions = df["region"].isna().sum()
    print("→ Missing region count:", missing_regions)

    df["region"] = df["region"].fillna("Unknown")

    # Temporary county = division
    df["county"] = df["division"]

    return df


def attach_geography(crime_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[DEBUG] Attaching geography columns…")

    mapping = load_station_mapping()

    merged = crime_df.merge(
        mapping[["raw", "station_code", "station_name", "division", "region", "county"]],
        left_on="garda_station",
        right_on="raw",
        how="left",
    )

    print("→ Final merged rows:", len(merged))
    missing_merge = merged["station_code"].isna().sum()
    print("→ Missing station_code after merge:", missing_merge)

    merged.drop(columns=["raw"], inplace=True)
    return merged
