import pandas as pd
from pathlib import Path
from garda_divisions import attach_geography

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def clean_recorded_crime():
    path = DATA_DIR / "recorded_crime.csv"
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Year": "year",
        "Garda Station": "garda_station",
        "Type of Offence": "offence",
        "VALUE": "incidents"
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["incidents"] = pd.to_numeric(df["incidents"], errors="coerce").fillna(0).astype(int)

    df_final = df[["year", "garda_station", "offence", "incidents"]].copy()

    basic_out = CLEAN_DIR / "recorded_crime_cleaned.csv"
    df_final.to_csv(basic_out, index=False)
    print(f"Saved cleaned Recorded Crime to {basic_out} (rows={len(df_final)})")

    # 🔹 Enrich with geography
    enriched = attach_geography(df_final)

    # 🔹 NEW: standardise region names (IMPORTANT)
    REGION_MAP = {
        "Eastern": "Eastern Region",
        "Eastern Region": "Eastern Region",
        "Southern": "Southern Region",
        "Southern Region": "Southern Region",
        "Western": "Western Region",
        "Western Region": "Western Region",
        "Northern": "Northern Region",
        "Northern Region": "Northern Region",
        "North Western": "North Western Region",
        "North Western Region": "North Western Region",
        "Dublin Metropolitan Region": "Dublin Metropolitan Region"
    }

    enriched["region"] = enriched["region"].map(REGION_MAP).fillna(enriched["region"])

    enriched_out = CLEAN_DIR / "recorded_crime_enriched.csv"
    enriched.to_csv(enriched_out, index=False)
    print(f"Saved ENRICHED Recorded Crime to {enriched_out} (rows={len(enriched)})")

    return df_final

if __name__ == "__main__":
    clean_recorded_crime()
