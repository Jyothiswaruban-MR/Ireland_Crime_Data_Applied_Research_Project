from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"


def load_enriched() -> pd.DataFrame:
    path = CLEAN_DIR / "recorded_crime_enriched.csv"
    df = pd.read_csv(path)
    return df


def agg_yearly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("year", as_index=False)["incidents"]
        .sum()
        .rename(columns={"incidents": "total_incidents"})
    )


def agg_region_year(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["region", "year"], as_index=False)["incidents"]
        .sum()
        .rename(columns={"incidents": "total_incidents"})
    )


def agg_division_year(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["division", "year"], as_index=False)["incidents"]
        .sum()
        .rename(columns={"incidents": "total_incidents"})
    )


def agg_station_year(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["station_name", "division", "year"], as_index=False)["incidents"]
        .sum()
        .rename(columns={"incidents": "total_incidents"})
    )


def agg_offence_year(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["offence", "year"], as_index=False)["incidents"]
        .sum()
        .rename(columns={"incidents": "total_incidents"})
    )


def save_all() -> None:
    df = load_enriched()

    outputs = {
        "agg_yearly.csv": agg_yearly(df),
        "agg_region_year.csv": agg_region_year(df),
        "agg_division_year.csv": agg_division_year(df),
        "agg_station_year.csv": agg_station_year(df),
        "agg_offence_year.csv": agg_offence_year(df),
    }

    for name, data in outputs.items():
        out_path = CLEAN_DIR / name
        data.to_csv(out_path, index=False)
        print(f"Saved {name} -> {out_path} (rows={len(data)})")


if __name__ == "__main__":
    save_all()
