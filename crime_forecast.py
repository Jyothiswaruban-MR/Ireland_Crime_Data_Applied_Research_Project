from pathlib import Path
from math import sqrt

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
DATA_PATH = CLEAN_DIR / "recorded_crime_enriched.csv"


def load_enriched() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def make_level_dataset(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    Aggregate data at a given level (e.g. 'region' or 'division')
    and year, to get one row per level-year pair.
    """
    subset = df.dropna(subset=[level, "year"])
    grouped = (
        subset.groupby([level, "year"], as_index=False)["incidents"]
        .sum()
        .rename(columns={"incidents": "total_incidents"})
    )
    return grouped


def train_and_evaluate_level(df: pd.DataFrame, level: str, train_end_year: int = 2021) -> None:
    """
    Train a RandomForestRegressor to predict total_incidents at the given level.
    Uses data up to train_end_year for training and evaluates on subsequent years.
    Saves predictions CSV in data/cleaned.
    """
    data = make_level_dataset(df, level)

    # Restrict to 2018–2023 as per project spec
    data = data[(data["year"] >= 2018) & (data["year"] <= 2023)].copy()

    # Encode level as categorical code
    data[level] = data[level].astype("category")
    data["level_code"] = data[level].cat.codes

    # Train/test split by year
    train_mask = data["year"] <= train_end_year
    train = data[train_mask]
    test = data[~train_mask]

    if test.empty or train.empty:
        print(f"[WARN] Not enough data to train/evaluate for level '{level}'.")
        return

    X_train = train[["year", "level_code"]]
    y_train = train["total_incidents"]

    X_test = test[["year", "level_code"]]
    y_test = test["total_incidents"]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)  # no squared arg for older sklearn
    rmse = sqrt(mse)

    print(f"\n=== {level.upper()} FORECAST (test years > {train_end_year}) ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # Save predictions
    out = test.copy()
    out["prediction"] = y_pred
    out_path = CLEAN_DIR / f"forecast_{level}_2022_2023.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    df = load_enriched()
    train_and_evaluate_level(df, "region", train_end_year=2021)
    train_and_evaluate_level(df, "division", train_end_year=2021)
