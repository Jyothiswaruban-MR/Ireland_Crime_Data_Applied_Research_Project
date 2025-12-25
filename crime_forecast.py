from pathlib import Path
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
DATA_PATH = CLEAN_DIR / "recorded_crime_enriched.csv"


def load_enriched() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


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


def plot_forecast(plot_df: pd.DataFrame, level: str) -> None:
    """
    Show Actual vs Predicted line plots for ALL levels.
    Predictions only exist for test years (2022–2023).
    """
    levels = sorted(plot_df[level].dropna().unique())

    print(f"Plotting {len(levels)} {level}(s)...")

    for lvl in levels:
        sub = plot_df[plot_df[level] == lvl].sort_values("year")

        plt.figure(figsize=(8, 4))
        plt.plot(sub["year"], sub["total_incidents"], marker="o", label="Actual")

        # Plot predictions only where available
        pred = pd.to_numeric(sub["prediction"], errors="coerce")
        mask = pred.notna()
        if mask.any():
            plt.plot(sub.loc[mask, "year"], pred.loc[mask], marker="o", label="Predicted")

        plt.title(f"{level.title()} Forecast: {lvl} (2018–2023)")
        plt.xlabel("Year")
        plt.ylabel("Total Incidents")
        plt.xticks(sorted(sub["year"].unique()))
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()  # blocking (close window to move to next)


def train_and_evaluate_level(
    df: pd.DataFrame,
    level: str,
    train_end_year: int = 2021,
    show_plots: bool = True,
) -> None:
    """
    Train RandomForestRegressor and optionally show plots for ALL levels.
    """
    data = make_level_dataset(df, level)

    # Restrict to 2018–2023
    data = data[(data["year"] >= 2018) & (data["year"] <= 2023)].copy()

    # Encode categorical level
    data[level] = data[level].astype("category")
    data["level_code"] = data[level].cat.codes

    # Train/test split
    train_mask = data["year"] <= train_end_year
    train = data[train_mask].copy()
    test = data[~train_mask].copy()

    if train.empty or test.empty:
        print(f"[WARN] Not enough data for level '{level}'.")
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
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n=== {level.upper()} FORECAST (test years > {train_end_year}) ===")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # Save test predictions
    out = test.copy()
    out["prediction"] = y_pred
    out_path = CLEAN_DIR / f"forecast_{level}_2022_2023.csv"
    out.to_csv(out_path, index=False)

    # Save full timeline for plotting
    plot_df = data.copy()
    plot_df["prediction"] = float("nan")
    plot_df.loc[~train_mask, "prediction"] = y_pred

    plot_out_path = CLEAN_DIR / f"forecast_{level}_plot_2018_2023.csv"
    plot_df.to_csv(plot_out_path, index=False)

    print(f"Saved forecast files for level '{level}'.")

    # Show ALL plots
    if show_plots:
        plot_forecast(plot_df, level)


def run_all(train_end_year: int = 2021, show_plots: bool = True) -> None:
    df = load_enriched()
    train_and_evaluate_level(df, "region", train_end_year, show_plots)
    train_and_evaluate_level(df, "division", train_end_year, show_plots)


if __name__ == "__main__":
    run_all(train_end_year=2021, show_plots=True)
