import pandas as pd
import os

data_dir = "data/cleaned"

# Only inspect Recorded Crime (main dataset for modelling)
datasets = {
    "recorded_crime": "recorded_crime_cleaned.csv",

    # The following datasets can be optionally inspected for documentation,
    # but they are NOT used for prediction or merging.
    # "detected_crime_by_sex": "detected_crime_by_sex_cleaned.csv",
    # "detection_rates": "detection_rates_cleaned.csv",
    # "victims_personal_crime": "victims_personal_crime_cleaned.csv"
}


def inspect_dataset(file_path, name):
    df = pd.read_csv(file_path)
    print(f"\n🔹 Dataset: {name}")
    print("Shape:", df.shape)

    print("\nColumns and types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSample unique values for first 3 columns:")
    for col in df.columns[:3]:
        print(f"{col}: {df[col].unique()[:10]}")

    # Additional useful previews for Recorded Crime
    if name == "recorded_crime":
        if "garda_station" in df.columns:
            print("\nExample Garda stations:", df["garda_station"].dropna().unique()[:10])
        if "offence" in df.columns:
            print("\nExample offence types:", df["offence"].dropna().unique()[:10])

    return df


def inspect_all():
    for name, filename in datasets.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            inspect_dataset(file_path, name)
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    inspect_all()
