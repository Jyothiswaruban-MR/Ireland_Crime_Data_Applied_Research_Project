import pandas as pd
import re

df = pd.read_csv("station_list.txt", header=None, names=["raw"])

def parse_station(row):
    text = row["raw"]

    # Extract station code (first 5 digits)
    station_code = re.match(r"(\d{5})", text)
    station_code = station_code.group(1) if station_code else None

    # Remove station code
    cleaned = re.sub(r"^\d{5}\s*", "", text)

    # Split into station name and division
    if "," in cleaned:
        station_name, division_part = cleaned.split(",", 1)
    else:
        station_name = cleaned
        division_part = ""

    station_name = station_name.strip()
    division = division_part.replace("Division", "").strip()

    return pd.Series([station_code, station_name, division])

df[["station_code", "station_name", "division"]] = df.apply(parse_station, axis=1)

df.to_csv("station_division_mapping.csv", index=False)
print("Saved station_division_mapping.csv with", len(df), "rows")
