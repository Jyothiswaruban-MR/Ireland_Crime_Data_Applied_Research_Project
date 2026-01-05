import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"

# -----------------------------
# Distribution of Recorded Crime Types Across Ireland
# -----------------------------

df_offence = pd.read_csv(CLEAN_DIR / "agg_offence_year.csv")
df = pd.read_csv(CLEAN_DIR / "agg_offence_year.csv")

crime_by_type = (
    df.groupby("offence", as_index=False)["total_incidents"]
      .sum()
      .sort_values("total_incidents", ascending=False)
)

# Converting to millions
crime_by_type["total_incidents"] = crime_by_type["total_incidents"] / 1_000_000

plt.figure(figsize=(14, 7))  
plt.bar(crime_by_type["offence"], crime_by_type["total_incidents"])

plt.xticks(
    rotation=90,
    fontsize=9,              
    ha="center"
)

plt.xlabel("Offence Type")
plt.ylabel("Total Recorded Incidents (Millions)")
plt.title("Distribution of Recorded Crime Types Across Ireland")

plt.ticklabel_format(style="plain", axis="y")

plt.subplots_adjust(bottom=0.35)  

plt.savefig(
    "Figure_4_4_Distribution_of_Crime_Types.png",
    dpi=300,
    bbox_inches="tight")
plt.show()


# -----------------------------
# Trend Analysis of Recorded Crimes over Time
# -----------------------------

df_yearly = pd.read_csv(CLEAN_DIR / "agg_yearly.csv")

plt.figure(figsize=(8, 5))
plt.plot(df_yearly["year"], df_yearly["total_incidents"])
plt.xlabel("Year")
plt.ylabel("Total Recorded Incidents")
plt.title("Trend Analysis of Recorded Crimes over Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_4_5_Crime_Trends_Over_Time.png", dpi=300)
plt.show()

# -----------------------------
# Regional Comparison of Crime Rates
# -----------------------------

df_region = pd.read_csv(CLEAN_DIR / "agg_region_year.csv")

region_totals = (
    df_region.groupby("region", as_index=False)["total_incidents"]
    .sum()
    .sort_values("total_incidents", ascending=False)
)

region_totals["total_incidents"] = region_totals["total_incidents"] / 1_000_000

plt.figure(figsize=(9, 5))
plt.bar(region_totals["region"], region_totals["total_incidents"])
plt.xticks(rotation=45, ha="right")
plt.xlabel("Region")
plt.ylabel("Total Recorded Incidents (Millions)")
plt.title("Regional Comparison of Crime Rates")
plt.tight_layout()
plt.savefig("Figure_4_6_Regional_Comparison.png", dpi=300)
plt.show()
