import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# =========================
# Paths & basic config
# =========================
BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"

REC_ENRICHED = CLEAN_DIR / "recorded_crime_enriched.csv"
DET_SEX_PATH = CLEAN_DIR / "detected_crime_by_sex_cleaned.csv"
DET_RATE_PATH = CLEAN_DIR / "detection_rates_cleaned.csv"
FORECAST_REGION_PATH = CLEAN_DIR / "forecast_region_2022_2023.csv"
FORECAST_DIV_PATH = CLEAN_DIR / "forecast_division_2022_2023.csv"

AGG_REGION_YEAR = CLEAN_DIR / "agg_region_year.csv"
AGG_DIV_YEAR = CLEAN_DIR / "agg_division_year.csv"
AGG_STATION_YEAR = CLEAN_DIR / "agg_station_year.csv"
AGG_OFFENCE_YEAR = CLEAN_DIR / "agg_offence_year.csv"
AGG_YEARLY = CLEAN_DIR / "agg_yearly.csv"

st.set_page_config(
    page_title="Ireland Crime Dashboard",
    layout="wide"
)
alt.themes.enable("dark")


# =========================
# Loaders
# =========================
@st.cache_data
def load_recorded() -> pd.DataFrame:
    df = pd.read_csv(REC_ENRICHED)
    # Expected columns: year, garda_station, offence, incidents, division, region, county, etc.
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["incidents"] = pd.to_numeric(df["incidents"], errors="coerce").fillna(0).astype(int)

    # --- Normalise region names so we don't have "Southern" vs "Southern Region" confusion ---
    if "region" in df.columns:
        region_map = {
            "Southern": "Southern Region",
            "Eastern": "Eastern Region",
            "Western": "Western Region",
            "Northern": "Northern Region",
            "North Western": "North Western Region",
            "Dublin Metropolitan": "Dublin Metropolitan Region",
        }
        df["region"] = df["region"].replace(region_map)

    df = df.dropna(subset=["year"])
    return df


@st.cache_data
def load_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


rec_df = load_recorded()
det_sex_df = load_optional(DET_SEX_PATH)
det_rate_df = load_optional(DET_RATE_PATH)
forecast_region_df = load_optional(FORECAST_REGION_PATH)
forecast_div_df = load_optional(FORECAST_DIV_PATH)

agg_region_df = load_optional(AGG_REGION_YEAR)
agg_div_df = load_optional(AGG_DIV_YEAR)
agg_station_df = load_optional(AGG_STATION_YEAR)
agg_offence_df = load_optional(AGG_OFFENCE_YEAR)
agg_yearly_df = load_optional(AGG_YEARLY)

min_year = int(rec_df["year"].min())
max_year = int(rec_df["year"].max())


# =========================
# Helper: Apply Global Filters
# =========================
def apply_filters(
    df: pd.DataFrame,
    year_range,
    region_sel: str,
    division_sel: str,
    offence_sel: str
) -> pd.DataFrame:
    f = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])].copy()

    if region_sel != "All regions" and "region" in f.columns:
        f = f[f["region"] == region_sel]

    if division_sel != "All divisions" and "division" in f.columns:
        f = f[f["division"] == division_sel]

    if offence_sel != "All offences":
        f = f[f["offence"] == offence_sel]

    return f


# =========================
# Helper: Insight generator
# =========================
def generate_insights(
    df_full: pd.DataFrame,
    year: int,
    offence: str | None,
    region: str | None,
    division: str | None,
) -> list[str]:

    insights: list[str] = []

    # Extract selected & previous year data
    cur = df_full[df_full["year"] == year].copy()
    prev = df_full[df_full["year"] == (year - 1)].copy()

    # Apply same filtering to both frames
    if region and region != "All regions":
        cur = cur[cur["region"] == region]
        prev = prev[prev["region"] == region]

    if division and division != "All divisions":
        cur = cur[cur["division"] == division]
        prev = prev[prev["division"] == division]

    if offence and offence != "All offences":
        cur = cur[cur["offence"] == offence]
        prev = prev[prev["offence"] == offence]

    # If nothing found
    if cur.empty:
        insights.append("No incidents found for this selection. Try changing filters.")
        return insights

    # =======================
    # INSIGHT 1 — Total YoY
    # =======================
    total_now = int(cur["incidents"].sum())
    total_prev = int(prev["incidents"].sum()) if not prev.empty else 0

    if total_prev > 0:
        diff = total_now - total_prev
        pct = diff / total_prev * 100
        direction = "increase" if diff > 0 else "decrease" if diff < 0 else "no change"
        scope = "Ireland"

        if division and division != "All divisions":
            scope = f"{division} division"
        elif region and region != "All regions":
            scope = f"{region} region"

        insights.append(
            f"In {year}, **{scope}** recorded **{total_now:,} incidents**, "
            f"a **{abs(pct):.1f}% {direction}** compared with {year - 1}."
        )
    else:
        insights.append(
            f"In {year}, a total of **{total_now:,} incidents** were recorded in the selected scope."
        )

    # =======================
    # INSIGHT 2 — Top Offence
    # =======================
    off_group = (
        cur.groupby("offence", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
    )

    if not off_group.empty:
        top = off_group.iloc[0]
        share = top["incidents"] / total_now * 100
        insights.append(
            f"The most common offence type was **{top['offence']}**, representing "
            f"**{share:.1f}%** of all incidents in {year}."
        )

    # ===========================
    # INSIGHT 3 — Simple ranking
    # ===========================
    # Keep insights compact: one extra comparison only

    # CASE 1 → User selected a region → compare divisions inside that region
    if region and region != "All regions":
        by_div = (
            cur.groupby("division", as_index=False)["incidents"]
            .sum()
            .sort_values("incidents", ascending=False)
        )
        if len(by_div) >= 2:
            top1 = by_div.iloc[0]
            top2 = by_div.iloc[1]
            diff = int(top1["incidents"] - top2["incidents"])
            insights.append(
                f"Within **{region}**, **{top1['division']} division** recorded the highest incidents "
                f"({top1['incidents']:,}), which is **{diff:,}** more than {top2['division']}."
            )

    # CASE 2 → User selected a division → compare offences inside that division
    elif division and division != "All divisions":
        by_off = (
            cur.groupby("offence", as_index=False)["incidents"]
            .sum()
            .sort_values("incidents", ascending=False)
        )
        if len(by_off) >= 2:
            o1 = by_off.iloc[0]
            o2 = by_off.iloc[1]
            diff = int(o1["incidents"] - o2["incidents"])
            insights.append(
                f"In **{division} division**, **{o1['offence']}** is the most frequent offence "
                f"({o1['incidents']:,}), occurring **{diff:,}** more times than {o2['offence']}."
            )

    # CASE 3 → National view → compare regions
    else:
        by_reg = (
            cur.groupby("region", as_index=False)["incidents"]
            .sum()
            .sort_values("incidents", ascending=False)
        )
        if len(by_reg) >= 2:
            r1 = by_reg.iloc[0]
            r2 = by_reg.iloc[1]
            diff = int(r1["incidents"] - r2["incidents"])
            insights.append(
                f"Nationally, **{r1['region']}** recorded the highest incidents "
                f"({r1['incidents']:,}), **{diff:,}** more than {r2['region']}."
            )

    return insights


# =========================
# Sidebar: Global controls
# =========================
st.sidebar.title("Ireland Crime Dashboard")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Recorded Crime Explorer",
        "Region & Division Insights",
        "Offence Analysis",
        "Forecasting Results",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Global Filters")

year_range = st.sidebar.slider(
    "Year range",
    min_year,
    max_year,
    (max_year - 5, max_year),
)

region_options = ["All regions"] + sorted(rec_df["region"].dropna().unique())
region_sel = st.sidebar.selectbox("Region", region_options)

# Division options depend on region
if region_sel != "All regions":
    div_options = ["All divisions"] + sorted(
        rec_df.loc[rec_df["region"] == region_sel, "division"].dropna().unique()
    )
else:
    div_options = ["All divisions"] + sorted(rec_df["division"].dropna().unique())

division_sel = st.sidebar.selectbox("Division", div_options)

offence_options = ["All offences"] + sorted(rec_df["offence"].dropna().unique())
offence_sel = st.sidebar.selectbox("Offence", offence_options)

st.sidebar.markdown("---")
st.sidebar.caption("Filters apply across most pages.")


# =========================
# Apply global filters once
# =========================
filtered_rec = apply_filters(rec_df, year_range, region_sel, division_sel, offence_sel)


# =========================
# PAGE: Overview
# =========================
if page == "Overview":
    st.markdown("## 🇮🇪 Overview")

    # KPI Row
    total_inc = int(filtered_rec["incidents"].sum())
    total_off = filtered_rec["offence"].nunique()
    total_div = filtered_rec["division"].nunique()
    total_station = filtered_rec["garda_station"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidents", f"{total_inc:,}")
    c2.metric("Offence types", total_off)
    c3.metric("Divisions", total_div)
    c4.metric("Garda stations", total_station)

    # National trend (ignores current filters; shows full Ireland)
    st.markdown("### National Trend (All Ireland)")

    if agg_yearly_df is not None and "year" in agg_yearly_df.columns:
        ydf = agg_yearly_df.copy()
        # Try to guess the incident column
        val_col = "incidents"
        for cand in ["total_incidents", "incidents", "VALUE"]:
            if cand in ydf.columns:
                val_col = cand
                break
        trend_df = ydf[["year", val_col]].rename(columns={val_col: "incidents"})
    else:
        trend_df = (
            rec_df.groupby("year", as_index=False)["incidents"]
            .sum()
        )

    chart_nat = (
        alt.Chart(trend_df)
        .mark_area(opacity=0.7)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("incidents:Q", title="Incidents"),
            tooltip=["year", "incidents"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart_nat, use_container_width=True)

    # Region comparison for last year in range, using filtered_rec
    st.markdown("### Crimes by Region (Selected Filters, Last Year in Range)")
    last_year = year_range[1]
    reg_view = filtered_rec[filtered_rec["year"] == last_year]

    if not reg_view.empty:
        reg_agg = (
            reg_view.groupby("region", as_index=False)["incidents"]
            .sum()
            .sort_values("incidents", ascending=False)
        )
        chart_reg = (
            alt.Chart(reg_agg)
            .mark_bar()
            .encode(
                x=alt.X("region:N", sort="-y", title="Region"),
                y=alt.Y("incidents:Q", title="Incidents"),
                tooltip=["region", "incidents"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_reg, use_container_width=True)
    else:
        st.info("No regional data for the selected filters and year range.")

    st.markdown("### Insight Summary (Narrative)")

    insight_year = year_range[1]
    insights = generate_insights(
        df_full=rec_df,
        year=insight_year,
        offence=offence_sel,
        region=region_sel,
        division=division_sel,
    )

    for i, text in enumerate(insights, start=1):
        st.markdown(f"**{i}.** {text}")


# =========================
# PAGE: Recorded Crime Explorer
# =========================
elif page == "Recorded Crime Explorer":
    st.markdown("##  Recorded Crime Explorer")

    st.markdown("Use the filters on the left to change year range, region, division and offence.")

    # Yearly trend for filtered scope
    trend_scope = (
        filtered_rec.groupby("year", as_index=False)["incidents"]
        .sum()
        .sort_values("year")
    )

    if not trend_scope.empty:
        chart_scope = (
            alt.Chart(trend_scope)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("incidents:Q", title="Incidents"),
                tooltip=["year", "incidents"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_scope, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

    # Station-level view
    st.markdown("### Top Garda Stations by Incidents (within filters)")

    station_agg = (
        filtered_rec.groupby("garda_station", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
        .head(20)
    )

    if not station_agg.empty:
        chart_station = (
            alt.Chart(station_agg)
            .mark_bar()
            .encode(
                x=alt.X("garda_station:N", sort="-y", title="Garda Station"),
                y=alt.Y("incidents:Q", title="Incidents"),
                tooltip=["garda_station", "incidents"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_station, use_container_width=True)
    else:
        st.info("No station-level records for current filters.")

    with st.expander("View sample underlying records"):
        st.dataframe(filtered_rec.head(200))


# =========================
# PAGE: Region & Division Insights
# =========================
elif page == "Region & Division Insights":
    st.markdown("##  Region & Division Insights")

    col1, col2 = st.columns(2)

    # Region trends
    with col1:
        st.markdown("### Region Trend (Total Incidents)")
        reg_trend = (
            rec_df.groupby(["region", "year"], as_index=False)["incidents"]
            .sum()
            .sort_values(["region", "year"])
        )
        chart_reg_trend = (
            alt.Chart(reg_trend)
            .mark_line()
            .encode(
                x="year:O",
                y="incidents:Q",
                color="region:N",
                tooltip=["region", "year", "incidents"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_reg_trend, use_container_width=True)

    # Division trends
    with col2:
        st.markdown("### Division Trend (Total Incidents)")
        div_trend = (
            rec_df.groupby(["division", "year"], as_index=False)["incidents"]
            .sum()
            .sort_values(["division", "year"])
        )
        chart_div_trend = (
            alt.Chart(div_trend)
            .mark_line()
            .encode(
                x="year:O",
                y="incidents:Q",
                color="division:N",
                tooltip=["division", "year", "incidents"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_div_trend, use_container_width=True)

    st.markdown("---")
    st.markdown("### Division Breakdown for Selected Year")

    year_for_div = st.slider("Choose year for division breakdown", min_year, max_year, max_year)
    div_year = rec_df[rec_df["year"] == year_for_div]

    if region_sel != "All regions":
        div_year = div_year[div_year["region"] == region_sel]

    div_agg = (
        div_year.groupby("division", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
    )

    if not div_agg.empty:
        chart_div_year = (
            alt.Chart(div_agg)
            .mark_bar()
            .encode(
                x=alt.X("division:N", sort="-y", title="Division"),
                y=alt.Y("incidents:Q", title="Incidents"),
                tooltip=["division", "incidents"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart_div_year, use_container_width=True)
    else:
        st.info("No division data for this year / filter combination.")


# =========================
# PAGE: Offence Analysis
# =========================
elif page == "Offence Analysis":
    st.markdown("##  Offence Analysis")

    # Offence mix in filtered data
    st.markdown("### Offence Composition (Within Filters)")

    off_mix = (
        filtered_rec.groupby("offence", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
    )

    if not off_mix.empty:
        chart_off_mix = (
            alt.Chart(off_mix.head(20))
            .mark_bar()
            .encode(
                x=alt.X("offence:N", sort="-y", title="Offence"),
                y=alt.Y("incidents:Q", title="Incidents"),
                tooltip=["offence", "incidents"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_off_mix, use_container_width=True)
    else:
        st.info("No offence data for current filters.")

    st.markdown("### Offence Trend Over Time (Ireland)")

    off_sel_trend = st.selectbox(
        "Choose an offence for trend view:",
        sorted(rec_df["offence"].unique())
    )

    off_trend = (
        rec_df[rec_df["offence"] == off_sel_trend]
        .groupby("year", as_index=False)["incidents"]
        .sum()
        .sort_values("year")
    )

    chart_off_trend = (
        alt.Chart(off_trend)
        .mark_line(point=True)
        .encode(
            x="year:O",
            y="incidents:Q",
            tooltip=["year", "incidents"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart_off_trend, use_container_width=True)



# =========================
# PAGE: Forecasting Results
# =========================
elif page == "Forecasting Results":
    st.markdown("##  Forecasting Results (Region & Division)")

    st.markdown(
        "Models were trained on historical data (e.g. 2018–2021) and evaluated on 2022–2023. "
        "Below are the predictions and, where possible, a comparison with actual incidents."
    )

    col1, col2 = st.columns(2)

    # -------------------------
    # Region forecast
    # -------------------------
    with col1:
        st.markdown("### Region-level Forecast (2022–2023)")

        if forecast_region_df is not None:
            st.dataframe(forecast_region_df.head(50))

            cols = set(forecast_region_df.columns)
            year_col = None
            reg_col = None
            true_col = None
            pred_col = None

            for c in cols:
                lc = c.lower()
                if lc == "year":
                    year_col = c
                if "region" in lc:
                    reg_col = c
                if "true" in lc or "actual" in lc:
                    true_col = c
                if "pred" in lc or "forecast" in lc:
                    pred_col = c

            # Fallback to your crime_forecast.py column names
            if true_col is None and "total_incidents" in forecast_region_df.columns:
                true_col = "total_incidents"
            if pred_col is None and "prediction" in forecast_region_df.columns:
                pred_col = "prediction"

            if year_col and reg_col and true_col and pred_col:
                sel_reg = st.selectbox(
                    "Choose region for forecast chart",
                    sorted(forecast_region_df[reg_col].unique()),
                    key="region_forecast_sel",
                )
                sub = forecast_region_df[forecast_region_df[reg_col] == sel_reg].copy()

                plot_df = sub[[year_col, true_col, pred_col]].melt(
                    id_vars=[year_col],
                    value_vars=[true_col, pred_col],
                    var_name="type",
                    value_name="incidents",
                )

                chart_f_reg = (
                    alt.Chart(plot_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(f"{year_col}:O", title="Year"),
                        y=alt.Y("incidents:Q", title="Incidents"),
                        color=alt.Color("type:N", title="Series"),
                        tooltip=[year_col, "type", "incidents"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_f_reg, use_container_width=True)
            else:
                st.info("Region forecast CSV does not have expected columns for plotting.")
        else:
            st.info("Region forecast file not found.")

    # -------------------------
    # Division forecast
    # -------------------------
    with col2:
        st.markdown("### Division-level Forecast (2022–2023)")

        if forecast_div_df is not None:
            st.dataframe(forecast_div_df.head(50))

            cols = set(forecast_div_df.columns)
            year_col = None
            div_col = None
            true_col = None
            pred_col = None

            for c in cols:
                lc = c.lower()
                if lc == "year":
                    year_col = c
                if "division" in lc:
                    div_col = c
                if "true" in lc or "actual" in lc:
                    true_col = c
                if "pred" in lc or "forecast" in lc:
                    pred_col = c

            # Fallback to crime_forecast.py column names
            if true_col is None and "total_incidents" in forecast_div_df.columns:
                true_col = "total_incidents"
            if pred_col is None and "prediction" in forecast_div_df.columns:
                pred_col = "prediction"

            if year_col and div_col and true_col and pred_col:
                sel_div = st.selectbox(
                    "Choose division for forecast chart",
                    sorted(forecast_div_df[div_col].unique()),
                    key="division_forecast_sel",
                )
                sub = forecast_div_df[forecast_div_df[div_col] == sel_div].copy()

                plot_df = sub[[year_col, true_col, pred_col]].melt(
                    id_vars=[year_col],
                    value_vars=[true_col, pred_col],
                    var_name="type",
                    value_name="incidents",
                )

                chart_f_div = (
                    alt.Chart(plot_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(f"{year_col}:O", title="Year"),
                        y=alt.Y("incidents:Q", title="Incidents"),
                        color=alt.Color("type:N", title="Series"),
                        tooltip=[year_col, "type", "incidents"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_f_div, use_container_width=True)
            else:
                st.info("Division forecast CSV does not have expected columns for plotting.")
        else:
            st.info("Division forecast file not found.")
