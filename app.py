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

# Existing (test years only)
FORECAST_REGION_PATH = CLEAN_DIR / "forecast_region_2022_2023.csv"
FORECAST_DIV_PATH = CLEAN_DIR / "forecast_division_2022_2023.csv"

# Plot-ready full timeline
FORECAST_REGION_PLOT_PATH = CLEAN_DIR / "forecast_region_plot_2018_2023.csv"
FORECAST_DIV_PLOT_PATH = CLEAN_DIR / "forecast_division_plot_2018_2023.csv"

AGG_REGION_YEAR = CLEAN_DIR / "agg_region_year.csv"
AGG_DIV_YEAR = CLEAN_DIR / "agg_division_year.csv"
AGG_STATION_YEAR = CLEAN_DIR / "agg_station_year.csv"
AGG_OFFENCE_YEAR = CLEAN_DIR / "agg_offence_year.csv"
AGG_YEARLY = CLEAN_DIR / "agg_yearly.csv"

st.set_page_config(page_title="Ireland Crime Dashboard", layout="wide")
alt.themes.enable("dark")

# =========================
# Global chart label helpers
# =========================
def pretty_label(name: str) -> str:
    """snake_case -> Title Case (e.g., total_incidents -> Total Incidents)"""
    return name.replace("_", " ").strip().title()

def x_field(field: str, *, sort=None, title: str | None = None):
    return alt.X(f"{field}:N", sort=sort, title=title or pretty_label(field))

def x_field_ordinal(field: str, *, title: str | None = None):
    return alt.X(f"{field}:O", title=title or pretty_label(field))

def x_field_quant(field: str, *, title: str | None = None):
    return alt.X(f"{field}:Q", title=title or pretty_label(field))

def y_field_quant(field: str, *, title: str | None = None):
    return alt.Y(f"{field}:Q", title=title or pretty_label(field))

def color_field(field: str, *, title: str | None = None):
    return alt.Color(f"{field}:N", title=title or pretty_label(field))

def tt(field: str, *, fmt: str | None = None, title: str | None = None):
    """Tooltip with pretty title; optional number format."""
    if fmt:
        return alt.Tooltip(f"{field}:Q", format=fmt, title=title or pretty_label(field))
    return alt.Tooltip(field, title=title or pretty_label(field))

# =========================
# Loaders
# =========================
@st.cache_data
def load_recorded() -> pd.DataFrame:
    df = pd.read_csv(REC_ENRICHED)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["incidents"] = pd.to_numeric(df["incidents"], errors="coerce").fillna(0).astype(int)

    # Normalise region names
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


# =========================
# Load data
# =========================
rec_df = load_recorded()

det_sex_df = load_optional(DET_SEX_PATH)
det_rate_df = load_optional(DET_RATE_PATH)

forecast_region_df = load_optional(FORECAST_REGION_PATH)
forecast_div_df = load_optional(FORECAST_DIV_PATH)

forecast_region_plot_df = load_optional(FORECAST_REGION_PLOT_PATH)
forecast_div_plot_df = load_optional(FORECAST_DIV_PLOT_PATH)

agg_region_df = load_optional(AGG_REGION_YEAR)
agg_div_df = load_optional(AGG_DIV_YEAR)
agg_station_df = load_optional(AGG_STATION_YEAR)
agg_offence_df = load_optional(AGG_OFFENCE_YEAR)
agg_yearly_df = load_optional(AGG_YEARLY)

min_year = int(rec_df["year"].min())
max_year = int(rec_df["year"].max())

region_options = ["All regions"] + sorted(rec_df["region"].dropna().unique())
division_options_all = ["All divisions"] + sorted(rec_df["division"].dropna().unique())
offence_options = ["All offences"] + sorted(rec_df["offence"].dropna().unique())

# =========================
# Session state defaults
# =========================
if "page" not in st.session_state:
    st.session_state.page = "Overview"

def reset_dashboard():
    st.session_state.clear()
    st.rerun()

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
    key="page",
)

show_filters = page not in ["Overview", "Forecasting Results"]

st.sidebar.markdown("---")

if show_filters:
    st.sidebar.markdown("### Global Filters")

    year_range = st.sidebar.slider(
        "Year range",
        min_year,
        max_year,
        (max_year - 5, max_year),
        key="year_range",
    )

    region_sel = st.sidebar.selectbox("Region", region_options, key="region")

    if region_sel != "All regions":
        div_options = ["All divisions"] + sorted(
            rec_df.loc[rec_df["region"] == region_sel, "division"].dropna().unique()
        )
    else:
        div_options = division_options_all

    division_sel = st.sidebar.selectbox("Division", div_options, key="division")
    offence_sel = st.sidebar.selectbox("Offence", offence_options, key="offence")

else:
    year_range = (min_year, max_year)
    region_sel = "All regions"
    division_sel = "All divisions"
    offence_sel = "All offences"

st.sidebar.markdown("---")

if st.sidebar.button(" Reset Dashboard"):
    reset_dashboard()

st.sidebar.caption("Filters apply across analytical sections.")

# =========================
# Apply global filters
# =========================
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    f = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])].copy()

    if region_sel != "All regions":
        f = f[f["region"] == region_sel]
    if division_sel != "All divisions":
        f = f[f["division"] == division_sel]
    if offence_sel != "All offences":
        f = f[f["offence"] == offence_sel]

    return f

filtered_rec = apply_filters(rec_df)

# =========================
# Insight generator
# =========================
def generate_insights(df_full: pd.DataFrame, year: int, offence: str, region: str, division: str) -> list[str]:
    insights: list[str] = []

    cur = df_full[df_full["year"] == year].copy()
    prev = df_full[df_full["year"] == (year - 1)].copy()

    if region != "All regions":
        cur = cur[cur["region"] == region]
        prev = prev[prev["region"] == region]

    if division != "All divisions":
        cur = cur[cur["division"] == division]
        prev = prev[prev["division"] == division]

    if offence != "All offences":
        cur = cur[cur["offence"] == offence]
        prev = prev[prev["offence"] == offence]

    if cur.empty:
        return ["No incidents found for the selected filters."]

    total_now = int(cur["incidents"].sum())
    total_prev = int(prev["incidents"].sum()) if not prev.empty else 0

    scope = "Ireland"
    if division != "All divisions":
        scope = f"{division} division"
    elif region != "All regions":
        scope = f"{region} region"

    if total_prev > 0:
        diff = total_now - total_prev
        pct = abs(diff / total_prev * 100)
        direction = "increase" if diff > 0 else "decrease" if diff < 0 else "no change"
        insights.append(
            f"In {year}, **{scope}** recorded **{total_now:,} incidents**, "
            f"a **{pct:.1f}% {direction}** compared with {year - 1}."
        )
    else:
        insights.append(f"In {year}, **{scope}** recorded **{total_now:,} incidents**.")

    by_off = (
        cur.groupby("offence", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
    )
    if not by_off.empty:
        top = by_off.iloc[0]
        share = top["incidents"] / total_now * 100
        insights.append(
            f"The most common offence was **{top['offence']}**, accounting for **{share:.1f}%** of incidents."
        )

    by_reg = (
        cur.groupby("region", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
    )
    if region == "All regions" and len(by_reg) >= 2:
        r1, r2 = by_reg.iloc[0], by_reg.iloc[1]
        insights.append(
            f"**{r1['region']}** recorded the highest incidents ({r1['incidents']:,}), "
            f"**{r1['incidents'] - r2['incidents']:,}** more than {r2['region']}."
        )

    return insights

# =========================
# Forecast chart builder
# =========================
def build_forecast_chart(
    df: pd.DataFrame,
    year_col: str,
    group_col: str,
    true_col: str,
    pred_col: str,
    selected_value: str,
    height: int = 300,
):
    sub = df[df[group_col] == selected_value].copy()
    sub[year_col] = pd.to_numeric(sub[year_col], errors="coerce")
    sub = sub.dropna(subset=[year_col]).sort_values(year_col)

    plot_df = sub[[year_col, true_col, pred_col]].melt(
        id_vars=[year_col],
        value_vars=[true_col, pred_col],
        var_name="Series",
        value_name="Incidents",
    )

    plot_df["Series"] = plot_df["Series"].map({true_col: "Actual", pred_col: "Predicted"}).fillna(plot_df["Series"])

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{year_col}:O", title=pretty_label(year_col)),
            y=alt.Y("Incidents:Q", title="Incidents"),
            color=alt.Color("Series:N", title="Series"),
            tooltip=[
                alt.Tooltip(f"{year_col}:O", title=pretty_label(year_col)),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Incidents:Q", title="Incidents", format=","),
            ],
        )
        .properties(height=height)
    )
    return chart

# =========================
# PAGE: Overview
# =========================
if page == "Overview":
    st.markdown("## 🇮reland Crime Stats Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidents", f"{int(filtered_rec['incidents'].sum()):,}")
    c2.metric("Offence types", filtered_rec["offence"].nunique())
    c3.metric("Divisions", filtered_rec["division"].nunique())
    c4.metric("Garda stations", filtered_rec["garda_station"].nunique())

    st.markdown("### National Crime Trend (All Ireland)")

    if agg_yearly_df is not None:
        trend_df = agg_yearly_df.rename(columns={"total_incidents": "incidents"})
    else:
        trend_df = rec_df.groupby("year", as_index=False)["incidents"].sum()

    chart_nat = (
        alt.Chart(trend_df)
        .mark_area(opacity=0.7)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("incidents:Q", title="Incidents"),
            tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("incidents:Q", title="Incidents", format=",")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart_nat, use_container_width=True, theme=None)

    st.markdown("### Crimes by Region (All Years)")

    reg_agg = (
        rec_df.groupby("region", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
    )

    chart_reg = (
        alt.Chart(reg_agg)
        .mark_bar()
        .encode(
            x=alt.X("region:N", sort="-y", title="Region"),
            y=alt.Y("incidents:Q", title="Incidents"),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("incidents:Q", title="Incidents", format=","),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart_reg, use_container_width=True, theme=None)

    st.markdown("### Offence Trend Over Time (Ireland)")

    off_sel = st.selectbox("Choose offence", sorted(rec_df["offence"].unique()), key="overview_offence_trend")

    off_trend = (
        rec_df[rec_df["offence"] == off_sel]
        .groupby("year", as_index=False)["incidents"]
        .sum()
    )

    chart_off = (
        alt.Chart(off_trend)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("incidents:Q", title="Incidents"),
            tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("incidents:Q", title="Incidents", format=",")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart_off, use_container_width=True, theme=None)

    st.markdown("### Key Insights")

    insights = generate_insights(
        df_full=rec_df,
        year=year_range[1],
        offence=offence_sel,
        region=region_sel,
        division=division_sel,
    )
    for i, txt in enumerate(insights, 1):
        st.markdown(f"**{i}.** {txt}")

# =========================
# PAGE: Recorded Crime Explorer
# =========================
elif page == "Recorded Crime Explorer":
    st.markdown("## 📊 Recorded Crime Explorer")

    st.markdown("### Crime Trend (Filtered Scope)")

    trend_scope = (
        filtered_rec.groupby("year", as_index=False)["incidents"]
        .sum()
        .sort_values("year")
    )

    chart_scope = (
        alt.Chart(trend_scope)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("incidents:Q", title="Incidents"),
            tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("incidents:Q", title="Incidents", format=",")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart_scope, use_container_width=True, theme=None)

    st.markdown("### Garda Stations by Incident Volume")

    sort_order = st.radio("Sort order", ["Highest incidents", "Lowest incidents"], horizontal=True)

    station_agg = filtered_rec.groupby("garda_station", as_index=False)["incidents"].sum()

    if sort_order == "Highest incidents":
        station_agg = station_agg.sort_values("incidents", ascending=False).head(20)
    else:
        station_agg = station_agg.sort_values("incidents", ascending=True).head(20)

    chart_station = (
        alt.Chart(station_agg)
        .mark_bar()
        .encode(
            x=alt.X("garda_station:N", sort="-y", title="Garda Station"),
            y=alt.Y("incidents:Q", title="Incidents"),
            tooltip=[
                alt.Tooltip("garda_station:N", title="Garda Station"),
                alt.Tooltip("incidents:Q", title="Incidents", format=","),
            ],
        )
        .properties(height=400)
    )
    st.altair_chart(chart_station, use_container_width=True, theme=None)

    st.markdown("### Insights")

    insights = generate_insights(
        df_full=rec_df,
        year=year_range[1],
        offence=offence_sel,
        region=region_sel,
        division=division_sel,
    )
    for i, txt in enumerate(insights, 1):
        st.markdown(f"**{i}.** {txt}")

# =========================
# PAGE: Region & Division Insights
# =========================
elif page == "Region & Division Insights":
    st.markdown("## 🗺️ Region & Division Insights")

    st.markdown("### Region Trends (Filtered Scope)")

    reg_trend = (
        filtered_rec.groupby(["region", "year"], as_index=False)["incidents"]
        .sum()
        .sort_values(["region", "year"])
    )

    chart_reg_trend = (
        alt.Chart(reg_trend)
        .mark_line()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("incidents:Q", title="Incidents"),
            color=alt.Color("region:N", title="Region"),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("incidents:Q", title="Incidents", format=","),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart_reg_trend, use_container_width=True, theme=None)

    st.markdown("### Division Breakdown (Filtered Scope)")

    div_agg = (
        filtered_rec.groupby("division", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
    )

    if div_agg.empty:
        st.info("No division data available for selected filters.")
    else:
        chart_div = (
            alt.Chart(div_agg)
            .mark_bar()
            .encode(
                x=alt.X("division:N", sort="-y", title="Division"),
                y=alt.Y("incidents:Q", title="Incidents"),
                tooltip=[
                    alt.Tooltip("division:N", title="Division"),
                    alt.Tooltip("incidents:Q", title="Incidents", format=","),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(chart_div, use_container_width=True, theme=None)

    st.markdown("### Insights")

    insights = generate_insights(
        df_full=rec_df,
        year=year_range[1],
        offence=offence_sel,
        region=region_sel,
        division=division_sel,
    )
    for i, txt in enumerate(insights, 1):
        st.markdown(f"**{i}.** {txt}")

# =========================
# PAGE: Offence Analysis
# =========================
elif page == "Offence Analysis":
    st.markdown("## ⚖️ Offence Analysis")

    st.markdown("### Offence Composition (Filtered Scope)")

    off_mix = (
        filtered_rec.groupby("offence", as_index=False)["incidents"]
        .sum()
        .sort_values("incidents", ascending=False)
    )

    if off_mix.empty:
        st.info("No offence data available for selected filters.")
    else:
        chart_off_mix = (
            alt.Chart(off_mix.head(20))
            .mark_bar()
            .encode(
                x=alt.X("offence:N", sort="-y", title="Offence"),
                y=alt.Y("incidents:Q", title="Incidents"),
                tooltip=[
                    alt.Tooltip("offence:N", title="Offence"),
                    alt.Tooltip("incidents:Q", title="Incidents", format=","),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_off_mix, use_container_width=True, theme=None)

    st.markdown("### Insights")

    insights = generate_insights(
        df_full=rec_df,
        year=year_range[1],
        offence=offence_sel,
        region=region_sel,
        division=division_sel,
    )
    for i, txt in enumerate(insights, 1):
        st.markdown(f"**{i}.** {txt}")

# =========================
# PAGE: Forecasting Results
# =========================
elif page == "Forecasting Results":
    st.markdown("## 🔮 Forecasting Results")

    st.markdown(
        "Forecasts were generated using a Random Forest Regressor trained on "
        "aggregated crime counts (2018–2021) and evaluated on 2022–2023. "
        "The plots below show **Actual vs Predicted** trends."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Region-level Forecast (2018–2023)")

        if forecast_region_plot_df is not None:
            df_use = forecast_region_plot_df.copy()

            cols = df_use.columns.str.lower()
            year_col = df_use.columns[cols == "year"][0]
            reg_col = df_use.columns[cols.str.contains("region")][0]

            true_col = next((c for c in df_use.columns if "true" in c.lower() or "actual" in c.lower()), "total_incidents")
            pred_col = next((c for c in df_use.columns if "pred" in c.lower()), "prediction")

            sel_reg = st.selectbox("Select region", sorted(df_use[reg_col].dropna().unique()), key="forecast_region_sel")

            chart_reg_forecast = build_forecast_chart(
                df=df_use,
                year_col=year_col,
                group_col=reg_col,
                true_col=true_col,
                pred_col=pred_col,
                selected_value=sel_reg,
                height=300,
            )
            st.altair_chart(chart_reg_forecast, use_container_width=True, theme=None)
            st.caption("Predicted values appear for **2022–2023** (test years).")

        elif forecast_region_df is not None:
            st.info("Plot-ready file not found. Showing 2022–2023 forecast only.")
            df_use = forecast_region_df.copy()

            cols = df_use.columns.str.lower()
            year_col = df_use.columns[cols == "year"][0]
            reg_col = df_use.columns[cols.str.contains("region")][0]
            true_col = next((c for c in df_use.columns if "true" in c.lower() or "actual" in c.lower()), "total_incidents")
            pred_col = next((c for c in df_use.columns if "pred" in c.lower()), "prediction")

            sel_reg = st.selectbox("Select region", sorted(df_use[reg_col].dropna().unique()), key="forecast_region_sel_fallback")

            st.altair_chart(
                build_forecast_chart(df_use, year_col, reg_col, true_col, pred_col, sel_reg, 300),
                use_container_width=True,
                theme=None,
            )
        else:
            st.info("Region-level forecast data not available.")

    with col2:
        st.markdown("### Division-level Forecast (2018–2023)")

        if forecast_div_plot_df is not None:
            df_use = forecast_div_plot_df.copy()

            cols = df_use.columns.str.lower()
            year_col = df_use.columns[cols == "year"][0]
            div_col = df_use.columns[cols.str.contains("division")][0]

            true_col = next((c for c in df_use.columns if "true" in c.lower() or "actual" in c.lower()), "total_incidents")
            pred_col = next((c for c in df_use.columns if "pred" in c.lower()), "prediction")

            sel_div = st.selectbox("Select division", sorted(df_use[div_col].dropna().unique()), key="forecast_div_sel")

            chart_div_forecast = build_forecast_chart(
                df=df_use,
                year_col=year_col,
                group_col=div_col,
                true_col=true_col,
                pred_col=pred_col,
                selected_value=sel_div,
                height=300,
            )
            st.altair_chart(chart_div_forecast, use_container_width=True, theme=None)
            st.caption("Predicted values appear for **2022–2023** (test years).")

        elif forecast_div_df is not None:
            st.info("Plot-ready file not found. Showing 2022–2023 forecast only.")
            df_use = forecast_div_df.copy()

            cols = df_use.columns.str.lower()
            year_col = df_use.columns[cols == "year"][0]
            div_col = df_use.columns[cols.str.contains("division")][0]
            true_col = next((c for c in df_use.columns if "true" in c.lower() or "actual" in c.lower()), "total_incidents")
            pred_col = next((c for c in df_use.columns if "pred" in c.lower()), "prediction")

            sel_div = st.selectbox("Select division", sorted(df_use[div_col].dropna().unique()), key="forecast_div_sel_fallback")

            st.altair_chart(
                build_forecast_chart(df_use, year_col, div_col, true_col, pred_col, sel_div, 300),
                use_container_width=True,
                theme=None,
            )
        else:
            st.info("Division-level forecast data not available.")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:0.9em; opacity:0.7;">
        Developed by <b>Jyothiswaruban Madurai Ravishankar</b><br>
        MSc – Information Systems with Computing, Dublin Business School<br>
        Applied Research Project – Irish Crime Data Analytics
    </div>
    """,
    unsafe_allow_html=True,
)
