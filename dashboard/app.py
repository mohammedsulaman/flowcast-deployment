from dashboard.config import DATA_DIR
from pathlib import Path
import json
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
DATA = DATA_DIR
REPORTS = BASE / "reports"

st.set_page_config(
    page_title="FlowCast",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 FlowCast")
st.caption("Smart Traffic Flow Prediction System")


@st.cache_data
def load_data():
    return pd.read_csv(
        DATA / "flowcast_cleaned.csv",
        parse_dates=["timestamp"]
    )


@st.cache_data
def load_metrics():
    files = {
        "Traffic Volume": "baseline_metrics.json",
        "Congestion": "congestion_metrics.json",
        "Travel Time": "travel_time_metrics.json"
    }
    results = {}
    for name, filename in files.items():
        path = REPORTS / filename
        if path.exists():
            results[name] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return results


df = load_data()
metrics = load_metrics()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Traffic Analysis",
        "Weather & Congestion",
        "Model Performance",
        "Interactive Predictions"
    ]
)

roads = sorted(df["road_id"].unique())

selected_roads = st.sidebar.multiselect(
    "Select roads",
    roads,
    default=roads[:3]
)

filtered = df[df["road_id"].isin(selected_roads)].copy()

if page == "Overview":
    st.header("Traffic Overview")

    a, b, c, d = st.columns(4)
    a.metric("Total records", f"{len(df):,}")
    b.metric("Road segments", df["road_id"].nunique())
    c.metric(
        "Average traffic volume",
        f"{df['traffic_volume'].mean():.1f}"
    )
    d.metric(
        "Average travel time",
        f"{df['travel_time'].mean():.2f}"
    )

    st.subheader("Daily average traffic volume")
    daily = (
        df.set_index("timestamp")
        .resample("D")["traffic_volume"]
        .mean()
    )
    st.line_chart(daily)

elif page == "Traffic Analysis":
    st.header("Traffic Analysis")

    if filtered.empty:
        st.warning("Select at least one road.")
    else:
        daily = (
            filtered.assign(
                day=filtered["timestamp"].dt.date
            )
            .groupby(["day", "road_id"])["traffic_volume"]
            .mean()
            .unstack()
        )

        st.subheader("Traffic volume by day")
        st.line_chart(daily)

        hourly = (
            filtered.assign(
                hour=filtered["timestamp"].dt.hour
            )
            .groupby("hour")["traffic_volume"]
            .mean()
        )

        st.subheader("Average traffic volume by hour")
        st.bar_chart(hourly)

        st.subheader("Road statistics")
        summary = filtered.groupby("road_id").agg(
            average_volume=("traffic_volume", "mean"),
            average_speed=("avg_speed", "mean"),
            average_travel_time=("travel_time", "mean")
        )
        st.dataframe(summary.round(2))

elif page == "Weather & Congestion":
    st.header("Weather & Congestion")

    st.subheader("Weather conditions")
    st.bar_chart(
        df["weather_condition"].value_counts()
    )

    st.subheader("Congestion levels")
    st.bar_chart(
        df["congestion_level"]
        .fillna("Unknown")
        .value_counts()
    )

    st.subheader("Traffic volume by weather")
    weather_summary = (
        df.groupby("weather_condition")["traffic_volume"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(weather_summary)

elif page == "Model Performance":
    st.header("Machine Learning Performance")

    for name, result in metrics.items():
        st.subheader(name)
        st.json(result.get("test", {}))

    st.info(
        "Accident-risk prediction is experimental "
        "and is not used for operational alerts."
    )

st.divider()
st.caption(
    "FlowCast Research Prototype | "
    "Historical data: January–May 2025"
)


if page == "Interactive Predictions":
    from predictions import render_prediction_page
    render_prediction_page()


# FLOWCAST STEP 107 RELIABILITY PANEL
if page == "Interactive Predictions":
    from dashboard.reliability import show_reliability_panel
    show_reliability_panel()
