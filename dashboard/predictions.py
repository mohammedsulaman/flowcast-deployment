from dashboard.model_loader import load_model
from dashboard.config import DATA_DIR
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
DATA = DATA_DIR
MODELS = BASE / "models"


@st.cache_resource
def load_prediction_models():
    names = {
        "volume": "traffic_volume_random_forest.joblib",
        "congestion": "congestion_random_forest.joblib",
        "travel": "travel_time_random_forest.joblib",
    }
    return {
        key: load_model(Path(MODELS / filename).name)
        for key, filename in names.items()
    }


@st.cache_data
def load_prediction_history():
    frames = []
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(
            DATA / f"flowcast_{split}.csv",
            parse_dates=["timestamp"]
        )
        frame["split"] = split
        frames.append(frame)

    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(
        ["road_id", "timestamp"]
    ).reset_index(drop=True)

    group = history.groupby("road_id", sort=False)

    # Match the historical travel-time features used in training.
    history["travel_time_lag_1"] = group["travel_time"].shift(1)
    history["travel_time_lag_2"] = group["travel_time"].shift(2)

    previous = group["travel_time"].shift(1)
    history["travel_time_rolling_3"] = (
        previous.groupby(history["road_id"], sort=False)
        .rolling(3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    history["next_timestamp"] = group["timestamp"].shift(-1)
    history["next_split"] = group["split"].shift(-1)
    history["actual_next_volume"] = group["traffic_volume"].shift(-1)
    history["actual_next_travel"] = group["travel_time"].shift(-1)

    history["valid_next_interval"] = (
        (history["next_timestamp"] - history["timestamp"])
        == pd.Timedelta(minutes=30)
    ) & history["split"].eq(history["next_split"])

    return history


def predict_from_row(bundle, row):
    model = bundle["model"]
    features = bundle["features"]
    codes = bundle["road_codes"]

    values = row.copy()
    road_id = values["road_id"]

    if road_id not in codes:
        raise ValueError("Road not found in saved model mapping.")

    values["road_code"] = codes[road_id]

    missing = [
        feature for feature in features
        if feature not in values.index
        or pd.isna(values[feature])
    ]

    if missing:
        raise ValueError(
            "Missing required historical features: "
            + ", ".join(missing)
        )

    x = pd.DataFrame(
        [{feature: values[feature] for feature in features}],
        columns=features
    ).astype("float32")

    return model.predict(x)[0], x


def render_prediction_page():
    st.header("Interactive ML Predictions")
    st.caption(
        "Historical model demonstration using recorded traffic "
        "observations. Not a live forecast or safety alert."
    )

    bundles = load_prediction_models()
    history = load_prediction_history()

    split = st.selectbox(
        "Dataset",
        ["test", "validation", "train"],
        format_func=lambda x: x.title()
    )

    subset = history.loc[history["split"] == split]

    roads = sorted(subset["road_id"].unique())
    road = st.selectbox("Road segment", roads)

    road_data = subset.loc[
        subset["road_id"] == road
    ].sort_values("timestamp")

    selected_time = st.selectbox(
        "Historical timestamp",
        road_data["timestamp"].tolist(),
        index=len(road_data) - 1,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
    )

    selected = road_data.loc[
        road_data["timestamp"] == selected_time
    ].iloc[0]

    st.write(
        "**Road:**", selected.get("road_name", road),
        " | **Timestamp:**", selected_time
    )

    if st.button("Generate Predictions", type="primary"):
        st.subheader("Model predictions")

        try:
            volume, _ = predict_from_row(
                bundles["volume"], selected
            )

            travel, _ = predict_from_row(
                bundles["travel"], selected
            )

            congestion, congestion_x = predict_from_row(
                bundles["congestion"], selected
            )

            a, b, c = st.columns(3)

            a.metric(
                "Next 30-min traffic volume",
                f"{max(0, float(volume)):.1f}"
            )

            b.metric(
                "Next 30-min travel time",
                f"{max(0, float(travel)):.2f}"
            )

            c.metric(
                "Current congestion class",
                str(congestion)
            )

            st.subheader("Congestion class probabilities")

            classifier = bundles["congestion"]["model"]
            probabilities = classifier.predict_proba(
                congestion_x
            )[0]

            probability_table = pd.DataFrame({
                "Congestion level": classifier.classes_,
                "Probability": probabilities
            }).set_index("Congestion level")

            st.bar_chart(probability_table)

            if bool(selected["valid_next_interval"]):
                st.subheader("Recorded next-interval values")

                comparison = pd.DataFrame({
                    "Metric": [
                        "Traffic volume",
                        "Travel time"
                    ],
                    "Predicted": [
                        float(volume),
                        float(travel)
                    ],
                    "Recorded": [
                        float(selected["actual_next_volume"]),
                        float(selected["actual_next_travel"])
                    ]
                })

                st.dataframe(
                    comparison.round(3),
                    hide_index=True,
                    use_container_width=True
                )

                st.caption(
                    "Recorded values are historical observations, "
                    "not future information supplied to the models. "
                    "Some recorded targets may have been imputed; "
                    "use saved evaluation reports for validated accuracy."
                )
            else:
                st.info(
                    "No matching next 30-minute observation "
                    "exists within this dataset split."
                )

            st.success("Predictions generated successfully.")

        except (ValueError, KeyError, TypeError) as error:
            st.warning(
                "This timestamp cannot be predicted: "
                + str(error)
                + ". Select another timestamp."
            )
