from pathlib import Path
import json
import streamlit as st

REPORT = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "forecast_reliability.json"
)

def show_reliability_panel():
    st.divider()
    st.subheader("Forecast Reliability & Data Continuity")

    if not REPORT.exists():
        st.info("Forecast reliability report not available.")
        return

    try:
        data = json.loads(REPORT.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.warning(f"Unable to load reliability report: {exc}")
        return

    volume = data["traffic_volume"]
    travel = data["travel_time"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Traffic history with gaps",
            f'{volume["irregular_history_percent"]:.2f}%'
        )

    with col2:
        st.metric(
            "Travel-time history with gaps",
            f'{travel["irregular_history_percent"]:.2f}%'
        )

    st.markdown("#### Historical Test Performance")

    rows = []

    for label, result in [
        ("Traffic Volume", volume),
        ("Travel Time", travel)
    ]:
        for history_key, history_label in [
            ("regular_history", "Regular"),
            ("irregular_history", "Irregular")
        ]:
            metrics = result[history_key]
            rows.append({
                "Model": label,
                "History": history_label,
                "Test rows": metrics["rows"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"]
            })

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Historical data continuity: traffic-volume prediction "
        "errors were higher for irregular histories in the test "
        "set. Travel-time errors were similar. These are aggregate "
        "results, not confidence scores for individual predictions."
    )

    st.caption(
        "Regular history means six consecutive observed "
        "30-minute gaps. Irregular history means at least "
        "one gap did not meet that requirement."
    )
