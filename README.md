# FlowCast - Smart Traffic Flow Prediction System

### AI & Machine Learning Internship Project

FlowCast is a machine learning-based traffic analytics and prediction system designed to analyze historical road traffic patterns, predict traffic volume and travel time, and classify congestion conditions.

The project includes an interactive Streamlit dashboard for exploring traffic datasets, reviewing model performance, and generating historical demonstration predictions.

## Project Objectives

- Analyze traffic patterns across multiple roads.
- Predict traffic volume for the next 30-minute interval.
- Estimate travel time for the next 30-minute interval.
- Classify current traffic congestion conditions.
- Visualize traffic trends and model results.
- Evaluate model reliability using chronological data splits.

## Technologies Used

| Category | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Storage | Joblib, Gzip |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Version Control | Git, GitHub |

## Machine Learning Models

FlowCast uses Random Forest models for its packaged dashboard.

| Task | Model | Test Performance |
|---|---|---|
| Traffic Volume Prediction | Random Forest Regressor | MAE: 41.684, R²: 0.9546 |
| Travel Time Prediction | Random Forest Regressor | MAE: 0.4295, R²: 0.8294 |
| Congestion Classification | Random Forest Classifier | Accuracy: 86.17%, Macro F1: 0.7573 |

The traffic-volume and travel-time models predict the next eligible 30-minute observation. The congestion model estimates the current congestion category using historical features.

These metrics come from chronological held-out testing on the full development datasets, not the smaller deployment demonstration datasets.

## Dashboard Features

The Streamlit dashboard provides:

1. **Overview** - Traffic dataset summary and key statistics.
2. **Traffic Analysis** - Historical traffic patterns and trends.
3. **Weather & Congestion** - Weather-related traffic analysis.
4. **Model Performance** - Evaluation results for trained models.
5. **Interactive Predictions** - Historical demonstration predictions for selected roads and timestamps.

## Project Structure

```text
flowcast-deployment/
|-- streamlit_app.py
|-- dashboard/
|   |-- app.py
|   |-- predictions.py
|   |-- reliability.py
|   |-- config.py
|   |-- model_loader.py
|-- models/
|   |-- traffic_volume_random_forest.joblib.gz
|   |-- congestion_random_forest.joblib.gz
|   |-- travel_time_random_forest.joblib.gz
|-- data/
|   |-- deployment/
|-- reports/
|-- requirements.txt
|-- runtime.txt
|-- README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mohammedsulaman/flowcast-deployment.git
cd flowcast-deployment
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL shown in the terminal, usually:

`http://localhost:8501`

## Dataset and Evaluation

The development dataset contains traffic observations across 25 roads from January through May 2025.

The machine learning workflow uses chronological training, validation, and test splits to reduce future-data leakage.

Models use historical lag and rolling-window features. The deployed package includes smaller historical demonstration datasets to reduce repository size.

## Reliability and Limitations

- Predictions are historical demonstrations, not live traffic forecasts.
- Forecasting requires recent historical observations.
- Irregular observation intervals can affect feature reliability.
- The congestion classifier predicts the current congestion category rather than the next interval.
- Accident-risk modeling was explored during development but is not included in the deployed dashboard because its predictive performance was insufficient for operational safety alerts.
- Reported test metrics should not be interpreted as performance guarantees on new roads or future traffic conditions.

## Future Improvements

- Integrate real-time traffic feeds.
- Extend forecasting to multiple future intervals.
- Add additional road networks and geographic regions.
- Improve performance under irregular observations.
- Explore more advanced temporal forecasting models.
- Add automated model monitoring.

## Internship Project

**Project:** FlowCast - Smart Traffic Flow Prediction System

**Organization:** InternMo

**Category:** Artificial Intelligence / Machine Learning

**Repository:** https://github.com/mohammedsulaman/flowcast-deployment

## License

No open-source license has been assigned. Public visibility does not automatically grant permission to reuse or redistribute the project.
