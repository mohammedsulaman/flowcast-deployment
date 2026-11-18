# FlowCast — Streamlit Deployment

FlowCast is a traffic-flow prediction demonstration.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run streamlit_app.py
```

## Deployment

1. Create a separate GitHub repository.
2. Upload the contents of this deployment bundle.
3. In Streamlit Community Cloud, select the repository.
4. Set the entry point to `streamlit_app.py`.
5. Deploy the application.

The application uses compressed Random Forest models
and lightweight historical demonstration datasets.

Predictions are historical demonstrations, not live
traffic forecasts. Accident-risk predictions are not
included in this deployment.
