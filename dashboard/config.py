from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent

DEPLOYMENT_MODE = (
    os.getenv("FLOWCAST_DEPLOYMENT_MODE", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

DATA_DIR = (
    ROOT / "data" / "deployment"
    if DEPLOYMENT_MODE
    else ROOT / "data" / "processed"
)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
