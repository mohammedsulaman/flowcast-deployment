from pathlib import Path
import gzip
import io
import os

import joblib

ROOT = Path(__file__).resolve().parent.parent

def load_model(filename):
    """Load a trusted FlowCast model from local or compressed assets."""

    deployment = (
        os.getenv("FLOWCAST_DEPLOYMENT_MODE", "false")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )

    if deployment:
        compressed = (
            ROOT / "models" / f"{filename}.gz"
        )

        if not compressed.exists():
            compressed = (
                ROOT / "deployment_assets"
                / "models"
                / f"{filename}.gz"
            )

        if not compressed.exists():
            raise FileNotFoundError(
                f"Compressed model missing: {filename}"
            )

        with gzip.open(compressed, "rb") as stream:
            return joblib.load(io.BytesIO(stream.read()))

    original = ROOT / "models" / filename

    if not original.exists():
        raise FileNotFoundError(original)

    return joblib.load(original)
