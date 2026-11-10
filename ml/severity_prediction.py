
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Load trained models
customer_model = joblib.load(
    ARTIFACT_DIR / "customer_severity_model.pkl"
)

duration_model = joblib.load(
    ARTIFACT_DIR / "duration_model.pkl"
)

zone_encoder = joblib.load(
    ARTIFACT_DIR / "ercot_zone_encoder.pkl"
)

# Reference values used for severity calculation
MAX_CUSTOMERS = 219964.74683500358
MAX_DURATION = 24.738202506940873


def predict_severity(input_data, predicted_usage):

    zone = input_data["ERCOT_zone"]

    if zone not in zone_encoder.classes_:
        raise ValueError(f"Unknown ERCOT zone: {zone}")

    selected_date = pd.to_datetime(input_data["date"])

    temperature_mean = input_data["temperature_mean"]
    temperature_max = input_data["temperature_max"]
    temperature_min = input_data["temperature_min"]

    features = pd.DataFrame([{
        "zone_usage": predicted_usage,
        "temperature_mean": temperature_mean,
        "temperature_max": temperature_max,
        "temperature_min": temperature_min,
        "humidity": input_data["humidity"],
        "precipitation_mm": input_data["precipitation_mm"],
        "wind_speed_mps": input_data["wind_speed_mps"],
        "snowfall_cm": input_data["snowfall_cm"],
        "CDD": max(temperature_mean - 18, 0),
        "temp_range": temperature_max - temperature_min,
        "weekend": int(selected_date.dayofweek in [5, 6]),
        "ERCOT_zone_encoded": int(
            zone_encoder.transform([zone])[0]
        )
    }])

    expected_customers = max(
        0, float(customer_model.predict(features)[0])
    )

    expected_duration = max(
        0, float(duration_model.predict(features)[0])
    )

    severity_score = (
        (expected_customers / MAX_CUSTOMERS) * 0.6
        + (expected_duration / MAX_DURATION) * 0.4
    ) * 100

    if severity_score >= 75:
        severity_level = "Critical"
    elif severity_score >= 50:
        severity_level = "High"
    elif severity_score >= 25:
        severity_level = "Medium"
    else:
        severity_level = "Low"

    return {
        "expected_customers": expected_customers,
        "expected_duration": expected_duration,
        "severity_score": severity_score,
        "severity_level": severity_level
    }
