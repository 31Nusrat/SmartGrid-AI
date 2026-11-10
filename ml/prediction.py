
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from ml.severity_prediction import predict_severity


# ==========================================
# 1. LOAD TRAINED MODELS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

regression_model = joblib.load(
    ARTIFACT_DIR / "usage_regression_model.pkl"
)

classification_artifact = joblib.load(
    ARTIFACT_DIR / "usage_classification_model.pkl"
)

classification_model = classification_artifact["model"]
usage_threshold = classification_artifact["threshold"]

print("All SmartGrid ML models loaded successfully!")


# ==========================================
# 2. PREPARE INPUT FEATURES
# ==========================================

def prepare_input(input_data):

    df = pd.DataFrame([input_data])

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.day_of_year

    df["CDD"] = np.maximum(
        0, df["temperature_mean"] - 18.3
    )

    df["HDD"] = np.maximum(
        0, 18.3 - df["temperature_mean"]
    )

    numeric_features = [
        "latitude",
        "longitude",
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation_mm",
        "wind_speed_mps",
        "snowfall_cm",
        "humidity",
        "year",
        "month",
        "dayofyear",
        "CDD",
        "HDD"
    ]

    categorical_features = ["ERCOT_zone"]

    feature_columns = numeric_features + categorical_features

    return df[feature_columns]


# ==========================================
# 3. RUN COMPLETE PREDICTION
# ==========================================

def predict_grid_status(input_data):

    X = prepare_input(input_data)

    # --------------------------------------
    # A. ELECTRICITY USAGE REGRESSION
    # --------------------------------------

    predicted_usage = float(
        regression_model.predict(X)[0]
    )

    # --------------------------------------
    # B. HIGH USAGE CLASSIFICATION
    # --------------------------------------

    risk_status = int(
        classification_model.predict(X)[0]
    )

    probabilities = classification_model.predict_proba(X)[0]

    positive_class_index = list(
        classification_model.classes_
    ).index(1)

    high_usage_probability = (
        float(probabilities[positive_class_index]) * 100
    )

    if risk_status == 1:
        status_text = "High Usage Risk"
    else:
        status_text = "Normal Usage"

    # --------------------------------------
    # C. OUTAGE SEVERITY PREDICTION
    # --------------------------------------

    severity_result = predict_severity(
        input_data,
        predicted_usage
    )

    expected_customers = severity_result[
        "expected_customers"
    ]

    expected_duration = severity_result[
        "expected_duration"
    ]

    severity_score = severity_result[
        "severity_score"
    ]

    severity_level = severity_result[
        "severity_level"
    ]

    # --------------------------------------
    # D. CREATE RAG QUERY
    # --------------------------------------

    rag_query = f"""
    Analyze the SmartGrid electricity prediction.

    ELECTRICITY USAGE:
    Predicted electricity usage: {predicted_usage:.2f}
    Usage threshold: {usage_threshold:.2f}

    HIGH USAGE CLASSIFICATION:
    Risk status: {status_text}
    High usage probability: {high_usage_probability:.2f}%

    OUTAGE SEVERITY ESTIMATION:
    Expected customers affected: {expected_customers:.2f}
    Expected outage duration: {expected_duration:.2f}
    Severity score: {severity_score:.2f}%
    Severity level: {severity_level}

    Based on the provided knowledge documents:

    1. Explain the possible reasons behind the predicted
       electricity demand and severity level.

    2. Explain potential grid-related concerns.

    3. Provide recommendations for consumers and
       electricity grid operators.

    4. Suggest preventive measures where appropriate.

    Use the retrieved documents to support your explanation.
    Do not invent facts that are not supported by the documents.
    """

    # --------------------------------------
    # E. RETURN ALL PREDICTIONS
    # --------------------------------------

    return {
        "predicted_usage": round(predicted_usage, 2),

        "risk_status": status_text,

        "risk_probability": round(
            high_usage_probability, 2
        ),

        "usage_threshold": round(
            float(usage_threshold), 2
        ),

        "expected_customers": round(
            expected_customers, 2
        ),

        "expected_duration": round(
            expected_duration, 2
        ),

        "severity_score": round(
            severity_score, 2
        ),

        "severity_level": severity_level,

        "rag_query": rag_query
    }


# ==========================================
# 4. TEST COMPLETE PREDICTION
# ==========================================

if __name__ == "__main__":

    sample_input = {
        "latitude": 32.7767,
        "longitude": -96.7970,
        "ERCOT_zone": "NORTH",
        "date": "2026-07-15",
        "temperature_mean": 38,
        "temperature_max": 42,
        "temperature_min": 34,
        "precipitation_mm": 0,
        "wind_speed_mps": 4,
        "snowfall_cm": 0,
        "humidity": 55
    }

    result = predict_grid_status(sample_input)

    print("\n========== SMARTGRID PREDICTION ==========")

    for key, value in result.items():

        if key != "rag_query":
            print(f"{key}: {value}")

    print("\n========== RAG QUERY ==========")
    print(result["rag_query"])
