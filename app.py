
import streamlit as st
import pandas as pd
from datetime import date

from rag.smartgrid_pipeline import run_smartgrid


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartGrid AI | Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1220;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background-color: #111c30;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #f8fafc;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
}

.section-title {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
}

.metric-box {
    background: linear-gradient(135deg, #172840, #142033);
    border: 1px solid #263a55;
    padding: 22px;
    border-radius: 14px;
    min-height: 130px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 750;
    margin-top: 8px;
}

.metric-note {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 5px;
}

.status-high {
    background-color: #451c29;
    color: #fca5a5;
    border: 1px solid #9f3349;
    padding: 15px;
    border-radius: 10px;
    font-weight: 700;
}

.status-normal {
    background-color: #12352c;
    color: #86efac;
    border: 1px solid #267653;
    padding: 15px;
    border-radius: 10px;
    font-weight: 700;
}

.stButton > button {
    background-color: #087fce;
    color: white;
    border-radius: 9px;
    border: none;
    height: 48px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #0b9bf0;
    color: white;
}

div[data-testid="stExpander"] {
    border: 1px solid #263a55;
    border-radius: 10px;
}

hr {
    border-color: #263a55;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. SESSION STATE
# ==========================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "last_input" not in st.session_state:
    st.session_state.last_input = None


# ==========================================
# 4. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">⚡ SmartGrid AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Electricity Usage Prediction '
    '& Consumer Advisory Platform'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Prototype dashboard using synthetic Texas-like data. "
    "Predictions are illustrative high-usage indicators, "
    "not official ERCOT forecasts or outage alerts."
)

st.divider()


# ==========================================
# 5. SIDEBAR INPUTS
# ==========================================

st.sidebar.title("⚙️ Prediction Settings")

st.sidebar.caption(
    "Enter the location and weather conditions."
)

st.sidebar.subheader("📍 Location")

zone = st.sidebar.selectbox(
    "ERCOT Zone",
    ["COAST", "NORTH", "SOUTH", "WEST"]
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=32.75,
    format="%.5f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=-96.77,
    format="%.5f"
)

selected_date = st.sidebar.date_input(
    "Prediction Date",
    value=date.today()
)

st.sidebar.divider()

st.sidebar.subheader("🌦️ Weather Conditions")

temperature_mean = st.sidebar.number_input(
    "Average Temperature (°C)",
    value=25.0,
    step=1.0
)

temperature_max = st.sidebar.number_input(
    "Maximum Temperature (°C)",
    value=32.0,
    step=1.0
)

temperature_min = st.sidebar.number_input(
    "Minimum Temperature (°C)",
    value=18.0,
    step=1.0
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=60
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (m/s)",
    min_value=0.0,
    value=5.0
)

precipitation = st.sidebar.number_input(
    "Precipitation (mm)",
    min_value=0.0,
    value=0.0
)

snowfall = st.sidebar.number_input(
    "Snowfall (cm)",
    min_value=0.0,
    value=0.0
)

st.sidebar.divider()

predict_button = st.sidebar.button(
    "⚡ Generate Prediction",
    type="primary",
    use_container_width=True
)


# ==========================================
# 6. INPUT VALIDATION
# ==========================================

if predict_button:

    if temperature_min > temperature_mean:
        st.sidebar.error(
            "Minimum temperature cannot exceed average temperature."
        )

    elif temperature_mean > temperature_max:
        st.sidebar.error(
            "Average temperature cannot exceed maximum temperature."
        )

    else:

        input_data = {
            "latitude": latitude,
            "longitude": longitude,
            "ERCOT_zone": zone,
            "date": selected_date.isoformat(),
            "temperature_mean": temperature_mean,
            "temperature_max": temperature_max,
            "temperature_min": temperature_min,
            "precipitation_mm": precipitation,
            "wind_speed_mps": wind_speed,
            "snowfall_cm": snowfall,
            "humidity": humidity
        }

        try:

            with st.spinner(
                "Running ML models, retrieving PDF knowledge "
                "and generating Gemini explanation..."
            ):

                result = run_smartgrid(input_data)

            st.session_state.prediction_result = result
            st.session_state.last_input = input_data

            st.sidebar.success("Prediction completed!")

        except Exception as e:

            st.error("Prediction could not be completed.")
            st.exception(e)


# ==========================================
# 7. DISPLAY DASHBOARD
# ==========================================

result = st.session_state.prediction_result
input_data = st.session_state.last_input

if result is None:

    st.markdown(
        '<div class="section-title">'
        '📊 Energy Intelligence Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Configure the weather and location conditions "
        "in the sidebar, then click Generate Prediction."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="metric-box">'
            '<div class="metric-label">REGRESSION</div>'
            '<div class="metric-value">Usage Forecast</div>'
            '<div class="metric-note">Predict electricity usage</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="metric-box">'
            '<div class="metric-label">CLASSIFICATION</div>'
            '<div class="metric-value">Risk Detection</div>'
            '<div class="metric-note">Identify high-usage risk</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            '<div class="metric-box">'
            '<div class="metric-label">GENERATIVE AI</div>'
            '<div class="metric-value">RAG + Gemini</div>'
            '<div class="metric-note">Generate grounded explanations</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.info(
        "Your dashboard is ready. Enter prediction conditions "
        "using the sidebar to begin."
    )


else:

    # ======================================
    # 8. PREDICTION SUMMARY
    # ======================================

    st.markdown(
        '<div class="section-title">📈 Prediction Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">PREDICTED ELECTRICITY USAGE</div>
                <div class="metric-value">
                    {result['predicted_usage']:,.2f}
                </div>
                <div class="metric-note">
                    Model output in dataset units
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">HIGH-USAGE PROBABILITY</div>
                <div class="metric-value">
                    {result['risk_probability']:.2f}%
                </div>
                <div class="metric-note">
                    Classifier-estimated probability
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">USAGE THRESHOLD</div>
                <div class="metric-value">
                    {result['usage_threshold']:,.2f}
                </div>
                <div class="metric-note">
                    Training-set 75th percentile
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # ======================================
    # 9. RISK ASSESSMENT
    # ======================================

    st.markdown(
        '<div class="section-title">🚦 Risk Assessment</div>',
        unsafe_allow_html=True
    )

    risk_status = result["risk_status"].strip().lower()

    is_high_risk = risk_status in [
        "high usage risk",
        "high usage",
        "high risk"
    ]

    if is_high_risk:

        st.markdown(
            '<div class="status-high">'
            '⚠️ HIGH USAGE RISK DETECTED'
            '</div>',
            unsafe_allow_html=True
        )

        st.warning(
            "The model classified this input as high usage. "
            "This does not confirm actual grid stress or an outage."
        )

    else:

        st.markdown(
            '<div class="status-normal">'
            '✅ NORMAL USAGE CLASSIFICATION'
            '</div>',
            unsafe_allow_html=True
        )

        st.success(
            "The classifier did not identify this input "
            "as high usage."
        )

    st.write("")

    st.caption(
        "The high-usage threshold was derived from the synthetic "
        "training dataset. This is not an official ERCOT threshold."
    )

    # ======================================
    # 10. USAGE COMPARISON
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Usage vs Threshold</div>',
        unsafe_allow_html=True
    )

    usage_data = pd.DataFrame({
        "Category": [
            "Predicted Usage",
            "High Usage Threshold"
        ],
        "Value": [
            result["predicted_usage"],
            result["usage_threshold"]
        ]
    })

    st.bar_chart(
        usage_data,
        x="Category",
        y="Value",
        horizontal=True
    )

    difference = (
        result["predicted_usage"] -
        result["usage_threshold"]
    )

    if difference > 0:

        st.write(
            f"Predicted usage is **{difference:,.2f}** "
            "above the model's high-usage threshold."
        )

    else:

        st.write(
            f"Predicted usage is **{abs(difference):,.2f}** "
            "below the model's high-usage threshold."
        )

    # ======================================
    # 11. WEATHER SUMMARY
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">🌦️ Input Conditions</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Average Temperature",
            f"{input_data['temperature_mean']} °C"
        )

    with col2:
        st.metric(
            "Humidity",
            f"{input_data['humidity']}%"
        )

    with col3:
        st.metric(
            "Wind Speed",
            f"{input_data['wind_speed_mps']} m/s"
        )

    with col4:
        st.metric(
            "Precipitation",
            f"{input_data['precipitation_mm']} mm"
        )

    st.write(
        f"**Zone:** {input_data['ERCOT_zone']}  |  "
        f"**Date:** {input_data['date']}  |  "
        f"**Location:** {input_data['latitude']}, "
        f"{input_data['longitude']}"
    )

    # ======================================
    # 12. GEMINI EXPLANATION
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🤖 AI-Powered Grid Advisory'
        '</div>',
        unsafe_allow_html=True
    )

    explanation = result["explanation"]

    if "temporarily unavailable" in explanation.lower():

        st.warning(explanation)

    else:

        st.markdown(explanation)

    # ======================================
    # 13. RETRIEVED PDF SOURCES
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📚 Knowledge Retrieval Sources'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "These are the document sections retrieved by the "
        "RAG system to provide context for the AI explanation."
    )

    for index, source in enumerate(
        result["sources"],
        start=1
    ):

        with st.expander(
            f"📄 Source {index}: {source['document']}"
        ):

            st.write(
                "**Document:**",
                source["document"]
            )

            st.write(
                "**Page:**",
                source["page"]
            )

            st.write(
                "**Cosine Similarity Score:**",
                f"{source['similarity_score']:.4f}"
            )


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "⚡ SmartGrid AI | Texas Electricity Usage "
    "Prediction & Consumer Advisory Prototype"
)

st.caption(
    "Disclaimer: This project uses synthetic data. "
    "Predictions are illustrative and are not official "
    "ERCOT forecasts, grid-stress determinations or outage alerts."
)
