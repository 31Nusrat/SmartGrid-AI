

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

/* ==========================================
   SMARTGRID AI — MIDNIGHT NAVY DARK THEME
========================================== */

:root {
    --bg: #0B1220;
    --sidebar: #111D30;
    --panel: #192943;
    --panel-soft: #203653;

    --accent: #4385C5;
    --accent-hover: #5B9BDB;
    --accent-dark: #2D659B;

    --text: #EAF2FC;
    --heading: #F3F7FF;
    --muted: #A9BCD3;

    --border: #304967;
    --input: #192943;

    --shadow: 0 8px 25px rgba(0, 0, 0, 0.22);
}

/* MAIN APPLICATION */

.stApp {
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(67, 133, 197, 0.12),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            #0B1220 0%,
            #101C30 55%,
            #0B1220 100%
        );
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

[data-testid="stHeader"] {
    background: rgba(11, 18, 32, 0.95);
}

[data-testid="stToolbar"] {
    background: transparent;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111D30 0%,
        #15243A 55%,
        #111D30 100%
    ) !important;
    border-right: 1px solid #304967;
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #C5D6EA !important;
    font-weight: 600;
}

/* SIDEBAR INPUTS */

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: #192943 !important;
    color: #EAF2FC !important;
    border: 1px solid #304967 !important;
    border-radius: 10px !important;
    -webkit-text-fill-color: #EAF2FC !important;
}

/* SELECT BOXES */

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #192943 !important;
    color: #EAF2FC !important;
    border: 1px solid #304967 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #EAF2FC !important;
    -webkit-text-fill-color: #EAF2FC !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] * {
    background-color: #192943 !important;
    color: #EAF2FC !important;
}

[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    fill: #A9BCD3 !important;
}

/* DROPDOWN MENU */

[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="menu"] ul,
[role="listbox"] {
    background: #192943 !important;
    color: #EAF2FC !important;
}

[role="option"] {
    background: #192943 !important;
    color: #EAF2FC !important;
}

[role="option"]:hover {
    background: #304967 !important;
}

/* PREDICTION DATE */

[data-testid="stSidebar"] [data-testid="stDateInput"],
[data-testid="stSidebar"] [data-testid="stDateInput"] > div,
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] > div {
    background-color: #203653 !important;
    border-color: #456487 !important;
    border-radius: 10px !important;
    color-scheme: dark !important;
}

[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    background: #203653 !important;
    color: #EAF2FC !important;
    -webkit-text-fill-color: #EAF2FC !important;
    border: none !important;
    box-shadow: none !important;
    color-scheme: dark !important;
}

[data-testid="stSidebar"] [data-testid="stDateInput"] button {
    background: #203653 !important;
    color: #DCEBFF !important;
    border: none !important;
}

[data-testid="stSidebar"] [data-testid="stDateInput"] svg {
    color: #DCEBFF !important;
    fill: #DCEBFF !important;
    stroke: #DCEBFF !important;
}

/* DATE PICKER POPUP */

[data-baseweb="calendar"],
[data-baseweb="calendar"] *,
[data-baseweb="popover"] [role="grid"] {
    color: #EAF2FC !important;
}

[data-baseweb="calendar"] {
    background: #192943 !important;
}

[data-baseweb="calendar"] button {
    color: #EAF2FC !important;
}

[data-baseweb="calendar"] button:hover {
    background: #304967 !important;
}

/* NUMBER INPUT */

[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #203653 !important;
    color: #DCEBFF !important;
    border: 1px solid #304967 !important;
}

/* HEADINGS AND TEXT */

.main-title {
    font-size: 39px;
    font-weight: 800;
    color: #F3F7FF;
    letter-spacing: -1px;
}

.subtitle {
    color: #A9BCD3;
    font-size: 15px;
    margin-top: 5px;
}

.section-title {
    color: #DCEBFF;
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 15px;
}

.stMarkdown,
.stText,
p,
li {
    color: inherit;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #A9BCD3;
}

/* METRIC CARDS */

.metric-box {
    background: linear-gradient(
        145deg,
        #192943,
        #203653
    );
    border: 1px solid #304967;
    padding: 22px;
    border-radius: 16px;
    min-height: 135px;
    box-shadow: var(--shadow);
    transition: all 0.25s ease;
}

.metric-box:hover {
    background: #203653;
    border-color: #4385C5;
    transform: translateY(-3px);
    box-shadow: 0 13px 30px rgba(0, 0, 0, 0.3);
}

.metric-label {
    color: #A9BCD3;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.7px;
}

.metric-value {
    color: #F3F7FF;
    font-size: 27px;
    font-weight: 750;
    margin-top: 10px;
}

.metric-note {
    color: #A9BCD3;
    font-size: 12px;
    margin-top: 6px;
}

/* RISK STATUS */

.status-high {
    background: #492735;
    color: #FFB8C5;
    border: 1px solid #824052;
    padding: 15px;
    border-radius: 12px;
    font-weight: 700;
}

.status-normal {
    background: #193C37;
    color: #9AE6C8;
    border: 1px solid #286B5B;
    padding: 15px;
    border-radius: 12px;
    font-weight: 700;
}

/* GENERAL BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        110deg,
        #4385C5,
        #5B9BDB
    );
    color: #FFFFFF !important;
    border: 1px solid #4385C5;
    border-radius: 11px;
    min-height: 46px;
    font-weight: 750;
    box-shadow: 0 5px 16px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(
        110deg,
        #2D659B,
        #4385C5
    );
    border-color: #5B9BDB;
    color: #FFFFFF !important;
    transform: translateY(-1px);
}

/* READY TO ANALYZE BUTTON */

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: #4385C5 !important;
    color: #FFFFFF !important;
    border: 1px solid #5B9BDB !important;
    font-weight: 750 !important;
}

.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background: #2D659B !important;
    color: #FFFFFF !important;
    border-color: #5B9BDB !important;
}

/* MAIN INPUTS */

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTextArea textarea,
[data-testid="stChatInput"] textarea {
    background: #192943 !important;
    color: #EAF2FC !important;
    border: 1px solid #304967 !important;
    border-radius: 10px !important;
    -webkit-text-fill-color: #EAF2FC !important;
}

[data-baseweb="select"] > div {
    background: #192943 !important;
    color: #EAF2FC !important;
    border-color: #304967 !important;
    border-radius: 10px !important;
}

input::placeholder,
textarea::placeholder {
    color: #91A8C2 !important;
    -webkit-text-fill-color: #91A8C2 !important;
}

/* SLIDERS AND TOGGLES */

[data-testid="stSlider"] [role="slider"] {
    background: #5B9BDB !important;
    border-color: #5B9BDB !important;
}

[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
    color: #C5D6EA !important;
}

/* EXPANDERS, METRICS AND CHAT */

[data-testid="stExpander"],
[data-testid="stMetric"],
[data-testid="stChatMessage"] {
    background: #192943 !important;
    border: 1px solid #304967 !important;
    border-radius: 13px !important;
}

[data-testid="stMetric"] {
    padding: 15px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #EAF2FC !important;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* CHATBOT FLOATING BUTTON */

div[data-testid="stPopover"] {
    position: fixed !important;
    right: 26px !important;
    bottom: 22px !important;
    z-index: 999999 !important;
    width: auto !important;
}

div[data-testid="stPopover"] button {
    background: #304967 !important;
    background-color: #304967 !important;
    color: #DCEBFF !important;
    border: 1px solid #456487 !important;
    border-radius: 999px !important;
    box-shadow: 0 5px 16px rgba(0, 0, 0, 0.3) !important;
    font-weight: 800 !important;
}

div[data-testid="stPopover"] button svg {
    color: #DCEBFF !important;
    stroke: #DCEBFF !important;
    fill: none !important;
}

div[data-testid="stPopover"] button p,
div[data-testid="stPopover"] button span {
    color: #DCEBFF !important;
}

div[data-testid="stPopover"] button:hover {
    background: #4385C5 !important;
    color: #FFFFFF !important;
}

div[data-testid="stPopover"] [data-testid="stPopoverBody"] {
    background: #192943 !important;
    color: #EAF2FC !important;
    border: 1px solid #304967 !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
}

/* CHAT INPUT */

[data-testid="stChatInput"] {
    background: #192943 !important;
    border-color: #304967 !important;
}

[data-testid="stChatInput"] textarea {
    color: #EAF2FC !important;
}

/* DIVIDERS AND SCROLLBAR */

hr {
    border-color: #304967 !important;
}

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #111D30;
}

::-webkit-scrollbar-thumb {
    background: #456487;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #5B9BDB;
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_sources" not in st.session_state:
    st.session_state.chat_sources = {}
if "chat_chunks" not in st.session_state:
    st.session_state.chat_chunks = None
if "chat_embeddings" not in st.session_state:
    st.session_state.chat_embeddings = None

# ==========================================
# 4. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">⚡ SmartGrid AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Electricity Demand Forecasting '
    '& Consumer Advisory Platform'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# Centered primary action, placed near the top of the dashboard.
st.markdown(
    '<div style="text-align:center;color:#A9BCD3;font-size:12px;'
    'font-weight:700;letter-spacing:1px;margin-bottom:8px;">READY TO ANALYZE</div>',
    unsafe_allow_html=True
)
left, center, right = st.columns([1.3, 1.4, 1.3])
with center:
    predict_button = st.button(
        "⚡ Generate Prediction",
        type="primary",
        use_container_width=True
    )
st.write("")


# ==========================================
# 5. SIDEBAR INPUTS
# ==========================================

st.sidebar.title("⚙️ Prediction Settings")

st.sidebar.caption(
    "Configure location and weather conditions."
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



# ==========================================
# 6. INPUT VALIDATION AND PREDICTION
# ==========================================

if predict_button:

    if temperature_min > temperature_mean:

        st.error(
            "Minimum temperature cannot exceed average temperature."
        )

    elif temperature_mean > temperature_max:

        st.error(
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
                "Running ML models, retrieving PDF knowledge and generating AI advisory..."
            ):

                result = run_smartgrid(input_data)

            st.session_state.prediction_result = result
            st.session_state.last_input = input_data

            st.success(
                "Prediction completed successfully!"
            )

        except Exception as e:

            st.error("Prediction could not be completed.")
            st.exception(e)


# ==========================================
# 7. DASHBOARD DISPLAY
# ==========================================

result = st.session_state.prediction_result
input_data = st.session_state.last_input


if result is None:

    st.markdown(
        '<div class="section-title">'
        '📍Š Energy Intelligence Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Configure the location and weather conditions "
        "in the sidebar to generate an energy demand prediction."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">MACHINE LEARNING</div>
                <div class="metric-value">Demand Forecast</div>
                <div class="metric-note">Electricity usage prediction</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">RISK ANALYTICS</div>
                <div class="metric-value">Risk Detection</div>
                <div class="metric-note">High usage and severity estimation</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="metric-box">
                <div class="metric-label">GENERATIVE AI</div>
                <div class="metric-value">RAG + Groq LLM</div>
                <div class="metric-note">Intelligent energy advisory</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Your energy intelligence dashboard is ready. "
        "Enter prediction conditions to get started."
    )


else:

    # ======================================
    # 8. PREDICTION OVERVIEW
    # ======================================

    st.markdown(
        '<div class="section-title">📍ˆ Prediction Overview</div>',
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
                    Model prediction
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
                    Model-defined usage threshold
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # ======================================
    # 9. OUTAGE IMPACT AND SEVERITY
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '⚡ Outage Impact & Severity Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Expected Customers Affected",
            f"{result['expected_customers']:,.0f}"
        )

    with col2:

        st.metric(
            "Expected Outage Duration",
            f"{result['expected_duration']:.2f}"
        )

    with col3:

        st.metric(
            "Severity Score",
            f"{result['severity_score']:.2f}%"
        )

    with col4:

        st.metric(
            "Severity Level",
            result['severity_level']
        )

    # Severity status
    severity_level = result["severity_level"]

    if severity_level in ["Critical", "High"]:

        st.error(
            f"⚠️ {severity_level.upper()} SEVERITY ESTIMATED"
        )

    elif severity_level == "Medium":

        st.warning(
            "⚠️ MEDIUM SEVERITY ESTIMATED"
        )

    else:

        st.success(
            "NORMAL / LOW SEVERITY ESTIMATED"
        )

    # ======================================
    # 10. RISK ASSESSMENT
    # ======================================

    st.divider()

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
            "The model has identified conditions associated "
            "with elevated electricity usage."
        )

    else:

        st.markdown(
            '<div class="status-normal">'
            '✅ NORMAL USAGE CLASSIFICATION'
            '</div>',
            unsafe_allow_html=True
        )

        st.success(
            "The model has classified the input as normal usage."
        )

    st.write("")

    # ======================================
    # 11. USAGE COMPARISON
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">📍Š Usage vs Threshold</div>',
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
    # 12. WEATHER SUMMARY
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
    # 13. AI-POWERED GRID ADVISORY
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
    # 14. KNOWLEDGE RETRIEVAL SOURCES
    # ======================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📍š Knowledge Retrieval Sources'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Reference documents retrieved to support "
        "the AI-generated advisory."
    )

    for index, source in enumerate(
        result["sources"],
        start=1
    ):

        with st.expander(
            f"📍„ Source {index}: {source['document']}"
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
                "**Similarity Score:**",
                f"{source['similarity_score']:.4f}"
            )


# ==========================================
# 15. FOOTER
# ==========================================

st.divider()

st.caption(
    "⚡ SmartGrid AI | Intelligent Energy Forecasting "
    "& Consumer Advisory"
)

st.caption(
    "© 2026 SmartGrid AI. All rights reserved."
)

# ==========================================
# 16. FLOATING AI CHATBOT - AGENTIC AI
# ==========================================

from rag.chatbot import initialize_chatbot
from agent.smartgrid_agent import run_smartgrid_agent


with st.popover(
    "💬 Chat with AI",
    help="Open the SmartGrid AI Assistant"
):

    st.markdown("### 🤖 SmartGrid AI Agent")

    st.caption(
        "Ask about electricity demand, grid stress, "
        "outages, weather impacts and consumer safety."
    )

    # --------------------------------------
    # INITIALIZE KNOWLEDGE BASE
    # --------------------------------------

    if (
        st.session_state.chat_chunks is None
        or st.session_state.chat_embeddings is None
    ):

        with st.spinner("Preparing SmartGrid knowledge base..."):

            (
                st.session_state.chat_chunks,
                st.session_state.chat_embeddings
            ) = initialize_chatbot()


    # --------------------------------------
    # CHAT HEADER AND CLEAR BUTTON
    # --------------------------------------

    header_col, clear_col = st.columns([3, 1])

    with header_col:

        st.caption("Powered by Agentic AI + ML + RAG + Groq")

    with clear_col:

        if st.button(
            "Clear",
            key="floating_clear_chat"
        ):

            st.session_state.chat_history = []
            st.session_state.chat_sources = {}

            st.rerun()


    # --------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------

    chat_area = st.container(height=350)

    with chat_area:

        if not st.session_state.chat_history:

            st.info(
                "Try asking: What causes electricity "
                "grid stress in Texas?"
            )


        for message_index, message in enumerate(
            st.session_state.chat_history
        ):

            with st.chat_message(message["role"]):

                st.markdown(message["content"])


                # DISPLAY SOURCES FOR ASSISTANT
                if message["role"] == "assistant":

                    refs = st.session_state.chat_sources.get(
                        message_index,
                        []
                    )

                    if refs:

                        with st.expander("📚 Retrieved PDF Sources"):

                            for number, ref in enumerate(
                                refs,
                                start=1
                            ):

                                st.markdown(
                                    f"**{number}. {ref.get('document', 'Unknown document')}**"
                                )

                                page = ref.get("page", "N/A")
                                score = ref.get("similarity_score")

                                if score is not None:
                                    st.caption(
                                        f"Page {page} | "
                                        f"Similarity: {float(score):.4f}"
                                    )
                                else:
                                    st.caption(f"Page {page}")

                    # DISPLAY AGENT TOOL ACTIONS
                    actions = message.get("actions", [])

                    if actions:
                        with st.expander("⚙️ Agent Actions"):
                            for action in actions:
                                st.write("✓", action)


    # --------------------------------------
    # CHAT INPUT
    # --------------------------------------

    with st.form(
        "floating_chat_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Your question",
            placeholder="Ask SmartGrid AI...",
            label_visibility="collapsed"
        )

        send_question = st.form_submit_button(
            "Send message",
            use_container_width=True
        )


    # --------------------------------------
    # RUN AGENTIC AI
    # --------------------------------------

    if send_question and question.strip():

        with st.spinner(
            "SmartGrid AI Agent is thinking..."
        ):

            try:

                # Keep previous conversation for the agent
                previous_history = list(
                    st.session_state.chat_history
                )


                # Run the agent
                result = run_smartgrid_agent(

                    question=question.strip(),

                    chunks=st.session_state.chat_chunks,

                    embeddings=st.session_state.chat_embeddings,

                    # Latest dashboard inputs
                    prediction_input=st.session_state.last_input,

                    # Previous conversation
                    chat_history=previous_history
                )


                # ----------------------------------
                # SAVE USER MESSAGE
                # ----------------------------------

                st.session_state.chat_history.append({

                    "role": "user",

                    "content": question.strip()

                })


                # ----------------------------------
                # SAVE ASSISTANT RESPONSE
                # ----------------------------------

                answer = result.get(
                    "answer",
                    "Sorry, I could not generate a response."
                )

                st.session_state.chat_history.append({

                    "role": "assistant",

                    "content": answer

                })


                # ----------------------------------
                # SAVE SOURCES
                # ----------------------------------

                assistant_index = (
                    len(st.session_state.chat_history) - 1
                )

                st.session_state.chat_sources[
                    assistant_index
                ] = result.get("sources", [])


                # ----------------------------------
                # DISPLAY AGENT ACTIONS
                # ----------------------------------

                actions = result.get("actions", [])

                if actions:

                    st.session_state.chat_history[-1][
                        "actions"
                    ] = actions


                # Refresh chatbot
                st.rerun()


            except Exception as exc:

                st.error(
                    "The SmartGrid AI Agent could not "
                    "process this question."
                )

                st.exception(exc)
