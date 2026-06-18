from joblib import load
import streamlit as st
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.main-title {
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:bold;
}

.sub-title {
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}

.metric-card {
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    box-shadow:0px 0px 10px rgba(255,255,255,0.08);
}

.result-success {
    background:#16a34a;
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.result-danger {
    background:#dc2626;
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

div[data-testid="stSidebar"] {
    background:#111827;
}

.stButton > button {
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    background:linear-gradient(90deg,#ef4444,#f97316);
    color:white;
}

.stButton > button:hover {
    transform:scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
try:
    model = load("knn_heart_model.pkl")
    scaler = load("heart_scaler.pkl")
    expected_columns = load("heart_columns.pkl")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.info(
        "Make sure knn_heart_model.pkl, heart_scaler.pkl and "
        "heart_columns.pkl are present in your repository."
    )
    st.stop()

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-Powered Heart Health Risk Assessment</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- INFO CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
    <h3>🩺 AI Model</h3>
    <p>KNN Classifier</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
    <h3>📊 Analysis</h3>
    <p>11 Health Parameters</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
    <h3>⚡ Prediction</h3>
    <p>Instant Results</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📝 Patient Information")

age = st.sidebar.slider("Age", 18, 100, 40)

sex = st.sidebar.selectbox(
    "Gender",
    ["M", "F"]
)

chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.sidebar.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar >120",
    [0, 1]
)

resting_ecg = st.sidebar.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.sidebar.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.sidebar.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)

oldpeak = st.sidebar.slider(
    "Oldpeak",
    0.0,
    6.0,
    1.0
)

st_slope = st.sidebar.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# ---------------- SUMMARY ----------------
left, right = st.columns([2, 1])

with left:
    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Age",
            "Gender",
            "Blood Pressure",
            "Cholesterol",
            "Maximum HR"
        ],
        "Value": [
            age,
            sex,
            resting_bp,
            cholesterol,
            max_hr
        ]
    })

    st.dataframe(summary, use_container_width=True)

with right:
    st.subheader("📈 Quick Stats")
    st.metric("Age", age)
    st.metric("Max HR", max_hr)
    st.metric("Cholesterol", cholesterol)

st.write("")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Heart Disease Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    with st.spinner("🔍 Analyzing patient data..."):
        time.sleep(1)
        prediction = model.predict(scaled_input)[0]

    st.write("")

    if prediction == 1:
        st.markdown("""
        <div class="result-danger">
        ⚠️ HIGH RISK OF HEART DISEASE
        </div>
        """, unsafe_allow_html=True)

        st.error(
            "Please consult a healthcare professional for further medical evaluation."
        )

    else:
        st.markdown("""
        <div class="result-success">
        ✅ LOW RISK OF HEART DISEASE
        </div>
        """, unsafe_allow_html=True)

        st.success(
            "Heart health indicators appear normal."
        )

# ---------------- FOOTER ----------------
st.write("---")
st.markdown(
    "<center style='color:lightgray;'>❤️ Developed by Mitul Sai</center>",
    unsafe_allow_html=True
)
