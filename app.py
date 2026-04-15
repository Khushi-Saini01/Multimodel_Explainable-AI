# =========================================================
# 🏥 HOSPITAL-GRADE AI DASHBOARD (IIT + CLINICAL UI)
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image
from xgboost import XGBClassifier
import shap
import plotly.graph_objects as go
import datetime

# =========================================================
# CONFIG (HOSPITAL UI STYLE)
# =========================================================
st.set_page_config(
    page_title="AI Hospital System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f5f7fb;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("🏥 AI-Powered Hospital Clinical Decision System")

# =========================================================
# SIDEBAR - DOCTOR CONTROL PANEL
# =========================================================
st.sidebar.markdown("## 🏥 Doctor Control Panel")

doctor_name = st.sidebar.text_input("Doctor Name", "Dr. Sharma")
patient_id = st.sidebar.text_input("Patient ID", "P-1001")

st.sidebar.markdown("---")
st.sidebar.header("🧾 Patient Vitals")

age = st.sidebar.number_input("Age", 1, 100, 50)
bp = st.sidebar.number_input("Blood Pressure", 50, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 400, 200)
sugar = st.sidebar.number_input("Sugar Level", 50, 300, 110)
smoking = st.sidebar.selectbox("Smoking", [0, 1])
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])

st.sidebar.markdown("---")
ecg_file = st.sidebar.file_uploader("📊 ECG File", type=["csv"])
xray_file = st.sidebar.file_uploader("🩻 X-Ray Image", type=["png", "jpg", "jpeg"])

# =========================================================
# TABULAR MODEL
# =========================================================
@st.cache_resource
def train_model():
    df = pd.read_csv("data/heart.csv")
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss"
    )
    model.fit(X, y)
    return model, X.columns

model, feature_cols = train_model()

# =========================================================
# SHAP EXPLAINER
# =========================================================
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# =========================================================
# INPUT BUILDER
# =========================================================
def build_input():
    arr = np.zeros(len(feature_cols))

    mapping = {
        "Age": age,
        "RestingBP": bp,
        "Cholesterol": chol,
        "FastingBS": sugar,
        "Smoking": smoking,
        "Diabetes": diabetes
    }

    for i, c in enumerate(feature_cols):
        if c in mapping:
            arr[i] = mapping[c]

    return arr.reshape(1, -1)

# =========================================================
# ECG + X-RAY MODELS (DEMO)
# =========================================================
@st.cache_resource
def load_ecg():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation="relu", input_shape=(187, 1)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

@st.cache_resource
def load_xray():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(128,128,3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

ecg_model = load_ecg()
xray_model = load_xray()

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🧠 RUN DIAGNOSTIC ANALYSIS"):

    st.markdown("## 📊 Clinical Results Dashboard")

    inp = build_input()
    tab_prob = float(model.predict_proba(inp)[0][1])

    # ECG
    ecg_prob = 0.0
    if ecg_file:
        try:
            ecg = pd.read_csv(ecg_file)
            ecg = ecg.values.reshape(-1, ecg.shape[1], 1)
            ecg_prob = float(ecg_model.predict(ecg[:1])[0][0])
        except:
            ecg_prob = 0.0

    # =====================================================
    # 🩻 X-RAY (ADDED FEATURE)
    # =====================================================
    xray_prob = 0.0
    heatmap_img = None
    boxed_img = None
    original_img = None

    if xray_file:
        try:
            img_pil = Image.open(xray_file).convert("RGB")
            original_img = img_pil

            img = np.array(img_pil)
            img_resized = cv2.resize(img, (128, 128)) / 255.0
            img_input = img_resized.reshape(1,128,128,3)

            xray_prob = float(xray_model.predict(img_input)[0][0])

            # 🔥 HEATMAP
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7,7), 0)

            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            heatmap_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

            # 🔴 INFECTED REGION
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxed_img = img.copy()

            for c in contours:
                if cv2.contourArea(c) > 800:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(boxed_img,(x,y),(x+w,y+h),(0,0,255),2)
                    cv2.putText(boxed_img, "Affected", (x,y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        except:
            pass

    # FINAL FUSION
    weights = np.array([0.5, 0.25, 0.25])
    probs = np.array([tab_prob, ecg_prob, xray_prob])
    final_risk = float(np.dot(weights, probs))

    # METRICS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Patient ID", patient_id)
    col2.metric("Doctor", doctor_name)
    col3.metric("Tabular Risk", f"{tab_prob:.2f}")
    col4.metric("Final Risk", f"{final_risk:.2f}")

    st.markdown("---")

    # GAUGE
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_risk * 100,
        title={'text': "Cardiovascular Risk Level"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 🩻 X-RAY VISUAL COMPARISON (ADDED)
    # =====================================================
    if heatmap_img is not None:

        st.subheader("🩻 X-Ray AI Analysis")

        col1, col2, col3 = st.columns(3)

        col1.image(original_img, caption="🟢 Original", use_container_width=True)
        col2.image(heatmap_img, caption="🔥 AI Heatmap", use_container_width=True)
        col3.image(boxed_img, caption="🔴 Infected Region", use_container_width=True)

        st.info("Heatmap = AI focus | Red box = suspected infection area")

    # STATUS
    if final_risk < 0.3:
        st.success("🟢 LOW RISK PATIENT")
    elif final_risk < 0.7:
        st.warning("🟡 MODERATE RISK")
    else:
        st.error("🔴 HIGH RISK")

    # =====================================================
    # STATUS
    # =====================================================
    if final_risk < 0.3:
        st.success("🟢 LOW RISK PATIENT")
    elif final_risk < 0.7:
        st.warning("🟡 MODERATE RISK")
    else:
        st.error("🔴 HIGH RISK")

    # =====================================================
    # AI EXPLANATION
    # =====================================================
    st.subheader("🧠 AI Clinical Insights")

    if final_risk > 0.7:
        st.error("High cardiovascular risk due to abnormal BP + cholesterol pattern.")
    elif final_risk > 0.3:
        st.warning("Moderate risk detected. Lifestyle modification recommended.")
    else:
        st.success("Normal physiological pattern observed.")

    # =====================================================
    # 📊 SHAP (FULL UPGRADE)
    # =====================================================
    st.subheader("📊 Feature Risk Contribution (SHAP)")

    try:
        df_inp = pd.DataFrame(inp, columns=feature_cols)
        shap_values = explainer.shap_values(df_inp)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_values = shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Impact": shap_values
        }).sort_values("Impact", ascending=False)

        st.bar_chart(shap_df.set_index("Feature"))

        # =====================================================
        # 🧠 MOST IMPORTANT FEATURE
        # =====================================================
        top = shap_df.iloc[0]

        st.markdown("### 🧠 Key Clinical Driver")

        if top["Impact"] > 0:
            st.error(f"🔴 Primary Risk Factor: {top['Feature']}")
            st.write("This feature increases cardiovascular risk significantly.")
        else:
            st.success(f"🟢 Primary Protective Factor: {top['Feature']}")
            st.write("This feature reduces cardiovascular risk.")

        # =====================================================
        # 🔍 TOP 3 FACTORS
        # =====================================================
        st.markdown("### 🔍 Top 3 Clinical Factors")

        for i in range(min(3, len(shap_df))):
            row = shap_df.iloc[i]

            if row["Impact"] > 0:
                st.write(f"🔴 {row['Feature']} → Increases risk")
            else:
                st.write(f"🟢 {row['Feature']} → Reduces risk")

        # =====================================================
        # 🧠 DOCTOR EXPLANATION
        # =====================================================
        st.markdown("### 🧠 AI Medical Explanation")

        explanation = []

        for i in range(min(2, len(shap_df))):
            row = shap_df.iloc[i]
            if row["Impact"] > 0:
                explanation.append(f"high influence of {row['Feature']}")
            else:
                explanation.append(f"protective effect of {row['Feature']}")

        st.info("Risk is mainly due to " + " and ".join(explanation))

    except Exception:
        st.warning("SHAP explanation unavailable")

    # =====================================================
    # REPORT
    # =====================================================
    st.subheader("📁 Medical Report")

    report = f"""
    HOSPITAL AI REPORT
    -------------------------
    Patient ID: {patient_id}
    Doctor: {doctor_name}
    Date: {datetime.datetime.now()}

    Tabular Risk: {tab_prob:.2f}
    ECG Risk: {ecg_prob:.2f}
    X-Ray Risk: {xray_prob:.2f}
    Final Risk: {final_risk:.2f}
    """

    st.download_button(
        "📥 Download Patient Report",
        report,
        file_name=f"{patient_id}_report.txt"
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("🏥 Hospital AI System and  Clinical Decision Support Dashboard")
