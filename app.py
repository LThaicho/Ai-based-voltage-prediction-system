
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

NOMINAL = 230.0
LOW = 207.0
HIGH = 253.0

MODEL_PATH = Path("models/voltage_model.pkl")

st.set_page_config(
    page_title="AI Voltage Monitor",
    page_icon="⚡",
    layout="centered",
)

st.title("⚡ AI-Based Voltage Condition Monitoring")
st.caption("Machine-learning demonstration using simulated electrical data.")

st.warning(
    "Simulation only. Do not connect this application to mains voltage "
    "or use it as a real electrical protection system."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

st.subheader("Simulated electrical inputs")

voltage = st.slider("Voltage (V)", 180.0, 280.0, 230.0, 0.5)
current = st.slider("Current (A)", 0.0, 30.0, 8.0, 0.1)
power_factor = st.slider("Power factor", 0.50, 1.00, 0.95, 0.01)
previous_voltage = st.slider(
    "Previous voltage reading (V)", 180.0, 280.0, 230.0, 0.5
)

voltage_change = voltage - previous_voltage
deviation = voltage - NOMINAL

row = pd.DataFrame(
    [[
        voltage,
        current,
        power_factor,
        previous_voltage,
        voltage_change,
        deviation,
    ]],
    columns=[
        "voltage_v",
        "current_a",
        "power_factor",
        "previous_voltage_v",
        "voltage_change_v",
        "deviation_v",
    ],
)

row = row[features]
prediction = model.predict(row)[0]
probabilities = model.predict_proba(row)[0]
confidence = probabilities.max()

st.divider()
st.subheader("Model result")

col1, col2 = st.columns(2)
col1.metric("Predicted condition", prediction.title())
col2.metric("Model confidence", f"{confidence:.1%}")

if voltage < LOW:
    st.error("UNDERVOLTAGE: Voltage is below 207 V.")
elif voltage > HIGH:
    st.error("OVERVOLTAGE: Voltage is above 253 V.")
else:
    st.success("NORMAL: Voltage is within 207–253 V.")

st.caption(
    "Voltage thresholds are illustrative assumptions for this project. "
    "The displayed threshold alert is separate from the ML prediction."
)

st.subheader("Class probabilities")
probability_table = pd.DataFrame({
    "Condition": model.classes_,
    "Probability": probabilities,
})
st.bar_chart(probability_table.set_index("Condition"))

st.subheader("Input summary")
st.dataframe(row, use_container_width=True)
