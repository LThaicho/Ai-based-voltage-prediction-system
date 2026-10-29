"""Safe synthetic ML voltage condition demonstration."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

NOMINAL, LOW, HIGH = 230.0, 207.0, 253.0

def label(v):
    return "undervoltage" if v < LOW else "overvoltage" if v > HIGH else "normal"

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    normal = rng.normal(230, 6, 1200)
    low = rng.normal(196, 5, 450)
    high = rng.normal(264, 5, 450)
    voltage = np.clip(np.concatenate([normal, low, high]), 180, 280)
    data = pd.DataFrame({"voltage_v": voltage, "current_a": rng.uniform(1,18,len(voltage)), "power_factor": rng.uniform(.75,1,len(voltage))})
    data["deviation_v"] = data.voltage_v - NOMINAL
    data["condition"] = data.voltage_v.map(label)
    features = ["voltage_v", "current_a", "power_factor", "deviation_v"]
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(data[features], data.condition)
    return model, features

st.set_page_config(page_title="AI Voltage Monitor", page_icon="⚡")
st.title("AI Voltage Condition Monitor")
st.warning("Software-only demonstration using simulated data. Never connect this project to mains electricity or use it as protection equipment.")
v = st.slider("Simulated voltage (V)", 180.0, 280.0, 230.0, .1)
i = st.slider("Simulated current (A)", 0.0, 30.0, 8.0, .1)
pf = st.slider("Power factor", .50, 1.00, .95, .01)
model, features = train_model()
row = pd.DataFrame([[v, i, pf, v-NOMINAL]], columns=features)
prediction = model.predict(row)[0]
confidence = model.predict_proba(row)[0].max()
st.metric("Predicted condition", prediction.title(), f"{confidence:.0%} model confidence")
st.caption("Labels: below 207 V = undervoltage, 207-253 V = normal, above 253 V = overvoltage")
if prediction == "normal": st.success("Normal simulated condition")
else: st.error(f"Alert: {prediction.title()} detected")
