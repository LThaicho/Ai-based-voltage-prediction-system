
from pathlib import Path

import numpy as np
import pandas as pd

NOMINAL_VOLTAGE = 230.0
LOW_THRESHOLD = 207.0
HIGH_THRESHOLD = 253.0
SAMPLES = 3000


def classify_voltage(voltage):
    if voltage < LOW_THRESHOLD:
        return "undervoltage"
    if voltage > HIGH_THRESHOLD:
        return "overvoltage"
    return "normal"


def main():
    rng = np.random.default_rng(42)

    # Generate voltage readings for three operating conditions.
    normal = rng.normal(230, 6, 1800)
    low = rng.normal(196, 5, 600)
    high = rng.normal(264, 5, 600)

    voltage = np.clip(
        np.concatenate([normal, low, high]), 180, 280
    )
    rng.shuffle(voltage)

    current = rng.uniform(1, 18, SAMPLES)
    power_factor = rng.uniform(0.75, 1.0, SAMPLES)

    # Simulate previous readings for trend analysis.
    previous_voltage = voltage + rng.normal(0, 2, SAMPLES)
    voltage_change = voltage - previous_voltage

    data = pd.DataFrame({
        "voltage_v": voltage,
        "current_a": current,
        "power_factor": power_factor,
        "previous_voltage_v": previous_voltage,
        "voltage_change_v": voltage_change,
    })

    data["deviation_v"] = data["voltage_v"] - NOMINAL_VOLTAGE
    data["condition"] = data["voltage_v"].map(classify_voltage)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "voltage_data.csv"

    data.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Total readings: {len(data)}")
    print("\nCondition counts:")
    print(data["condition"].value_counts())
    print("\nFirst five rows:")
    print(data.head())


if __name__ == "__main__":
    main()
