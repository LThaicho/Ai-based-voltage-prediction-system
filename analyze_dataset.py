
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/voltage_data.csv")
OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n--- Dataset Overview ---")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nStatistical summary:")
print(df.describe().round(2))
print("\nCondition distribution:")
print(df["condition"].value_counts())

# Graph 1: Number of readings in each condition
df["condition"].value_counts().reindex(
    ["normal", "undervoltage", "overvoltage"]
).plot(kind="bar")

plt.title("Voltage Condition Distribution")
plt.xlabel("Condition")
plt.ylabel("Number of Readings")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "condition_distribution.png")
plt.close()

# Graph 2: Voltage distribution
plt.figure()
plt.hist(df["voltage_v"], bins=30, edgecolor="black")
plt.title("Distribution of Voltage Readings")
plt.xlabel("Voltage (V)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "voltage_distribution.png")
plt.close()

# Graph 3: Voltage by condition
plt.figure()
df.boxplot(column="voltage_v", by="condition")
plt.title("Voltage by Condition")
plt.suptitle("")
plt.xlabel("Condition")
plt.ylabel("Voltage (V)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "voltage_by_condition.png")
plt.close()

print(f"\nAnalysis complete. Graphs saved in: {OUTPUT_DIR.resolve()}")
