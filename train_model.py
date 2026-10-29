
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/voltage_data.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "voltage_model.pkl"
REPORT_PATH = Path("model_evaluation.txt")
PLOT_PATH = Path("confusion_matrix.png")

FEATURES = [
    "voltage_v",
    "current_a",
    "power_factor",
    "previous_voltage_v",
    "voltage_change_v",
    "deviation_v",
]
TARGET = "condition"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run python generate_dataset.py first."
        )

    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test, predictions, zero_division=0
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES},
        MODEL_PATH,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title("Voltage Condition Classification")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation: {REPORT_PATH}")
    print(f"Saved confusion matrix: {PLOT_PATH}")


if __name__ == "__main__":
    main()
