# AI-Based Overvoltage and Undervoltage Prediction System

## Project Overview

This project uses machine learning to classify electrical voltage readings into three conditions:

- **Normal:** Voltage is within the defined operating range.
- **Undervoltage:** Voltage is below 207 V.
- **Overvoltage:** Voltage is above 253 V.

The system uses a Random Forest classifier trained on a synthetic dataset. A Streamlit dashboard allows users to enter simulated electrical parameters, view the predicted voltage condition, and examine class probabilities and threshold-based alerts.

**Note:** This is a software demonstration using simulated data. It is not connected to live electrical equipment and must not be used as a real-world protection system.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Matplotlib and Seaborn
- Streamlit

## Project Structure

```text
Ai-based-voltage-prediction-system/
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── voltage_data.csv
└── models/
    └── voltage_model.pkl
```

The project also generates model evaluation results and a confusion matrix during training.

## Installation

Install Python 3.10 or later, then clone this repository:

```bash
git clone https://github.com/LThaicho/Ai-based-voltage-prediction-system.git
cd Ai-based-voltage-prediction-system
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

If the dataset or trained model needs to be regenerated, run:

```bash
python generate_dataset.py
python train_model.py
```

Start the dashboard:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Model Evaluation

The Random Forest classifier is evaluated using a held-out test dataset. The training script generates evaluation metrics and a confusion matrix.

The current test accuracy is approximately 99.83% on the synthetic dataset. This result does not establish equivalent performance on real electrical measurements because the dataset's labels are generated using predefined voltage thresholds.

## Future Improvements

- Train and validate the model using real electrical measurement data.
- Add time-series forecasting to estimate impending voltage deviations.
- Integrate supported sensors for live monitoring.
- Add configurable alert thresholds and more extensive testing.

## Author

Developed as an academic machine-learning project.
