# Restaurant Order Cancellation Prediction System

A machine learning web app that predicts whether a restaurant order is likely to be **cancelled**, so restaurant managers can spot high-risk orders early and act — confirm with the customer, prioritise preparation or adjust staffing. A trained **Random Forest** model is served through an interactive **Streamlit** dashboard.

---

## Table of contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Model inputs](#model-inputs)
4. [Tech stack](#tech-stack)
5. [Project structure](#project-structure)
6. [Prerequisites](#prerequisites)
7. [Run locally](#run-locally)
8. [Using the app](#using-the-app)
9. [Retraining or replacing the model](#retraining-or-replacing-the-model)
10. [Deploy to Streamlit Community Cloud](#deploy-to-streamlit-community-cloud)
11. [Troubleshooting](#troubleshooting)
12. [Author](#author)

---

## Features

- Enter an order's details in a clean, styled form
- One click on **Predict Cancellation Risk** returns:
  - the prediction — **likely to be cancelled** or **likely to be completed**
  - the cancellation **probability**
- **Order analysis insights** that explain the risk factors:
  - order value (low / medium / high)
  - customer rating (poor / average / excellent)
  - cancellation history (none / some / high)
- Behind the model: data cleaning and preprocessing, exploratory data analysis, feature engineering, model training and evaluation, and visualisations of key insights

## How it works

```
Order details (form)
      │
      ▼
Build feature row ── one-hot encode delivery type and day of week
      │              fill any other training columns with 0
      ▼              reorder to match model_columns.pkl
Random Forest (random_forest_order_cancellation.pkl)
      │
      ▼
prediction (0 / 1) + probability of cancellation  →  result card + insights
```

## Model inputs

| Input | Type / range |
|---|---|
| Order amount ($) | number ≥ 0 |
| Number of items | 1 – 50 |
| Distance (km) | 0 – 50 |
| Customer rating | 1.0 – 5.0 |
| Previous cancellations | 0 – 10 |
| Delivery duration (minutes) | 1 – 200 |
| Delivery type | home delivery / pick-up |
| Day of week | Mon – Sun |

## Tech stack

| Part | Technology |
|---|---|
| Language | Python 3 |
| Web app | Streamlit |
| Data | pandas, NumPy |
| Machine learning | scikit-learn (Random Forest), XGBoost (explored during training) |
| Model storage | joblib |
| Visualisation | Matplotlib, Seaborn |

## Project structure

```
restaurant_orders_cancellation_prediction/
├── app.py                                   # Streamlit dashboard
├── random_forest_order_cancellation.pkl     # Trained Random Forest model (~47 MB)
├── model_columns.pkl                        # Feature columns in training order
├── requirements.txt                         # Python dependencies
└── README.md
```

## Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.9 – 3.12 | `python --version` |
| pip | comes with Python | `pip --version` |
| Git | any | `git --version` |

> The model was saved with scikit-learn. Use a scikit-learn version close to the one used for training, otherwise loading the `.pkl` file can fail or print version warnings.

## Run locally

```bash
# 1. Clone
git clone https://github.com/amashi2003/restaurant_orders_cancellation_prediction.git
cd restaurant_orders_cancellation_prediction

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

Streamlit opens **http://localhost:8501** in your browser.

## Using the app

1. Fill in the order amount, number of items, distance, customer rating, previous cancellations and delivery duration.
2. Choose the delivery type and day of the week.
3. Click **🚀 Predict Cancellation Risk**.
4. Read the result card (with probability) and the insights below it.

## Retraining or replacing the model

1. Train a new model in a notebook using the same feature engineering (one-hot encoding of `delivery_type` and `day_of_week`).
2. Save both the model and the list of training columns:

   ```python
   import joblib
   joblib.dump(model, "random_forest_order_cancellation.pkl")
   joblib.dump(list(X_train.columns), "model_columns.pkl")
   ```

3. Replace the two `.pkl` files and restart `streamlit run app.py`.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub (the model file is ~47 MB, under GitHub's 100 MB limit).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → choose this repo, branch `main`, file `app.py`.
3. Click **Deploy**. Dependencies are installed from `requirements.txt`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment, or run `python -m streamlit run app.py` |
| `FileNotFoundError: random_forest_order_cancellation.pkl` | Run the command from the project folder |
| `InconsistentVersionWarning` or unpickling errors | Install the scikit-learn version used for training, e.g. `pip install "scikit-learn==1.x.y"` |
| `KeyError` on columns | `model_columns.pkl` doesn't match the model — save both from the same training run |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

## Author

Amashi Vithanage — IT23145948
SLIIT, BSc in IT Specializing in Data Science
