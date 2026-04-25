# SME Valuation Model

A machine learning system for valuing Small and Medium Enterprises (SMEs) in the GCC region, with a primary focus on the Saudi Arabian market. The model predicts business valuations in SAR using an XGBoost regression pipeline and exports to ONNX format for seamless integration with .NET backends.

---

## Features

- **XGBoost Regression** with automated hyperparameter tuning via `RandomizedSearchCV`
- **KSA-first** dataset and feature set with full GCC country support
- **Dual-currency output** (SAR + conversion support)
- **ONNX export** for production deployment in .NET / C# environments
- **Input validation pipeline** with configurable rules per feature
- **Auto-computed features** (e.g. `market_demand_score` derived from industry, city, and country)
- **Optional features** with sensible defaults for missing inputs

---

## Project Structure

```
sme-valuation-model/
├── config/
│   ├── currency_config.json       # SAR conversion rates
│   ├── feature_importance.json    # Model feature importances (auto-generated)
│   ├── feature_order.json         # Feature schema for backend integration
│   ├── region_config.json         # GCC region/city mappings
│   └── validation_rules.json      # Per-field validation rules
├── data/
│   └── sme_valuation_mixed_dataset.csv   # Training dataset
├── models/                        # Saved model artifacts (not tracked by git)
│   ├── trained_model.pkl
│   └── ksa_valuation_model.onnx
├── notebooks/
│   └── exploration.ipynb          # Exploratory data analysis
├── scripts/
│   └── generate_ksa_dataset.py    # Dataset generation script
├── src/
│   ├── config.py                  # Centralized configuration & constants
│   ├── preprocess.py              # Data loading & preprocessing pipeline
│   ├── train.py                   # Model training & hyperparameter tuning
│   ├── evaluate.py                # Model evaluation (R², MAE, RMSE, MAPE)
│   ├── export_onnx.py             # ONNX model export
│   ├── inference.py               # ONNX Runtime inference demo
│   ├── validation.py              # Input validation logic
│   ├── currency_converter.py      # SAR currency conversion utilities
│   └── market_estimator.py        # Market demand score estimation
├── tests/
│   └── test_model.py              # Unit tests
├── requirements.txt
└── README.md
```

---

## Input Features

| Feature | Type | Required | Notes |
|---|---|---|---|
| `industry` | string | ✅ | One of 12 supported industries |
| `city` | string | ✅ | Must match country |
| `country` | string | ✅ | GCC countries supported |
| `annual_revenue_sar` | float | ✅ | Minimum 100,000 SAR |
| `annual_expenses_sar` | float | ✅ | Must not exceed revenue |
| `annual_profit_sar` | float | ✅ | Can be negative |
| `profit_margin` | float | ✅ | Range: -0.5 to 0.6 |
| `years_in_operation` | float | ✅ | |
| `number_of_employees` | int | ✅ | Minimum 1 |
| `monthly_customers` | int | ✅ | Minimum 10 |
| `revenue_growth_rate` | float | ✅ | Range: -0.5 to 1.0 |
| `competition_level` | int | ✅ | Score 1–10 |
| `owner_dependency_score` | int | ✅ | Score 1–10 |
| `scalability_score` | int | ✅ | Score 1–10 |
| `market_demand_score` | int | ⚙️ Auto | Auto-computed from industry, city & country |
| `customer_growth_rate` | float | ➕ Optional | Default: 0.0 |
| `location_score` | int | ➕ Optional | Default: 5 |
| `revenue_stability_score` | int | ➕ Optional | Default: 6 |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shahdelhadad/sme-valuation-model.git
cd sme-valuation-model
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model

Loads the dataset, preprocesses it, runs hyperparameter tuning, and saves the model + config files:

```bash
python src/train.py
```

### Evaluate the model

Prints R², MAE, RMSE, and MAPE on the held-out test set:

```bash
python src/evaluate.py
```

### Export to ONNX

Converts the trained `.pkl` pipeline to an ONNX model for backend deployment:

```bash
python src/export_onnx.py
```

### Run inference (ONNX)

Feeds sample data through the ONNX runtime to verify the exported model:

```bash
python src/inference.py
```

### Run tests

```bash
python -m unittest tests/test_model.py
```

---

## Tech Stack

| Category | Library |
|---|---|
| ML Model | `xgboost`, `scikit-learn` |
| ONNX Export | `skl2onnx`, `onnxmltools`, `onnxconverter-common` |
| ONNX Inference | `onnxruntime` |
| Data | `pandas`, `numpy` |
| Serialization | `joblib` |
| Testing | `pytest` |

---

## Supported Industries

`Technology` · `Healthcare` · `Retail` · `Manufacturing` · `Education` · `Hospitality` · `Construction` · `Logistics` · `Finance` · `Food & Beverage` · `Real Estate` · `Professional Services`

## Supported Countries

`Saudi Arabia` · `United Arab Emirates` · `Qatar` · `Kuwait` · `Bahrain` · `Oman`
