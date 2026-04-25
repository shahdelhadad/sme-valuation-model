import os
import sys
import csv
import json
import datetime
import numpy as np
import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.validation import validate_inputs, ValidationError
from src.market_estimator import estimate_market_demand
from src.currency_converter import format_dual_currency


def log_missing_feature(feature: str, default_value, log_path: str = config.MISSING_FEATURES_LOG):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)

    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'feature', 'default_used', 'note'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.datetime.now().isoformat(),
            'feature': feature,
            'default_used': default_value,
            'note': f'Missing optional feature: {feature} -> default used'
        })

    print(f"  [INFO] Missing optional feature: {feature} -> default {default_value} applied")



def fill_optional_defaults(data: dict) -> dict:
    for feature, default in config.OPTIONAL_DEFAULTS.items():
        if feature not in data or data[feature] is None:
            log_missing_feature(feature, default)
            data[feature] = default
    return data



def build_onnx_inputs(input_dict: dict) -> dict:
    inputs_onnx = {}

    for col in config.CATEGORICAL_FEATURES:
        val = input_dict.get(col, '')
        inputs_onnx[col] = np.array([[str(val)]], dtype=object)

    for col in config.NUMERICAL_FEATURES:
        val = input_dict.get(col, 0.0)
        inputs_onnx[col] = np.array([[float(val)]], dtype=np.float32)

    return inputs_onnx



def predict(user_input: dict, model_path: str = config.MODEL_ONNX_PATH) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX model not found at: {model_path}\n"
            "Please run: python src/train.py && python src/export_onnx.py"
        )

    user_input = fill_optional_defaults(user_input.copy())

    validated = validate_inputs(user_input)

    market_demand = estimate_market_demand(
        industry=validated['industry'],
        city=validated['city'],
        country=validated['country']
    )
    validated['market_demand_score'] = market_demand
    print(f"  [INFO] Auto market_demand_score = {market_demand} "
          f"({validated['industry']} in {validated['city']})")

    inputs_onnx = build_onnx_inputs(validated)

    session = ort.InferenceSession(model_path)
    raw_predictions = session.run(None, inputs_onnx)
    valuation_sar = float(raw_predictions[0][0][0])

    result = format_dual_currency(valuation_sar)
    result['market_demand_score_used'] = market_demand

    return result


def demonstrate_inference():
    print("\n" + "=" * 55)
    print("  KSA SME VALUATION MODEL — INFERENCE DEMO")
    print("=" * 55)

    example_input = {
        "country": "Saudi Arabia",
        "industry": "Technology",
        "city": "Riyadh",
        "annual_revenue_sar": 8_500_000,
        "annual_expenses_sar": 5_950_000,
        "annual_profit_sar": 2_550_000,
        "profit_margin": 0.30,
        "number_of_employees": 55,
        "years_in_operation": 7,
        "monthly_customers": 180,
        "revenue_growth_rate": 0.25,
        "competition_level": 6,
        "owner_dependency_score": 4,
        "scalability_score": 8,
        # Optional features intentionally omitted to test defaults:
        # "location_score": <omitted>
        # "customer_growth_rate": <omitted>
        # "revenue_stability_score": <omitted>
    }

    print("\nExample Input:")
    for k, v in example_input.items():
        print(f"  {k}: {v}")

    print("\nRunning inference pipeline...")
    result = predict(example_input)

    print("\n" + "=" * 55)
    print("  PREDICTION RESULT")
    print("=" * 55)
    print(f"  Valuation (SAR): {result['valuation_sar']:>20,.2f}")
    print(f"  Valuation (USD): {result['valuation_usd']:>20,.2f}")
    print(f"  Market Demand Score Used: {result['market_demand_score_used']}")
    print("=" * 55)

    print("\nFull JSON Output:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    demonstrate_inference()