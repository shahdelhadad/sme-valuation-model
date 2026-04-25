import os
import sys
import joblib
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from xgboost import XGBRegressor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config


def export_to_onnx():
    if not os.path.exists(config.MODEL_PKL_PATH):
        print(f"Error: Trained model not found at {config.MODEL_PKL_PATH}")
        print("Please run train.py first.")
        return

    print(f"Loading trained model from: {config.MODEL_PKL_PATH}")
    pipeline = joblib.load(config.MODEL_PKL_PATH)

    update_registered_converter(
        XGBRegressor, 'XGBoostXGBRegressor',
        calculate_linear_regressor_output_shapes, convert_xgboost
    )

    print("Defining ONNX input tensor types...")
    initial_types = []
    for col in config.CATEGORICAL_FEATURES:
        initial_types.append((col, StringTensorType([None, 1])))
    for col in config.NUMERICAL_FEATURES:
        initial_types.append((col, FloatTensorType([None, 1])))

    print("Converting pipeline to ONNX format...")
    onnx_model = convert_sklearn(pipeline, initial_types=initial_types, target_opset={'': 15, 'ai.onnx.ml': 3})

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    print(f"Saving optimized ONNX model to: {config.MODEL_ONNX_PATH}")
    with open(config.MODEL_ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Export successful: {config.MODEL_ONNX_PATH}")
    print(f"  Model size: {os.path.getsize(config.MODEL_ONNX_PATH) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    export_to_onnx()
