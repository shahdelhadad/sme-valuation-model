import os
import sys
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.preprocess import load_data, preprocess_data


def export_feature_importance(pipeline, output_path: str):
    xgb_model = pipeline.named_steps['model']
    importances = [float(i) for i in xgb_model.feature_importances_]

    preprocessor = pipeline.named_steps['preprocessor']
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(config.CATEGORICAL_FEATURES)
    num_features = np.array(config.NUMERICAL_FEATURES)
    all_features = list(cat_features) + list(num_features)

    importance_dict = dict(zip(all_features, importances))
    sorted_importance = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sorted_importance, f, indent=2)
    print(f"Feature importance saved to: {output_path}")
    return sorted_importance


def export_feature_order(output_path: str):
    feature_order = {
        "categorical_features": config.CATEGORICAL_FEATURES,
        "numerical_features": config.NUMERICAL_FEATURES,
        "all_features_ordered": config.ALL_FEATURES_ORDERED,
        "required_features": [
            f for f in config.NUMERICAL_FEATURES
            if f not in config.OPTIONAL_FEATURES and f not in config.AUTO_COMPUTED_FEATURES
        ] + config.CATEGORICAL_FEATURES,
        "optional_features": config.OPTIONAL_FEATURES,
        "auto_computed_features": config.AUTO_COMPUTED_FEATURES,
        "optional_defaults": config.OPTIONAL_DEFAULTS,
        "target_column": config.TARGET_COL
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(feature_order, f, indent=2)
    print(f"Feature order saved to: {output_path}")
    return feature_order


def export_validation_rules(output_path: str):
    rules = {
        "annual_revenue_sar": {"type": "float", "minimum": 100000, "required": True},
        "annual_expenses_sar": {"type": "float", "minimum": 0, "note": "must not exceed annual_revenue_sar", "required": True},
        "annual_profit_sar": {"type": "float", "note": "can be negative", "required": True},
        "profit_margin": {"type": "float", "minimum": -0.5, "maximum": 0.6, "required": True},
        "number_of_employees": {"type": "int", "minimum": 1, "required": True},
        "years_in_operation": {"type": "float", "minimum": 0, "required": True},
        "monthly_customers": {"type": "int", "minimum": 10, "required": True},
        "revenue_growth_rate": {"type": "float", "minimum": -0.5, "maximum": 1.0, "required": True},
        "customer_growth_rate": {"type": "float", "minimum": -0.5, "maximum": 1.0, "required": False, "default": 0.0},
        "market_demand_score": {"type": "int", "minimum": 1, "maximum": 10, "required": False, "note": "auto-computed from industry + city + country"},
        "competition_level": {"type": "int", "minimum": 1, "maximum": 10, "required": True},
        "owner_dependency_score": {"type": "int", "minimum": 1, "maximum": 10, "required": True},
        "scalability_score": {"type": "int", "minimum": 1, "maximum": 10, "required": True},
        "location_score": {"type": "int", "minimum": 1, "maximum": 10, "required": False, "default": 5},
        "revenue_stability_score": {"type": "int", "minimum": 1, "maximum": 10, "required": False, "default": 6},
        "industry": {"type": "string", "allowed_values": config.ALLOWED_INDUSTRIES, "required": True},
        "city": {"type": "string", "allowed_values": config.KSA_CITIES, "note": "must match country", "required": True},
        "country": {"type": "string", "allowed_values": config.ALLOWED_COUNTRIES, "required": True, "default": "Saudi Arabia"},
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(rules, f, indent=2)
    print(f"Validation rules saved to: {output_path}")
    return rules


def train_model():
    print("=" * 50)
    print("  KSA SME VALUATION MODEL — TRAINING")
    print("=" * 50)

    print("\nLoading KSA dataset...")
    df = load_data()
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")

    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    print("\nBuilding model pipeline...")
    model = XGBRegressor(
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        objective='reg:squarederror'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    print("\nSetting up hyperparameter tuning for XGBoost...")
    param_grid = {
        'model__n_estimators': [100, 300, 500],
        'model__max_depth': [6, 10, 15],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.8, 1.0],
    }

    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid,
        n_iter=10, cv=3, scoring='neg_mean_absolute_error', 
        random_state=config.RANDOM_STATE, n_jobs=-1, verbose=1
    )

    print("Training and tuning the model... (this may take several minutes)")
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_
    print("Training complete. Best parameters found:")
    print(search.best_params_)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print("\n" + "=" * 50)
    print("  NEW MODEL METRICS")
    print("=" * 50)
    print(f"  R²:        {r2:.4f}")
    print(f"  MAE (SAR): {mae:>15,.2f}")
    print(f"  RMSE(SAR): {rmse:>15,.2f}")
    print(f"  MAPE:      {mape:.2f}%")
    print("=" * 50)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    print(f"\nSaving model to: {config.MODEL_PKL_PATH}")
    joblib.dump(pipeline, config.MODEL_PKL_PATH)
    print("Model saved successfully.")

    os.makedirs(config.CONFIG_DIR, exist_ok=True)
    print("\nExporting feature importance...")
    export_feature_importance(pipeline, config.FEATURE_IMPORTANCE_PATH)

    print("Exporting feature order...")
    export_feature_order(config.FEATURE_ORDER_PATH)

    print("Exporting validation rules...")
    export_validation_rules(config.VALIDATION_RULES_PATH)

    return pipeline


if __name__ == "__main__":
    train_model()
