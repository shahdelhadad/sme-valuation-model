import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
LOGS_DIR = os.path.join(PROJECT_ROOT, "data")

DATA_PATH = os.path.join(DATA_DIR, "sme_valuation_ksa_dataset.csv")
DATA_PATH_ORIGINAL = os.path.join(DATA_DIR, "sme_valuation_mixed_dataset.csv")

MODEL_PKL_PATH = os.path.join(MODELS_DIR, "trained_model.pkl")
MODEL_ONNX_PATH = os.path.join(MODELS_DIR, "ksa_valuation_model.onnx")

FEATURE_ORDER_PATH = os.path.join(CONFIG_DIR, "feature_order.json")
FEATURE_IMPORTANCE_PATH = os.path.join(CONFIG_DIR, "feature_importance.json")
VALIDATION_RULES_PATH = os.path.join(CONFIG_DIR, "validation_rules.json")
REGION_CONFIG_PATH = os.path.join(CONFIG_DIR, "region_config.json")
CURRENCY_CONFIG_PATH = os.path.join(CONFIG_DIR, "currency_config.json")

MISSING_FEATURES_LOG = os.path.join(LOGS_DIR, "missing_feature_logs.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 300
MAX_DEPTH = 12

TARGET_COL = 'valuation_sar'
DROP_COLS = ['business_id']

CATEGORICAL_FEATURES = ['industry', 'city', 'country']

NUMERICAL_FEATURES = [
    'years_in_operation', 'number_of_employees', 'annual_revenue_sar',
    'annual_expenses_sar', 'annual_profit_sar', 'profit_margin',
    'revenue_growth_rate', 'customer_growth_rate', 'monthly_customers',
    'market_demand_score', 'competition_level', 'owner_dependency_score',
    'scalability_score', 'location_score', 'revenue_stability_score'
]

OPTIONAL_FEATURES = ['location_score', 'customer_growth_rate', 'revenue_stability_score']

OPTIONAL_DEFAULTS = {
    'location_score': 5,
    'customer_growth_rate': 0.0,
    'revenue_stability_score': 6
}

AUTO_COMPUTED_FEATURES = ['market_demand_score']

ALL_FEATURES_ORDERED = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

ALLOWED_INDUSTRIES = [
    'Technology', 'Healthcare', 'Retail', 'Manufacturing',
    'Education', 'Hospitality', 'Construction', 'Logistics',
    'Finance', 'Food & Beverage', 'Real Estate', 'Professional Services'
]

ALLOWED_COUNTRIES = [
    'Saudi Arabia', 'United Arab Emirates', 'Qatar',
    'Kuwait', 'Bahrain', 'Oman'
]

KSA_CITIES = [
    'Riyadh', 'Jeddah', 'Dammam', 'Mecca', 'Medina',
    'Khobar', 'Tabuk', 'Abha', 'Hail', 'Najran'
]

ALLOWED_CITIES_BY_COUNTRY = {
    'Saudi Arabia': KSA_CITIES,
    'United Arab Emirates': [],
    'Qatar': [],
    'Kuwait': [],
    'Bahrain': [],
    'Oman': []
}
