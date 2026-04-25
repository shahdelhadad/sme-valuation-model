import os
import sys
import random
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

# KSA city distribution: major cities appear 3x more often
KSA_CITY_WEIGHTS = {
    'Riyadh': 25,
    'Jeddah': 22,
    'Dammam': 15,
    'Mecca': 8,
    'Medina': 8,
    'Khobar': 7,
    'Tabuk': 5,
    'Abha': 4,
    'Hail': 3,
    'Najran': 3,
}

# Map legacy industry names from original dataset → standardized KSA industry names
INDUSTRY_MAPPING = {
    # Common patterns in EGY/mixed datasets
    'Tech Startup': 'Technology',
    'Technology': 'Technology',
    'IT Services': 'Technology',
    'Software': 'Technology',
    'Healthcare': 'Healthcare',
    'Medical': 'Healthcare',
    'Clinic': 'Healthcare',
    'Pharmacy': 'Healthcare',
    'Retail': 'Retail',
    'E-commerce': 'Retail',
    'Wholesale': 'Retail',
    'Manufacturing': 'Manufacturing',
    'Factory': 'Manufacturing',
    'Industrial': 'Manufacturing',
    'Education': 'Education',
    'Training': 'Education',
    'School': 'Education',
    'Hospitality': 'Hospitality',
    'Hotel': 'Hospitality',
    'Tourism': 'Hospitality',
    'Restaurant': 'Food & Beverage',
    'Food': 'Food & Beverage',
    'Catering': 'Food & Beverage',
    'Construction': 'Construction',
    'Real Estate': 'Real Estate',
    'Property': 'Real Estate',
    'Logistics': 'Logistics',
    'Transport': 'Logistics',
    'Shipping': 'Logistics',
    'Finance': 'Finance',
    'Banking': 'Finance',
    'Insurance': 'Finance',
    'Consulting': 'Professional Services',
    'Legal': 'Professional Services',
    'Accounting': 'Professional Services',
    'Service Business': 'Professional Services',
    'Services': 'Professional Services',
}

FALLBACK_INDUSTRIES = config.ALLOWED_INDUSTRIES


def map_industry(original: str) -> str:
    for key, standard in INDUSTRY_MAPPING.items():
        if key.lower() in original.lower():
            return standard
    # Hash-based deterministic fallback so the same unmapped name always maps to the same industry
    idx = hash(original) % len(FALLBACK_INDUSTRIES)
    return FALLBACK_INDUSTRIES[idx]


def generate_ksa_city() -> str:
    cities = list(KSA_CITY_WEIGHTS.keys())
    weights = list(KSA_CITY_WEIGHTS.values())
    return random.choices(cities, weights=weights, k=1)[0]


def generate_ksa_dataset(
    input_path: str = config.DATA_PATH_ORIGINAL,
    output_path: str = config.DATA_PATH,
    seed: int = config.RANDOM_STATE
):
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading original dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ── Step 1: Rename currency columns ──────────────────────────────────────
    rename_map = {
        'annual_revenue_egp': 'annual_revenue_sar',
        'annual_expenses_egp': 'annual_expenses_sar',
        'annual_profit_egp': 'annual_profit_sar',
        'valuation_egp': 'valuation_sar',
    }
    df = df.rename(columns=rename_map)
    print("  Renamed EGP columns → SAR")

    # ── Step 2: Map industry names to standardized KSA names ─────────────────
    df['industry'] = df['industry'].apply(map_industry)
    print(f"  Industry distribution:\n{df['industry'].value_counts().to_string()}")

    # ── Step 3: Replace city values with KSA cities ──────────────────────────
    df['city'] = [generate_ksa_city() for _ in range(len(df))]
    print(f"\n  City distribution:\n{df['city'].value_counts().to_string()}")

    # ── Step 4: Add country column (default: Saudi Arabia) ───────────────────
    df['country'] = 'Saudi Arabia'
    print("\n  Added country = 'Saudi Arabia'")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nKSA dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    generate_ksa_dataset()
