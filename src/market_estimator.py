import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

INDUSTRY_BASE_DEMAND = {
    'Technology': 9,
    'Healthcare': 9,
    'Retail': 7,
    'Manufacturing': 8,
    'Education': 8,
    'Hospitality': 7,
    'Construction': 8,
    'Logistics': 8,
    'Finance': 8,
    'Food & Beverage': 7,
    'Real Estate': 7,
    'Professional Services': 8,
}

KSA_CITY_ADJUSTMENT = {
    'Riyadh': 1,
    'Jeddah': 1,
    'Dammam': 1,
    'Mecca': 0,
    'Medina': 0,
    'Khobar': 0,
    'Tabuk': -1,
    'Abha': -1,
    'Hail': -1,
    'Najran': -1,
}

DEFAULT_INDUSTRY_DEMAND = 7
DEFAULT_CITY_ADJUSTMENT = -1


def estimate_market_demand(industry: str, city: str, country: str) -> int:
    base = INDUSTRY_BASE_DEMAND.get(industry, DEFAULT_INDUSTRY_DEMAND)

    if country == 'Saudi Arabia':
        adjustment = KSA_CITY_ADJUSTMENT.get(city, DEFAULT_CITY_ADJUSTMENT)
    else:
        adjustment = 0

    score = base + adjustment
    score = max(1, min(10, score))
    return score


if __name__ == "__main__":
    tests = [
        ('Technology', 'Riyadh', 'Saudi Arabia'),
        ('Retail', 'Abha', 'Saudi Arabia'),
        ('Healthcare', 'Jeddah', 'Saudi Arabia'),
        ('Logistics', 'Najran', 'Saudi Arabia'),
    ]
    for industry, city, country in tests:
        score = estimate_market_demand(industry, city, country)
        print(f"{industry} in {city}: market_demand_score = {score}")
