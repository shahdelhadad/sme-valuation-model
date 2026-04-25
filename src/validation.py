import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config


class ValidationError(Exception):
    pass


def validate_annual_revenue_sar(value) -> float:
    v = float(value)
    if v < 100_000:
        raise ValidationError(f"annual_revenue_sar must be >= 100,000 SAR. Got: {v}")
    return v


def validate_annual_expenses_sar(value, revenue_sar: Optional[float] = None) -> float:
    v = float(value)
    if v < 0:
        raise ValidationError(f"annual_expenses_sar must be >= 0. Got: {v}")
    if revenue_sar is not None and v > revenue_sar:
        raise ValidationError(
            f"annual_expenses_sar ({v:,.2f}) must NOT exceed annual_revenue_sar ({revenue_sar:,.2f})"
        )
    return v


def validate_annual_profit_sar(value) -> float:
    return float(value) 


def validate_profit_margin(value) -> float:
    v = float(value)
    if not (-0.5 <= v <= 0.6):
        raise ValidationError(f"profit_margin must be between -0.50 and 0.60. Got: {v}")
    return v


def validate_number_of_employees(value) -> int:
    v = int(value)
    if v < 1:
        raise ValidationError(f"number_of_employees must be >= 1. Got: {v}")
    return v


def validate_years_in_operation(value) -> float:
    v = float(value)
    if v < 0:
        raise ValidationError(f"years_in_operation must be >= 0. Got: {v}")
    return v


def validate_monthly_customers(value) -> int:
    v = int(value)
    if v < 10:
        raise ValidationError(f"monthly_customers must be >= 10. Got: {v}")
    return v



def validate_revenue_growth_rate(value) -> float:
    v = float(value)
    if not (-0.5 <= v <= 1.0):
        raise ValidationError(f"revenue_growth_rate must be between -0.50 and 1.00. Got: {v}")
    return v


def validate_customer_growth_rate(value) -> float:
    v = float(value)
    if not (-0.5 <= v <= 1.0):
        raise ValidationError(f"customer_growth_rate must be between -0.50 and 1.00. Got: {v}")
    return v


def validate_score_feature(name: str, value) -> int:
    v = int(value)
    if not (1 <= v <= 10):
        raise ValidationError(f"{name} must be between 1 and 10. Got: {v}")
    return v


def validate_market_demand_score(value) -> int:
    return validate_score_feature('market_demand_score', value)


def validate_competition_level(value) -> int:
    return validate_score_feature('competition_level', value)


def validate_owner_dependency_score(value) -> int:
    return validate_score_feature('owner_dependency_score', value)


def validate_scalability_score(value) -> int:
    return validate_score_feature('scalability_score', value)


def validate_location_score(value) -> int:
    return validate_score_feature('location_score', value)


def validate_revenue_stability_score(value) -> int:
    return validate_score_feature('revenue_stability_score', value)


def validate_industry(value: str) -> str:
    if value not in config.ALLOWED_INDUSTRIES:
        raise ValidationError(
            f"Unknown industry: '{value}'. Allowed: {config.ALLOWED_INDUSTRIES}"
        )
    return value


def validate_country(value: str) -> str:
    if value not in config.ALLOWED_COUNTRIES:
        raise ValidationError(
            f"Unknown country: '{value}'. Allowed: {config.ALLOWED_COUNTRIES}"
        )
    return value


def validate_city(value: str, country: str) -> str:
    allowed_cities = config.ALLOWED_CITIES_BY_COUNTRY.get(country, [])
    if allowed_cities and value not in allowed_cities:
        raise ValidationError(
            f"Unknown city: '{value}' for country '{country}'. Allowed: {allowed_cities}"
        )
    return value


def validate_profit_consistency(
    revenue: float, expenses: float, profit: float, tolerance: float = 0.05
) -> None:
    expected_profit = revenue - expenses
    if abs(expected_profit) > 0:
        diff_ratio = abs(profit - expected_profit) / abs(expected_profit)
        if diff_ratio > tolerance:
            raise ValidationError(
                f"profit inconsistency: expected ≈ {expected_profit:,.2f} "
                f"(revenue - expenses), got {profit:,.2f} "
                f"(difference: {diff_ratio*100:.1f}% > {tolerance*100}% tolerance)"
            )


def validate_profit_margin_consistency(
    profit: float, revenue: float, claimed_margin: float, tolerance: float = 0.05
) -> None:
    if abs(revenue) > 0:
        computed_margin = profit / revenue
        diff = abs(computed_margin - claimed_margin)
        if diff > tolerance:
            raise ValidationError(
                f"profit_margin inconsistency: computed {computed_margin:.4f}, "
                f"provided {claimed_margin:.4f} (difference {diff:.4f} > {tolerance} tolerance)"
            )


def validate_revenue_per_customer(revenue: float, monthly_customers: int) -> None:
    annual_customers = monthly_customers * 12
    if annual_customers > 0:
        avg = revenue / annual_customers
        if not (5 <= avg <= 100_000):
            raise ValidationError(
                f"avg_revenue_per_customer = {avg:,.2f} SAR is outside valid range [5, 100,000]. "
                f"Check annual_revenue_sar ({revenue:,.2f}) and monthly_customers ({monthly_customers})."
            )



def validate_inputs(data: dict) -> dict:
    errors = []
    validated = {}

    def _run(fn, *args):
        try:
            return fn(*args)
        except (ValidationError, ValueError, TypeError) as e:
            errors.append(str(e))
            return None

    country = _run(validate_country, data.get('country', ''))
    validated['country'] = country

    industry = _run(validate_industry, data.get('industry', ''))
    validated['industry'] = industry

    if country is not None:
        city = _run(validate_city, data.get('city', ''), country)
    else:
        city = data.get('city', '')
    validated['city'] = city

    revenue = _run(validate_annual_revenue_sar, data.get('annual_revenue_sar', 0))
    validated['annual_revenue_sar'] = revenue

    expenses = _run(validate_annual_expenses_sar, data.get('annual_expenses_sar', 0), revenue)
    validated['annual_expenses_sar'] = expenses

    profit = _run(validate_annual_profit_sar, data.get('annual_profit_sar', 0))
    validated['annual_profit_sar'] = profit

    margin = _run(validate_profit_margin, data.get('profit_margin', 0))
    validated['profit_margin'] = margin

    validated['number_of_employees'] = _run(validate_number_of_employees, data.get('number_of_employees', 0))
    validated['years_in_operation'] = _run(validate_years_in_operation, data.get('years_in_operation', 0))
    monthly_customers = _run(validate_monthly_customers, data.get('monthly_customers', 0))
    validated['monthly_customers'] = monthly_customers

    validated['revenue_growth_rate'] = _run(validate_revenue_growth_rate, data.get('revenue_growth_rate', 0))
    validated['customer_growth_rate'] = _run(validate_customer_growth_rate, data.get('customer_growth_rate', 0))

    validated['competition_level'] = _run(validate_competition_level, data.get('competition_level', 5))
    validated['owner_dependency_score'] = _run(validate_owner_dependency_score, data.get('owner_dependency_score', 5))
    validated['scalability_score'] = _run(validate_scalability_score, data.get('scalability_score', 5))
    validated['location_score'] = _run(validate_location_score, data.get('location_score', 5))
    validated['revenue_stability_score'] = _run(validate_revenue_stability_score, data.get('revenue_stability_score', 5))

    if not errors and None not in [revenue, expenses, profit, margin, monthly_customers]:
        _run(validate_profit_consistency, revenue, expenses, profit)
        _run(validate_profit_margin_consistency, profit, revenue, margin)
        _run(validate_revenue_per_customer, revenue, monthly_customers)

    if errors:
        raise ValidationError(f"Input validation failed with {len(errors)} error(s):\n" + "\n".join(f"  • {e}" for e in errors))

    return validated


if __name__ == "__main__":
    sample = {
        'country': 'Saudi Arabia',
        'industry': 'Technology',
        'city': 'Riyadh',
        'annual_revenue_sar': 5_000_000,
        'annual_expenses_sar': 3_500_000,
        'annual_profit_sar': 1_500_000,
        'profit_margin': 0.30,
        'number_of_employees': 45,
        'years_in_operation': 6,
        'monthly_customers': 120,
        'revenue_growth_rate': 0.20,
        'customer_growth_rate': 0.15,
        'competition_level': 6,
        'owner_dependency_score': 4,
        'scalability_score': 8,
        'location_score': 9,
        'revenue_stability_score': 7,
    }
    result = validate_inputs(sample)
    print("Validation passed.")
    for k, v in result.items():
        print(f"  {k}: {v}")
