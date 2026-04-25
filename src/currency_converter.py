import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

SAR_TO_USD = 0.2665
USD_TO_SAR = 3.75


def convert_sar_to_usd(value_sar: float) -> float:
    return round(value_sar * SAR_TO_USD, 2)


def convert_usd_to_sar(value_usd: float) -> float:
    return round(value_usd * USD_TO_SAR, 2)


def update_exchange_rate(new_rate: float):
    global SAR_TO_USD
    SAR_TO_USD = new_rate
    print(f"Exchange rate updated: 1 SAR = {new_rate} USD")


def format_dual_currency(value_sar: float) -> dict:
    return {
        "valuation_sar": round(value_sar, 2),
        "valuation_usd": convert_sar_to_usd(value_sar)
    }


if __name__ == "__main__":
    test_value = 1_000_000
    result = format_dual_currency(test_value)
    print(f"SAR: {result['valuation_sar']:,.2f}")
    print(f"USD: {result['valuation_usd']:,.2f}")
