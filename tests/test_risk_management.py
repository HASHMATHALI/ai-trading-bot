import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_management import calculate_position_size, evaluate_risk


def test_high_volatility_and_low_confidence_force_hold():
    assert evaluate_risk(1, 0.9, atr_pct=3.0, max_atr_pct=2.5).startswith("HOLD")
    assert evaluate_risk(1, 0.5, atr_pct=1.0, confidence_threshold=0.6).startswith("HOLD")


def test_signal_is_executed_when_safe():
    assert evaluate_risk(1, 0.8, atr_pct=1.0) == "BUY"
    assert evaluate_risk(0, 0.8, atr_pct=1.0) == "SELL"


def test_position_size_risks_two_percent():
    # $10,000 * 2% = $200 risk; stop distance 1.5 * 4 = $6 -> 33.33 shares
    assert round(calculate_position_size(10_000, 0.02, atr=4, price=100), 2) == 33.33


def test_position_never_exceeds_capital():
    shares = calculate_position_size(10_000, 0.02, atr=0.1, price=500)
    assert shares * 500 <= 10_000 * 0.95 + 1e-6


def test_degenerate_inputs_return_zero():
    assert calculate_position_size(10_000, 0.02, atr=0, price=100) == 0.0
    assert calculate_position_size(10_000, 0.02, atr=4, price=0) == 0.0
