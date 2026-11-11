"""Rule-based overrides that sit between the model and any simulated order."""


def evaluate_risk(signal: int, confidence: float, atr_pct: float,
                  max_atr_pct: float = 2.5, confidence_threshold: float = 0.55) -> str:
    """Overrides the model when conditions are dangerous.

    Args:
        signal: 1 for up, 0 for down, as predicted by the model.
        confidence: probability the model gives its own prediction (0.0 to 1.0).
        atr_pct: Average True Range as a percentage of price. Using a percentage
            keeps one threshold meaningful for a $20 stock and a $700 ETF alike.
        max_atr_pct: volatility cap above which nothing is traded.
        confidence_threshold: minimum confidence to act.

    Returns:
        "BUY", "SELL" or a "HOLD (...)" string explaining the override.
    """
    if atr_pct > max_atr_pct:
        return f"HOLD (volatility {atr_pct:.1f}% > {max_atr_pct:.1f}% cap)"
    if confidence < confidence_threshold:
        return f"HOLD (confidence {confidence:.0%} < {confidence_threshold:.0%})"
    return "BUY" if signal == 1 else "SELL"


def calculate_position_size(capital: float, risk_per_trade: float, atr: float, price: float) -> float:
    """Number of shares to trade.

    Shares = amount at risk / stop distance (1.5 x ATR), capped so the position
    never costs more than 95% of available capital.
    """
    if atr <= 0 or price <= 0 or capital <= 0:
        return 0.0
    risk_amount = capital * risk_per_trade
    stop_distance = 1.5 * atr
    return min(risk_amount / stop_distance, (capital * 0.95) / price)
