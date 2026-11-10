def evaluate_risk(signal: int, confidence: float, current_atr: float, atr_threshold: float = 5.0, confidence_threshold: float = 0.55) -> str:
    """
    STRICT Rules-based risk management engine.
    Overrides ML signal if conditions are dangerous.
    
    Args:
        signal (int): 1 for Buy, 0 for Sell/Hold from ML model.
        confidence (float): Probability from model (0.0 to 1.0)
        current_atr (float): Current volatility measure
        atr_threshold (float): Max allowed ATR before trading halts
        confidence_threshold (float): Minimum confidence to execute
    
    Returns:
        str: Final action string ['BUY', 'SELL', 'HOLD']
    """
    
    # 1. High Volatility Override
    if current_atr > atr_threshold:
        return "HOLD (High Volatility)"
    
    # 2. Low Confidence Override
    if confidence < confidence_threshold:
        return f"HOLD (Low Confidence: {confidence:.2f})"
        
    # 3. Model Signal Execution
    if signal == 1:
        return "BUY"
    else:
        return "SELL"


def calculate_position_size(capital: float, risk_per_trade: float, atr: float, price: float) -> float:
    """
    Calculates the number of shares to trade based on Average True Range limits.
    Risk equation: Shares = Total Risk Amount / Initial Stop Loss Distance (1.5 * ATR),
    capped so the position never costs more than 95% of available capital.
    """
    if atr <= 0 or price <= 0 or capital <= 0:
        return 0.0
        
    risk_amount = capital * risk_per_trade
    stop_distance = 1.5 * atr  # Conservative stop loss
    
    position_size = risk_amount / stop_distance
    
    # Capital cap check: shares * price may not exceed capital (leave 5% margin)
    max_affordable = (capital * 0.95) / price
    return min(position_size, max_affordable)
