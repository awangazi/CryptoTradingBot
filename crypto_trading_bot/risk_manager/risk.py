"""
risk.py

Implements advanced risk management: ATR-based stop-loss, dynamic position sizing (Kelly Criterion, agent confidence), max exposure per asset, and circuit breaker logic.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from ..utils.logging import TradeLogger

logger = TradeLogger(log_file="risk_manager.log")

class RiskManager:
    """
    Manages risk controls and position sizing for the trading bot.
    """
    def __init__(self, config):
        self.config = config
        self.max_exposure_per_asset = float(config.get("MAX_EXPOSURE_PER_ASSET", 0.3))
        self.circuit_breaker_threshold = float(config.get("CIRCUIT_BREAKER_THRESHOLD", 0.1))
        self.circuit_breaker_window = int(config.get("CIRCUIT_BREAKER_WINDOW_MINUTES", 60))

    def calculate_atr(self, df: pd.DataFrame, period: int = 14):
        """
        Calculate Average True Range (ATR).
        Requires 'high', 'low', 'close' columns in the DataFrame.
        """
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            logger.log_event({"event": "calculate_atr_failed", "reason": "Missing required columns (high, low, close)"})
            return None
        try:
            atr = ta.atr(df['high'], df['low'], df['close'], length=period)
            logger.log_event({"event": "calculate_atr_success", "period": period})
            return atr
        except Exception as e:
            logger.log_event({"event": "calculate_atr_error", "error": str(e)})
            return None

    def calculate_atr_stop_loss(self, df: pd.DataFrame, atr_period: int = 14, multiplier: float = 2.0):
        """
        Calculate ATR-based stop-loss level.
        Returns stop-loss price for long and short positions.
        """
        atr = self.calculate_atr(df, period=atr_period)
        if atr is None or atr.empty:
            return None, None

        last_close = df['close'].iloc[-1]
        last_atr = atr.iloc[-1]

        long_stop_loss = last_close - (last_atr * multiplier)
        short_stop_loss = last_close + (last_atr * multiplier)
        logger.log_event({"event": "calculate_atr_stop_loss_success", "last_close": last_close, "last_atr": last_atr, "long_stop": long_stop_loss, "short_stop": short_stop_loss})
        return long_stop_loss, short_stop_loss

    def dynamic_position_size(self, portfolio_value: float, confidence: float, kelly_fraction: float = 1.0, win_prob: float = 0.55, win_loss_ratio: float = 1.5):
        """
        Calculate position size using Kelly Criterion and agent confidence.
        Args:
            portfolio_value (float): Current total value of the portfolio.
            confidence (float): Agent's confidence in the trade (0 to 1).
            kelly_fraction (float): Fraction of Kelly criterion to use (0 to 1).
            win_prob (float): Estimated probability of winning the trade.
            win_loss_ratio (float): Estimated ratio of average win size to average loss size.
        Returns:
            float: Calculated position size in quote currency (e.g., USDT).
        """
        # Kelly Criterion formula: f = (p * (b + 1) - 1) / b
        # where p = win probability, b = win/loss ratio
        # Simplified Kelly: f = p - (1 - p) / b
        if win_loss_ratio <= 0:
            logger.log_event({"event": "dynamic_position_size_failed", "reason": "Win/loss ratio must be positive"})
            return 0.0 # Avoid division by zero

        kelly_f = win_prob - (1 - win_prob) / win_loss_ratio

        # Adjust based on confidence and Kelly fraction
        position_fraction = max(0, min(1, kelly_f * kelly_fraction * confidence))
        position_size = portfolio_value * position_fraction

        logger.log_event({
            "event": "dynamic_position_size_calculated",
            "portfolio_value": portfolio_value,
            "confidence": confidence,
            "kelly_f": kelly_f,
            "position_fraction": position_fraction,
            "position_size": position_size
        })
        return position_size

    def check_max_exposure(self, symbol: str, current_exposure_value: float, portfolio_value: float):
        """
        Check if adding a new position would exceed the maximum allowed exposure per asset.
        Args:
            symbol (str): The asset symbol.
            current_exposure_value (float): The current value of the position in this asset.
            portfolio_value (float): The total portfolio value.
        Returns:
            bool: True if exposure limit is not exceeded, False otherwise.
        """
        max_allowed_value = portfolio_value * self.max_exposure_per_asset
        if current_exposure_value >= max_allowed_value:
            logger.log_event({
                "event": "max_exposure_exceeded",
                "symbol": symbol,
                "current_exposure": current_exposure_value,
                "max_allowed": max_allowed_value,
                "portfolio_value": portfolio_value
            })
            return False
        return True

    def circuit_breaker(self, portfolio_history: pd.DataFrame, current_time: pd.Timestamp):
        """
        Check if the circuit breaker threshold has been hit.
        Args:
            portfolio_history (pd.DataFrame): DataFrame with 'timestamp' and 'net_worth' columns.
            current_time (pd.Timestamp): The current timestamp.
        Returns:
            bool: True if trading should be halted, False otherwise.
        """
        if portfolio_history.empty:
            return False

        window_start_time = current_time - pd.Timedelta(minutes=self.circuit_breaker_window)
        recent_history = portfolio_history[portfolio_history.index >= window_start_time]

        if recent_history.empty:
            return False

        start_worth = recent_history['net_worth'].iloc[0]
        current_worth = recent_history['net_worth'].iloc[-1]
        drawdown = (start_worth - current_worth) / start_worth

        if drawdown >= self.circuit_breaker_threshold:
            logger.log_event({
                "event": "circuit_breaker_tripped",
                "start_time": str(window_start_time),
                "current_time": str(current_time),
                "start_worth": start_worth,
                "current_worth": current_worth,
                "drawdown": drawdown,
                "threshold": self.circuit_breaker_threshold
            })
            return True
        return False