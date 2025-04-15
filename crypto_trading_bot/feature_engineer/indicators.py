"""
indicators.py

Computes technical indicators such as EMA, MACD, RSI, Bollinger Bands, and others for feature engineering.
"""

import pandas as pd
import pandas_ta as ta

class TechnicalIndicators:
    """
    Computes various technical indicators for a given price DataFrame.
    """
    def __init__(self):
        pass

    def ema(self, df: pd.DataFrame, period: int = 20, column: str = "close"):
        """
        Compute Exponential Moving Average (EMA).
        """
        ema = ta.ema(df[column], length=period)
        return ema

    def macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close"):
        """
        Compute MACD and signal line.
        """
        macd = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
        return macd

    def rsi(self, df: pd.DataFrame, period: int = 14, column: str = "close"):
        """
        Compute Relative Strength Index (RSI).
        """
        rsi = ta.rsi(df[column], length=period)
        return rsi

    def bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, column: str = "close"):
        """
        Compute Bollinger Bands.
        """
        bbands = ta.bbands(df[column], length=period, std=std_dev)
        return bbands