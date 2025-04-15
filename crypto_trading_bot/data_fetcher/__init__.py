"""
data_fetcher package

Aggregates all data ingestion modules: market, onchain, sentiment, and macro.
"""

from .market import MarketDataFetcher
from .onchain import OnChainDataFetcher
from .sentiment import SentimentDataFetcher
from .macro import MacroDataFetcher