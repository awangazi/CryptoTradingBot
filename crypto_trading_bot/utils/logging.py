"""
logging.py

Provides logging utilities for the trading bot, including detailed trade logs in JSON format.
"""

import logging
import json

class TradeLogger:
    """
    Logs trades and events in structured JSON format.
    """
    def __init__(self, log_file: str = "trade_logs.json"):
        self.log_file = log_file
        self.logger = logging.getLogger("TradeLogger")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_file)
        self.logger.addHandler(handler)

    def log_trade(self, trade_data: dict):
        """
        Log a trade with all relevant metadata.
        """
        self.logger.info(json.dumps(trade_data))

    def log_event(self, event: dict):
        """
        Log a general event.
        """
        self.logger.info(json.dumps(event))