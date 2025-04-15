"""
config.py

Loads configuration from .env and other sources for the trading bot.
"""

import os
from dotenv import load_dotenv

class ConfigLoader:
    """
    Loads and provides access to configuration variables.
    """
    def __init__(self, env_path: str = ".env"):
        load_dotenv(env_path)
        self.config = {key: os.getenv(key) for key in os.environ.keys()}

    def get(self, key: str, default=None):
        """
        Get a configuration value by key.
        """
        return self.config.get(key, default)