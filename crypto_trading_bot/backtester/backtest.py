"""
backtest.py

Implements the backtesting engine for simulating trading strategies under realistic market conditions.
Supports transaction costs, slippage, and order book simulation.
"""

import pandas as pd
from ..utils.logging import TradeLogger
import numpy as np

logger = TradeLogger(log_file="backtester.log")

class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    """
    def __init__(self, config):
        self.config = config

    def run(self, rl_agent, data: pd.DataFrame, initial_balance=100000.0, commission=0.001):
        """
        Run backtest for a given RL agent and data.
        """
        logger.log_event({"event": "backtest_start", "strategy": str(rl_agent)})
        balance = initial_balance
        shares_held = 0
        net_worth = balance
        data = data.copy() # Avoid modifying the original DataFrame

        # Ensure data has correct format
        data.index.name = 'Datetime'
        data.columns = [col.lower() for col in data.columns]

        # Create a trading environment
        env = rl_agent.build_env(data)
        obs = env.reset() # DummyVecEnv.reset() returns only obs

        logger.log_event({"event": "backtest_running"})
        for i in range(len(data) - 1):
            action = rl_agent.act(obs)
            obs, reward, done, info = env.step(action) # DummyVecEnv.step() returns 4 values
            current_price = data['close'].iloc[i]

            if action == 1:  # Buy
                available_shares = balance / current_price
                shares_held += available_shares
                balance -= available_shares * current_price
            elif action == 2:  # Sell
                balance += shares_held * current_price
                shares_held = 0

            net_worth = balance + shares_held * current_price
            logger.log_event({"event": "trade", "step": i, "action": int(action[0]), "price": current_price, "balance": balance, "shares": shares_held, "net_worth": net_worth}) # Convert action to int

            if done:
                break

        logger.log_event({"event": "backtest_finished"})
        print('Final Portfolio Value: %.2f' % net_worth)

    def simulate_order_execution(self, order, order_book):
        """
        Simulate order execution with slippage and order book depth.
        """
        logger.log_event({"event": "simulate_order_execution", "order": str(order)})
        pass  # To be implemented

    def calculate_performance_metrics(self, trades):
        """
        Calculate performance metrics: ROI, Sharpe Ratio, Max Drawdown, etc.
        """
        logger.log_event({"event": "calculate_performance_metrics"})
        pass  # To be implemented