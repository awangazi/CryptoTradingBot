"""
agent.py

Implements the deep reinforcement learning (DRL) trading agent using Stable Baselines3 (PPO/DQN).
Handles training, inference, experience replay, and model persistence.
"""

import os
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

class TradingEnv(gym.Env):
    """
    A simple trading environment for a single asset.
    """
    def __init__(self, df, lookback_window=5): # Reduced lookback_window
        super(TradingEnv, self).__init__()
        self.df = df
        self.lookback_window = lookback_window
        self.start_index = lookback_window # Ensure start_index is correct
        self.current_step = self.start_index
        self.max_steps = len(df) - 1
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_drawdown = 0

        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space: price data (lookback window) + balance + shares held
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.lookback_window * len(df.columns) + 2,), dtype=np.float32)

    def _next_observation(self):
        # Get the prices for the last 'lookback_window' steps
        start = max(0, self.current_step - self.lookback_window)
        end = self.current_step
        obs = self.df.iloc[start:end].values.flatten()

        # Add the agent's balance and shares held
        obs = np.append(obs, [self.balance, self.shares_held])
        return obs

    def _take_action(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        # Execute actions
        if action == 1:  # Buy
            available_shares = self.balance / current_price
            self.shares_held += available_shares
            self.balance -= available_shares * current_price
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price
        drawdown = (self.initial_balance - self.net_worth) / self.initial_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        obs = self._next_observation()
        reward = self.net_worth - self.initial_balance # Simple reward: net worth increase
        truncated = False # Add truncated for gym 0.26 compatibility
        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        self.current_step = self.start_index
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_drawdown = 0
        obs = self._next_observation()
        info = {}
        if seed is not None:
            info['seed'] = seed
        return obs, info

class RLTradingAgent:
    """
    Deep RL agent for trading using Stable Baselines3.
    """
    def __init__(self, config):
        self.config = config
        self.model = None

    def build_env(self, data: pd.DataFrame):
        """
        Build or update the trading environment for the agent.
        """
        env = TradingEnv(data)
        env = DummyVecEnv([lambda: env])  # Vectorize the environment
        return env

    def train(self, env, timesteps: int = 100_000):
        """
        Train the RL agent on the given environment.
        """
        self.model = PPO("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=timesteps)

    def act(self, obs):
        """
        Select an action given the current state.
        """
        action, _states = self.model.predict(obs, deterministic=True)
        return action

    def save(self, path: str):
        """
        Save the trained model to disk.
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Load a trained model from disk.
        """
        self.model = PPO.load(path)