# CryptoTradingBot
Put in the API in the .env file and start the program

# Crypto Trading Bot

## Overview

This project is a self-learning crypto trading bot developed in Python but it can be improved upon. The bot autonomously fetches and processes diverse data sources, performs comprehensive backtesting, dynamically learns via reinforcement learning, and executes trades with advanced risk management. The design incorporates modularity, allowing independent updates to each component.

## Core Features

*   **Data Ingestion & Preprocessing:**
    *   Market Data: Retrieves historical and real-time OHLCV data from Binance, CoinGecko, and Kraken.
    *   On-Chain & Alternative Data: Integrates on-chain metrics (exchange inflows/outflows) using the Dune Analytics API.
    *   Sentiment & News: Pulls sentiment from social platforms (Twitter, Reddit) and news feeds (NewsAPI). Applies NLP to score sentiment.
    *   Macroeconomic & Geopolitical Events: Monitors global headlines and maps them to crypto market signals.
*   **Feature Engineering & Market Regime Detection:**
    *   Computes technical indicators: EMA, MACD, RSI, Bollinger Bands, etc.
    *   Uses unsupervised learning to detect and label market regimes dynamically.
    *   Fuses signals via weighted averaging and reinforcement learning.
*   **Learning & Adaptation:**
    *   Implements a Reinforcement Learning Engine using a DRL agent (e.g., Stable Baselines3 PPO/DQN) to simulate trading.
    *   Defines a reward function as a weighted sum of ROI, Sharpe Ratio, and Max Drawdown.
    *   Employs continuous learning with experience replay and periodic retraining.
    *   Includes a meta-learning loop for self-critique and signal weight adjustment.
*   **Backtesting & Simulation:**
    *   Uses a backtesting framework incorporating realistic market conditions (transaction costs, slippage, order book simulation).
    *   Runs simulations on various market regimes (bull, bear, sideways, stress-test scenarios).
*   **Execution & Risk Management:**
    *   Connects to Binance (or chosen exchange) via API.
    *   Implements market and limit orders with dynamic slippage estimation.
    *   Includes ATR-based stop-loss, dynamic position sizing, maximum exposure per asset, and a circuit breaker.
*   **Explainability & Monitoring:**
    *   Records detailed trade logs in JSON format.
    *   Provides a simple dashboard using Streamlit to show real-time performance, cumulative returns, drawdowns, and signal overlays on price charts.
    *   Saves model weights, training histories, and performance metrics for reproducibility.
*   **Modularity & Future-Readiness:**
    *   Organized into modular components: `data_fetcher`, `feature_engineer`, `rl_agent`, `backtester`, `executor`, `risk_manager`, and `dashboard`.
    *   Designed with clear interfaces for adding future data sources and strategies.

## Getting Started

### Prerequisites

*   Python 3.10+
*   Pip package manager

### Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    cd crypto_trading_bot
    ```

2.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Configure the environment variables:

    *   Create a `.env` file in the root directory.
    *   Add the following environment variables with your API keys and credentials:

        ```
        BINANCE_API_KEY=your_binance_api_key
        BINANCE_API_SECRET=your_binance_api_secret
        KRAKEN_API_KEY=your_kraken_api_key
        KRAKEN_API_SECRET=your_kraken_api_secret
        DUNE_API_KEY=your_dune_api_key
        TWITTER_BEARER_TOKEN=your_twitter_bearer_token
        REDDIT_CLIENT_ID=your_reddit_client_id
        REDDIT_CLIENT_SECRET=your_reddit_client_secret
        REDDIT_USER_AGENT=YourBotName (by /u/YourRedditUsername)
        NEWSAPI_KEY=your_newsapi_key
        STREAMLIT_PASSWORD=your_dashboard_password
        ```

        **Important:** Replace the placeholder values with your actual API keys and credentials.

4.  Find a suitable public Dune Analytics query for exchange flows and set the `DUNE_EXCHANGE_FLOW_QUERY_ID` in your `.env` file.

### Usage

1.  Run the main script:

    ```bash
    python main.py
    ```

    (Note: The specific usage will depend on how the `main.py` script is structured. Add more details here as the project evolves.)

2.  Access the Streamlit dashboard (if implemented) by running:

    ```bash
    streamlit run dashboard/app.py
    ```

    and navigating to the URL provided.

## Shortcomings

*   **Data Ingestion:**
    *   The Twitter API integration is limited by the free tier, which may result in limited or no data.
    *   The on-chain data integration relies on public Dune Analytics queries. The availability and quality of these queries may vary. Whale flow data is not currently implemented due to the lack of reliable free APIs.
    *   The macroeconomic event fetching uses a simplified keyword-based approach, which may not capture the nuances of geopolitical events.
*   **Market Regime Detection:**
    *   The market regime detection is a basic implementation and may not accurately identify all market regimes.
*   **Backtesting:**
    *   The backtesting framework may not fully simulate all real-world market conditions.
*   **Reinforcement Learning:**
    *   The RL agent's performance is highly dependent on the reward function and training data. Further optimization and fine-tuning may be required.
*   **Dune Analytics Query:**
    *   The `DUNE_EXCHANGE_FLOW_QUERY_ID` needs to be manually configured with a valid Dune Analytics query ID. The bot does not automatically search for or validate the query.

## Contributing

Contributions are welcome! Please submit a pull request with your proposed changes.

## License

[Choose a license, e.g., MIT License]
