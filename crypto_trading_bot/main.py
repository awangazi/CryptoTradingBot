"""
main.py

Entry point for the crypto trading bot.
Orchestrates data ingestion, feature engineering, RL agent, backtesting, execution, and monitoring.
"""

import os # Add os import
import pandas as pd # Added for printing DataFrame

from .utils import ConfigLoader, TradeLogger # Use explicit relative import
from .data_fetcher import MarketDataFetcher, OnChainDataFetcher, SentimentDataFetcher, MacroDataFetcher # Keep others commented for now # Use explicit relative import
from .feature_engineer import TechnicalIndicators # Import TechnicalIndicators
from .rl_agent import RLTradingAgent, MetaLearningLoop # Import RLTradingAgent and MetaLearningLoop
from .backtester import Backtester # Import Backtester
from .executor import TradeExecutor # Import TradeExecutor
from .risk_manager import RiskManager # Import RiskManager
import numpy as np

def main():
    """
    Main orchestration function for the trading bot pipeline.
    """
    # Load configuration
    # Note: Ensure you have a .env file in the crypto_trading_bot directory
    # with BINANCE_API_KEY and BINANCE_SECRET_KEY if needed for private endpoints.
    # Public data fetching might work without keys for some exchanges/endpoints.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(script_dir, '.env')
    config = ConfigLoader(env_path=env_file_path)
    logger = TradeLogger(log_file="main_log.json") # Use a different log file for main
    logger.log_event({"event": "main_start"})

    # Initialize modules (examples)
    market_fetcher = MarketDataFetcher(config)
    onchain_fetcher = OnChainDataFetcher(config) # Initialize OnChainDataFetcher
    sentiment_fetcher = SentimentDataFetcher(config) # Initialize SentimentDataFetcher
    macro_fetcher = MacroDataFetcher(config) # Initialize MacroDataFetcher
    indicators = TechnicalIndicators() # Initialize TechnicalIndicators
    # regime_detector = MarketRegimeDetector()
    rl_agent = RLTradingAgent(config) # Initialize RLTradingAgent
    meta_learner = MetaLearningLoop(config) # Initialize MetaLearningLoop
    backtester = Backtester(config) # Initialize Backtester
    executor = TradeExecutor(config) # Initialize TradeExecutor
    risk_manager = RiskManager(config) # Initialize RiskManager

    # --- Test MarketDataFetcher, TechnicalIndicators, and RLTradingAgent ---
    logger.log_event({"event": "testing_market_fetcher_indicators_and_rl_agent"})
    symbol = 'BTC/USDT'
    start_date = '2024-01-01 00:00:00'
    end_date = '2024-01-31 00:00:00' # Fetch 14 hours of data
    interval = '1h'

    historical_data = market_fetcher.fetch_historical(symbol, start_date, end_date, interval)

    if historical_data is not None and not historical_data.empty:
        print(f"Successfully fetched {len(historical_data)} rows for {symbol} ({interval})")
        # print("Data Head:") # Commented out for brevity
        # print(historical_data.head())
        logger.log_event({"event": "market_fetcher_test_success", "rows": len(historical_data)})

        # Print DataFrame info before RSI calculation
        # print("\nDataFrame Info:") # Commented out for brevity
        # historical_data.info()

        # Calculate RSI
        rsi = indicators.rsi(historical_data)
        # print("\nRSI Head:") # Commented out for brevity
        # print(rsi.head())
        logger.log_event({"event": "rsi_calculation_success", "rows": len(rsi)})

        # Build and train the RL agent
        env = rl_agent.build_env(historical_data)
        rl_agent.train(env, timesteps=1000) # Reduced timesteps for testing

        # Get a sample action
        obs = env.reset()
        action = rl_agent.act(obs)
        print(f"\nSample Action: {action}")
        logger.log_event({"event": "rl_agent_test_success", "action": str(action)})

        # --- Test Backtester ---
        logger.log_event({"event": "testing_backtester"})
        backtester.run(rl_agent, historical_data) # Pass rl_agent directly
        # --- End Test ---

        # --- Test RiskManager ---
        logger.log_event({"event": "testing_risk_manager"})
        # Test ATR Stop Loss
        long_stop, short_stop = risk_manager.calculate_atr_stop_loss(historical_data)
        if long_stop is not None and short_stop is not None:
            print(f"\nATR Stop Loss: Long={long_stop:.2f}, Short={short_stop:.2f}")
            logger.log_event({"event": "risk_manager_atr_test_success", "long_stop": long_stop, "short_stop": short_stop})
        else:
            print("\nFailed to calculate ATR stop loss.")
            logger.log_event({"event": "risk_manager_atr_test_failed"})

        # Test Dynamic Position Size
        portfolio_value = 100000.0 # Example portfolio value
        confidence = 0.8 # Example agent confidence
        position_size = risk_manager.dynamic_position_size(portfolio_value, confidence)
        print(f"Dynamic Position Size (Confidence={confidence}): {position_size:.2f}")
        logger.log_event({"event": "risk_manager_pos_size_test_success", "position_size": position_size})
        # --- End Test ---

        # --- Test MetaLearningLoop ---
        logger.log_event({"event": "testing_meta_learning"})
        # Create some example trades
        trades_data = {
            'timestamp': pd.to_datetime(['2024-01-02 00:00:00', '2024-01-02 01:00:00', '2024-01-02 02:00:00']),
            'pnl': [100, -50, 75],
            'rsi': [60, 40, 55],
            'volatility': [0.01, 0.02, 0.015]
        }
        trades = pd.DataFrame(trades_data).set_index('timestamp')

        meta_learner.run_periodic_update(trades)
        print("\nMeta-Learning Loop completed. New signal weights:")
        print(meta_learner.signal_weights)
        logger.log_event({"event": "meta_learning_test_success", "new_weights": meta_learner.signal_weights})
        # --- End Test ---

    else:
        print(f"Failed to fetch historical data for {symbol}.")
        logger.log_event({"event": "market_fetcher_test_failed"})

    # --- Test OnChainDataFetcher ---
    logger.log_event({"event": "testing_onchain_fetcher"})
    onchain_start_date = '2024-01-01 00:00:00'
    onchain_end_date = '2024-01-01 05:00:00'
    whale_flows = onchain_fetcher.fetch_whale_flows(symbol, onchain_start_date, onchain_end_date)
    if whale_flows is not None and not whale_flows.empty:
        print(f"Successfully fetched {len(whale_flows)} rows of whale flow data for {symbol}")
        # print("Whale Flows Head:") # Commented out for brevity
        # print(whale_flows.head())
        logger.log_event({"event": "onchain_fetcher_test_success", "rows": len(whale_flows)})
    else:
        print(f"Failed to fetch whale flow data for {symbol}.")
        logger.log_event({"event": "onchain_fetcher_test_failed"})

    # --- Test SentimentDataFetcher ---
    logger.log_event({"event": "testing_sentiment_fetcher"})
    sentiment_start_date = '2024-01-01T00:00:00Z' # Twitter API uses RFC3339 format
    sentiment_end_date = '2024-01-01T01:00:00Z'
    twitter_sentiment = sentiment_fetcher.fetch_twitter_sentiment(symbol, sentiment_start_date, sentiment_end_date)
    if twitter_sentiment is not None and not twitter_sentiment.empty:
        print(f"Successfully fetched {len(twitter_sentiment)} rows of Twitter sentiment data for {symbol}")
        print("Twitter Sentiment Head:")
        print(twitter_sentiment.head())
        logger.log_event({"event": "sentiment_fetcher_test_success", "rows": len(twitter_sentiment)})
    else:
        print(f"Failed to fetch Twitter sentiment data for {symbol}.")
        logger.log_event({"event": "sentiment_fetcher_test_failed"})

    # --- Test MacroDataFetcher ---
    logger.log_event({"event": "testing_macro_fetcher"})
    macro_start_date = '2024-01-01' # NewsAPI uses YYYY-MM-DD format
    macro_end_date = '2024-01-01'
    macro_events = macro_fetcher.fetch_macro_events(macro_start_date, macro_end_date)
    if macro_events is not None and not macro_events.empty:
        print(f"Successfully fetched {len(macro_events)} rows of macro event data")
        # print("Macro Events Head:") # Commented out for brevity
        # print(macro_events.head())
        logger.log_event({"event": "macro_fetcher_test_success", "rows": len(macro_events)})
    else:
        print(f"Failed to fetch macro event data.")
        logger.log_event({"event": "macro_fetcher_test_failed"})
    # --- End Test ---

    # --- Test TradeExecutor ---
    logger.log_event({"event": "testing_executor"})
    if executor.exchange: # Check if executor initialized correctly
        test_symbol = 'BTC/USDT' # Use a valid symbol for testing
        test_amount = 0.0001 # Use a very small amount for testing
        print(f"\nAttempting to place a test market buy order for {test_amount} {test_symbol}...")
        order_result = executor.place_market_order(test_symbol, 'buy', test_amount)
        if order_result:
            print("Test market order placed successfully:")
            print(order_result)
            logger.log_event({"event": "executor_test_order_success", "order": order_result})
        else:
            print("Failed to place test market order.")
            logger.log_event({"event": "executor_test_order_failed"})
    else:
        print("\nTradeExecutor not initialized correctly (check API keys). Skipping live order test.")
        logger.log_event({"event": "executor_test_skipped"})
    # --- End Test ---


    # Orchestration and pipeline logic goes here (currently replaced by test)
    # Example: Fetch data, engineer features, detect regimes, train agent, backtest, execute trades, monitor, etc.
    # pass # Original pass removed

    logger.log_event({"event": "main_end"})


if __name__ == "__main__":
    # To run correctly with relative imports, execute from the parent directory (Desktop in this case)
    # using the command: python -m crypto_trading_bot.main
    main()