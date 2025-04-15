"""
market.py

Handles fetching historical and real-time OHLCV market data from Binance, CoinGecko, and Kraken.
Provides unified interfaces for data retrieval and fallback logic.
"""

import ccxt
import pandas as pd
from pycoingecko import CoinGeckoAPI # Added for CoinGecko
from datetime import datetime, timezone, timedelta # Added timedelta
import time
from ..utils.logging import TradeLogger

# Initialize logger (adjust as needed based on how config/logger are passed)
logger = TradeLogger(log_file="data_fetcher.log")

class MarketDataFetcher:
    """
    Fetches historical and real-time OHLCV data from multiple exchanges.
    """
    def __init__(self, config):
        self.config = config
        # Initialize exchanges based on config (API keys needed for private endpoints)
        # For public data, keys might not be strictly necessary but good practice
        self.binance = ccxt.binance({
            'apiKey': config.get('BINANCE_API_KEY'),
            'secret': config.get('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True, # Adjust for clock skew
            }
        })
        self.coingecko = CoinGeckoAPI() # Initialize CoinGecko client (free tier)
        self.kraken = ccxt.kraken({ # Initialize Kraken
             # Kraken public endpoints don't strictly require keys
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
            }
        })

    def fetch_historical(self, symbol: str, start_date_str: str, end_date_str: str, interval: str = "1h"):
        """
        Fetch historical OHLCV data for a given symbol and date range from Binance.

        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT').
            start_date_str (str): Start date in 'YYYY-MM-DD HH:MM:SS' format.
            end_date_str (str): End date in 'YYYY-MM-DD HH:MM:SS' format.
            interval (str): Time interval (e.g., '1m', '5m', '1h', '1d').

        Returns:
            pd.DataFrame: DataFrame with OHLCV data, or None if fetching fails.
        """
        logger.log_event({"event": "fetch_historical_start", "symbol": symbol, "interval": interval, "start": start_date_str, "end": end_date_str})
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

        # --- Attempt 1: CoinGecko (Free Tier, good for longer history) ---
        df = self._fetch_historical_coingecko(symbol, start_dt, end_dt)
        if df is not None and not df.empty:
            logger.log_event({"event": "fetch_historical_success", "source": "coingecko", "symbol": symbol, "rows": len(df)})
            return df
        else:
            logger.log_event({"event": "fetch_historical_failed", "source": "coingecko", "symbol": symbol})

        # --- Attempt 2: Binance ---
        df = self._fetch_historical_exchange(self.binance, symbol, start_dt, end_dt, interval)
        if df is not None and not df.empty:
            logger.log_event({"event": "fetch_historical_success", "source": "binance", "symbol": symbol, "rows": len(df)})
            return df
        else:
            logger.log_event({"event": "fetch_historical_failed", "source": "binance", "symbol": symbol})

        # --- Attempt 3: Kraken ---
        df = self._fetch_historical_exchange(self.kraken, symbol, start_dt, end_dt, interval)
        if df is not None and not df.empty:
            logger.log_event({"event": "fetch_historical_success", "source": "kraken", "symbol": symbol, "rows": len(df)})
            return df
        else:
            logger.log_event({"event": "fetch_historical_failed", "source": "kraken", "symbol": symbol})

        logger.log_event({"event": "fetch_historical_error", "symbol": symbol, "error": "Failed to fetch data from all sources"})
        return None

    def _fetch_historical_coingecko(self, symbol: str, start_dt: datetime, end_dt: datetime):
        """Helper to fetch historical data from CoinGecko."""
        logger.log_event({"event": "fetch_coingecko_attempt", "symbol": symbol})
        try:
            # CoinGecko uses coin IDs, need mapping from symbol like 'BTC/USDT'
            # This is a simplification, a robust solution needs a proper mapping service/cache
            base_currency = symbol.split('/')[0].lower()
            quote_currency = symbol.split('/')[1].lower() # Usually 'usd' for coingecko price charts

            # Find CoinGecko ID for the base currency
            coins_list = self.coingecko.get_coins_list()
            coin_id = None
            for coin in coins_list:
                if coin['symbol'] == base_currency and 'bitcoin' in coin['id']: # Simple heuristic, needs improvement
                     coin_id = coin['id']
                     break
                elif coin['symbol'] == base_currency: # Fallback if specific match fails
                    coin_id = coin['id']
                    # Don't break immediately, prefer more specific matches if available

            if not coin_id:
                logger.log_event({"event": "fetch_coingecko_error", "symbol": symbol, "error": f"Could not find CoinGecko ID for {base_currency}"})
                return None

            # CoinGecko API takes start/end timestamps in Unix seconds
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            # Determine 'days' parameter for CoinGecko API
            # 'daily' interval data is returned for requests > 90 days
            # Hourly data for requests 1 to 90 days
            # Minute data for requests < 1 day (not directly supported by get_coin_market_chart_range_by_id)
            # We will fetch daily and resample later if needed, or adjust based on interval
            delta_days = (end_dt - start_dt).days
            if delta_days <= 0: delta_days = 1 # Fetch at least one day

            # Fetch market chart data
            chart_data = self.coingecko.get_coin_market_chart_range_by_id(
                id=coin_id,
                vs_currency=quote_currency,
                from_timestamp=start_ts,
                to_timestamp=end_ts
                # Note: CoinGecko free API might return daily data regardless of range sometimes.
            )

            if not chart_data or not chart_data.get('prices'):
                logger.log_event({"event": "fetch_coingecko_empty", "symbol": symbol, "coin_id": coin_id})
                return None

            # Convert CoinGecko data [timestamp, value] to OHLCV DataFrame
            # CoinGecko provides price, market_cap, total_volumes separately
            prices_df = pd.DataFrame(chart_data['prices'], columns=['timestamp', 'close'])
            volumes_df = pd.DataFrame(chart_data['total_volumes'], columns=['timestamp', 'volume'])
            # Market cap is often included but not needed for OHLCV
            # market_caps_df = pd.DataFrame(chart_data['market_caps'], columns=['timestamp', 'market_cap'])

            # Convert timestamps from ms to datetime and set index
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms', utc=True)
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms', utc=True)
            prices_df.set_index('timestamp', inplace=True)
            volumes_df.set_index('timestamp', inplace=True)

            # Merge price and volume data
            df = prices_df.join(volumes_df, how='inner')

            # CoinGecko free API often only gives 'close' price for daily.
            # We can approximate OHLC if needed, but for now, we'll use 'close' for all.
            # This is a limitation of the free tier for historical daily data.
            # If higher resolution (hourly) is returned, it might have OHLC implicitly.
            # For simplicity here, we create OHLC from close.
            df['open'] = df['close']
            df['high'] = df['close']
            df['low'] = df['close']

            # Reorder columns to standard OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index.name = 'Datetime'
            df.columns = [col.lower() for col in df.columns]

            # Filter exact date range
            df = df[(df.index >= start_dt) & (df.index < end_dt)]

            logger.log_event({"event": "fetch_coingecko_success", "symbol": symbol, "rows": len(df)})
            return df

        except Exception as e:
            logger.log_event({"event": "fetch_coingecko_error", "symbol": symbol, "error": f"Unexpected error: {e}"})
            return None


    def _fetch_historical_exchange(self, exchange: ccxt.Exchange, symbol: str, start_dt: datetime, end_dt: datetime, interval: str):
        """Helper function to fetch historical OHLCV data from a ccxt exchange."""
        logger.log_event({"event": "fetch_exchange_attempt", "exchange": exchange.id, "symbol": symbol})
        try:
            since = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            limit = 1000 # Standard limit for many exchanges
            timeframe_duration_ms = exchange.parse_timeframe(interval) * 1000
            all_ohlcv = []

            while since < end_ms:
                try:
                    # logger.log_event({"event": "fetching_chunk", "exchange": exchange.id, "symbol": symbol, "since": since})
                    ohlcv = exchange.fetch_ohlcv(symbol, interval, since, limit)

                    if not ohlcv:
                        # logger.log_event({"event": "no_data_received", "exchange": exchange.id, "symbol": symbol, "since": since})
                        break # No more data

                    # Filter out data beyond the requested end_date precisely
                    ohlcv = [candle for candle in ohlcv if candle[0] < end_ms]
                    if not ohlcv:
                        break

                    all_ohlcv.extend(ohlcv)
                    last_timestamp = ohlcv[-1][0]
                    since = last_timestamp + timeframe_duration_ms # Move to the next interval start

                    # Check if we are stuck in a loop (e.g., timestamp not advancing)
                    if len(all_ohlcv) > limit and all_ohlcv[-limit][0] == last_timestamp:
                         logger.log_event({"event": "fetch_stuck_warning", "exchange": exchange.id, "symbol": symbol, "timestamp": last_timestamp})
                         since += timeframe_duration_ms # Force advance

                    # Respect rate limits
                    if exchange.enableRateLimit:
                        time.sleep(exchange.rateLimit / 1000)

                except ccxt.RateLimitExceeded as e:
                    logger.log_event({"event": "fetch_exchange_error", "exchange": exchange.id, "symbol": symbol, "error": f"RateLimitExceeded: {e}. Sleeping..."})
                    time.sleep(60) # Wait longer for rate limit exceeded
                except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                    logger.log_event({"event": "fetch_exchange_error", "exchange": exchange.id, "symbol": symbol, "error": f"Temporary Network/Exchange Error: {e}. Retrying soon..."})
                    time.sleep(10) # Wait before retrying network issues
                except ccxt.ExchangeError as e: # Catch other specific exchange errors
                    logger.log_event({"event": "fetch_exchange_error", "exchange": exchange.id, "symbol": symbol, "error": f"ExchangeError: {e}"})
                    return None # Non-recoverable exchange error for this attempt
                except Exception as e: # Catch any other unexpected error during chunk fetch
                    logger.log_event({"event": "fetch_exchange_error", "exchange": exchange.id, "symbol": symbol, "error": f"Unexpected chunk fetch error: {e}"})
                    return None # Non-recoverable for this attempt


            if not all_ohlcv:
                logger.log_event({"event": "fetch_exchange_empty", "exchange": exchange.id, "symbol": symbol})
                return None

            # Convert to DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = 'Datetime'
            df.columns = [col.lower() for col in df.columns]

            # Ensure data is within the exact requested range
            df = df[(df.index >= start_dt) & (df.index < end_dt)]

            # Remove potential duplicates just in case pagination logic overlapped
            df = df[~df.index.duplicated(keep='first')]

            logger.log_event({"event": "fetch_exchange_success", "exchange": exchange.id, "symbol": symbol, "rows": len(df)})
            return df

        except Exception as e: # Catch errors during the overall _fetch_historical_exchange call
            logger.log_event({"event": "fetch_exchange_error", "exchange": exchange.id, "symbol": symbol, "error": f"Overall fetch error: {e}"})
            return None

    def fetch_realtime(self, symbol: str, interval: str = "1m"):
        """
        Fetch the latest OHLCV candle for a given symbol.
        Note: Real-time streaming often requires websockets, this is a simplified polling approach.
        """
        logger.log_event({"event": "fetch_realtime_start", "symbol": symbol, "interval": interval})
        # Attempt 1: Binance
        df = self._fetch_realtime_exchange(self.binance, symbol, interval)
        if df is not None and not df.empty:
            logger.log_event({"event": "fetch_realtime_success", "source": "binance", "symbol": symbol, "timestamp": str(df.index[0])})
            return df
        else:
            logger.log_event({"event": "fetch_realtime_failed", "source": "binance", "symbol": symbol})

        # Attempt 2: Kraken
        df = self._fetch_realtime_exchange(self.kraken, symbol, interval)
        if df is not None and not df.empty:
            logger.log_event({"event": "fetch_realtime_success", "source": "kraken", "symbol": symbol, "timestamp": str(df.index[0])})
            return df
        else:
            logger.log_event({"event": "fetch_realtime_failed", "source": "kraken", "symbol": symbol})

        logger.log_event({"event": "fetch_realtime_error", "symbol": symbol, "error": "Failed to fetch real-time data from all sources"})
        return None

    def _fetch_realtime_exchange(self, exchange: ccxt.Exchange, symbol: str, interval: str):
        """Helper to fetch the latest candle from a specific exchange."""
        logger.log_event({"event": "fetch_realtime_exchange_attempt", "exchange": exchange.id, "symbol": symbol})
        try:
            # Fetch the most recent 2 candles to get the last completed one
            ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=2)
            if not ohlcv or len(ohlcv) < 2: # Need at least 2 candles
                logger.log_event({"event": "fetch_realtime_exchange_empty", "exchange": exchange.id, "symbol": symbol})
                return None

            # Get the second to last candle (last fully completed one)
            last_candle = ohlcv[-2]

            df = pd.DataFrame([last_candle], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = 'Datetime'
            df.columns = [col.lower() for col in df.columns]

            return df

        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            logger.log_event({"event": "fetch_realtime_exchange_error", "exchange": exchange.id, "symbol": symbol, "error": f"{type(e).__name__}: {e}"})
            return None
        except Exception as e:
            logger.log_event({"event": "fetch_realtime_exchange_error", "exchange": exchange.id, "symbol": symbol, "error": f"Unexpected error: {e}"})
            return None


    # Removed fallback_to_kraken as logic is now integrated into fetch_historical and fetch_realtime