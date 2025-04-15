"""
onchain.py

Fetches and processes on-chain metrics such as whale flows, exchange inflows/outflows, and other blockchain analytics.
Integrates with public APIs like Glassnode and Dune Analytics.
"""

import pandas as pd
from dune_client.types import QueryParameter
from dune_client.client import DuneClient
from dune_client.query import QueryBase
from datetime import datetime, timezone # Added datetime, timezone
import time # Added time
from ..utils.logging import TradeLogger
import os # Added os to get API key from env

logger = TradeLogger(log_file="onchain_fetcher.log")

class OnChainDataFetcher:
    """
    Fetches on-chain metrics for supported cryptocurrencies.
    """
    def __init__(self, config):
        self.config = config
        # Dune API Key
        self.dune_api_key = config.get("DUNE_API_KEY")
        if not self.dune_api_key:
            logger.log_event({"event": "dune_api_key_missing", "severity": "WARNING"})
            self.dune_client = None
        else:
            try:
                self.dune_client = DuneClient(self.dune_api_key)
                logger.log_event({"event": "dune_client_initialized"})
            except Exception as e:
                logger.log_event({"event": "dune_client_init_error", "error": str(e)})
                self.dune_client = None
    def fetch_whale_flows(self, symbol: str, start_date: str, end_date: str):
        """
        Fetch whale transaction flows.
        NOTE: Reliable, free historical whale flow data via API is difficult to obtain.
              This often requires paid services (Glassnode, CryptoQuant) or finding/creating
              a specific Dune Analytics query tracking large wallet movements.
              This function is currently a placeholder.
        """
        logger.log_event({"event": "fetch_whale_flows_skipped", "symbol": symbol, "reason": "Not implemented due to lack of reliable free API source."})
        # To implement this, you would need to:
        # 1. Find a suitable public Dune query (e.g., searching Dune for "whale transactions [symbol]")
        # 2. Get the query ID.
        # 3. Use self._fetch_dune_query(query_id, params={'symbol': symbol, ...})
        # 4. Parse the results appropriately.
        return None # Placeholder

    def fetch_exchange_flows(self, symbol: str, start_date: str, end_date: str):
        """
        Fetch exchange inflow/outflow data using a Dune Analytics query.
        Requires DUNE_API_KEY in .env and a valid QUERY_ID in config.
        """
        logger.log_event({"event": "fetch_exchange_flows_start", "symbol": symbol, "start": start_date, "end": end_date})

        if not self.dune_client:
            logger.log_event({"event": "fetch_exchange_flows_error", "symbol": symbol, "error": "Dune client not initialized (check API key)"})
            return None

        # --- IMPORTANT ---
        # You MUST find a suitable public Dune query for exchange flows for your desired asset(s).
        # Search on Dune Analytics (e.g., "bitcoin exchange netflow").
        # Get the Query ID from the URL (the number).
        # Store this ID in your config file or .env under a key like 'DUNE_EXCHANGE_FLOW_QUERY_ID'.
        query_id = self.config.get("DUNE_EXCHANGE_FLOW_QUERY_ID") # Get Query ID from config

        if not query_id:
             logger.log_event({"event": "fetch_exchange_flows_error", "symbol": symbol, "error": "DUNE_EXCHANGE_FLOW_QUERY_ID not found in config"})
             return None

        try:
            query_id = int(query_id) # Ensure it's an integer
        except ValueError:
             logger.log_event({"event": "fetch_exchange_flows_error", "symbol": symbol, "error": f"Invalid DUNE_EXCHANGE_FLOW_QUERY_ID: {query_id}"})
             return None

        # Define parameters for the Dune query (these depend HEAVILY on the specific query)
        # Common parameters might include asset symbol/address, start/end dates.
        # Check the query page on Dune to see what parameters it accepts.
        # Example parameters (ADAPT TO YOUR CHOSEN QUERY):
        params = [
            QueryParameter.text_type(name="Symbol", value=symbol.split('/')[0]), # Assuming query takes base symbol
            QueryParameter.date_type(name="StartDate", value=start_date), # Format 'YYYY-MM-DD HH:MM:SS'
            QueryParameter.date_type(name="EndDate", value=end_date),     # Format 'YYYY-MM-DD HH:MM:SS'
        ]

        logger.log_event({"event": "fetching_dune_query", "query_id": query_id, "params": {p.name: p.value for p in params}})

        try:
            results_df = self._fetch_dune_query(query_id, params)

            if results_df is None or results_df.empty:
                logger.log_event({"event": "fetch_exchange_flows_empty", "symbol": symbol, "query_id": query_id})
                return None

            # --- Data Cleaning & Formatting ---
            # The column names and data types depend entirely on the Dune query results.
            # Inspect the results_df and adapt the cleaning below.
            # Example: Assuming columns 'time' (datetime) and 'netflow' (numeric)
            if 'time' not in results_df.columns or 'netflow' not in results_df.columns:
                 logger.log_event({
                     "event": "fetch_exchange_flows_invalid_data",
                     "symbol": symbol,
                     "query_id": query_id,
                     "error": "Required columns ('time', 'netflow') not found in Dune results",
                     "columns_found": list(results_df.columns)
                 })
                 return None

            # Convert timestamp column (adjust column name and format if needed)
            try:
                results_df['timestamp'] = pd.to_datetime(results_df['time'], utc=True)
            except Exception as e:
                 logger.log_event({"event": "fetch_exchange_flows_invalid_data", "symbol": symbol, "query_id": query_id, "error": f"Failed to parse timestamp column 'time': {e}"})
                 return None

            results_df.set_index('timestamp', inplace=True)
            results_df.index.name = 'Datetime'

            # Select and rename the relevant column (e.g., 'netflow')
            # Ensure the flow data is numeric
            try:
                results_df['flow'] = pd.to_numeric(results_df['netflow'])
            except Exception as e:
                 logger.log_event({"event": "fetch_exchange_flows_invalid_data", "symbol": symbol, "query_id": query_id, "error": f"Failed to convert 'netflow' column to numeric: {e}"})
                 return None

            # Filter by date range again just in case query parameters weren't exact
            start_dt_obj = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            end_dt_obj = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            results_df = results_df[(results_df.index >= start_dt_obj) & (results_df.index < end_dt_obj)]


            logger.log_event({"event": "fetch_exchange_flows_success", "symbol": symbol, "query_id": query_id, "rows": len(results_df)})
            return results_df[['flow']] # Return only the flow column

        except Exception as e:
            logger.log_event({"event": "fetch_exchange_flows_error", "symbol": symbol, "query_id": query_id, "error": str(e)})
            return None


    def _fetch_dune_query(self, query_id: int, params: list[QueryParameter] = None):
        """Helper function to fetch results from a Dune Analytics query."""
        if not self.dune_client:
            logger.log_event({"event": "dune_fetch_error", "query_id": query_id, "error": "Dune client not initialized"})
            return None
        try:
            query = QueryBase(query_id=query_id, params=params or [])
            logger.log_event({"event": "dune_refresh_start", "query_id": query_id})
            results = self.dune_client.refresh_into_dataframe(query)
            logger.log_event({"event": "dune_refresh_complete", "query_id": query_id, "rows": len(results)})
            # Basic rate limiting - Dune free tier has limits
            time.sleep(2) # Add a small delay after each query
            return results
        except Exception as e:
            # Handle common Dune errors if possible (e.g., invalid query ID, parameter mismatch)
            logger.log_event({"event": "dune_fetch_error", "query_id": query_id, "error": str(e)})
            return None

    def fetch_custom_metric(self, query_id: int, params: list[QueryParameter]):
        """
        Fetch results from a specific Dune Analytics query ID with given parameters.
        """
        logger.log_event({"event": "fetch_custom_metric_start", "query_id": query_id, "params": {p.name: p.value for p in params}})
        return self._fetch_dune_query(query_id, params)