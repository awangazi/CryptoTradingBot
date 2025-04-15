"""
regime.py

Detects and labels market regimes (bull, bear, sideways, stress) using unsupervised learning on volatility, volume, and returns.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ..utils.logging import TradeLogger

logger = TradeLogger(log_file="regime_detector.log")

class MarketRegimeDetector:
    """
    Uses clustering algorithms to detect and label market regimes.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None

    def compute_features(self, df: pd.DataFrame, volatility_window=20, return_window=1):
        """
        Compute features (volatility, returns) for clustering.
        Requires 'close' column in the DataFrame.
        """
        logger.log_event({"event": "compute_regime_features_start"})
        if 'close' not in df.columns:
            logger.log_event({"event": "compute_regime_features_failed", "reason": "Missing 'close' column"})
            return None

        features = pd.DataFrame(index=df.index)
        try:
            # Calculate log returns
            features['returns'] = np.log(df['close'] / df['close'].shift(return_window))
            # Calculate rolling volatility (standard deviation of log returns)
            features['volatility'] = features['returns'].rolling(window=volatility_window).std()

            # Drop NaN values created by rolling calculations and shifts
            features.dropna(inplace=True)

            logger.log_event({"event": "compute_regime_features_success", "rows": len(features)})
            return features

        except Exception as e:
            logger.log_event({"event": "compute_regime_features_error", "error": str(e)})
            return None

    def detect_regimes(self, features: pd.DataFrame, method: str = "kmeans", n_clusters: int = 3):
        """
        Detect market regimes using k-Means clustering.
        Args:
            features (pd.DataFrame): DataFrame containing features like 'returns', 'volatility'.
            method (str): Clustering method (currently only 'kmeans' supported).
            n_clusters (int): Number of regimes to detect.
        Returns:
            pd.Series: Series containing the regime label for each timestamp, or None if failed.
        """
        logger.log_event({"event": "detect_regimes_start", "method": method, "n_clusters": n_clusters})
        if features is None or features.empty:
            logger.log_event({"event": "detect_regimes_failed", "reason": "Input features are empty or None"})
            return None

        if method.lower() != "kmeans":
            logger.log_event({"event": "detect_regimes_failed", "reason": f"Unsupported method: {method}. Only 'kmeans' is supported."})
            # TODO: Implement other methods like GMM if needed
            return None

        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(features)

            # Apply K-Means clustering
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set n_init explicitly
            regime_labels = self.kmeans.fit_predict(scaled_features)

            regimes = pd.Series(regime_labels, index=features.index, name='regime')
            logger.log_event({"event": "detect_regimes_success", "clusters_found": len(np.unique(regime_labels))})
            return regimes

        except Exception as e:
            logger.log_event({"event": "detect_regimes_error", "error": str(e)})
            return None

    def label_regimes(self, regimes: pd.Series, features: pd.DataFrame):
        """
        Assign human-readable labels (e.g., 'Low Volatility', 'High Volatility') to detected regimes based on cluster centroids.
        This is a simplified labeling based on volatility. More sophisticated labeling could be used.
        """
        if regimes is None or features is None or self.kmeans is None:
            logger.log_event({"event": "label_regimes_failed", "reason": "Regimes, features, or kmeans model is None"})
            print("Debug: Regimes, features, or kmeans model is None") # Debug print
            return None
        logger.log_event({"event": "label_regimes_start"})
        try:
            # Get cluster centers (in scaled space)
            centers_scaled = self.kmeans.cluster_centers_
            print(f"Debug: Scaled centers shape: {centers_scaled.shape}") # Debug print

            # Inverse transform to get centers in original feature space
            centers_original = self.scaler.inverse_transform(centers_scaled)
            print(f"Debug: Original centers shape: {centers_original.shape}") # Debug print
            print(f"Debug: Original centers:\n{centers_original}") # Debug print


            # Assuming 'volatility' is the second feature (index 1)
            if centers_original.shape[1] < 2:
                 logger.log_event({"event": "label_regimes_error", "reason": "Not enough columns in cluster centers for volatility"})
                 print("Debug: Not enough columns in cluster centers for volatility") # Debug print
                 return None
            volatility_centers = centers_original[:, 1]
            print(f"Debug: Volatility centers: {volatility_centers}") # Debug print


            # Sort cluster indices by volatility
            volatility_sorted_indices = np.argsort(volatility_centers)
            print(f"Debug: Sorted indices by volatility: {volatility_sorted_indices}") # Debug print


            # Create labels based on volatility ranking
            label_map = {}
            if len(volatility_sorted_indices) == 3:
                label_map[int(volatility_sorted_indices[0])] = "Low Volatility" # Convert key to int
                label_map[int(volatility_sorted_indices[1])] = "Medium Volatility" # Convert key to int
                label_map[int(volatility_sorted_indices[2])] = "High Volatility" # Convert key to int
            else: # Generic labels if not 3 clusters
                 for i, cluster_idx in enumerate(volatility_sorted_indices):
                     label_map[int(cluster_idx)] = f"Regime {i}" # Convert key to int
            print(f"Debug: Label map: {label_map}") # Debug print


            labeled_regimes = regimes.map(label_map)
            print(f"Debug: Labeled regimes head:\n{labeled_regimes.head()}") # Debug print
            logger.log_event({"event": "label_regimes_success", "label_map": label_map})
            return labeled_regimes

        except Exception as e:
            logger.log_event({"event": "label_regimes_error", "error": str(e)})
            print(f"Debug: Error during labeling: {e}") # Debug print
            return None