"""
sentiment.py

Fetches and processes sentiment data from social platforms (Twitter, Reddit) and news feeds (NewsAPI, GDELT).
Applies NLP models (SpaCy, FinBERT, or OpenAI) to score sentiment on a normalized scale.
"""

import tweepy
import praw # Added for Reddit
from newsapi import NewsApiClient # Added for NewsAPI
import pandas as pd
from datetime import datetime, timedelta, timezone # Added datetime utils
from ..utils.logging import TradeLogger
from transformers import pipeline # For free sentiment analysis
import os # Added for env vars

logger = TradeLogger(log_file="sentiment_fetcher.log")

class SentimentDataFetcher:
    """
    Fetches and scores sentiment from social and news sources.
    """
    def __init__(self, config):
        self.config = config
        # Twitter Client
        self.twitter_bearer_token = config.get("TWITTER_BEARER_TOKEN")
        self.twitter_client = None
        if self.twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=self.twitter_bearer_token)
                logger.log_event({"event": "twitter_client_init_success"})
            except Exception as e:
                logger.log_event({"event": "twitter_client_init_failed", "error": str(e)})
        else:
            logger.log_event({"event": "twitter_client_init_skipped", "reason": "TWITTER_BEARER_TOKEN not found in environment variables"})

        # Reddit Client
        self.reddit_client_id = config.get("REDDIT_CLIENT_ID")
        self.reddit_client_secret = config.get("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = config.get("REDDIT_USER_AGENT") # e.g., "CryptoBot by YourUsername"
        self.reddit_client = None
        if self.reddit_client_id and self.reddit_client_secret and self.reddit_user_agent:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent,
                    read_only=True # Read-only mode is sufficient
                )
                logger.log_event({"event": "reddit_client_init_success"})
            except Exception as e:
                logger.log_event({"event": "reddit_client_init_failed", "error": str(e)})
        else:
            logger.log_event({"event": "reddit_client_init_skipped", "reason": "Missing Reddit credentials in environment variables"})

        # NewsAPI Client
        self.newsapi_key = config.get("NEWSAPI_KEY")
        self.newsapi_client = None
        if self.newsapi_key:
            try:
                self.newsapi_client = NewsApiClient(api_key=self.newsapi_key)
                logger.log_event({"event": "newsapi_client_init_success"})
            except Exception as e:
                logger.log_event({"event": "newsapi_client_init_failed", "error": str(e)})
        else:
            logger.log_event({"event": "newsapi_client_init_skipped", "reason": "NEWSAPI_KEY not found in environment variables"})
        # Initialize sentiment analysis pipeline (Hugging Face Transformers)
        self.sentiment_pipeline = pipeline("sentiment-analysis") # Default model

    def fetch_twitter_sentiment(self, symbol: str, start_time: str, end_time: str, max_results=10):
        """
        Fetch and score sentiment from Twitter for a given symbol and date range.
        WARNING: Twitter API v2 free tier is very limited. This may return few or no results.
        """
        logger.log_event({"event": "fetch_twitter_sentiment_start", "symbol": symbol, "start": start_time, "end": end_time})
        if not self.twitter_client:
            logger.log_event({"event": "fetch_twitter_sentiment_skipped", "reason": "Twitter client not initialized (check TWITTER_BEARER_TOKEN)"})
            return None

        try:
            # -------------------------------------------------------------------
            # Note: search_recent_tweets only covers ~last 7 days on standard v2 access.
            # Max results for free tier might be very low (e.g., 10 per request).

            query = f"{symbol} lang:en -is:retweet" # Example query
            # Convert dates to Twitter API format (ISO 8601)
            # Ensure dates are timezone-aware (UTC)
            try:
                start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                # Twitter API requires RFC 3339 format
                start_time_iso = start_dt.isoformat().replace('+00:00', 'Z')
                end_time_iso = end_dt.isoformat().replace('+00:00', 'Z')
            except ValueError:
                 logger.log_event({"event": "fetch_twitter_sentiment_error", "symbol": symbol, "error": "Invalid date format. Use YYYY-MM-DD HH:MM:SS"})
                 return None

            tweets = self.twitter_client.search_recent_tweets(
                query,
                start_time=start_time_iso,
                end_time=end_time_iso,
                max_results=max_results, # Note: Free tier might enforce lower limits (e.g., 10)
                tweet_fields=["created_at", "public_metrics"] # Added public_metrics for potential filtering
            )

            if not tweets.data:
                logger.log_event({"event": "fetch_twitter_sentiment_empty", "symbol": symbol})
                return None

            sentiments = []
            for tweet in tweets.data:
                try:
                    # Use the Hugging Face Transformers pipeline for sentiment analysis
                    text_to_analyze = tweet.text
                    sentiment_result = self.sentiment_pipeline(text_to_analyze)[0]
                    label = sentiment_result['label']
                    score = sentiment_result['score']

                    # Convert label/score to a single score [-1, 1]
                    sentiment_score = score if label == 'POSITIVE' else -score

                    # Consider weighting by engagement (e.g., retweet_count) if needed
                    # weight = 1 + tweet.public_metrics.get('retweet_count', 0) * 0.1 # Example weighting

                    sentiments.append({
                        "timestamp": tweet.created_at,
                        "sentiment_score": sentiment_score,
                        # "weight": weight # Optional
                        })
                except Exception as analysis_error:
                     logger.log_event({"event": "sentiment_analysis_error", "source": "twitter", "tweet_id": tweet.id, "error": str(analysis_error)})

            df = pd.DataFrame(sentiments)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True) # Ensure UTC
                df.set_index('timestamp', inplace=True)
                df.index.name = 'Datetime'
                logger.log_event({"event": "fetch_twitter_sentiment_success", "symbol": symbol, "rows": len(df)})
                return df
            else:
                logger.log_event({"event": "fetch_twitter_sentiment_empty", "symbol": symbol})
                return None

        except tweepy.TweepyException as e:
            logger.log_event({"event": "fetch_twitter_sentiment_error", "symbol": symbol, "error": str(e)})
            return None
        except Exception as e:
            logger.log_event({"event": "fetch_twitter_sentiment_error", "symbol": symbol, "error": str(e)})
            return None

    # Removed score_sentiment placeholder as pipeline is used directly
    def fetch_reddit_sentiment(self, symbol: str, start_date: str, end_date: str):
        """
        Fetch and score sentiment from Reddit (e.g., r/CryptoCurrency) for a given symbol.
        Note: PRAW doesn't easily support historical search by date range without Pushshift or paid tiers.
              This fetches recent posts matching the symbol.
        """
        logger.log_event({"event": "fetch_reddit_sentiment_start", "symbol": symbol})
        if not self.reddit_client:
            logger.log_event({"event": "fetch_reddit_sentiment_skipped", "reason": "Reddit client not initialized (check credentials)"})
            return None

        # Define relevant subreddits (consider making this configurable)
        subreddits_to_search = ["CryptoCurrency", "Bitcoin", "Ethereum", "Altcoin"] # Example list
        query = symbol.split('/')[0] # Use base symbol for search
        limit_per_subreddit = 25 # Number of recent posts to fetch per subreddit

        sentiments = []
        try:
            for sub_name in subreddits_to_search:
                logger.log_event({"event": "searching_subreddit", "subreddit": sub_name, "query": query})
                subreddit = self.reddit_client.subreddit(sub_name)
                # Search for recent submissions matching the query
                for submission in subreddit.search(query, sort="new", time_filter="week", limit=limit_per_subreddit):
                     # Combine title and selftext for analysis
                    text_to_analyze = submission.title + " " + submission.selftext
                    try:
                        sentiment_result = self.sentiment_pipeline(text_to_analyze)[0]
                        label = sentiment_result['label']
                        score = sentiment_result['score']
                        sentiment_score = score if label == 'POSITIVE' else -score
                        submission_time = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)

                        # Optional: Filter by the provided date range if needed, though search is limited
                        # start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                        # end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                        # if not (start_dt <= submission_time < end_dt):
                        #     continue

                        sentiments.append({
                            "timestamp": submission_time,
                            "sentiment_score": sentiment_score,
                            "source": f"reddit/{sub_name}"
                        })
                    except Exception as analysis_error:
                        logger.log_event({"event": "sentiment_analysis_error", "source": "reddit", "submission_id": submission.id, "error": str(analysis_error)})

            if not sentiments:
                logger.log_event({"event": "fetch_reddit_sentiment_empty", "symbol": symbol})
                return None

            df = pd.DataFrame(sentiments)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = 'Datetime'
            logger.log_event({"event": "fetch_reddit_sentiment_success", "symbol": symbol, "rows": len(df)})
            return df[['sentiment_score']] # Return only score

        except Exception as e:
            logger.log_event({"event": "fetch_reddit_sentiment_error", "symbol": symbol, "error": str(e)})
            return None

    def fetch_news_sentiment(self, symbol: str, start_date: str, end_date: str, page_size=20):
        """
        Fetch and score sentiment from news sources using NewsAPI.
        WARNING: NewsAPI free tier has limitations (e.g., 100 requests/day, limited history).
        """
        logger.log_event({"event": "fetch_news_sentiment_start", "symbol": symbol, "start": start_date, "end": end_date})
        if not self.newsapi_client:
            logger.log_event({"event": "fetch_news_sentiment_skipped", "reason": "NewsAPI client not initialized (check NEWSAPI_KEY)"})
            return None

        try:
            # Format dates for NewsAPI (YYYY-MM-DD)
            start_date_news = start_date.split(' ')[0]
            end_date_news = end_date.split(' ')[0]
            query = symbol.split('/')[0] # Use base symbol

            # Fetch articles
            articles_response = self.newsapi_client.get_everything(
                q=query,
                from_param=start_date_news,
                to=end_date_news,
                language='en',
                sort_by='publishedAt', # Or 'relevancy', 'popularity'
                page_size=page_size # Max 100 for paid, lower for free tier? Check docs.
            )

            if articles_response['status'] != 'ok' or articles_response['totalResults'] == 0:
                logger.log_event({"event": "fetch_news_sentiment_empty", "symbol": symbol, "response_status": articles_response.get('status')})
                return None

            sentiments = []
            for article in articles_response['articles']:
                # Combine title and description for analysis
                text_to_analyze = (article.get('title') or '') + " " + (article.get('description') or '')
                if not text_to_analyze.strip():
                    continue

                try:
                    sentiment_result = self.sentiment_pipeline(text_to_analyze)[0]
                    label = sentiment_result['label']
                    score = sentiment_result['score']
                    sentiment_score = score if label == 'POSITIVE' else -score
                    published_time = pd.to_datetime(article['publishedAt'], utc=True)

                    sentiments.append({
                        "timestamp": published_time,
                        "sentiment_score": sentiment_score,
                        "source": article['source']['name']
                    })
                except Exception as analysis_error:
                     logger.log_event({"event": "sentiment_analysis_error", "source": "newsapi", "article_url": article.get('url'), "error": str(analysis_error)})


            if not sentiments:
                logger.log_event({"event": "fetch_news_sentiment_empty", "symbol": symbol})
                return None

            df = pd.DataFrame(sentiments)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.set_index('timestamp', inplace=True)
            df.index.name = 'Datetime'
            logger.log_event({"event": "fetch_news_sentiment_success", "symbol": symbol, "rows": len(df)})
            return df[['sentiment_score']] # Return only score

        except Exception as e:
            # Catch potential NewsAPI errors (e.g., rate limits, invalid key)
            logger.log_event({"event": "fetch_news_sentiment_error", "symbol": symbol, "error": str(e)})
            return None

    # Removed normalize_sentiment - assuming pipeline output mapped to [-1, 1] is sufficient for now.