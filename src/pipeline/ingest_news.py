"""
News Ingestion Pipeline for Titan V8
Phase 1.1: Build 7-Year News Dataset (2018-2025)

This module creates a unified news dataset by merging:
- Kaggle backfill data (2018-2020)
- Polygon.io API data (2020-2025)

Usage:
    python -m src.pipeline.ingest_news
"""

import os
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yaml
from polygon import RESTClient
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str = "conf/base/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_polygon_client() -> RESTClient:
    """Initialize Polygon.io REST client from environment variable."""
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError(
            "POLYGON_API_KEY not found in environment. "
            "Please set it in your .env file."
        )
    return RESTClient(api_key)


# =============================================================================
# TEXT CLEANING UTILITIES
# =============================================================================

def clean_motley_fool_text(text: str) -> str:
    """
    Remove Motley Fool disclosure footer from news text.
    
    Common patterns:
    - "The Motley Fool has a disclosure policy..."
    - "The Motley Fool has positions in..."
    - "The Motley Fool recommends..."
    
    Args:
        text: Raw news text
    
    Returns:
        Cleaned text with disclosure removed
    """
    if not isinstance(text, str):
        return ""
    
    # Patterns to remove
    patterns = [
        r"The Motley Fool has a disclosure policy.*$",
        r"The Motley Fool has positions? in.*$",
        r"The Motley Fool recommends.*$",
        r"The Motley Fool owns shares of.*$",
        r"\*Stock Advisor returns as of.*$",
        r"Suzanne Frey, an executive at Alphabet.*$",
        r"John Mackey, CEO of Whole Foods Market.*$",
    ]
    
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned


def clean_news_text(text: str) -> str:
    """
    General text cleaning for news articles.
    
    Args:
        text: Raw news text
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove Motley Fool disclosures
    text = clean_motley_fool_text(text)
    
    # Remove HTML tags if any
    text = re.sub(r"<[^>]+>", "", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


# =============================================================================
# STEP A: KAGGLE INGESTION (2018-2020)
# =============================================================================

def download_kaggle_dataset(output_path: str) -> bool:
    """
    Attempt to download Kaggle dataset using kaggle API.
    
    Args:
        output_path: Directory to save the dataset
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("  📥 Downloading Kaggle dataset...")
        api.dataset_download_files(
            "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests",
            path=output_path,
            unzip=True
        )
        print("  ✓ Kaggle download complete")
        return True
        
    except ImportError:
        print("  ⚠️ kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"  ⚠️ Kaggle API error: {e}")
        return False


def ingest_kaggle_news(config: dict) -> pd.DataFrame:
    """
    Ingest news data from Kaggle dataset (2018-2020).
    
    Dataset: miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
    
    Args:
        config: Configuration dictionary with tickers
    
    Returns:
        DataFrame with standardized columns: date, ticker, raw_text, sentiment_score, source
    """
    print("\n📰 STEP A: Kaggle News Ingestion (2018-2020)")
    print("-" * 50)
    
    raw_path = Path(config["data"]["raw_path"])
    csv_path = raw_path / "analyst_ratings_processed.csv"
    
    # Check if file exists
    if not csv_path.exists():
        print(f"  ⚠️ File not found: {csv_path}")
        
        # Try to download via Kaggle API
        if not download_kaggle_dataset(str(raw_path)):
            raise FileNotFoundError(
                f"\n❌ Please download the Kaggle dataset manually:\n"
                f"   1. Go to: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests\n"
                f"   2. Download 'analyst_ratings_processed.csv'\n"
                f"   3. Place it in: {csv_path}\n"
            )
    
    print(f"  📂 Loading: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df):,} rows")
    
    # Get our tickers
    our_tickers = set(config["data"]["tickers"])
    
    # Filter to our tickers
    # The dataset uses 'stock' column for ticker symbols
    if "stock" in df.columns:
        df = df[df["stock"].isin(our_tickers)]
    elif "ticker" in df.columns:
        df = df[df["ticker"].isin(our_tickers)]
    else:
        print(f"  ⚠️ Available columns: {list(df.columns)}")
        raise ValueError("Cannot find ticker column in Kaggle dataset")
    
    print(f"  ✓ Filtered to {len(df):,} rows for our tickers")
    
    # Standardize columns
    standardized = pd.DataFrame()
    
    # Date parsing
    if "date" in df.columns:
        standardized["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        print("  ⚠️ No date column found")
        return pd.DataFrame()
    
    # Ticker
    if "stock" in df.columns:
        standardized["ticker"] = df["stock"]
    elif "ticker" in df.columns:
        standardized["ticker"] = df["ticker"]
    
    # Headline -> raw_text
    if "title" in df.columns:
        standardized["raw_text"] = df["title"].apply(clean_news_text)
    elif "headline" in df.columns:
        standardized["raw_text"] = df["headline"].apply(clean_news_text)
    else:
        print("  ⚠️ No headline/title column found")
        return pd.DataFrame()
    
    # Sentiment - default to neutral (0.0)
    standardized["sentiment_score"] = 0.0
    
    # Source label
    standardized["source"] = "kaggle"
    
    # Drop rows with missing data
    standardized = standardized.dropna(subset=["date", "ticker", "raw_text"])
    standardized = standardized[standardized["raw_text"].str.len() > 10]
    
    print(f"  ✓ Standardized: {len(standardized):,} news items")
    
    return standardized


# =============================================================================
# STEP B: POLYGON INGESTION (2020-2025)
# =============================================================================

def map_polygon_sentiment(insights: list) -> float:
    """
    Map Polygon sentiment insights to numeric score.
    
    Args:
        insights: List of insight objects from Polygon
    
    Returns:
        Sentiment score: 1.0 (positive), -1.0 (negative), 0.0 (neutral)
    """
    if not insights:
        return 0.0
    
    sentiments = []
    for insight in insights:
        sentiment = getattr(insight, "sentiment", None)
        if sentiment:
            if sentiment == "positive":
                sentiments.append(1.0)
            elif sentiment == "negative":
                sentiments.append(-1.0)
            else:
                sentiments.append(0.0)
    
    if sentiments:
        return sum(sentiments) / len(sentiments)
    return 0.0


def ingest_polygon_news(config: dict) -> pd.DataFrame:
    """
    Ingest news data from Polygon.io API (2020-2025).
    
    Args:
        config: Configuration dictionary with tickers
    
    Returns:
        DataFrame with standardized columns: date, ticker, raw_text, sentiment_score, source
    """
    print("\n📰 STEP B: Polygon News Ingestion (2020-2025)")
    print("-" * 50)
    
    client = get_polygon_client()
    tickers = config["data"]["tickers"]
    
    # Polygon news API typically has ~5 year history
    # We'll fetch from 2020 onwards to complement Kaggle
    polygon_start = "2020-01-01"
    polygon_end = datetime.now().strftime("%Y-%m-%d")
    
    all_news = []
    
    for ticker in tqdm(tickers, desc="Fetching ticker news"):
        try:
            news_items = []
            
            # Fetch news for this ticker
            for article in client.list_ticker_news(
                ticker=ticker,
                published_utc_gte=polygon_start,
                published_utc_lte=polygon_end,
                limit=1000,
                order="desc"
            ):
                # Extract fields
                published = getattr(article, "published_utc", None)
                title = getattr(article, "title", "") or ""
                description = getattr(article, "description", "") or ""
                insights = getattr(article, "insights", []) or []
                
                # Combine title and description
                raw_text = f"{title}. {description}".strip()
                raw_text = clean_news_text(raw_text)
                
                if not raw_text or len(raw_text) < 10:
                    continue
                
                # Get sentiment
                sentiment_score = map_polygon_sentiment(insights)
                
                news_items.append({
                    "date": pd.to_datetime(published, utc=True),
                    "ticker": ticker,
                    "raw_text": raw_text,
                    "sentiment_score": sentiment_score,
                    "source": "polygon"
                })
            
            all_news.extend(news_items)
            time.sleep(0.15)  # Rate limiting
            
        except Exception as e:
            print(f"\n  ⚠️ Error fetching news for {ticker}: {e}")
            continue
    
    if not all_news:
        print("  ⚠️ No news fetched from Polygon")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_news)
    print(f"\n  ✓ Fetched: {len(df):,} news items from Polygon")
    
    return df


# =============================================================================
# STEP C: MERGE AND DEDUPLICATE
# =============================================================================

def merge_news_datasets(kaggle_df: pd.DataFrame, polygon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Kaggle and Polygon news datasets with deduplication.
    
    Args:
        kaggle_df: Standardized Kaggle news
        polygon_df: Standardized Polygon news
    
    Returns:
        Merged and deduplicated DataFrame
    """
    print("\n🔗 STEP C: Merge and Deduplicate")
    print("-" * 50)
    
    # Concatenate
    dfs_to_concat = []
    
    if not kaggle_df.empty:
        print(f"  📊 Kaggle: {len(kaggle_df):,} items")
        dfs_to_concat.append(kaggle_df)
    
    if not polygon_df.empty:
        print(f"  📊 Polygon: {len(polygon_df):,} items")
        dfs_to_concat.append(polygon_df)
    
    if not dfs_to_concat:
        raise ValueError("No news data to merge!")
    
    merged = pd.concat(dfs_to_concat, ignore_index=True)
    print(f"  ✓ Combined: {len(merged):,} items")
    
    # Normalize dates for deduplication (date only, no time)
    merged["date_only"] = merged["date"].dt.date
    
    # Create text hash for faster deduplication
    merged["text_hash"] = merged["raw_text"].str.lower().str[:100]
    
    # Deduplicate based on date, ticker, and text similarity
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["date_only", "ticker", "text_hash"])
    after_dedup = len(merged)
    
    print(f"  ✓ Deduplicated: {before_dedup - after_dedup:,} duplicates removed")
    print(f"  ✓ Final count: {after_dedup:,} unique news items")
    
    # Clean up temp columns
    merged = merged.drop(columns=["date_only", "text_hash"])
    
    # Sort by date
    merged = merged.sort_values("date").reset_index(drop=True)
    
    # Ensure date is timezone-aware UTC
    if merged["date"].dt.tz is None:
        merged["date"] = merged["date"].dt.tz_localize("UTC")
    
    return merged


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to Parquet format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False, engine="pyarrow")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n💾 Saved to: {output_path} ({file_size_mb:.2f} MB)")


def main():
    """Main entry point for news ingestion pipeline."""
    print("\n" + "=" * 60)
    print("🚀 TITAN V8 NEWS INGESTION PIPELINE")
    print("    Phase 1.1: 7-Year News Dataset (2018-2025)")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    tickers = config["data"]["tickers"]
    print(f"\nTickers: {tickers}")
    print(f"Date Range: 2018-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
    
    # Step A: Kaggle ingestion (2018-2020)
    try:
        kaggle_df = ingest_kaggle_news(config)
    except FileNotFoundError as e:
        print(e)
        print("\n⚠️ Continuing without Kaggle data...")
        kaggle_df = pd.DataFrame()
    except Exception as e:
        print(f"\n⚠️ Kaggle ingestion failed: {e}")
        kaggle_df = pd.DataFrame()
    
    # Step B: Polygon ingestion (2020-2025)
    try:
        polygon_df = ingest_polygon_news(config)
    except Exception as e:
        print(f"\n⚠️ Polygon ingestion failed: {e}")
        polygon_df = pd.DataFrame()
    
    # Step C: Merge and deduplicate
    merged_df = merge_news_datasets(kaggle_df, polygon_df)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("📊 NEWS DATASET SUMMARY")
    print("=" * 60)
    print(f"Total items: {len(merged_df):,}")
    print(f"Unique tickers: {merged_df['ticker'].nunique()}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"\nItems by source:")
    print(merged_df["source"].value_counts().to_string())
    print(f"\nItems by ticker:")
    print(merged_df["ticker"].value_counts().head(10).to_string())
    print(f"\nSentiment distribution:")
    print(f"  Positive (>0): {(merged_df['sentiment_score'] > 0).sum():,}")
    print(f"  Neutral (=0):  {(merged_df['sentiment_score'] == 0).sum():,}")
    print(f"  Negative (<0): {(merged_df['sentiment_score'] < 0).sum():,}")
    print("=" * 60)
    
    # Save to parquet
    output_path = Path(config["data"]["processed_path"]) / "news_base.parquet"
    save_to_parquet(merged_df, output_path)
    
    print("\n✅ News ingestion complete!")
    print("Next step: Run embedding generation pipeline.")


if __name__ == "__main__":
    main()

