"""
SNAP Feature Engineering Pipeline for Titan V8
Phase 1.2: Transform News Text into Volatility Signals

Two modes:
- LITE: TF-IDF + TruncatedSVD (fast, no GPU needed)
- FULL: Sentence-Transformers (requires more memory)

Features generated:
- shock_index: Keyword-based severity score
- news_count: Daily headline count
- sentiment_avg: Average sentiment per day
- news_pca_0..19: 20-dimensional news theme vectors
- novelty_score: Cosine distance from previous day's news

Phase 10 UPDATE: "Overnight News Split" Strategy
- Groups by 'effective_date' (not original publish date)
- effective_date = date when news impacts volatility
- After-hours news (>= 4 PM ET) impacts NEXT day
- This ensures we only use news the market hasn't priced in

Usage:
    python -m src.pipeline.process_news
    python -m src.pipeline.process_news --mode full  # Use sentence-transformers
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Check if we should use sentence-transformers (full mode)
USE_SENTENCE_TRANSFORMERS = "--mode" in sys.argv and "full" in sys.argv


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str = "conf/base/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_vectorization_mode() -> str:
    """
    Get vectorization mode based on command line args.
    
    Returns:
        "lite" for TF-IDF, "full" for sentence-transformers
    """
    if USE_SENTENCE_TRANSFORMERS:
        return "full"
    return "lite"


# =============================================================================
# STEP A: KEYWORD SCORING (The "Shock" Scalar)
# =============================================================================

# Severity keywords and weights
HIGH_SEVERITY_KEYWORDS = [
    'fraud', 'bankruptcy', 'sec investigation', 'default', 
    'sanction', 'crash', 'plunge', 'indictment', 'scandal',
    'collapse', 'crisis', 'halt', 'suspend'
]

MEDIUM_SEVERITY_KEYWORDS = [
    'downgrade', 'miss', 'hike', 'warning', 'lawsuit', 
    'resigns', 'cut', 'decline', 'fall', 'drop', 'concern',
    'investigation', 'probe', 'delay', 'recall'
]


def calculate_shock_score(text: str) -> float:
    """
    Calculate keyword-based shock score for news text.
    
    High severity keywords (weight 3): fraud, bankruptcy, etc.
    Medium severity keywords (weight 1): downgrade, miss, etc.
    
    Args:
        text: News headline/text
    
    Returns:
        Shock score (0-5, capped)
    """
    if not isinstance(text, str):
        return 0.0
    
    text_lower = text.lower()
    score = 0.0
    
    # High severity (weight 3)
    for keyword in HIGH_SEVERITY_KEYWORDS:
        if keyword in text_lower:
            score += 3.0
    
    # Medium severity (weight 1)
    for keyword in MEDIUM_SEVERITY_KEYWORDS:
        if keyword in text_lower:
            score += 1.0
    
    # Cap at 5.0
    return min(score, 5.0)


# =============================================================================
# STEP B: VECTORIZATION (The "Theme" Vector)
# =============================================================================

def generate_tfidf_embeddings(texts: list, n_components: int = 384) -> np.ndarray:
    """
    Generate TF-IDF embeddings with SVD reduction (LITE mode).
    
    This is a lightweight alternative to sentence-transformers that
    doesn't require GPU or heavy ML models.
    
    Args:
        texts: List of news texts
        n_components: Output dimension (matches MiniLM-L6 for compatibility)
    
    Returns:
        Numpy array of embeddings (n_texts, n_components)
    """
    print(f"  🔢 Generating TF-IDF embeddings for {len(texts):,} texts...")
    
    # Clean texts
    texts = [str(t) if t else "" for t in texts]
    
    # TF-IDF vectorization
    print("     Step 1/2: TF-IDF vectorization...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    print(f"     TF-IDF shape: {tfidf_matrix.shape}")
    
    # Reduce dimensions with TruncatedSVD
    print(f"     Step 2/2: SVD reduction to {n_components} dims...")
    n_comp = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix)
    
    print(f"     Explained variance: {svd.explained_variance_ratio_.sum():.1%}")
    
    return embeddings


def generate_sentence_embeddings(texts: list, batch_size: int = 256) -> np.ndarray:
    """
    Generate embeddings using sentence-transformers (FULL mode).
    
    Requires more memory but produces higher quality embeddings.
    
    Args:
        texts: List of news texts
        batch_size: Batch size for encoding
    
    Returns:
        Numpy array of embeddings
    """
    import torch
    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  📦 Loading all-MiniLM-L6-v2 on {device.upper()}...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print(f"  🔢 Generating embeddings for {len(texts):,} texts...")
    texts = [str(t) if t else "" for t in texts]
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True
    )
    
    return embeddings


def generate_embeddings(texts: list, mode: str = "lite") -> np.ndarray:
    """
    Generate text embeddings using specified mode.
    
    Args:
        texts: List of news texts
        mode: "lite" (TF-IDF) or "full" (sentence-transformers)
    
    Returns:
        Numpy array of embeddings
    """
    if mode == "full":
        return generate_sentence_embeddings(texts)
    else:
        return generate_tfidf_embeddings(texts)


# =============================================================================
# STEP C: DIMENSIONALITY REDUCTION (PCA with Anti-Leakage)
# =============================================================================

def fit_pca_with_antileakage(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    train_cutoff: str = "2022-01-01",
    n_components: int = 20
) -> tuple:
    """
    Fit PCA on training data only to prevent data leakage.
    
    STRICT ANTI-LEAKAGE:
    1. Split data: Train (date < cutoff), Rest (date >= cutoff)
    2. Fit PCA ONLY on Train embeddings
    3. Transform ALL embeddings
    
    Args:
        df: DataFrame with 'date' column
        embeddings: Full embedding matrix
        train_cutoff: Date string for train/test split
        n_components: Number of PCA components
    
    Returns:
        Tuple of (pca_model, transformed_embeddings)
    """
    from sklearn.decomposition import IncrementalPCA
    
    print(f"\n  📉 Fitting PCA with anti-leakage (cutoff: {train_cutoff})...")
    
    # Get train mask
    df_dates = pd.to_datetime(df["date"]).dt.tz_localize(None)
    train_mask = df_dates < pd.to_datetime(train_cutoff)
    
    train_count = train_mask.sum()
    total_count = len(df)
    
    print(f"     Train samples: {train_count:,} ({100*train_count/total_count:.1f}%)")
    print(f"     Test samples:  {total_count - train_count:,} ({100*(total_count-train_count)/total_count:.1f}%)")
    
    # Adjust n_components if needed
    max_components = min(n_components, embeddings.shape[1], train_count - 1)
    if max_components < n_components:
        print(f"     ⚠️ Adjusting n_components from {n_components} to {max_components}")
        n_components = max_components
    
    # Extract train embeddings
    train_embeddings = embeddings[train_mask]
    
    # Fit PCA on train only
    pca = IncrementalPCA(n_components=n_components, batch_size=1024)
    pca.fit(train_embeddings)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"     Explained variance: {100*explained_var:.1f}%")
    
    # Transform ALL embeddings
    print(f"  🔄 Transforming all embeddings...")
    pca_embeddings = pca.transform(embeddings)
    
    return pca, pca_embeddings


# =============================================================================
# STEP D: DAILY AGGREGATION (Ticker Level)
# =============================================================================

def aggregate_daily_features(df: pd.DataFrame, pca_cols: list, use_effective_date: bool = True) -> pd.DataFrame:
    """
    Aggregate news features to daily ticker level.
    
    Phase 10 UPDATE: Uses 'effective_date' instead of 'date' by default.
    This ensures proper causality - we only use news that hasn't been
    priced in yet at market open.
    
    Aggregations:
    - shock_index: Sum of shock scores
    - news_count: Number of headlines
    - sentiment_avg: Mean sentiment
    - news_pca_X: Mean of PCA vectors
    
    Args:
        df: DataFrame with news features
        pca_cols: List of PCA column names
        use_effective_date: If True, group by effective_date; else by date
    
    Returns:
        Aggregated daily DataFrame
    """
    print("\n  📊 Aggregating to daily ticker level...")
    
    df = df.copy()
    
    # Determine which date column to use
    if use_effective_date and 'effective_date' in df.columns:
        date_col = 'effective_date'
        print(f"     ✓ Using 'effective_date' (Overnight News Split)")
    else:
        date_col = 'date'
        print(f"     ℹ️ Using original 'date' (no effective_date found)")
    
    # Ensure date is date-only (no time)
    df["agg_date"] = pd.to_datetime(df[date_col]).dt.date
    df["agg_date"] = pd.to_datetime(df["agg_date"])
    
    # Build aggregation dictionary
    agg_dict = {
        "shock_score": "sum",        # Sum of shock scores
        "sentiment_score": "mean",   # Average sentiment
        "raw_text": "count"          # Count of headlines
    }
    
    # Add PCA columns (mean)
    for col in pca_cols:
        agg_dict[col] = "mean"
    
    # Also track publish hour distribution (for debugging)
    if 'publish_hour_et' in df.columns:
        agg_dict['publish_hour_et'] = lambda x: x.median()
    
    # Group by aggregation date and ticker
    daily = df.groupby(["agg_date", "ticker"]).agg(agg_dict).reset_index()
    
    # Rename columns
    daily = daily.rename(columns={
        "agg_date": "date",  # Final output uses "date" (= effective_date)
        "shock_score": "shock_index",
        "sentiment_score": "sentiment_avg",
        "raw_text": "news_count"
    })
    
    # Drop publish_hour_et if present (not needed downstream)
    if 'publish_hour_et' in daily.columns:
        daily = daily.drop(columns=['publish_hour_et'])
    
    print(f"     Aggregated to {len(daily):,} daily records")
    print(f"     Date range: {daily['date'].min()} to {daily['date'].max()}")
    
    return daily


# =============================================================================
# STEP E: NOVELTY SCORE (Time-Series Feature)
# =============================================================================

def calculate_novelty_scores(df: pd.DataFrame, pca_cols: list) -> pd.DataFrame:
    """
    Calculate novelty score as cosine distance from previous day.
    
    novelty_score = 1 - cosine_similarity(today_vector, yesterday_vector)
    
    Args:
        df: Daily aggregated DataFrame
        pca_cols: List of PCA column names
    
    Returns:
        DataFrame with novelty_score added
    """
    print("\n  🆕 Calculating novelty scores...")
    
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Initialize novelty column
    df["novelty_score"] = 0.0
    
    # Process each ticker
    for ticker in tqdm(df["ticker"].unique(), desc="Computing novelty"):
        ticker_mask = df["ticker"] == ticker
        ticker_indices = df[ticker_mask].index.tolist()
        
        if len(ticker_indices) < 2:
            continue
        
        # Get PCA vectors for this ticker
        ticker_vectors = df.loc[ticker_indices, pca_cols].values
        
        # Calculate novelty for each day (skip first)
        for i in range(1, len(ticker_indices)):
            today_vec = ticker_vectors[i].reshape(1, -1)
            yesterday_vec = ticker_vectors[i-1].reshape(1, -1)
            
            # Handle zero vectors
            if np.allclose(today_vec, 0) or np.allclose(yesterday_vec, 0):
                novelty = 0.0
            else:
                cos_sim = cosine_similarity(today_vec, yesterday_vec)[0, 0]
                novelty = 1 - cos_sim
            
            df.loc[ticker_indices[i], "novelty_score"] = novelty
    
    # Fill any remaining NaN
    df["novelty_score"] = df["novelty_score"].fillna(0.0)
    
    return df


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


def main(mode: str = None, use_effective_date: bool = True):
    """
    Main entry point for SNAP feature engineering.
    
    Args:
        mode: "lite" (TF-IDF) or "full" (sentence-transformers). Auto-detected if None.
        use_effective_date: If True, use effective_date for aggregation (Overnight News Split)
    """
    print("\n" + "=" * 60)
    print("🚀 TITAN V8 SNAP FEATURE ENGINEERING")
    print("    Phase 1.2: News → Volatility Signals")
    if use_effective_date:
        print("    Phase 10: Overnight News Split (effective_date)")
    print("=" * 60)
    
    # =========================================
    # Mode Selection
    # =========================================
    if mode is None:
        mode = get_vectorization_mode()
    print(f"\n🖥️  MODE: {mode.upper()}")
    if mode == "full":
        print("   ✓ Using sentence-transformers (high quality)")
    else:
        print("   ✓ Using TF-IDF + SVD (fast, lightweight)")
        print("   ℹ️  Run with --mode full for sentence-transformers")
    
    # Load configuration
    config = load_config()
    
    # =========================================
    # Load Input Data
    # =========================================
    input_path = Path(config["data"]["processed_path"]) / "news_base.parquet"
    print(f"\n📂 Loading: {input_path}")
    
    df = pd.read_parquet(input_path)
    print(f"   ✓ Loaded {len(df):,} news items")
    print(f"   ✓ Tickers: {df['ticker'].unique().tolist()}")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    
    # =========================================
    # STEP A: Keyword Scoring
    # =========================================
    print("\n" + "-" * 50)
    print("📌 STEP A: Keyword Scoring (Shock Score)")
    print("-" * 50)
    
    print("  ⚡ Calculating shock scores...")
    df["shock_score"] = df["raw_text"].apply(calculate_shock_score)
    
    shock_stats = df["shock_score"].describe()
    print(f"   ✓ Shock score stats:")
    print(f"      Mean: {shock_stats['mean']:.3f}")
    print(f"      Max:  {shock_stats['max']:.1f}")
    print(f"      Non-zero: {(df['shock_score'] > 0).sum():,} ({100*(df['shock_score'] > 0).mean():.1f}%)")
    
    # =========================================
    # STEP B: Vectorization
    # =========================================
    print("\n" + "-" * 50)
    print("📌 STEP B: Vectorization (Embeddings)")
    print("-" * 50)
    
    embeddings = generate_embeddings(df["raw_text"].tolist(), mode=mode)
    
    print(f"   ✓ Embedding shape: {embeddings.shape}")
    
    # =========================================
    # STEP C: PCA Dimensionality Reduction
    # =========================================
    print("\n" + "-" * 50)
    print("📌 STEP C: PCA Reduction (Anti-Leakage)")
    print("-" * 50)
    
    pca, pca_embeddings = fit_pca_with_antileakage(
        df, embeddings, 
        train_cutoff="2022-01-01",
        n_components=20
    )
    
    # Add PCA columns
    pca_cols = [f"news_pca_{i}" for i in range(20)]
    for i, col in enumerate(pca_cols):
        df[col] = pca_embeddings[:, i]
    
    print(f"   ✓ Added {len(pca_cols)} PCA columns")
    
    # =========================================
    # STEP D: Daily Aggregation
    # =========================================
    print("\n" + "-" * 50)
    print("📌 STEP D: Daily Aggregation")
    print("-" * 50)
    
    daily_df = aggregate_daily_features(df, pca_cols, use_effective_date=use_effective_date)
    
    # =========================================
    # STEP E: Novelty Score
    # =========================================
    print("\n" + "-" * 50)
    print("📌 STEP E: Novelty Score")
    print("-" * 50)
    
    daily_df = calculate_novelty_scores(daily_df, pca_cols)
    
    novelty_stats = daily_df["novelty_score"].describe()
    print(f"   ✓ Novelty score stats:")
    print(f"      Mean: {novelty_stats['mean']:.3f}")
    print(f"      Std:  {novelty_stats['std']:.3f}")
    
    # =========================================
    # Save Output
    # =========================================
    output_path = Path(config["data"]["processed_path"]) / "news_features.parquet"
    save_to_parquet(daily_df, output_path)
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    print("📊 SNAP FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Mode used: {mode.upper()}")
    print(f"Input rows:  {len(df):,}")
    print(f"Output rows: {len(daily_df):,}")
    print(f"Output shape: {daily_df.shape}")
    print(f"\nOutput columns:")
    print(f"  {list(daily_df.columns)}")
    
    print(f"\n📋 Top 5 rows:")
    display_cols = ["date", "ticker", "shock_index", "news_count", 
                    "sentiment_avg", "novelty_score", "news_pca_0", "news_pca_1"]
    print(daily_df[display_cols].head().to_string())
    
    print("=" * 60)
    print("\n✅ SNAP feature engineering complete!")
    print("Next step: Merge with price features for modeling.")


if __name__ == "__main__":
    main()

