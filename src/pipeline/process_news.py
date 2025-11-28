"""
SNAP Feature Engineering Pipeline for Titan V8
Phase 1.2: Transform News Text into Volatility Signals

Optimized for Apple M1 Max (MPS acceleration)

Features generated:
- shock_index: Keyword-based severity score
- news_count: Daily headline count
- sentiment_avg: Average sentiment per day
- news_pca_0..19: 20-dimensional news theme vectors
- novelty_score: Cosine distance from previous day's news

Usage:
    python -m src.pipeline.process_news
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str = "conf/base/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """
    Get optimal device for M1 Max acceleration.
    
    Returns:
        "mps" for Apple Silicon, "cpu" otherwise
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


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

def load_embedding_model(device: str) -> SentenceTransformer:
    """
    Load sentence transformer model optimized for device.
    
    Args:
        device: "mps", "cuda", or "cpu"
    
    Returns:
        SentenceTransformer model
    """
    print(f"  📦 Loading all-MiniLM-L6-v2 on {device.upper()}...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    return model


def generate_embeddings(
    model: SentenceTransformer,
    texts: list,
    device: str,
    batch_size: int = 256
) -> np.ndarray:
    """
    Generate embeddings for news texts with MPS optimization.
    
    Args:
        model: SentenceTransformer model
        texts: List of news texts
        device: Device to use
        batch_size: Batch size (256 works well on M1 Max)
    
    Returns:
        Numpy array of embeddings (n_texts, embedding_dim)
    """
    print(f"  🔢 Generating embeddings for {len(texts):,} texts...")
    print(f"     Device: {device.upper()}, Batch size: {batch_size}")
    
    # Convert to list and handle None values
    texts = [str(t) if t else "" for t in texts]
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True
    )
    
    return embeddings


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
    print(f"\n  📉 Fitting PCA with anti-leakage (cutoff: {train_cutoff})...")
    
    # Get train mask
    df_dates = pd.to_datetime(df["date"]).dt.tz_localize(None)
    train_mask = df_dates < pd.to_datetime(train_cutoff)
    
    train_count = train_mask.sum()
    total_count = len(df)
    
    print(f"     Train samples: {train_count:,} ({100*train_count/total_count:.1f}%)")
    print(f"     Test samples:  {total_count - train_count:,} ({100*(total_count-train_count)/total_count:.1f}%)")
    
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

def aggregate_daily_features(df: pd.DataFrame, pca_cols: list) -> pd.DataFrame:
    """
    Aggregate news features to daily ticker level.
    
    Aggregations:
    - shock_index: Sum of shock scores
    - news_count: Number of headlines
    - sentiment_avg: Mean sentiment
    - news_pca_X: Mean of PCA vectors
    
    Args:
        df: DataFrame with news features
        pca_cols: List of PCA column names
    
    Returns:
        Aggregated daily DataFrame
    """
    print("\n  📊 Aggregating to daily ticker level...")
    
    # Ensure date is date-only (no time)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["date"] = pd.to_datetime(df["date"])
    
    # Build aggregation dictionary
    agg_dict = {
        "shock_score": "sum",        # Sum of shock scores
        "sentiment_score": "mean",   # Average sentiment
        "raw_text": "count"          # Count of headlines
    }
    
    # Add PCA columns (mean)
    for col in pca_cols:
        agg_dict[col] = "mean"
    
    # Group by date and ticker
    daily = df.groupby(["date", "ticker"]).agg(agg_dict).reset_index()
    
    # Rename columns
    daily = daily.rename(columns={
        "shock_score": "shock_index",
        "sentiment_score": "sentiment_avg",
        "raw_text": "news_count"
    })
    
    print(f"     Aggregated to {len(daily):,} daily records")
    
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


def main():
    """Main entry point for SNAP feature engineering."""
    print("\n" + "=" * 60)
    print("🚀 TITAN V8 SNAP FEATURE ENGINEERING")
    print("    Phase 1.2: News → Volatility Signals")
    print("=" * 60)
    
    # =========================================
    # Device Selection (M1 Max Optimization)
    # =========================================
    device = get_device()
    print(f"\n🖥️  DEVICE: {device.upper()}")
    if device == "mps":
        print("   ✓ Apple Silicon MPS acceleration enabled!")
    elif device == "cuda":
        print("   ✓ NVIDIA CUDA acceleration enabled!")
    else:
        print("   ⚠️ Running on CPU (slower)")
    
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
    
    model = load_embedding_model(device)
    embeddings = generate_embeddings(
        model, 
        df["raw_text"].tolist(), 
        device=device,
        batch_size=256
    )
    
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
    
    daily_df = aggregate_daily_features(df, pca_cols)
    
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
    print(f"Device used: {device.upper()}")
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

