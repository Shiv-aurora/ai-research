"""
Run Deep Learning News Agent (Phase 9)

Trains a Global Sector-Aware LSTM on news features to predict
volatility residuals from the HAR model.

Key Features:
- 5-day sliding window of news features
- Sector embedding learns industry-specific patterns
- 2-layer LSTM with dropout
- AdamW optimizer with learning rate scheduling

Usage:
    python scripts/run_deep_news.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def main():
    """Run the deep news agent training."""
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("🧠 PHASE 9: DEEP LEARNING NEWS AGENT")
    print("   Global Sector-Aware LSTM for News Features")
    print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check PyTorch and device
    print(f"\n📊 Environment:")
    print(f"   PyTorch version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        device = "mps (Apple Silicon)"
    elif torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
    print(f"   Device: {device}")
    
    # Import and run agent
    from src.agents.news_lstm import NewsLSTMAgent
    
    # Initialize agent with hyperparameters
    agent = NewsLSTMAgent(
        seq_length=5,           # 5-day sliding window
        hidden_size=64,         # LSTM hidden dimension
        num_layers=2,           # 2-layer LSTM
        embedding_dim=8,        # Sector embedding size
        dropout=0.2,            # Regularization
        learning_rate=0.001,    # AdamW learning rate
        weight_decay=1e-5,      # L2 regularization
        batch_size=64,          # Batch size
        epochs=50,              # Max epochs
        patience=10             # Early stopping patience
    )
    
    # Load data
    df = agent.load_and_process_data()
    
    # Train model
    metrics = agent.train(df)
    
    # Results comparison
    print("\n" + "=" * 70)
    print("📊 PHASE 9 RESULTS COMPARISON")
    print("=" * 70)
    
    # Baselines
    lgbm_baseline = -0.0138  # LightGBM from Phase 7
    sector_baseline = -0.0249  # Sector News Agent from Phase 8
    
    lstm_r2 = metrics['test']['R2']
    
    print(f"\n   {'Model':<35} {'Test R²':>12}")
    print("   " + "-" * 49)
    print(f"   {'LightGBM (Phase 7)':<35} {lgbm_baseline:>12.4f}")
    print(f"   {'Sector LightGBM (Phase 8)':<35} {sector_baseline:>12.4f}")
    print(f"   {'🧠 Global LSTM (Phase 9)':<35} {lstm_r2:>12.4f}")
    print("   " + "-" * 49)
    
    # Delta
    delta_lgbm = lstm_r2 - lgbm_baseline
    delta_sector = lstm_r2 - sector_baseline
    
    print(f"\n   Improvement vs LightGBM:  {delta_lgbm:+.4f} ({delta_lgbm*100:+.2f}%)")
    print(f"   Improvement vs Sector:    {delta_sector:+.4f} ({delta_sector*100:+.2f}%)")
    
    # Verdict
    print("\n" + "=" * 70)
    print("🏆 VERDICT")
    print("=" * 70)
    
    if lstm_r2 > 0.05:
        print(f"""
   ✅ LSTM ACHIEVES SIGNIFICANT POSITIVE R²!
   
   Test R²: {lstm_r2:.4f} ({lstm_r2*100:.2f}%)
   
   The deep learning approach successfully learns temporal
   patterns in news features that traditional methods miss.
        """)
    elif lstm_r2 > 0:
        print(f"""
   ⚠️ LSTM IS POSITIVE BUT MARGINAL
   
   Test R²: {lstm_r2:.4f} ({lstm_r2*100:.2f}%)
   
   The LSTM improves over LightGBM but the signal is weak.
   News may have limited predictive power for residuals.
        """)
    else:
        print(f"""
   ❌ LSTM STILL NEGATIVE
   
   Test R²: {lstm_r2:.4f} ({lstm_r2*100:.2f}%)
   
   Even deep learning cannot extract a positive signal.
   This confirms news has LIMITED predictive power for
   next-day volatility residuals.
   
   RECOMMENDATION:
   - Accept that news improves same-day vol (40% R²)
   - But cannot predict next-day residuals
   - Focus on HAR + VIX + Calendar for predictions
        """)
    
    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n   Duration: {duration}")
    print(f"   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return agent, metrics


if __name__ == "__main__":
    main()

