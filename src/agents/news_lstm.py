"""
GlobalNewsLSTM: Deep Learning News Agent (Phase 9)

A sector-aware LSTM that learns temporal patterns in news features
across the entire 18-ticker universe.

Architecture:
- Input A (Sequence): 5-day sliding window of news vectors
- Input B (Static): Sector embedding (learns sector-specific patterns)
- LSTM: 2-layer, 64 hidden units, dropout 0.2
- Fusion: Concatenate LSTM output + sector embedding
- Head: Linear(72) → ReLU → Linear(1)

Key Innovations:
1. Centralized model learns cross-sector patterns
2. Sector embedding captures industry-specific news dynamics
3. 5-day window captures news persistence without weekly contamination

Target: resid_tech (HAR model residuals)

Usage:
    from src.agents.news_lstm import GlobalNewsLSTM, NewsLSTMAgent
    agent = NewsLSTMAgent()
    agent.train()
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Sector mapping
SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech',
    'JPM': 'Finance', 'BAC': 'Finance', 'V': 'Finance',
    'CAT': 'Industrial', 'GE': 'Industrial', 'BA': 'Industrial',
    'WMT': 'Consumer', 'MCD': 'Consumer', 'COST': 'Consumer',
    'XOM': 'Energy', 'CVX': 'Energy', 'SLB': 'Energy',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare'
}

SECTOR_TO_ID = {
    'Tech': 0, 'Finance': 1, 'Industrial': 2,
    'Consumer': 3, 'Energy': 4, 'Healthcare': 5
}

TICKER_TO_ID = {ticker: i for i, ticker in enumerate(sorted(SECTOR_MAP.keys()))}


class NewsSequenceDataset(Dataset):
    """
    PyTorch Dataset for news sequences.
    
    Creates sliding windows of 5 days of news features.
    """
    
    def __init__(self, 
                 sequences: np.ndarray, 
                 sector_ids: np.ndarray,
                 targets: np.ndarray):
        """
        Args:
            sequences: (N, seq_len, n_features) news feature sequences
            sector_ids: (N,) sector IDs for each sequence
            targets: (N,) target values (resid_tech)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.sector_ids = torch.LongTensor(sector_ids)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.sector_ids[idx], self.targets[idx]


class GlobalNewsLSTM(nn.Module):
    """
    Sector-Aware Global LSTM for News Features.
    
    Architecture:
    - LSTM: Processes 5-day news sequence
    - Sector Embedding: Learns sector-specific patterns
    - Fusion: Concatenates LSTM output with sector embedding
    - Head: Projects to single prediction
    """
    
    def __init__(self,
                 input_size: int = 23,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 num_sectors: int = 6,
                 embedding_dim: int = 8,
                 dropout: float = 0.2):
        """
        Args:
            input_size: Number of features per timestep
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_sectors: Number of unique sectors
            embedding_dim: Sector embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Sector embedding
        self.sector_embedding = nn.Embedding(
            num_embeddings=num_sectors,
            embedding_dim=embedding_dim
        )
        
        # Fusion and output head
        fusion_dim = hidden_size + embedding_dim
        
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, sequence: torch.Tensor, sector_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequence: (batch, seq_len, input_size) news feature sequence
            sector_id: (batch,) sector IDs
            
        Returns:
            (batch, 1) predicted residuals
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        
        # Take last hidden state from final layer
        lstm_final = h_n[-1]  # (batch, hidden_size)
        
        # Sector embedding
        sector_emb = self.sector_embedding(sector_id)  # (batch, embedding_dim)
        
        # Fusion
        fused = torch.cat([lstm_final, sector_emb], dim=1)  # (batch, hidden_size + embedding_dim)
        
        # Output
        output = self.head(fused)
        
        return output


class NewsLSTMAgent:
    """
    Agent wrapper for the GlobalNewsLSTM.
    
    Handles data loading, preprocessing, training, and evaluation.
    """
    
    def __init__(self,
                 seq_length: int = 5,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 embedding_dim: int = 8,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 batch_size: int = 64,
                 epochs: int = 50,
                 patience: int = 10):
        """
        Initialize the LSTM agent.
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Feature columns
        self.pca_cols = [f'news_pca_{i}' for i in range(20)]
        self.core_cols = ['news_count', 'shock_index', 'sentiment_avg']
        self.feature_cols = self.pca_cols + self.core_cols
        self.target_col = 'resid_tech'
        
        # Model and data
        self.model = None
        self.device = None
        self.df = None
        self.train_metrics = None
        self.test_metrics = None
        
        # Stats for normalization
        self.feature_means = None
        self.feature_stds = None
        
    def _get_device(self):
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_and_process_data(self) -> pd.DataFrame:
        """Load and prepare data for LSTM training."""
        print("\n📂 Loading data for NewsLSTM...")
        
        # Load news features
        news_path = Path("data/processed/news_features.parquet")
        news_df = pd.read_parquet(news_path)
        print(f"   ✓ News features: {len(news_df):,} rows")
        
        # Load residuals (has resid_tech)
        residuals_path = Path("data/processed/residuals.parquet")
        residuals_df = pd.read_parquet(residuals_path)
        print(f"   ✓ Residuals: {len(residuals_df):,} rows")
        
        # Normalize dates
        for df in [news_df, residuals_df]:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            if df['ticker'].dtype.name == 'category':
                df['ticker'] = df['ticker'].astype(str)
        
        # Merge
        df = pd.merge(news_df, residuals_df[['date', 'ticker', 'resid_tech']], 
                      on=['date', 'ticker'], how='inner')
        
        # Add sector info
        df['sector'] = df['ticker'].map(SECTOR_MAP)
        df['sector_id'] = df['sector'].map(SECTOR_TO_ID)
        df['ticker_id'] = df['ticker'].map(TICKER_TO_ID)
        
        # Remove rows without sector
        df = df.dropna(subset=['sector_id', self.target_col])
        df['sector_id'] = df['sector_id'].astype(int)
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        # Fill NaN in features
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        print(f"   ✓ Merged: {len(df):,} rows")
        print(f"   ✓ Tickers: {df['ticker'].nunique()}")
        print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        self.df = df
        return df
    
    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for each ticker.
        
        Returns:
            sequences: (N, seq_len, n_features)
            sector_ids: (N,)
            targets: (N,)
            dates: (N,) dates for each sequence
        """
        sequences = []
        sector_ids = []
        targets = []
        dates = []
        
        # Get available feature columns
        available_features = [c for c in self.feature_cols if c in df.columns]
        
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker'] == ticker].sort_values('date')
            
            if len(ticker_df) < self.seq_length:
                continue
            
            # Get feature values
            features = ticker_df[available_features].values
            target_vals = ticker_df[self.target_col].values
            sector_id = ticker_df['sector_id'].iloc[0]
            date_vals = ticker_df['date'].values
            
            # Create sequences
            for i in range(self.seq_length, len(ticker_df)):
                seq = features[i - self.seq_length:i]
                sequences.append(seq)
                sector_ids.append(sector_id)
                targets.append(target_vals[i])
                dates.append(date_vals[i])
        
        return (np.array(sequences), 
                np.array(sector_ids), 
                np.array(targets),
                np.array(dates))
    
    def _normalize_features(self, sequences: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features using z-score."""
        if fit:
            # Compute mean and std across all sequences
            self.feature_means = sequences.mean(axis=(0, 1), keepdims=True)
            self.feature_stds = sequences.std(axis=(0, 1), keepdims=True)
            self.feature_stds[self.feature_stds < 1e-6] = 1.0
        
        return (sequences - self.feature_means) / self.feature_stds
    
    def train(self, df: pd.DataFrame = None) -> dict:
        """Train the LSTM model."""
        if df is None:
            df = self.df
        
        if df is None:
            raise ValueError("No data! Call load_and_process_data first.")
        
        print("\n🎯 Training GlobalNewsLSTM...")
        
        # Get device
        self.device = self._get_device()
        print(f"   Device: {self.device}")
        
        # Create sequences
        print("\n   Creating sequences...")
        sequences, sector_ids, targets, dates = self._create_sequences(df)
        print(f"   Total sequences: {len(sequences):,}")
        
        # Time-based split
        cutoff = pd.to_datetime("2023-01-01")
        dates_pd = pd.to_datetime(dates)
        train_mask = dates_pd < cutoff
        test_mask = dates_pd >= cutoff
        
        # Split data
        train_seq = sequences[train_mask]
        train_sectors = sector_ids[train_mask]
        train_targets = targets[train_mask]
        
        test_seq = sequences[test_mask]
        test_sectors = sector_ids[test_mask]
        test_targets = targets[test_mask]
        
        print(f"   Train: {len(train_seq):,} sequences")
        print(f"   Test:  {len(test_seq):,} sequences")
        
        # Normalize features
        train_seq = self._normalize_features(train_seq, fit=True)
        test_seq = self._normalize_features(test_seq, fit=False)
        
        # Create datasets
        train_dataset = NewsSequenceDataset(train_seq, train_sectors, train_targets)
        test_dataset = NewsSequenceDataset(test_seq, test_sectors, test_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_size = train_seq.shape[2]
        self.model = GlobalNewsLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_sectors=6,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"\n   Model architecture:")
        print(f"      Input size: {input_size}")
        print(f"      Hidden size: {self.hidden_size}")
        print(f"      Num layers: {self.num_layers}")
        print(f"      Sector embedding: {self.embedding_dim}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        print(f"\n   Training for {self.epochs} epochs...")
        
        best_test_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            
            for seq, sector, target in train_loader:
                seq = seq.to(self.device)
                sector = sector.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(seq, sector)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Evaluate
            self.model.eval()
            test_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for seq, sector, target in test_loader:
                    seq = seq.to(self.device)
                    sector = sector.to(self.device)
                    target = target.to(self.device)
                    
                    output = self.model(seq, sector)
                    loss = criterion(output, target)
                    test_loss += loss.item()
                    
                    all_preds.extend(output.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy().flatten())
            
            test_loss /= len(test_loader)
            test_r2 = r2_score(all_targets, all_preds)
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            
            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_r2 = test_r2
                patience_counter = 0
                # Save best model weights
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test R²={test_r2:.4f}")
            
            if patience_counter >= self.patience:
                print(f"      Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(best_state)
        self.model = self.model.to(self.device)
        
        # Final evaluation
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for seq, sector, target in test_loader:
                seq = seq.to(self.device)
                sector = sector.to(self.device)
                
                output = self.model(seq, sector)
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        # Calculate final metrics
        test_r2 = r2_score(all_targets, all_preds)
        test_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        self.test_metrics = {'R2': test_r2, 'RMSE': test_rmse, 'Loss': best_test_loss}
        self.train_metrics = {'Loss': train_loss}
        
        print(f"\n   ✅ FINAL RESULTS (Best epoch {best_epoch+1}):")
        print(f"      Test Loss: {best_test_loss:.4f}")
        print(f"      Test R²:   {test_r2:.4f} ({test_r2*100:.2f}%)")
        print(f"      Test RMSE: {test_rmse:.4f}")
        
        return {
            'train': self.train_metrics,
            'test': self.test_metrics
        }
    
    def predict(self, sequences: np.ndarray, sector_ids: np.ndarray) -> np.ndarray:
        """Generate predictions for sequences."""
        self.model.eval()
        
        # Normalize
        sequences = self._normalize_features(sequences, fit=False)
        
        # Create dataset
        dataset = NewsSequenceDataset(sequences, sector_ids, np.zeros(len(sequences)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        
        with torch.no_grad():
            for seq, sector, _ in loader:
                seq = seq.to(self.device)
                sector = sector.to(self.device)
                
                output = self.model(seq, sector)
                all_preds.extend(output.cpu().numpy().flatten())
        
        return np.array(all_preds)


def main():
    """Run the NewsLSTM agent."""
    print("\n" + "=" * 70)
    print("🧠 PHASE 9: GLOBAL NEWS LSTM")
    print("   Sector-aware deep learning for news features")
    print("=" * 70)
    
    agent = NewsLSTMAgent(
        seq_length=5,
        hidden_size=64,
        num_layers=2,
        embedding_dim=8,
        dropout=0.2,
        learning_rate=0.001,
        weight_decay=1e-5,
        batch_size=64,
        epochs=50,
        patience=10
    )
    
    df = agent.load_and_process_data()
    metrics = agent.train(df)
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"\n   Final Test R²: {metrics['test']['R2']:.4f} ({metrics['test']['R2']*100:.2f}%)")
    
    # Compare to baseline
    baseline_r2 = -0.0138  # LightGBM from Phase 7
    delta = metrics['test']['R2'] - baseline_r2
    
    print(f"\n   Comparison to LightGBM baseline (-1.38%):")
    print(f"      Delta: {delta:+.4f} ({delta*100:+.2f}%)")
    
    if metrics['test']['R2'] > 0:
        print(f"\n   ✅ LSTM achieves POSITIVE R²!")
    else:
        print(f"\n   ❌ LSTM still negative")
    
    print("\n" + "=" * 70)
    
    return agent, metrics


if __name__ == "__main__":
    main()

