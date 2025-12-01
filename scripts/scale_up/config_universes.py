"""
Stock Universe Configurations for RIVE Scale-Up Phase

Contains two universes for RIVE (Regime-Integrated Volatility Ensemble) evaluation:
1. TOP_50_ACTIVE: High volatility, social-driven stocks
2. GICS_BALANCED_55: Blue-chip stocks across 11 GICS sectors
"""

# =============================================================================
# UNIVERSE A: TOP 50 MOST ACTIVE U.S. STOCKS
# =============================================================================
# Most actively traded stocks - includes crypto miners, meme stocks, speculative tech
# Expected characteristics: High volatility, Reddit/social influence, variable liquidity

TOP_50_ACTIVE = [
    'NVDA', 'HBI', 'INTC', 'BITF', 'BBAI', 'OPEN', 'BMNR', 'CLSK', 'PLUG', 'SNAP',
    'AAL', 'TSLA', 'CIFR', 'ONDS', 'MARA', 'WULF', 'NIO', 'F', 'SOFI', 'VALE',
    'GOOGL', 'BTG', 'GRAB', 'ACHR', 'PFE', 'CRCL', 'DNN', 'IPG', 'IBRX', 'RIOT',
    'AMZN', 'AAPL', 'AG', 'WBD', 'APLD', 'NGD', 'GOOG', 'IREN', 'NU', 'RIG',
    'BAC', 'AMD', 'RIVN', 'SMR', 'TMC', 'PLTR', 'CDE', 'ITUB', 'AGNC', 'SPY'
]

# Sector mapping for High Octane stocks (approximate - many are speculative)
SECTOR_MAP_ACTIVE = {
    # Info Tech
    'NVDA': 'Info Tech', 'INTC': 'Info Tech', 'AAPL': 'Info Tech', 'AMD': 'Info Tech',
    'PLTR': 'Info Tech', 'GOOG': 'Info Tech', 'GOOGL': 'Info Tech',
    # Crypto/Blockchain (classify as Info Tech)
    'BITF': 'Info Tech', 'CLSK': 'Info Tech', 'CIFR': 'Info Tech', 'MARA': 'Info Tech',
    'WULF': 'Info Tech', 'RIOT': 'Info Tech', 'APLD': 'Info Tech', 'IREN': 'Info Tech',
    # Consumer Discretionary
    'TSLA': 'Consumer Disc', 'AMZN': 'Consumer Disc', 'NIO': 'Consumer Disc',
    'RIVN': 'Consumer Disc', 'F': 'Consumer Disc', 'GRAB': 'Consumer Disc',
    # Communication Services
    'SNAP': 'Comm Services', 'WBD': 'Comm Services',
    # Financials
    'SOFI': 'Financials', 'BAC': 'Financials', 'NU': 'Financials', 'ITUB': 'Financials',
    'AGNC': 'Real Estate',  # REIT
    # Industrials
    'AAL': 'Industrials', 'ACHR': 'Industrials', 'PLUG': 'Industrials', 'GE': 'Industrials',
    # Materials/Mining
    'VALE': 'Materials', 'BTG': 'Materials', 'AG': 'Materials', 'NGD': 'Materials',
    'CDE': 'Materials', 'DNN': 'Materials',
    # Energy
    'RIG': 'Energy',
    # Health Care
    'PFE': 'Health Care', 'IBRX': 'Health Care',
    # Utilities
    'SMR': 'Utilities',
    # Real Estate
    'OPEN': 'Real Estate',
    # Other/Speculative
    'HBI': 'Consumer Disc', 'BBAI': 'Info Tech', 'BMNR': 'Materials',
    'ONDS': 'Info Tech', 'CRCL': 'Materials', 'IPG': 'Comm Services',
    'TMC': 'Materials', 'SPY': 'Index'
}


# =============================================================================
# UNIVERSE B: S&P 500 GICS SECTOR-BALANCED 55
# =============================================================================
# 5 blue-chip stocks per GICS sector - institutional quality, high liquidity
# Expected characteristics: Moderate volatility, traditional news, complete data

GICS_BALANCED_55 = [
    # Information Technology (5)
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL',
    # Health Care (5)
    'LLY', 'UNH', 'JNJ', 'MRK', 'ABBV',
    # Financials (5)
    'BRK.B', 'JPM', 'V', 'MA', 'BAC',
    # Consumer Discretionary (5)
    'AMZN', 'TSLA', 'HD', 'MCD', 'BKNG',
    # Communication Services (5)
    'GOOGL', 'GOOG', 'META', 'NFLX', 'TMUS',
    # Industrials (5)
    'CAT', 'UNP', 'GE', 'UBER', 'RTX',
    # Consumer Staples (5)
    'WMT', 'PG', 'COST', 'KO', 'PEP',
    # Energy (5)
    'XOM', 'CVX', 'COP', 'WMB', 'EOG',
    # Utilities (5)
    'NEE', 'SO', 'DUK', 'CEG', 'AEP',
    # Materials (5)
    'LIN', 'SHW', 'FCX', 'ECL', 'NEM',
    # Real Estate (5)
    'PLD', 'AMT', 'EQIX', 'WELL', 'SPG'
]

# Complete sector mapping for S&P 500 leaders
SECTOR_MAP_SP500 = {
    # Information Technology
    'AAPL': 'Info Tech', 'MSFT': 'Info Tech', 'NVDA': 'Info Tech',
    'AVGO': 'Info Tech', 'ORCL': 'Info Tech',
    # Health Care
    'LLY': 'Health Care', 'UNH': 'Health Care', 'JNJ': 'Health Care',
    'MRK': 'Health Care', 'ABBV': 'Health Care',
    # Financials
    'BRK.B': 'Financials', 'JPM': 'Financials', 'V': 'Financials',
    'MA': 'Financials', 'BAC': 'Financials',
    # Consumer Discretionary
    'AMZN': 'Consumer Disc', 'TSLA': 'Consumer Disc', 'HD': 'Consumer Disc',
    'MCD': 'Consumer Disc', 'BKNG': 'Consumer Disc',
    # Communication Services
    'GOOGL': 'Comm Services', 'GOOG': 'Comm Services', 'META': 'Comm Services',
    'NFLX': 'Comm Services', 'TMUS': 'Comm Services',
    # Industrials
    'CAT': 'Industrials', 'UNP': 'Industrials', 'GE': 'Industrials',
    'UBER': 'Industrials', 'RTX': 'Industrials',
    # Consumer Staples
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'COST': 'Consumer Staples',
    'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'WMB': 'Energy', 'EOG': 'Energy',
    # Utilities
    'NEE': 'Utilities', 'SO': 'Utilities', 'DUK': 'Utilities',
    'CEG': 'Utilities', 'AEP': 'Utilities',
    # Materials
    'LIN': 'Materials', 'SHW': 'Materials', 'FCX': 'Materials',
    'ECL': 'Materials', 'NEM': 'Materials',
    # Real Estate
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'EQIX': 'Real Estate',
    'WELL': 'Real Estate', 'SPG': 'Real Estate'
}


# =============================================================================
# SECTOR LISTS (for analysis)
# =============================================================================

GICS_SECTORS = [
    'Info Tech',
    'Health Care',
    'Financials',
    'Consumer Disc',
    'Comm Services',
    'Industrials',
    'Consumer Staples',
    'Energy',
    'Utilities',
    'Materials',
    'Real Estate'
]

# Universe metadata
UNIVERSE_METADATA = {
    'Top_50_Active': {
        'name': 'Top 50 Most Active U.S. Stocks',
        'tickers': TOP_50_ACTIVE,
        'sector_map': SECTOR_MAP_ACTIVE,
        'description': 'Most actively traded stocks - crypto miners, meme stocks, speculative tech',
        'expected_volatility': 'Very High',
        'news_source': 'Social/Reddit heavy',
        'performance': '61.12% R²'
    },
    'GICS_Balanced_55': {
        'name': 'S&P 500 GICS Sector-Balanced 55',
        'tickers': GICS_BALANCED_55,
        'sector_map': SECTOR_MAP_SP500,
        'description': 'Blue-chip stocks across 11 GICS sectors',
        'expected_volatility': 'Moderate',
        'news_source': 'Traditional financial news',
        'performance': '22.44% R²'
    }
}


def get_universe(name: str) -> dict:
    """Get universe configuration by name."""
    if name not in UNIVERSE_METADATA:
        raise ValueError(f"Unknown universe: {name}. Available: {list(UNIVERSE_METADATA.keys())}")
    return UNIVERSE_METADATA[name]


def get_sector(ticker: str, universe: str = 'GICS_Balanced_55') -> str:
    """Get sector for a ticker from specified universe."""
    sector_map = UNIVERSE_METADATA[universe]['sector_map']
    return sector_map.get(ticker, 'Unknown')


if __name__ == "__main__":
    print("=" * 60)
    print("RIVE - SCALE-UP UNIVERSE CONFIGURATIONS")
    print("=" * 60)
    
    for universe_name, metadata in UNIVERSE_METADATA.items():
        print(f"\n📊 {metadata['name']}")
        print(f"   Tickers: {len(metadata['tickers'])}")
        print(f"   Description: {metadata['description']}")
        print(f"   Expected Volatility: {metadata['expected_volatility']}")
        print(f"   News Source: {metadata['news_source']}")
        
        # Count by sector
        sector_counts = {}
        for ticker in metadata['tickers']:
            sector = metadata['sector_map'].get(ticker, 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        print(f"   Sectors: {len(sector_counts)}")
        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            print(f"      {sector}: {count} stocks")

