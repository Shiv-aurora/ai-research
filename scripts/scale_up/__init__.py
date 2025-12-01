"""
RIVE Scale-Up Phase: Multi-Universe Evaluation

This module contains infrastructure to test RIVE on large stock universes:
- Top 50 Active: Most actively traded U.S. stocks
- GICS Balanced 55: S&P 500 sector-balanced (5 per sector)
"""

from .config_universes import (
    TOP_50_ACTIVE,
    GICS_BALANCED_55,
    SECTOR_MAP_ACTIVE,
    SECTOR_MAP_SP500
)

__all__ = [
    'TOP_50_ACTIVE',
    'GICS_BALANCED_55',
    'SECTOR_MAP_ACTIVE',
    'SECTOR_MAP_SP500'
]
