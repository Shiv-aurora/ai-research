"""
Scale-Up Phase: Multi-Universe Evaluation

This module contains infrastructure to test Titan V15 on large stock universes:
- High Octane 50: Most active/volatile stocks
- SP500 Sector Leaders 55: Blue-chip stocks across 11 GICS sectors
"""

from .config_universes import (
    MOST_ACTIVE_50,
    SP500_SECTOR_LEADERS_55,
    SECTOR_MAP_ACTIVE,
    SECTOR_MAP_SP500
)

__all__ = [
    'MOST_ACTIVE_50',
    'SP500_SECTOR_LEADERS_55',
    'SECTOR_MAP_ACTIVE',
    'SECTOR_MAP_SP500'
]

