"""
Route Configurations for ETH Empirical Backtest.

Route A (Conservative): CAGR 25-30%, Leverage 1.0-1.2, MaxDD ≤30%, Sharpe ≥0.9
Route B (Balanced): CAGR 35-40%, Leverage 1.3-1.5, MaxDD ≤40%, Sharpe ≥0.8
Route C (Aggressive): CAGR 40-45%, Leverage 2.0-2.5, MaxDD ≤55%
"""

# =============================================================================
# Route A: Conservative (穩健型)
# Target: CAGR 25-30%, MaxDD ≤30%, Sharpe ≥0.9
# =============================================================================

ROUTE_A_CONFIG = {
    'name': 'Route A (Conservative)',
    'description': 'Stable returns with strict risk control',
    
    # Leverage and Position Sizing
    'leverage': 1.0,
    'max_leverage': 1.2,
    'base_position_size': 0.8,  # Conservative base position
    
    # Factor Weights (sum to 1.0)
    'factor_weights': {
        'trend_persistence': 0.35,
        'chop_filter': 0.25,
        'regime_sizing': 0.25,
        'vol_breakout': 0.0,      # Disabled for conservative
        'crash_protection': 0.15,
    },
    
    # Factor Parameters
    'trend_persistence_params': {
        'adx_period': 14,
        'adx_threshold': 30,      # Higher threshold = more selective
        'hurst_period': 20,
    },
    
    'chop_filter_params': {
        'chop_period': 14,
        'chop_threshold': 55,     # Lower threshold = more filtering
    },
    
    'regime_params': {
        'adx_period': 14,
        'sma_period': 50,
        'regime_multipliers': {
            0: 0.3,    # RANGING: very low exposure
            1: 1.0,    # TRENDING_UP: full exposure
            -1: 1.0,   # TRENDING_DOWN: full exposure
            2: 0.5,    # VOLATILE: reduced
        },
    },
    
    'crash_protection_params': {
        'vol_lookback': 20,
        'crash_vol_mult': 1.5,    # Stricter crash detection
        'momentum_lookback': 24,
    },
    
    # Signal Generation
    'signal_params': {
        'momentum_lookback': 48,
        'ema_fast': 12,
        'ema_slow': 26,
        'donchian_period': 20,
        'holding_period_min': 12,  # Minimum 12 hours
    },
    
    # Risk Management
    'risk_params': {
        'max_drawdown_target': 0.30,
        'volatility_target': 0.25,
        'atr_stop_mult': 2.5,
    },
    
    # Optimization Bounds
    'opt_bounds': {
        'leverage': (1.0, 1.2),
        'adx_threshold': (25, 40),
        'chop_threshold': (50, 65),
        'momentum_lookback': (24, 72),
    },
}


# =============================================================================
# Route B: Balanced (折衷型)
# Target: CAGR 35-40%, MaxDD ≤40%, Sharpe ≥0.8
# =============================================================================

ROUTE_B_CONFIG = {
    'name': 'Route B (Balanced)',
    'description': 'Balanced risk-return with moderate leverage',
    
    # Leverage and Position Sizing
    'leverage': 1.3,
    'max_leverage': 1.5,
    'base_position_size': 0.9,
    
    # Factor Weights
    'factor_weights': {
        'trend_persistence': 0.30,
        'chop_filter': 0.20,
        'regime_sizing': 0.25,
        'vol_breakout': 0.10,     # Moderate breakout exposure
        'crash_protection': 0.15,
    },
    
    # Factor Parameters
    'trend_persistence_params': {
        'adx_period': 14,
        'adx_threshold': 25,
        'hurst_period': 20,
    },
    
    'chop_filter_params': {
        'chop_period': 14,
        'chop_threshold': 60,
    },
    
    'regime_params': {
        'adx_period': 14,
        'sma_period': 50,
        'regime_multipliers': {
            0: 0.4,
            1: 1.1,
            -1: 1.1,
            2: 0.6,
        },
    },
    
    'vol_breakout_params': {
        'bb_period': 20,
        'bb_std': 2.0,
        'atr_period': 14,
    },
    
    'crash_protection_params': {
        'vol_lookback': 20,
        'crash_vol_mult': 1.8,
        'momentum_lookback': 24,
    },
    
    # Signal Generation
    'signal_params': {
        'momentum_lookback': 36,
        'ema_fast': 12,
        'ema_slow': 26,
        'donchian_period': 24,
        'holding_period_min': 8,
    },
    
    # Risk Management
    'risk_params': {
        'max_drawdown_target': 0.40,
        'volatility_target': 0.35,
        'atr_stop_mult': 2.0,
    },
    
    # Optimization Bounds
    'opt_bounds': {
        'leverage': (1.2, 1.5),
        'adx_threshold': (20, 35),
        'chop_threshold': (55, 70),
        'momentum_lookback': (18, 60),
    },
}


# =============================================================================
# Route C: Aggressive (積極型)
# Target: CAGR 40-45%, MaxDD ≤55%
# =============================================================================

ROUTE_C_CONFIG = {
    'name': 'Route C (Aggressive)',
    'description': 'Maximum returns with higher risk tolerance',
    
    # Leverage and Position Sizing
    'leverage': 2.0,
    'max_leverage': 2.5,
    'base_position_size': 1.0,
    
    # Factor Weights
    'factor_weights': {
        'trend_persistence': 0.25,
        'chop_filter': 0.10,
        'regime_sizing': 0.20,
        'vol_breakout': 0.30,     # High breakout weight
        'crash_protection': 0.15,
    },
    
    # Factor Parameters
    'trend_persistence_params': {
        'adx_period': 14,
        'adx_threshold': 20,      # Lower threshold = more trades
        'hurst_period': 20,
    },
    
    'chop_filter_params': {
        'chop_period': 14,
        'chop_threshold': 65,     # Higher threshold = less filtering
    },
    
    'regime_params': {
        'adx_period': 14,
        'sma_period': 50,
        'regime_multipliers': {
            0: 0.5,
            1: 1.3,
            -1: 1.3,
            2: 0.8,              # Still active in volatile
        },
    },
    
    'vol_breakout_params': {
        'bb_period': 20,
        'bb_std': 1.8,           # Tighter bands for more signals
        'atr_period': 14,
    },
    
    'crash_protection_params': {
        'vol_lookback': 20,
        'crash_vol_mult': 2.5,   # Less strict crash detection
        'momentum_lookback': 24,
    },
    
    # Signal Generation
    'signal_params': {
        'momentum_lookback': 24,
        'ema_fast': 10,
        'ema_slow': 21,
        'donchian_period': 20,
        'holding_period_min': 4,
    },
    
    # Risk Management
    'risk_params': {
        'max_drawdown_target': 0.55,
        'volatility_target': 0.50,
        'atr_stop_mult': 1.5,
    },
    
    # Optimization Bounds
    'opt_bounds': {
        'leverage': (1.8, 2.5),
        'adx_threshold': (15, 30),
        'chop_threshold': (60, 75),
        'momentum_lookback': (12, 48),
    },
}


# Route mapping
ROUTE_CONFIGS = {
    'A': ROUTE_A_CONFIG,
    'B': ROUTE_B_CONFIG,
    'C': ROUTE_C_CONFIG,
}


def get_route_config(route: str) -> dict:
    """Get configuration for a specific route."""
    return ROUTE_CONFIGS.get(route.upper(), ROUTE_A_CONFIG)


def get_all_routes() -> list:
    """Get list of all route names."""
    return list(ROUTE_CONFIGS.keys())

