"""
Definitive comparison: Geometric Brownian Motion vs Merton Jump Diffusion.
Calibrated with real Binance data from 2020-2026.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from scipy import stats
from datetime import datetime
from src.models.jump_diffusion import MultivariateJumpDiffusion
from src.config import (
    PARAMS_FILE,
    DASHBOARD_RESULTS,
    DEFAULT_SYMBOLS,
    TRADING_DAYS_PER_YEAR,
    RANDOM_SEED,
    DEFAULT_N_SIMULATIONS,
    DEFAULT_N_DAYS,
    DEFAULT_TIME_HORIZON
)

print("=" * 80)
print("GBM vs JUMP DIFFUSION - DEFINITIVE COMPARISON")
print("=" * 80)

# ============================================
# 1. LOAD CALIBRATED PARAMETERS
# ============================================
print("\nLOADING REAL CALIBRATED PARAMETERS...")

if not PARAMS_FILE.exists():
    print(f"ERROR: Parameters file not found at {PARAMS_FILE}")
    print("Run first: python test/calibrate_all_jumps.py")
    sys.exit(1)

with open(PARAMS_FILE, 'r') as f:
    params = json.load(f)

print(f"Assets found: {list(params['assets'].keys())}")

# ============================================
# 2. SIMULATION CONFIGURATION
# ============================================
SYMBOLS = [s for s in DEFAULT_SYMBOLS if s in params['assets']]
N_SIMULATIONS = DEFAULT_N_SIMULATIONS
N_DAYS = DEFAULT_N_DAYS
T = DEFAULT_TIME_HORIZON
DT = T / N_DAYS

print(f"\nSIMULATION CONFIGURATION:")
print(f"   Simulations: {N_SIMULATIONS:,}")
print(f"   Time horizon: {N_DAYS} days (1 year)")
print(f"   Assets: {SYMBOLS}")

# ============================================
# 3. EXTRACT PARAMETERS BY ASSET
# ============================================
gbm_params = {}
jump_params = {}
initial_prices = {}

for symbol in SYMBOLS:
    if symbol in params['assets']:
        asset_data = params['assets'][symbol]

        print(f"\n{symbol}:")

        # ----- GBM PARAMETERS -----
        mu_value = asset_data.get('mu', 0.10)
        sigma_value = asset_data.get('sigma', 0.50)

        gbm_params[symbol] = {
            'mu': mu_value,
            'sigma': sigma_value
        }

        # ----- INITIAL PRICE -----
        price = asset_data.get('final_price', None)
        if price is None:
            # Try alternative keys
            for key in ['last_price', 'close', 'price']:
                if key in asset_data:
                    price = asset_data[key]
                    break

        if price is None:
            # Default values
            defaults = {'BTC': 50000, 'ETH': 3000, 'SOL': 100, 'BNB': 300, 'ADA': 0.5}
            price = defaults.get(symbol, 100)
            print(f"   Warning: No price found, using default: ${price:,.2f}")
        else:
            print(f"   Price: ${price:,.2f}")

        initial_prices[symbol] = price

        # ----- JUMP DIFFUSION PARAMETERS -----
        if 'jump_diffusion' in params and symbol in params['jump_diffusion']:
            jp = params['jump_diffusion'][symbol]

            jump_params[symbol] = {
                'mu_diff': jp.get('mu_diffusion', mu_value),
                'sigma_diff': jp.get('sigma_diffusion', sigma_value),
                'lambda_jump': jp.get('lambda_jump', 5.0),
                'mu_jump': jp.get('mu_jump', -0.02),
                'sigma_jump': jp.get('sigma_jump', 0.05)
            }
            print(f"   Jump: λ={jump_params[symbol]['lambda_jump']:.2f}, "
                  f"μ_j={jump_params[symbol]['mu_jump'] * 100:.2f}%")
        else:
            print(f"   Warning: No jump params, using defaults")
            jump_params[symbol] = {
                'mu_diff': mu_value,
                'sigma_diff': sigma_value,
                'lambda_jump': 5.0,
                'mu_jump': -0.02,
                'sigma_jump': 0.05
            }

# ============================================
# 4. CORRELATION MATRIX
# ============================================
print("\nLOADING CORRELATION MATRIX...")

if 'correlation_matrix' in params:
    try:
        corr_dict = params['correlation_matrix']
        n = len(SYMBOLS)
        corr_matrix = np.zeros((n, n))

        for i, s1 in enumerate(SYMBOLS):
            for j, s2 in enumerate(SYMBOLS):
                if s1 in corr_dict and s2 in corr_dict[s1]:
                    corr_matrix[i, j] = corr_dict[s1][s2]
                else:
                    corr_matrix[i, j] = 0.7 if i != j else 1.0

        print("   Correlation matrix loaded from JSON")
    except Exception as e:
        print(f"   Error loading correlation matrix: {e}")
        print("   Using default correlation matrix")
        corr_matrix = np.array([
                                   [1.00, 0.83, 0.75, 0.70, 0.68],
                                   [0.83, 1.00, 0.78, 0.72, 0.70],
                                   [0.75, 0.78, 1.00, 0.69, 0.67],
                                   [0.70, 0.72, 0.69, 1.00, 0.65],
                                   [0.68, 0.70, 0.67, 0.65, 1.00]
                               ][:n, :n])
else:
    print("   No correlation matrix found, using default")
    corr_matrix = np.array([
                               [1.00, 0.83, 0.75, 0.70, 0.68],
                               [0.83, 1.00, 0.78, 0.72, 0.70],
                               [0.75, 0.78, 1.00, 0.69, 0.67],
                               [0.70, 0.72, 0.69, 1.00, 0.65],
                               [0.68, 0.70, 0.67, 0.65, 1.00]
                           ][:len(SYMBOLS), :len(SYMBOLS)])

print("\nCORRELATION MATRIX:")
print(np.round(corr_matrix, 3))

# ============================================
# 5. VECTORIZED GBM SIMULATION
# ============================================
print("\n" + "=" * 80)
print("SIMULATING GBM - 50,000 TRAJECTORIES")
print("=" * 80)


def simulate_gbm_vectorized(S0, mu, sigma, corr_matrix, T, n_days, n_sims):
    """Fully vectorized multivariate GBM simulation"""
    n_assets = len(S0)
    dt = T / n_days

    # Regularized Cholesky decomposition
    corr_reg = corr_matrix + 1e-6 * np.eye(n_assets)
    L = np.linalg.cholesky(corr_reg)

    # Generate correlated shocks
    np.random.seed(RANDOM_SEED)
    Z = np.random.randn(n_days, n_sims, n_assets)
    Z_corr = Z @ L.T

    # Drift and diffusion components
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z_corr

    # Cumulative log returns
    log_returns = drift + diffusion
    log_cumsum = np.cumsum(log_returns, axis=0)

    # Prices
    log_prices = np.log(S0) + np.vstack([np.zeros((1, n_sims, n_assets)), log_cumsum])

    return np.exp(log_prices).transpose(2, 0, 1)


# Prepare arrays
S0_array = np.array([initial_prices[s] for s in SYMBOLS])
mu_array = np.array([gbm_params[s]['mu'] for s in SYMBOLS])
sigma_array = np.array([gbm_params[s]['sigma'] for s in SYMBOLS])

# Run GBM simulation
gbm_paths = simulate_gbm_vectorized(
    S0_array, mu_array, sigma_array, corr_matrix, T, N_DAYS, N_SIMULATIONS
)
gbm_returns = (gbm_paths[:, -1, :] / S0_array.reshape(-1, 1)) - 1

print(f"GBM simulation complete: {gbm_paths.shape}")
print(f"   BTC mean return: {np.mean(gbm_returns[0, :]) * 100:.2f}%")

# ============================================
# 6. JUMP DIFFUSION SIMULATION
# ============================================
print("\n" + "=" * 80)
print("SIMULATING JUMP DIFFUSION - 50,000 TRAJECTORIES")
print("=" * 80)

try:
    jump_model = MultivariateJumpDiffusion(
        symbols=SYMBOLS,
        initial_prices=[initial_prices[s] for s in SYMBOLS],
        mus=[jump_params[s]['mu_diff'] for s in SYMBOLS],
        sigmas=[jump_params[s]['sigma_diff'] for s in SYMBOLS],
        lambda_jumps=[jump_params[s]['lambda_jump'] for s in SYMBOLS],
        mu_jumps=[jump_params[s]['mu_jump'] for s in SYMBOLS],
        sigma_jumps=[jump_params[s]['sigma_jump'] for s in SYMBOLS],
        correlation_matrix=corr_matrix,
        seed=RANDOM_SEED
    )

    jump_paths = jump_model.simulate(T=T, n_days=N_DAYS, n_simulations=N_SIMULATIONS)
    jump_returns = (jump_paths[:, -1, :] / S0_array.reshape(-1, 1)) - 1

    print(f"Jump Diffusion simulation complete: {jump_paths.shape}")
    print(f"   BTC mean return: {np.mean(jump_returns[0, :]) * 100:.2f}%")

except Exception as e:
    print(f"ERROR in Jump Diffusion simulation: {e}")
    print("   Using GBM as fallback...")
    jump_returns = gbm_returns.copy()
    jump_paths = gbm_paths.copy()

# ============================================
# 7. COMPARATIVE METRICS
# ============================================
print("\n" + "=" * 80)
print("COMPARATIVE METRICS - 1 YEAR HORIZON")
print("=" * 80)


def calculate_all_metrics(returns, name):
    """Calculate complete risk metrics for return series"""
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()

    return {
        'model': name,
        'expected_return': np.mean(returns) * 100,
        'volatility': np.std(returns) * 100,
        'sharpe': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        'var_95': var_95 * 100,
        'var_99': var_99 * 100,
        'cvar_95': cvar_95 * 100,
        'cvar_99': cvar_99 * 100,
        'prob_loss_10': np.mean(returns < -0.10) * 100,
        'prob_loss_20': np.mean(returns < -0.20) * 100,
        'prob_loss_30': np.mean(returns < -0.30) * 100,
        'prob_loss_50': np.mean(returns < -0.50) * 100,
        'prob_loss_70': np.mean(returns < -0.70) * 100,
        'max_loss': returns.min() * 100,
        'max_gain': returns.max() * 100,
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns)
    }


# Calculate metrics for all assets
comparison = {}
for i, symbol in enumerate(SYMBOLS):
    comparison[symbol] = {
        'GBM': calculate_all_metrics(gbm_returns[i, :], 'GBM'),
        'JUMP': calculate_all_metrics(jump_returns[i, :], 'JUMP')
    }

# ============================================
# 8. EXECUTIVE SUMMARY - BTC FOCUS
# ============================================
btc_gbm = comparison['BTC']['GBM']
btc_jump = comparison['BTC']['JUMP']

print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY - BTC")
print("=" * 80)

print(f"""
PROBABILITY OF LOSS >20% IN 1 YEAR:
   GBM predicts:  {btc_gbm['prob_loss_20']:.1f}%
   JUMP predicts: {btc_jump['prob_loss_20']:.1f}%
   Difference:    +{btc_jump['prob_loss_20'] - btc_gbm['prob_loss_20']:.1f}%
   Risk factor:   {btc_jump['prob_loss_20'] / btc_gbm['prob_loss_20']:.1f}x

VaR 95%:
   GBM:  {btc_gbm['var_95']:.1f}%
   JUMP: {btc_jump['var_95']:.1f}%

KURTOSIS (fat tails):
   GBM:  {btc_gbm['kurtosis']:.1f}
   JUMP: {btc_jump['kurtosis']:.1f}
""")

# ============================================
# 9. SAVE RESULTS FOR DASHBOARD
# ============================================
print("\nSAVING DASHBOARD RESULTS...")

dashboard_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_simulations': N_SIMULATIONS,
    'assets': SYMBOLS,
    'btc': {
        'gbm_prob_loss_20': float(btc_gbm['prob_loss_20']),
        'jump_prob_loss_20': float(btc_jump['prob_loss_20']),
        'gbm_var_95': float(btc_gbm['var_95']),
        'jump_var_95': float(btc_jump['var_95']),
        'gbm_kurtosis': float(btc_gbm['kurtosis']),
        'jump_kurtosis': float(btc_jump['kurtosis'])
    }
}

# Ensure directory exists
DASHBOARD_RESULTS.parent.mkdir(parents=True, exist_ok=True)

with open(DASHBOARD_RESULTS, 'w') as f:
    json.dump(dashboard_data, f, indent=2, default=str)

print(f"   Results saved to: {DASHBOARD_RESULTS}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETED - READY FOR DASHBOARD")
print("=" * 80)
print(f"\nNext step: streamlit run app/Home.py")