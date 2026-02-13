"""
Definitive comparison: Geometric Brownian Motion vs Merton Jump Diffusion.
USING ONLY REAL DATA FROM parameters.json - NO DEFAULTS, NO FALLBACKS.
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
    RANDOM_SEED,
    DEFAULT_N_SIMULATIONS,
    DEFAULT_N_DAYS,
    DEFAULT_TIME_HORIZON
)

print("=" * 80)
print("GBM vs JUMP DIFFUSION - REAL DATA ONLY")
print("=" * 80)

# ============================================
# 1. LOAD REAL CALIBRATED PARAMETERS
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
# 2. VERIFY WE HAVE EVERYTHING WE NEED
# ============================================
SYMBOLS = [s for s in DEFAULT_SYMBOLS if s in params['assets']]

# Verify we have jump diffusion parameters for all assets
if 'jump_diffusion' not in params:
    print("ERROR: No jump diffusion parameters found in parameters.json")
    print("Run: python test/calibrate_all_jumps.py")
    sys.exit(1)

# ============================================
# 3. EXTRACT REAL PARAMETERS - NO DEFAULTS
# ============================================
print("\nEXTRACTING REAL PARAMETERS FROM CALIBRATION:")
print("-" * 50)

gbm_params = {}
jump_params = {}
initial_prices = {}

for symbol in SYMBOLS:
    asset_data = params['assets'][symbol]
    jump_data = params['jump_diffusion'][symbol]

    print(f"\n{symbol}:")

    # ----- REAL GBM PARAMETERS -----
    mu_value = asset_data['mu']  # Will FAIL if missing - GOOD!
    sigma_value = asset_data['sigma']  # Will FAIL if missing - GOOD!

    gbm_params[symbol] = {
        'mu': mu_value,
        'sigma': sigma_value
    }

    print(f"   GBM μ: {mu_value*100:.2f}%")
    print(f"   GBM σ: {sigma_value*100:.2f}%")

    # ----- REAL INITIAL PRICE -----
    # MUST exist in parameters.json from calculate_params.py
    if 'final_price' not in asset_data:
        print(f"   ERROR: No final_price for {symbol}")
        print("   Run calculate_params.py first")
        sys.exit(1)

    price = asset_data['final_price']
    initial_prices[symbol] = price
    print(f"   Current price: ${price:,.2f}")

    # ----- REAL JUMP DIFFUSION PARAMETERS -----
    jump_params[symbol] = {
        'mu_diff': jump_data['mu_diffusion'],
        'sigma_diff': jump_data['sigma_diffusion'],
        'lambda_jump': jump_data['lambda_jump'],
        'mu_jump': jump_data['mu_jump'],
        'sigma_jump': jump_data['sigma_jump']
    }

    print(f"   Jump λ: {jump_data['lambda_jump']:.2f} jumps/year")
    print(f"   Jump μ_j: {jump_data['mu_jump']*100:.2f}%")
    print(f"   Jump σ_j: {jump_data['sigma_jump']*100:.2f}%")

# ============================================
# 4. REAL CORRELATION MATRIX - NO DEFAULTS
# ============================================
print("\n" + "=" * 50)
print("LOADING REAL CORRELATION MATRIX")
print("=" * 50)

if 'correlation_matrix' not in params:
    print("ERROR: No correlation matrix found in parameters.json")
    print("Run calculate_params.py to generate correlation matrix")
    sys.exit(1)

try:
    corr_dict = params['correlation_matrix']
    n = len(SYMBOLS)
    corr_matrix = np.zeros((n, n))

    for i, s1 in enumerate(SYMBOLS):
        for j, s2 in enumerate(SYMBOLS):
            # This will FAIL if correlation doesn't exist - GOOD!
            corr_matrix[i, j] = corr_dict[s1][s2]

    print("\nREAL CORRELATION MATRIX (from historical data):")
    print(np.round(corr_matrix, 3))

    # Print key correlations
    if 'BTC' in SYMBOLS and 'ETH' in SYMBOLS:
        btc_eth_corr = corr_dict['BTC']['ETH']
        print(f"\nBTC-ETH correlation: {btc_eth_corr:.3f}")

except Exception as e:
    print(f"ERROR loading correlation matrix: {e}")
    print("Cannot proceed without real correlation data")
    sys.exit(1)

# ============================================
# 5. SIMULATION CONFIGURATION
# ============================================
N_SIMULATIONS = DEFAULT_N_SIMULATIONS
N_DAYS = DEFAULT_N_DAYS
T = DEFAULT_TIME_HORIZON

print(f"\nSIMULATION CONFIGURATION:")
print(f"   Simulations: {N_SIMULATIONS:,}")
print(f"   Time horizon: {N_DAYS} days (1 year)")
print(f"   Assets: {SYMBOLS}")

# ============================================
# 6. VECTORIZED GBM SIMULATION
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

print(f"\nInitial prices (REAL): {dict(zip(SYMBOLS, S0_array))}")

# Run GBM simulation
gbm_paths = simulate_gbm_vectorized(
    S0_array, mu_array, sigma_array, corr_matrix, T, N_DAYS, N_SIMULATIONS
)
gbm_returns = (gbm_paths[:, -1, :] / S0_array.reshape(-1, 1)) - 1

print(f"GBM simulation complete: {gbm_paths.shape}")
print(f"   BTC mean return: {np.mean(gbm_returns[0, :]) * 100:.2f}%")

# ============================================
# 7. JUMP DIFFUSION SIMULATION
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
    print("Check that all parameters are valid")
    sys.exit(1)

# ============================================
# 8. COMPARATIVE METRICS
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
# 9. EXECUTIVE SUMMARY - REAL RESULTS ONLY
# ============================================
btc_gbm = comparison['BTC']['GBM']
btc_jump = comparison['BTC']['JUMP']

print("\n" + "=" * 80)
print("EXECUTIVE SUMMARY - BTC (REAL DATA)")
print("=" * 80)

print(f"""
REAL CALIBRATED PARAMETERS (from 5 years of Binance data):
   • BTC drift (μ): {gbm_params['BTC']['mu']*100:.2f}% annual
   • BTC volatility (σ): {gbm_params['BTC']['sigma']*100:.2f}% annual
   • Jump frequency (λ): {jump_params['BTC']['lambda_jump']:.2f} jumps/year
   • Average jump size (μ_j): {jump_params['BTC']['mu_jump']*100:.2f}%
   • BTC-ETH correlation: {corr_dict['BTC']['ETH']:.3f}

SIMULATION RESULTS (50,000 scenarios, 1 year horizon):
   PROBABILITY OF LOSS >20%:
      GBM predicts:  {btc_gbm['prob_loss_20']:.1f}%
      JUMP predicts: {btc_jump['prob_loss_20']:.1f}%
      Difference:    +{btc_jump['prob_loss_20'] - btc_gbm['prob_loss_20']:.1f}%
      Risk factor:   {btc_jump['prob_loss_20'] / btc_gbm['prob_loss_20']:.1f}x

   VaR 95%:
      GBM:  {btc_gbm['var_95']:.1f}%
      JUMP: {btc_jump['var_95']:.1f}%

   KURTOSIS (fat tails, normal = 3):
      GBM:  {btc_gbm['kurtosis']:.1f}
      JUMP: {btc_jump['kurtosis']:.1f}
""")

# ============================================
# 10. SAVE RESULTS FOR DASHBOARD - ALL ASSETS!
# ============================================
print("\nSAVING DASHBOARD RESULTS FOR ALL ASSETS...")

dashboard_data = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'n_simulations': N_SIMULATIONS,
    'assets': SYMBOLS,
    'results': {}  # <--- NUEVO: aquí van TODOS los activos
}

# Guardar TODOS los activos
for symbol in SYMBOLS:
    gbm_metrics = comparison[symbol]['GBM']
    jump_metrics = comparison[symbol]['JUMP']

    dashboard_data['results'][symbol] = {
        'gbm_prob_loss_20': float(gbm_metrics['prob_loss_20']),
        'jump_prob_loss_20': float(jump_metrics['prob_loss_20']),
        'gbm_var_95': float(gbm_metrics['var_95']),
        'jump_var_95': float(jump_metrics['var_95']),
        'gbm_kurtosis': float(gbm_metrics['kurtosis']),
        'jump_kurtosis': float(jump_metrics['kurtosis']),
        'gbm_expected_return': float(gbm_metrics['expected_return']),
        'jump_expected_return': float(jump_metrics['expected_return'])
    }

# Mantener BTC en raíz para compatibilidad con dashboards viejos
if 'BTC' in SYMBOLS:
    dashboard_data['btc'] = dashboard_data['results']['BTC']

# Guardar archivo
with open(DASHBOARD_RESULTS, 'w') as f:
    json.dump(dashboard_data, f, indent=2, default=str)

print(f"✅ Saved results for: {list(dashboard_data['results'].keys())}")