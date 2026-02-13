"""
Crypto Risk Engine - Centralized Configuration
All paths and global settings are defined here.
"""

from pathlib import Path

# ============================================
# PROJECT ROOT - Works on any machine
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================
# DATA DIRECTORIES - Single source of truth
# ============================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
APP_DIR = PROJECT_ROOT / "app"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "test"

# ============================================
# CRITICAL FILES - All generated outputs
# ============================================
PARAMS_FILE = PROCESSED_DATA_DIR / "parameters.json"
DASHBOARD_RESULTS = PROCESSED_DATA_DIR / "dashboard_results.json"
SIMULATION_RESULTS = PROCESSED_DATA_DIR / "simulation_results.json"

# ============================================
# ASSET CONFIGURATION
# ============================================
DEFAULT_SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA']
TRADING_DAYS_PER_YEAR = 252

# ============================================
# DAILY FILES - Auto-generated paths
# ============================================
def get_daily_path(symbol: str) -> Path:
    """Get path for daily feather file by symbol"""
    return PROCESSED_DATA_DIR / f"{symbol.lower()}_daily.feather"

DAILY_FILES = {
    'BTC': get_daily_path('BTC'),
    'ETH': get_daily_path('ETH'),
    'SOL': get_daily_path('SOL'),
    'BNB': get_daily_path('BNB'),
    'ADA': get_daily_path('ADA')
}

# ============================================
# SIMULATION DEFAULTS
# ============================================
RANDOM_SEED = 42
DEFAULT_N_SIMULATIONS = 50000
DEFAULT_N_DAYS = 252
DEFAULT_TIME_HORIZON = 1.0