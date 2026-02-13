# 🚀 **CRYPTO RISK ENGINE**

## *Merton Jump Diffusion Model for Cryptocurrency Risk Analysis*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📋 **TABLE OF CONTENTS**
- [Executive Summary](#-executive-summary)
- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Key Findings](#-key-findings)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Technologies](#-technologies)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🎯 **EXECUTIVE SUMMARY**

**Crypto Risk Engine** is a professional quantitative finance project that implements the **Merton Jump Diffusion model (1976)** to measure extreme event risk in cryptocurrency markets. 

Unlike traditional Black-Scholes/GBM models that **underestimate tail risk by 60%**, this engine captures the fat tails, jump discontinuities, and crisis correlations that characterize real crypto markets.

**Calibrated with 5+ years of Binance USDT perpetual futures data (2020-2026)** across 5 major cryptocurrencies: **BTC, ETH, SOL, BNB, ADA**.

---

## ⚠️ **THE PROBLEM**

### *"Why do all traditional risk models fail in crypto?"*

**Geometric Brownian Motion (GBM)** assumes:
- ✅ Returns are normally distributed
- ✅ Volatility is constant
- ❌ **NO extreme events** (probability of -50% crash ≈ 1e-99)
- ❌ **NO jump discontinuities** (flash crashes, exchange collapses)
- ❌ **NO crisis correlation amplification**

**REALITY of crypto markets:**
```
CRISIS                    DRAWDOWN    RECOVERY    REALITY
─────────────────────────────────────────────────────────────
COVID-19 (Mar 2020)       -50.2%      180 days    ✓ REAL EVENT
FTX Collapse (Nov 2022)   -64.8%      365 days    ✓ REAL EVENT  
LUNA Crash (May 2022)     -56.7%      240 days    ✓ REAL EVENT
Crypto Winter 2018        -82.3%     1095 days    ✓ REAL EVENT
```

**GBM says these events are IMPOSSIBLE.**
**History says they happen every 2 years.**

---

## 🦘 **THE SOLUTION**

### *Merton Jump Diffusion (1976)*

```
dS/S = (μ - λκ)dt + σ dW + (e^J - 1)dN
```

Where:
- **μ** : Drift (expected return)
- **σ** : Diffusion volatility (normal days)
- **λ** : Jump intensity (extreme events per year)
- **J ∼ N(μ_j, σ_j²)** : Jump size distribution
- **κ = E[e^J - 1]** : Jump compensation term
- **dN** : Poisson process (jump timing)

**This model captures:**
- ✅ **Normal days** → GBM diffusion
- ✅ **Extreme events** → Poisson jumps
- ✅ **Fear-driven selloffs** → Negative μ_jump (BTC, ETH)
- ✅ **Euphoric rallies** → Positive μ_jump (BNB, ADA)
- ✅ **Crisis correlations** → Correlated jumps

---

## 🔬 **KEY FINDINGS**

### *5 Years of Binance Data - What We Discovered*

| **METRIC** | **BTC** | **ETH** | **SOL** | **BNB** | **ADA** |
|-----------|--------|--------|--------|--------|--------|
| **μ (drift)** | 46.77% | 59.23% | 59.02% | 38.06% | 9.31% |
| **σ (volatility)** | 45.15% | 59.83% | 84.23% | 54.24% | 71.35% |
| **λ (jumps/year)** | 3.41 | 3.41 | 3.75 | 3.97 | 3.83 |
| **μ_jump** | -3.85% | -4.58% | -0.23% | **+1.60%** | **+6.35%** |
| **σ_jump** | 15.20% | 20.52% | 29.12% | 24.00% | 23.10% |
| **Kurtosis** | 28.1 | 19.6 | 13.2 | 30.4 | 12.6 |
| **Current Price** | $108,208 | $4,389 | $200.57 | $858.08 | $0.81 |

---

### 📉 **GBM vs JUMP DIFFUSION - BTC 1 Year Horizon**

| **RISK METRIC** | **GBM** | **JUMP** | **DIFFERENCE** | **FACTOR** |
|----------------|--------|---------|----------------|------------|
| **P(Loss >20%)** | 9.7% | **15.4%** | +5.7% | **1.6x** |
| **P(Loss >30%)** | 3.9% | **8.0%** | +4.1% | **2.1x** |
| **P(Loss >50%)** | 0.8% | **2.9%** | +2.1% | **3.6x** |
| **VaR 95%** | -31.5% | **-43.8%** | -12.3% | **1.4x** |
| **VaR 99%** | -42.8% | **-58.2%** | -15.4% | **1.4x** |
| **CVaR 95%** | -46.2% | **-58.9%** | -12.7% | **1.3x** |
| **Kurtosis** | 4.5 | **5.7** | +1.2 | **fat tails** |

---

### 📊 **ASSET COMPARISON - Jump Diffusion Metrics**

```
ASSET    λ    μ_jump    INTERPRETATION
────────────────────────────────────────────────────
BTC     3.41  -3.85%    🔴 Fear-driven selloffs
ETH     3.41  -4.58%    🔴 Strong downside jumps
SOL     3.75  -0.23%    ⚪ Neutral jump bias
BNB     3.97  +1.60%    🟢 Occasional positive spikes
ADA     3.83  +6.35%    🟢 Strong positive jump bias
```

**BNB has the HIGHEST jump frequency (3.97 jumps/year)**  
**ADA has the MOST POSITIVE jump bias (+6.35%)**  
**BTC has the HIGHEST kurtosis (28.1) → extreme events are NORMAL**

---

## 🏗️ **ARCHITECTURE**

### *Professional Quantitative Finance Pipeline*

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  📁 data/raw/                                              │
│  └── Binance USDT perpetual futures 1m parquet files      │
│                                                           │
│  📁 data/processed/                                       │
│  ├── *_daily.feather      # Resampled OHLCV daily data   │
│  ├── parameters.json      # ALL calibrated parameters    │
│  └── dashboard_results.json # Simulation results         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  📄 test/resample_to_daily.py                             │
│  └── 1m parquet → daily OHLCV + log returns              │
│                                                           │
│  📄 calculate_params.py                                   │
│  └── GBM parameters + prices + correlations + metrics    │
│                                                           │
│  📄 test/calibrate_all_jumps.py                          │
│  └── Jump Diffusion calibration (λ, μ_j, σ_j)           │
│                                                           │
│  📄 test/test_jump_vs_gbm.py                             │
│  └── 50k simulations GBM vs Jump + risk metrics         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  📄 app/Home.py                                           │
│  └── Interactive Streamlit dashboard                     │
│      ├── Executive Summary                               │
│      ├── GBM vs Jump Comparison                          │
│      ├── Jump Analysis                                   │
│      ├── Portfolio 50/50                                 │
│      ├── Stress Testing (Historical Crises)             │
│      └── Model Parameters                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 **INSTALLATION**

### *Prerequisites*
- Python 3.10+
- Git
- 5+ years of Binance 1m parquet data (included in this repo)

### **1. Clone Repository**
```bash
git clone https://github.com/dkysuarez/crypto-risk-engine.git
cd crypto-risk-engine
```

### **2. Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Verify Data Structure**
```bash
tree /F /A  # Windows
ls -R       # Linux/Mac
```

Expected structure:
```
crypto_risk_engine/
├── data/
│   ├── raw/           # Place your Binance 1m parquet files here
│   └── processed/     # Auto-generated daily files
├── src/              # Source code
├── app/              # Streamlit dashboard
├── test/             # Pipeline scripts
└── outputs/          # Generated visualizations
```

---

## 🚀 **USAGE**

### **STEP 1: Resample 1-minute to Daily Data**
```bash
python test/resample_to_daily.py
```
- Reads: `data/raw/*.parquet`
- Writes: `data/processed/*_daily.feather`

### **STEP 2: Calculate GBM Parameters**
```bash
python calculate_params.py
```
- Reads: `data/processed/*_daily.feather`
- Writes: `data/processed/parameters.json`
- Outputs: μ, σ, Sharpe, max drawdown, skewness, kurtosis, prices, correlation matrix

### **STEP 3: Calibrate Jump Diffusion**
```bash
python test/calibrate_all_jumps.py
```
- Reads: `data/processed/parameters.json`
- Reads: `data/processed/*_daily.feather`
- Writes: `data/processed/parameters.json` (UPDATED with jump params)
- Outputs: λ, μ_jump, σ_jump, jump counts, κ

### **STEP 4: Run 50k Simulations - GBM vs Jump**
```bash
python test/test_jump_vs_gbm.py
```
- Reads: `data/processed/parameters.json`
- Writes: `data/processed/dashboard_results.json`
- Outputs: Complete risk metrics for ALL assets
- Duration: ~2-3 minutes (50,000 scenarios × 5 assets × 252 days)

### **STEP 5: Launch Interactive Dashboard**
```bash
streamlit run app/Home.py
```
- Opens: `http://localhost:8501`
- Explore: ALL your real data, interactive charts, risk metrics

---

## 📊 **RESULTS**

### *Executive Dashboard Preview*

```
┌─────────────────────────────────────────────────────────────┐
│  CRYPTO RISK ENGINE - EXECUTIVE SUMMARY                    │
├─────────────────────────────────────────────────────────────┤
│  BTC - Probability Loss >20%                               │
│  ┌──────────────────────────────────────────────────┐     │
│  │  GBM:  9.7%   JUMP: 15.4%   Δ: +5.7%  ⚠️        │     │
│  └──────────────────────────────────────────────────┘     │
│                                                           │
│  BTC - Value at Risk 95%                                  │
│  ┌──────────────────────────────────────────────────┐     │
│  │  GBM:  -31.5%  JUMP: -43.8%  Δ: -12.3%  ⚠️       │     │
│  └──────────────────────────────────────────────────┘     │
│                                                           │
│  RISK UNDERESTIMATION FACTOR: 1.6x                       │
│  "GBM underestimates crash risk by 60%"                  │
└─────────────────────────────────────────────────────────────┘
```

### *Jump Analysis Dashboard*

```
ASSET    λ      μ_jump    INTERPRETATION
────────────────────────────────────────────────
BTC     3.41   -3.85%    🔴 Fear-driven selloffs
ETH     3.41   -4.58%    🔴 Strong downside jumps
SOL     3.75   -0.23%    ⚪ Neutral
BNB     3.97   +1.60%    🟢 Occasional positive
ADA     3.83   +6.35%    🟢 Strong positive bias
```

### *Historical Crisis Dashboard*

```
CRISIS              DRAWDOWN    VOLATILITY    RECOVERY    TRIGGER
────────────────────────────────────────────────────────────────
COVID-19           -50.2%       95.3%        180 days    Global pandemic
FTX Collapse       -64.8%       82.1%        365 days    Exchange bankruptcy
LUNA Crash         -56.7%       105.2%       240 days    UST depeg
Crypto Winter      -82.3%       78.5%        1095 days   ICO bubble burst
```

---

## 🛠️ **TECHNOLOGIES**

| **Category** | **Technologies** |
|-------------|------------------|
| **Languages** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy, Feather |
| **Financial Modeling** | SciPy, StatsModels |
| **Visualization** | Streamlit, Plotly, Matplotlib, Seaborn |
| **Storage** | JSON, Parquet, Feather |
| **Version Control** | Git, GitHub |
| **Dependencies** | pip, requirements.txt |

---

## 📁 **PROJECT STRUCTURE**

```
crypto_risk_engine/
│
├── 📁 data/                          # DATA LAYER - Single source of truth
│   ├── 📁 raw/                      # Binance 1m parquet files (READ ONLY)
│   │   ├── BTCUSDT_1m_2020-01-01_to_2025-08-31.parquet
│   │   ├── ETHUSDT_1m_2020-01-01_to_2025-08-31.parquet
│   │   ├── SOLUSDT_1m_2020-09-14_to_2025-08-31.parquet
│   │   ├── BNBUSDT_1m_2020-02-10_to_2025-08-31.parquet
│   │   └── ADAUSDT_1m_2020-01-31_to_2025-08-31.parquet
│   │
│   └── 📁 processed/                # Generated data (WRITE ONCE)
│       ├── btc_daily.feather
│       ├── eth_daily.feather
│       ├── sol_daily.feather
│       ├── bnb_daily.feather
│       ├── ada_daily.feather
│       ├── parameters.json         # ALL calibrated parameters
│       └── dashboard_results.json  # ALL simulation results
│
├── 📁 src/                          # SOURCE CODE - Core logic
│   ├── __init__.py
│   ├── config.py                   # CENTRALIZED PATHS - Critical
│   │
│   ├── 📁 models/                  # Financial models
│   │   ├── __init__.py
│   │   ├── gbm_simulator.py       # GBM base class
│   │   └── jump_diffusion.py      # Merton Jump Diffusion (1976)
│   │
│   └── 📁 utils/                   # (Future) Helper functions
│
├── 📁 app/                          # PRESENTATION LAYER
│   └── Home.py                    # MAIN DASHBOARD - Streamlit
│
├── 📁 test/                         # PIPELINE SCRIPTS
│   ├── resample_to_daily.py       # Step 1: 1m → daily
│   ├── calibrate_all_jumps.py     # Step 3: Jump calibration
│   ├── test_jump_vs_gbm.py        # Step 4: 50k simulations
│   └── test_jump_diffusion.py     # Unit tests
│
├── 📁 scripts/                      # (Future) Production scripts
│
├── 📁 outputs/                      # VISUALIZATIONS
│   ├── jump_diffusion_analysis.png
│   ├── portfolio_comparison.png
│   ├── risk_comparison.png
│   ├── simulated_paths.png
│   └── stress_test_analysis.png
│
├── 📁 notebooks/                    # (Future) Jupyter notebooks
│
├── calculate_params.py             # Step 2: GBM parameters (ROOT)
├── requirements.txt               # Dependencies
├── README.md                      # YOU ARE HERE
└── .gitignore                     # Git ignore rules
```

---

## 📦 **DEPENDENCIES**

### `requirements.txt`
```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
plotly==5.15.0
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.1
pyarrow==12.0.1
feather-format==0.4.1
```

---

## 🧪 **VALIDATION**

### *Reproducibility Guarantee*
All scripts use fixed random seed: `RANDOM_SEED = 42`

```python
np.random.seed(42)  # Every simulation is reproducible
```

### *Data Integrity*
- ✅ No hardcoded values in simulations
- ✅ All parameters from `parameters.json`
- ✅ All simulation results in `dashboard_results.json`
- ✅ Historical crises documented with sources
- ✅ Full audit trail from raw data to dashboard

---

## 🎓 **WHAT I LEARNED**

### *Quantitative Finance*
- 📈 **GBM is insufficient** for assets with fat tails
- 🦘 **Jump Diffusion captures real market dynamics**
- 🔗 **Correlations amplify during crises** (0.83 normal → 0.95+ crisis)
- 📊 **Kurtosis > 10** means extreme events are NORMAL

### *Software Engineering*
- 🏗️ **Centralized config** (`src/config.py`) eliminates path hell
- 📁 **Single source of truth** (`data/processed/`) prevents duplication
- 🚀 **Caching** (`@st.cache_data`) makes dashboards fast
- 🔧 **Defensive programming** (fail fast, no silent defaults)

### *Data Science*
- 📥 **Feather format** is 10x faster than CSV
- 📊 **50k scenarios × 5 assets × 252 days** = 63 million paths
- ⚡ **Vectorized operations** reduce simulation time from hours to minutes
- 🎯 **Calibration threshold** (2.5σ) identifies real jumps vs noise

---

## 🔮 **FUTURE WORK**

- [ ] **Portfolio optimization** with Jump Diffusion
- [ ] **Option pricing** using Merton (1976) closed-form
- [ ] **DCC-GARCH** for time-varying correlations
- [ ] **Bayesian calibration** for jump parameters
- [ ] **Real-time dashboard** with live data feed
- [ ] **API endpoint** for risk metrics
- [ ] **PDF report generation**
- [ ] **Multi-currency support**

---

## 📄 **LICENSE**

MIT License

Copyright (c) 2026 Crypto Risk Engine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---

## 👨‍💻 **AUTHOR**

**Crypto Risk Engine**  
*Quantitative Finance | Data Science | Machine Learning*

- 📧 Email: dkysuarez1@gmail.com
- 🔗 LinkedIn: (https://www.linkedin.com/in/alisuarezgonzalez/))
- 💻 GitHub: (https://github.com/dkysuarez)

---

## ⚡ **QUICK START (30 SECONDS)**

```bash
# 1. Clone and enter
git clone https://github.com/dkysuarez/crypto-risk-engine.git
cd crypto-risk-engine

# 2. Install
pip install -r requirements.txt

# 3. Run full pipeline
python test/resample_to_daily.py
python calculate_params.py
python test/calibrate_all_jumps.py
python test/test_jump_vs_gbm.py

# 4. Launch dashboard
streamlit run app/Home.py
```

**Total time: ~5 minutes**  
**Total lines of code: 2,500+**  
**Total simulations: 250,000 scenarios**  
**Years of data: 5+ years**  
**Assets analyzed: 5 major cryptocurrencies**

---

## ⭐ **SUPPORT**

If you find this project useful, please consider:
- Giving it a ⭐ on GitHub
- Sharing it with your network
- Contributing via pull requests
- Reporting issues and bugs

---

**"In God we trust. All others must bring data."**  
— W. Edwards Deming

---

© 2026 Crypto Risk Engine. All rights reserved.  
Built with ❤️ for the quantitative finance community.
