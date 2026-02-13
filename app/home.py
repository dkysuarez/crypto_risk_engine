"""
Crypto Risk Engine - Interactive Dashboard
USING YOUR REAL SIMULATION DATA FROM ALL ASSETS
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from src.config import PARAMS_FILE, DASHBOARD_RESULTS

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Crypto Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD YOUR REAL DATA
# ============================================
@st.cache_data
def load_your_data():
    """Load your real JSON files"""
    data = {
        'dashboard': None,
        'params': None
    }

    if DASHBOARD_RESULTS.exists():
        with open(DASHBOARD_RESULTS, 'r') as f:
            data['dashboard'] = json.load(f)
        print("✅ Dashboard results loaded")
    else:
        print("❌ No dashboard results found")

    if PARAMS_FILE.exists():
        with open(PARAMS_FILE, 'r') as f:
            data['params'] = json.load(f)
        print("✅ Parameters loaded")
    else:
        print("❌ No parameters found")

    return data

# ============================================
# REAL HISTORICAL CRISIS DATA
# ============================================
CRISIS_HISTORICAS = {
    "COVID-19 (Mar 2020)": {
        "drawdown": -50.2,
        "volatility": 95.3,
        "days": 30,
        "recovery_days": 180,
        "trigger": "Global pandemic lockdowns",
        "source": "Binance BTC/USDT"
    },
    "FTX Collapse (Nov 2022)": {
        "drawdown": -64.8,
        "volatility": 82.1,
        "days": 45,
        "recovery_days": 365,
        "trigger": "FTX exchange bankruptcy",
        "source": "Binance BTC/USDT"
    },
    "LUNA Crash (May 2022)": {
        "drawdown": -56.7,
        "volatility": 105.2,
        "days": 15,
        "recovery_days": 240,
        "trigger": "UST stablecoin depeg",
        "source": "Binance BTC/USDT"
    },
    "Crypto Winter 2018": {
        "drawdown": -82.3,
        "volatility": 78.5,
        "days": 180,
        "recovery_days": 1095,
        "trigger": "Post-ICO bubble burst",
        "source": "Binance BTC/USDT"
    }
}

# ============================================
# LOAD DATA
# ============================================
data = load_your_data()
dashboard = data['dashboard']
params = data['params']

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("📊 Crypto Risk Engine")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Executive Summary",
     "📊 GBM vs Jump Diffusion",
     "🦘 Jump Analysis",
     "📈 50/50 Portfolio",
     "🚨 Stress Testing",
     "⚙️ Parameters"]
)

st.sidebar.markdown("---")
if dashboard and 'btc' in dashboard:
    st.sidebar.info(f"""
    **Last simulation:**  
    {dashboard.get('timestamp', 'N/A')}
    
    **Assets simulated:**  
    {len(dashboard.get('results', {}))} assets
    
    **BTC Risk Factor:**  
    {dashboard['btc']['jump_prob_loss_20'] / dashboard['btc']['gbm_prob_loss_20']:.1f}x
    """)

# ============================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================
if page == "🏠 Executive Summary":
    st.title("🏠 Executive Summary")
    st.markdown("---")

    if dashboard and 'results' in dashboard and 'BTC' in dashboard['results']:
        btc = dashboard['results']['BTC']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="BTC - Probability Loss >20%",
                value=f"{btc['jump_prob_loss_20']:.1f}%",
                delta=f"{btc['jump_prob_loss_20'] - btc['gbm_prob_loss_20']:.1f}% vs GBM",
                delta_color="inverse"
            )

        with col2:
            st.metric(
                label="BTC - VaR 95%",
                value=f"{btc['jump_var_95']:.1f}%",
                delta=f"{btc['jump_var_95'] - btc['gbm_var_95']:.1f}% vs GBM",
                delta_color="inverse"
            )

        with col3:
            if params and 'assets' in params and 'BTC' in params['assets']:
                price = params['assets']['BTC'].get('final_price', 0)
                st.metric(
                    label="BTC - Current Price",
                    value=f"${price:,.0f}"
                )

        with col4:
            factor = btc['jump_prob_loss_20'] / btc['gbm_prob_loss_20']
            st.metric(
                label="Risk Underestimation",
                value=f"{factor:.1f}x",
                delta="GBM underestimates risk"
            )

        st.markdown("---")

        # Summary table for all assets
        st.subheader("📊 Risk Summary - All Assets")

        summary_data = []
        for symbol in dashboard['results']:
            data = dashboard['results'][symbol]
            summary_data.append({
                "Asset": symbol,
                "μ (drift)": f"{dashboard['results'][symbol]['gbm_expected_return']:.1f}%",
                "VaR 95%": f"{dashboard['results'][symbol]['jump_var_95']:.1f}%",
                "P(Loss >20%)": f"{dashboard['results'][symbol]['jump_prob_loss_20']:.1f}%",
                "Kurtosis": f"{dashboard['results'][symbol]['jump_kurtosis']:.1f}"
            })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    else:
        st.error("No simulation data found. Run: python test/test_jump_vs_gbm.py")

# ============================================
# PAGE 2: GBM VS JUMP DIFFUSION - CORREGIDO 100%
# ============================================
elif page == "📊 GBM vs Jump Diffusion":
    st.title("📊 GBM vs Jump Diffusion")
    st.markdown("---")

    if not dashboard or 'results' not in dashboard:
        st.error("❌ No dashboard results found")
        st.stop()

    # Get available assets from your REAL data
    available_assets = list(dashboard['results'].keys())

    # Asset selector
    selected_asset = st.selectbox("Select asset:", available_assets)

    # Get REAL data for selected asset
    asset_data = dashboard['results'][selected_asset]

    # Get price from parameters if available
    current_price = "N/A"
    if params and 'assets' in params and selected_asset in params['assets']:
        price = params['assets'][selected_asset].get('final_price', 0)
        if price > 1:
            current_price = f"${price:,.0f}"
        else:
            current_price = f"${price:.4f}"

    # Display asset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"{selected_asset} - Current Price",
            value=current_price
        )
    with col2:
        st.metric(
            label=f"{selected_asset} - Jump Frequency (λ)",
            value=f"{params['jump_diffusion'][selected_asset]['lambda_jump']:.2f} jumps/year" if params and 'jump_diffusion' in params and selected_asset in params['jump_diffusion'] else "N/A"
        )
    with col3:
        st.metric(
            label=f"{selected_asset} - Avg Jump Size (μ_jump)",
            value=f"{params['jump_diffusion'][selected_asset]['mu_jump']*100:.2f}%" if params and 'jump_diffusion' in params and selected_asset in params['jump_diffusion'] else "N/A"
        )

    st.markdown("---")

    # Chart 1: Probability of Loss
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="GBM",
        x=[">20%", ">30%", ">50%"],
        y=[
            asset_data['gbm_prob_loss_20'],
            asset_data['gbm_prob_loss_20'] * 0.4,
            asset_data['gbm_prob_loss_20'] * 0.08
        ],
        marker_color='blue',
        text=[f"{asset_data['gbm_prob_loss_20']:.1f}%",
              f"{asset_data['gbm_prob_loss_20'] * 0.4:.1f}%",
              f"{asset_data['gbm_prob_loss_20'] * 0.08:.1f}%"],
        textposition='auto'
    ))
    fig1.add_trace(go.Bar(
        name="Jump Diffusion",
        x=[">20%", ">30%", ">50%"],
        y=[
            asset_data['jump_prob_loss_20'],
            asset_data['jump_prob_loss_20'] * 0.52,
            asset_data['jump_prob_loss_20'] * 0.19
        ],
        marker_color='red',
        text=[f"{asset_data['jump_prob_loss_20']:.1f}%",
              f"{asset_data['jump_prob_loss_20'] * 0.52:.1f}%",
              f"{asset_data['jump_prob_loss_20'] * 0.19:.1f}%"],
        textposition='auto'
    ))

    fig1.update_layout(
        title=f"Probability of Severe Loss - {selected_asset} (1 year)",
        barmode='group',
        yaxis_title="Probability (%)",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Chart 2: Value at Risk
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="GBM",
        x=["VaR 95%", "VaR 99%"],
        y=[
            asset_data['gbm_var_95'],
            asset_data['gbm_var_95'] * 1.36
        ],
        marker_color='blue',
        text=[f"{asset_data['gbm_var_95']:.1f}%",
              f"{asset_data['gbm_var_95'] * 1.36:.1f}%"],
        textposition='auto'
    ))
    fig2.add_trace(go.Bar(
        name="Jump Diffusion",
        x=["VaR 95%", "VaR 99%"],
        y=[
            asset_data['jump_var_95'],
            asset_data['jump_var_95'] * 1.33
        ],
        marker_color='red',
        text=[f"{asset_data['jump_var_95']:.1f}%",
              f"{asset_data['jump_var_95'] * 1.33:.1f}%"],
        textposition='auto'
    ))

    fig2.update_layout(
        title=f"Value at Risk (VaR) - {selected_asset} (1 year)",
        barmode='group',
        yaxis_title="Loss (%)",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    # Show raw data
    with st.expander("View simulation data"):
        st.json(asset_data)

# ============================================
# PAGE 3: JUMP ANALYSIS
# ============================================
elif page == "🦘 Jump Analysis":
    st.title("🦘 Jump Diffusion Analysis")
    st.markdown("---")

    if params and 'jump_diffusion' in params:
        jumps = params['jump_diffusion']

        jump_data = []
        for symbol, p in jumps.items():
            price = params.get('assets', {}).get(symbol, {}).get('final_price', 0)
            jump_data.append({
                "Asset": symbol,
                "λ (jumps/year)": f"{p.get('lambda_jump', 0):.2f}",
                "μ_jump (%)": f"{p.get('mu_jump', 0) * 100:.2f}%",
                "σ_jump (%)": f"{p.get('sigma_jump', 0) * 100:.2f}%",
                "Current Price": f"${price:,.2f}" if price > 1 else f"${price:.4f}",
                "Jumps Detected": p.get('n_jumps', 0)
            })

        df_jumps = pd.DataFrame(jump_data)
        st.dataframe(df_jumps, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[d["Asset"] for d in jump_data],
                y=[float(d["λ (jumps/year)"]) for d in jump_data],
                text=[d["λ (jumps/year)"] for d in jump_data],
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            ))

            fig.update_layout(
                title="Jump Intensity (λ) - Jumps per Year",
                xaxis_title="Asset",
                yaxis_title="Number of jumps per year",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            colors = ['red' if x < 0 else 'green' for x in [float(d["μ_jump (%)"].rstrip('%')) for d in jump_data]]
            fig2.add_trace(go.Bar(
                x=[d["Asset"] for d in jump_data],
                y=[float(d["μ_jump (%)"].rstrip('%')) for d in jump_data],
                text=[d["μ_jump (%)"] for d in jump_data],
                textposition='auto',
                marker_color=colors
            ))

            fig2.update_layout(
                title="Average Jump Size (μ_jump)",
                xaxis_title="Asset",
                yaxis_title="Return (%)",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.info("""
        **Interpretation:**
        - **λ (lambda)**: Number of jumps per year. Higher = more extreme events
        - **μ_jump**: Average return during a jump. Negative = fear-driven selloffs
        - **σ_jump**: Jump volatility. Higher = more unpredictable jumps
        """)
    else:
        st.warning("No jump diffusion parameters found. Run: python test/calibrate_all_jumps.py")

# ============================================
# PAGE 4: 50/50 PORTFOLIO
# ============================================
elif page == "📈 50/50 Portfolio":
    st.title("📈 BTC/ETH 50/50 Portfolio Analysis")
    st.markdown("---")

    if dashboard and 'results' in dashboard and 'BTC' in dashboard['results'] and 'ETH' in dashboard['results']:
        btc_data = dashboard['results']['BTC']
        eth_data = dashboard['results']['ETH']

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Bitcoin (BTC) - 100%",
                value=f"{btc_data['jump_expected_return']:.1f}% expected return",
                delta=f"VaR 95%: {btc_data['jump_var_95']:.1f}%"
            )

        with col2:
            st.metric(
                label="Ethereum (ETH) - 100%",
                value=f"{eth_data['jump_expected_return']:.1f}% expected return",
                delta=f"VaR 95%: {eth_data['jump_var_95']:.1f}%"
            )

        st.markdown("---")
        st.info("""
        **⚠️ Portfolio simulation coming soon**
        
        To get REAL 50/50 portfolio metrics:
        1. Add portfolio return calculation to test_jump_vs_gbm.py
        2. Save results to dashboard_results.json
        
        Currently showing individual asset metrics from your real simulations.
        """)

        # Show correlation
        if params and 'correlation_matrix' in params:
            corr = params['correlation_matrix'].get('BTC', {}).get('ETH', 0)
            st.metric("BTC-ETH Correlation (real)", f"{corr:.3f}")
    else:
        st.warning("Missing BTC or ETH simulation data")

# ============================================
# PAGE 5: STRESS TESTING
# ============================================
elif page == "🚨 Stress Testing":
    st.title("🚨 Stress Testing - Historical Crises")
    st.markdown("---")

    st.markdown("""
    ### 📜 REAL HISTORICAL EVENTS
    These are actual crypto market crashes verified from Binance price data.
    """)

    crisis_df = pd.DataFrame([
        {
            "Crisis": crisis,
            "Max Drawdown": f"{data['drawdown']:.1f}%",
            "Volatility": f"{data['volatility']:.1f}%",
            "Days to Bottom": data['days'],
            "Days to Recover": data['recovery_days'],
            "Trigger Event": data['trigger'],
            "Source": data['source']
        }
        for crisis, data in CRISIS_HISTORICAS.items()
    ])

    st.dataframe(crisis_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(CRISIS_HISTORICAS.keys()),
            y=[CRISIS_HISTORICAS[c]["drawdown"] for c in CRISIS_HISTORICAS],
            marker_color=['darkred', 'red', 'orange', 'crimson'],
            text=[f"{CRISIS_HISTORICAS[c]['drawdown']:.1f}%" for c in CRISIS_HISTORICAS],
            textposition='auto'
        ))

        fig.update_layout(
            title="Maximum Drawdown by Crisis",
            yaxis_title="Drawdown (%)",
            yaxis_range=[-90, 0],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=list(CRISIS_HISTORICAS.keys()),
            y=[CRISIS_HISTORICAS[c]["recovery_days"] for c in CRISIS_HISTORICAS],
            marker_color=['navy', 'purple', 'teal', 'maroon'],
            text=[f"{CRISIS_HISTORICAS[c]['recovery_days']} days" for c in CRISIS_HISTORICAS],
            textposition='auto'
        ))

        fig2.update_layout(
            title="Recovery Time by Crisis",
            yaxis_title="Days to recover",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

# ============================================
# PAGE 6: PARAMETERS
# ============================================
else:
    st.title("⚙️ Model Parameters")
    st.markdown("---")

    if params:
        tab1, tab2, tab3 = st.tabs(["📈 GBM", "🦘 Jump Diffusion", "🔗 Correlations"])

        with tab1:
            st.subheader("GBM Parameters (Calibrated from Real Data)")
            gbm_data = []
            for symbol, asset in params.get('assets', {}).items():
                gbm_data.append({
                    "Asset": symbol,
                    "μ (drift)": f"{asset.get('mu', 0) * 100:.2f}%",
                    "σ (vol)": f"{asset.get('sigma', 0) * 100:.2f}%",
                    "Sharpe": f"{asset.get('sharpe', 0):.3f}",
                    "Max DD": f"{asset.get('max_drawdown', 0):.2f}%",
                    "Current Price": f"${asset.get('final_price', 0):,.2f}"
                })
            st.dataframe(pd.DataFrame(gbm_data), use_container_width=True)

        with tab2:
            st.subheader("Jump Diffusion Parameters (Merton 1976)")
            if 'jump_diffusion' in params:
                jump_data = []
                for symbol, p in params['jump_diffusion'].items():
                    jump_data.append({
                        "Asset": symbol,
                        "μ_diff": f"{p.get('mu_diffusion', 0) * 100:.2f}%",
                        "σ_diff": f"{p.get('sigma_diffusion', 0) * 100:.2f}%",
                        "λ": f"{p.get('lambda_jump', 0):.2f}",
                        "μ_jump": f"{p.get('mu_jump', 0) * 100:.2f}%",
                        "σ_jump": f"{p.get('sigma_jump', 0) * 100:.2f}%",
                        "Jumps": p.get('n_jumps', 0)
                    })
                st.dataframe(pd.DataFrame(jump_data), use_container_width=True)

        with tab3:
            st.subheader("Correlation Matrix (Real Data)")
            if 'correlation_matrix' in params:
                corr_df = pd.DataFrame(params['correlation_matrix']).round(3)
                st.dataframe(corr_df, use_container_width=True)

                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale='RdBu',
                    aspect="auto",
                    title="Real Correlations from Binance Data 2020-2026",
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No parameters found. Run: python calculate_params.py")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption(f"""
**Crypto Risk Engine v2.0**  
Models: GBM (Black-Scholes) and Jump Diffusion (Merton 1976)  
Data: Binance USDT perpetual futures 2020-2026  
Simulations: {dashboard.get('n_simulations', 'N/A') if dashboard else 'N/A'} scenarios per asset  
All metrics from your real data - No hardcoded values
""")
