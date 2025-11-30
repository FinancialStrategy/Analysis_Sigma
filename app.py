#pip install streamlit yfinance pandas numpy plotly pyportfolioopt matplotlib seaborn quantstats arch statsmodels scikit-learn
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import quantstats as qs
from arch import arch_model
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Institutional Quantitative Analysis",
    layout="wide",
    page_icon="📈"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .main-header { font-family: 'Helvetica Neue', sans-serif; font-size: 2.5rem; color: #0e1117; font-weight: 700; }
    .sub-header { font-family: 'Helvetica Neue', sans-serif; font-size: 1.5rem; color: #4c566a; }
    div[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #d6d9e0; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. Data Acquisition & Processing
# -----------------------------------------------------------------------------
TICKS = {
    "Gold": "GC=F",
    "Oil": "CL=F",
    "Platinum": "PL=F",
    "Copper": "HG=F",
    "Silver": "SI=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X"
}

@st.cache_data
def get_market_data(start_date="2015-01-01"):
    tickers_list = list(TICKS.values())
    data = yf.download(tickers_list, start=start_date, group_by='ticker', auto_adjust=True)
    close_prices = pd.DataFrame()
    for name, ticker in TICKS.items():
        try:
            if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                series = data[ticker]['Close']
            else:
                series = data['Close'] if len(tickers_list) == 1 else data[ticker]
            close_prices[name] = series
        except KeyError:
            continue
    return close_prices

def clean_data(df):
    # Forward fill then drop remaining NaNs (aligns to youngest asset)
    return df.ffill().dropna()

# -----------------------------------------------------------------------------
# Helper Functions for Advanced Analysis
# -----------------------------------------------------------------------------
def calculate_risk_contribution(weights, cov_matrix):
    """Calculates the percentage risk contribution of each asset."""
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
    risk_contribution = np.multiply(weights, marginal_contribution)
    # Normalize to percentages
    risk_contribution_pct = risk_contribution / portfolio_volatility
    return risk_contribution_pct, portfolio_volatility

def run_garch_optimization(returns):
    """Iterates through ARMA/GARCH params to find best AIC."""
    best_aic = np.inf
    best_model = None
    best_params = (1, 1)
    
    # Grid search for GARCH(p,q) - simplified for performance
    # Usually GARCH(1,1) is sufficient, but we will test up to (2,2)
    # We use a Constant Mean (ARMA(0,0)) for volatility modeling focus, 
    # but the user asked for ARMA degree checks. We will test AR(1) vs Constant Mean.
    
    mean_models = ['Constant', 'AR']
    qs = [1, 2]
    ps = [1, 2]
    
    results_log = []

    for mean_type in mean_models:
        for p in ps:
            for q in qs:
                try:
                    # Rescale returns for numerical stability in GARCH
                    am = arch_model(returns * 100, vol='Garch', p=p, q=q, mean=mean_type, lags=1 if mean_type=='AR' else 0)
                    res = am.fit(disp='off')
                    results_log.append({
                        'Model': f"{mean_type}-GARCH({p},{q})",
                        'AIC': res.aic,
                        'BIC': res.bic,
                        'Result': res
                    })
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_model = res
                        best_params = (mean_type, p, q)
                except:
                    continue
    
    return best_model, best_params, pd.DataFrame(results_log)

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------

# Sidebar Controls
st.sidebar.title("Configuration")
initial_investment = st.sidebar.number_input("Initial Capital (USD)", value=10_000_000, step=1_000_000)
start_date_input = st.sidebar.date_input("Data Start Date", value=pd.to_datetime("2015-01-01"))
risk_free_rate = st.sidebar.slider("Risk-Free Rate (Annual %)", 0.0, 10.0, 4.5, 0.1) / 100

st.markdown('<div class="main-header">Institutional Quantitative Suite</div>', unsafe_allow_html=True)
st.markdown(f"**Universe:** {', '.join(TICKS.keys())}")

# Load Data
with st.spinner('Fetching real-time market data...'):
    raw_data = get_market_data(start_date=start_date_input)
    df = clean_data(raw_data)

if df.empty:
    st.error("Insufficient data overlap. Please adjust start date.")
    st.stop()

# Returns Calculation
daily_returns = df.pct_change().dropna()
n_assets = len(df.columns)

# Tabs
tab_port, tab_qs, tab_risk = st.tabs(["Portfolio Optimization", "Asset Intelligence (QuantStats)", "Advanced Risk (GARCH & PCA)"])

# -----------------------------------------------------------------------------
# TAB 1: Portfolio Optimization (Existing & Refined)
# -----------------------------------------------------------------------------
with tab_port:
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Dataset Start", str(df.index.min().date()))
    col_kpi2.metric("Dataset End", str(df.index.max().date()))
    col_kpi3.metric("Trading Days", len(df))

    st.markdown("### Efficient Frontier & Optimization")
    col_opt1, col_opt2 = st.columns([1, 2])

    with col_opt1:
        st.info("Optimization Constraints")
        target_obj = st.radio("Objective", ["Max Sharpe Ratio", "Min Volatility", "Equal Weight"])
        
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)
        ef = EfficientFrontier(mu, S)
        
        if target_obj == "Max Sharpe Ratio":
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            weights_cleaned = ef.clean_weights()
            final_weights = np.array(list(weights_cleaned.values()))
        elif target_obj == "Min Volatility":
            weights = ef.min_volatility()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            weights_cleaned = ef.clean_weights()
            final_weights = np.array(list(weights_cleaned.values()))
        else:
            # Equal Weight Manual
            final_weights = np.array([1/n_assets] * n_assets)
            # Calculate manual performance metrics
            port_ret = np.sum(mu * final_weights)
            port_vol = np.sqrt(np.dot(final_weights.T, np.dot(S, final_weights)))
            port_sharpe = (port_ret - risk_free_rate) / port_vol
            perf = (port_ret, port_vol, port_sharpe)
            weights_cleaned = dict(zip(df.columns, final_weights))

        # Weights Table
        w_df = pd.DataFrame.from_dict(weights_cleaned, orient='index', columns=['Weight'])
        w_df = w_df[w_df['Weight'] > 0.0001]
        w_df['Allocation ($)'] = w_df['Weight'] * initial_investment
        w_df['Weight'] = w_df['Weight'].apply(lambda x: f"{x:.2%}")
        w_df['Allocation ($)'] = w_df['Allocation ($)'].apply(lambda x: f"${x:,.2f}")
        st.table(w_df)

        st.write(f"**Exp. Return:** {perf[0]:.2%}")
        st.write(f"**Volatility:** {perf[1]:.2%}")
        st.write(f"**Sharpe Ratio:** {perf[2]:.2f}")

    with col_opt2:
        # Frontier Plot
        # Generate random portfolios
        n_samples = 3000
        w_rand = np.random.dirichlet(np.ones(n_assets), n_samples)
        rets_rand = w_rand.dot(mu)
        stds_rand = np.sqrt(np.diag(w_rand @ S @ w_rand.T))
        sharpes_rand = (rets_rand - risk_free_rate) / stds_rand

        fig_frontier = go.Figure()
        fig_frontier.add_trace(go.Scatter(
            x=stds_rand, y=rets_rand, mode='markers',
            marker=dict(size=4, color=sharpes_rand, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
            name='Simulations'
        ))
        
        # Add Optimized Point
        fig_frontier.add_trace(go.Scatter(
            x=[perf[1]], y=[perf[0]], mode='markers+text',
            marker=dict(size=18, color='red', symbol='star'),
            text=[f"Chosen Strategy"], textposition="top left",
            name='Selected'
        ))

        fig_frontier.update_layout(title="Efficient Frontier", xaxis_title="Risk (Vol)", yaxis_title="Return", height=600, template="plotly_white")
        st.plotly_chart(fig_frontier, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: Asset Intelligence (QuantStats)
# -----------------------------------------------------------------------------
with tab_qs:
    st.markdown("### Institutional Performance Reporting")
    selected_asset = st.selectbox("Select Instrument for Deep Dive", df.columns)
    
    qs_col1, qs_col2 = st.columns([1, 3])
    
    asset_series = daily_returns[selected_asset]
    
    # Calculate Metrics
    # QuantStats usually works with a benchmark, but we can run 'basic' metrics or against zero
    with qs_col1:
        st.subheader("Key Metrics")
        
        # Manual calculation via QuantStats functions for control
        sharpe = qs.stats.sharpe(asset_series, rf=risk_free_rate)
        sortino = qs.stats.sortino(asset_series, rf=risk_free_rate)
        max_dd = qs.stats.max_drawdown(asset_series)
        win_rate = qs.stats.win_rate(asset_series)
        volatility = qs.stats.volatility(asset_series)
        calmar = qs.stats.calmar(asset_series)
        cvar = qs.stats.cvar(asset_series)
        
        metrics_df = pd.DataFrame({
            "Metric": ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Win Rate", "Annual Volatility", "Calmar Ratio", "CVaR (95%)"],
            "Value": [
                f"{sharpe:.2f}", 
                f"{sortino:.2f}", 
                f"{max_dd:.2%}", 
                f"{win_rate:.2%}", 
                f"{volatility:.2%}", 
                f"{calmar:.2f}", 
                f"{cvar:.2%}"
            ]
        })
        st.table(metrics_df)

    with qs_col2:
        st.subheader("Drawdown & Returns Analysis")
        
        # Cumulative Returns
        fig_cum = qs.plots.snapshot(asset_series, show=False, title=f"Cumulative Returns: {selected_asset}")
        # QS plots are matplotlib, convert/display
        st.pyplot(fig_cum)
        
        st.write("---")
        
        # Monthly Heatmap
        st.subheader("Monthly Returns Heatmap")
        try:
            fig_heat = qs.plots.monthly_heatmap(asset_series, show=False)
            st.pyplot(fig_heat)
        except:
            st.warning("Not enough data for heatmap.")

# -----------------------------------------------------------------------------
# TAB 3: Advanced Risk (Econometrics & PCA)
# -----------------------------------------------------------------------------
with tab_risk:
    st.markdown("### 1. Portfolio Structure & Risk Contribution")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("**Principal Component Analysis (PCA)**")
        # Standardize returns
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(daily_returns)
        
        pca = PCA()
        pca.fit(scaled_returns)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        pca_df = pd.DataFrame({
            'PC': [f"PC{i+1}" for i in range(len(explained_variance))],
            'Explained Variance': explained_variance,
            'Cumulative Variance': cumulative_variance
        })
        
        fig_pca = px.bar(pca_df, x='PC', y='Explained Variance', title="PCA: Explained Variance by Component")
        fig_pca.add_trace(go.Scatter(x=pca_df['PC'], y=pca_df['Cumulative Variance'], mode='lines+markers', name='Cumulative'))
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption("PC1 typically represents the 'Market Factor' (systematic risk).")

    with risk_col2:
        st.markdown("**Risk Contribution (to Portfolio Volatility)**")
        # Use the weights calculated in Tab 1
        rc_pct, p_vol = calculate_risk_contribution(final_weights, S.values)
        
        rc_df = pd.DataFrame({
            'Asset': df.columns,
            'Risk Contribution %': rc_pct * 100, # Display as 0-100 number
            'Weight %': final_weights * 100
        })
        
        fig_rc = px.bar(rc_df, x='Asset', y=['Risk Contribution %', 'Weight %'], barmode='group', 
                        title="Risk Contribution vs. Capital Allocation")
        st.plotly_chart(fig_rc, use_container_width=True)
        st.caption("If Risk Contribution > Weight, the asset is adding concentration risk to the portfolio.")

    st.markdown("---")
    st.markdown("### 2. Time Series Econometrics (GARCH)")
    
    ts_asset = st.selectbox("Select Asset for GARCH Modeling", df.columns, key="garch_select")
    ts_data = daily_returns[ts_asset].dropna()
    
    ts_col1, ts_col2 = st.columns([1, 2])
    
    with ts_col1:
        st.markdown("#### Step A: Pre-Estimation Tests")
        
        # 1. Stationarity (ADF)
        adf_result = adfuller(ts_data)
        st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
        st.write(f"**p-value:** {adf_result[1]:.4f}")
        if adf_result[1] < 0.05:
            st.success("✅ Series is Stationary")
        else:
            st.error("❌ Series is Non-Stationary (Diff required)")
            
        # 2. ARCH Effect (Engle's LM Test)
        # We test residuals of mean equation (assume constant mean for test)
        resid = ts_data - ts_data.mean()
        lm_test = het_arch(resid)
        st.write(f"**ARCH LM Test p-value:** {lm_test[1]:.4f}")
        if lm_test[1] < 0.05:
            st.success("✅ ARCH Effect Detected (GARCH appropriate)")
        else:
            st.warning("⚠️ No significant ARCH effect detected")
            
    with ts_col2:
        st.markdown("#### Step B: Best Model Selection (AIC)")
        
        if st.button("Run Model Search (ARMA-GARCH)"):
            with st.spinner(f"Optimizing GARCH parameters for {ts_asset}..."):
                best_model, best_params, log_df = run_garch_optimization(ts_data)
                
                st.write(f"**Best Model:** {best_params[0]}-GARCH({best_params[1]},{best_params[2]})")
                st.write(f"**AIC:** {best_model.aic:.2f}")
                
                # Display Top 3 Models
                st.dataframe(log_df.sort_values('AIC').head(3), height=150)
                
                # Plot Volatility
                st.subheader("Conditional Volatility (Annualized)")
                cond_vol = best_model.conditional_volatility
                # Annualize (assuming daily data)
                ann_vol = cond_vol * np.sqrt(252)
                
                fig_garch = px.line(x=ts_data.index, y=ann_vol, title=f"Estimated Volatility: {ts_asset}")
                fig_garch.update_layout(yaxis_title="Annualized Volatility (%)")
                st.plotly_chart(fig_garch, use_container_width=True)
                
                st.markdown("**Model Summary**")
                st.text(best_model.summary())
        else:
            st.info("Click button to perform grid search for best ARIMA/GARCH parameters.")