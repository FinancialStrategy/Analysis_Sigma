import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# --- DÜZELTİLEN SATIRLAR (8-10. Satırlar) ---
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, plotting
# --------------------------------------------
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

# Uyarıları gizle
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Ayarlar
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kurumsal Kantitatif Analiz",
    layout="wide",
    page_icon="📈"
)

# Özel Tasarım CSS
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
# 1. Veri Çekme ve İşleme
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
    # Eksik verileri doldur ve en genç varlığa göre hizala
    return df.ffill().dropna()

# -----------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------
def calculate_risk_contribution(weights, cov_matrix):
    """Her varlığın portföy riskine katkısını hesaplar."""
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
    risk_contribution = np.multiply(weights, marginal_contribution)
    risk_contribution_pct = risk_contribution / portfolio_volatility
    return risk_contribution_pct, portfolio_volatility

def run_garch_optimization(returns):
    """En iyi AIC değerine sahip GARCH modelini bulur."""
    best_aic = np.inf
    best_model = None
    best_params = (1, 1)
    
    mean_models = ['Constant', 'AR']
    qs = [1, 2]
    ps = [1, 2]
    
    results_log = []

    for mean_type in mean_models:
        for p in ps:
            for q in qs:
                try:
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
# Ana Uygulama Mantığı
# -----------------------------------------------------------------------------

# Kenar Çubuğu
st.sidebar.title("Ayarlar")
initial_investment = st.sidebar.number_input("Başlangıç Sermayesi (USD)", value=10_000_000, step=1_000_000)
start_date_input = st.sidebar.date_input("Veri Başlangıç Tarihi", value=pd.to_datetime("2015-01-01"))
risk_free_rate = st.sidebar.slider("Risksiz Faiz Oranı (Yıllık %)", 0.0, 10.0, 4.5, 0.1) / 100

st.markdown('<div class="main-header">Kurumsal Kantitatif Analiz Paketi</div>', unsafe_allow_html=True)
st.markdown(f"**Varlık Evreni:** {', '.join(TICKS.keys())}")

# Veri Yükleme
with st.spinner('Piyasa verileri çekiliyor...'):
    raw_data = get_market_data(start_date=start_date_input)
    df = clean_data(raw_data)

if df.empty:
    st.error("Yeterli veri yok. Lütfen tarihi değiştirin.")
    st.stop()

# Getiriler
daily_returns = df.pct_change().dropna()
n_assets = len(df.columns)

# Sekmeler
tab_port, tab_qs, tab_risk = st.tabs(["Portföy Optimizasyonu", "Varlık Analizi (QuantStats)", "İleri Risk (GARCH & PCA)"])

# -----------------------------------------------------------------------------
# SEKME 1: Portföy Optimizasyonu
# -----------------------------------------------------------------------------
with tab_port:
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Veri Başlangıcı", str(df.index.min().date()))
    col_kpi2.metric("Veri Bitişi", str(df.index.max().date()))
    col_kpi3.metric("İşlem Günü", len(df))

    st.markdown("### Etkin Sınır ve Optimizasyon")
    col_opt1, col_opt2 = st.columns([1, 2])

    with col_opt1:
        st.info("Optimizasyon Hedefi")
        target_obj = st.radio("Hedef", ["Maksimum Sharpe", "Minimum Volatilite", "Eşit Ağırlık"])
        
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)
        
        # Etkin Sınır Hesabı
        ef = EfficientFrontier(mu, S)
        
        if target_obj == "Maksimum Sharpe":
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            weights_cleaned = ef.clean_weights()
            final_weights = np.array(list(weights_cleaned.values()))
        elif target_obj == "Minimum Volatilite":
            weights = ef.min_volatility()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            weights_cleaned = ef.clean_weights()
            final_weights = np.array(list(weights_cleaned.values()))
        else:
            # Eşit Ağırlık
            final_weights = np.array([1/n_assets] * n_assets)
            port_ret = np.sum(mu * final_weights)
            port_vol = np.sqrt(np.dot(final_weights.T, np.dot(S, final_weights)))
            port_sharpe = (port_ret - risk_free_rate) / port_vol
            perf = (port_ret, port_vol, port_sharpe)
            weights_cleaned = dict(zip(df.columns, final_weights))

        # Ağırlık Tablosu
        w_df = pd.DataFrame.from_dict(weights_cleaned, orient='index', columns=['Ağırlık'])
        w_df = w_df[w_df['Ağırlık'] > 0.0001]
        w_df['Tutar ($)'] = w_df['Ağırlık'] * initial_investment
        w_df['Ağırlık'] = w_df['Ağırlık'].apply(lambda x: f"{x:.2%}")
        w_df['Tutar ($)'] = w_df['Tutar ($)'].apply(lambda x: f"${x:,.2f}")
        st.table(w_df)

        st.write(f"**Beklenen Getiri:** {perf[0]:.2%}")
        st.write(f"**Volatilite (Risk):** {perf[1]:.2%}")
        st.write(f"**Sharpe Oranı:** {perf[2]:.2f}")

    with col_opt2:
        # Simülasyon
        n_samples = 3000
        w_rand = np.random.dirichlet(np.ones(n_assets), n_samples)
        rets_rand = w_rand.dot(mu)
        stds_rand = np.sqrt(np.diag(w_rand @ S @ w_rand.T))
        sharpes_rand = (rets_rand - risk_free_rate) / stds_rand

        fig_frontier = go.Figure()
        fig_frontier.add_trace(go.Scatter(
            x=stds_rand, y=rets_rand, mode='markers',
            marker=dict(size=4, color=sharpes_rand, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")),
            name='Simülasyonlar'
        ))
        
        # Seçilen Nokta
        fig_frontier.add_trace(go.Scatter(
            x=[perf[1]], y=[perf[0]], mode='markers+text',
            marker=dict(size=18, color='red', symbol='star'),
            text=[f"Seçilen Strateji"], textposition="top left",
            name='Seçilen'
        ))

        fig_frontier.update_layout(title="Etkin Sınır (Efficient Frontier)", xaxis_title="Risk (Volatilite)", yaxis_title="Getiri", height=600, template="plotly_white")
        st.plotly_chart(fig_frontier, use_container_width=True)

# -----------------------------------------------------------------------------
# SEKME 2: Varlık Analizi (QuantStats)
# -----------------------------------------------------------------------------
with tab_qs:
    st.markdown("### Detaylı Performans Raporu")
    selected_asset = st.selectbox("İncelenecek Varlığı Seçin", df.columns)
    
    qs_col1, qs_col2 = st.columns([1, 3])
    
    asset_series = daily_returns[selected_asset]
    
    with qs_col1:
        st.subheader("Temel Metrikler")
        try:
            sharpe = qs.stats.sharpe(asset_series, rf=risk_free_rate)
            sortino = qs.stats.sortino(asset_series, rf=risk_free_rate)
            max_dd = qs.stats.max_drawdown(asset_series)
            win_rate = qs.stats.win_rate(asset_series)
            volatility = qs.stats.volatility(asset_series)
            calmar = qs.stats.calmar(asset_series)
            cvar = qs.stats.cvar(asset_series)
            
            metrics_df = pd.DataFrame({
                "Metrik": ["Sharpe", "Sortino", "Maks. Düşüş", "Kazanma Oranı", "Yıllık Volatilite", "Calmar", "CVaR (95%)"],
                "Değer": [f"{sharpe:.2f}", f"{sortino:.2f}", f"{max_dd:.2%}", f"{win_rate:.2%}", f"{volatility:.2%}", f"{calmar:.2f}", f"{cvar:.2%}"]
            })
            st.table(metrics_df)
        except Exception as e:
            st.error(f"Metrik hesaplanırken hata: {e}")

    with qs_col2:
        st.subheader("Kümülatif Getiri ve Aylık Performans")
        try:
            fig_cum = qs.plots.snapshot(asset_series, show=False, title=f"{selected_asset} Performans Özeti")
            st.pyplot(fig_cum)
        except:
            st.warning("Grafik oluşturulamadı.")
        
        st.write("---")
        try:
            fig_heat = qs.plots.monthly_heatmap(asset_series, show=False)
            st.pyplot(fig_heat)
        except:
            st.warning("Isı haritası için yeterli veri yok.")

# -----------------------------------------------------------------------------
# SEKME 3: İleri Risk (GARCH & PCA)
# -----------------------------------------------------------------------------
with tab_risk:
    st.markdown("### 1. Risk Yapısı ve PCA")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("**Temel Bileşen Analizi (PCA)**")
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(daily_returns)
        
        pca = PCA()
        pca.fit(scaled_returns)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        pca_df = pd.DataFrame({
            'Bileşen': [f"PC{i+1}" for i in range(len(explained_variance))],
            'Açıklanan Varyans': explained_variance,
            'Kümülatif': cumulative_variance
        })
        
        fig_pca = px.bar(pca_df, x='Bileşen', y='Açıklanan Varyans', title="PCA: Piyasa Risk Faktörleri")
        fig_pca.add_trace(go.Scatter(x=pca_df['Bileşen'], y=pca_df['Kümülatif'], mode='lines+markers', name='Kümülatif'))
        st.plotly_chart(fig_pca, use_container_width=True)

    with risk_col2:
        st.markdown("**Riske Katkı Analizi**")
        rc_pct, p_vol = calculate_risk_contribution(final_weights, S.values)
        
        rc_df = pd.DataFrame({
            'Varlık': df.columns,
            'Riske Katkı %': rc_pct * 100, 
            'Ağırlık %': final_weights * 100
        })
        
        fig_rc = px.bar(rc_df, x='Varlık', y=['Riske Katkı %', 'Ağırlık %'], barmode='group', 
                        title="Riske Katkı vs. Sermaye Ağırlığı")
        st.plotly_chart(fig_rc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 2. Ekonometrik Modelleme (GARCH)")
    
    ts_asset = st.selectbox("GARCH Modeli İçin Varlık Seçin", df.columns, key="garch_select")
    ts_data = daily_returns[ts_asset].dropna()
    
    ts_col1, ts_col2 = st.columns([1, 2])
    
    with ts_col1:
        st.markdown("#### Adım A: Ön Testler")
        try:
            adf_result = adfuller(ts_data)
            st.write(f"**ADF İstatistiği:** {adf_result[0]:.4f}")
            st.write(f"**p-değeri:** {adf_result[1]:.4f}")
            if adf_result[1] < 0.05:
                st.success("✅ Seri Durağan")
            else:
                st.error("❌ Seri Durağan Değil")
                
            resid = ts_data - ts_data.mean()
            lm_test = het_arch(resid)
            st.write(f"**ARCH LM Test p-değeri:** {lm_test[1]:.4f}")
            if lm_test[1] < 0.05:
                st.success("✅ ARCH Etkisi Var (GARCH uygun)")
            else:
                st.warning("⚠️ Belirgin ARCH etkisi yok")
        except Exception as e:
            st.error(f"Test hatası: {e}")
            
    with ts_col2:
        st.markdown("#### Adım B: Model Seçimi (AIC)")
        if st.button("En İyi Modeli Bul (ARMA-GARCH)"):
            with st.spinner(f"{ts_asset} için optimizasyon yapılıyor..."):
                try:
                    best_model, best_params, log_df = run_garch_optimization(ts_data)
                    
                    st.write(f"**En İyi Model:** {best_params[0]}-GARCH({best_params[1]},{best_params[2]})")
                    st.write(f"**AIC Değeri:** {best_model.aic:.2f}")
                    
                    st.dataframe(log_df.sort_values('AIC').head(3), height=150)
                    
                    cond_vol = best_model.conditional_volatility
                    ann_vol = cond_vol * np.sqrt(252)
                    
                    fig_garch = px.line(x=ts_data.index, y=ann_vol, title=f"Tahmin Edilen Volatilite: {ts_asset}")
                    st.plotly_chart(fig_garch, use_container_width=True)
                except Exception as e:
                     st.error(f"Model hatası: {e}")
        else:
            st.info("Modeli çalıştırmak için butona basın.")