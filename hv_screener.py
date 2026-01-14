"""
Historical Volatility Screener & Option Pricer
==============================================
A robust dashboard for Market Makers to analyze volatility regimes.
Supports Binance Global, Binance.US, and GeckoTerminal (DEX) for memecoins.

Features:
- Multi-Exchange Support (Bypass Geo-blocking)
- Spot vs Perpetual toggle (where available)
- RMS Volatility Calculation (7/14 blending)
- Black-Scholes Greeks Calculator
- Bulk CSV Data Export
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta, timezone
import time
import os
import io

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="MM Volatility Dashboard",
    page_icon="⚡",
)

st.title("⚡ Dynamic Volatility Dashboard")
st.markdown(
    """
    **Market Maker Risk Engine**
    Analyze historical volatility (RMS 7/14) to set optimal option/perp pricing.
    """
)

# -----------------------------------------------------------------------------
# 2. ASSET & CONFIGURATION
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_default_assets():
    """Returns a robust default list if CSV is missing."""
    return pd.DataFrame([
        # Majors
        {"Coin symbol": "BTC", "Name": "Bitcoin", "Pool": "eth_0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"}, 
        {"Coin symbol": "ETH", "Name": "Ethereum", "Pool": "eth_0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"},
        {"Coin symbol": "SOL", "Name": "Solana", "Pool": "solana_HgFq..."},
        # Mems / Alts
        {"Coin symbol": "DOGE", "Name": "Dogecoin", "Pool": ""},
        {"Coin symbol": "PEPE", "Name": "Pepe", "Pool": "eth_0xa43fe16908251ee70ef74718545e46dd2cf076d1"},
        {"Coin symbol": "WIF", "Name": "dogwifhat", "Pool": "solana_EP2ib6dYdEeqD8MfE2ezHCxX3kP3K2eLK79Us1TezJy3"},
        {"Coin symbol": "SHIB", "Name": "Shiba Inu", "Pool": ""},
        {"Coin symbol": "XRP", "Name": "Ripple", "Pool": ""},
        {"Coin symbol": "BNB", "Name": "Binance Coin", "Pool": ""},
    ])

@st.cache_data(show_spinner=False)
def load_asset_list(csv_path: str) -> pd.DataFrame:
    """Load the asset list from a CSV file."""
    if not os.path.exists(csv_path):
        return get_default_assets()
    
    try:
        df = pd.read_csv(csv_path)
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        return df.fillna("")
    except Exception:
        return get_default_assets()

def build_token_options(df: pd.DataFrame) -> dict:
    """Map display names to symbols/configs."""
    options = {}
    if df.empty: return options
        
    for _, row in df.iterrows():
        # Flexible column finding
        cols = df.columns
        sym_col = next((c for c in cols if 'symbol' in c.lower()), None)
        name_col = next((c for c in cols if 'name' in c.lower()), None)
        pool_col = next((c for c in cols if 'pool' in c.lower() or 'id' in c.lower()), None)
        
        if not sym_col: continue

        coin = str(row.get(sym_col, "")).strip().upper()
        name = str(row.get(name_col, "")).strip() if name_col else ""
        pool = str(row.get(pool_col, "")).strip() if pool_col else ""
        
        display = f"{coin} - {name}" if name else coin
        
        # Store metadata needed for different APIs
        options[display] = {
            "symbol": coin,
            "pool": pool  # Needed for GeckoTerminal
        }
    return options

# -----------------------------------------------------------------------------
# 3. DATA FETCHING ENGINES
# -----------------------------------------------------------------------------

def fetch_binance(symbol_pair, endpoint_url, start_ms, end_ms):
    """Generic Binance Fetcher."""
    params = {
        'symbol': symbol_pair,
        'interval': '1d',
        'limit': 1000,
        'startTime': start_ms,
        'endTime': end_ms
    }
    try:
        response = requests.get(endpoint_url, params=params, timeout=10)
        
        if response.status_code == 451:
            st.error(f"🚫 Geo-Blocked: Binance Global is unavailable in your region. Switch Data Source to **Binance.US** in the sidebar.")
            return pd.DataFrame()
        elif response.status_code != 200:
            return pd.DataFrame()

        data = response.json()
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        return df[cols]
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

def fetch_geckoterminal(network_pool):
    """
    Fetch OHLCV from GeckoTerminal (DEX Data).
    Format: network_address (e.g., 'eth_0x123...')
    """
    if "_" not in network_pool:
        # Try to infer or fail gracefully
        return pd.DataFrame()
        
    network, address = network_pool.split('_', 1)
    url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{address}/ohlcv/day"
    
    try:
        response = requests.get(url, params={'limit': 1000}, timeout=10)
        if response.status_code != 200: return pd.DataFrame()
        
        data = response.json()
        ohlcv_list = data.get('data', {}).get('attributes', {}).get('ohlcv_list', [])
        if not ohlcv_list: return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp').sort_index()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def get_crypto_data(token_meta: dict, source: str, market_type: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Router for different data sources."""
    symbol = token_meta['symbol']
    
    # 1. BINANCE GLOBAL
    if source == "Binance (Global)":
        if market_type == "Perpetuals":
            url = "https://fapi.binance.com/fapi/v1/klines"
        else:
            url = "https://api.binance.com/api/v3/klines"
        return fetch_binance(f"{symbol}USDT", url, start_ms, end_ms)

    # 2. BINANCE US
    elif source == "Binance.US":
        if market_type == "Perpetuals":
            st.warning("⚠️ Binance.US does not support Perpetuals. Switching to Spot data proxy.")
        # Always Spot for US
        url = "https://api.binance.us/api/v3/klines"
        # Try USD first, then USDT
        df = fetch_binance(f"{symbol}USD", url, start_ms, end_ms)
        if df.empty:
            df = fetch_binance(f"{symbol}USDT", url, start_ms, end_ms)
        return df

    # 3. GECKOTERMINAL (DEX)
    elif source == "GeckoTerminal (DEX)":
        pool_info = token_meta.get('pool', '')
        if not pool_info or "_" not in pool_info:
            st.warning(f"No pool address configured for {symbol}. Cannot fetch DEX data.")
            return pd.DataFrame()
        return fetch_geckoterminal(pool_info)
        
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# 4. MATH ENGINE
# -----------------------------------------------------------------------------

def calculate_volatility(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    annual_factor = np.sqrt(365)
    
    for w in windows:
        df[f'hv_{w}'] = df['log_ret'].rolling(window=w).std() * annual_factor
            
    # RMS Metric: Sqrt of average variance of 7d and 14d
    if 7 in windows and 14 in windows:
        df['rms_vol'] = np.sqrt((df['hv_7']**2 + df['hv_14']**2) / 2)
    else:
        # Fallback to first window
        df['rms_vol'] = df[f'hv_{windows[0]}']
        
    return df.dropna()

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return (0.0, 0.0, 0.0, 0.0, 0.0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Cumulative normal distribution
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1) # PDF for Greeks
    
    if option_type == 'call':
        price = S * nd1 - K * np.exp(-r * T) * nd2
        delta = nd1
        theta = (- (S * n_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * nd2) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = nd1 - 1
        theta = (- (S * n_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
    gamma = n_d1 / (S * sigma * np.sqrt(T))
    vega = S * n_d1 * np.sqrt(T) / 100 # Vega per 1% vol change
    
    return price, delta, gamma, theta, vega

# -----------------------------------------------------------------------------
# 5. UI LAYOUT
# -----------------------------------------------------------------------------

asset_path = os.path.join(os.path.dirname(__file__), 'asset list.csv')
asset_df = load_asset_list(asset_path)
token_options = build_token_options(asset_df)

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Data Source
    data_source = st.selectbox(
        "Data Source",
        ["Binance (Global)", "Binance.US", "GeckoTerminal (DEX)"],
        index=1,
        help="Use Binance.US if you are in the USA. Use GeckoTerminal for memecoins."
    )
    
    # 2. Market Type
    market_disabled = data_source != "Binance (Global)"
    market_type = st.radio(
        "Market Type", 
        ["Spot", "Perpetuals"], 
        index=0,
        disabled=market_disabled,
        help="Perpetuals only available on Binance Global."
    )
    
    st.divider()
    
    # 3. Time Range (New!)
    st.subheader("Data Range")
    # Default to 1 year back, but allow user to change
    default_start = datetime.now() - timedelta(days=365)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", datetime.now())
    
    if start_date > end_date:
        st.error("Start Date must be before End Date")
    
    st.divider()

    # 4. Asset Selector
    if token_options:
        default_sel = [list(token_options.keys())[0]]
        selected_display = st.multiselect("Assets", list(token_options.keys()), default=default_sel)
    else:
        st.error("No assets found.")
        selected_display = []
        
    # 5. Params
    st.subheader("Volatility Settings")
    windows_str = st.text_input("Days (comma sep)", "7,14,30")
    try:
        vol_windows = [int(x) for x in windows_str.split(',')]
    except:
        vol_windows = [7, 14, 30]
        
    st.subheader("Pricer Inputs")
    expiry_days = st.number_input("Days to Expiry", 1, 365, 30)
    strike_pct = st.slider("Strike (%)", 0.8, 1.2, 1.0, 0.01)

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------------------------------------

if selected_display:
    # Time setup using User Inputs
    start_dt_combined = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt_combined = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
    
    start_ms = int(start_dt_combined.timestamp() * 1000)
    end_ms = int(end_dt_combined.timestamp() * 1000)
    
    # List to collect all dataframes for bulk export
    all_data_collection = []

    for display_name in selected_display:
        meta = token_options[display_name]
        
        st.markdown(f"### {display_name}")
        
        # 1. Get Data
        df = get_crypto_data(meta, data_source, market_type, start_ms, end_ms)
        
        if df.empty:
            if data_source == "GeckoTerminal (DEX)":
                 st.info(f"No DEX data found. Configure Pool Address in CSV.")
            else:
                 st.info(f"No data. Try switching Data Source.")
            continue
            
        # 2. Calc Stats
        df_calc = calculate_volatility(df, vol_windows)
        latest = df_calc.iloc[-1]
        
        # Add to collection for bulk export
        df_export_prep = df_calc.copy()
        df_export_prep['Symbol'] = meta['symbol']
        all_data_collection.append(df_export_prep)
        
        # 3. Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${latest['close']:,.4f}")
        c2.metric("RMS Vol (7,14)", f"{latest['rms_vol']*100:.1f}%")
        c3.metric(f"{vol_windows[0]}d Vol", f"{latest.get(f'hv_{vol_windows[0]}', 0)*100:.1f}%")
        
        # 4. Chart & Table
        col_viz, col_tbl = st.columns([2, 1])
        
        with col_viz:
            fig = go.Figure()
            # Plot Volatility Curves
            for w in vol_windows:
                if f'hv_{w}' in df_calc:
                    fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc[f'hv_{w}'], name=f'{w}d HV'))
            
            # Plot RMS
            fig.add_trace(go.Scatter(
                x=df_calc.index, y=df_calc['rms_vol'], 
                name='RMS (Target)', line=dict(color='white', width=3, dash='dot')
            ))
            
            fig.update_layout(
                title="Volatility Term Structure",
                yaxis_title="Annualized Volatility",
                yaxis_tickformat='.0%',
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_tbl:
            st.subheader("Inventory Pricer")
            vol_input = latest['rms_vol']
            strike_price = latest['close'] * strike_pct
            
            cp, cd, cg, ct, cv = black_scholes(latest['close'], strike_price, expiry_days/365, 0.04, vol_input, 'call')
            pp, pd_val, pg, pt, pv = black_scholes(latest['close'], strike_price, expiry_days/365, 0.04, vol_input, 'put')
            
            greeks = pd.DataFrame({
                "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega"],
                "Call": [cp, cd, cg, ct, cv],
                "Put": [pp, pd_val, pg, pt, pv]
            }).set_index("Metric")
            
            st.table(greeks.style.format("{:.4f}"))
            
            # Individual Export
            csv = df_calc.to_csv().encode('utf-8')
            st.download_button(
                f"📥 Download {meta['symbol']} CSV",
                csv,
                f"{meta['symbol']}_vol_data.csv",
                "text/csv",
                key=f"dl_{meta['symbol']}"
            )

    # --- BULK EXPORT SECTION ---
    if all_data_collection:
        st.divider()
        st.subheader("📚 Bulk Export")
        st.write("Download historical data for ALL selected assets in one file.")
        
        # Concatenate all dataframes
        combined_df = pd.concat(all_data_collection)
        combined_csv = combined_df.to_csv().encode('utf-8')
        
        st.download_button(
            "📥 Download All Combined Data (CSV)",
            combined_csv,
            "volatility_dashboard_master.csv",
            "text/csv",
            key="dl_master"
        )
