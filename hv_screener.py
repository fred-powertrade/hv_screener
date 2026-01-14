"""
Historical Volatility Screener (Spot & Futures)
===============================================

This updated Streamlit app allows users to screen historical volatility (HV) for a
curated list of crypto assets. It now supports switching between **Binance Spot**
and **Binance USDT-M Futures (Perps)** data.

New Features
------------
* **Market Type Selector**: Toggle between 'Spot' and 'Futures' (Perps) in the sidebar.
  The app dynamically switches API endpoints (Binance Spot vs Binance FAPI).
* **Data Export**: A download button is available below the data table to export
  the calculated volatility metrics to a CSV file.

Standard Features
-----------------
* **Asset filtering** via `asset list.csv`.
* **Custom HV windows** (e.g., 7, 14, 30 days).
* **Normalised RMS metrics** (RMS of 7&14, 2&3).
* **Tenor comparison chart**.
* **Interactive Option Pricer**.

Usage
-----
Ensure `asset list.csv` is in the same directory.
Run with: `streamlit run app.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="Historical Volatility Screener",
    page_icon="📉",
)

st.title("📉 Historical Volatility Screener")
st.markdown(
    """
    **Market Maker Volatility Engine**
    
    Select your asset and market type (Spot or Perps) to analyze realized volatility regimes. 
    Use the RMS metrics to price inventory risk and the export function to download data for offline modeling.
    """
)

# -----------------------------------------------------------------------------
# 2. ASSET LIST LOADING
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_asset_list(csv_path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(csv_path):
            # Fallback for demo purposes if file is missing
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        return df.fillna("")
    except Exception as exc:
        st.error(f"Failed to load asset list: {exc}")
        return pd.DataFrame()

def build_token_options(df: pd.DataFrame) -> dict:
    options = {}
    for _, row in df.iterrows():
        coin = str(row.get("Coin symbol", "")).strip().upper()
        if not coin:
            continue
        common = str(row.get("Common Name", "")).strip()
        display = f"{coin} - {common}" if common else coin
        if display not in options:
            options[display] = f"{coin}USDT"
    return options

# -----------------------------------------------------------------------------
# 3. DATA FETCHING UTILITIES (UPDATED FOR SPOT/PERPS)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_crypto_data(symbol: str, market_type: str, interval: str = "1d", 
                   start_time: int | None = None, end_time: int | None = None, 
                   limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance (Spot or Futures).
    """
    # Switch URL based on market type
    if market_type == 'spot':
        base_url = "[https://api.binance.com/api/v3/klines](https://api.binance.com/api/v3/klines)"
    else: # perps
        base_url = "[https://fapi.binance.com/fapi/v1/klines](https://fapi.binance.com/fapi/v1/klines)"

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }
    if start_time is not None:
        params = int(start_time)
    if end_time is not None:
        params = int(end_time)
        
    try:
        response = requests.get(base_url, params=params)
        if response.status_code!= 200:
            return pd.DataFrame()
            
        data = response.json()
        if not data or not isinstance(data, list):
            return pd.DataFrame()
            
        # Structure is identical for Spot and Futures
        df = pd.DataFrame(
            data,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ],
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def get_current_price(symbol: str, market_type: str) -> float | None:
    """Get the latest price from Binance (Spot or Futures)."""
    try:
        if market_type == 'spot':
            url = f"[https://api.binance.com/api/v3/ticker/price?symbol=](https://api.binance.com/api/v3/ticker/price?symbol=){symbol}"
        else:
            url = f"[https://fapi.binance.com/fapi/v1/ticker/price?symbol=](https://fapi.binance.com/fapi/v1/ticker/price?symbol=){symbol}"
            
        response = requests.get(url)
        if response.status_code == 200:
            return float(response.json()['price'])
        return None
    except Exception:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def get_implied_vol(currency: str = 'BTC') -> float | None:
    """Fetch DVOL from Deribit (BTC/ETH only)."""
    url = "[https://www.deribit.com/api/v2/public/get_volatility_index_data](https://www.deribit.com/api/v2/public/get_volatility_index_data)"
    now_ms = int(time.time() * 1000)
    params = {
        "currency": currency,
        "start_timestamp": now_ms - 3600 * 1000,
        "end_timestamp": now_ms,
        "resolution": "3600",
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'result' in data and 'data' in data['result'] and data['result']['data']:
            return data['result']['data'][-1][1]
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# 4. MATHEMATICAL ENGINE
# -----------------------------------------------------------------------------

def calculate_hv_metrics(df: pd.DataFrame, vol_windows: list[int]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) < max(vol_windows) + 1:
        return pd.DataFrame()
        
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    annual_factor = np.sqrt(365)
    
    for w in vol_windows:
        df[f'hv_{w}'] = df['log_ret'].rolling(window=w).std() * annual_factor
        
    # Normalised RMS Calculations
    if 2 in vol_windows and 3 in vol_windows:
        df['normalized_23'] = np.sqrt((df['hv_2']**2 + df['hv_3']**2) / 2)
    if 7 in vol_windows and 14 in vol_windows:
        df['normalized_714'] = np.sqrt((df['hv_7']**2 + df['hv_14']**2) / 2)
        
    # Representative RMS for UI
    df['rms_vol'] = df['normalized_714'] if 'normalized_714' in df.columns else np.nan
    return df.dropna()

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return (price, delta, gamma, theta, vega)

# -----------------------------------------------------------------------------
# 5. USER INTERFACE
# -----------------------------------------------------------------------------

# Load assets
asset_csv_path = 'asset list.csv' # Assumes file is in same directory
asset_df = load_asset_list(asset_csv_path)
token_options = build_token_options(asset_df)

with st.sidebar:
    st.header("🔧 Settings")
    
    # 1. Market Type Selector (Spot vs Perps)
    market_mode = st.radio(
        "Market Source", 
       ,
        index=0,
        help="Switch between Binance Spot and USDT-M Perpetual Futures."
    )
    market_type = 'spot' if market_mode == 'Spot' else 'perps'
    
    # 2. Token Selection
    default_tokens =
    for name in:
        for k in token_options.keys():
            if k.startswith(name):
                default_tokens.append(k)
                break
                
    selected_display = st.multiselect(
        "Select Tokens (max 5)",
        options=list(token_options.keys()),
        default=default_tokens,
        max_selections=5
    )
    
    # 3. Date & Windows
    st.divider()
    today = datetime.now().date()
    default_start = today - timedelta(days=180)
    start_date = st.date_input("Start Date", value=default_start, max_value=today)
    end_date = st.date_input("End Date", value=today, min_value=start_date, max_value=today)
    
    windows_input = st.text_input("HV Windows (days)", value="2,3,7,14,30,60,90")
    vol_windows = sorted(list(set([int(x.strip()) for x in windows_input.split(',') if x.strip().isdigit()])))
    
    # 4. Tenor Comparison
    st.divider()
    st.subheader("Comparison Settings")
    tenor1 = st.selectbox("Short Tenor", vol_windows, index=0 if vol_windows else 0)
    tenor2 = st.selectbox("Long Tenor", vol_windows, index=1 if len(vol_windows)>1 else 0)
    
    # 5. Pricer Settings
    st.divider()
    st.subheader("Pricer Inputs")
    days_expiry = st.number_input("Days to Expiry", 1, 365, 30)
    strike_pct = st.slider("Strike Range (%)", 0.5, 1.5, (0.8, 1.2), 0.05)
    risk_free = st.number_input("Risk Free Rate", value=0.05)

# -----------------------------------------------------------------------------
# 6. MAIN LOGIC
# -----------------------------------------------------------------------------

if selected_display and vol_windows:
    start_dt = datetime(start_date.year, start_date.month, start_date.day, 8, 0)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, 8, 0)
    start_ms = int(time.mktime(start_dt.timetuple()) * 1000)
    end_ms = int(time.mktime(end_dt.timetuple()) * 1000)
    
    for display_name in selected_display:
        symbol = token_options.get(display_name)
        if not symbol: continue
        
        st.markdown(f"---")
        st.markdown(f"## {display_name} ({market_mode})")
        
        # Fetch Data
        raw_df = get_crypto_data(symbol, market_type, start_time=start_ms, end_time=end_ms)
        
        if raw_df.empty:
            st.warning(f"No data found for {symbol} on Binance {market_mode}. It might not be listed.")
            continue
            
        processed_df = calculate_hv_metrics(raw_df, vol_windows)
        
        if processed_df.empty:
            st.warning("Not enough data to calculate volatility.")
            continue
            
        latest = processed_df.iloc[-1]
        current_price = get_current_price(symbol, market_type) or latest['close']
        
        # Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Price", f"${current_price:,.4f}")
        m2.metric("RMS (7,14)", f"{latest.get('normalized_714', 0):.2%}")
        m3.metric("RMS (2,3)", f"{latest.get('normalized_23', 0):.2%}")
        m4.metric(f"{tenor1}d Vol", f"{latest.get(f'hv_{tenor1}', 0):.2%}")
        m5.metric(f"{tenor2}d Vol", f"{latest.get(f'hv_{tenor2}', 0):.2%}")
        
        # Charts & Table Layout
        c_chart, c_table = st.columns([2, 3])
        
        with c_chart:
            # Main Volatility Chart
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Plot user selected windows
            for idx, w in enumerate(vol_windows):
                if w in : # Show standard ones by default
                    fig.add_trace(go.Scatter(
                        x=processed_df.index, y=processed_df[f'hv_{w}'],
                        name=f'{w}d HV', line=dict(width=1.5)
                    ))
            
            # Highlight RMS
            if 'normalized_714' in processed_df.columns:
                fig.add_trace(go.Scatter(
                    x=processed_df.index, y=processed_df['normalized_714'],
                    name='RMS (7,14)', line=dict(color='white', width=3, dash='dot')
                ))
                
            fig.update_layout(
                title=f"Volatility Structure ({market_mode})",
                yaxis=dict(tickformat='.0%', title="Annualized Volatility"),
                height=450,
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tenor Comparison Chart
            if tenor1!= tenor2:
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(
                    x=processed_df.index, y=processed_df[f'hv_{tenor1}'], 
                    name=f'{tenor1}d', line=dict(color='cyan')
                ))
                fig_spread.add_trace(go.Scatter(
                    x=processed_df.index, y=processed_df[f'hv_{tenor2}'], 
                    name=f'{tenor2}d', line=dict(color='magenta')
                ))
                # Spread
                spread = processed_df[f'hv_{tenor1}'] - processed_df[f'hv_{tenor2}']
                fig_spread.add_trace(go.Scatter(
                    x=processed_df.index, y=spread, 
                    name='Spread', line=dict(color='yellow', width=1), yaxis='y2'
                ))
                
                fig_spread.update_layout(
                    title=f"Term Structure Spread ({tenor1}d vs {tenor2}d)",
                    height=300,
                    yaxis=dict(tickformat='.0%'),
                    yaxis2=dict(title="Spread", overlaying='y', side='right', tickformat='.2%'),
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=True
                )
                st.plotly_chart(fig_spread, use_container_width=True)

        with c_table:
            st.subheader("Historical Data")
            
            # Prepare Table
            display_cols = ['close']
            if 'normalized_714' in processed_df.columns: display_cols.append('normalized_714')
            display_cols += [f'hv_{w}' for w in vol_windows[:4]] # Show first 4 windows to save space
            
            table_df = processed_df[display_cols].copy().sort_index(ascending=False)
            
            # Format for display
            fmt_df = table_df.copy()
            for col in fmt_df.columns:
                if 'hv' in col or 'normalized' in col:
                    fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.2%}")
                else:
                    fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(fmt_df, height=600, use_container_width=True)
            
            # Export Button
            csv_data = table_df.to_csv().encode('utf-8')
            filename = f"{symbol}_{market_mode}_Volatility.csv"
            
            st.download_button(
                label="📥 Export Data to CSV",
                data=csv_data,
                file_name=filename,
                mime='text/csv',
                key=f"dl_{symbol}"
            )

        # 7. Options Pricer Integration
        with st.expander("🛠️ Market Maker Option Pricer", expanded=False):
            st.write("Pricing theoretical options based on current RMS Volatility")
            
            vol_input = latest.get('normalized_714', 0.5)
            t_years = days_expiry / 365.0
            
            strikes = np.linspace(strike_pct, strike_pct[3], 5)
            pricer_rows =
            
            for k_pct in strikes:
                K = current_price * k_pct
                # Call
                c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(current_price, K, t_years, risk_free, vol_input, 'call')
                # Put
                p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(current_price, K, t_years, risk_free, vol_input, 'put')
                
                pricer_rows.append({
                    "Strike": f"${K:,.2f} ({k_pct:.0%})",
                    "Call Price": f"{c_price:.2f}",
                    "Call Delta": f"{c_delta:.2f}",
                    "Put Price": f"{p_price:.2f}",
                    "Put Delta": f"{p_delta:.2f}",
                    "Vega": f"{c_vega:.2f}",
                    "Theta": f"{c_theta:.2f}"
                })
            
            st.table(pd.DataFrame(pricer_rows))

else:
    st.info("👈 Select tokens and configure windows in the sidebar to start.")
