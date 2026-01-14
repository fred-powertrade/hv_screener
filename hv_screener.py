"""
Historical Volatility Screener
===============================

This Streamlit app allows users to screen historical volatility (HV) for a
curated list of crypto assets. It leverages Binance's public API (Spot & Futures)
to fetch daily candle data and computes annualised historical volatility.

Key features:
- Toggle between Spot and Perpetual Futures data.
- Custom HV windows (e.g., 7, 14, 30 days).
- RMS Normalised metrics (7/14 day blending).
- Option Greeks pricer (Black-Scholes).
- Export data to CSV.
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
import io

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
    **Market Making Risk Dashboard**
    Select assets, choose your market type (Spot vs Perps), and analyze volatility regimes.
    """
)

# -----------------------------------------------------------------------------
# 2. ASSET LIST LOADING
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_asset_list(csv_path: str) -> pd.DataFrame:
    """Load the asset list from a CSV file."""
    if not os.path.exists(csv_path):
        # Fallback if file doesn't exist
        return pd.DataFrame([
            {"Coin symbol": "BTC", "Common Name": "Bitcoin"},
            {"Coin symbol": "ETH", "Common Name": "Ethereum"},
            {"Coin symbol": "SOL", "Common Name": "Solana"},
            {"Coin symbol": "DOGE", "Common Name": "Dogecoin"},
            {"Coin symbol": "PEPE", "Common Name": "Pepe"},
            {"Coin symbol": "WIF", "Common Name": "dogwifhat"},
        ])
    
    try:
        df = pd.read_csv(csv_path)
        return df.fillna("")
    except Exception as exc:
        st.error(f"Failed to load asset list: {exc}")
        return pd.DataFrame()


def build_token_options(df: pd.DataFrame) -> dict:
    """Construct a mapping from Display Name -> Symbol (base)."""
    options = {}
    if df.empty:
        return options
        
    for _, row in df.iterrows():
        coin = str(row.get("Coin symbol", "")).strip().upper()
        if not coin:
            continue
        common = str(row.get("Common Name", "")).strip()
        display = f"{coin} - {common}" if common else coin
        # Store just the coin symbol (e.g., BTC), we append USDT later based on market type
        options[display] = coin 
    return options


# -----------------------------------------------------------------------------
# 3. DATA FETCHING UTILITIES
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_crypto_data(symbol: str, market_type: str, interval: str = "1d", 
                    start_time: int | None = None, end_time: int | None = None, 
                    limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance (Spot or Futures).
    """
    # Select Endpoint based on Market Type
    if market_type == "Perpetuals":
        base_url = "https://fapi.binance.com/fapi/v1/klines"
        # Most perps are USDT margined, symbol usually same as spot (BTCUSDT)
        # Note: Some crazy alts might be 1000PEPEUSDT, but standard is Coin+USDT
        trading_pair = symbol 
    else:
        base_url = "https://api.binance.com/api/v3/klines"
        trading_pair = symbol

    params = {
        'symbol': trading_pair,
        'interval': interval,
        'limit': limit,
    }
    if start_time is not None:
        params['startTime'] = int(start_time)
    if end_time is not None:
        params['endTime'] = int(end_time)
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        if not data:
            return pd.DataFrame()
            
        # Binance returns list of lists
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


@st.cache_data(ttl=600, show_spinner=False)
def get_spot_price(symbol: str) -> float | None:
    """Get latest price (Spot API is generally reliable for reference price)."""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return float(response.json()['price'])
        return None
    except Exception:
        return None


def calculate_hv_metrics(df: pd.DataFrame, vol_windows: list[int]) -> pd.DataFrame:
    """Compute annualised historical volatility and RMS metrics."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    # Log Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Crypto trades 365 days a year
    annual_factor = np.sqrt(365)
    
    for w in vol_windows:
        if len(df) >= w:
            df[f'hv_{w}'] = df['log_ret'].rolling(window=w).std() * annual_factor
            
    # RMS Normalised Metrics (The "Secret Sauce")
    if 7 in vol_windows and 14 in vol_windows:
        df['normalized_714'] = np.sqrt((df['hv_7']**2 + df['hv_14']**2) / 2)
        # Use this as the main "RMS Vol" for the UI
        df['rms_vol'] = df['normalized_714']
    elif 'rms_vol' not in df.columns:
         # Fallback if 7/14 not selected
        df['rms_vol'] = np.nan
        
    return df.dropna()


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes Greeks."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = (term1 + term2) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return price, delta, gamma, theta, vega


# -----------------------------------------------------------------------------
# 4. SIDEBAR SETTINGS
# -----------------------------------------------------------------------------

# Load asset list
asset_csv_path = os.path.join(os.path.dirname(__file__), 'asset list.csv')
asset_df = load_asset_list(asset_csv_path)
token_options = build_token_options(asset_df)

with st.sidebar:
    st.header("🔧 Settings")
    
    # --- Market Selector (Spot vs Perps) ---
    market_type = st.radio("Market Data Source", ["Spot", "Perpetuals"], index=0)
    
    if not token_options:
        st.warning("No assets found in asset list.csv")
    else:
        # Filter keys for default selection
        default_tokens = [k for k in token_options.keys() if k.startswith("BTC") or k.startswith("ETH")]
        if not default_tokens and token_options:
            default_tokens = [list(token_options.keys())[0]]
            
        selected_display = st.multiselect(
            "Select Tokens",
            options=list(token_options.keys()),
            default=default_tokens[:2]
        )

        st.divider()
        st.subheader("Time Parameters")
        
        # Date inputs
        col_d1, col_d2 = st.columns(2)
        end_date = col_d2.date_input("End Date", datetime.now())
        start_date = col_d1.date_input("Start Date", end_date - timedelta(days=180))

        # Windows
        windows_input = st.text_input("HV Windows (days)", value="7,14,30,90")
        try:
            vol_windows = sorted(list(set([int(x.strip()) for x in windows_input.split(',') if x.strip().isdigit()])))
        except:
            vol_windows = [7, 14, 30]
            
        st.divider()
        st.subheader("Option Pricer Inputs")
        days_expiry = st.number_input("Days to Expiry", 1, 365, 30)
        strike_dist = st.slider("Strike Distance", 0.8, 1.2, 1.0, 0.05)
        risk_free = st.number_input("Risk Free Rate", 0.0, 1.0, 0.04, 0.01)


# -----------------------------------------------------------------------------
# 5. MAIN LOGIC
# -----------------------------------------------------------------------------

if selected_display and vol_windows:
    # Prepare timestamps
    start_dt = datetime(start_date.year, start_date.month, start_date.day, 8, 0, 0)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, 8, 0, 0)
    start_ms = int(time.mktime(start_dt.timetuple())) * 1000
    end_ms = int(time.mktime(end_dt.timetuple())) * 1000

    for display_name in selected_display:
        base_symbol = token_options[display_name]
        # Construct pair symbol (assume USDT for now)
        full_symbol = f"{base_symbol}USDT"
        
        st.markdown(f"## {display_name} ({market_type})")
        
        # Fetch Data
        raw_df = get_crypto_data(full_symbol, market_type, start_time=start_ms, end_time=end_ms)
        
        if raw_df.empty:
            st.error(f"No data found for {full_symbol} on {market_type}. Try switching market type or checking the symbol.")
            continue
            
        # Calculate Metrics
        df_vol = calculate_hv_metrics(raw_df, vol_windows)
        
        if df_vol.empty:
            st.warning("Not enough data to calculate volatility windows.")
            continue
            
        latest = df_vol.iloc[-1]
        
        # --- Top Metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Close Price", f"${latest['close']:.4f}")
        
        if 'rms_vol' in latest and not np.isnan(latest['rms_vol']):
            m2.metric("RMS Vol (7,14)", f"{latest['rms_vol']*100:.1f}%")
        else:
            m2.metric("RMS Vol", "N/A")
            
        if f'hv_{vol_windows[0]}' in latest:
            m3.metric(f"{vol_windows[0]}d Vol", f"{latest[f'hv_{vol_windows[0]}']*100:.1f}%")
        
        if len(vol_windows) > 1 and f'hv_{vol_windows[1]}' in latest:
            m4.metric(f"{vol_windows[1]}d Vol", f"{latest[f'hv_{vol_windows[1]}']*100:.1f}%")

        # --- Layout: Chart Left, Table/Export Right ---
        col_chart, col_data = st.columns([2, 1])
        
        with col_chart:
            fig = go.Figure()
            # Add HV traces
            colors = ['#00ff00', '#ff00ff', '#0000ff', '#ffa500']
            for i, w in enumerate(vol_windows):
                col_name = f'hv_{w}'
                if col_name in df_vol.columns:
                    fig.add_trace(go.Scatter(
                        x=df_vol.index, y=df_vol[col_name], 
                        name=f'{w}d HV',
                        line=dict(width=1.5)
                    ))
            
            # Add RMS trace if available
            if 'rms_vol' in df_vol.columns:
                fig.add_trace(go.Scatter(
                    x=df_vol.index, y=df_vol['rms_vol'],
                    name='RMS (7,14)',
                    line=dict(color='white', width=3, dash='dot')
                ))

            fig.update_layout(
                title=f"{full_symbol} Volatility Term Structure",
                yaxis=dict(tickformat=".0%", title="Annualized Volatility"),
                height=450,
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_data:
            st.subheader("Recent Data")
            
            # Prepare display table
            disp_cols = ['close'] + [c for c in df_vol.columns if 'hv_' in c or 'rms' in c]
            disp_df = df_vol[disp_cols].sort_index(ascending=False).head(15)
            
            # Formatting for display (Keep raw data for export)
            formatted_df = disp_df.copy()
            for c in formatted_df.columns:
                if c != 'close':
                    formatted_df[c] = formatted_df[c].apply(lambda x: f"{x*100:.1f}%")
                else:
                    formatted_df[c] = formatted_df[c].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(formatted_df, height=400, use_container_width=True)
            
            # --- EXPORT BUTTON ---
            # Export the full calculated dataset (not just the top 15 rows)
            csv_buffer = io.StringIO()
            df_vol.to_csv(csv_buffer)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label=f"📥 Download {full_symbol} Data (CSV)",
                data=csv_data,
                file_name=f"{full_symbol}_{market_type}_volatility.csv",
                mime="text/csv",
                key=f"dl_{full_symbol}"
            )

        # --- Option Pricer (Expander) ---
        with st.expander(f"🛠️ Option Pricer for {full_symbol}", expanded=False):
            # Use RMS vol if available, else first window
            calc_vol = latest['rms_vol'] if 'rms_vol' in latest and not np.isnan(latest['rms_vol']) else latest[f'hv_{vol_windows[0]}']
            
            c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(
                latest['close'], latest['close']*strike_dist, days_expiry/365, risk_free, calc_vol, 'call'
            )
            p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(
                latest['close'], latest['close']*strike_dist, days_expiry/365, risk_free, calc_vol, 'put'
            )
            
            pricer_df = pd.DataFrame({
                "Metric": ["Price", "Delta", "Gamma", "Theta", "Vega"],
                "Call": [c_price, c_delta, c_gamma, c_theta, c_vega],
                "Put": [p_price, p_delta, p_gamma, p_theta, p_vega]
            })
            st.table(pricer_df.set_index("Metric"))

else:
    st.info("Please select at least one token to view data.")
