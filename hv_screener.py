"""
Historical Volatility Screener
===============================

This Streamlit app allows users to screen historical volatility (HV) for a
curated list of crypto assets. It leverages Binance's public API (Spot & Futures)
to fetch daily candle data and computes annualised historical volatility.
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
            {"Coin symbol": "XRP", "Common Name": "Ripple"},
            {"Coin symbol": "BNB", "Common Name": "Binance Coin"},
        ])
    
    try:
        df = pd.read_csv(csv_path)
        # Clean column names just in case
        df.columns = [c.strip() for c in df.columns]
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
        # Handle cases where column names might slightly vary
        coin_col = next((c for c in df.columns if 'symbol' in c.lower()), None)
        name_col = next((c for c in df.columns if 'name' in c.lower()), None)
        
        if not coin_col: continue

        coin = str(row.get(coin_col, "")).strip().upper()
        if not coin:
            continue
            
        common = str(row.get(name_col, "")).strip() if name_col else ""
        display = f"{coin} - {common}" if common else coin
        # Store just the coin symbol (e.g., BTC)
        options[display] = coin 
    return options


# -----------------------------------------------------------------------------
# 3. DATA FETCHING UTILITIES
# -----------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_crypto_data(symbol: str, market_type: str, interval: str = "1d", 
                    start_time: int = None, end_time: int = None) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance (Spot or Futures).
    """
    # Select Endpoint based on Market Type
    if market_type == "Perpetuals":
        base_url = "https://fapi.binance.com/fapi/v1/klines"
    else:
        base_url = "https://api.binance.com/api/v3/klines"

    # Parameters
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1000 # Max limit for one call
    }
    
    # Only add time params if they are valid
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time
    
    try:
        # Timeout added to prevent hanging
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code != 200:
            st.error(f"API Error ({symbol}): {response.status_code} - {response.text}")
            return pd.DataFrame()
        
        data = response.json()
        
        # Check if empty list returned
        if not data:
            return pd.DataFrame()
            
        # Binance returns list of lists
        # [Open Time, Open, High, Low, Close, Volume, Close Time, ...]
        df = pd.DataFrame(
            data,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ],
        )
        # Convert timestamp (ms) to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()
        
        # Ensure numeric types
        cols_to_numeric = ['open', 'high', 'low', 'close', 'volume']
        df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Connection Error for {symbol}: {str(e)}")
        return pd.DataFrame()


def calculate_hv_metrics(df: pd.DataFrame, vol_windows: list[int]) -> pd.DataFrame:
    """Compute annualised historical volatility and RMS metrics."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    # Log Returns: ln(Pt / Pt-1)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Annualisation Factor: sqrt(365) for crypto
    annual_factor = np.sqrt(365)
    
    for w in vol_windows:
        # Require at least w observations
        df[f'hv_{w}'] = df['log_ret'].rolling(window=w).std() * annual_factor
            
    # RMS Normalised Metrics (The "Secret Sauce")
    # Formula: sqrt( (hv7^2 + hv14^2) / 2 )
    if 7 in vol_windows and 14 in vol_windows:
        df['normalized_714'] = np.sqrt((df['hv_7']**2 + df['hv_14']**2) / 2)
        df['rms_vol'] = df['normalized_714']
    elif 'rms_vol' not in df.columns:
        # Fallback: if 7/14 not available, use the first window available as "vol"
        first_window = vol_windows[0] if vol_windows else 7
        if f'hv_{first_window}' in df.columns:
            df['rms_vol'] = df[f'hv_{first_window}']
        else:
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
    else: # put
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
        default_keys = [k for k in token_options.keys() if "BTC" in k]
        default_val = default_keys[:1] if default_keys else [list(token_options.keys())[0]]
            
        selected_display = st.multiselect(
            "Select Tokens",
            options=list(token_options.keys()),
            default=default_val
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
        strike_dist = st.slider("Strike Distance", 0.8, 1.2, 1.0, 0.01)
        risk_free = st.number_input("Risk Free Rate", 0.0, 1.0, 0.04, 0.01)


# -----------------------------------------------------------------------------
# 5. MAIN LOGIC
# -----------------------------------------------------------------------------

if selected_display and vol_windows:
    # --- Fix Timestamp Logic: Use UTC explicitly ---
    # Convert dates to datetime objects at 00:00:00 UTC
    t_start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    t_end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
    
    # Convert to milliseconds for Binance API
    start_ms = int(t_start.timestamp() * 1000)
    end_ms = int(t_end.timestamp() * 1000)

    for display_name in selected_display:
        base_symbol = token_options[display_name]
        # Construct pair symbol (Assume USDT)
        full_symbol = f"{base_symbol}USDT"
        
        st.markdown(f"## {display_name} ({market_type})")
        
        # Fetch Data
        with st.spinner(f"Fetching data for {full_symbol}..."):
            raw_df = get_crypto_data(full_symbol, market_type, start_time=start_ms, end_time=end_ms)
        
        if raw_df.empty:
            st.warning(f"⚠️ No data returned for **{full_symbol}**. It might not exist on the {market_type} market.")
            continue
            
        # Calculate Metrics
        df_vol = calculate_hv_metrics(raw_df, vol_windows)
        
        if df_vol.empty:
            st.warning("Not enough data points to calculate the requested volatility windows.")
            continue
            
        latest = df_vol.iloc[-1]
        
        # --- Top Metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Close Price", f"${latest['close']:.4f}")
        
        # Safe display of RMS
        rms_val = latest.get('rms_vol', np.nan)
        m2.metric("RMS Vol (7,14)", f"{rms_val*100:.1f}%" if not np.isnan(rms_val) else "N/A")
            
        # Safe display of specific windows
        w1 = vol_windows[0]
        val_w1 = latest.get(f'hv_{w1}', np.nan)
        m3.metric(f"{w1}d Vol", f"{val_w1*100:.1f}%" if not np.isnan(val_w1) else "N/A")
        
        if len(vol_windows) > 1:
            w2 = vol_windows[1]
            val_w2 = latest.get(f'hv_{w2}', np.nan)
            m4.metric(f"{w2}d Vol", f"{val_w2*100:.1f}%" if not np.isnan(val_w2) else "N/A")

        # --- Layout: Chart Left, Table/Export Right ---
        col_chart, col_data = st.columns([2, 1])
        
        with col_chart:
            fig = go.Figure()
            # Add HV traces
            colors = ['#00ff00', '#ff00ff', '#0000ff', '#ffa500', '#00ced1', '#ff4500']
            
            # Plot HV windows
            for i, w in enumerate(vol_windows):
                col_name = f'hv_{w}'
                if col_name in df_vol.columns:
                    fig.add_trace(go.Scatter(
                        x=df_vol.index, y=df_vol[col_name], 
                        name=f'{w}d HV',
                        line=dict(width=1.5, color=colors[i % len(colors)])
                    ))
            
            # Add RMS trace (White Dotted)
            if 'rms_vol' in df_vol.columns:
                fig.add_trace(go.Scatter(
                    x=df_vol.index, y=df_vol['rms_vol'],
                    name='RMS (7,14)',
                    line=dict(color='white', width=3, dash='dot')
                ))

            fig.update_layout(
                title=f"Volatility Term Structure ({full_symbol})",
                yaxis=dict(tickformat=".1%", title="Annualized Volatility"),
                height=450,
                legend=dict(orientation="h", y=1.1, x=0),
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
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
                label=f"📥 Download CSV ({full_symbol})",
                data=csv_data,
                file_name=f"{full_symbol}_{market_type}_volatility.csv",
                mime="text/csv",
                key=f"dl_{full_symbol}"
            )

        # --- Option Pricer (Expander) ---
        with st.expander(f"🛠️ Option Pricer for {full_symbol}", expanded=False):
            # Use RMS vol if available, else first window
            calc_vol = latest.get('rms_vol', latest.get(f'hv_{vol_windows[0]}', 0.5))
            
            current_strike = latest['close'] * strike_dist
            
            c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(
                latest['close'], current_strike, days_expiry/365, risk_free, calc_vol, 'call'
            )
            p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(
                latest['close'], current_strike, days_expiry/365, risk_free, calc_vol, 'put'
            )
            
            pricer_df = pd.DataFrame({
                "Metric": ["Price ($)", "Delta", "Gamma", "Theta (Daily $)", "Vega (1% move $)"],
                "Call Option": [c_price, c_delta, c_gamma, c_theta, c_vega],
                "Put Option": [p_price, p_delta, p_gamma, p_theta, p_vega]
            })
            
            # Formatting table
            st.table(pricer_df.set_index("Metric"))

else:
    st.info("Please select at least one token from the sidebar to view data.")
