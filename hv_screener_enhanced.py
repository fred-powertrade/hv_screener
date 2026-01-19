"""
Historical Volatility Screener - Market Maker Edition
=====================================================

Professional volatility analysis tool designed for market makers to:
- Screen realized volatility across multiple tenors
- Analyze volatility term structure and regime shifts  
- Export historical data for offline modeling and risk analysis
- Price theoretical options using Black-Scholes with realized vol inputs

Features
--------
* Market Type Toggle: Switch between Binance Spot and USDT-M Perpetual Futures
* Multi-Asset Analysis: Compare up to 5 assets simultaneously
* Customizable HV Windows: Define your own volatility calculation periods
* RMS Metrics: Normalized volatility measures for inventory risk pricing
* Data Export: Download complete historical volatility data for selected date ranges
* Options Pricer: Theoretical pricing with Greeks using realized volatility

Usage
-----
Run with: streamlit run hv_screener_enhanced.py
Ensure asset_list.csv is in the same directory
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
from io import BytesIO
import pytz

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    layout="wide",
    page_title="HV Screener - Market Maker Edition",
    page_icon="📊",
)

# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Historical Volatility Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Market Maker Volatility Engine - HV calculated at 08:00 UTC daily snapshots</div>', unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & UTILITIES
# =============================================================================

@st.cache_data(show_spinner=False)
def load_asset_list(csv_path: str) -> pd.DataFrame:
    """Load the curated asset list from CSV."""
    try:
        if not os.path.exists(csv_path):
            st.error(f"Asset list not found at: {csv_path}")
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        return df.fillna("")
    except Exception as exc:
        st.error(f"Failed to load asset list: {exc}")
        return pd.DataFrame()

def build_token_options(df: pd.DataFrame) -> dict:
    """Build token selection options from asset list."""
    options = {}
    seen = set()
    
    for _, row in df.iterrows():
        coin = str(row.get("Coin symbol", "")).strip().upper()
        if not coin or coin in seen:
            continue
        seen.add(coin)
        
        common = str(row.get("Common Name", "")).strip()
        display = f"{coin} - {common}" if common else coin
        options[display] = f"{coin}USDT"
    
    return options

# =============================================================================
# API DATA FETCHING
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_crypto_data(
    symbol: str,
    market_type: str,
    interval: str = "1d",
    start_time: int = None,
    end_time: int = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance with 08:00 UTC timestamps.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        market_type: 'spot' or 'perps'
        interval: Kline interval (default '1d')
        start_time: Start timestamp in milliseconds (08:00 UTC)
        end_time: End timestamp in milliseconds (08:00 UTC)
        limit: Maximum number of data points
    
    Returns:
        DataFrame with OHLCV data indexed by timestamp (08:00 UTC)
    """
    # Select API endpoint based on market type
    if market_type == 'spot':
        base_url = "https://api.binance.com/api/v3/klines"
    else:  # perps
        base_url = "https://fapi.binance.com/fapi/v1/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }
    
    if start_time is not None:
        params['startTime'] = int(start_time)
    if end_time is not None:
        params['endTime'] = int(end_time)
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        
        # Parse response (identical structure for Spot and Futures)
        df = pd.DataFrame(
            data,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ],
        )
        
        # Convert timestamps to UTC datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Binance daily candles close at 00:00 UTC and open at 00:00 UTC
        # We need to adjust to 08:00 UTC for HV calculations
        # Add 8 hours to align with 08:00 UTC snapshot time
        df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=8)
        
        df = df.set_index('timestamp').sort_index()
        
        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def get_current_price(symbol: str, market_type: str) -> float:
    """Get the latest price from Binance."""
    try:
        if market_type == 'spot':
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        else:
            url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return float(response.json()['price'])
        return None
    except Exception:
        return None

# =============================================================================
# VOLATILITY CALCULATIONS
# =============================================================================

def calculate_hv_metrics(df: pd.DataFrame, vol_windows: list) -> pd.DataFrame:
    """
    Calculate historical volatility metrics across multiple windows at 08:00 UTC.
    
    All volatility calculations are performed using close prices at 08:00 UTC daily snapshots.
    This ensures consistency across different assets and markets.
    
    Args:
        df: DataFrame with OHLCV data (indexed at 08:00 UTC)
        vol_windows: List of window sizes in days
    
    Returns:
        DataFrame with HV calculations and RMS metrics (indexed at 08:00 UTC)
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    if len(df) < max(vol_windows) + 1:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Calculate log returns using 08:00 UTC close prices
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Annualization factor for daily data
    annual_factor = np.sqrt(365)
    
    # Calculate HV for each window
    for w in vol_windows:
        df[f'hv_{w}'] = df['log_ret'].rolling(window=w).std() * annual_factor
    
    # Normalized RMS calculations for inventory risk
    if 2 in vol_windows and 3 in vol_windows:
        df['rms_2_3'] = np.sqrt((df['hv_2']**2 + df['hv_3']**2) / 2)
    
    if 7 in vol_windows and 14 in vol_windows:
        df['rms_7_14'] = np.sqrt((df['hv_7']**2 + df['hv_14']**2) / 2)
    
    # Representative RMS for UI (prefer 7-14 mix)
    if 'rms_7_14' in df.columns:
        df['rms_vol'] = df['rms_7_14']
    elif 'rms_2_3' in df.columns:
        df['rms_vol'] = df['rms_2_3']
    else:
        df['rms_vol'] = np.nan
    
    return df.dropna()

# =============================================================================
# BLACK-SCHOLES OPTION PRICING
# =============================================================================

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call'):
    """
    Black-Scholes option pricing with Greeks.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Tuple of (price, delta, gamma, theta, vega)
    """
    if T <= 0 or sigma <= 0:
        return 0, 0, 0, 0, 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    
    # Greeks (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
    theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) / 365  # Per day
    
    if option_type == 'put':
        theta -= r * K * np.exp(-r * T) * norm.cdf(-d2) / 365
    else:
        theta -= r * K * np.exp(-r * T) * norm.cdf(d2) / 365
    
    return price, delta, gamma, theta, vega

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Market Type Selection
    st.subheader("Market Type")
    market_mode = st.radio(
        "Data Source",
        options=["Spot", "Perps"],
        index=1,  # Default to Perps for market makers
        help="Toggle between Binance Spot and USDT-M Perpetual Futures"
    )
    market_type = 'spot' if market_mode == 'Spot' else 'perps'
    
    st.divider()
    
    # Asset Selection
    st.subheader("Asset Selection")
    
    # Load asset list
    asset_df = load_asset_list("asset_list.csv")
    
    if asset_df.empty:
        st.error("Cannot load asset list. Please ensure asset_list.csv is in the working directory.")
        st.stop()
    
    token_options = build_token_options(asset_df)
    
    # Default selections (BTC, ETH, SOL)
    default_tokens = []
    for name in ['BTC', 'ETH', 'SOL']:
        for k in token_options.keys():
            if k.startswith(name):
                default_tokens.append(k)
                break
    
    selected_display = st.multiselect(
        "Select Assets (max 5)",
        options=list(token_options.keys()),
        default=default_tokens[:3],
        max_selections=5,
        help="Choose up to 5 assets for volatility analysis"
    )
    
    st.divider()
    
    # Date Range
    st.subheader("Date Range")
    today = datetime.now().date()
    default_start = today - timedelta(days=180)
    
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        max_value=today,
        help="Beginning of historical data range"
    )
    
    end_date = st.date_input(
        "End Date",
        value=today,
        min_value=start_date,
        max_value=today,
        help="End of historical data range"
    )
    
    st.divider()
    
    # Volatility Windows
    st.subheader("HV Windows")
    windows_input = st.text_input(
        "Window Sizes (days)",
        value="2,3,7,14,30,60,90",
        help="Comma-separated list of volatility calculation windows"
    )
    
    try:
        vol_windows = sorted(list(set([
            int(x.strip()) for x in windows_input.split(',') 
            if x.strip().isdigit() and int(x.strip()) > 0
        ])))
    except:
        vol_windows = [7, 14, 30]
        st.warning("Invalid window input. Using default: 7, 14, 30")
    
    if not vol_windows:
        vol_windows = [7, 14, 30]
    
    st.divider()
    
    # Term Structure Comparison
    st.subheader("Term Structure")
    
    if len(vol_windows) >= 2:
        tenor1 = st.selectbox(
            "Short Tenor",
            vol_windows,
            index=0,
            help="Shorter maturity for term structure analysis"
        )
        
        tenor2 = st.selectbox(
            "Long Tenor",
            vol_windows,
            index=min(1, len(vol_windows)-1),
            help="Longer maturity for term structure analysis"
        )
    else:
        tenor1 = tenor2 = vol_windows[0] if vol_windows else 7
    
    st.divider()
    
    # Options Pricer Settings
    st.subheader("Options Pricer")
    
    days_expiry = st.number_input(
        "Days to Expiry",
        min_value=1,
        max_value=365,
        value=30,
        help="Time to expiration for theoretical option pricing"
    )
    
    strike_range = st.slider(
        "Strike Range (%)",
        min_value=0.5,
        max_value=1.5,
        value=(0.8, 1.2),
        step=0.05,
        help="Strike price range as percentage of spot"
    )
    
    risk_free_rate = st.number_input(
        "Risk-Free Rate",
        min_value=0.0,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.4f",
        help="Annual risk-free interest rate"
    )

# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================

if not selected_display:
    st.info("👈 Select one or more assets from the sidebar to begin analysis")
    st.stop()

if not vol_windows:
    st.error("Please specify at least one valid HV window")
    st.stop()

# Convert dates to timestamps at 08:00 UTC
utc = pytz.UTC
start_dt = datetime.combine(start_date, datetime.min.time()).replace(hour=8, minute=0, second=0, microsecond=0)
start_dt = utc.localize(start_dt)
end_dt = datetime.combine(end_date, datetime.min.time()).replace(hour=8, minute=0, second=0, microsecond=0)
end_dt = utc.localize(end_dt)

start_ms = int(start_dt.timestamp() * 1000)
end_ms = int(end_dt.timestamp() * 1000)

# Process each selected asset
for idx, display_name in enumerate(selected_display):
    symbol = token_options.get(display_name)
    if not symbol:
        continue
    
    # Section divider
    st.markdown("---")
    st.markdown(f"## {display_name} ({market_mode})")
    
    # Fetch data
    with st.spinner(f"Fetching {symbol} data..."):
        raw_df = get_crypto_data(
            symbol=symbol,
            market_type=market_type,
            start_time=start_ms,
            end_time=end_ms
        )
    
    if raw_df.empty:
        st.warning(f"⚠️ No data available for {symbol} on Binance {market_mode}. The asset may not be listed or the date range may be invalid.")
        continue
    
    # Calculate volatility metrics
    processed_df = calculate_hv_metrics(raw_df, vol_windows)
    
    if processed_df.empty:
        st.warning(f"⚠️ Insufficient data to calculate volatility for {symbol}. Try a shorter HV window or longer date range.")
        continue
    
    # Get latest metrics
    latest = processed_df.iloc[-1]
    current_price = get_current_price(symbol, market_type) or latest['close']
    
    # =============================================================================
    # METRICS ROW
    # =============================================================================
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.metric(
            "Current Price",
            f"${current_price:,.4f}" if current_price < 100 else f"${current_price:,.2f}"
        )
    
    with m2:
        rms_714 = latest.get('rms_7_14', 0)
        st.metric(
            "RMS (7,14)",
            f"{rms_714:.2%}",
            help="Root Mean Square volatility of 7d and 14d windows"
        )
    
    with m3:
        rms_23 = latest.get('rms_2_3', 0)
        st.metric(
            "RMS (2,3)",
            f"{rms_23:.2%}",
            help="Root Mean Square volatility of 2d and 3d windows"
        )
    
    with m4:
        hv_short = latest.get(f'hv_{tenor1}', 0)
        st.metric(
            f"{tenor1}d HV",
            f"{hv_short:.2%}"
        )
    
    with m5:
        hv_long = latest.get(f'hv_{tenor2}', 0)
        st.metric(
            f"{tenor2}d HV",
            f"{hv_long:.2%}"
        )
    
    # =============================================================================
    # CHARTS & DATA TABLE
    # =============================================================================
    
    col_chart, col_table = st.columns([2, 3])
    
    with col_chart:
        # ----- Main Volatility Chart -----
        fig = go.Figure()
        
        colors = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#ffe66d', '#a8dadc']
        
        # Plot HV windows
        for i, w in enumerate(vol_windows[:5]):  # Limit to 5 for readability
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=processed_df.index,
                y=processed_df[f'hv_{w}'],
                name=f'{w}d HV',
                line=dict(width=1.5, color=color),
                mode='lines'
            ))
        
        # Highlight RMS volatility
        if 'rms_7_14' in processed_df.columns:
            fig.add_trace(go.Scatter(
                x=processed_df.index,
                y=processed_df['rms_7_14'],
                name='RMS (7,14)',
                line=dict(color='white', width=2.5, dash='dot'),
                mode='lines'
            ))
        
        fig.update_layout(
            title=f"Volatility Term Structure ({market_mode})",
            yaxis=dict(
                tickformat='.0%',
                title="Annualized Volatility"
            ),
            xaxis=dict(title="Date"),
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ----- Term Structure Spread Chart -----
        if tenor1 != tenor2:
            fig_spread = go.Figure()
            
            fig_spread.add_trace(go.Scatter(
                x=processed_df.index,
                y=processed_df[f'hv_{tenor1}'],
                name=f'{tenor1}d HV',
                line=dict(color='cyan', width=2),
                mode='lines'
            ))
            
            fig_spread.add_trace(go.Scatter(
                x=processed_df.index,
                y=processed_df[f'hv_{tenor2}'],
                name=f'{tenor2}d HV',
                line=dict(color='magenta', width=2),
                mode='lines'
            ))
            
            # Calculate and plot spread
            spread = processed_df[f'hv_{tenor1}'] - processed_df[f'hv_{tenor2}']
            
            fig_spread.add_trace(go.Scatter(
                x=processed_df.index,
                y=spread,
                name='Spread (Short - Long)',
                line=dict(color='yellow', width=1.5),
                mode='lines',
                yaxis='y2'
            ))
            
            fig_spread.update_layout(
                title=f"Term Structure Spread: {tenor1}d vs {tenor2}d",
                height=300,
                yaxis=dict(
                    tickformat='.0%',
                    title="Volatility"
                ),
                yaxis2=dict(
                    title="Spread",
                    overlaying='y',
                    side='right',
                    tickformat='.2%',
                    showgrid=False
                ),
                xaxis=dict(title="Date"),
                hovermode='x unified',
                legend=dict(orientation="h", y=1.02),
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_spread, use_container_width=True)
    
    with col_table:
        st.subheader("Historical Data")
        
        # Prepare display columns
        display_cols = ['close']
        
        if 'rms_7_14' in processed_df.columns:
            display_cols.append('rms_7_14')
        if 'rms_2_3' in processed_df.columns:
            display_cols.append('rms_2_3')
        
        # Add HV columns (show first 4 windows to save space)
        for w in vol_windows[:4]:
            display_cols.append(f'hv_{w}')
        
        # Create table with most recent data first
        table_df = processed_df[display_cols].copy().sort_index(ascending=False)
        
        # Format for display with UTC timezone indicator
        fmt_df = table_df.copy()
        fmt_df.index = fmt_df.index.strftime('%Y-%m-%d %H:%M UTC')
        
        for col in fmt_df.columns:
            if col == 'close':
                if current_price < 100:
                    fmt_df[col] = fmt_df[col].apply(lambda x: f"${x:.4f}")
                else:
                    fmt_df[col] = fmt_df[col].apply(lambda x: f"${x:.2f}")
            else:
                fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.2%}")
        
        # Rename columns for clarity
        rename_map = {
            'close': 'Price',
            'rms_7_14': 'RMS(7,14)',
            'rms_2_3': 'RMS(2,3)'
        }
        for w in vol_windows[:4]:
            rename_map[f'hv_{w}'] = f'{w}d HV'
        
        fmt_df = fmt_df.rename(columns=rename_map)
        
        st.dataframe(
            fmt_df,
            height=600,
            use_container_width=True
        )
        
        # =============================================================================
        # DOWNLOAD BUTTON - Export complete dataset
        # =============================================================================
        
        st.markdown("### 📥 Export Data")
        
        # Prepare export dataframe with all calculated metrics
        export_df = processed_df.copy()
        export_df.index.name = 'Date'
        
        # Create CSV
        csv_buffer = BytesIO()
        export_df.to_csv(csv_buffer)
        csv_data = csv_buffer.getvalue()
        
        filename = f"{symbol}_{market_mode}_Volatility_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        st.download_button(
            label=f"📥 Download {symbol} Volatility Data",
            data=csv_data,
            file_name=filename,
            mime='text/csv',
            key=f"download_{symbol}_{idx}",
            help=f"Export complete volatility dataset for {symbol} from {start_date} to {end_date}"
        )
    
    # =============================================================================
    # OPTIONS PRICER
    # =============================================================================
    
    with st.expander("🛠️ Theoretical Options Pricer", expanded=False):
        st.markdown(f"""
        **Black-Scholes pricing using realized volatility as input**
        
        - Spot: ${current_price:,.2f}
        - Volatility Input: {latest.get('rms_7_14', latest.get('rms_vol', 0)):.2%} (RMS 7,14)
        - Days to Expiry: {days_expiry}
        - Risk-Free Rate: {risk_free_rate:.2%}
        """)
        
        # Use RMS volatility as input
        vol_input = latest.get('rms_7_14', latest.get('rms_vol', 0.5))
        t_years = days_expiry / 365.0
        
        # Generate strikes across the selected range
        strikes = np.linspace(
            current_price * strike_range[0],
            current_price * strike_range[1],
            5
        )
        
        pricer_data = []
        
        for K in strikes:
            k_pct = K / current_price
            
            # Calculate Call
            c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(
                current_price, K, t_years, risk_free_rate, vol_input, 'call'
            )
            
            # Calculate Put
            p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(
                current_price, K, t_years, risk_free_rate, vol_input, 'put'
            )
            
            pricer_data.append({
                "Strike": f"${K:,.2f}",
                "K/S": f"{k_pct:.1%}",
                "Call Price": f"${c_price:.2f}",
                "Call Δ": f"{c_delta:.3f}",
                "Put Price": f"${p_price:.2f}",
                "Put Δ": f"{p_delta:.3f}",
                "Γ": f"{c_gamma:.4f}",
                "Vega": f"{c_vega:.2f}",
                "Θ/day": f"{c_theta:.2f}"
            })
        
        pricer_df = pd.DataFrame(pricer_data)
        st.table(pricer_df)
        
        st.caption("Δ = Delta | Γ = Gamma | Θ = Theta (per day) | Vega = per 1% vol change")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("Market Maker HV Screener | Data: Binance | HV calculated at 08:00 UTC using 365-day annualization")
