import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import io  # For CSV buffering

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="MM Volatility Dashboard")

st.title("⚡ Dynamic Volatility Dashboard (RMS Model)")
st.markdown("""
**Strategy:** Market Maker (MM) Risk Engine for Volatile Markets (e.g., Memecoins/Altcoins)  
**Metric:** Root Mean Square (RMS) of Custom Historical Volatility Windows  
**Customizable:** By date range, token, data source (CoinGecko, Kraken Futures, Binance Perpetuals), vol windows. With export options for historical data.
""")

# ==========================================
# 2. DATA ENGINE (Multi-source)
# ==========================================
@st.cache_data(ttl=600)
def get_crypto_data(source, symbol, start_time=None, end_time=None, days=90):
    df = pd.DataFrame()
    
    if source == "Binance Perpetuals":
        base_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1d&limit=1000"
        if start_time:
            base_url += f"&startTime={start_time}"
        if end_time:
            base_url += f"&endTime={end_time}"
        try:
            response = requests.get(base_url)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp').sort_index()
                df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except:
            pass
    
    elif source == "CoinGecko":
        base_url = f"https://api.coingecko.com/api/v3/coins/{symbol}/ohlc?vs_currency=usd&days={days}"
        try:
            response = requests.get(base_url)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp').sort_index()
                df['volume'] = np.nan  # CoinGecko OHLC doesn't include volume
        except:
            pass
    
    elif source == "Kraken Futures":
        base_url = f"https://futures.kraken.com/derivatives/api/v3/history?symbol={symbol}&resolution=1D"
        if start_time:
            base_url += f"&from={start_time}"
        if end_time:
            base_url += f"&to={end_time}"
        try:
            response = requests.get(base_url)
            if response.status_code == 200:
                data = response.json()
                if 'history' in data['result']:
                    hist = data['result']['history']
                    df = pd.DataFrame(hist)
                    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
                    df = df.set_index('timestamp').sort_index()
                    df = df[['open', 'high', 'low', 'close', 'volume']]
        except:
            pass
    
    if df.empty:
        st.warning(f"No data for {symbol} on {source}. Check symbol availability.")
    return df

@st.cache_data(ttl=600)
def get_spot_price(source, symbol):
    price = None
    
    if source == "Binance Perpetuals":
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                price = float(response.json()['markPrice'])
        except:
            pass
    
    elif source == "CoinGecko":
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                price = response.json()[symbol]['usd']
        except:
            pass
    
    elif source == "Kraken Futures":
        url = f"https://futures.kraken.com/derivatives/api/v3/ticker?symbol={symbol}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                price = response.json()['result']['last']
        except:
            pass
    
    return price

@st.cache_data(ttl=600)
def get_implied_vol(currency='BTC'):
    try:
        url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
        now_ms = int(time.time() * 1000)
        params = {
            "currency": currency,
            "start_timestamp": now_ms - 3600 * 1000,
            "end_timestamp": now_ms,
            "resolution": "3600"
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'result' in data and 'data' in data['result'] and data['result']['data']:
            return data['result']['data'][-1][4]
        return None
    except:
        return None

# ==========================================
# 3. MATHEMATICAL ENGINE
# ==========================================
def calculate_metrics(df, vol_windows=[2, 3, 7, 14, 30, 60, 90]):
    if len(df) < max(vol_windows) + 1:
        return pd.DataFrame()
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    ANNUAL_FACTOR = np.sqrt(365)
    
    for w in vol_windows:
        df[f'hv_{w}'] = df['log_ret'].rolling(window=w).std() * ANNUAL_FACTOR
    
    if 2 in vol_windows and 3 in vol_windows:
        df['normalized_23'] = np.sqrt((df['hv_2']**2 + df['hv_3']**2) / 2)
    if 7 in vol_windows and 14 in vol_windows:
        df['normalized_714'] = np.sqrt((df['hv_7']**2 + df['hv_14']**2) / 2)
    
    df['rms_vol'] = df['normalized_714'] if 'normalized_714' in df else np.nan
    
    return df.dropna()

# ==========================================
# 4. BLACK-SCHOLES PRICER
# ==========================================
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

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

# ==========================================
# 5. USER INTERFACE LAYOUT
# ==========================================
with st.sidebar:
    st.header("🔧 Settings")
    
    # Source selector
    source = st.selectbox("Data Source", ["Binance Perpetuals", "CoinGecko", "Kraken Futures"])
    
    # Token selector
    token_options = {
        "Bitcoin": {"Binance Perpetuals": "BTCUSDT", "CoinGecko": "bitcoin", "Kraken Futures": "PF_XBTUSD"},
        "Ethereum": {"Binance Perpetuals": "ETHUSDT", "CoinGecko": "ethereum", "Kraken Futures": "PF_ETHUSD"},
        "PEPE": {"Binance Perpetuals": "PEPEUSDT", "CoinGecko": "pepe", "Kraken Futures": "PF_PEPEUSD"}
    }
    selected_token = st.selectbox("Select Token", list(token_options.keys()))
    symbol = token_options[selected_token][source]
    
    st.divider()
    
    # Date customization
    today = datetime.today()
    end_date = st.date_input("End Date", value=today)
    start_date = st.date_input("Start Date", value=today - timedelta(days=90))
    if start_date > end_date:
        st.error("Start date must be before end date.")
    
    start_time = int(time.mktime(start_date.timetuple())) * 1000
    end_time = int(time.mktime(end_date.timetuple())) * 1000
    days_range = (end_date - start_date).days + 1
    
    # Vol windows
    vol_windows_str = st.text_input("Vol Windows (comma-separated days)", value="2,3,7,14,30,60,90")
    vol_windows = [int(w.strip()) for w in vol_windows_str.split(',') if w.strip().isdigit()]
    
    st.divider()
    
    st.subheader("Option Pricing Inputs")
    strike_min_pct, strike_max_pct = st.slider("Strike Distance Range (%)", 0.5, 2.0, (0.8, 1.2), 0.01)
    days_expiry = st.number_input("Days to Expiry", min_value=1, value=30)
    risk_free = st.number_input("Risk Free Rate", value=0.05)
    
    # Optional: Upload pricer file
    uploaded_file = st.file_uploader("Upload Options Pricer File (CSV/Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            df_upload = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df_upload)} rows. Customize as needed.")
        except:
            st.warning("Couldn't read file.")

# Fetch and process data for selected token
raw_df = get_crypto_data(source, symbol, start_time=start_time, end_time=end_time, days=days_range)
if not raw_df.empty:
    processed_df = calculate_metrics(raw_df, vol_windows)
    
    if len(processed_df) > 0:
        latest = processed_df.iloc[-1]
        spot_price = get_spot_price(source, symbol)
        
        # Implied Vol (only for BTC/ETH)
        iv = None
        if selected_token in ["Bitcoin", "Ethereum"]:
            deribit_currency = 'BTC' if selected_token == "Bitcoin" else 'ETH'
            iv = get_implied_vol(deribit_currency)
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Spot Price Today", f"${spot_price:.2f}" if spot_price else "N/A")
        col2.metric("RMS Vol (7,14)", f"{latest['rms_vol']*100:.2f}%" if 'rms_vol' in latest else "N/A")
        col3.metric("Normalized (2/3)", f"{latest['normalized_23']*100:.2f}%" if 'normalized_23' in latest else "N/A")
        col4.metric("Normalized (7/14)", f"{latest['normalized_714']*100:.2f}%" if 'normalized_714' in latest else "N/A")
        col5.metric("Expected IV (30d Deribit)", f"{iv:.2f}%" if iv else "N/A")
        
        # HV Table
        st.subheader("Historical Volatility Table")
        table_columns = ['close'] + ['normalized_23', 'normalized_714'] + [f'hv_{w}' for w in sorted(vol_windows)]
        if all(col in processed_df.columns for col in table_columns):
            table_df = processed_df[table_columns]
            table_df = table_df.iloc[-20:].reset_index()
            table_df.columns = ['dt', 'close', 'normalized(2/3)', 'normalized(7/14)'] + [f'hv_{w}' for w in sorted(vol_windows)]
            st.dataframe(table_df)
        else:
            st.warning("Insufficient vol windows for table.")
        
        # Visualization
        st.subheader("📈 Historical Volatility Chart (Past HV)")
        fig = go.Figure()
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']
        for i, w in enumerate(vol_windows):
            if f'hv_{w}' in processed_df:
                fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df[f'hv_{w}'],
                                         name=f'HV {w}d', line=dict(color=colors[i % len(colors)])))
        if 'normalized_23' in processed_df:
            fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['normalized_23'],
                                     name='Normalized (2/3)', line=dict(color='black', dash='dash')))
        if 'normalized_714' in processed_df:
            fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['normalized_714'],
                                     name='Normalized (7/14)', line=dict(color='gray', dash='dash')))
        
        fig.update_layout(
            yaxis=dict(title="Volatility", tickformat='.0%'),
            height=500,
            title=f"{selected_token} Historical Volatility Regimes ({source})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Option Pricer
        st.subheader("🛠️ Market Maker Inventory Pricer (Shadow Options)")
        current_spot = spot_price if spot_price else latest['close']
        time_years = days_expiry / 365.0
        rms_vol = latest['rms_vol'] if 'rms_vol' in latest else 0.0
        
        pricer_data = {"Metric": ["Price", "Delta", "Gamma", "Theta", "Vega"]}
        for strike_pct in np.linspace(strike_min_pct, strike_max_pct, 3):
            target_strike = current_spot * strike_pct
            c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(current_spot, target_strike, time_years, risk_free, rms_vol, 'call')
            p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(current_spot, target_strike, time_years, risk_free, rms_vol, 'put')
            
            col_name = f"Call ({strike_pct:.2f}x Spot)"
            pricer_data[col_name] = [f"{c_price:.4f}", f"{c_delta:.4f}", f"{c_gamma:.4f}", f"{c_theta:.4f}", f"{c_vega:.4f}"]
            
            col_name = f"Put ({strike_pct:.2f}x Spot)"
            pricer_data[col_name] = [f"{p_price:.4f}", f"{p_delta:.4f}", f"{p_gamma:.4f}", f"{p_theta:.4f}", f"{p_vega:.4f}"]
        
        st.table(pd.DataFrame(pricer_data).set_index("Metric"))
        
        # Export for selected token
        csv_buffer = io.StringIO()
        processed_df.to_csv(csv_buffer)
        st.download_button(
            label="Download Historical Data for Selected Token",
            data=csv_buffer.getvalue(),
            file_name=f"historical_vol_{selected_token}_{source}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"Not enough data for {selected_token} (need at least {max(vol_windows) + 1} days).")
else:
    st.warning("No data found.")

# Export for all tokens
if st.button("Export Historical Data for All Tokens"):
    all_dfs = []
    for tok in token_options.keys():
        sym = token_options[tok][source]
        raw = get_crypto_data(source, sym, start_time=start_time, end_time=end_time, days=days_range)
        if not raw.empty:
            proc = calculate_metrics(raw, vol_windows)
            if not proc.empty:
                proc['token'] = tok
                proc['source'] = source
                all_dfs.append(proc)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs)
        csv_buffer_all = io.StringIO()
        combined_df.to_csv(csv_buffer_all)
        st.download_button(
            label="Download All Tokens Data",
            data=csv_buffer_all.getvalue(),
            file_name=f"all_historical_vol_{source}.csv",
            mime="text/csv"
        )
    else:
        st.error("No data available for any tokens.")
