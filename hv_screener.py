"""
Historical Volatility Screener
===============================

This Streamlit app allows users to screen historical volatility (HV) for a
curated list of crypto assets.  It leverages Binance's public API to fetch
daily candle data and computes annualised historical volatility over multiple
rolling windows.  The app also supports comparing two different HV tenors
side‑by‑side for the same token, enabling a quick view of how longer and
shorter windows diverge over time.  The layout is designed to mirror the
side‑by‑side chart and table found in the supplied screenshot: the HV
curves are plotted on the left, while a compact table of recent values is
displayed on the right.

Key features
------------

* **Asset filtering** — only tokens from the provided `asset list.csv` are
  available for selection.  The app automatically appends `USDT` to the
  symbol when querying Binance, and gracefully handles missing pairs.
* **Custom HV windows** — input a comma‑separated list of window lengths (in
  days).  For each window `w` the app computes the annualised HV using
  rolling log returns.
* **Normalised RMS metrics** — if windows 2 & 3 or 7 & 14 are present, their
  root mean square (RMS) is computed as in the reference implementation.
* **Tenor comparison** — select two window lengths from your list.  A
  dedicated chart overlays their HV curves to facilitate tenor comparison.
* **Multi‑asset support** — select multiple tokens and view their charts and
  tables sequentially.  Each token has its own section with metrics,
  tables and charts.

The app is self‑contained and can be run locally with:

```bash
streamlit run hv_screener.py
```

Make sure that `asset list.csv` resides in the same directory as this
script.  The file was supplied by the user and contains the coin symbol,
common name and an optional CoinGecko API identifier.
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
    Use this tool to inspect and compare historical volatility across a curated
    list of crypto assets.  Select one or more tokens from the side bar and
    specify your desired HV windows.  For each token the app will display a
    multi‑line HV chart alongside a recent data table.  You can also pick two
    windows to compare their HV curves directly.
    """
)

# -----------------------------------------------------------------------------
# 2. ASSET LIST LOADING
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_asset_list(csv_path: str) -> pd.DataFrame:
    """Load the asset list from a CSV file and return a DataFrame.

    The CSV is expected to have columns: ``Coin symbol``, ``Common Name`` and
    ``CG API ID``.  Missing values are filled with empty strings.
    """
    try:
        df = pd.read_csv(csv_path)
        return df.fillna("")
    except Exception as exc:
        st.error(f"Failed to load asset list: {exc}")
        return pd.DataFrame(columns=["Coin symbol", "Common Name", "CG API ID"])


def build_token_options(df: pd.DataFrame) -> dict:
    """Construct a mapping from a display name to Binance symbol.

    Each row in the data frame contributes a key of the form
    ``"{symbol} - {common_name}"`` (if a common name exists) or just ``symbol``.
    The value is the Binance trading symbol obtained by appending ``USDT`` to
    the coin symbol.  Duplicate display names are de‑duplicated by keeping the
    first occurrence.
    """
    options = {}
    for _, row in df.iterrows():
        coin = str(row.get("Coin symbol", "")).strip().upper()
        if not coin:
            continue
        common = str(row.get("Common Name", "")).strip()
        display = f"{coin} - {common}" if common else coin
        # Avoid overwriting if the display name already exists
        if display not in options:
            options[display] = f"{coin}USDT"
    return options


# -----------------------------------------------------------------------------
# 3. DATA FETCHING UTILITIES
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_crypto_data(symbol: str, interval: str = "1d", start_time: int | None = None,
                    end_time: int | None = None, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given symbol from Binance.

    This helper contacts Binance's public REST API to retrieve candlestick
    information.  To maximise compatibility with Streamlit Community Cloud (which
    restricts calls to ``api.binance.com``【804757919967632†L58-L96】), the function uses the
    ``data.binance.com`` domain.  If you deploy this app in an environment
    without such restrictions, you can revert to ``api.binance.com`` by
    changing the ``base_url`` variable.

    Parameters
    ----------
    symbol : str
        The trading pair symbol (e.g., ``BTCUSDT``).
    interval : str, optional
        Candle interval (e.g., ``'1d'`` for daily).  Defaults to ``'1d'``.
    start_time : int, optional
        Millisecond timestamp for the beginning of the query range.
    end_time : int, optional
        Millisecond timestamp for the end of the query range.
    limit : int, optional
        Maximum number of data points to return.  Binance caps at 1000 per call.

    Returns
    -------
    pandas.DataFrame
        Data indexed by timestamp with columns: open, high, low, close and volume.

    Notes
    -----
    This function is cached to avoid repeated calls.  If the Binance API
    returns an error or no data, an empty DataFrame is returned instead.
    """
    # Use the data.binance.com endpoint to avoid 403 errors on Streamlit Cloud
    base_url = "https://data.binance.com/api/v3/klines"
    params: dict[str, int | str] = {
        'symbol': symbol,
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
    """Get the latest spot price for a symbol from Binance.

    To increase the likelihood of success on Streamlit Cloud, this function
    queries the ``data.binance.com`` domain rather than ``api.binance.com``.
    See the discussion on the Streamlit forums regarding 403 errors【804757919967632†L58-L96】
    for more context.  If your environment permits, you can switch back to
    ``api.binance.com`` by altering the URL below.

    Returns ``None`` if the price cannot be fetched.
    """
    url = f"https://data.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return float(response.json().get('price', 'nan'))
        else:
            return None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_implied_vol(currency: str = 'BTC') -> float | None:
    """Fetch the latest implied volatility index from Deribit.

    Deribit provides a volatility index (DVOL) based on a 30‑day implied
    volatility.  This function queries the last hour of DVOL data and returns
    the closing value.  It is only used for BTC and ETH where such data is
    available.
    """
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    now_ms = int(time.time() * 1000)
    params: dict[str, int | str] = {
        "currency": currency,
        "start_timestamp": now_ms - 3600 * 1000,
        "end_timestamp": now_ms,
        "resolution": "3600",
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'result' in data and 'data' in data['result'] and data['result']['data']:
            latest_vol = data['result']['data'][-1][4]  # closing vol
            return latest_vol
    except Exception:
        pass
    return None


def calculate_hv_metrics(df: pd.DataFrame, vol_windows: list[int]) -> pd.DataFrame:
    """Compute historical volatility metrics for a given OHLCV DataFrame.

    The function calculates log returns and rolling standard deviations for each
    window length provided in ``vol_windows``.  The resulting volatility is
    annualised using ``sqrt(365)`` (since crypto trades every day).  In
    addition, two normalised RMS metrics are computed if the requisite
    windows are present:

    * ``normalized_23`` is the RMS of ``hv_2`` and ``hv_3``.
    * ``normalized_714`` is the RMS of ``hv_7`` and ``hv_14``.

    The function returns a DataFrame with the computed columns and drops
    any rows containing NaNs.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if len(df) < max(vol_windows) + 1:
        # not enough data for the largest window
        return pd.DataFrame()
    # calculate log returns
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    # annualisation factor
    annual_factor = np.sqrt(365)
    # compute rolling std for each window
    for w in vol_windows:
        df[f'hv_{w}'] = df['log_ret'].rolling(window=w).std() * annual_factor
    # compute RMS normalised windows
    if 2 in vol_windows and 3 in vol_windows:
        df['normalized_23'] = np.sqrt((df['hv_2']**2 + df['hv_3']**2) / 2)
    if 7 in vol_windows and 14 in vol_windows:
        df['normalized_714'] = np.sqrt((df['hv_7']**2 + df['hv_14']**2) / 2)
    # a representative RMS for UI metrics (e.g., 7/14) if available
    df['rms_vol'] = df['normalized_714'] if 'normalized_714' in df.columns else np.nan
    return df.dropna()


def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = 'call') -> tuple[float, float, float, float, float]:
    """Calculate option greeks using the Black–Scholes model.

    Returns price, delta, gamma, theta and vega.  If the time to expiry ``T``
    is non‑positive, zeros are returned.
    """
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
    return (price, delta, gamma, theta, vega)


# -----------------------------------------------------------------------------
# 4. USER INTERFACE: SIDEBAR CONTROLS
# -----------------------------------------------------------------------------

# Load asset list and build token options
asset_csv_path = os.path.join(os.path.dirname(__file__), 'asset list.csv')
asset_df = load_asset_list(asset_csv_path)
token_options = build_token_options(asset_df)

with st.sidebar:
    st.header("🔧 Settings")
    if not token_options:
        st.error("No tokens available from the asset list.")
    else:
        # Multi‑select for tokens; default to a few popular names if present
        default_tokens = []
        for name in ["BTC", "ETH", "DOGE"]:
            for k in token_options.keys():
                if k.startswith(name):
                    default_tokens.append(k)
                    break
        selected_display = st.multiselect(
            "Select Tokens (max 5)",
            options=list(token_options.keys()),
            default=default_tokens,
            max_selections=5,
            help="Choose up to five assets to display simultaneously."
        )
        # Date range inputs
        st.subheader("Date Range")
        today = datetime.now().date()
        # default to last 180 days
        default_start = today - timedelta(days=180)
        start_date = st.date_input("Start Date", value=default_start, max_value=today)
        end_date = st.date_input("End Date", value=today, min_value=start_date, max_value=today)
        if start_date > end_date:
            st.error("Start date must be on or before the end date.")
        # Volatility windows
        st.subheader("HV Windows (days)")
        windows_input = st.text_input(
            "Enter comma‑separated integers", value="2,3,7,14,30,60,90"
        )
        vol_windows: list[int] = []
        for part in windows_input.split(','):
            part = part.strip()
            if part.isdigit():
                vol_windows.append(int(part))
        # ensure unique and sorted
        vol_windows = sorted(set(vol_windows))
        if not vol_windows:
            st.error("Please enter at least one valid HV window.")
        # Tenor comparison selection
        st.subheader("Tenor Comparison")
        tenor1 = st.selectbox(
            "Select first HV window", vol_windows, index=0 if vol_windows else 0
        )
        tenor2 = st.selectbox(
            "Select second HV window", vol_windows, index=1 if len(vol_windows) > 1 else 0
        )
        # Option pricer inputs
        st.subheader("Option Pricer (Experimental)")
        strike_range = st.slider(
            "Strike Distance Range (%)", 0.5, 2.0, (0.8, 1.2), 0.01
        )
        days_expiry = st.number_input(
            "Days to Expiry", min_value=1, value=30, step=1
        )
        risk_free = st.number_input("Risk Free Rate", value=0.05, step=0.001)


# -----------------------------------------------------------------------------
# 5. MAIN CONTENT: PROCESS AND DISPLAY FOR EACH SELECTED TOKEN
# -----------------------------------------------------------------------------

if selected_display and vol_windows and start_date <= end_date:
    # convert dates to milliseconds for Binance API
    # Convert the selected dates to timestamps at 08:00 UTC.  Binance daily
    # candles close at 00:00 UTC, but many settlement procedures reference 08:00
    # UTC.  To align with this, we anchor both the start and end of the
    # retrieval window at 08:00 instead of midnight.  This effectively
    # shifts the sampling window to start and end at 08:00.
    start_dt = datetime(start_date.year, start_date.month, start_date.day, 8, 0, 0)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, 8, 0, 0)
    start_ms = int(time.mktime(start_dt.timetuple())) * 1000
    end_ms = int(time.mktime(end_dt.timetuple())) * 1000
    # iterate through selected tokens
    for display_name in selected_display:
        symbol = token_options.get(display_name)
        if not symbol:
            continue
        st.markdown(f"## {display_name} ({symbol})")
        # fetch data
        with st.spinner(f"Fetching data for {symbol}…"):
            raw_df = get_crypto_data(symbol, start_time=start_ms, end_time=end_ms,
                                     limit=min((end_ms - start_ms) // (24*3600*1000) + 1, 1000))
        if raw_df.empty:
            st.warning(
                f"No data returned for {symbol}. This pair may not exist on Binance or the date range is too long."
            )
            continue
        # compute metrics
        processed_df = calculate_hv_metrics(raw_df, vol_windows)
        if processed_df.empty:
            st.warning(
                f"Not enough data for {display_name} to compute HV for the specified windows."
            )
            continue
        latest = processed_df.iloc[-1]
        spot_price = get_spot_price(symbol)
        # Deribit IV (only for BTC/ETH)
        iv_value: float | None = None
        if display_name.startswith("BTC"):
            iv_value = get_implied_vol('BTC')
        elif display_name.startswith("ETH"):
            iv_value = get_implied_vol('ETH')
        # show key metrics as columns
        col_metrics = st.columns(5)
        col_metrics[0].metric(
            "Spot Price", f"${spot_price:.2f}" if spot_price else "N/A"
        )
        col_metrics[1].metric(
            "RMS Vol (7/14)", f"{latest['rms_vol']*100:.2f}%" if not np.isnan(latest['rms_vol']) else "N/A"
        )
        col_metrics[2].metric(
            "Normalised (2/3)", f"{latest['normalized_23']*100:.2f}%" if 'normalized_23' in latest else "N/A"
        )
        col_metrics[3].metric(
            "Normalised (7/14)", f"{latest['normalized_714']*100:.2f}%" if 'normalized_714' in latest else "N/A"
        )
        col_metrics[4].metric(
            "Implied Vol (Deribit)", f"{iv_value:.2f}%" if iv_value else "N/A"
        )
        # chart and table side by side
        chart_col, table_col = st.columns([2, 1])
        # Chart: multi HV lines and normalised curves
        with chart_col:
            st.subheader("HV Chart")
            fig = go.Figure()
            colour_palette = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            for idx, w in enumerate(vol_windows):
                fig.add_trace(go.Scatter(
                    x=processed_df.index,
                    y=processed_df[f'hv_{w}'],
                    name=f'HV {w}d',
                    line=dict(color=colour_palette[idx % len(colour_palette)]),
                ))
            # add normalised lines if available
            if 'normalized_23' in processed_df.columns:
                fig.add_trace(go.Scatter(
                    x=processed_df.index,
                    y=processed_df['normalized_23'],
                    name='Normalized (2/3)',
                    line=dict(color='black', dash='dash'),
                ))
            if 'normalized_714' in processed_df.columns:
                fig.add_trace(go.Scatter(
                    x=processed_df.index,
                    y=processed_df['normalized_714'],
                    name='Normalized (7/14)',
                    line=dict(color='grey', dash='dash'),
                ))
            fig.update_layout(
                height=400,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                yaxis=dict(title='Volatility', tickformat='.0%'),
                xaxis=dict(title=None),
                margin=dict(l=40, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        # Tenor comparison chart
        with chart_col:
            if tenor1 != tenor2 and tenor1 in vol_windows and tenor2 in vol_windows:
                st.subheader(f"Comparison: {tenor1}d vs {tenor2}d")
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(
                    x=processed_df.index,
                    y=processed_df[f'hv_{tenor1}'],
                    name=f'HV {tenor1}d',
                    line=dict(color='blue'),
                ))
                fig_cmp.add_trace(go.Scatter(
                    x=processed_df.index,
                    y=processed_df[f'hv_{tenor2}'],
                    name=f'HV {tenor2}d',
                    line=dict(color='red'),
                ))
                # optionally show the spread between the two
                spread = processed_df[f'hv_{tenor1}'] - processed_df[f'hv_{tenor2}']
                fig_cmp.add_trace(go.Scatter(
                    x=processed_df.index,
                    y=spread,
                    name=f'Spread ({tenor1}d - {tenor2}d)',
                    line=dict(color='green', dash='dot'),
                    yaxis='y2'
                ))
                fig_cmp.update_layout(
                    height=400,
                    yaxis=dict(title='HV', tickformat='.0%'),
                    yaxis2=dict(
                        title='Spread',
                        overlaying='y',
                        side='right',
                        tickformat='.0%'
                    ),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)
        # Table: last 20 rows with selected columns
        with table_col:
            st.subheader("Recent HV Table")
            table_cols = ['close']
            if 'normalized_23' in processed_df.columns:
                table_cols.append('normalized_23')
            if 'normalized_714' in processed_df.columns:
                table_cols.append('normalized_714')
            table_cols += [f'hv_{w}' for w in vol_windows]
            table_df = processed_df[table_cols].iloc[-20:].copy()
            table_df = table_df.reset_index()
            # rename columns for readability
            rename_map: dict[str, str] = {
                'timestamp': 'dt',
                'close': 'close',
                'normalized_23': 'normalized(2/3)',
                'normalized_714': 'normalized(7/14)',
            }
            for w in vol_windows:
                rename_map[f'hv_{w}'] = f'hv_{w}'
            table_df.rename(columns=rename_map, inplace=True)
            # format numeric columns as percent for vol columns
            for col in table_df.columns:
                if col.startswith('hv_') or col.startswith('normalized'):
                    table_df[col] = (table_df[col] * 100).map(lambda x: f"{x:.2f}%")
                if col == 'close':
                    table_df[col] = table_df[col].map(lambda x: f"{x:.4f}")
            st.dataframe(table_df, hide_index=True)
        # Option pricer
        with st.expander("Option Pricer", expanded=False):
            current_spot = spot_price if spot_price else latest['close']
            T_years = days_expiry / 365.0
            sigma = latest['rms_vol'] if not np.isnan(latest['rms_vol']) else 0.0
            pricer_data: dict[str, list[str]] = {"Metric": ["Price", "Delta", "Gamma", "Theta", "Vega"]}
            strikes = np.linspace(strike_range[0], strike_range[1], 3)
            for pct in strikes:
                strike = current_spot * pct
                c_price, c_delta, c_gamma, c_theta, c_vega = black_scholes(
                    current_spot, strike, T_years, risk_free, sigma, 'call'
                )
                p_price, p_delta, p_gamma, p_theta, p_vega = black_scholes(
                    current_spot, strike, T_years, risk_free, sigma, 'put'
                )
                pricer_data[f"Call ({pct:.2f}x) "] = [
                    f"{c_price:.4f}", f"{c_delta:.4f}", f"{c_gamma:.4f}", f"{c_theta:.4f}", f"{c_vega:.4f}"
                ]
                pricer_data[f"Put ({pct:.2f}x) "] = [
                    f"{p_price:.4f}", f"{p_delta:.4f}", f"{p_gamma:.4f}", f"{p_theta:.4f}", f"{p_vega:.4f}"
                ]
            st.table(pd.DataFrame(pricer_data).set_index("Metric"))
else:
    st.info("Configure your settings in the sidebar to begin.")
