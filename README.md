# Historical Volatility Screener - Market Maker Edition

Professional-grade volatility analysis tool designed specifically for market makers to analyze realized volatility across crypto assets, evaluate term structures, and export data for offline risk modeling.

## 🎯 Key Features

### For Market Makers
- **Multi-Tenor Analysis**: Track volatility across 2d, 3d, 7d, 14d, 30d, 60d, and 90d windows
- **RMS Metrics**: Normalized volatility measures for inventory risk pricing
- **Term Structure Analysis**: Compare short vs long-dated realized volatility and identify inversions
- **Data Export**: Download complete historical volatility datasets for your selected date range
- **Theoretical Pricer**: Black-Scholes option pricing using realized vol as input

### Technical Capabilities
- **Dual Market Support**: Toggle between Binance Spot and USDT-M Perpetual Futures
- **Multi-Asset Comparison**: Analyze up to 5 assets simultaneously
- **Customizable Windows**: Define your own volatility calculation periods
- **Real-time Pricing**: Live price feeds from Binance
- **Interactive Charts**: Plotly-powered visualizations with zoom and export capabilities

## 📋 Requirements

```bash
streamlit
pandas
numpy
requests
plotly
scipy
pytz
```

Install all dependencies:
```bash
pip install streamlit pandas numpy requests plotly scipy pytz
```

## 🚀 Getting Started

### 1. Setup
Ensure both files are in the same directory:
- `hv_screener_enhanced.py`
- `asset_list.csv`

### 2. Run the Application
```bash
streamlit run hv_screener_enhanced.py
```

The app will open in your default browser at `http://localhost:8501`

### 3. Configuration

**In the Sidebar:**

1. **Market Type**: Choose between Spot or Perps (Perpetual Futures)
2. **Asset Selection**: Select up to 5 assets from the curated list
3. **Date Range**: Define your analysis period (default: last 180 days)
4. **HV Windows**: Customize volatility calculation periods (default: 2,3,7,14,30,60,90)
5. **Term Structure**: Select short and long tenors for spread analysis
6. **Options Pricer**: Set expiry, strike range, and risk-free rate

## 📊 Understanding the Metrics

### Timestamp Convention: 08:00 UTC
**All HV calculations use 08:00 UTC daily snapshots.** This ensures:
- Consistency across different assets and markets
- Alignment with common market maker practices
- Clean daily boundaries for volatility calculations
- Reproducible results across different time zones

The app fetches Binance daily candles (which close at 00:00 UTC) and adjusts timestamps to 08:00 UTC for all volatility calculations.

### RMS (Root Mean Square) Volatility
- **RMS (7,14)**: Primary metric for inventory risk - normalized volatility across 7d and 14d windows
- **RMS (2,3)**: Ultra-short-term volatility for intraday/scalping strategies
- Formula: `RMS = √((HV₁² + HV₂²) / 2)`

### Historical Volatility (HV)
- **Timestamp**: All calculations use 08:00 UTC daily snapshots
- **Returns**: Calculated using log returns: `ln(Close_t / Close_t-1)`
- **Annualization**: Using 365-day convention (not 252 trading days)
- Formula: `HV = StdDev(log_returns) × √365`

**Why 08:00 UTC?**
- Standard market maker convention
- Aligns with Asian market close / European market open
- Avoids midnight boundary effects
- Ensures consistent daily snapshots across all assets

### Term Structure Spread
- Shows the difference between short and long-dated volatility
- Positive spread → backwardation (short vol > long vol)
- Negative spread → contango (long vol > short vol)
- Useful for identifying regime shifts and mean-reversion opportunities

## 💾 Data Export

### What Gets Exported
The download button exports a CSV file containing:
- **Date index**: Daily timestamps at 08:00 UTC
- **OHLCV data**: Open, High, Low, Close, Volume
- **Log returns**: Daily log returns (calculated from 08:00 UTC closes)
- **All HV calculations**: Every window you specified
- **RMS metrics**: Normalized volatility measures

### File Naming Convention
```
{SYMBOL}_{MARKET_TYPE}_Volatility_{START_DATE}_{END_DATE}.csv
```

Example: `BTCUSDT_Perps_Volatility_20240719_20250119.csv`

### Use Cases for Exported Data
- Backtesting trading strategies
- Risk model calibration
- Correlation analysis across assets
- Custom volatility forecasting models
- Integration with proprietary risk systems

## 🛠️ Options Pricer

The built-in Black-Scholes pricer uses **realized volatility** (not implied vol) as the volatility input.

### Inputs
- **Spot Price**: Current market price from Binance
- **Strike Range**: Selectable range around spot (default: 80% to 120%)
- **Volatility**: Automatically uses RMS (7,14) from historical data
- **Days to Expiry**: User-defined (1-365 days)
- **Risk-Free Rate**: User-defined annual rate

### Outputs (Greeks)
- **Price**: Theoretical option value
- **Delta (Δ)**: Sensitivity to spot price changes
- **Gamma (Γ)**: Rate of change of delta
- **Theta (Θ)**: Time decay per day
- **Vega**: Sensitivity to 1% volatility change

### Market Maker Usage
- Price inventory hedges using realized vol
- Estimate gamma exposure across strikes
- Calculate vega risk for volatility positions
- Compare theoretical values to market prices for edge detection

## 📈 Workflow Example

### Daily Volatility Check
1. Select your core trading assets (e.g., BTC, ETH, SOL)
2. Review RMS (7,14) for current volatility regime
3. Check term structure spread for backwardation/contango
4. Compare current levels to historical ranges
5. Export data if regime shift detected

### Pre-Trade Analysis
1. Select the specific asset you're trading
2. Analyze volatility across multiple tenors
3. Use options pricer to estimate hedge costs
4. Export data for detailed modeling if needed
5. Document volatility assumptions for trade plan

### Risk Monitoring
1. Set up 5-asset dashboard with your portfolio holdings
2. Monitor RMS metrics for sudden spikes
3. Track term structure for early warning signals
4. Export periodic snapshots for compliance/audit

## 🔧 Customization Tips

### Adding Custom Assets
Edit `asset_list.csv` to add new tokens:
```csv
Coin symbol,Common Name,CG API ID
NEWTOKEN,New Token Name,coingecko-id
```

**Important**: The symbol must be listed on Binance (Spot or Futures) as `{SYMBOL}USDT`

### Adjusting HV Windows
For different trading strategies:
- **Scalping**: 1,2,3,5
- **Day Trading**: 3,5,7,14
- **Swing Trading**: 7,14,30,60
- **Position Trading**: 30,60,90,180

### Performance Optimization
- Reduce date range for faster loading
- Limit assets to 3-4 for smoother performance
- Data is cached for 10 minutes (600 seconds)

## ⚠️ Important Notes

### Data Source
- **Binance API**: All price and volume data
- **Spot**: `api.binance.com`
- **Futures**: `fapi.binance.com`

### Limitations
- Requires active internet connection
- Data availability depends on Binance listing history
- Some assets may not have perpetual futures
- Historical data limited to what Binance provides

### API Rate Limits
The app uses Streamlit's caching to minimize API calls:
- Price data cached for 60 seconds
- OHLCV data cached for 600 seconds (10 minutes)

If you hit rate limits, wait a few minutes before refreshing.

## 🐛 Troubleshooting

### "No data available for {SYMBOL}"
- Verify the asset is listed on Binance
- Check if it's available in the selected market (Spot vs Perps)
- Try a shorter date range
- Some newer assets don't have long price history

### "Insufficient data to calculate volatility"
- Increase your date range
- Reduce the maximum HV window size
- The asset needs at least `max(windows) + 1` days of data

### Charts not loading
- Check your internet connection
- Refresh the browser
- Clear Streamlit cache (rerun the app)

### Download button not working
- Ensure the asset has processed data
- Try a different browser
- Check browser download settings/permissions

## 📞 Support

For issues specific to:
- **Binance data**: Check [Binance API Status](https://www.binance.com/en/support/announcement)
- **Streamlit**: See [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

This tool is provided as-is for market making and trading analysis purposes.

**Disclaimer**: This tool is for informational purposes only. Theoretical prices are estimates and should not be used as the sole basis for trading decisions. Always verify calculations independently and understand the risks involved in trading derivatives.

---

**Version**: 2.0  
**Last Updated**: January 2025  
**Optimized for**: Professional market makers and volatility traders
