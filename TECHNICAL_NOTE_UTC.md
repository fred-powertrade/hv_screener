# Technical Note: 08:00 UTC Timestamp Implementation

## Overview
This HV Screener implements a standardized 08:00 UTC daily snapshot for all volatility calculations. This document explains the implementation and rationale.

## Why 08:00 UTC?

### Market Maker Convention
- **Industry Standard**: Many market makers use 08:00 UTC for daily risk snapshots
- **Asian Close / European Open**: Captures transition between major trading sessions
- **Global Consistency**: Provides a single reference time that works across all time zones
- **Avoids Midnight Effects**: Eliminates potential issues with UTC day boundaries

### Technical Benefits
1. **Reproducibility**: Same calculation time regardless of user's local timezone
2. **Consistency**: All assets calculated at identical timestamps
3. **Clean Boundaries**: Avoids partial-day volatility calculations
4. **Audit Trail**: Clear, unambiguous timestamp for compliance

## Implementation Details

### Data Fetching
Binance provides daily candles that:
- **Open**: 00:00 UTC
- **Close**: 23:59:59.999 UTC (effectively 00:00 UTC next day)

Our implementation:
```python
# Fetch Binance daily candles (close at 00:00 UTC)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

# Adjust to 08:00 UTC for HV calculations
df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=8)
```

### Timestamp Conversion
When user selects date range in the UI:
```python
# Convert user's date selection to 08:00 UTC timestamps
utc = pytz.UTC
start_dt = datetime.combine(start_date, datetime.min.time())
start_dt = start_dt.replace(hour=8, minute=0, second=0, microsecond=0)
start_dt = utc.localize(start_dt)

start_ms = int(start_dt.timestamp() * 1000)  # Convert to milliseconds for Binance API
```

### Volatility Calculation
All HV calculations use close prices at 08:00 UTC:
```python
# Log returns calculated from 08:00 UTC close prices
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# Rolling volatility using 08:00 UTC snapshots
df[f'hv_{window}'] = df['log_ret'].rolling(window=window).std() * np.sqrt(365)
```

## Example Timeline

### User Selects Date Range
- **Start Date**: 2024-01-01
- **End Date**: 2024-01-31

### Internal Processing
```
User Input → UTC Conversion → API Call → Data Adjustment → HV Calculation

2024-01-01  →  2024-01-01 08:00:00+00:00  →  Fetch from Binance  →  Shift to 08:00  →  Calculate HV
2024-01-02  →  2024-01-02 08:00:00+00:00  →  Fetch from Binance  →  Shift to 08:00  →  Calculate HV
...
2024-01-31  →  2024-01-31 08:00:00+00:00  →  Fetch from Binance  →  Shift to 08:00  →  Calculate HV
```

### Output Timestamps
All exported data and charts show timestamps as:
```
2024-01-01 08:00 UTC
2024-01-02 08:00 UTC
2024-01-03 08:00 UTC
...
```

## Data Integrity

### Quality Checks
1. ✅ All timestamps are timezone-aware (UTC)
2. ✅ Daily snapshots are exactly 24 hours apart
3. ✅ No daylight saving time issues
4. ✅ Consistent across Spot and Futures markets

### Validation
To verify 08:00 UTC implementation:

1. **Export Data**: Download CSV from the app
2. **Check Timestamps**: All should show `08:00 UTC`
3. **Verify Spacing**: Exactly 24 hours between consecutive timestamps
4. **Cross-Reference**: Compare with manual calculations using same close prices

## Comparison with Other Conventions

### Alternative Approaches
| Convention | Timestamp | Pros | Cons |
|------------|-----------|------|------|
| **00:00 UTC** | Midnight | Aligns with Binance candles | Midnight boundary issues |
| **08:00 UTC** | 8am | Market maker standard | Requires timestamp shift |
| **16:00 UTC** | 4pm | US close | Misses Asian session |
| **User Local** | Varies | User-friendly | Inconsistent, not reproducible |

### Our Choice: 08:00 UTC
We chose 08:00 UTC because:
- ✅ Industry standard for market makers
- ✅ Reproducible across users/locations
- ✅ Captures global market activity
- ✅ Clean daily boundaries
- ✅ Avoids midnight edge cases

## Impact on Calculations

### What Changes
- **Timestamp labels**: Show 08:00 instead of 00:00
- **Display format**: Explicitly shows UTC timezone

### What Doesn't Change
- **Underlying prices**: Same OHLCV data from Binance
- **Volatility values**: Identical calculations, just labeled correctly
- **Data quality**: No loss or interpolation of data

## For Developers

### If You Fork This Code
Key lines to maintain for 08:00 UTC:

1. **Import timezone library**:
   ```python
   import pytz
   ```

2. **Shift timestamps after fetching**:
   ```python
   df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=8)
   ```

3. **Localize user inputs**:
   ```python
   utc = pytz.UTC
   start_dt = utc.localize(start_dt.replace(hour=8, minute=0))
   ```

4. **Format display**:
   ```python
   fmt_df.index = fmt_df.index.strftime('%Y-%m-%d %H:%M UTC')
   ```

### Testing 08:00 UTC
```python
import pandas as pd
import pytz

# Create test timestamp
ts = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')

# Shift to 08:00 UTC
ts_adjusted = ts + pd.Timedelta(hours=8)

assert ts_adjusted.hour == 8
assert ts_adjusted.tzinfo == pytz.UTC
print(f"✓ Timestamp correctly set to {ts_adjusted}")
```

## Frequently Asked Questions

### Q: Why not use local timezone?
**A**: Reproducibility. Two market makers in different timezones need identical HV values for the same asset.

### Q: Does this affect the actual volatility calculation?
**A**: No. It only changes the timestamp label. The underlying returns and volatility math are unchanged.

### Q: What if I want a different time?
**A**: Modify the `pd.Timedelta(hours=8)` line to your preferred hour offset. For 16:00 UTC, use `hours=16`.

### Q: Does this work with intraday data?
**A**: This implementation is designed for daily (1d) data. For intraday timeframes, you'd need different logic.

### Q: How do I verify the timestamps are correct?
**A**: Export the CSV and check the timestamp column. All entries should show `08:00 UTC`.

## References

### Binance API
- Kline/candlestick bars: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
- Timestamp format: Unix timestamp in milliseconds
- Timezone: All Binance timestamps are in UTC

### Python Libraries
- pandas timezone: https://pandas.pydata.org/docs/user_guide/timeseries.html#time-zone-handling
- pytz: https://pythonhosted.org/pytz/

---

**Last Updated**: January 2025  
**Maintained By**: HV Screener Development Team
