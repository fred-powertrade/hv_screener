"""
PowerTrade Risk Management Dashboard - Mission Control
======================================================
Real-time monitoring dashboard for crypto financial operations.
Data persisted in Supabase for continuity across sessions.
"""

import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set
from collections import deque

import streamlit as st
import pandas as pd

# =============================================================================
# SUPABASE CLIENT
# =============================================================================
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

@st.cache_resource
def get_supabase_client() -> Optional[Any]:
    """Initialize Supabase client"""
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        
        if not url or not key:
            return None
        
        # Validate URL format
        if not url.startswith("https://") or "supabase.co" not in url:
            st.error(f"Invalid SUPABASE_URL format. Should be: https://xxxxx.supabase.co")
            return None
        
        # Validate key format (should start with eyJ)
        if not key.startswith("eyJ"):
            st.error("Invalid SUPABASE_KEY format. Should start with 'eyJ...'")
            return None
        
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Supabase connection error: {e}")
        return None

# =============================================================================
# TIMEZONE CONFIGURATION
# =============================================================================
UTC = timezone.utc

def utc_now() -> datetime:
    return datetime.now(UTC)

def make_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt

def to_utc_datetime(obj, end_of_day: bool = False) -> datetime:
    if obj is None:
        return None
    if isinstance(obj, datetime):
        dt = make_aware(obj)
        if end_of_day:
            return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return dt
    try:
        year, month, day = obj.year, obj.month, obj.day
        if end_of_day:
            return datetime(year, month, day, 23, 59, 59, 999999, tzinfo=UTC)
        return datetime(year, month, day, 0, 0, 0, 0, tzinfo=UTC)
    except AttributeError:
        return utc_now()

def get_expiry_time(dt: datetime) -> datetime:
    aware_dt = make_aware(dt)
    return datetime(aware_dt.year, aware_dt.month, aware_dt.day, 8, 0, 0, 0, tzinfo=UTC)

def get_current_expiry_period():
    now = utc_now()
    today_expiry = get_expiry_time(now)
    if now < today_expiry:
        return today_expiry - timedelta(days=1), today_expiry
    return today_expiry, today_expiry + timedelta(days=1)

def format_date(obj) -> str:
    try:
        return obj.strftime('%Y-%m-%d') if hasattr(obj, 'strftime') else str(obj)
    except:
        return str(obj)

# =============================================================================
# SLACK SDK
# =============================================================================
try:
    from slack_sdk import WebClient
    from slack_sdk.socket_mode import SocketModeClient
    from slack_sdk.socket_mode.response import SocketModeResponse
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

# =============================================================================
# ACCOUNT CLASSIFICATION
# =============================================================================

# Institutional accounts
INSTITUTIONAL_ACCOUNTS = {"5687"}

# Market maker classification
MM_ACCOUNTS = {
    "ericma988@gmail.com": {"tag": "🤖 MM", "name": "MM"},
    "deltix-mm@power.trade": {"tag": "🔧 Deltix", "name": "Deltix-MM"},
    "trading@zorus.capital": {"tag": "🏢 MM-Int", "name": "MM Internal"},
}
MM_ACCOUNT_IDS = {"18613": {"tag": "🏢 MM-Int", "name": "MM Internal"}}

def get_account_tag(account_id: str = None, email: str = None) -> str:
    """Get account classification tag"""
    if account_id and account_id in INSTITUTIONAL_ACCOUNTS:
        return "🏛️ Inst"
    if account_id and account_id in MM_ACCOUNT_IDS:
        return MM_ACCOUNT_IDS[account_id]["tag"]
    if email:
        email_lower = email.lower()
        for mm_email, info in MM_ACCOUNTS.items():
            if mm_email in email_lower:
                return info["tag"]
    if account_id and account_id.startswith("dex+"):
        return "🌐 DEX"
    return "👤 Retail"

def is_mm_trade(account_id: str = None, email: str = None) -> bool:
    """Check if trade involves market maker"""
    if account_id and account_id in MM_ACCOUNT_IDS:
        return True
    if email:
        email_lower = email.lower()
        for mm_email in MM_ACCOUNTS:
            if mm_email in email_lower:
                return True
    return False

def is_institutional(account_id: str) -> bool:
    return account_id in INSTITUTIONAL_ACCOUNTS if account_id else False

# =============================================================================
# CONFIGURATION
# =============================================================================

def get_channel_config():
    try:
        return {
            "withdrawals": st.secrets.get("CHANNEL_WITHDRAWALS", ""),
            "trades": st.secrets.get("CHANNEL_TRADES", ""),
            "dex_api": st.secrets.get("CHANNEL_DEX_API", ""),
            "liquidations": st.secrets.get("CHANNEL_LIQUIDATIONS", ""),
            "institutional": st.secrets.get("CHANNEL_INSTITUTIONAL", ""),
        }
    except:
        return {k: "" for k in ["withdrawals", "trades", "dex_api", "liquidations", "institutional"]}

# =============================================================================
# INSTRUMENT CLASSIFICATION
# =============================================================================

def classify_instrument(symbol: str) -> str:
    if not symbol:
        return "UNKNOWN"
    symbol_upper = symbol.upper()
    
    # Options - must end with C or P
    if symbol_upper.endswith('C') or symbol_upper.endswith('P'):
        # 10+ digit date = hourly/10-min expiry = DEGEN options
        if re.search(r'-\d{10,}', symbol_upper):
            return "DEGEN"
        # 8-digit date = daily expiry = regular OPTIONS
        elif re.search(r'-\d{8}', symbol_upper):
            return "OPTIONS"
    
    # Perpetual: PAXG-USD-PERPETUAL, LDO1000-USD-PERPETUAL
    if 'PERPETUAL' in symbol_upper or 'PERP' in symbol_upper:
        return "PERPETUAL"
    
    # Spot: BTC-USD, ETH-USDC (exactly two parts, letters only)
    if re.match(r'^[A-Z]+-[A-Z]+$', symbol_upper):
        return "SPOT"
    
    return "OTHER"

# =============================================================================
# EMOJI MAPPING
# =============================================================================
EVENT_EMOJIS = {
    "withdrawal_completed": "🔻",
    "withdrawal_requested": "⚠️",
    "deposit": "💰",
    "trade": "💵",
    "liquidation": "🚨",
    "institutional_alert": "🔔",
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ParsedEvent:
    timestamp: datetime
    channel_type: str
    event_type: str
    account_id: Optional[str] = None
    amount: Optional[float] = None
    asset: Optional[str] = None
    symbol: Optional[str] = None
    instrument_type: Optional[str] = None
    size: Optional[float] = None
    price: Optional[float] = None
    value: Optional[float] = None
    email: Optional[str] = None
    raw_message: str = ""
    rfq_details: Optional[str] = None
    is_high_priority: bool = False
    is_market_maker: bool = False
    account_tag: str = ""
    side: Optional[str] = None  # BUY/SELL
    pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        emoji = EVENT_EMOJIS.get(self.event_type, "📝")
        return {
            "Timestamp (UTC)": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "": emoji,  # Emoji column
            "Type": self.event_type.replace("_", " ").title(),
            "Tag": self.account_tag or get_account_tag(self.account_id, self.email),
            "Account ID": self.account_id or "-",
            "Amount": self.amount,
            "Asset": self.asset or "-",
            "Symbol": self.symbol or "-",
            "Instrument": self.instrument_type or "-",
            "Size": self.size,
            "Price": self.price,
            "Value (USD)": self.value,
            "Side": self.side or "-",
            "Email": self.email or "-",
        }


# =============================================================================
# MESSAGE PARSERS
# =============================================================================

class MessageParser:
    # Withdrawals - handle both completed and requested
    WITHDRAWAL_COMPLETED = re.compile(
        r"Account\s+ID\s+(\d+)\s+completed\s+a\s+withdrawal\s+for\s+([\d.,]+)\s+(\w+)",
        re.IGNORECASE
    )
    WITHDRAWAL_REQUESTED = re.compile(
        r"Account\s+ID\s+(\d+)\s+requested\s+a\s+withdrawal\s+for\s+([\d.,]+)\s+(\w+)",
        re.IGNORECASE
    )
    
    # Trades - flexible pattern for ALL formats:
    # *PAXG-USD-PERPETUAL 0.01 @5083.2*　Value: 50.759181  (with asterisks, perpetual)
    # PAXG-20260213-4900.00P 10 @57.01　Value: 50811.325   (no asterisks, options)
    # AXS-20260118-1.2000C 8000 @0.0122　Value: 8479.2     (no asterisks, options small strike)
    TRADE_PATTERN = re.compile(
        r"\*?([A-Z][A-Z0-9]*(?:-[A-Z0-9.]+)+[CP]?)\s+([\d.]+)\s+@([\d.]+)\*?\s+Value:\s*([\d.]+)",
        re.IGNORECASE
    )
    TRADE_ACCOUNT = re.compile(
        r"Account:\s*(\d+)\s+(?:<mailto:[^|]+\|)?([^\s<>\n]+)",
        re.IGNORECASE
    )
    TRADE_SIDE = re.compile(r"(Buyer|Seller)\s+(TAKER|MAKER)", re.IGNORECASE)
    
    # Deposits
    DEPOSIT_PATTERN = re.compile(
        r"Account\s+(\d+)\s+deposited\s+`([\d.,]+)\s+(\w+)`",
        re.IGNORECASE
    )
    
    # Liquidations
    LIQUIDATION_PATTERN = re.compile(r"Liquidation\s+(\d+):", re.IGNORECASE)
    LIQUIDATION_SUMMARY = re.compile(r"Liquidation\s+(\d+):\s*(-?\d+)\s*balances.*?After:\s*`(-?[\d.,]+)\s*USD`", re.IGNORECASE | re.DOTALL)

    @classmethod
    def parse_withdrawal(cls, message: str, timestamp: datetime) -> Optional[ParsedEvent]:
        match = cls.WITHDRAWAL_COMPLETED.search(message)
        event_subtype = "completed"
        
        if not match:
            match = cls.WITHDRAWAL_REQUESTED.search(message)
            event_subtype = "requested"
        
        if match:
            amount_str = match.group(2).replace(",", "")
            asset = match.group(3).upper()
            account_id = match.group(1)
            return ParsedEvent(
                timestamp=make_aware(timestamp),
                channel_type="withdrawals",
                event_type=f"withdrawal_{event_subtype}",
                account_id=account_id,
                amount=float(amount_str),
                asset=asset,
                value=float(amount_str) if asset in ["USDC", "USDT", "USD"] else None,
                raw_message=message[:300],
                account_tag=get_account_tag(account_id),
            )
        return None

    @classmethod
    def parse_trade(cls, message: str, timestamp: datetime) -> Optional[ParsedEvent]:
        trade_match = cls.TRADE_PATTERN.search(message)
        if not trade_match:
            return None
        
        symbol = trade_match.group(1).rstrip('*')
        size = float(trade_match.group(2))
        price = float(trade_match.group(3))
        value = float(trade_match.group(4))
        
        # Find all accounts in the trade
        account_matches = cls.TRADE_ACCOUNT.findall(message)
        account_ids = [m[0] for m in account_matches]
        emails = [m[1].lower().rstrip('>').strip() for m in account_matches]
        
        primary_account = account_ids[0] if account_ids else None
        primary_email = emails[0] if emails else None
        
        # Determine side
        side_matches = cls.TRADE_SIDE.findall(message)
        side = side_matches[0][0].upper() if side_matches else None
        
        # Check for MM and institutional
        is_mm = is_mm_trade(primary_account, ", ".join(emails))
        is_inst = any(acc in INSTITUTIONAL_ACCOUNTS for acc in account_ids)
        
        instrument_type = classify_instrument(symbol)
        
        return ParsedEvent(
            timestamp=make_aware(timestamp),
            channel_type="trades",
            event_type="trade",
            symbol=symbol,
            instrument_type=instrument_type,
            size=size,
            price=price,
            value=value,
            account_id=primary_account,
            email=", ".join(emails) if emails else None,
            raw_message=message[:300],
            is_market_maker=is_mm,
            account_tag=get_account_tag(primary_account, primary_email),
            side=side,
            is_high_priority=is_inst or value > 50000,  # Flag big trades & institutional
        )

    @classmethod
    def parse_dex_api(cls, message: str, timestamp: datetime) -> Optional[ParsedEvent]:
        deposit_match = cls.DEPOSIT_PATTERN.search(message)
        if deposit_match:
            amount_str = deposit_match.group(2).replace(",", "")
            asset = deposit_match.group(3).upper()
            account_id = deposit_match.group(1)
            return ParsedEvent(
                timestamp=make_aware(timestamp),
                channel_type="dex_api",
                event_type="deposit",
                account_id=account_id,
                amount=float(amount_str),
                asset=asset,
                value=float(amount_str) if asset in ["USDC", "USDT", "USD"] else None,
                raw_message=message[:300],
                account_tag=get_account_tag(account_id),
            )
        return None

    @classmethod
    def parse_liquidation(cls, message: str, timestamp: datetime) -> Optional[ParsedEvent]:
        # Try to get summary info
        summary_match = cls.LIQUIDATION_SUMMARY.search(message)
        if summary_match:
            account_id = summary_match.group(1)
            loss_str = summary_match.group(3).replace(",", "")
            return ParsedEvent(
                timestamp=make_aware(timestamp),
                channel_type="liquidations",
                event_type="liquidation",
                account_id=account_id,
                value=abs(float(loss_str)),
                raw_message=message[:300],
                is_high_priority=True,
                account_tag=get_account_tag(account_id),
            )
        
        # Fallback to basic pattern
        match = cls.LIQUIDATION_PATTERN.search(message)
        if match and "balances" in message.lower():
            account_id = match.group(1)
            return ParsedEvent(
                timestamp=make_aware(timestamp),
                channel_type="liquidations",
                event_type="liquidation",
                account_id=account_id,
                raw_message=message[:300],
                is_high_priority=True,
                account_tag=get_account_tag(account_id),
            )
        return None

    @classmethod
    def parse_institutional(cls, message: str, timestamp: datetime) -> Optional[ParsedEvent]:
        if "Trade Notifications" in message or "Inst Accounts" in message:
            return ParsedEvent(
                timestamp=make_aware(timestamp),
                channel_type="institutional",
                event_type="institutional_alert",
                raw_message=message[:300],
                is_high_priority=True,
            )
        return None


# =============================================================================
# DATA STORE WITH SUPABASE PERSISTENCE
# =============================================================================

class ThreadSafeDataStore:
    def __init__(self, max_events: int = 500000):
        self._lock = threading.Lock()
        self._events: deque = deque(maxlen=max_events)
        self._last_update = utc_now()
        self._history_loaded = False
        self._last_event_timestamp: Optional[datetime] = None
        self._supabase = get_supabase_client()
        self._pending_events: List[ParsedEvent] = []
        # Alert tracking
        self._recent_alerts: deque = deque(maxlen=50)
        self._last_alert_check = utc_now()

    def get_recent_alerts(self, minutes: int = 5) -> List[ParsedEvent]:
        """Get high priority events from last N minutes"""
        cutoff = utc_now() - timedelta(minutes=minutes)
        with self._lock:
            return [e for e in self._events 
                    if e.is_high_priority and make_aware(e.timestamp) >= cutoff]

    def get_whale_trades(self, minutes: int = 5, threshold: float = 10000) -> List[ParsedEvent]:
        """Get large trades from last N minutes"""
        cutoff = utc_now() - timedelta(minutes=minutes)
        with self._lock:
            return [e for e in self._events 
                    if e.event_type == "trade" 
                    and (e.value or 0) >= threshold
                    and make_aware(e.timestamp) >= cutoff]

    def get_institutional_trades(self, minutes: int = 5) -> List[ParsedEvent]:
        """Get institutional trades from last N minutes"""
        cutoff = utc_now() - timedelta(minutes=minutes)
        with self._lock:
            return [e for e in self._events 
                    if e.event_type == "trade"
                    and e.account_id in INSTITUTIONAL_ACCOUNTS
                    and make_aware(e.timestamp) >= cutoff]

    def _event_to_db_row(self, event: ParsedEvent) -> dict:
        """Convert ParsedEvent to database row"""
        return {
            "timestamp": event.timestamp.isoformat(),
            "channel_type": event.channel_type,
            "event_type": event.event_type,
            "account_id": event.account_id,
            "amount": float(event.amount) if event.amount else None,
            "asset": event.asset,
            "symbol": event.symbol,
            "instrument_type": event.instrument_type,
            "size": float(event.size) if event.size else None,
            "price": float(event.price) if event.price else None,
            "value": float(event.value) if event.value else None,
            "email": event.email,
            "raw_message": event.raw_message[:500] if event.raw_message else None,
            "is_market_maker": event.is_market_maker,
        }

    def _db_row_to_event(self, row: dict) -> ParsedEvent:
        """Convert database row to ParsedEvent"""
        ts = row.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        
        account_id = row.get("account_id")
        email = row.get("email")
        
        return ParsedEvent(
            timestamp=make_aware(ts) if ts else utc_now(),
            channel_type=row.get("channel_type", ""),
            event_type=row.get("event_type", ""),
            account_id=account_id,
            amount=float(row["amount"]) if row.get("amount") else None,
            asset=row.get("asset"),
            symbol=row.get("symbol"),
            instrument_type=row.get("instrument_type"),
            size=float(row["size"]) if row.get("size") else None,
            price=float(row["price"]) if row.get("price") else None,
            value=float(row["value"]) if row.get("value") else None,
            email=email,
            raw_message=row.get("raw_message", ""),
            is_market_maker=row.get("is_market_maker", False),
            account_tag=get_account_tag(account_id, email),
        )

    def save_to_supabase(self, event: ParsedEvent):
        """Save single event to Supabase"""
        if not self._supabase:
            return
        try:
            self._supabase.table("events").upsert(
                self._event_to_db_row(event),
                on_conflict="timestamp,channel_type,account_id,event_type,value"
            ).execute()
        except Exception as e:
            # Silently handle duplicates
            if "duplicate" not in str(e).lower():
                st.warning(f"DB save error: {e}")

    def save_batch_to_supabase(self, events: List[ParsedEvent]):
        """Save multiple events to Supabase in batches"""
        if not self._supabase or not events:
            return
        
        rows = [self._event_to_db_row(e) for e in events]
        
        # Insert in batches of 500
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            try:
                self._supabase.table("events").upsert(
                    batch,
                    on_conflict="timestamp,channel_type,account_id,event_type,value"
                ).execute()
            except Exception as e:
                if "duplicate" not in str(e).lower():
                    st.warning(f"Batch save error: {e}")

    def load_from_supabase(self, days: int = 30) -> int:
        """Load events from Supabase"""
        if not self._supabase:
            return 0
        
        try:
            cutoff = (utc_now() - timedelta(days=days)).isoformat()
            
            # Fetch in pages
            all_rows = []
            page_size = 1000
            offset = 0
            
            while True:
                response = self._supabase.table("events")\
                    .select("*")\
                    .gte("timestamp", cutoff)\
                    .order("timestamp", desc=False)\
                    .range(offset, offset + page_size - 1)\
                    .execute()
                
                rows = response.data
                if not rows:
                    break
                
                all_rows.extend(rows)
                offset += page_size
                
                if len(rows) < page_size:
                    break
            
            # Convert to events
            events = [self._db_row_to_event(row) for row in all_rows]
            
            with self._lock:
                self._events.clear()
                for event in events:
                    self._events.append(event)
                    if not self._last_event_timestamp or event.timestamp > self._last_event_timestamp:
                        self._last_event_timestamp = event.timestamp
                self._history_loaded = True
            
            return len(events)
        
        except Exception as e:
            st.error(f"Error loading from Supabase: {e}")
            return 0

    def add_event(self, event: ParsedEvent):
        with self._lock:
            self._events.append(event)
            self._last_update = utc_now()
            if not self._last_event_timestamp or event.timestamp > self._last_event_timestamp:
                self._last_event_timestamp = event.timestamp
        
        # Save to Supabase
        self.save_to_supabase(event)

    def add_events_bulk(self, events: List[ParsedEvent]):
        with self._lock:
            for event in events:
                self._events.append(event)
                if not self._last_event_timestamp or event.timestamp > self._last_event_timestamp:
                    self._last_event_timestamp = event.timestamp
            self._last_update = utc_now()
        
        # Save to Supabase
        self.save_batch_to_supabase(events)

    def get_last_event_timestamp(self) -> Optional[datetime]:
        with self._lock:
            return self._last_event_timestamp

    def get_last_event_timestamp_from_db(self) -> Optional[datetime]:
        """Get the most recent event timestamp from Supabase"""
        if not self._supabase:
            return None
        try:
            response = self._supabase.table("events")\
                .select("timestamp")\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()
            
            if response.data:
                ts = response.data[0]["timestamp"]
                if isinstance(ts, str):
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return None
        except:
            return None

    def load_from_csv(self, csv_content: str) -> int:
        """Load from CSV and save to Supabase"""
        try:
            df = pd.read_csv(pd.io.common.StringIO(csv_content))
            events = []
            for _, row in df.iterrows():
                try:
                    ts = pd.to_datetime(row['timestamp']).replace(tzinfo=UTC)
                    account_id = str(row.get('account_id', '')) if pd.notna(row.get('account_id')) else None
                    email = row.get('email') if pd.notna(row.get('email')) else None
                    event = ParsedEvent(
                        timestamp=ts,
                        channel_type=row.get('channel_type', ''),
                        event_type=row.get('event_type', ''),
                        account_id=account_id,
                        amount=float(row['amount']) if pd.notna(row.get('amount')) else None,
                        asset=row.get('asset') if pd.notna(row.get('asset')) else None,
                        symbol=row.get('symbol') if pd.notna(row.get('symbol')) else None,
                        instrument_type=row.get('instrument_type') if pd.notna(row.get('instrument_type')) else None,
                        size=float(row['size']) if pd.notna(row.get('size')) else None,
                        price=float(row['price']) if pd.notna(row.get('price')) else None,
                        value=float(row['value']) if pd.notna(row.get('value')) else None,
                        email=email,
                        raw_message=str(row.get('raw_message', ''))[:200] if pd.notna(row.get('raw_message')) else '',
                        is_market_maker=bool(row.get('is_market_maker', False)),
                        account_tag=get_account_tag(account_id, email),
                    )
                    events.append(event)
                except:
                    continue
            
            events.sort(key=lambda e: e.timestamp)
            self.add_events_bulk(events)  # This also saves to Supabase
            self._history_loaded = True
            return len(events)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return 0

    def set_history_loaded(self):
        with self._lock:
            self._history_loaded = True

    def is_history_loaded(self) -> bool:
        with self._lock:
            return self._history_loaded

    def get_events(self, start_date=None, end_date=None, account_filter: Set[str] = None, 
                   excluded_accounts: Set[str] = None, show_mm: bool = True) -> List[ParsedEvent]:
        with self._lock:
            events = list(self._events)
        
        if account_filter:
            events = [e for e in events if e.account_id in account_filter]
        if excluded_accounts:
            events = [e for e in events if e.account_id not in excluded_accounts]
        if not show_mm:
            events = [e for e in events if not e.is_market_maker]
        
        if start_date:
            try:
                start_dt = to_utc_datetime(start_date)
                events = [e for e in events if make_aware(e.timestamp) >= start_dt]
            except:
                pass
        
        if end_date:
            try:
                end_dt = to_utc_datetime(end_date, end_of_day=True)
                events = [e for e in events if make_aware(e.timestamp) <= end_dt]
            except:
                pass
        
        return events

    def get_events_last_24h(self) -> List[ParsedEvent]:
        cutoff = utc_now() - timedelta(hours=24)
        with self._lock:
            return [e for e in self._events if make_aware(e.timestamp) >= cutoff]

    def get_unique_accounts_24h(self) -> Set[str]:
        events = self.get_events_last_24h()
        return {e.account_id for e in events if e.account_id}

    def get_deposits_24h(self) -> List[ParsedEvent]:
        events = self.get_events_last_24h()
        return [e for e in events if e.event_type == "deposit"]

    def get_all_account_ids(self) -> List[str]:
        with self._lock:
            ids = {e.account_id for e in self._events if e.account_id}
            return sorted(list(ids))

    def get_last_update(self) -> datetime:
        with self._lock:
            return self._last_update


@st.cache_resource
def get_data_store_v5():
    return ThreadSafeDataStore()

def get_data_store():
    return get_data_store_v5()


# =============================================================================
# SLACK FUNCTIONS
# =============================================================================

def fetch_channel_history(client, channel_id: str, days: int = 7) -> List[dict]:
    messages = []
    oldest = (utc_now() - timedelta(days=days)).timestamp()
    cursor = None
    
    while True:
        try:
            response = client.conversations_history(
                channel=channel_id, oldest=str(oldest), limit=200, cursor=cursor
            )
            messages.extend(response.get("messages", []))
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(0.3)
        except Exception as e:
            st.warning(f"Error fetching {channel_id}: {e}")
            break
    return messages


def load_historical_data(client, channel_config: dict, data_store: ThreadSafeDataStore, 
                         days: int = 7, status_container=None):
    all_events = []
    last_ts = data_store.get_last_event_timestamp()
    channel_id_to_type = {v: k for k, v in channel_config.items() if v}
    total = len(channel_id_to_type)
    
    if status_container:
        progress = status_container.progress(0, text="Starting...")
    
    for idx, (channel_id, channel_type) in enumerate(channel_id_to_type.items()):
        if status_container:
            progress.progress(idx / total, text=f"Fetching {channel_type}...")
        
        messages = fetch_channel_history(client, channel_id, days)
        
        parsed = 0
        for msg in messages:
            if msg.get("subtype") == "channel_join":
                continue
            
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            if not text.strip():
                continue
            
            try:
                timestamp = datetime.fromtimestamp(float(ts), tz=UTC)
            except:
                continue
            
            if last_ts and timestamp <= last_ts:
                continue
            
            event = None
            if channel_type == "withdrawals":
                event = MessageParser.parse_withdrawal(text, timestamp)
            elif channel_type == "trades":
                event = MessageParser.parse_trade(text, timestamp)
            elif channel_type == "dex_api":
                event = MessageParser.parse_dex_api(text, timestamp)
            elif channel_type == "liquidations":
                event = MessageParser.parse_liquidation(text, timestamp)
            elif channel_type == "institutional":
                event = MessageParser.parse_institutional(text, timestamp)
            
            if event:
                all_events.append(event)
                parsed += 1
        
        if status_container:
            status_container.caption(f"✓ {channel_type}: {parsed} events")
    
    if status_container:
        progress.progress(1.0, text="Done!")
    
    if all_events:
        all_events.sort(key=lambda e: e.timestamp)
        data_store.add_events_bulk(all_events)
    data_store.set_history_loaded()
    
    if status_container:
        status_container.success(f"✅ Added {len(all_events)} new events")


class SlackListener:
    def __init__(self, app_token: str, bot_token: str, channel_config: dict):
        self.channel_id_to_type = {v: k for k, v in channel_config.items() if v}
        self.client = WebClient(token=bot_token)
        self.socket_client = SocketModeClient(app_token=app_token, web_client=self.client)
        self.data_store = get_data_store()
        self._running = False

    def _process_message(self, channel_id: str, text: str, ts: str):
        try:
            timestamp = datetime.fromtimestamp(float(ts), tz=UTC)
        except:
            timestamp = utc_now()

        channel_type = self.channel_id_to_type.get(channel_id)
        if not channel_type:
            return

        event = None
        if channel_type == "withdrawals":
            event = MessageParser.parse_withdrawal(text, timestamp)
        elif channel_type == "trades":
            event = MessageParser.parse_trade(text, timestamp)
        elif channel_type == "dex_api":
            event = MessageParser.parse_dex_api(text, timestamp)
        elif channel_type == "liquidations":
            event = MessageParser.parse_liquidation(text, timestamp)
        elif channel_type == "institutional":
            event = MessageParser.parse_institutional(text, timestamp)

        if event:
            self.data_store.add_event(event)

    def _handler(self, client, req):
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        if req.type == "events_api":
            event = req.payload.get("event", {})
            if event.get("type") == "message":
                subtype = event.get("subtype", "")
                if subtype != "channel_join":
                    self._process_message(event.get("channel", ""), event.get("text", ""), event.get("ts", ""))

    def start(self):
        if self._running:
            return
        self._running = True
        self.socket_client.socket_mode_request_listeners.append(self._handler)
        
        def run():
            self.socket_client.connect()
            while self._running:
                time.sleep(1)
        
        threading.Thread(target=run, daemon=True).start()


# =============================================================================
# STATISTICS
# =============================================================================

def calculate_account_stats(events: List[ParsedEvent], account_id: str) -> Dict[str, Any]:
    """Calculate PnL and stats for a specific account"""
    account_events = [e for e in events if e.account_id == account_id]
    
    deposits = sum(e.value or 0 for e in account_events if e.event_type == "deposit")
    withdrawals = sum(e.value or 0 for e in account_events if "withdrawal" in e.event_type)
    
    trades = [e for e in account_events if e.event_type == "trade"]
    trade_volume = sum(e.value or 0 for e in trades)
    
    # Simple PnL: deposits - withdrawals (actual PnL would need position tracking)
    net_flow = deposits - withdrawals
    
    return {
        "deposits": deposits,
        "withdrawals": withdrawals,
        "net_flow": net_flow,
        "trade_count": len(trades),
        "trade_volume": trade_volume,
        "total_events": len(account_events),
    }


def calculate_summary_stats(events: List[ParsedEvent]) -> Dict[str, Any]:
    deposits = [e for e in events if e.event_type == "deposit"]
    # Separate withdrawal types - only count completed for totals
    withdrawals_completed = [e for e in events if e.event_type == "withdrawal_completed"]
    withdrawals_requested = [e for e in events if e.event_type == "withdrawal_requested"]
    trades = [e for e in events if e.event_type == "trade"]
    
    total_deposits = sum(e.value or 0 for e in deposits)
    # Only sum COMPLETED withdrawals
    total_withdrawals = sum(e.value or 0 for e in withdrawals_completed)
    total_trade_volume = sum(e.value or 0 for e in trades)
    
    # Instrument breakdown
    inst_stats = {}
    for e in trades:
        inst = e.instrument_type or "UNKNOWN"
        if inst not in inst_stats:
            inst_stats[inst] = {"count": 0, "volume": 0}
        inst_stats[inst]["count"] += 1
        inst_stats[inst]["volume"] += e.value or 0
    
    return {
        "total_events": len(events),
        "total_deposits": total_deposits,
        "total_withdrawals": total_withdrawals,
        "deposit_count": len(deposits),
        "withdrawal_completed_count": len(withdrawals_completed),
        "withdrawal_requested_count": len(withdrawals_requested),
        "net_flow": total_deposits - total_withdrawals,
        "total_trade_volume": total_trade_volume,
        "trade_count": len(trades),
        "avg_trade_size": total_trade_volume / len(trades) if trades else 0,
        "instrument_stats": inst_stats,
        "unique_accounts": len({e.account_id for e in events if e.account_id}),
        "liquidation_count": sum(1 for e in events if e.event_type == "liquidation"),
        "institutional_count": sum(1 for e in events if e.event_type == "institutional_alert"),
    }


# =============================================================================
# UI COMPONENTS
# =============================================================================

def style_event_row(row: pd.Series) -> List[str]:
    tag = row.get("Tag", "")
    event_type = row.get("Type", "").lower()
    
    # MM highlighting
    if "MM" in tag or "Deltix" in tag:
        return ["background-color: rgba(147, 51, 234, 0.2)"] * len(row)
    # Institutional
    if "Inst" in tag:
        return ["background-color: rgba(255, 215, 0, 0.2)"] * len(row)
    # Event type colors
    if "deposit" in event_type:
        return ["background-color: rgba(0, 255, 0, 0.1)"] * len(row)
    if "withdrawal" in event_type:
        return ["background-color: rgba(255, 0, 0, 0.1)"] * len(row)
    if "trade" in event_type:
        return ["background-color: rgba(0, 100, 255, 0.1)"] * len(row)
    if "liquidation" in event_type:
        return ["background-color: rgba(255, 165, 0, 0.2)"] * len(row)
    return [""] * len(row)


def render_summary(title: str, stats: Dict, period: str):
    st.markdown(f"### {title}")
    st.caption(f"Period: {period}")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("💰 Deposits", f"${stats['total_deposits']:,.2f}")
        st.metric("Count", stats['deposit_count'])
    with c2:
        # Show only completed withdrawals in main metric
        st.metric("🔻 Withdrawals (Completed)", f"${stats['total_withdrawals']:,.2f}")
        st.caption(f"✅ {stats.get('withdrawal_completed_count', 0)} completed | ⏳ {stats.get('withdrawal_requested_count', 0)} pending")
    with c3:
        delta = "Inflow" if stats['net_flow'] >= 0 else "Outflow"
        st.metric("📊 Net Flow", f"${abs(stats['net_flow']):,.2f}", delta=delta,
                  delta_color="normal" if stats['net_flow'] >= 0 else "inverse")
        st.metric("👥 Accounts", stats['unique_accounts'])
    with c4:
        st.metric("💵 Trade Volume", f"${stats['total_trade_volume']:,.2f}")
        st.metric("🔄 Trades", stats['trade_count'])
    
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("🚨 Liquidations", stats['liquidation_count'])
    with c6:
        st.metric("🏛️ Institutional", stats['institutional_count'])
    with c7:
        st.metric("📊 Avg Trade", f"${stats['avg_trade_size']:,.2f}")
    
    # Instrument breakdown
    if stats['instrument_stats']:
        st.markdown("---")
        st.markdown("**📈 Instrument Breakdown**")
        num_cols = min(len(stats['instrument_stats']), 5)  # Max 5 columns
        cols = st.columns(num_cols)
        icons = {"OPTIONS": "🟡", "DEGEN": "🔥", "PERPETUAL": "🔵", "SPOT": "🟢", "OTHER": "⚪"}
        for i, (inst, data) in enumerate(sorted(stats['instrument_stats'].items(), 
                                                  key=lambda x: x[1]["volume"], reverse=True)):
            with cols[i % num_cols]:
                st.markdown(f"**{icons.get(inst, '⚪')} {inst}**")
                st.caption(f"{data['count']} trades | ${data['volume']:,.0f}")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="PowerTrade Risk Dashboard", page_icon="🎯", layout="wide")
    
    st.markdown("""<style>
    .live-counter {background: linear-gradient(90deg, #1e3a5f, #2d5a87); padding: 1rem; 
                   border-radius: 0.5rem; margin-bottom: 1rem;}
    .alert-box {background-color: #ff4444; color: white; padding: 1rem; border-radius: 0.5rem;
                font-weight: bold; animation: pulse 2s infinite;}
    .whale-alert {background: linear-gradient(90deg, #ff6b00, #ff9500); color: white; 
                  padding: 1rem; border-radius: 0.5rem; font-weight: bold; animation: pulse 1.5s infinite;}
    .inst-alert {background: linear-gradient(90deg, #9333ea, #c084fc); color: white;
                 padding: 1rem; border-radius: 0.5rem; font-weight: bold; animation: glow 1s infinite;}
    @keyframes pulse {0%,100%{opacity:1; transform: scale(1);}50%{opacity:0.8; transform: scale(1.02);}}
    @keyframes glow {0%,100%{box-shadow: 0 0 5px #9333ea;}50%{box-shadow: 0 0 20px #c084fc;}}
    </style>""", unsafe_allow_html=True)

    # Session state
    if "listener_started" not in st.session_state:
        st.session_state.listener_started = False
    if "show_mm" not in st.session_state:
        st.session_state.show_mm = True
    if "excluded_accounts" not in st.session_state:
        st.session_state.excluded_accounts = []
    if "selected_accounts" not in st.session_state:
        st.session_state.selected_accounts = []

    data_store = get_data_store()
    now = utc_now()
    today = now.date()

    # === SIDEBAR ===
    with st.sidebar:
        st.title("⚙️ Controls")
        st.markdown(f"🕐 **{now.strftime('%Y-%m-%d %H:%M:%S')} UTC**")
        
        st.divider()
        
        # Database status
        supabase = get_supabase_client()
        if supabase:
            st.success("🗄️ Supabase Connected")
            
            # Auto-load from Supabase on first run
            if "db_loaded" not in st.session_state:
                with st.spinner("Loading from database..."):
                    count = data_store.load_from_supabase(days=60)
                    st.session_state.db_loaded = True
                    if count > 0:
                        st.toast(f"Loaded {count:,} events from database")
                        st.rerun()
        else:
            st.warning("🗄️ Supabase not configured")
            st.caption("Add SUPABASE_URL and SUPABASE_KEY to secrets")
        
        st.divider()
        
        # Data source (CSV backup option)
        st.subheader("📊 Data Source")
        uploaded = st.file_uploader("Upload history CSV", type=['csv'], 
                                     help="Optional: Import from CSV (will also save to database)")
        if uploaded and st.button("📥 Load CSV"):
            count = data_store.load_from_csv(uploaded.getvalue().decode('utf-8'))
            if count > 0:
                st.success(f"✅ Loaded {count:,} events")
                st.rerun()
        
        event_count = len(data_store.get_events())
        last_ts = data_store.get_last_event_timestamp()
        if event_count > 0:
            st.info(f"📦 {event_count:,} events in memory")
            if last_ts:
                st.caption(f"Latest: {last_ts.strftime('%Y-%m-%d %H:%M')} UTC")
        
        # Show DB stats
        if supabase:
            db_last_ts = data_store.get_last_event_timestamp_from_db()
            if db_last_ts:
                st.caption(f"DB latest: {db_last_ts.strftime('%Y-%m-%d %H:%M')} UTC")
        
        st.divider()
        
        if not st.session_state.listener_started:
            if st.button("🔄 Connect Slack", type="primary", use_container_width=True):
                if SLACK_AVAILABLE:
                    try:
                        app_token = st.secrets["SLACK_APP_TOKEN"]
                        bot_token = st.secrets["SLACK_BOT_TOKEN"]
                        channel_config = get_channel_config()
                        client = WebClient(token=bot_token)
                        
                        days = min((utc_now() - last_ts).days + 1, 30) if last_ts else 7
                        status = st.empty()
                        load_historical_data(client, channel_config, data_store, days, status)
                        
                        SlackListener(app_token, bot_token, channel_config).start()
                        st.session_state.listener_started = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.success("🟢 Live Connected")
        
        st.divider()
        
        # Date range
        st.subheader("📅 Date Range")
        start_date = st.date_input("Start", value=today - timedelta(days=7), max_value=today)
        end_date = st.date_input("End", value=today, max_value=today)
        
        st.divider()
        
        # Account filter
        st.subheader("🔍 Account Filter")
        all_accounts = data_store.get_all_account_ids()
        selected = st.multiselect("Show only accounts:", options=all_accounts, 
                                   default=st.session_state.selected_accounts,
                                   help="Leave empty to show all")
        st.session_state.selected_accounts = selected
        
        st.divider()
        
        # MM toggle
        st.session_state.show_mm = st.checkbox("🤖 Show MM trades", value=st.session_state.show_mm)
        
        st.divider()
        
        # Excluded accounts
        st.subheader("🚫 Excluded")
        new_exc = st.text_input("Exclude Account ID")
        if new_exc and st.button("Add"):
            if new_exc not in st.session_state.excluded_accounts:
                st.session_state.excluded_accounts.append(new_exc)
                st.rerun()
        for acc in st.session_state.excluded_accounts:
            c1, c2 = st.columns([3, 1])
            c1.text(acc)
            if c2.button("❌", key=f"rm_{acc}"):
                st.session_state.excluded_accounts.remove(acc)
                st.rerun()

    # === MAIN CONTENT ===
    st.title("🎯 PowerTrade Risk Dashboard")
    
    # Live counters
    events_24h = data_store.get_events_last_24h()
    active_accounts = data_store.get_unique_accounts_24h()
    deposits_24h = data_store.get_deposits_24h()
    
    st.markdown('<div class="live-counter">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🟢 Active Accounts (24h)", len(active_accounts))
    with m2:
        dep_value = sum(e.value or 0 for e in deposits_24h)
        st.metric("💰 Deposits (24h)", f"${dep_value:,.0f}")
        st.caption(f"{len(deposits_24h)} transactions")
    with m3:
        st.metric("📊 Events (24h)", len(events_24h))
    with m4:
        trades_24h = [e for e in events_24h if e.event_type == "trade"]
        st.metric("💵 Trades (24h)", len(trades_24h))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === ALERTS SECTION ===
    whale_trades = data_store.get_whale_trades(minutes=10, threshold=10000)
    inst_trades = data_store.get_institutional_trades(minutes=10)
    
    alert_col1, alert_col2 = st.columns(2)
    
    # Whale alerts
    with alert_col1:
        if whale_trades:
            st.markdown('<div class="whale-alert">🐋 WHALE TRADES DETECTED!</div>', unsafe_allow_html=True)
            for trade in whale_trades[-3:]:
                st.warning(f"🐋 **${trade.value:,.0f}** | {trade.symbol} | Acc: {trade.account_id} | {trade.timestamp.strftime('%H:%M:%S')}")
    
    # Institutional alerts
    with alert_col2:
        if inst_trades:
            st.markdown('<div class="inst-alert">🏛️ INSTITUTIONAL ACTIVITY!</div>', unsafe_allow_html=True)
            for trade in inst_trades[-3:]:
                st.error(f"🏛️ **${trade.value:,.0f}** | {trade.symbol} | Acc: {trade.account_id} | {trade.timestamp.strftime('%H:%M:%S')}")
    
    st.divider()
    
    # Get filtered events
    account_filter = set(selected) if selected else None
    excluded = set(st.session_state.excluded_accounts)
    events = data_store.get_events(start_date, end_date, account_filter, excluded, st.session_state.show_mm)
    
    # Account summary if filtering
    if selected and len(selected) == 1:
        acc_id = selected[0]
        acc_stats = calculate_account_stats(events, acc_id)
        st.subheader(f"📊 Account {acc_id} Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("💰 Deposits", f"${acc_stats['deposits']:,.2f}")
        with c2:
            st.metric("🔻 Withdrawals", f"${acc_stats['withdrawals']:,.2f}")
        with c3:
            st.metric("📊 Net Flow", f"${acc_stats['net_flow']:,.2f}")
        with c4:
            st.metric("🔄 Trades", acc_stats['trade_count'])
        st.divider()
    
    # Tabs
    tabs = st.tabs(["📊 Summary", "📜 Ledger", "📈 Instruments", "🐋 Whales", "🚨 Liquidations"])
    
    with tabs[0]:
        period_tabs = st.tabs(["Daily", "Weekly", "MTD", "YTD", "Custom"])
        periods = ["daily", "weekly", "mtd", "ytd", None]
        
        for i, ptab in enumerate(period_tabs):
            with ptab:
                if periods[i]:
                    now_ts = utc_now()
                    today_8am = get_expiry_time(now_ts)
                    if periods[i] == "daily":
                        start = today_8am - timedelta(days=1) if now_ts < today_8am else today_8am
                        label = f"Since {start.strftime('%m-%d %H:%M')} UTC"
                    elif periods[i] == "weekly":
                        start = today_8am - timedelta(days=7)
                        label = "Last 7 days"
                    elif periods[i] == "mtd":
                        start = datetime(now_ts.year, now_ts.month, 1, 8, 0, 0, tzinfo=UTC)
                        label = now_ts.strftime('%B %Y')
                    else:
                        start = datetime(now_ts.year, 1, 1, 8, 0, 0, tzinfo=UTC)
                        label = now_ts.strftime('%Y')
                    
                    with data_store._lock:
                        pevents = [e for e in data_store._events if make_aware(e.timestamp) >= start]
                    if account_filter:
                        pevents = [e for e in pevents if e.account_id in account_filter]
                    if excluded:
                        pevents = [e for e in pevents if e.account_id not in excluded]
                    if not st.session_state.show_mm:
                        pevents = [e for e in pevents if not e.is_market_maker]
                else:
                    pevents = events
                    label = f"{format_date(start_date)} to {format_date(end_date)}"
                
                stats = calculate_summary_stats(pevents)
                render_summary(["Daily", "Weekly", "MTD", "YTD", "Custom"][i], stats, label)

    with tabs[1]:
        st.subheader("📜 Live Ledger")
        if events:
            df = pd.DataFrame([e.to_dict() for e in events])
            df = df.sort_values("Timestamp (UTC)", ascending=False).reset_index(drop=True)
            st.dataframe(df.style.apply(style_event_row, axis=1), use_container_width=True, height=500)
            st.download_button("📥 CSV", df.to_csv(index=False), 
                             f"events_{format_date(start_date)}_{format_date(end_date)}.csv")
        else:
            st.info("No events")

    with tabs[2]:
        st.subheader("📈 Instruments")
        trades = [e for e in events if e.event_type == "trade"]
        if trades:
            inst_data = {}
            for t in trades:
                inst = t.instrument_type or "UNKNOWN"
                if inst not in inst_data:
                    inst_data[inst] = {"count": 0, "volume": 0, "symbols": {}}
                inst_data[inst]["count"] += 1
                inst_data[inst]["volume"] += t.value or 0
                if t.symbol:
                    inst_data[inst]["symbols"].setdefault(t.symbol, {"count": 0, "volume": 0})
                    inst_data[inst]["symbols"][t.symbol]["count"] += 1
                    inst_data[inst]["symbols"][t.symbol]["volume"] += t.value or 0
            
            icons = {"OPTIONS": "🟡", "PERPETUAL": "🔵", "SPOT": "🟢"}
            cols = st.columns(len(inst_data))
            for i, (inst, d) in enumerate(sorted(inst_data.items(), key=lambda x: x[1]["volume"], reverse=True)):
                with cols[i]:
                    st.markdown(f"### {icons.get(inst, '⚪')} {inst}")
                    st.metric("Trades", d["count"])
                    st.metric("Volume", f"${d['volume']:,.0f}")
            
            for inst, d in inst_data.items():
                with st.expander(f"{icons.get(inst, '⚪')} {inst} Symbols"):
                    sym_df = pd.DataFrame([{"Symbol": s, "Trades": v["count"], "Volume": v["volume"]} 
                                          for s, v in d["symbols"].items()])
                    st.dataframe(sym_df.sort_values("Volume", ascending=False), hide_index=True)
        else:
            st.info("No trades")

    with tabs[3]:
        st.subheader("🐋 Whales (>$10,000)")
        if events:
            df = pd.DataFrame([e.to_dict() for e in events])
            whale_df = df[pd.to_numeric(df["Value (USD)"], errors='coerce').fillna(0) > 10000]
            whale_df = whale_df.sort_values("Timestamp (UTC)", ascending=False)
            if not whale_df.empty:
                st.dataframe(whale_df.style.apply(style_event_row, axis=1), height=400, use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Volume", f"${whale_df['Value (USD)'].sum():,.0f}")
                c2.metric("Count", len(whale_df))
                c3.metric("Avg", f"${whale_df['Value (USD)'].mean():,.0f}")
            else:
                st.info("No whales")

    with tabs[4]:
        st.subheader("🚨 Liquidations")
        liqs = [e for e in events if e.event_type == "liquidation"]
        if liqs:
            st.metric("Total", len(liqs))
            for e in sorted(liqs, key=lambda x: x.timestamp, reverse=True)[:30]:
                st.warning(f"⏰ {e.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Account {e.account_id} | {e.raw_message[:100]}")
        else:
            st.success("✅ No liquidations")

    # Auto refresh
    if st.session_state.listener_started:
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()
