import json
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ==========================================
# CONFIGURATION (Optimized for $1,000 Account)
# ==========================================
API_KEY = os.environ["ALPACA_API_KEY"].strip()
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"].strip()
PAPER = True  # Set False only when ready for live money

TICKERS = ["TSLA", "NVDA", "AAPL", "META", "QQQ"]
CASH_ETF = "SGOV"
ATR_MULTIPLIER = 3.5
TOTAL_TARGET_WEIGHT = 0.98
CASH_BUFFER = 0.02
WEIGHT_PER_SYMBOL = TOTAL_TARGET_WEIGHT / len(TICKERS)
MIN_NOTIONAL_ORDER = 5.00  # Alpaca minimum fractional order is $1; $5 avoids dust
STOP_STATE_FILE = "stop_levels.json"

# Initialize Alpaca Clients
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


# ==========================================
# TRAILING STOP STATE STORAGE
# ==========================================
def load_stops() -> dict[str, float]:
    """Loads saved trailing stop prices for fractional positions."""
    if os.path.exists(STOP_STATE_FILE):
        try:
            with open(STOP_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_stops(stops: dict[str, float]) -> None:
    """Saves updated trailing stop prices."""
    with open(STOP_STATE_FILE, "w") as f:
        json.dump(stops, f, indent=2)


# ==========================================
# INDICATOR CALCULATIONS
# ==========================================
def get_indicators(symbol: str) -> dict | None:
    """Fetches daily bars and calculates EMA(12), EMA(26), SMA(200), and ATR(14)."""
    start_date = datetime.now() - timedelta(days=365)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(request).df
    if bars.empty or len(bars) < 205:
        print(f"[{symbol}] Not enough historical bars for SMA 200.")
        return None

    df = bars.reset_index()

    # EMA 12 & EMA 26
    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()

    # SMA 200
    df["sma_200"] = df["close"].rolling(window=200).mean()

    # ATR 14
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14).mean()

    latest = df.iloc[-1]
    return {
        "close": float(latest["close"]),
        "ema_fast": float(latest["ema_fast"]),
        "ema_slow": float(latest["ema_slow"]),
        "sma_200": float(latest["sma_200"]),
        "atr": float(latest["atr_14"]),
    }


# ==========================================
# MAIN STRATEGY EXECUTION
# ==========================================
def run_strategy() -> None:
    print(f"\n--- Running Fractional Strategy Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    available_cash = float(account.cash)

    # Map current open positions (fractional qty and market value)
    positions = {
        p.symbol: {
            "qty": float(p.qty),
            "market_value": float(p.market_value),
        }
        for p in trading_client.get_all_positions()
    }

    stops = load_stops()
    pending_entry_dollars = 0.0
    new_entries: list[tuple[str, float, float]] = []

    # 1. Check existing positions for trailing stop exits/ratchets & collect new entries
    for symbol in TICKERS:
        ind = get_indicators(symbol)
        if not ind:
            continue

        close = ind["close"]
        trend_bullish = ind["ema_fast"] > ind["ema_slow"]
        above_regime = close > ind["sma_200"]
        candidate_stop = round(close - (ATR_MULTIPLIER * ind["atr"]), 2)

        # Case A: Already holding fractional position
        if symbol in positions and positions[symbol]["qty"] > 0:
            current_stop = stops.get(symbol, candidate_stop)

            # Exit if price fell below our ratcheted trailing stop
            if close <= current_stop:
                print(f"[{symbol}] Stop triggered (${close:.2f} <= ${current_stop:.2f}). Closing position...")
                trading_client.close_position(symbol)
                stops.pop(symbol, None)
                time.sleep(2)
            else:
                # Ratchet stop upward only
                new_stop = max(current_stop, candidate_stop)
                stops[symbol] = new_stop
                print(f"[{symbol}] Holding | Price: ${close:.2f} | Trailing Stop: ${new_stop:.2f}")
            continue

        # Clean up stale stop if position was closed externally
        stops.pop(symbol, None)

        # Skip if an order is already pending
        open_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        if len(open_orders) > 0:
            continue

        # Case B: New bullish trend entry using fractional dollar slice
        if trend_bullish and above_regime:
            target_dollars = round(portfolio_value * WEIGHT_PER_SYMBOL, 2)
            if target_dollars >= MIN_NOTIONAL_ORDER:
                new_entries.append((symbol, target_dollars, candidate_stop))
                pending_entry_dollars += target_dollars

    # 2. Free up cash from fractional SGOV if needed for new entries
    sgov_pos = positions.get(CASH_ETF)
    if pending_entry_dollars > available_cash and sgov_pos and sgov_pos["market_value"] > MIN_NOTIONAL_ORDER:
        shortfall = round((pending_entry_dollars - available_cash) * 1.02, 2)
        dollars_to_sell = min(round(sgov_pos["market_value"], 2), shortfall)

        if dollars_to_sell >= MIN_NOTIONAL_ORDER:
            print(f"[{CASH_ETF}] Selling ${dollars_to_sell:.2f} of SGOV to fund entries...")
            sell_sgov = MarketOrderRequest(
                symbol=CASH_ETF,
                notional=dollars_to_sell,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            trading_client.submit_order(sell_sgov)
            time.sleep(3)

    # 3. Submit fractional dollar BUY orders for new entries
    for symbol, dollars, initial_stop in new_entries:
        refreshed_cash = float(trading_client.get_account().cash)
        order_dollars = min(dollars, round(refreshed_cash - (portfolio_value * CASH_BUFFER), 2))
        if order_dollars < MIN_NOTIONAL_ORDER:
            print(f"[{symbol}] Skipping entry—insufficient deployable cash (${order_dollars:.2f}).")
            continue

        print(f"[{symbol}] Submitting fractional BUY for ${order_dollars:.2f} (Initial Stop: ${initial_stop:.2f})...")
        buy_req = MarketOrderRequest(
            symbol=symbol,
            notional=order_dollars,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(buy_req)
        stops[symbol] = initial_stop
        time.sleep(2)

    # 4. Sweep remaining idle cash into fractional SGOV
    time.sleep(2)
    refreshed_account = trading_client.get_account()
    current_cash = float(refreshed_account.cash)
    min_cash_reserve = portfolio_value * CASH_BUFFER
    deployable_cash = round(current_cash - min_cash_reserve, 2)

    if deployable_cash >= MIN_NOTIONAL_ORDER:
        print(f"[{CASH_ETF}] Sweeping ${deployable_cash:.2f} idle cash into SGOV...")
        buy_sgov = MarketOrderRequest(
            symbol=CASH_ETF,
            notional=deployable_cash,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(buy_sgov)

    save_stops(stops)
    print("--- Strategy Check Complete ---")


if __name__ == "__main__":
    clock = trading_client.get_clock()
    if clock.is_open:
        run_strategy()
    else:
        print(f"Market is currently closed. Next open: {clock.next_open}")
