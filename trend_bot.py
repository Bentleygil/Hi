import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopOrderRequest,
    ReplaceOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderType,
    QueryOrderStatus,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ==========================================
# CONFIGURATION (Reads from GitHub Secrets)
# ==========================================
API_KEY = os.environ["ALPACA_API_KEY"].strip()
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"].strip()
PAPER = True  # Paper trading enabled

TICKERS = ["TSLA", "NVDA", "AAPL", "META", "QQQ"]
CASH_ETF = "SGOV"
ATR_MULTIPLIER = 3.5
TOTAL_TARGET_WEIGHT = 0.98
CASH_BUFFER = 0.02
WEIGHT_PER_SYMBOL = TOTAL_TARGET_WEIGHT / len(TICKERS)

# Initialize Alpaca Clients
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


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

    # ATR 14 (Simple Moving Average of True Range to match QuantConnect)
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


def get_latest_price(symbol: str) -> float:
    """Gets the most recent daily close price for cash sizing."""
    start_date = datetime.now() - timedelta(days=10)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(request).df
    return float(bars["close"].iloc[-1])


# ==========================================
# ORDER & STOP MANAGEMENT
# ==========================================
def get_open_stop_order(symbol: str):
    """Finds an active GTC stop order for a given symbol."""
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    orders = trading_client.get_orders(filter=req)
    for order in orders:
        if order.type == OrderType.STOP and order.side == OrderSide.SELL:
            return order
    return None


def place_or_update_stop(symbol: str, qty: int, new_stop: float) -> None:
    """Places a new GTC stop order or ratchets up an existing stop order."""
    new_stop = round(new_stop, 2)
    existing_stop = get_open_stop_order(symbol)

    if existing_stop:
        current_stop_price = float(existing_stop.stop_price)
        current_qty = int(float(existing_stop.qty))

        if new_stop > current_stop_price or qty != current_qty:
            target_stop = max(new_stop, current_stop_price)
            print(f"[{symbol}] Updating stop to ${target_stop} (Qty: {qty})")
            replace_req = ReplaceOrderRequest(qty=qty, stop_price=target_stop)
            trading_client.replace_order_by_id(existing_stop.id, replace_req)
        return

    print(f"[{symbol}] Placing new GTC stop at ${new_stop} (Qty: {qty})")
    stop_req = StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=new_stop,
    )
    trading_client.submit_order(stop_req)


# ==========================================
# MAIN STRATEGY EXECUTION
# ==========================================
def run_strategy() -> None:
    print(f"\n--- Running Strategy Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    available_cash = float(account.cash)

    positions = {p.symbol: int(float(p.qty)) for p in trading_client.get_all_positions()}

    sgov_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[CASH_ETF])
    )
    for o in sgov_orders:
        trading_client.cancel_order_by_id(o.id)

    pending_entry_dollars = 0.0
    new_entries = []

    for symbol in TICKERS:
        ind = get_indicators(symbol)
        if not ind:
            continue

        close = ind["close"]
        trend_bullish = ind["ema_fast"] > ind["ema_slow"]
        above_regime = close > ind["sma_200"]
        stop_price = close - (ATR_MULTIPLIER * ind["atr"])

        if symbol in positions and positions[symbol] > 0:
            place_or_update_stop(symbol, positions[symbol], stop_price)
            continue

        open_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        if len(open_orders) > 0:
            continue

        if trend_bullish and above_regime:
            target_dollars = portfolio_value * WEIGHT_PER_SYMBOL
            qty = int(target_dollars // close)
            if qty > 0:
                est_cost = qty * close
                new_entries.append((symbol, qty, stop_price))
                pending_entry_dollars += est_cost

    sgov_qty = positions.get(CASH_ETF, 0)
    if pending_entry_dollars > available_cash and sgov_qty > 0:
        sgov_price = get_latest_price(CASH_ETF)
        shortfall = (pending_entry_dollars - available_cash) * 1.01
        shares_to_sell = min(sgov_qty, int(shortfall // sgov_price) + 1)

        if shares_to_sell > 0:
            print(f"[{CASH_ETF}] Selling {shares_to_sell} shares to fund new entries...")
            sell_sgov = MarketOrderRequest(
                symbol=CASH_ETF,
                qty=shares_to_sell,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            trading_client.submit_order(sell_sgov)
            time.sleep(3)

    for symbol, qty, initial_stop in new_entries:
        print(f"[{symbol}] Submitting BUY order for {qty} shares...")
        buy_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(buy_req)
        time.sleep(2)
        place_or_update_stop(symbol, qty, initial_stop)

    time.sleep(2)
    refreshed_account = trading_client.get_account()
    current_cash = float(refreshed_account.cash)
    min_cash_reserve = portfolio_value * CASH_BUFFER
    deployable_cash = current_cash - min_cash_reserve

    sgov_price = get_latest_price(CASH_ETF)
    if deployable_cash >= sgov_price:
        qty_to_buy = int(deployable_cash // sgov_price)
        if qty_to_buy > 0:
            print(f"[{CASH_ETF}] Sweeping idle cash into {qty_to_buy} shares...")
            buy_sgov = MarketOrderRequest(
                symbol=CASH_ETF,
                qty=qty_to_buy,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            trading_client.submit_order(buy_sgov)

    print("--- Strategy Check Complete ---")


if __name__ == "__main__":
    clock = trading_client.get_clock()
    if clock.is_open:
        run_strategy()
    else:
        print(f"Market is currently closed. Next open: {clock.next_open}")
