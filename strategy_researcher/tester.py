"""
Tester --- Runs a strategy against live BTC 5-min Polymarket markets.

Phase 3 of the daily cycle. Uses the CORRECT Polymarket price sources:

  - Chainlink BTC/USD via RTDS WebSocket (wss://ws-live-data.polymarket.com)
    This is the EXACT price Polymarket resolves against. Coinbase/Kraken
    prices diverge enough to generate wrong signals.

  - CLOB API (https://clob.polymarket.com) for real token prices / market odds.
    No more estimating odds from price delta.

  - GAMMA API for PTB (Price to Beat) and market resolution.

Resolution rule: close >= open => UP, close < open => DOWN (ties go UP).
"""

import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_prices import get_ptb_from_api, get_market_resolution, get_live_btc_price

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
RTDS_WS_URL = "wss://ws-live-data.polymarket.com"
STRATEGIES_DIR = Path(__file__).parent / "strategies"


# ---------------------------------------------------------------------------
# Chainlink RTDS WebSocket price feed
# ---------------------------------------------------------------------------

class ChainlinkPriceFeed:
    """
    Background WebSocket to Polymarket RTDS for the authoritative
    Chainlink BTC/USD price — the same feed markets resolve against.

    Usage:
        feed = ChainlinkPriceFeed()
        feed.start()
        price = feed.price          # latest Chainlink BTC/USD or None
        feed.stop()
    """

    def __init__(self):
        self._price: float | None = None
        self._price_ts: float = 0.0  # time.time() of last update
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False

    # -- public API --

    def start(self):
        """Connect in a daemon thread. Non-blocking."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Give the WebSocket a moment to connect
        deadline = time.time() + 5
        while not self._connected and time.time() < deadline:
            time.sleep(0.2)
        if self._connected:
            logger.info("Chainlink RTDS feed connected")
        else:
            logger.warning("Chainlink RTDS feed did not connect in 5s (will keep trying)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def price(self) -> float | None:
        """Latest Chainlink BTC/USD price, or None if unavailable."""
        # Stale after 30s — treat as disconnected
        if self._price and (time.time() - self._price_ts) > 30:
            return None
        return self._price

    @property
    def connected(self) -> bool:
        return self._connected and self.price is not None

    # -- internals --

    def _run(self):
        """WebSocket loop with auto-reconnect."""
        try:
            import websocket
        except ImportError:
            logger.error(
                "websocket-client not installed. "
                "Install with: pip install websocket-client"
            )
            self._running = False
            return

        while self._running:
            try:
                ws = websocket.WebSocket()
                ws.settimeout(10)
                ws.connect(RTDS_WS_URL)
                self._connected = True

                # Subscribe to Chainlink BTC/USD
                sub_msg = json.dumps({
                    "type": "subscribe",
                    "channel": "crypto_prices_chainlink",
                    "filter": {"symbol": "btc/usd"},
                })
                ws.send(sub_msg)
                logger.debug("Subscribed to crypto_prices_chainlink btc/usd")

                while self._running:
                    try:
                        raw = ws.recv()
                        if not raw:
                            continue
                        msg = json.loads(raw)
                        self._handle_message(msg)
                    except websocket.WebSocketTimeoutException:
                        continue
                    except websocket.WebSocketConnectionClosedException:
                        logger.warning("RTDS WebSocket closed, reconnecting...")
                        break

            except Exception as e:
                logger.warning(f"RTDS WebSocket error: {e}")
                self._connected = False

            # Reconnect back-off
            if self._running:
                time.sleep(2)

    def _handle_message(self, msg: dict):
        """Extract price from an RTDS message."""
        # RTDS messages vary in format; handle the common shapes
        price = None

        # Shape 1: {"price": 84250.0, "timestamp": ...}
        if "price" in msg:
            try:
                price = float(msg["price"])
            except (ValueError, TypeError):
                pass

        # Shape 2: {"data": {"price": 84250.0, ...}}
        if price is None and isinstance(msg.get("data"), dict):
            try:
                price = float(msg["data"]["price"])
            except (ValueError, TypeError, KeyError):
                pass

        # Shape 3: {"data": [{"price": 84250.0, ...}]}
        if price is None and isinstance(msg.get("data"), list):
            for item in msg["data"]:
                if isinstance(item, dict) and "price" in item:
                    try:
                        price = float(item["price"])
                        break
                    except (ValueError, TypeError):
                        pass

        if price and 10_000 < price < 500_000:
            self._price = price
            self._price_ts = time.time()


# Singleton feed — shared across the test run
_chainlink_feed: ChainlinkPriceFeed | None = None


def _get_chainlink_feed() -> ChainlinkPriceFeed:
    """Get or create the singleton Chainlink price feed."""
    global _chainlink_feed
    if _chainlink_feed is None:
        _chainlink_feed = ChainlinkPriceFeed()
        _chainlink_feed.start()
    return _chainlink_feed


def _stop_chainlink_feed():
    global _chainlink_feed
    if _chainlink_feed:
        _chainlink_feed.stop()
        _chainlink_feed = None


# ---------------------------------------------------------------------------
# CLOB API — real token prices
# ---------------------------------------------------------------------------

def _get_market_tokens(window_start: int) -> dict | None:
    """
    Get UP/DOWN token IDs and prices from GAMMA API for a given window.

    Returns:
        {
            "token_up": str,       -- CLOB token ID for UP outcome
            "token_down": str,     -- CLOB token ID for DOWN outcome
            "condition_id": str,
            "closed": bool,
        }
        or None on failure.
    """
    slug = f"btc-updown-5m-{window_start}"
    try:
        r = requests.get(f"{GAMMA_HOST}/events?slug={slug}", timeout=10)
        r.raise_for_status()
        events = r.json()
        if not events or not events[0].get("markets"):
            return None

        market = events[0]["markets"][0]
        tokens = json.loads(market.get("clobTokenIds", "[]"))
        if len(tokens) < 2:
            return None

        return {
            "token_up": tokens[0],
            "token_down": tokens[1],
            "condition_id": market.get("conditionId", ""),
            "closed": market.get("closed", False),
        }
    except Exception as e:
        logger.warning(f"GAMMA token lookup failed: {e}")
        return None


def _get_clob_price(token_id: str, side: str = "BUY") -> float | None:
    """
    Get the current best price for a token from CLOB API.
    side: "BUY" or "SELL"
    """
    try:
        r = requests.get(
            f"{CLOB_HOST}/price",
            params={"token_id": token_id, "side": side},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        price = float(data.get("price", 0))
        if 0 < price < 1:
            return price
    except Exception as e:
        logger.debug(f"CLOB price fetch failed: {e}")
    return None


def _get_clob_midpoint(token_id: str) -> float | None:
    """Get the midpoint price for a token from CLOB API."""
    try:
        r = requests.get(
            f"{CLOB_HOST}/midpoint",
            params={"token_id": token_id},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        mid = float(data.get("mid", 0))
        if 0 < mid < 1:
            return mid
    except Exception as e:
        logger.debug(f"CLOB midpoint fetch failed: {e}")
    return None


def get_market_odds(window_start: int) -> tuple[dict, dict | None]:
    """
    Get real market odds from CLOB API for a given window.

    Returns:
        (
            {"up": float, "down": float},    -- market odds (0-1)
            market_tokens dict or None        -- for later use in order placement
        )

    Falls back to GAMMA API outcomePrices if CLOB fails.
    """
    tokens = _get_market_tokens(window_start)
    if not tokens:
        return {"up": 0.50, "down": 0.50}, None

    # Try CLOB midpoint first (most accurate)
    up_mid = _get_clob_midpoint(tokens["token_up"])
    down_mid = _get_clob_midpoint(tokens["token_down"])

    if up_mid and down_mid:
        return {"up": round(up_mid, 4), "down": round(down_mid, 4)}, tokens

    # Try CLOB buy price
    up_price = _get_clob_price(tokens["token_up"], "BUY")
    down_price = _get_clob_price(tokens["token_down"], "BUY")

    if up_price and down_price:
        return {"up": round(up_price, 4), "down": round(down_price, 4)}, tokens

    # Fallback: GAMMA API outcomePrices
    try:
        slug = f"btc-updown-5m-{window_start}"
        r = requests.get(f"{GAMMA_HOST}/events?slug={slug}", timeout=10)
        r.raise_for_status()
        events = r.json()
        if events and events[0].get("markets"):
            market = events[0]["markets"][0]
            prices = json.loads(market.get("outcomePrices", "[]"))
            if len(prices) >= 2:
                return {
                    "up": round(float(prices[0]), 4),
                    "down": round(float(prices[1]), 4),
                }, tokens
    except Exception:
        pass

    return {"up": 0.50, "down": 0.50}, tokens


# ---------------------------------------------------------------------------
# BTC price — Chainlink first, CEX fallback
# ---------------------------------------------------------------------------

def get_btc_price() -> float | None:
    """
    Get BTC price from the best available source:
    1. Chainlink RTDS WebSocket (authoritative — what Polymarket resolves against)
    2. Coinbase/Kraken spot (fallback — close but not exact)
    """
    feed = _get_chainlink_feed()
    if feed.connected and feed.price:
        return feed.price

    # Fallback to CEX
    return get_live_btc_price()


def get_btc_price_source() -> str:
    """Return which price source is active."""
    feed = _get_chainlink_feed()
    if feed.connected and feed.price:
        return "chainlink"
    return "cex_fallback"


# ---------------------------------------------------------------------------
# Price history from Chainlink (with CEX fallback)
# ---------------------------------------------------------------------------

# Rolling buffer of Chainlink prices, populated by the tester loop
_price_buffer: list[tuple[float, float]] = []  # [(timestamp, price), ...]
_price_buffer_lock = threading.Lock()


def _record_price(price: float):
    """Add a price to the rolling buffer (called every loop iteration)."""
    now = time.time()
    with _price_buffer_lock:
        _price_buffer.append((now, price))
        # Keep last 25 minutes
        cutoff = now - 1500
        while _price_buffer and _price_buffer[0][0] < cutoff:
            _price_buffer.pop(0)


def _get_price_history(lookback_minutes: int = 20) -> list[float]:
    """
    Build price history from the rolling buffer (1-min interval samples).

    If the buffer doesn't have enough data yet (early in the run),
    falls back to Coinbase 1-min candles.
    """
    cutoff = time.time() - lookback_minutes * 60
    with _price_buffer_lock:
        relevant = [(ts, p) for ts, p in _price_buffer if ts >= cutoff]

    if len(relevant) >= lookback_minutes:
        # Sample one price per minute (take the last reading in each minute bucket)
        buckets: dict[int, float] = {}
        for ts, p in relevant:
            minute_key = int(ts // 60)
            buckets[minute_key] = p
        prices = [buckets[k] for k in sorted(buckets.keys())]
        return prices[-lookback_minutes:]

    # Not enough Chainlink data yet — fall back to Coinbase candles
    try:
        from data.fetch_prices import fetch_coinbase_candles
        candles = fetch_coinbase_candles(days=1, granularity=60)
        if candles:
            recent = candles[-lookback_minutes:]
            return [c["close"] for c in recent]
    except Exception as e:
        logger.debug(f"Coinbase candle fetch failed: {e}")

    # Absolute fallback
    price = get_btc_price()
    if price:
        return [price] * min(lookback_minutes, 5)
    return [84000.0, 84000.0]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(btc_price: float, price_history: list[float],
                   market_odds: dict, time_in_window: float,
                   trade_results: list[dict], volume: float) -> dict:
    """Build the context dict passed to decide()."""
    resolved = [t for t in trade_results if t.get("result") is not None]
    wins = sum(1 for t in resolved if t["result"] == "WIN")
    win_rate = wins / len(resolved) if resolved else 0.0

    return {
        "btc_price": btc_price,
        "price_history": price_history,
        "market_odds": market_odds,
        "time_in_window": time_in_window,
        "recent_trades": trade_results[-20:],
        "win_rate": win_rate,
        "volume": volume,
    }


# ---------------------------------------------------------------------------
# Trade logging
# ---------------------------------------------------------------------------

def _log_trade(path: Path, entry: dict):
    """Append a trade entry to the day's JSONL log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_trades(path: Path) -> list[dict]:
    """Load all trade entries from a JSONL file."""
    if not path.exists():
        return []
    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[dict]) -> dict:
    """
    Compute performance metrics from a list of trade results.

    Returns dict with: total_trades, total_decisions, wins, losses, skips,
    win_rate, total_pnl, avg_pnl, sharpe_ratio, max_drawdown,
    best_trade, worst_trade, avg_confidence.
    """
    executed = [t for t in trades if t.get("action") in ("UP", "DOWN")]
    skips = [t for t in trades if t.get("action") == "SKIP"]
    resolved = [t for t in executed if t.get("result") is not None]

    wins = sum(1 for t in resolved if t["result"] == "WIN")
    losses = len(resolved) - wins

    pnls = [t.get("pnl", 0.0) for t in resolved if t.get("pnl") is not None]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / len(pnls) if pnls else 0.0

    # Sharpe ratio (annualized, 288 five-minute windows per day)
    if len(pnls) >= 2:
        mean_pnl = total_pnl / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.001
        sharpe = (mean_pnl / std_pnl) * math.sqrt(288)
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    confidences = [t.get("confidence", 0.5) for t in executed]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "total_trades": len(executed),
        "total_decisions": len(trades),
        "wins": wins,
        "losses": losses,
        "skips": len(skips),
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0.0,
        "total_pnl": round(total_pnl, 4),
        "avg_pnl": round(avg_pnl, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "best_trade": round(max(pnls), 4) if pnls else 0.0,
        "worst_trade": round(min(pnls), 4) if pnls else 0.0,
        "avg_confidence": round(avg_conf, 3),
    }


# ---------------------------------------------------------------------------
# Live / dry-run tester
# ---------------------------------------------------------------------------

def run_test(decide_fn, day: int,
             duration_hours: float = 24.0,
             entry_seconds: int = 30,
             bet_size: float = 1.0,
             dry_run: bool = True) -> dict:
    """
    Run a strategy against live BTC 5-min Polymarket markets.

    Price sources:
      btc_price    -> Chainlink RTDS WebSocket (exact resolution price)
      market_odds  -> CLOB API midpoint/buy prices (real market odds)
      PTB          -> GAMMA API eventMetadata.priceToBeat
      resolution   -> GAMMA API market outcome

    Args:
        decide_fn: The strategy's decide(context) -> dict function
        day: Day number (for logging)
        duration_hours: How long to run the test
        entry_seconds: How many seconds before window close to make decisions
        bet_size: Simulated bet size in USD
        dry_run: If True, don't place real orders

    Returns:
        {"trades": list[dict], "metrics": dict, "duration_hours": float}
    """
    log_path = STRATEGIES_DIR / f"day_{day:02d}_trades.jsonl"
    trade_results = _load_trades(log_path)  # Resume if interrupted
    traded_windows = {t.get("window_start") for t in trade_results}

    start_time = time.time()
    end_time = start_time + duration_hours * 3600
    current_window = None
    ptb = None
    market_tokens = None

    # Start Chainlink price feed
    feed = _get_chainlink_feed()

    mode = "DRY RUN" if dry_run else "LIVE"
    price_src = "Chainlink RTDS" if feed.connected else "CEX fallback (Chainlink connecting...)"
    print(f"\n  {'='*60}")
    print(f"  STRATEGY TESTER -- Day {day}")
    print(f"  Mode: {mode} | Duration: {duration_hours}h | Entry: T-{entry_seconds}s")
    print(f"  Bet size: ${bet_size:.2f}")
    print(f"  Price source: {price_src}")
    print(f"  Market odds: CLOB API (real token prices)")
    print(f"  {'='*60}\n")

    try:
        while time.time() < end_time:
            ts = int(time.time())
            window_start = ts - (ts % 300)
            window_end = window_start + 300
            seconds_left = window_end - ts
            time_in_window = 300 - seconds_left

            # Record current price for history buffer
            current_price = get_btc_price()
            if current_price:
                _record_price(current_price)

            # New window
            if window_start != current_window:
                current_window = window_start

                # Get PTB from GAMMA API (Chainlink price at window boundary)
                ptb_api = get_ptb_from_api(window_start)
                if ptb_api:
                    ptb = ptb_api
                else:
                    # Fallback: use current Chainlink price at boundary
                    ptb = get_btc_price()

                # Pre-fetch market tokens for CLOB queries
                market_tokens = _get_market_tokens(window_start)

                if ptb:
                    src = get_btc_price_source()
                    now_str = datetime.now().strftime("%H:%M:%S")
                    token_status = "tokens OK" if market_tokens else "no tokens"
                    print(f"  [{now_str}] Window {window_start} | PTB: ${ptb:,.2f} | "
                          f"src={src} | {token_status}")

            # Decision zone: entry_seconds before window close
            if (entry_seconds >= seconds_left > 3
                    and window_start not in traded_windows
                    and ptb):

                live_price = get_btc_price()
                if not live_price:
                    time.sleep(2)
                    continue

                # Get real market odds from CLOB API
                market_odds, _ = get_market_odds(window_start)

                # Build price history (Chainlink buffer, CEX fallback)
                price_history = _get_price_history(20)

                # Get volume from CLOB spread as a proxy
                volume = 150000.0  # TODO: fetch from data-api.polymarket.com

                # Build context and call strategy
                context = _build_context(
                    btc_price=live_price,
                    price_history=price_history,
                    market_odds=market_odds,
                    time_in_window=time_in_window,
                    trade_results=trade_results,
                    volume=volume,
                )

                try:
                    decision = decide_fn(context)
                except Exception as e:
                    logger.error(f"Strategy crashed: {e}")
                    decision = {"action": "SKIP", "confidence": 0.0,
                                "reasoning": f"Strategy error: {e}"}

                action = decision.get("action", "SKIP")
                confidence = decision.get("confidence", 0.0)
                reasoning = decision.get("reasoning", "")
                now_str = datetime.now().strftime("%H:%M:%S")
                price_src = get_btc_price_source()

                if action == "SKIP":
                    print(f"  [{now_str}] SKIP | conf={confidence:.2f} | {reasoning[:60]}")
                    entry = {
                        "window_start": window_start,
                        "action": "SKIP",
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "btc_price": live_price,
                        "ptb": ptb,
                        "market_odds": market_odds,
                        "price_source": price_src,
                        "result": None,
                        "pnl": None,
                    }
                    _log_trade(log_path, entry)
                    trade_results.append(entry)
                    traded_windows.add(window_start)

                elif action in ("UP", "DOWN"):
                    delta_pct = (live_price - ptb) / ptb * 100 if ptb else 0

                    print(f"\n  {'='*55}")
                    print(f"  [{now_str}] {action} | conf={confidence:.2f} | "
                          f"{seconds_left}s left | src={price_src}")
                    print(f"  delta={delta_pct:+.4f}% | PTB=${ptb:,.2f} | "
                          f"Live=${live_price:,.2f}")
                    print(f"  Odds: UP={market_odds['up']:.3f} DOWN={market_odds['down']:.3f}")
                    print(f"  Reason: {reasoning[:80]}")

                    if dry_run:
                        print(f"  [DRY RUN] Simulated ${bet_size:.2f} on {action}")
                    else:
                        from live_trader import place_fok_order
                        place_fok_order(action, bet_size)

                    traded_windows.add(window_start)

                    # Wait for resolution
                    print(f"  Waiting {seconds_left + 5}s for resolution...")
                    time.sleep(seconds_left + 10)

                    # Check resolution — GAMMA API first (authoritative), CEX fallback
                    resolution = get_market_resolution(window_start)
                    close_price = get_btc_price()

                    if resolution:
                        won = resolution == action
                    elif close_price and ptb:
                        # Fallback: match Polymarket's rule (>= is UP)
                        actual = "UP" if close_price >= ptb else "DOWN"
                        won = actual == action
                    else:
                        won = None

                    if won is not None:
                        # P&L based on real market odds at entry
                        buy_price = market_odds.get(
                            "up" if action == "UP" else "down", 0.5
                        )
                        if won:
                            pnl = bet_size * (1.0 - buy_price)
                        else:
                            pnl = -bet_size * buy_price

                        icon = "WIN" if won else "LOSS"
                        close_str = f"${close_price:,.2f}" if close_price else "N/A"
                        print(f"  {icon} | Close: {close_str} | P&L: ${pnl:+.4f}")
                    else:
                        pnl = None
                        print(f"  UNRESOLVED | Could not determine outcome")

                    entry = {
                        "window_start": window_start,
                        "action": action,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "btc_price": live_price,
                        "ptb": ptb,
                        "close_price": close_price,
                        "market_odds": market_odds,
                        "delta_pct": delta_pct,
                        "price_source": price_src,
                        "result": ("WIN" if won else "LOSS") if won is not None else None,
                        "pnl": round(pnl, 4) if pnl is not None else None,
                        "bet_size": bet_size,
                    }
                    _log_trade(log_path, entry)
                    trade_results.append(entry)

                    # Print running stats
                    metrics = compute_metrics(trade_results)
                    print(f"  Running: {metrics['wins']}W/{metrics['losses']}L | "
                          f"WR={metrics['win_rate']}% | "
                          f"P&L=${metrics['total_pnl']:+.4f}")
                    print(f"  {'='*55}\n")

                    continue  # Skip the sleep at the end

            time.sleep(5)

    except KeyboardInterrupt:
        print(f"\n  Test interrupted by user (Ctrl+C)")

    finally:
        _stop_chainlink_feed()

    # Final metrics
    metrics = compute_metrics(trade_results)
    actual_hours = (time.time() - start_time) / 3600

    print(f"\n  {'='*60}")
    print(f"  DAY {day} TEST COMPLETE")
    print(f"  Duration: {actual_hours:.1f}h")
    print(f"  Price source used: {get_btc_price_source()}")
    print(f"  Trades: {metrics['total_trades']} | Skips: {metrics['skips']}")
    print(f"  Wins: {metrics['wins']} | Losses: {metrics['losses']}")
    print(f"  Win Rate: {metrics['win_rate']}%")
    print(f"  Total P&L: ${metrics['total_pnl']:+.4f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']}")
    print(f"  Max Drawdown: ${metrics['max_drawdown']:.4f}")
    print(f"  {'='*60}\n")

    return {
        "trades": trade_results,
        "metrics": metrics,
        "duration_hours": round(actual_hours, 2),
    }


# ---------------------------------------------------------------------------
# Backtest (historical candles — offline)
# ---------------------------------------------------------------------------

def run_backtest(decide_fn, day: int,
                 candles: list[dict] = None,
                 bet_size: float = 1.0) -> dict:
    """
    Run a strategy against historical candle data (offline backtest).

    NOTE: Uses Coinbase candles, not Chainlink historical data.
    Odds are simulated at 50/50. This is a rough sanity check only —
    live dry-run is the real test.
    """
    if candles is None:
        from data.fetch_prices import load_cached_candles
        candles = load_cached_candles()

    if len(candles) < 25:
        logger.error("Not enough candle data for backtest (need at least 25)")
        return {"trades": [], "metrics": compute_metrics([]), "duration_hours": 0}

    trade_results = []

    for i in range(20, len(candles)):
        candle = candles[i]
        ptb = candle["open"]
        close_price = candle["close"]

        history_candles = candles[max(0, i - 20):i]
        price_history = [c["close"] for c in history_candles]

        # 50/50 odds — we don't have historical CLOB data
        market_odds = {"up": 0.50, "down": 0.50}

        context = _build_context(
            btc_price=ptb,
            price_history=price_history,
            market_odds=market_odds,
            time_in_window=240.0,
            trade_results=trade_results,
            volume=150000.0,
        )

        try:
            decision = decide_fn(context)
        except Exception as e:
            decision = {"action": "SKIP", "confidence": 0.0, "reasoning": f"Error: {e}"}

        action = decision.get("action", "SKIP")
        confidence = decision.get("confidence", 0.0)
        reasoning = decision.get("reasoning", "")

        if action in ("UP", "DOWN"):
            # Polymarket rule: >= is UP
            actual = "UP" if close_price >= ptb else "DOWN"
            won = actual == action

            buy_price = market_odds.get("up" if action == "UP" else "down", 0.5)
            pnl = bet_size * (1.0 - buy_price) if won else -bet_size * buy_price

            trade_results.append({
                "window_start": candle["open_time"],
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "btc_price": ptb,
                "ptb": ptb,
                "close_price": close_price,
                "market_odds": market_odds,
                "price_source": "backtest_coinbase",
                "result": "WIN" if won else "LOSS",
                "pnl": round(pnl, 4),
                "bet_size": bet_size,
            })
        else:
            trade_results.append({
                "window_start": candle["open_time"],
                "action": "SKIP",
                "confidence": confidence,
                "reasoning": reasoning,
                "btc_price": ptb,
                "ptb": ptb,
                "market_odds": market_odds,
                "price_source": "backtest_coinbase",
                "result": None,
                "pnl": None,
            })

    metrics = compute_metrics(trade_results)
    return {
        "trades": trade_results,
        "metrics": metrics,
        "duration_hours": len(candles) * 5 / 60,
    }
