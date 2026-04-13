"""
Tester --- Runs a strategy against live BTC 5-min Polymarket markets.

Phase 3 of the daily cycle:
1. Monitors BTC 5-min windows continuously
2. Calls the day's decide() function each window
3. Logs every decision and outcome (dry-run or live)
4. Computes real-time metrics: win rate, P&L, Sharpe, max drawdown

Reuses infrastructure from data/fetch_prices.py and live_trader.py.
"""

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Reuse existing infrastructure
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_prices import (
    get_ptb_from_api,
    get_market_resolution,
    get_live_btc_price,
    fetch_coinbase_candles,
)
from live_trader import get_current_market

STRATEGIES_DIR = Path(__file__).parent / "strategies"
TRADE_LOG_DIR = Path(__file__).parent / "strategies"


def _get_price_history(lookback_minutes: int = 20) -> list[float]:
    """
    Get recent BTC price history (1-min intervals).
    Uses Coinbase candles for the last N minutes.
    Falls back to repeated spot price checks if candle fetch fails.
    """
    try:
        candles = fetch_coinbase_candles(days=1, granularity=60)
        if candles:
            # Take last N candles, extract close prices
            recent = candles[-lookback_minutes:]
            return [c["close"] for c in recent]
    except Exception as e:
        logger.debug(f"Candle fetch failed, using spot prices: {e}")

    # Fallback: just return current price repeated
    price = get_live_btc_price()
    if price:
        return [price] * min(lookback_minutes, 5)
    return [84000.0, 84000.0]  # absolute fallback


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


def compute_metrics(trades: list[dict]) -> dict:
    """
    Compute performance metrics from a list of trade results.

    Returns:
        {
            "total_trades": int,
            "total_decisions": int,  -- including skips
            "wins": int,
            "losses": int,
            "skips": int,
            "win_rate": float,       -- wins / (wins + losses)
            "total_pnl": float,
            "avg_pnl": float,
            "sharpe_ratio": float,
            "max_drawdown": float,
            "best_trade": float,
            "worst_trade": float,
            "avg_confidence": float,
        }
    """
    executed = [t for t in trades if t.get("action") in ("UP", "DOWN")]
    skips = [t for t in trades if t.get("action") == "SKIP"]
    resolved = [t for t in executed if t.get("result") is not None]

    wins = sum(1 for t in resolved if t["result"] == "WIN")
    losses = len(resolved) - wins

    pnls = [t.get("pnl", 0.0) for t in resolved if t.get("pnl") is not None]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / len(pnls) if pnls else 0.0

    # Sharpe ratio (annualized, assuming 288 trades/day)
    if len(pnls) >= 2:
        mean_pnl = total_pnl / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.001
        # Annualize: sqrt(288 * 365) periods per year
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

    # Confidence stats
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


def run_test(decide_fn, day: int,
             duration_hours: float = 24.0,
             entry_seconds: int = 30,
             bet_size: float = 1.0,
             dry_run: bool = True) -> dict:
    """
    Run a strategy against live BTC 5-min markets.

    Args:
        decide_fn: The strategy's decide(context) -> dict function
        day: Day number (for logging)
        duration_hours: How long to run the test
        entry_seconds: How many seconds before window close to make decisions
        bet_size: Simulated bet size in USD
        dry_run: If True, don't place real orders

    Returns:
        {
            "trades": list[dict],   -- all trade entries
            "metrics": dict,        -- computed metrics
            "duration_hours": float,
        }
    """
    log_path = STRATEGIES_DIR / f"day_{day:02d}_trades.jsonl"
    trade_results = _load_trades(log_path)  # Resume if interrupted

    # Track which windows we've already traded
    traded_windows = {t.get("window_start") for t in trade_results}

    start_time = time.time()
    end_time = start_time + duration_hours * 3600
    current_window = None
    ptb = None

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"\n  {'='*60}")
    print(f"  STRATEGY TESTER -- Day {day}")
    print(f"  Mode: {mode} | Duration: {duration_hours}h | Entry: T-{entry_seconds}s")
    print(f"  Bet size: ${bet_size:.2f}")
    print(f"  {'='*60}\n")

    try:
        while time.time() < end_time:
            ts = int(time.time())
            window_start = ts - (ts % 300)
            window_end = window_start + 300
            seconds_left = window_end - ts
            time_in_window = 300 - seconds_left

            # New window
            if window_start != current_window:
                current_window = window_start

                # Get PTB for this window
                ptb_api = get_ptb_from_api(window_start)
                if ptb_api:
                    ptb = ptb_api
                else:
                    ptb = get_live_btc_price()

                if ptb:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{now}] Window {window_start} | PTB: ${ptb:,.2f}")

            # Decision zone: entry_seconds before window close
            if (entry_seconds >= seconds_left > 3
                    and window_start not in traded_windows
                    and ptb):

                live_price = get_live_btc_price()
                if not live_price:
                    time.sleep(2)
                    continue

                # Get price history
                price_history = _get_price_history(20)

                # Get market odds
                market = get_current_market(window_start)
                if market and market.get("prices"):
                    market_odds = {
                        "up": market["prices"][0],
                        "down": market["prices"][1],
                    }
                else:
                    # Estimate from price delta
                    delta = (live_price - ptb) / ptb * 100 if ptb else 0
                    up_odds = min(0.95, max(0.05, 0.5 + delta * 5))
                    market_odds = {"up": round(up_odds, 3), "down": round(1 - up_odds, 3)}

                # Build context and call strategy
                context = _build_context(
                    btc_price=live_price,
                    price_history=price_history,
                    market_odds=market_odds,
                    time_in_window=time_in_window,
                    trade_results=trade_results,
                    volume=150000.0,  # placeholder
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

                now = datetime.now().strftime("%H:%M:%S")

                if action == "SKIP":
                    print(f"  [{now}] SKIP | conf={confidence:.2f} | {reasoning[:60]}")
                    entry = {
                        "window_start": window_start,
                        "action": "SKIP",
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "btc_price": live_price,
                        "ptb": ptb,
                        "market_odds": market_odds,
                        "result": None,
                        "pnl": None,
                    }
                    _log_trade(log_path, entry)
                    trade_results.append(entry)
                    traded_windows.add(window_start)

                elif action in ("UP", "DOWN"):
                    delta_pct = (live_price - ptb) / ptb * 100 if ptb else 0

                    print(f"\n  {'='*55}")
                    print(f"  [{now}] {action} | conf={confidence:.2f} | {seconds_left}s left")
                    print(f"  delta={delta_pct:+.4f}% | PTB=${ptb:,.2f} | Live=${live_price:,.2f}")
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

                    # Check resolution
                    resolution = get_market_resolution(window_start)
                    close_price = get_live_btc_price()

                    if resolution:
                        won = resolution == action
                    elif close_price and ptb:
                        actual = "UP" if close_price >= ptb else "DOWN"
                        won = actual == action
                    else:
                        won = None

                    if won is not None:
                        if won:
                            # Approximate payout from market odds
                            buy_price = market_odds.get("up" if action == "UP" else "down", 0.5)
                            pnl = bet_size * (1.0 - buy_price)
                        else:
                            buy_price = market_odds.get("up" if action == "UP" else "down", 0.5)
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

    # Final metrics
    metrics = compute_metrics(trade_results)
    actual_hours = (time.time() - start_time) / 3600

    print(f"\n  {'='*60}")
    print(f"  DAY {day} TEST COMPLETE")
    print(f"  Duration: {actual_hours:.1f}h")
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


def run_backtest(decide_fn, day: int,
                 candles: list[dict] = None,
                 bet_size: float = 1.0) -> dict:
    """
    Run a strategy against historical candle data (offline backtest).

    Simulates 5-min windows using actual candle OHLC data.
    Useful for quick testing before committing to live dry-run.

    Args:
        decide_fn: The strategy's decide(context) function
        day: Day number (for logging)
        candles: Historical candle data (from fetch_prices). If None, loads cache.
        bet_size: Simulated bet size

    Returns:
        Same format as run_test()
    """
    if candles is None:
        from data.fetch_prices import load_cached_candles
        candles = load_cached_candles()

    if len(candles) < 25:
        logger.error("Not enough candle data for backtest (need at least 25)")
        return {"trades": [], "metrics": compute_metrics([]), "duration_hours": 0}

    trade_results = []

    # Group candles into 5-min windows (every 5 candles if 1-min, or use directly if 5-min)
    # Assume candles are 5-min granularity (300s)
    for i in range(20, len(candles)):
        candle = candles[i]
        ptb = candle["open"]
        close_price = candle["close"]

        # Build price history from previous candles
        history_candles = candles[max(0, i - 20):i]
        price_history = [c["close"] for c in history_candles]

        # Simulate market odds (neutral since we don't have historical odds)
        delta = (close_price - ptb) / ptb * 100 if ptb else 0
        market_odds = {"up": 0.50, "down": 0.50}

        context = _build_context(
            btc_price=ptb,  # Use open as "current" price
            price_history=price_history,
            market_odds=market_odds,
            time_in_window=240.0,  # Simulate near end of window
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
            actual = "UP" if close_price >= ptb else "DOWN"
            won = actual == action

            buy_price = market_odds.get("up" if action == "UP" else "down", 0.5)
            if won:
                pnl = bet_size * (1.0 - buy_price)
            else:
                pnl = -bet_size * buy_price

            trade_results.append({
                "window_start": candle["open_time"],
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "btc_price": ptb,
                "ptb": ptb,
                "close_price": close_price,
                "market_odds": market_odds,
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
                "result": None,
                "pnl": None,
            })

    metrics = compute_metrics(trade_results)
    return {
        "trades": trade_results,
        "metrics": metrics,
        "duration_hours": len(candles) * 5 / 60,  # Approximate
    }
