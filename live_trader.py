"""
Live RL Trader — Uses the trained PPO model + Qwen strategy config for real trading.

Key improvements over original:
- Gets PTB from Polymarket GAMMA API (no Playwright needed)
- Uses Chainlink-aligned prices (Polymarket resolves against Chainlink BTC/USD)
- Strategy config acts as gate on RL model decisions (Qwen's updates matter)
- Checks actual market resolution via API for accurate win/loss tracking
- Proper error logging (no silent except:pass)

Usage:
    python live_trader.py           # Dry run
    python live_trader.py --live    # Real trades (needs VPN Japan)
"""

import time
import json
import sys
import os
import re
import logging
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from data.fetch_prices import get_ptb_from_api, get_market_resolution, get_live_btc_price

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "configs" / "strategy.json"
TRADE_LOG_PATH = Path(__file__).parent / "logs" / "live_trades.jsonl"
MODEL_PATH = Path(__file__).parent / "models" / "best_model.zip"
GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
SAFE_ADDRESS = os.getenv("SAFE_ADDRESS", "")


def load_strategy_config() -> dict:
    """Load current strategy from Qwen-updated config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg
    return {
        "min_confidence": 0.80,
        "entry_seconds": 30,
        "max_entry_price": 0.95,
        "bet_size": 1.00,
        "min_abs_delta_pct": 0.02,
        "avoid_hours_utc": [],
        "max_consecutive_losses_pause": 5,
    }


def get_btc_price() -> float | None:
    """Get live BTC price. Uses the same function as fetch_prices module."""
    return get_live_btc_price()


def get_price_to_beat(window_start: int) -> float | None:
    """
    Get Price To Beat from Polymarket GAMMA API.
    Parses the market question text for the exact Chainlink reference price.
    Falls back to CEX price if API doesn't return PTB.
    """
    ptb = get_ptb_from_api(window_start)
    if ptb:
        return ptb

    # Fallback: CEX price (less accurate but better than nothing)
    logger.warning("Could not get PTB from API, falling back to CEX price")
    return get_btc_price()


def build_observation(ptb: float, live_price: float, time_remaining_sec: int,
                      recent_wins: int, recent_total: int, recent_pnl: float,
                      bankroll: float, initial_bankroll: float) -> np.ndarray:
    """Build observation vector matching training environment."""
    import math

    if not ptb or not live_price or ptb == 0:
        return np.zeros(10, dtype=np.float32)

    delta_pct = (live_price - ptb) / ptb * 100
    delta_normalized = float(np.tanh(delta_pct / 0.1))
    time_remaining_pct = time_remaining_sec / 300

    # Volatility (simplified)
    vol = abs(delta_pct) / 10  # rough proxy

    recent_wr = recent_wins / max(recent_total, 1)
    recent_pnl_norm = float(np.tanh(recent_pnl / 10))

    # Time of day
    hour = (int(time.time()) % 86400) / 3600
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    # Token prices
    abs_delta = abs(delta_pct)
    if abs_delta >= 0.10: wp = 0.92
    elif abs_delta >= 0.05: wp = 0.80
    elif abs_delta >= 0.02: wp = 0.65
    elif abs_delta >= 0.005: wp = 0.55
    else: wp = 0.50

    yes_price = wp if delta_pct >= 0 else 1 - wp
    no_price = 1 - yes_price
    bankroll_health = min(bankroll / initial_bankroll, 2.0) - 1.0

    return np.array([
        delta_normalized, time_remaining_pct, vol, recent_wr, recent_pnl_norm,
        hour_sin, hour_cos, yes_price, no_price, bankroll_health
    ], dtype=np.float32)


def get_current_market(window_start: int) -> dict | None:
    """Get current Polymarket market data."""
    slug = f"btc-updown-5m-{window_start}"
    try:
        r = requests.get(f"{GAMMA_HOST}/events?slug={slug}", timeout=10)
        r.raise_for_status()
        events = r.json()
        if events and events[0].get("markets"):
            m = events[0]["markets"][0]
            tokens = json.loads(m.get("clobTokenIds", "[]"))
            prices = json.loads(m.get("outcomePrices", "[]"))
            return {
                "token_yes": tokens[0] if tokens else None,
                "token_no": tokens[1] if len(tokens) > 1 else None,
                "prices": [float(p) for p in prices] if prices else [0.5, 0.5],
                "question": m.get("question", ""),
                "closed": m.get("closed", False),
            }
    except requests.RequestException as e:
        logger.error(f"GAMMA API error: {e}")
    return None


def apply_strategy_gate(action: int, delta_pct: float, config: dict,
                        consecutive_losses: int) -> tuple[int, str | None]:
    """
    Apply Qwen's strategy config as a gate on RL model decisions.
    Returns (possibly overridden action, override reason or None).

    This is how Qwen's config updates affect the RL model's live behavior.
    """
    if action == 0:  # SKIP — no gate needed
        return action, None

    abs_delta = abs(delta_pct)

    # Gate 1: Minimum delta threshold
    min_delta = config.get("min_abs_delta_pct", 0.02)
    if abs_delta < min_delta:
        return 0, f"delta {abs_delta:.4f}% < min {min_delta}%"

    # Gate 2: Time-of-day filter
    current_hour_utc = (int(time.time()) % 86400) // 3600
    avoid_hours = config.get("avoid_hours_utc", [])
    if current_hour_utc in avoid_hours:
        return 0, f"hour {current_hour_utc} UTC in avoid list"

    # Gate 3: Consecutive loss circuit breaker
    max_losses = config.get("max_consecutive_losses_pause", 5)
    if consecutive_losses >= max_losses:
        return 0, f"{consecutive_losses} consecutive losses >= {max_losses}"

    return action, None


def place_fok_order(direction: str, amount: float) -> bool:
    """Place a FOK order via CLOB."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        ts = int(time.time())
        window_start = ts - (ts % 300)
        market = get_current_market(window_start)
        if not market:
            return False

        token_id = market["token_yes"] if direction == "UP" else market["token_no"]

        client = ClobClient(
            host=CLOB_HOST, key=PRIVATE_KEY,
            chain_id=POLYGON, signature_type=2, funder=SAFE_ADDRESS,
        )
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

        args = MarketOrderArgs(token_id=token_id, amount=float(amount),
                              side=BUY, order_type=OrderType.FOK)
        signed = client.create_market_order(args)
        result = client.post_order(signed, OrderType.FOK)

        return result.get("success", False)
    except Exception as e:
        logger.error(f"Order failed: {e}")
        return False


def check_resolution(window_start: int, direction: str, ptb: float) -> tuple[bool | None, float | None]:
    """
    Check actual market resolution via GAMMA API.
    Falls back to CEX price comparison if API doesn't have resolution yet.

    Returns (won: bool|None, close_price: float|None)
    """
    # Try API resolution first (most accurate)
    resolution = get_market_resolution(window_start)
    if resolution:
        won = resolution == direction
        close_price = get_btc_price()  # For logging only
        return won, close_price

    # Fallback: CEX price comparison (less accurate)
    close_price = get_btc_price()
    if close_price and ptb:
        actual = "UP" if close_price >= ptb else "DOWN"
        return actual == direction, close_price

    return None, close_price


def log_trade(trade: dict):
    """Log trade to file for Qwen analysis."""
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADE_LOG_PATH, "a") as f:
        f.write(json.dumps(trade) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 60)
    print("  RL TRADER + QWEN BRAIN -- Live Trading System")
    print(f"  Mode: {'LIVE' if args.live else 'DRY RUN'}")
    print("  Price source: Chainlink via GAMMA API (PTB) + CEX (live)")
    print("=" * 60)

    # Load RL model
    model = None
    try:
        from stable_baselines3 import PPO
        if MODEL_PATH.exists():
            model = PPO.load(str(MODEL_PATH), device="cpu")
            print(f"  RL model loaded: {MODEL_PATH}")
        else:
            print(f"  No trained model found at {MODEL_PATH}")
            print(f"  Run: python agent/train.py first")
    except ImportError:
        print("  stable-baselines3 not installed -- using rule-based fallback")

    # State
    config = load_strategy_config()
    bankroll = 25.0
    initial_bankroll = bankroll
    recent_wins = 0
    recent_total = 0
    recent_pnl = 0.0
    consecutive_losses = 0

    print(f"\n  Strategy config (v{config.get('version', 0)}):")
    print(f"    Min confidence: {config.get('min_confidence', 0.80):.0%}")
    print(f"    Min delta: {config.get('min_abs_delta_pct', 0.02):.3f}%")
    print(f"    Entry: T-{config.get('entry_seconds', 30)}s")
    print(f"    Bet size: ${config.get('bet_size', 1.00):.2f}")
    print(f"\n  Monitoring BTC... (Ctrl+C to stop)\n")

    current_window = None
    ptb = None
    traded_this_window = False

    try:
        while True:
            ts = int(time.time())
            window_start = ts - (ts % 300)
            window_end = window_start + 300
            seconds_left = window_end - ts
            entry_sec = config.get("entry_seconds", 30)

            # New window -- get PTB
            if window_start != current_window:
                current_window = window_start
                traded_this_window = False
                config = load_strategy_config()  # Reload in case Qwen updated it

                print(f"\n  Window {window_start}")
                ptb = get_price_to_beat(window_start)
                if ptb:
                    print(f"  PTB: ${ptb:,.2f} (from {'API' if get_ptb_from_api(window_start) else 'CEX fallback'})")
                else:
                    print(f"  Could not get PTB for this window")

            live_price = get_btc_price()

            # Trading zone
            if entry_sec >= seconds_left > 3 and not traded_this_window and ptb and live_price:
                obs = build_observation(
                    ptb, live_price, seconds_left,
                    recent_wins, recent_total, recent_pnl,
                    bankroll, initial_bankroll
                )

                delta_pct = (live_price - ptb) / ptb * 100

                # Get action from RL model (or rule-based fallback)
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    # Rule-based fallback
                    abs_delta = abs(delta_pct)
                    if abs_delta >= 0.10: conf = 0.95
                    elif abs_delta >= 0.05: conf = 0.80
                    elif abs_delta >= 0.02: conf = 0.60
                    else: conf = 0.20

                    if conf >= config.get("min_confidence", 0.80):
                        action = 1 if delta_pct >= 0 else 2
                    else:
                        action = 0

                # Apply Qwen's strategy gate (works for BOTH RL and rule-based)
                original_action = action
                action, gate_reason = apply_strategy_gate(
                    action, delta_pct, config, consecutive_losses
                )

                action_names = {0: "SKIP", 1: "BUY UP", 2: "BUY DOWN"}
                now = datetime.now().strftime("%H:%M:%S")

                if gate_reason and original_action != 0:
                    print(f"  [{now}] GATE: {action_names[original_action]} -> SKIP | {gate_reason}")

                if action == 0:
                    print(f"  [{now}] SKIP | delta {delta_pct:+.4f}% | {seconds_left}s")
                else:
                    direction = "UP" if action == 1 else "DOWN"
                    bet = config.get("bet_size", 1.00)

                    print(f"\n  {'='*55}")
                    print(f"  {action_names[action]} | {seconds_left}s left")
                    print(f"  delta {delta_pct:+.4f}% | PTB: ${ptb:,.2f} | Live: ${live_price:,.2f}")
                    print(f"  Bet: ${bet:.2f}")

                    success = False
                    if args.live:
                        success = place_fok_order(direction, bet)
                    else:
                        print(f"  [DRY RUN] Would buy {direction} ${bet:.2f}")
                        success = True

                    if success:
                        traded_this_window = True
                        print(f"  Waiting {seconds_left}s for resolution...")
                        time.sleep(seconds_left + 10)

                        # Check resolution (API first, then CEX fallback)
                        won, close_price = check_resolution(window_start, direction, ptb)

                        if won is not None:
                            if won:
                                pnl = bet * 0.8  # approx
                                consecutive_losses = 0
                            else:
                                pnl = -bet
                                consecutive_losses += 1

                            bankroll += pnl
                            recent_pnl += pnl
                            recent_total += 1
                            if won:
                                recent_wins += 1

                            icon = "WIN" if won else "LOSS"
                            close_str = f"${close_price:,.2f}" if close_price else "N/A"
                            print(f"\n  {icon} | Close: {close_str} | P&L: ${pnl:+.2f}")
                            print(f"  Session: {recent_wins}W/{recent_total-recent_wins}L | "
                                  f"Bankroll: ${bankroll:.2f} | "
                                  f"Consec losses: {consecutive_losses}")

                            # Log for Qwen
                            log_trade({
                                "timestamp": datetime.now().isoformat(),
                                "direction": direction,
                                "result": "WIN" if won else "LOSS",
                                "pnl": pnl,
                                "delta_pct": delta_pct,
                                "ptb": ptb,
                                "close": close_price,
                                "entry_seconds_before_close": seconds_left,
                                "model_action": original_action,
                                "gate_override": gate_reason is not None,
                                "gate_reason": gate_reason,
                                "strategy_version": config.get("version", 0),
                                "using_rl_model": model is not None,
                            })

                        continue

            # Status
            if ptb and live_price and seconds_left % 30 == 0 and seconds_left > entry_sec:
                delta = (live_price - ptb) / ptb * 100 if ptb else 0
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] ${live_price:,.2f} "
                      f"delta {delta:+.4f}% | {seconds_left}s")

            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n  SESSION SUMMARY")
        print(f"  Trades: {recent_total} | Wins: {recent_wins}")
        wr = recent_wins / max(recent_total, 1)
        print(f"  Win Rate: {wr:.1%}")
        print(f"  P&L: ${recent_pnl:+.2f}")
        print(f"  Bankroll: ${bankroll:.2f}")


if __name__ == "__main__":
    main()
