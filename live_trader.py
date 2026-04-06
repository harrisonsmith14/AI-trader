"""
Live RL Trader — Uses the trained PPO model + Qwen strategy config for real trading.
The RL model makes fast decisions every 5 minutes.
The Qwen brain updates the config nightly.

Usage:
    python live_trader.py           # Dry run
    python live_trader.py --live    # Real trades (needs VPN Japan)
"""

import time
import json
import sys
import os
import requests
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

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
    }


def get_btc_price() -> float:
    """Get live BTC price from multiple sources."""
    sources = [
        ("coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot",
         lambda r: float(r.json()["data"]["amount"])),
        ("kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
         lambda r: float(r.json()["result"]["XXBTZUSD"]["c"][0])),
    ]
    prices = []
    for name, url, parse in sources:
        try:
            r = requests.get(url, timeout=3)
            prices.append(parse(r))
        except:
            pass
    return sum(prices) / len(prices) if prices else None


def scrape_price_to_beat(window_start: int) -> float:
    """Scrape Polymarket for Price To Beat (simplified version)."""
    # This uses the approach from pc_v5.py — browser scraping
    # For the RL trader we'll try the Playwright approach
    try:
        from playwright.sync_api import sync_playwright
        import re
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
            page = browser.new_page()
            slug = f"btc-updown-5m-{window_start}"
            page.goto(f"https://polymarket.com/event/{slug}", 
                     wait_until='domcontentloaded', timeout=45000)
            time.sleep(3)
            
            content = page.content()
            browser.close()
            
            patterns = [
                r'[Pp]rice\s*[Tt]o\s*[Bb]eat[^$]*?\$\s*([\d,]+\.?\d*)',
                r'>\s*\$\s*(6[5-9],\d{3}\.\d{2})\s*<',
                r'>\s*\$\s*(7[0-4],\d{3}\.\d{2})\s*<',
                r'>\s*\$\s*(8[0-9],\d{3}\.\d{2})\s*<',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    price = float(match.group(1).replace(',', ''))
                    if 20000 < price < 200000:
                        return price
    except:
        pass
    
    # Fallback to CEX average
    return get_btc_price()


def build_observation(ptb: float, live_price: float, time_remaining_sec: int,
                      recent_wins: int, recent_total: int, recent_pnl: float,
                      bankroll: float, initial_bankroll: float) -> np.ndarray:
    """Build observation vector matching training environment."""
    import numpy as np
    
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
    import math
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


def get_current_market(window_start: int) -> dict:
    """Get current Polymarket market data."""
    slug = f"btc-updown-5m-{window_start}"
    try:
        r = requests.get(f"{GAMMA_HOST}/events?slug={slug}", timeout=10)
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
            }
    except:
        pass
    return None


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
        print(f"  ❌ Order failed: {e}")
        return False


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
    
    print("=" * 60)
    print("  🤖 RL TRADER + 🧠 QWEN BRAIN — Live Trading System")
    print(f"  Mode: {'🔴 LIVE' if args.live else '🔵 DRY RUN'}")
    print("=" * 60)
    
    # Load RL model
    try:
        from stable_baselines3 import PPO
        if MODEL_PATH.exists():
            model = PPO.load(str(MODEL_PATH))
            print(f"  ✅ RL model loaded: {MODEL_PATH}")
        else:
            print(f"  ⚠️  No trained model found at {MODEL_PATH}")
            print(f"  Run: python agent/train.py first")
            model = None
    except ImportError:
        print("  ⚠️  stable-baselines3 not installed — using rule-based fallback")
        model = None
    
    # State
    config = load_strategy_config()
    bankroll = 25.0
    initial_bankroll = bankroll
    recent_wins = 0
    recent_total = 0
    recent_pnl = 0.0
    session_trades = []
    
    print(f"\n  Strategy config (v{config.get('version', 0)}):")
    print(f"    Min confidence: {config.get('min_confidence', 0.80):.0%}")
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
            
            # New window — get PTB
            if window_start != current_window:
                current_window = window_start
                traded_this_window = False
                config = load_strategy_config()  # Reload in case Qwen updated it
                
                print(f"\n  📌 Window {window_start}")
                ptb = scrape_price_to_beat(window_start)
                if ptb:
                    print(f"  📌 PTB: ${ptb:,.2f}")
            
            live_price = get_btc_price()
            
            # Trading zone
            if entry_sec >= seconds_left > 3 and not traded_this_window and ptb and live_price:
                obs = build_observation(
                    ptb, live_price, seconds_left,
                    recent_wins, recent_total, recent_pnl,
                    bankroll, initial_bankroll
                )
                
                # Get action from RL model (or rule-based fallback)
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    # Rule-based fallback using Qwen's config
                    delta_pct = (live_price - ptb) / ptb * 100
                    abs_delta = abs(delta_pct)
                    
                    if abs_delta >= 0.10: conf = 0.95
                    elif abs_delta >= 0.05: conf = 0.80
                    elif abs_delta >= 0.02: conf = 0.60
                    else: conf = 0.20
                    
                    if conf >= config.get("min_confidence", 0.80):
                        action = 1 if delta_pct >= 0 else 2
                    else:
                        action = 0
                
                action_names = {0: "SKIP", 1: "BUY UP", 2: "BUY DOWN"}
                now = datetime.now().strftime("%H:%M:%S")
                
                delta_pct = (live_price - ptb) / ptb * 100
                
                if action == 0:
                    print(f"  [{now}] SKIP | Δ{delta_pct:+.4f}% | {seconds_left}s")
                else:
                    direction = "UP" if action == 1 else "DOWN"
                    bet = config.get("bet_size", 1.00)
                    
                    print(f"\n  {'='*55}")
                    print(f"  🎯 {action_names[action]} | {seconds_left}s left")
                    print(f"  Δ{delta_pct:+.4f}% | PTB: ${ptb:,.2f} | Live: ${live_price:,.2f}")
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
                        time.sleep(seconds_left + 5)
                        
                        close_price = get_btc_price()
                        if close_price:
                            actual = "UP" if close_price >= ptb else "DOWN"
                            won = actual == direction
                            
                            if won:
                                pnl = bet * 0.8  # approx
                            else:
                                pnl = -bet
                            
                            bankroll += pnl
                            recent_pnl += pnl
                            recent_total += 1
                            if won:
                                recent_wins += 1
                            
                            icon = "✅ WIN" if won else "❌ LOSS"
                            print(f"\n  {icon} | Close: ${close_price:,.2f} | P&L: ${pnl:+.2f}")
                            print(f"  Session: {recent_wins}W/{recent_total-recent_wins}L | Bankroll: ${bankroll:.2f}")
                            
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
                                "confidence_tier": "80+" if abs(delta_pct) >= 0.05 else "60-80",
                                "model_action": action,
                            })
                        
                        continue
            
            # Status
            if ptb and live_price and seconds_left % 30 == 0 and seconds_left > entry_sec:
                delta = (live_price - ptb) / ptb * 100 if ptb else 0
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] ${live_price:,.2f} Δ{delta:+.4f}% | {seconds_left}s")
            
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
