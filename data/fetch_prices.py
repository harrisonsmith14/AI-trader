"""
Price Data Fetcher — Gets BTC/USD OHLC candles for training and backtesting.

Polymarket resolves BTC 5-min markets against Chainlink's BTC/USD data stream:
https://data.chain.link/streams/btc-usd

Chainlink Data Streams requires a paid subscription for historical data,
so we fetch from Coinbase/Kraken (which Chainlink aggregates from) as a proxy.
For live trading, the exact PTB comes from Polymarket's GAMMA API.

Usage:
    python data/fetch_prices.py                    # Fetch 30 days, save to data/candles.json
    python data/fetch_prices.py --days 7           # Fetch 7 days
    python data/fetch_prices.py --source kraken    # Use Kraken instead of Coinbase
"""

import json
import time
import requests
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(__file__).parent / "candles.json"

# --- Coinbase API ---

def fetch_coinbase_candles(days: int = 30, granularity: int = 300) -> list[dict]:
    """
    Fetch BTC-USD candles from Coinbase Exchange API.

    Coinbase returns max 300 candles per request.
    For 5-min candles (granularity=300), that's ~25 hours per request.

    Returns list of {open_time, open, high, low, close} sorted by time ascending.
    """
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    all_candles = []
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    current_start = start

    logger.info(f"Fetching {days} days of Coinbase BTC-USD candles (granularity={granularity}s)...")

    while current_start < end:
        current_end = min(current_start + timedelta(seconds=granularity * 300), end)

        params = {
            "granularity": granularity,
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.warning(f"Coinbase request failed: {e}")
            time.sleep(2)
            current_start = current_end
            continue

        # Coinbase format: [timestamp, low, high, open, close, volume]
        for candle in data:
            if len(candle) >= 5:
                all_candles.append({
                    "open_time": int(candle[0]) * 1000,  # Convert to ms
                    "open": float(candle[3]),
                    "high": float(candle[2]),
                    "low": float(candle[1]),
                    "close": float(candle[4]),
                })

        current_start = current_end
        time.sleep(0.35)  # Rate limit: ~3 requests/sec

    # Sort ascending by time, deduplicate
    all_candles.sort(key=lambda c: c["open_time"])
    seen = set()
    deduped = []
    for c in all_candles:
        if c["open_time"] not in seen:
            seen.add(c["open_time"])
            deduped.append(c)

    logger.info(f"Fetched {len(deduped)} candles from Coinbase")
    return deduped


# --- Kraken API ---

def fetch_kraken_candles(days: int = 30, interval: int = 5) -> list[dict]:
    """
    Fetch BTC-USD candles from Kraken API.

    Kraken returns up to 720 entries per request (~60 hours for 5-min candles).
    Use `since` parameter for pagination.

    Returns list of {open_time, open, high, low, close} sorted by time ascending.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    all_candles = []
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    end_ts = int(datetime.now(timezone.utc).timestamp())

    logger.info(f"Fetching {days} days of Kraken BTC-USD candles (interval={interval}m)...")

    while since < end_ts:
        params = {
            "pair": "XBTUSD",
            "interval": interval,
            "since": since,
        }

        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            logger.warning(f"Kraken request failed: {e}")
            time.sleep(2)
            since += 720 * interval * 60
            continue

        if data.get("error"):
            logger.warning(f"Kraken API error: {data['error']}")
            break

        result = data.get("result", {})
        ohlc = result.get("XXBTZUSD", [])

        if not ohlc:
            break

        for candle in ohlc:
            # Kraken format: [time, open, high, low, close, vwap, volume, count]
            all_candles.append({
                "open_time": int(candle[0]) * 1000,  # Convert to ms
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
            })

        # Use the 'last' field for pagination
        last = result.get("last")
        if last:
            new_since = int(last)
            if new_since <= since:
                break
            since = new_since
        else:
            break

        time.sleep(1)  # Kraken rate limit

    # Sort and deduplicate
    all_candles.sort(key=lambda c: c["open_time"])
    seen = set()
    deduped = []
    for c in all_candles:
        if c["open_time"] not in seen:
            seen.add(c["open_time"])
            deduped.append(c)

    logger.info(f"Fetched {len(deduped)} candles from Kraken")
    return deduped


# --- Polymarket PTB from GAMMA API ---

GAMMA_HOST = "https://gamma-api.polymarket.com"


def get_ptb_from_api(window_start: int) -> float | None:
    """
    Get the Price To Beat from Polymarket's GAMMA API.

    The PTB is in the event's eventMetadata.priceToBeat field — this is the
    exact Chainlink BTC/USD price at the window start that Polymarket uses
    for resolution. No Playwright needed.
    """
    slug = f"btc-updown-5m-{window_start}"
    try:
        r = requests.get(f"{GAMMA_HOST}/events?slug={slug}", timeout=10)
        r.raise_for_status()
        events = r.json()

        if not events:
            return None

        event = events[0]

        # Primary: eventMetadata.priceToBeat (exact Chainlink price)
        metadata = event.get("eventMetadata")
        if metadata:
            ptb = metadata.get("priceToBeat")
            if ptb:
                price = float(ptb)
                if 10000 < price < 500000:
                    return price

        # Fallback: parse from market question/description text
        import re
        markets = event.get("markets", [])
        if markets:
            market = markets[0]
            for text in [market.get("question", ""), market.get("description", ""),
                        market.get("title", ""), event.get("title", "")]:
                match = re.search(r'\$\s*([\d,]+\.?\d*)', text)
                if match:
                    price = float(match.group(1).replace(',', ''))
                    if 10000 < price < 500000:
                        return price

    except requests.RequestException as e:
        logger.warning(f"GAMMA API request failed: {e}")

    return None


def get_market_resolution(window_start: int) -> str | None:
    """
    Check actual market resolution from GAMMA API.

    Returns "UP", "DOWN", or None if not yet resolved.
    """
    slug = f"btc-updown-5m-{window_start}"
    try:
        r = requests.get(f"{GAMMA_HOST}/events?slug={slug}", timeout=10)
        r.raise_for_status()
        events = r.json()

        if not events or not events[0].get("markets"):
            return None

        market = events[0]["markets"][0]

        # Check if market is resolved
        if not market.get("closed"):
            return None

        # Check winning outcome
        outcome_prices = json.loads(market.get("outcomePrices", "[]"))
        if outcome_prices:
            # After resolution, winning outcome price = 1.0, losing = 0.0
            yes_price = float(outcome_prices[0]) if outcome_prices else 0
            if yes_price > 0.9:
                return "UP"
            elif yes_price < 0.1:
                return "DOWN"

    except requests.RequestException as e:
        logger.warning(f"GAMMA API resolution check failed: {e}")

    return None


def get_live_btc_price() -> float | None:
    """
    Get live BTC price from multiple exchange sources.
    Used as fallback when Chainlink direct access isn't available.

    Returns average of available sources, or None if all fail.
    """
    sources = [
        ("coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot",
         lambda r: float(r.json()["data"]["amount"])),
        ("kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
         lambda r: float(r.json()["result"]["XXBTZUSD"]["c"][0])),
    ]

    prices = []
    for name, url, parse in sources:
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            price = parse(r)
            prices.append(price)
        except Exception as e:
            logger.debug(f"{name} price fetch failed: {e}")

    return sum(prices) / len(prices) if prices else None


# --- Cache Management ---

def load_cached_candles(path: str | Path = DEFAULT_CACHE_PATH) -> list[dict]:
    """Load candles from local cache file."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_candles(candles: list[dict], path: str | Path = DEFAULT_CACHE_PATH):
    """Save candles to local cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(candles, f)
    logger.info(f"Saved {len(candles)} candles to {path}")


def update_cache(path: str | Path = DEFAULT_CACHE_PATH, source: str = "coinbase",
                 days: int = 30) -> list[dict]:
    """
    Fetch fresh candles and merge with existing cache.
    Only fetches data newer than the latest cached candle.
    """
    path = Path(path)
    existing = load_cached_candles(path)

    if existing:
        latest_ts = max(c["open_time"] for c in existing)
        latest_dt = datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc)
        fetch_days = max(1, (datetime.now(timezone.utc) - latest_dt).days + 1)
        logger.info(f"Cache has {len(existing)} candles, latest: {latest_dt.isoformat()}")
        logger.info(f"Fetching {fetch_days} days of new data...")
    else:
        fetch_days = days
        logger.info(f"No cache found, fetching {fetch_days} days...")

    # Fetch new data
    if source == "coinbase":
        new_candles = fetch_coinbase_candles(days=fetch_days)
    elif source == "kraken":
        new_candles = fetch_kraken_candles(days=fetch_days)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'coinbase' or 'kraken'.")

    # Merge: existing + new, deduplicate by timestamp
    all_candles = existing + new_candles
    all_candles.sort(key=lambda c: c["open_time"])

    seen = set()
    merged = []
    for c in all_candles:
        if c["open_time"] not in seen:
            seen.add(c["open_time"])
            merged.append(c)

    save_candles(merged, path)
    logger.info(f"Cache updated: {len(merged)} total candles ({len(merged) - len(existing)} new)")
    return merged


# --- CLI ---

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Fetch BTC/USD candle data for training")
    parser.add_argument("--days", type=int, default=30, help="Days of history to fetch")
    parser.add_argument("--source", choices=["coinbase", "kraken"], default="coinbase",
                       help="Data source (default: coinbase)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_CACHE_PATH),
                       help="Output file path")
    parser.add_argument("--fresh", action="store_true",
                       help="Ignore existing cache, fetch everything fresh")
    args = parser.parse_args()

    if args.fresh:
        if args.source == "coinbase":
            candles = fetch_coinbase_candles(days=args.days)
        else:
            candles = fetch_kraken_candles(days=args.days)
        save_candles(candles, args.output)
    else:
        candles = update_cache(path=args.output, source=args.source, days=args.days)

    if candles:
        first = datetime.fromtimestamp(candles[0]["open_time"] / 1000, tz=timezone.utc)
        last = datetime.fromtimestamp(candles[-1]["open_time"] / 1000, tz=timezone.utc)
        print(f"\n  Total candles: {len(candles)}")
        print(f"  Range: {first.strftime('%Y-%m-%d %H:%M')} to {last.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"  Saved to: {args.output}")
    else:
        print("  No candles fetched. Check your internet connection.")


if __name__ == "__main__":
    main()
