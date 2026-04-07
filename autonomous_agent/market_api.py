"""
Polymarket Weather Market Interface — Discovers and interacts with weather markets.

Finds active daily temperature markets, gets bracket prices, places orders,
and checks resolutions.

Weather markets on Polymarket:
- "Highest temperature in NYC on April 8?"
- Brackets in 2°F ranges: 50-51, 52-53, 54-55, etc.
- Resolution from airport weather stations via Weather Underground
"""

import re
import json
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

GAMMA_HOST = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"


def find_weather_markets(cities: list[str] = None) -> list[dict]:
    """
    Search Polymarket for active daily temperature markets.

    Returns list of markets with bracket info:
    [
        {
            "city": "NYC",
            "date": "2026-04-08",
            "title": "Highest temperature in NYC on April 8?",
            "slug": "highest-temperature-in-nyc-on-april-8",
            "event_id": "12345",
            "brackets": [
                {"range": "54-55", "low": 54, "high": 55, "outcome": "54-55°F",
                 "price": 0.42, "token_id": "abc123"},
                ...
            ],
            "active": True,
            "closed": False,
        }
    ]
    """
    markets = []

    # Search for temperature markets
    search_terms = ["highest temperature", "temperature"]
    if cities:
        search_terms = [f"highest temperature in {c}" for c in cities]

    for term in search_terms:
        try:
            r = requests.get(
                f"{GAMMA_HOST}/events",
                params={"tag": "temperature", "active": "true", "closed": "false", "limit": 50},
                timeout=15,
            )

            if r.status_code == 200:
                events = r.json()
                for event in events:
                    parsed = _parse_weather_event(event)
                    if parsed:
                        markets.append(parsed)

            # Also try searching by title
            r = requests.get(
                f"{GAMMA_HOST}/events",
                params={"title": term, "active": "true", "closed": "false", "limit": 50},
                timeout=15,
            )

            if r.status_code == 200:
                events = r.json()
                for event in events:
                    parsed = _parse_weather_event(event)
                    if parsed and parsed["slug"] not in [m["slug"] for m in markets]:
                        markets.append(parsed)

        except requests.RequestException as e:
            logger.warning(f"Market search failed for '{term}': {e}")

    # Try the series slug approach
    try:
        r = requests.get(
            f"{GAMMA_HOST}/events",
            params={"seriesSlug": "nyc-temperature", "active": "true", "limit": 20},
            timeout=15,
        )
        if r.status_code == 200:
            for event in r.json():
                parsed = _parse_weather_event(event)
                if parsed and parsed["slug"] not in [m["slug"] for m in markets]:
                    markets.append(parsed)
    except requests.RequestException:
        pass

    logger.info(f"Found {len(markets)} weather markets")
    return markets


def _parse_weather_event(event: dict) -> dict | None:
    """Parse a GAMMA API event into our weather market format."""
    title = event.get("title", "")
    slug = event.get("slug", "")

    # Check if this is a temperature market
    if not re.search(r'(?i)(temperature|temp)', title):
        return None

    # Extract city
    city = None
    city_patterns = {
        "NYC": r'(?i)\b(NYC|New York|LaGuardia)\b',
        "Chicago": r'(?i)\b(Chicago|O\'?Hare)\b',
        "LA": r'(?i)\b(LA|Los Angeles|LAX)\b',
        "SF": r'(?i)\b(SF|San Francisco|SFO)\b',
        "Atlanta": r'(?i)\b(Atlanta|Hartsfield)\b',
        "London": r'(?i)\b(London)\b',
    }
    for city_key, pattern in city_patterns.items():
        if re.search(pattern, title):
            city = city_key
            break

    if not city:
        # Try to extract any city name
        city = title  # Use full title as identifier

    # Extract date from title
    date_match = re.search(r'(?:on\s+)?(\w+\s+\d{1,2})', title)
    date_str = date_match.group(1) if date_match else ""

    # Parse brackets from markets
    brackets = []
    event_markets = event.get("markets", [])
    for market in event_markets:
        question = market.get("question", "")
        outcomes = market.get("outcomes", "")

        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except (json.JSONDecodeError, TypeError):
                outcomes = []

        prices_raw = market.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            try:
                prices = json.loads(prices_raw)
            except (json.JSONDecodeError, TypeError):
                prices = []
        else:
            prices = prices_raw

        tokens_raw = market.get("clobTokenIds", "[]")
        if isinstance(tokens_raw, str):
            try:
                tokens = json.loads(tokens_raw)
            except (json.JSONDecodeError, TypeError):
                tokens = []
        else:
            tokens = tokens_raw

        # Parse temperature range from question or outcomes
        temp_match = re.search(r'(\d{1,3})\s*[-–]\s*(\d{1,3})', question)
        if temp_match:
            low = int(temp_match.group(1))
            high = int(temp_match.group(2))
            price = float(prices[0]) if prices else 0.5
            token_id = tokens[0] if tokens else None

            brackets.append({
                "range": f"{low}-{high}",
                "low": low,
                "high": high,
                "outcome": f"{low}-{high}°F",
                "price": round(price, 4),
                "token_id": token_id,
                "market_id": market.get("id"),
                "question": question,
            })
        elif outcomes:
            # Try to parse from outcomes list
            for idx, outcome in enumerate(outcomes):
                temp_match = re.search(r'(\d{1,3})', str(outcome))
                if temp_match:
                    temp = int(temp_match.group(1))
                    price = float(prices[idx]) if idx < len(prices) else 0.5
                    token_id = tokens[idx] if idx < len(tokens) else None
                    brackets.append({
                        "range": str(outcome),
                        "low": temp,
                        "high": temp + 1,
                        "outcome": str(outcome),
                        "price": round(price, 4),
                        "token_id": token_id,
                        "market_id": market.get("id"),
                    })

    if not brackets:
        return None

    # Sort brackets by temperature
    brackets.sort(key=lambda b: b["low"])

    return {
        "city": city,
        "date": date_str,
        "title": title,
        "slug": slug,
        "event_id": event.get("id"),
        "brackets": brackets,
        "active": event.get("active", True),
        "closed": event.get("closed", False),
    }


def get_bracket_prices(event_slug: str) -> list[dict]:
    """Get current prices for all brackets in a weather market."""
    try:
        r = requests.get(f"{GAMMA_HOST}/events?slug={event_slug}", timeout=10)
        r.raise_for_status()
        events = r.json()
        if events:
            parsed = _parse_weather_event(events[0])
            if parsed:
                return parsed["brackets"]
    except requests.RequestException as e:
        logger.warning(f"Failed to get bracket prices for {event_slug}: {e}")
    return []


def check_market_resolution(event_slug: str) -> dict | None:
    """
    Check if a weather market has resolved and what the outcome was.

    Returns:
        {"resolved": True, "winning_bracket": "54-55", "actual_temp": 55}
        or None if not resolved.
    """
    try:
        r = requests.get(f"{GAMMA_HOST}/events?slug={event_slug}", timeout=10)
        r.raise_for_status()
        events = r.json()

        if not events:
            return None

        event = events[0]
        if not event.get("closed"):
            return None

        # Find which bracket won (price closest to 1.0)
        markets = event.get("markets", [])
        winning_bracket = None
        actual_temp = None

        for market in markets:
            prices_raw = market.get("outcomePrices", "[]")
            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw

            if prices and float(prices[0]) > 0.9:
                question = market.get("question", "")
                temp_match = re.search(r'(\d{1,3})\s*[-–]\s*(\d{1,3})', question)
                if temp_match:
                    winning_bracket = f"{temp_match.group(1)}-{temp_match.group(2)}"
                    actual_temp = int(temp_match.group(1))  # Lower bound of winning bracket

        if winning_bracket:
            return {
                "resolved": True,
                "winning_bracket": winning_bracket,
                "actual_temp": actual_temp,
            }

    except requests.RequestException as e:
        logger.warning(f"Resolution check failed for {event_slug}: {e}")

    return None


def place_bracket_order(token_id: str, amount: float, dry_run: bool = True) -> bool:
    """
    Place an order on a specific temperature bracket.

    For now, this uses the same CLOB approach as the BTC trader.
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would buy token {token_id} for ${amount:.2f}")
        return True

    try:
        import os
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        private_key = os.getenv("PRIVATE_KEY", "")
        safe_address = os.getenv("SAFE_ADDRESS", "")

        client = ClobClient(
            host=CLOB_HOST, key=private_key,
            chain_id=POLYGON, signature_type=2, funder=safe_address,
        )
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

        args = MarketOrderArgs(token_id=token_id, amount=float(amount),
                              side=BUY, order_type=OrderType.FOK)
        signed = client.create_market_order(args)
        result = client.post_order(signed, OrderType.FOK)

        return result.get("success", False)
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return False


# --- CLI for testing ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Polymarket Weather Market Scanner\n")
    print("Searching for active weather markets...")

    markets = find_weather_markets(["NYC", "Chicago", "LA"])

    if not markets:
        print("  No weather markets found via standard search.")
        print("  Trying direct search...")
        # Try a broader search
        try:
            r = requests.get(
                f"{GAMMA_HOST}/events",
                params={"limit": 20, "active": "true"},
                timeout=15,
            )
            if r.status_code == 200:
                events = r.json()
                for e in events[:5]:
                    print(f"  - {e.get('title', 'untitled')} [{e.get('slug', '')}]")
        except Exception as e:
            print(f"  Search failed: {e}")
    else:
        for m in markets:
            print(f"\n  {m['title']}")
            print(f"  City: {m['city']} | Date: {m['date']}")
            print(f"  Brackets:")
            for b in m["brackets"][:5]:
                print(f"    {b['range']}°F @ ${b['price']:.2f}")
            if len(m["brackets"]) > 5:
                print(f"    ... and {len(m['brackets']) - 5} more")
