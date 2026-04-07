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

    Uses three complementary approaches to discover markets:
    1. Tag-based search: GET /events?tag=Weather&closed=false&limit=100
    2. Slug-contains search: GET /events?slug_contains=temperature&closed=false&limit=100
    3. Text search: GET /public-search?q=temperature&events_status=active

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
    seen_slugs = set()

    def _collect(events):
        for event in events:
            parsed = _parse_weather_event(event, cities)
            if parsed and parsed["slug"] not in seen_slugs:
                seen_slugs.add(parsed["slug"])
                markets.append(parsed)

    # --- Approach 1: Tag-based search (tag is case-sensitive: "Weather") ---
    try:
        r = requests.get(
            f"{GAMMA_HOST}/events",
            params={"tag": "Weather", "closed": "false", "limit": 100},
            timeout=15,
        )
        if r.status_code == 200:
            _collect(r.json())
            logger.debug(f"Tag=Weather returned {len(r.json())} events")
    except requests.RequestException as e:
        logger.warning(f"Tag search failed: {e}")

    # --- Approach 2: Slug-contains search (undocumented but used in the wild) ---
    slug_patterns = ["temperature", "weather", "temp-"]
    for pattern in slug_patterns:
        try:
            r = requests.get(
                f"{GAMMA_HOST}/events",
                params={"slug_contains": pattern, "closed": "false", "limit": 100},
                timeout=15,
            )
            if r.status_code == 200:
                _collect(r.json())
                logger.debug(f"slug_contains={pattern} returned {len(r.json())} events")
        except requests.RequestException as e:
            logger.warning(f"Slug search for '{pattern}' failed: {e}")

    # --- Approach 3: Public text search endpoint ---
    search_queries = ["highest temperature", "temperature"]
    if cities:
        search_queries = [f"highest temperature {c}" for c in cities] + search_queries

    for query in search_queries:
        try:
            r = requests.get(
                f"{GAMMA_HOST}/public-search",
                params={
                    "q": query,
                    "events_status": "active",
                    "limit_per_type": 20,
                },
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                # public-search returns {"events": [...], "markets": [...], ...}
                events = []
                if isinstance(data, dict):
                    events = data.get("events", [])
                elif isinstance(data, list):
                    events = data
                _collect(events)
                logger.debug(f"public-search q='{query}' returned {len(events)} events")
        except requests.RequestException as e:
            logger.warning(f"Public search for '{query}' failed: {e}")

    logger.info(f"Found {len(markets)} weather markets")
    return markets


def _parse_weather_event(event: dict, cities: list[str] = None) -> dict | None:
    """Parse a GAMMA API event into our weather market format.

    Handles both bracket-style markets ("50-51°F") and threshold-style
    markets ("Will NYC high exceed 55°F?"). Also checks the groupItemTitle
    field which some markets use instead of question.
    """
    title = event.get("title", "")
    slug = event.get("slug", "")

    # Check if this is a temperature market (check title, slug, and description)
    searchable = f"{title} {slug} {event.get('description', '')}"
    if not re.search(r'(?i)(temperature|temp\b|°f|degrees)', searchable):
        return None

    # Extract city
    city = _extract_city(title)

    # If cities filter is set, skip non-matching cities
    if cities and city not in cities and city != title:
        return None

    # If we couldn't match a known city, use the title as fallback
    if not city:
        city = title

    # Extract date from title (handles "April 8", "on April 8", "Apr 8, 2026")
    date_str = ""
    date_match = re.search(
        r'(?:on\s+)?(\w+\.?\s+\d{1,2}(?:\s*,?\s*\d{4})?)', title
    )
    if date_match:
        date_str = date_match.group(1).strip().rstrip(",")

    # Parse brackets/outcomes from the event's sub-markets
    brackets = []
    event_markets = event.get("markets", [])
    for market in event_markets:
        bracket = _parse_market_bracket(market)
        if bracket:
            brackets.append(bracket)

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


# City alias map — matches Polymarket naming conventions
CITY_ALIASES = {
    "NYC": [r'NYC', r'New York', r'LaGuardia', r'KLGA'],
    "Chicago": [r'Chicago', r"O'?Hare", r'KORD'],
    "LA": [r'LA\b', r'Los Angeles', r'LAX', r'KLAX'],
    "SF": [r'SF\b', r'San Francisco', r'SFO', r'KSFO'],
    "Atlanta": [r'Atlanta', r'Hartsfield', r'KATL'],
    "Miami": [r'Miami', r'KMIA'],
    "Denver": [r'Denver', r'KDEN'],
    "London": [r'London', r'EGLC'],
    "Helsinki": [r'Helsinki'],
    "Hong Kong": [r'Hong Kong'],
    "Seoul": [r'Seoul'],
}


def _extract_city(text: str) -> str | None:
    """Extract a recognized city from market text."""
    for city_key, aliases in CITY_ALIASES.items():
        pattern = r'(?i)\b(' + '|'.join(aliases) + r')\b'
        if re.search(pattern, text):
            return city_key
    return None


def _parse_market_bracket(market: dict) -> dict | None:
    """Parse a single sub-market into a bracket dict.

    Handles multiple question formats:
    - Bracket range: "50-51" or "50–51" in question/groupItemTitle
    - Threshold: "55°F or higher" / "below 45°F"
    - Outcome labels: outcomes=["50-51°F", "52-53°F"]
    """
    # Try question first, then groupItemTitle (some markets use this)
    question = market.get("question", "") or market.get("groupItemTitle", "")

    # Skip closed/resolved sub-markets with extreme prices
    prices = _parse_json_field(market.get("outcomePrices", "[]"))
    if prices:
        try:
            yes_price = float(prices[0])
            if yes_price > 0.98 or yes_price < 0.02:
                return None  # Already resolved, skip
        except (ValueError, IndexError):
            pass

    tokens = _parse_json_field(market.get("clobTokenIds", "[]"))
    outcomes = _parse_json_field(market.get("outcomes", "[]"))

    # --- Try bracket range pattern: "50-51", "50–51", "50 - 51" ---
    temp_match = re.search(r'(\d{1,3})\s*[-–]\s*(\d{1,3})', question)
    if temp_match:
        low = int(temp_match.group(1))
        high = int(temp_match.group(2))
        price = float(prices[0]) if prices else 0.5
        token_id = tokens[0] if tokens else None

        return {
            "range": f"{low}-{high}",
            "low": low,
            "high": high,
            "outcome": f"{low}-{high}°F",
            "price": round(price, 4),
            "token_id": token_id,
            "market_id": market.get("id"),
            "question": question,
        }

    # --- Try threshold pattern: "55°F or higher", "below 45°F" ---
    threshold_match = re.search(
        r'(\d{1,3})\s*°?\s*[fF]?\s*(?:or\s+)?(?:higher|above|more)',
        question
    )
    if threshold_match:
        temp = int(threshold_match.group(1))
        price = float(prices[0]) if prices else 0.5
        token_id = tokens[0] if tokens else None
        return {
            "range": f"{temp}+",
            "low": temp,
            "high": temp + 20,
            "outcome": f"{temp}°F or higher",
            "price": round(price, 4),
            "token_id": token_id,
            "market_id": market.get("id"),
            "question": question,
        }

    below_match = re.search(
        r'(?:below|under|less\s+than)\s*(\d{1,3})\s*°?\s*[fF]?',
        question
    )
    if below_match:
        temp = int(below_match.group(1))
        price = float(prices[0]) if prices else 0.5
        token_id = tokens[0] if tokens else None
        return {
            "range": f"<{temp}",
            "low": temp - 20,
            "high": temp,
            "outcome": f"Below {temp}°F",
            "price": round(price, 4),
            "token_id": token_id,
            "market_id": market.get("id"),
            "question": question,
        }

    # --- Try parsing from outcomes list ---
    if outcomes:
        for idx, outcome in enumerate(outcomes):
            temp_match = re.search(r'(\d{1,3})\s*[-–]\s*(\d{1,3})', str(outcome))
            if temp_match:
                low = int(temp_match.group(1))
                high = int(temp_match.group(2))
                price = float(prices[idx]) if idx < len(prices) else 0.5
                token_id = tokens[idx] if idx < len(tokens) else None
                return {
                    "range": f"{low}-{high}",
                    "low": low,
                    "high": high,
                    "outcome": str(outcome),
                    "price": round(price, 4),
                    "token_id": token_id,
                    "market_id": market.get("id"),
                }

    return None


def _parse_json_field(raw) -> list:
    """Parse a JSON field that may be a string or already a list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    return []


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
        print("  Trying diagnostic queries...")
        # Diagnostic: show what each approach returns
        for label, params in [
            ("tag=Weather", {"tag": "Weather", "closed": "false", "limit": 5}),
            ("slug_contains=temperature", {"slug_contains": "temperature", "closed": "false", "limit": 5}),
        ]:
            try:
                r = requests.get(f"{GAMMA_HOST}/events", params=params, timeout=15)
                print(f"\n  [{label}] status={r.status_code}, count={len(r.json()) if r.status_code == 200 else 'N/A'}")
                if r.status_code == 200:
                    for e in r.json()[:3]:
                        print(f"    - {e.get('title', 'untitled')} [{e.get('slug', '')}]")
            except Exception as e:
                print(f"  [{label}] failed: {e}")
        # Also try public-search
        try:
            r = requests.get(
                f"{GAMMA_HOST}/public-search",
                params={"q": "temperature", "limit_per_type": 5},
                timeout=15,
            )
            print(f"\n  [public-search q=temperature] status={r.status_code}")
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    for key, val in data.items():
                        if isinstance(val, list):
                            print(f"    {key}: {len(val)} results")
                            for item in val[:3]:
                                print(f"      - {item.get('title', item.get('question', str(item)[:80]))}")
        except Exception as e:
            print(f"  [public-search] failed: {e}")
    else:
        for m in markets:
            print(f"\n  {m['title']}")
            print(f"  City: {m['city']} | Date: {m['date']}")
            print(f"  Brackets:")
            for b in m["brackets"][:5]:
                print(f"    {b['range']}°F @ ${b['price']:.2f}")
            if len(m["brackets"]) > 5:
                print(f"    ... and {len(m['brackets']) - 5} more")
