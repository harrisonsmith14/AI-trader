"""
Researcher — Discovers new BTC 5-min trading strategies via web search + LLM.

Phase 1 of the daily cycle:
1. Search the web for trading strategy ideas (DuckDuckGo, no API key needed)
2. Feed search results to Qwen for analysis
3. Qwen picks or invents a strategy it hasn't tried before
4. Returns strategy description, concept, and source URLs

Falls back to Qwen's own knowledge if web search is unavailable.
"""

import json
import logging
import re
import time
import requests
from pathlib import Path
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
STRATEGY_LOG_PATH = Path(__file__).parent / "strategy_log.json"

# Search queries rotated across days to get diverse strategies
SEARCH_QUERIES = [
    "BTC short term momentum trading strategy 5 minute",
    "bitcoin mean reversion scalping strategy",
    "crypto order flow trading signals",
    "BTC volume weighted price prediction short term",
    "bitcoin RSI MACD 5 minute trading",
    "crypto microstructure trading strategy",
    "BTC bollinger bands scalping",
    "bitcoin VWAP trading strategy intraday",
    "crypto sentiment analysis trading signals",
    "BTC price action pattern recognition short term",
    "bitcoin stochastic oscillator trading",
    "crypto EMA crossover strategy 5 minute",
    "BTC support resistance breakout trading",
    "bitcoin volatility trading strategy scalping",
    "crypto rate of change momentum indicator",
    "BTC keltner channel trading strategy",
    "bitcoin fibonacci retracement scalping",
    "crypto relative volume trading signal",
    "BTC donchian channel breakout strategy",
    "bitcoin williams percent R trading",
    "crypto ATR based trading strategy short term",
]

# Strategy categories to guide diverse exploration
STRATEGY_CATEGORIES = [
    "momentum / trend following",
    "mean reversion / contrarian",
    "volume analysis / order flow",
    "volatility breakout",
    "moving average crossover",
    "oscillator-based (RSI, stochastic, etc.)",
    "price action / candlestick patterns",
    "statistical / quantitative",
    "sentiment / behavioral",
    "multi-indicator fusion",
    "channel-based (Bollinger, Keltner, Donchian)",
    "rate-of-change / momentum divergence",
    "VWAP / volume profile",
    "microstructure / spread analysis",
    "machine-learning inspired (simple rules derived from ML concepts)",
    "entropy / information-theoretic",
    "regime detection (trending vs ranging)",
    "gap / opening range breakout",
    "adaptive threshold strategies",
    "fractal / multi-timeframe analysis",
    "market making / grid-inspired",
]


def load_strategy_log() -> list[dict]:
    """Load the log of all previously tried strategies."""
    if STRATEGY_LOG_PATH.exists():
        with open(STRATEGY_LOG_PATH) as f:
            return json.load(f)
    return []


def save_strategy_log(log: list[dict]):
    """Save the strategy log."""
    STRATEGY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STRATEGY_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def _get_previous_strategy_names(log: list[dict]) -> list[str]:
    """Extract names/concepts of previously tried strategies."""
    return [entry.get("strategy_name", "") for entry in log]


def _search_web(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo HTML (no API key needed).
    Returns list of {title, url, snippet}.
    """
    results = []
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    }

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        html = r.text

        # Parse result blocks from DuckDuckGo HTML
        # Each result is in a <div class="result"> block
        result_blocks = re.findall(
            r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>.*?'
            r'<a class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )

        for href, title, snippet in result_blocks[:num_results]:
            # Clean HTML tags from title and snippet
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            results.append({
                "title": clean_title,
                "url": href,
                "snippet": clean_snippet,
            })

    except Exception as e:
        logger.warning(f"Web search failed for '{query}': {e}")

    return results


def _call_ollama(prompt: str, model: str, temperature: float = 0.5) -> str | None:
    """Call Ollama and return response text."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": 16384},
            },
            timeout=300,
        )
        if r.status_code != 200:
            logger.error(f"Ollama HTTP {r.status_code}")
            return None
        return r.json()["message"]["content"]
    except requests.RequestException as e:
        logger.error(f"Ollama connection failed: {e}")
        return None


def research_strategy(day: int, model: str = "qwen3:8b") -> dict | None:
    """
    Research and discover a new trading strategy for the given day.

    Returns:
        {
            "strategy_name": str,       -- short name (e.g., "RSI Divergence Momentum")
            "category": str,            -- strategy category
            "description": str,         -- detailed description of the strategy
            "key_concept": str,         -- core idea in 1-2 sentences
            "indicators": list[str],    -- indicators/signals used
            "sources": list[dict],      -- web sources [{title, url, snippet}]
            "day": int,
        }
        or None on failure.
    """
    log = load_strategy_log()
    previous_names = _get_previous_strategy_names(log)

    # Phase 1a: Web search for strategy ideas
    logger.info(f"Day {day}: Searching for trading strategy ideas...")

    query_idx = (day - 1) % len(SEARCH_QUERIES)
    search_results = _search_web(SEARCH_QUERIES[query_idx])

    # Also search a second query for more diversity
    alt_query_idx = (query_idx + 7) % len(SEARCH_QUERIES)
    search_results += _search_web(SEARCH_QUERIES[alt_query_idx])

    # Format search results for Qwen
    if search_results:
        sources_text = "\n".join(
            f"  [{i+1}] {r['title']}\n      URL: {r['url']}\n      {r['snippet']}"
            for i, r in enumerate(search_results)
        )
    else:
        sources_text = "  (Web search unavailable -- use your own knowledge of trading strategies)"

    # Phase 1b: Get the suggested category for this day
    category_idx = (day - 1) % len(STRATEGY_CATEGORIES)
    suggested_category = STRATEGY_CATEGORIES[category_idx]

    # Phase 1c: Build prompt for Qwen
    previous_str = "\n".join(f"  - Day {i+1}: {name}" for i, name in enumerate(previous_names))
    if not previous_str:
        previous_str = "  (None yet -- this is day 1)"

    prompt = f"""You are a quantitative trading researcher. Your task is to identify a NEW trading strategy
for BTC 5-minute UP/DOWN binary markets on Polymarket.

## How BTC 5-Min Markets Work
- Every 5 minutes, a new market asks: "Will BTC go UP or DOWN?"
- Binary outcome: UP or DOWN (based on BTC price at start vs end of 5-min window)
- Each side has a price (e.g., UP @ $0.52, DOWN @ $0.48)
- If you buy UP at $0.52 and BTC goes up, you get $1.00 (profit: $0.48)
- If BTC goes down, you lose $0.52

## Web Search Results (inspiration)
{sources_text}

## Suggested Category for Today
{suggested_category}
(You can pick a different category if something more interesting emerges from the research)

## Previously Tried Strategies (DO NOT REPEAT)
{previous_str}

## Your Task
1. Based on the search results and your knowledge, identify ONE specific trading strategy
2. It MUST be different from all previously tried strategies
3. It should be applicable to 5-minute BTC price movements
4. It should be implementable with just price history data (OHLC candles, computed indicators)
5. You can INVENT a novel strategy by combining ideas -- don't just copy

## Output Format (respond with ONLY this JSON, no other text)
```json
{{
    "strategy_name": "Short descriptive name (3-5 words)",
    "category": "Category from the list above",
    "description": "Detailed 3-5 sentence description of how the strategy works. Be specific about entry/exit signals, thresholds, and logic.",
    "key_concept": "Core idea in 1-2 sentences",
    "indicators": ["indicator1", "indicator2", "..."],
    "source_indices": [1, 3]
}}
```
The source_indices should reference which search results inspired this strategy (use [] if invented from scratch).
Use ONLY ASCII characters -- no unicode, no special symbols."""

    response = _call_ollama(prompt, model, temperature=0.6)
    if not response:
        logger.error("Failed to get response from Ollama")
        return None

    # Extract JSON from response
    json_match = re.search(r'```json\s*\n(.*?)\n\s*```', response, re.DOTALL)
    if not json_match:
        # Try bare JSON
        json_match = re.search(r'\{[^{}]*"strategy_name"[^{}]*\}', response, re.DOTALL)

    if not json_match:
        logger.error("Could not parse strategy JSON from Qwen response")
        logger.debug(f"Response was: {response[:500]}")
        return None

    try:
        json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
        strategy_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from Qwen: {e}")
        return None

    # Validate required fields
    required = ["strategy_name", "category", "description", "key_concept", "indicators"]
    for field in required:
        if field not in strategy_data:
            logger.error(f"Missing field '{field}' in strategy data")
            return None

    # Check for duplicates
    new_name = strategy_data["strategy_name"].lower().strip()
    for prev in previous_names:
        if prev.lower().strip() == new_name:
            logger.warning(f"Strategy '{new_name}' already tried. Asking Qwen to pick another.")
            # Retry once with stronger constraint
            return _retry_with_different_strategy(day, model, previous_names, search_results)

    # Attach sources
    source_indices = strategy_data.get("source_indices", [])
    sources = []
    for idx in source_indices:
        if 1 <= idx <= len(search_results):
            sources.append(search_results[idx - 1])

    result = {
        "strategy_name": strategy_data["strategy_name"],
        "category": strategy_data["category"],
        "description": strategy_data["description"],
        "key_concept": strategy_data["key_concept"],
        "indicators": strategy_data["indicators"],
        "sources": sources,
        "day": day,
    }

    logger.info(f"Day {day}: Selected strategy: {result['strategy_name']} ({result['category']})")
    return result


def _retry_with_different_strategy(day: int, model: str,
                                   previous_names: list[str],
                                   search_results: list[dict]) -> dict | None:
    """Retry strategy selection with a stronger uniqueness constraint."""
    category_idx = (day - 1 + 11) % len(STRATEGY_CATEGORIES)  # Different category
    suggested_category = STRATEGY_CATEGORIES[category_idx]

    previous_str = "\n".join(f"  - {name}" for name in previous_names)

    prompt = f"""You are a quantitative trading researcher. You MUST invent a COMPLETELY NEW
trading strategy for BTC 5-minute UP/DOWN binary markets.

## CRITICAL: These strategies have ALREADY been tried. Do NOT repeat any of them:
{previous_str}

## Suggested new category: {suggested_category}

## Requirements
- Must work on 5-minute BTC price data
- Must be implementable with price history (OHLC candles)
- Must be genuinely different from all listed strategies above
- Can combine ideas or invent something novel

## Output Format (respond with ONLY this JSON)
```json
{{
    "strategy_name": "Short descriptive name (3-5 words)",
    "category": "Category",
    "description": "Detailed 3-5 sentence description",
    "key_concept": "Core idea in 1-2 sentences",
    "indicators": ["indicator1", "indicator2"],
    "source_indices": []
}}
```
Use ONLY ASCII characters."""

    response = _call_ollama(prompt, model, temperature=0.7)
    if not response:
        return None

    json_match = re.search(r'```json\s*\n(.*?)\n\s*```', response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{[^{}]*"strategy_name"[^{}]*\}', response, re.DOTALL)
    if not json_match:
        return None

    try:
        json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    return {
        "strategy_name": data.get("strategy_name", f"Day {day} Strategy"),
        "category": data.get("category", "unknown"),
        "description": data.get("description", ""),
        "key_concept": data.get("key_concept", ""),
        "indicators": data.get("indicators", []),
        "sources": [],
        "day": day,
    }
