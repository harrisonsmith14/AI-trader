"""
Researcher --- Discovers new BTC 5-min trading strategies via Brave Search + LLM.

Phase 1 of the daily cycle:
1. Search the web via Brave Search API for trading strategy ideas
2. Fetch actual page content from top results (not just snippets)
3. Feed real strategy details to Qwen for analysis
4. Qwen picks or invents a strategy it hasn't tried before
5. Returns strategy description, concept, and source URLs

Requires BRAVE_API_KEY in environment or .env file.
"""

import json
import logging
import os
import re
import time
import requests
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

OLLAMA_URL = "http://localhost:11434"
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
STRATEGY_LOG_PATH = Path(__file__).parent / "strategy_log.json"

# Search queries rotated across days -- specific enough to find real strategies
SEARCH_QUERIES = [
    "bitcoin 5 minute scalping strategy with RSI entry rules backtest results",
    "BTC mean reversion trading strategy short timeframe quantitative",
    "crypto order flow imbalance trading strategy microstructure",
    "bitcoin VWAP deviation scalping strategy with entry exit rules",
    "BTC momentum divergence RSI MACD 5 minute trading rules",
    "crypto volatility breakout bollinger band squeeze strategy",
    "bitcoin EMA crossover scalping strategy 5 minute backtest",
    "BTC stochastic RSI oversold overbought 5 minute trading",
    "crypto rate of change momentum strategy short term signals",
    "bitcoin keltner channel ATR breakout strategy intraday",
    "BTC fibonacci retracement scalping levels strategy",
    "crypto volume profile VWAP trading strategy signals",
    "bitcoin donchian channel breakout strategy short timeframe",
    "BTC williams percent R overbought oversold trading rules",
    "crypto ATR trailing stop scalping strategy 5 minute",
    "bitcoin heikin ashi trend following strategy short term",
    "BTC ichimoku cloud scalping strategy 5 minute signals",
    "crypto on-balance volume divergence trading strategy",
    "bitcoin hull moving average crossover strategy scalping",
    "BTC elder ray bull bear power trading strategy",
    "crypto adaptive moving average regime detection strategy",
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


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter. Skips script/style tags."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._pieces.append(text)

    def get_text(self) -> str:
        return " ".join(self._pieces)


def _html_to_text(html: str) -> str:
    """Extract readable text from HTML, skipping scripts/styles."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        return parser.get_text()
    except Exception:
        # Fallback: strip all tags with regex
        return re.sub(r'<[^>]+>', ' ', html)


# ---------------------------------------------------------------------------
# Strategy log
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Brave Search API
# ---------------------------------------------------------------------------

def _brave_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Search via Brave Search API. Returns structured results with
    title, url, description, and (optionally) extra_snippets.

    Requires BRAVE_API_KEY environment variable.
    """
    if not BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not set -- web search unavailable")
        return []

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "count": num_results,
        "text_decorations": False,
        "search_lang": "en",
    }

    try:
        r = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        logger.warning(f"Brave Search failed for '{query}': {e}")
        return []

    results = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
            "extra_snippets": item.get("extra_snippets", []),
            "page_age": item.get("page_age", ""),
        })

    logger.info(f"Brave Search: {len(results)} results for '{query[:50]}...'")
    return results


def _fetch_page_content(url: str, max_chars: int = 3000) -> str:
    """
    Fetch a web page and extract its text content.
    Returns the first max_chars of readable text, or empty string on failure.
    """
    # Skip URLs that are unlikely to have useful text content
    skip_domains = ["youtube.com", "youtu.be", "twitter.com", "x.com",
                    "reddit.com", "instagram.com", "tiktok.com"]
    for domain in skip_domains:
        if domain in url:
            return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        r.raise_for_status()

        # Only process HTML
        content_type = r.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return ""

        text = _html_to_text(r.text)

        # Clean up: collapse whitespace, take first max_chars
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]

    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main research function
# ---------------------------------------------------------------------------

def research_strategy(day: int, model: str = "qwen3:8b") -> dict | None:
    """
    Research and discover a new trading strategy for the given day.

    Pipeline:
      1. Brave Search for strategy ideas (2 queries for diversity)
      2. Fetch actual page content from top 3 results
      3. Build a rich prompt with real strategy details for Qwen
      4. Qwen selects/invents a strategy, ensuring no repeats

    Returns:
        {
            "strategy_name": str,
            "category": str,
            "description": str,
            "key_concept": str,
            "indicators": list[str],
            "sources": list[dict],
            "day": int,
        }
        or None on failure.
    """
    log = load_strategy_log()
    previous_names = _get_previous_strategy_names(log)

    # ── Step 1: Brave Search ─────────────────────────────────────
    logger.info(f"Day {day}: Searching for trading strategy ideas...")

    query_idx = (day - 1) % len(SEARCH_QUERIES)
    search_results = _brave_search(SEARCH_QUERIES[query_idx], num_results=5)

    # Second query for diversity
    alt_query_idx = (query_idx + 7) % len(SEARCH_QUERIES)
    search_results += _brave_search(SEARCH_QUERIES[alt_query_idx], num_results=5)

    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for r in search_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            deduped.append(r)
    search_results = deduped

    # ── Step 2: Fetch actual page content from top results ───────
    logger.info(f"Day {day}: Fetching page content from top results...")

    pages_fetched = 0
    for result in search_results:
        if pages_fetched >= 3:
            break
        content = _fetch_page_content(result["url"])
        if content and len(content) > 200:
            result["page_content"] = content
            pages_fetched += 1
            logger.info(f"  Fetched {len(content)} chars from {result['url'][:60]}...")
        time.sleep(0.5)  # Be polite

    # ── Step 3: Build rich prompt ────────────────────────────────
    sources_text = _format_search_results(search_results)

    category_idx = (day - 1) % len(STRATEGY_CATEGORIES)
    suggested_category = STRATEGY_CATEGORIES[category_idx]

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
- You only have access to price history (OHLC candles at 1-min intervals) and current market odds

## Web Research Results
Below are REAL search results with actual page content. Read them carefully for strategy ideas.

{sources_text}

## Suggested Category for Today
{suggested_category}
(You can pick a different category if something better emerges from the research above)

## Previously Tried Strategies (DO NOT REPEAT THESE)
{previous_str}

## Your Task
1. READ the web research results above carefully. Extract specific, actionable trading rules.
2. Identify ONE specific trading strategy based on what you read.
3. It MUST be different from all previously tried strategies listed above.
4. It should be applicable to 5-minute BTC price movements.
5. It should be implementable with just price history data (last 20 one-minute close prices).
6. Be SPECIFIC -- include exact indicator parameters, thresholds, and decision rules.
7. You can COMBINE ideas from multiple sources or ADAPT a strategy to the 5-min timeframe.

## Output Format (respond with ONLY this JSON, no other text)
```json
{{
    "strategy_name": "Short descriptive name (3-5 words)",
    "category": "Category from the list above",
    "description": "Detailed 3-5 sentence description of how the strategy works. Include SPECIFIC parameters: indicator periods, thresholds, entry/exit conditions. Be precise enough that a developer could implement it from this description alone.",
    "key_concept": "Core idea in 1-2 sentences",
    "indicators": ["indicator1", "indicator2"],
    "source_indices": [1, 3]
}}
```
The source_indices should reference which search results (by number) inspired this strategy.
Use [] if you invented it from scratch.
Use ONLY ASCII characters -- no unicode, no special symbols."""

    # ── Step 4: Qwen selects/invents ─────────────────────────────
    response = _call_ollama(prompt, model, temperature=0.6)
    if not response:
        logger.error("Failed to get response from Ollama")
        return None

    # Extract JSON from response
    json_match = re.search(r'```json\s*\n(.*?)\n\s*```', response, re.DOTALL)
    if not json_match:
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
            logger.warning(f"Strategy '{new_name}' already tried. Retrying...")
            return _retry_with_different_strategy(day, model, previous_names, search_results)

    # Attach sources
    source_indices = strategy_data.get("source_indices", [])
    sources = []
    for idx in source_indices:
        if 1 <= idx <= len(search_results):
            src = search_results[idx - 1]
            sources.append({
                "title": src["title"],
                "url": src["url"],
                "snippet": src["snippet"],
            })

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


def _format_search_results(results: list[dict]) -> str:
    """Format search results with page content into a readable block for Qwen."""
    if not results:
        return "  (No web results found -- use your own knowledge of trading strategies)"

    parts = []
    for i, r in enumerate(results, 1):
        block = f"  [{i}] {r['title']}\n"
        block += f"      URL: {r['url']}\n"
        block += f"      Snippet: {r['snippet']}\n"

        # Include extra snippets from Brave if available
        extra = r.get("extra_snippets", [])
        if extra:
            block += f"      Additional context: {' ... '.join(extra[:2])}\n"

        # Include fetched page content (the real value-add)
        page_content = r.get("page_content", "")
        if page_content:
            # Trim to ~1500 chars to leave room for multiple sources
            trimmed = page_content[:1500]
            block += f"      --- Page Content ---\n"
            block += f"      {trimmed}\n"
            block += f"      --- End Page Content ---\n"

        parts.append(block)

    return "\n".join(parts)


def _retry_with_different_strategy(day: int, model: str,
                                   previous_names: list[str],
                                   search_results: list[dict]) -> dict | None:
    """Retry strategy selection with a stronger uniqueness constraint."""
    category_idx = (day - 1 + 11) % len(STRATEGY_CATEGORIES)
    suggested_category = STRATEGY_CATEGORIES[category_idx]

    previous_str = "\n".join(f"  - {name}" for name in previous_names)

    # Include any page content we already fetched
    sources_text = _format_search_results(search_results[:5]) if search_results else ""

    prompt = f"""You are a quantitative trading researcher. You MUST invent a COMPLETELY NEW
trading strategy for BTC 5-minute UP/DOWN binary markets.

## CRITICAL: These strategies have ALREADY been tried. Do NOT repeat any of them:
{previous_str}

## Suggested new category: {suggested_category}

## Available research (for inspiration only):
{sources_text}

## Requirements
- Must work on 5-minute BTC price data (you get last 20 one-minute close prices)
- Must be genuinely different from all listed strategies above
- Include SPECIFIC parameters: periods, thresholds, decision rules
- Can combine ideas from the research or invent something novel

## Output Format (respond with ONLY this JSON)
```json
{{
    "strategy_name": "Short descriptive name (3-5 words)",
    "category": "Category",
    "description": "Detailed 3-5 sentence description with specific parameters",
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
