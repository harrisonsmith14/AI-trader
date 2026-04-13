"""
Implementer --- Qwen writes executable strategy code from a research description.

Phase 2 of the daily cycle:
1. Takes the strategy description from the researcher
2. Qwen generates a Python decide() function implementing the strategy
3. Code is validated in a sandbox (syntax, safety, test run)
4. Returns validated strategy code ready for testing
"""

import ast
import json
import logging
import re
import traceback
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
STRATEGIES_DIR = Path(__file__).parent / "strategies"

# Imports blocked in strategy code (same as autonomous_agent/sandbox.py)
BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "requests", "urllib", "http", "socket", "ftplib",
    "smtplib", "email", "pickle", "shelve",
    "ctypes", "multiprocessing", "threading",
    "__import__", "eval", "exec", "compile",
    "open",
}

# Allowed imports for strategies (pure computation only)
ALLOWED_IMPORTS = {"math", "statistics", "collections", "itertools", "functools"}


def validate_strategy_code(code: str) -> tuple[bool, str]:
    """
    Validate BTC strategy code for safety and correctness.

    The strategy must define decide(context) -> dict with:
        action: "UP" | "DOWN" | "SKIP"
        confidence: float 0.0-1.0
        reasoning: str

    Returns (is_valid, error_message).
    """
    # 1. Strip non-ASCII (Qwen sometimes outputs unicode)
    code = code.encode("ascii", errors="ignore").decode("ascii")

    # 2. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # 3. Check for dangerous imports/calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in BLOCKED_IMPORTS:
                    return False, f"Blocked import: {module}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module in BLOCKED_IMPORTS:
                    return False, f"Blocked import: {module}"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "__import__", "open"):
                    return False, f"Blocked function call: {node.func.id}"

    # 4. Check that decide() exists
    func_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]
    if "decide" not in func_names:
        return False, "Missing required function: decide()"

    # 5. Test with sample BTC context data
    sample_context = {
        "btc_price": 84250.0,
        "price_history": [
            84200.0, 84210.0, 84230.0, 84220.0, 84240.0,
            84235.0, 84250.0, 84260.0, 84255.0, 84250.0,
            84245.0, 84240.0, 84250.0, 84260.0, 84270.0,
            84265.0, 84260.0, 84255.0, 84250.0, 84248.0,
        ],
        "market_odds": {"up": 0.52, "down": 0.48},
        "time_in_window": 120.0,
        "recent_trades": [],
        "win_rate": 0.0,
        "volume": 150000.0,
    }

    try:
        namespace = {"__builtins__": __builtins__}
        exec(compile(tree, "<strategy>", "exec"), namespace)

        decide_fn = namespace.get("decide")
        if not callable(decide_fn):
            return False, "decide is not callable"

        result = decide_fn(sample_context)

        # 6. Validate output format
        if not isinstance(result, dict):
            return False, f"decide() must return dict, got {type(result).__name__}"

        if "action" not in result:
            return False, "decide() result missing 'action' key"

        action = result["action"]
        if action not in ("UP", "DOWN", "SKIP"):
            return False, f"Invalid action: {action}. Must be 'UP', 'DOWN', or 'SKIP'"

        if "confidence" not in result:
            return False, "decide() result missing 'confidence' key"

        conf = result["confidence"]
        if not isinstance(conf, (int, float)):
            return False, f"confidence must be numeric, got {type(conf).__name__}"

        if "reasoning" not in result:
            return False, "decide() result missing 'reasoning' key"

    except Exception:
        return False, f"Runtime error: {traceback.format_exc()}"

    return True, "OK"


def _call_ollama(prompt: str, model: str, temperature: float = 0.3) -> str | None:
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


def implement_strategy(research: dict, day: int,
                       model: str = "qwen3:8b",
                       max_attempts: int = 3) -> str | None:
    """
    Generate and validate strategy code from research description.

    Args:
        research: Output from researcher.research_strategy()
        day: Day number (1-21)
        model: Ollama model name
        max_attempts: Max validation retries

    Returns:
        Validated strategy code string, or None on failure.
    """
    strategy_name = research["strategy_name"]
    description = research["description"]
    key_concept = research["key_concept"]
    indicators = ", ".join(research["indicators"])

    prompt = f"""You are a quantitative Python developer. Write a trading strategy function for BTC 5-minute
UP/DOWN binary markets on Polymarket.

## Strategy to Implement
Name: {strategy_name}
Category: {research['category']}
Description: {description}
Key Concept: {key_concept}
Indicators: {indicators}

## Function Interface (MUST follow exactly)

```python
def decide(context: dict) -> dict:
    \"\"\"
    context contains:
        btc_price: float         -- current BTC price (e.g., 84250.0)
        price_history: list      -- last 20 prices (1-min intervals), oldest first
        market_odds: dict        -- {{"up": 0.52, "down": 0.48}}
        time_in_window: float    -- seconds into current 5-min window (0-300)
        recent_trades: list      -- last 20 trades with outcomes
        win_rate: float          -- overall win rate (0.0-1.0)
        volume: float            -- recent trading volume

    Returns:
        {{
            "action": "UP" or "DOWN" or "SKIP",
            "confidence": float,       -- 0.0 to 1.0
            "reasoning": str           -- why this decision
        }}
    \"\"\"
```

## Rules
1. You can use: math, statistics (Python stdlib). No other imports.
2. You CANNOT use: os, sys, requests, subprocess, open, eval, exec, or any I/O
3. Use ONLY ASCII characters -- no degree symbols, em dashes, smart quotes, or unicode
4. All context values are guaranteed to be the correct type (never None)
5. price_history is always a list of at least 2 floats
6. The strategy should be NON-TRIVIAL -- actually implement the described strategy logic
7. Include a docstring at the top explaining the strategy in 2-3 sentences
8. Handle edge cases gracefully (e.g., not enough price history -> SKIP)
9. confidence should reflect how confident the signal is (0.5 = neutral, 1.0 = very strong)
10. SKIP when the signal is weak or unclear -- don't force trades
11. Consider market_odds -- if the market already prices in your signal, the edge is gone

## BTC 5-Min Market Context
- You're predicting whether BTC price will be HIGHER or LOWER at the end of the 5-min window
- price_history gives you the last 20 minutes of 1-minute prices
- market_odds tell you what the market currently thinks (0.52 up means market leans UP)
- A good strategy finds an edge the market hasn't priced in

## Output
Write the COMPLETE strategy as a single Python code block. Start with the docstring.
```python
# your code here
```"""

    for attempt in range(max_attempts):
        logger.info(f"Day {day}: Generating strategy code (attempt {attempt + 1}/{max_attempts})...")

        response = _call_ollama(prompt, model)
        if not response:
            continue

        # Extract Python code
        code_match = re.search(r'```python\s*\n(.*?)\n\s*```', response, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\s*\n(.*?)\n\s*```', response, re.DOTALL)
        if not code_match:
            logger.warning(f"Attempt {attempt + 1}: No code block found in response")
            continue

        code = code_match.group(1)
        # Clean non-ASCII
        code = code.encode("ascii", errors="ignore").decode("ascii")

        # Validate
        is_valid, error = validate_strategy_code(code)

        if is_valid:
            logger.info(f"Day {day}: Strategy code validated successfully")
            # Save to strategies directory
            _save_strategy(code, day, research)
            return code
        else:
            logger.warning(f"Attempt {attempt + 1}: Validation failed: {error}")
            # On next attempt, include the error in the prompt
            prompt = _build_fix_prompt(code, error, research)

    logger.error(f"Day {day}: Failed to generate valid strategy after {max_attempts} attempts")
    return None


def _build_fix_prompt(failed_code: str, error: str, research: dict) -> str:
    """Build a prompt that includes the previous failure for Qwen to fix."""
    return f"""Your previous strategy code had an error. Fix it.

## Strategy
Name: {research['strategy_name']}
Description: {research['description']}

## Previous Code (FAILED)
```python
{failed_code}
```

## Error
{error}

## Requirements
- Function signature: decide(context) -> dict
- context has: btc_price (float), price_history (list of floats), market_odds (dict with "up"/"down"),
  time_in_window (float), recent_trades (list), win_rate (float), volume (float)
- Return: {{"action": "UP"/"DOWN"/"SKIP", "confidence": float 0-1, "reasoning": str}}
- Allowed imports: math, statistics only
- NO os/sys/requests/subprocess/open/eval/exec
- ASCII only -- no unicode characters
- All context values are guaranteed non-None

Fix the error and output the COMPLETE corrected strategy in a ```python block."""


def _save_strategy(code: str, day: int, research: dict):
    """Save strategy code to the strategies directory."""
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    # Create filename from day and strategy name
    safe_name = re.sub(r'[^a-z0-9_]', '_', research["strategy_name"].lower())
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')
    filename = f"day_{day:02d}_{safe_name}.py"

    path = STRATEGIES_DIR / filename
    path.write_text(code, encoding="utf-8")
    logger.info(f"Strategy saved to {path}")


def load_strategy(day: int):
    """
    Load and return the decide() function for a given day's strategy.
    Scans the strategies directory for a file matching day_NN_*.py.
    """
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    pattern = f"day_{day:02d}_*.py"
    matches = list(STRATEGIES_DIR.glob(pattern))

    if not matches:
        logger.error(f"No strategy file found for day {day}")
        return None

    path = matches[0]
    code = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(code)
        namespace = {"__builtins__": __builtins__}
        exec(compile(tree, str(path), "exec"), namespace)
        decide_fn = namespace.get("decide")
        if callable(decide_fn):
            return decide_fn
        logger.error(f"decide() not found in {path}")
    except Exception as e:
        logger.error(f"Failed to load strategy from {path}: {e}")

    return None
