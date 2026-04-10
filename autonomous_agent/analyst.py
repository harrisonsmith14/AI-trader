"""
Analyst — Qwen's Brain. Reviews trading results and rewrites the strategy.

This is where the AI reasons about what's working, what's failing,
and writes improved Python strategy code. The strategy evolves through
real experience, not backtesting.
"""

import re
import json
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

from . import journal, sandbox

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


def analyze_and_rewrite(model: str = DEFAULT_MODEL,
                        trigger_reason: str = "scheduled") -> dict:
    """
    Main analysis function. Sends trade journal + current strategy to Qwen,
    gets back an improved strategy, validates it, and deploys.

    Returns:
        {
            "rewrote": bool,
            "old_version": int,
            "new_version": int,
            "analysis": str,
            "changes_summary": str,
            "error": str | None,
        }
    """
    # Load current strategy
    current_code = sandbox.get_current_strategy_code()
    current_version = sandbox.get_current_version()

    # Load journal data
    trades = journal.get_recent_entries(days=14, entry_type="trade")
    skips = journal.get_recent_entries(days=14, entry_type="skip")
    observations = journal.get_recent_entries(days=14, entry_type="observation")
    resolutions = journal.get_recent_entries(days=14, entry_type="resolution")
    stats = journal.get_trade_stats(days=30)

    total_data_points = len(trades) + len(skips) + len(observations)

    if total_data_points < 5:
        return {
            "rewrote": False,
            "old_version": current_version,
            "new_version": current_version,
            "analysis": f"Not enough data yet ({total_data_points} points, need 5+). Collecting more observations.",
            "changes_summary": "No changes",
            "error": None,
        }

    # Get version history summary so Qwen can see if it's spiraling
    version_history = _get_version_history(current_version)

    # Build the prompt
    prompt = _build_prompt(
        current_code, current_version, trades, skips,
        observations, resolutions, stats, trigger_reason,
        version_history
    )

    # Call Qwen
    logger.info(f"Sending analysis to {model}...")
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 16384},
            },
            timeout=300,
        )

        if r.status_code != 200:
            return _error_result(current_version, f"Ollama HTTP {r.status_code}")

        response_text = r.json()["message"]["content"]

    except requests.RequestException as e:
        return _error_result(current_version, f"Ollama connection failed: {e}")

    # Extract Python code from response
    code_match = re.search(r'```python\s*\n(.*?)\n\s*```', response_text, re.DOTALL)
    if not code_match:
        # Try without language specifier
        code_match = re.search(r'```\s*\n(.*?)\n\s*```', response_text, re.DOTALL)

    if not code_match:
        logger.warning("Qwen didn't output a code block. Keeping current strategy.")
        # Extract analysis even without code
        analysis = response_text[:500]
        return {
            "rewrote": False,
            "old_version": current_version,
            "new_version": current_version,
            "analysis": analysis,
            "changes_summary": "Qwen analyzed but chose not to rewrite (or output was malformed)",
            "error": None,
        }

    new_code = code_match.group(1)

    # Extract analysis text (everything before the code block)
    analysis_text = response_text[:response_text.find("```")].strip()
    if len(analysis_text) > 1000:
        analysis_text = analysis_text[:1000] + "..."

    # Validate the new strategy
    new_version = current_version + 1
    is_valid, error = sandbox.validate_code(new_code)

    if not is_valid:
        logger.warning(f"New strategy failed validation: {error}")
        return {
            "rewrote": False,
            "old_version": current_version,
            "new_version": current_version,
            "analysis": analysis_text,
            "changes_summary": f"Strategy rejected: {error}",
            "error": error,
        }

    # Deploy the new strategy
    success = sandbox.deploy_strategy(new_code, new_version)

    if success:
        logger.info(f"Strategy updated: v{current_version} -> v{new_version}")
        return {
            "rewrote": True,
            "old_version": current_version,
            "new_version": new_version,
            "analysis": analysis_text,
            "changes_summary": f"Strategy rewritten from v{current_version} to v{new_version}",
            "error": None,
        }
    else:
        return _error_result(current_version, "Deploy failed after validation passed")


def _get_version_history(current_version: int, lookback: int = 5) -> str:
    """Extract version docstrings to show how the strategy has evolved."""
    import re
    versions_dir = sandbox.VERSIONS_DIR
    if not versions_dir.exists():
        return "No version history yet."

    history = []
    start = max(1, current_version - lookback)
    for v in range(start, current_version + 1):
        path = versions_dir / f"strategy_v{v:03d}.py"
        if not path.exists():
            continue
        try:
            code = path.read_text(encoding="utf-8", errors="ignore")
            # Extract docstring
            doc_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
            if doc_match:
                doc = doc_match.group(1).strip()
                # Take first 300 chars
                summary = doc[:300]
                history.append(f"v{v}: {summary}")
        except Exception:
            pass

    return "\n\n".join(history) if history else "No readable version history."


def _build_prompt(current_code: str, version: int,
                  trades: list, skips: list, observations: list,
                  resolutions: list, stats: dict,
                  trigger_reason: str,
                  version_history: str = "") -> str:
    """Build the analysis prompt for Qwen."""

    # Format trades for display
    trades_str = ""
    if trades:
        sample = trades[-20:]  # Last 20
        for t in sample:
            result = t.get("result", "pending")
            pnl = t.get("pnl", "?")
            trades_str += (
                f"  {t.get('city')} {t.get('date')} | "
                f"Forecast: {t.get('nws_forecast')}°F | "
                f"Bought: {t.get('bracket_chosen')} @ ${t.get('bracket_price', '?')} | "
                f"Result: {result} | P&L: ${pnl}\n"
                f"    Reasoning: {t.get('reasoning', '')}\n"
            )
    else:
        trades_str = "  No trades yet — strategy has been observing only.\n"

    # Format observations (what happened on markets we skipped)
    obs_str = ""
    if observations:
        sample = observations[-15:]
        for o in sample:
            obs_str += (
                f"  {o.get('city')} {o.get('date')} | "
                f"Forecast: {o.get('nws_forecast')}°F | "
                f"Actual: {o.get('actual_temp')}°F | "
                f"Winner: {o.get('winning_bracket')} @ ${o.get('best_bracket_price', '?')} | "
                f"Hypothetical P&L: ${o.get('hypothetical_pnl', '?')}\n"
            )
    else:
        obs_str = "  No observations yet.\n"

    # Format skips
    skips_str = ""
    if skips:
        sample = skips[-10:]
        for s in sample:
            skips_str += (
                f"  {s.get('city')} {s.get('date')} | "
                f"Forecast: {s.get('nws_forecast')}°F | "
                f"Reasoning: {s.get('reasoning', '')}\n"
            )

    # Stats summary
    stats_str = json.dumps(stats, indent=2)

    return f"""You are an autonomous AI learning to predict weather on Polymarket.
You write Python trading strategies that evolve based on real market results.
Your strategy is a Python function that takes market context and returns a trading decision.

## Trigger Reason
{trigger_reason}

## Your Current Strategy (v{version})
```python
{current_code}
```

## Trade Results (most recent)
{trades_str}

## Markets You Skipped (what would have happened)
{obs_str}

## Recent Skip Decisions
{skips_str}

## Performance Statistics
{stats_str}

## Your Recent Strategy Evolution (last 5 versions)
Look at this carefully — if you see a pattern of progressively MORE conservative
strategies (smaller positions, higher thresholds, more checks), you are in a
death spiral. Reverse direction.

{version_history}

## How These Markets Work (READ CAREFULLY)

Polymarket has daily "Highest temperature in [City] on [Date]?" markets:
- Outcomes are 2 degree F brackets: 50-51, 52-53, 54-55, etc.
- Most markets have 5-10 brackets total
- Each bracket has a price (probability). Buying at 0.40 means market thinks 40% chance.
- If you buy the WINNING bracket at 0.40, you get 1.00 back (profit: 0.60)
- If wrong, you lose the price you paid

### CRITICAL: Win Rate Math
- Random guessing on a 5-bracket market = 20% win rate
- Random guessing on a 10-bracket market = 10% win rate
- A 25-30% win rate means you are BETTER than random
- The question is NOT "how often do I win" but "is my expected value positive"
- Buying a 0.05 bracket and winning 1 in 10 times = positive EV
- Buying a 0.50 bracket and winning 6 in 10 times = positive EV
- DO NOT panic about win rate alone — look at P&L

### What Matters: Edge, Not Win Rate
- Edge = your_estimated_probability - market_price
- Positive edge means market is underpricing this outcome
- A strategy with 20% win rate buying 0.05 brackets is profitable
- A strategy with 60% win rate buying 0.80 brackets is unprofitable

## Available Data in context
Your decide(context) function receives:
- city, date: which market
- nws_forecast: NWS official forecast (degrees F)
- gfs_mean, gfs_low, gfs_high, gfs_spread: GFS ensemble forecast range
- brackets: list of available brackets with current market prices
- historical_bias: average (forecast - actual) over recent days (always a number, never None)
- recent_trades: your last 20 trades with outcomes
- win_rate, total_pnl: your overall performance

## CRITICAL Rules to Avoid Common Mistakes

### 1. Don't Death-Spiral
If you keep making the strategy MORE conservative every iteration:
- Position size shrinking each version → STOP, go back to 1-5% range
- Edge threshold rising each version → STOP, lower it
- Confidence ceiling lowering each version → STOP, raise it
This is a feedback loop. You see losses, get scared, trade less, get less data,
see more variance in small sample, get more scared. Break the cycle.

### 2. Position Size Sanity
- Reasonable: 0.5% to 5% of bankroll per trade
- Too small: anything below 0.1% is meaningless — you can't profit
- Too big: above 10% is dangerous

### 3. Calibration vs Aggression
Look at your losing trades. Were you off by:
- 0 brackets (wrong direction entirely) → bad strategy, rethink approach
- 1 bracket → calibration issue, adjust forecast bias correction
- 2+ brackets → forecast was wrong, not your fault, this is normal variance

### 4. Use Historical Bias
If NWS forecast is 70F but actual is usually 67F (bias = +3F), then:
- adjusted_forecast = nws_forecast - historical_bias
- This is the SINGLE BIGGEST improvement you can make
- Track per-city bias if you can

### 5. Find Mispriced Brackets
The market is not perfectly efficient. Look for:
- Cheap brackets (under 0.10) when forecast is exactly there → great EV
- Expensive brackets (over 0.50) where market is overconfident → skip
- Brackets adjacent to forecast that market underprices → buy

### 6. Don't Over-Engineer
- 5 simple rules > 50 complex rules
- Every safety check you add reduces trades
- You need DATA (more trades) to learn patterns
- Trading 5x per day with 30% win rate teaches you more than trading 0x with 0% win rate

## Your Task
1. Look at WHAT WENT WRONG in losing trades — were you off by 1 bracket? Wrong direction?
2. Look at WHAT WENT RIGHT in winning trades — what was the pattern?
3. Identify the SINGLE biggest improvement you can make
4. Write a strategy that fixes that ONE thing
5. Keep all the things that were working

## Anti-Patterns (Things to NOT Do)
- DO NOT shrink position size if it's already below 1%
- DO NOT add more safety checks if you have more than 5 already
- DO NOT raise edge threshold above 0.20
- DO NOT raise confidence threshold above 0.70
- DO NOT add features without removing old ones (keep code simple)

## Rules for your code
- Function signature: decide(context) -> dict
- Return: {{"action": "BUY" or "SKIP", "bracket": "54-55" or None, "confidence": float, "reasoning": str}}
- You can use: math, statistics, any pure Python logic
- You CANNOT use: os, sys, requests, subprocess, open, or any I/O
- Use ONLY ASCII characters in your code — no degree symbols, em dashes, or unicode. Write "degrees F" not the symbol
- Be specific in your reasoning — explain WHY you make each decision
- All numeric context values (nws_forecast, gfs_mean, historical_bias, etc.) are guaranteed numbers, never None
- If you don't have enough data yet, it's OK to SKIP MOST markets but trade SOME — you need data

Output the complete strategy.py code in a ```python block.
Start with a docstring that explains your strategy in 3-4 sentences (plain English).
Focus on the ONE THING you changed from the previous version and WHY."""


def fix_crash(error_msg: str, context: dict,
              model: str = DEFAULT_MODEL) -> dict:
    """
    Emergency self-healing: when the strategy crashes at runtime,
    send the error + code + context to Qwen to fix immediately.

    Returns same format as analyze_and_rewrite().
    """
    current_code = sandbox.get_current_strategy_code()
    current_version = sandbox.get_current_version()

    # Format the context values (skip large lists for brevity)
    context_summary = {k: v for k, v in context.items()
                       if k not in ("recent_trades", "brackets")}
    context_summary["brackets_count"] = len(context.get("brackets", []))
    context_summary["brackets_sample"] = context.get("brackets", [])[:3]

    prompt = f"""Your trading strategy CRASHED with a runtime error. Fix it immediately.

## Error
```
{error_msg}
```

## The Context Data That Was Passed
```json
{json.dumps(context_summary, indent=2, default=str)}
```

## Important Notes About Context Values
- All numeric fields (nws_forecast, gfs_mean, gfs_spread, historical_bias, etc.) are guaranteed to be float or int, never None
- brackets is always a list (may be empty)
- recent_trades is always a list (may be empty)
- win_rate and total_pnl are always numbers

## Current Strategy Code (v{current_version}) — THIS CRASHED
```python
{current_code}
```

## Your Task
1. Identify the bug from the error message
2. Fix it while keeping the strategy logic intact
3. Add defensive checks so similar crashes don't happen again
4. Output the COMPLETE fixed strategy.py in a ```python block

Do NOT change the strategy logic — just fix the crash. Keep the same approach,
just make it robust against edge cases.
Use ONLY ASCII characters — no degree symbols, em dashes, or unicode."""

    logger.info(f"Sending crash fix to {model}...")
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 16384},
            },
            timeout=300,
        )

        if r.status_code != 200:
            return _error_result(current_version, f"Ollama HTTP {r.status_code}")

        response_text = r.json()["message"]["content"]

    except requests.RequestException as e:
        return _error_result(current_version, f"Ollama connection failed: {e}")

    # Extract code
    code_match = re.search(r'```python\s*\n(.*?)\n\s*```', response_text, re.DOTALL)
    if not code_match:
        code_match = re.search(r'```\s*\n(.*?)\n\s*```', response_text, re.DOTALL)

    if not code_match:
        return _error_result(current_version, "Qwen couldn't produce a fix")

    new_code = code_match.group(1)
    new_version = current_version + 1

    is_valid, error = sandbox.validate_code(new_code)
    if not is_valid:
        return _error_result(current_version, f"Fix failed validation: {error}")

    success = sandbox.deploy_strategy(new_code, new_version)
    if success:
        analysis = response_text[:response_text.find("```")].strip()
        if len(analysis) > 500:
            analysis = analysis[:500] + "..."
        return {
            "rewrote": True,
            "old_version": current_version,
            "new_version": new_version,
            "analysis": analysis,
            "changes_summary": f"Crash fix: v{current_version} -> v{new_version}",
            "error": None,
        }

    return _error_result(current_version, "Deploy failed")


def _error_result(version: int, error: str) -> dict:
    return {
        "rewrote": False,
        "old_version": version,
        "new_version": version,
        "analysis": "",
        "changes_summary": f"Error: {error}",
        "error": error,
    }


def check_ollama(model: str = DEFAULT_MODEL) -> bool:
    """Verify Ollama is running and model is available."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if any(model.split(":")[0] in m for m in models):
            return True
        logger.warning(f"Model {model} not found. Available: {models}")
        return False
    except Exception:
        logger.error("Ollama not running. Start with: ollama serve")
        return False
