"""
Qwen Strategist — The Meta-Learning Brain

Analyzes trading performance and outputs TWO things:
1. strategy.json — trading parameters + gate filters for live trading
2. reward_config.json — reward shaping weights that change HOW the RL model learns

This is the "slow thinking" layer that guides the "fast trading" layer.

Usage:
    python brain/qwen_strategist.py                    # Analyze live trades
    python brain/qwen_strategist.py --backtest-mode    # Analyze backtest results
    python brain/qwen_strategist.py --dry-run          # Don't save changes
"""

import json
import os
import re
import sys
import requests
import random
from datetime import datetime, timedelta
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "strategy.json"
REWARD_CONFIG_PATH = PROJECT_ROOT / "configs" / "reward_config.json"
LOGS_PATH = PROJECT_ROOT / "logs"
TRADE_LOG_PATH = LOGS_PATH / "live_trades.jsonl"
BACKTEST_LOG_PATH = LOGS_PATH / "backtest_results.jsonl"
BACKTEST_TRADES_PATH = LOGS_PATH / "backtest_trades.jsonl"


def load_recent_trades(days: int = 7) -> list:
    """Load recent live trade logs."""
    if not TRADE_LOG_PATH.exists():
        return []

    cutoff = datetime.now() - timedelta(days=days)
    trades = []

    with open(TRADE_LOG_PATH) as f:
        for line in f:
            try:
                t = json.loads(line.strip())
                ts = datetime.fromisoformat(t.get("timestamp", "2000-01-01"))
                if ts > cutoff:
                    trades.append(t)
            except (json.JSONDecodeError, ValueError):
                pass

    return trades


def load_backtest_trades() -> list:
    """Load detailed backtest trade logs."""
    if not BACKTEST_TRADES_PATH.exists():
        return []
    trades = []
    with open(BACKTEST_TRADES_PATH) as f:
        for line in f:
            try:
                trades.append(json.loads(line.strip()))
            except (json.JSONDecodeError, ValueError):
                pass
    return trades


def load_backtest_summary() -> dict | None:
    """Load most recent backtest summary."""
    if not BACKTEST_LOG_PATH.exists():
        return None
    last_line = None
    with open(BACKTEST_LOG_PATH) as f:
        for line in f:
            if line.strip():
                last_line = line
    if last_line:
        try:
            return json.loads(last_line.strip())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def load_current_strategy() -> dict:
    """Load current strategy config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {
        "min_confidence": 0.80,
        "entry_seconds": 30,
        "max_entry_price": 0.95,
        "bet_size": 1.00,
        "min_abs_delta_pct": 0.02,
        "avoid_hours_utc": [],
        "max_consecutive_losses_pause": 5,
        "notes": "Initial default config",
        "updated": "never",
        "version": 0,
    }


def load_reward_config() -> dict:
    """Load current reward config."""
    if REWARD_CONFIG_PATH.exists():
        with open(REWARD_CONFIG_PATH) as f:
            return json.load(f)
    return {
        "skip_penalty": -0.02,
        "win_bonus_multiplier": 1.0,
        "loss_penalty_multiplier": 1.0,
        "high_confidence_bonus": 0.05,
        "high_confidence_threshold": 0.05,
        "low_confidence_penalty": -0.05,
        "low_confidence_threshold": 0.02,
        "no_funds_penalty": -0.1,
        "version": 0,
    }


def _compute_trade_stats(trades: list) -> dict:
    """Compute detailed stats from a list of trades."""
    if not trades:
        return {}

    total = len(trades)
    wins = sum(1 for t in trades if t.get("result") == "WIN")
    losses = total - wins
    pnls = [t.get("reward", t.get("pnl", 0)) for t in trades]
    total_pnl = sum(pnls)

    # By delta magnitude
    by_delta = {"strong": {"w": 0, "l": 0}, "medium": {"w": 0, "l": 0}, "weak": {"w": 0, "l": 0}}
    for t in trades:
        d = abs(t.get("delta_pct", 0))
        if d >= 0.05:
            bucket = "strong"
        elif d >= 0.02:
            bucket = "medium"
        else:
            bucket = "weak"
        if t.get("result") == "WIN":
            by_delta[bucket]["w"] += 1
        else:
            by_delta[bucket]["l"] += 1

    # By action
    by_action = {}
    for t in trades:
        a = t.get("action", "unknown")
        if a not in by_action:
            by_action[a] = {"w": 0, "l": 0}
        if t.get("result") == "WIN":
            by_action[a]["w"] += 1
        else:
            by_action[a]["l"] += 1

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": f"{wins/total*100:.1f}%",
        "total_pnl": f"${total_pnl:+.2f}",
        "by_delta_strength": by_delta,
        "by_action": by_action,
    }


def analyze_for_retraining(trades: list, backtest_summary: dict | None,
                           current_reward_config: dict,
                           current_strategy: dict,
                           model: str = DEFAULT_MODEL) -> dict:
    """
    Main analysis function called by orchestrator.
    Analyzes trades and outputs both strategy and reward config changes.

    Returns:
        {
            "strategy_changes": [...],
            "reward_config_changes": [...],
            "analysis": str,
            "confidence": str,
            "raw_response": str,
        }
    """
    if not trades:
        return {"error": "No trades to analyze", "strategy_changes": [], "reward_config_changes": []}

    stats = _compute_trade_stats(trades)

    # Sample individual trades for Qwen (up to 50)
    sample_trades = random.sample(trades, min(50, len(trades)))
    sample_str = json.dumps(sample_trades, indent=1)

    backtest_str = ""
    if backtest_summary:
        backtest_str = f"""
## Most Recent Backtest Summary
- Win rate: {backtest_summary.get('win_rate', 'N/A')}
- Total P&L: ${backtest_summary.get('total_pnl', 0):+.2f}
- Avg P&L per trade: ${backtest_summary.get('avg_pnl_per_trade', 0):+.4f}
- Sharpe ratio: {backtest_summary.get('sharpe_ratio', 'N/A')}
- Max drawdown: {backtest_summary.get('max_drawdown', 'N/A')}
- Skip rate: {backtest_summary.get('skip_rate', 'N/A')}
"""

    prompt = f"""You are an AI trading strategist optimizing a Polymarket BTC 5-minute prediction bot.
The bot uses a PPO reinforcement learning model. You control TWO things:

1. **Strategy config** — gates/filters applied to live trading decisions
2. **Reward config** — weights that shape HOW the RL model learns during retraining

Your reward config changes are the most powerful lever. By adjusting reward weights,
you change what the RL model optimizes for in its next training cycle.

## Current Strategy Config
```json
{json.dumps(current_strategy, indent=2)}
```

## Current Reward Config (controls RL learning)
```json
{json.dumps(current_reward_config, indent=2)}
```

## Reward Config Parameter Guide
- `skip_penalty`: Penalty for not trading (more negative = model trades more often)
- `win_bonus_multiplier`: Multiplier on profit reward (higher = model values wins more)
- `loss_penalty_multiplier`: Multiplier on loss penalty (higher = model avoids losses more)
- `high_confidence_bonus`: Extra reward for correctly trading when |delta| > threshold
- `high_confidence_threshold`: Delta % threshold for "high confidence" (e.g., 0.05 = 0.05%)
- `low_confidence_penalty`: Extra penalty for trading when |delta| < threshold
- `low_confidence_threshold`: Delta % threshold for "low confidence" (e.g., 0.02 = 0.02%)
- `no_funds_penalty`: Penalty for trying to trade with insufficient bankroll

## Trade Statistics
{json.dumps(stats, indent=2)}
{backtest_str}

## Sample Individual Trades (examine these carefully)
{sample_str}

## Your Task
Analyze the data and recommend changes to BOTH configs. Think step by step:

1. Which delta ranges are profitable vs unprofitable?
2. Is the model trading too often or not enough?
3. Are losses concentrated in weak-signal trades?
4. What reward adjustments would teach the model to avoid losing patterns?

Output your analysis and recommendations in this exact JSON format:
```json
{{
  "analysis": "Your plain English analysis of what's happening and why",
  "strategy_changes": [
    {{"param": "min_abs_delta_pct", "old": 0.02, "new": 0.03, "reason": "why"}}
  ],
  "reward_config_changes": [
    {{"param": "low_confidence_penalty", "old": -0.05, "new": -0.15, "reason": "why"}},
    {{"param": "skip_penalty", "old": -0.02, "new": -0.01, "reason": "why"}}
  ],
  "confidence": "high/medium/low",
  "expected_improvement": "what you expect to happen after retraining"
}}
```

Be specific about WHY each change will help. Only suggest changes you're confident will improve performance.
If current configs are working well, suggest no changes."""

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 8192},
            },
            timeout=180,
        )

        if r.status_code != 200:
            return {"error": f"Ollama HTTP {r.status_code}",
                    "strategy_changes": [], "reward_config_changes": []}

        response_text = r.json()["message"]["content"]

        # Extract JSON from response
        match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response_text, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
            result["raw_response"] = response_text
            return result
        else:
            return {
                "analysis": response_text,
                "strategy_changes": [],
                "reward_config_changes": [],
                "error": "Could not parse JSON from response",
                "raw_response": response_text,
            }

    except Exception as e:
        return {"error": str(e), "strategy_changes": [], "reward_config_changes": []}


def apply_strategy_changes(current: dict, analysis: dict) -> dict:
    """Apply strategy config changes."""
    new = current.copy()

    for change in analysis.get("strategy_changes", []):
        param = change.get("param")
        new_val = change.get("new")
        if param and new_val is not None and param in new:
            new[param] = new_val

    new["notes"] = analysis.get("analysis", "")[:500]
    new["updated"] = datetime.now().isoformat()
    new["version"] = current.get("version", 0) + 1
    return new


def apply_reward_config_changes(current: dict, analysis: dict) -> dict:
    """Apply reward config changes with safety clamping."""
    new = current.copy()

    # Safety bounds to prevent degenerate reward configs
    bounds = {
        "skip_penalty": (-0.5, 0.0),
        "win_bonus_multiplier": (0.5, 3.0),
        "loss_penalty_multiplier": (0.5, 3.0),
        "high_confidence_bonus": (0.0, 0.5),
        "high_confidence_threshold": (0.01, 0.20),
        "low_confidence_penalty": (-0.5, 0.0),
        "low_confidence_threshold": (0.005, 0.10),
        "no_funds_penalty": (-1.0, 0.0),
    }

    for change in analysis.get("reward_config_changes", []):
        param = change.get("param")
        new_val = change.get("new")
        if param and new_val is not None and param in new:
            # Clamp to safety bounds
            if param in bounds:
                lo, hi = bounds[param]
                new_val = max(lo, min(hi, new_val))
            new[param] = new_val

    new["updated"] = datetime.now().isoformat()
    new["version"] = current.get("version", 0) + 1
    new["last_reasoning"] = analysis.get("analysis", "")[:300]
    return new


def save_config(config: dict, path: Path):
    """Save config to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def log_analysis(analysis: dict, trades_analyzed: int, mode: str):
    """Log the analysis for review."""
    log_path = LOGS_PATH / "strategy_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "trades_analyzed": trades_analyzed,
        "strategy_changes": len(analysis.get("strategy_changes", [])),
        "reward_config_changes": len(analysis.get("reward_config_changes", [])),
        "analysis_summary": analysis.get("analysis", "")[:200],
        "confidence": analysis.get("confidence", "unknown"),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true",
                       help="Analyze but don't save changes")
    parser.add_argument("--backtest-mode", action="store_true",
                       help="Analyze backtest results instead of live trades")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    print("  Qwen Strategist -- Meta-Learning Brain")
    print("=" * 50)

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  Ollama: connected ({len(models)} models)")

        # Find best available model
        preferred = ["qwen3:8b", "qwen2.5:7b", "qwen2.5-coder:7b", "llama3.1:8b"]
        model = args.model
        for p in preferred:
            if any(p.split(":")[0] in m for m in models):
                for m_name in models:
                    if p.split(":")[0] in m_name:
                        model = m_name
                        break
                break
        print(f"  Model: {model}")
    except Exception:
        print("  Ollama not running. Start with: ollama serve")
        return

    # Load data based on mode
    if args.backtest_mode:
        trades = load_backtest_trades()
        backtest_summary = load_backtest_summary()
        mode = "backtest"
        print(f"  Mode: backtest analysis")
    else:
        trades = load_recent_trades(days=args.days)
        backtest_summary = None
        mode = "live"
        print(f"  Mode: live trade analysis")

    current_strategy = load_current_strategy()
    current_reward = load_reward_config()

    print(f"  Trades to analyze: {len(trades)}")
    print(f"  Strategy config v{current_strategy.get('version', 0)}")
    print(f"  Reward config v{current_reward.get('version', 0)}")

    if len(trades) < 5:
        print(f"\n  Not enough trades to analyze ({len(trades)} found, need >= 5)")
        print(f"  Run backtesting or live trading first to collect data.")
        return

    # Analyze
    print(f"\n  Analyzing with {model}...")
    analysis = analyze_for_retraining(
        trades, backtest_summary, current_reward, current_strategy, model
    )

    if "error" in analysis and not analysis.get("strategy_changes") and not analysis.get("reward_config_changes"):
        print(f"  Analysis failed: {analysis['error']}")
        return

    print(f"\n  Analysis:")
    print(f"  {analysis.get('analysis', '')[:300]}")

    # Strategy changes
    strat_changes = analysis.get("strategy_changes", [])
    if strat_changes:
        print(f"\n  Strategy changes ({len(strat_changes)}):")
        for c in strat_changes:
            print(f"    {c['param']}: {c.get('old')} -> {c.get('new')} | {c.get('reason', '')}")
    else:
        print(f"\n  No strategy changes recommended")

    # Reward config changes
    reward_changes = analysis.get("reward_config_changes", [])
    if reward_changes:
        print(f"\n  Reward config changes ({len(reward_changes)}):")
        for c in reward_changes:
            print(f"    {c['param']}: {c.get('old')} -> {c.get('new')} | {c.get('reason', '')}")
    else:
        print(f"\n  No reward config changes recommended")

    # Apply
    if not args.dry_run:
        if strat_changes:
            new_strategy = apply_strategy_changes(current_strategy, analysis)
            save_config(new_strategy, CONFIG_PATH)
            print(f"\n  Strategy config updated (v{new_strategy['version']}): {CONFIG_PATH}")

        if reward_changes:
            new_reward = apply_reward_config_changes(current_reward, analysis)
            save_config(new_reward, REWARD_CONFIG_PATH)
            print(f"  Reward config updated (v{new_reward['version']}): {REWARD_CONFIG_PATH}")
    elif args.dry_run:
        print(f"\n  (dry-run -- no changes saved)")

    # Log
    log_analysis(analysis, len(trades), mode)
    print(f"\n  Analysis logged")


if __name__ == "__main__":
    main()
