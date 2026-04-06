"""
Qwen3 Strategist — The Meta-Learning Brain
Runs every night, reads trade logs, reasons about what's working/failing,
and updates the RL agent's config in plain English reasoning.

This is the "slow thinking" layer that guides the "fast trading" layer.

Usage:
    python brain/qwen_strategist.py          # Run analysis
    python brain/qwen_strategist.py --model  # Specify Ollama model
"""

import json
import os
import sys
import requests
from datetime import datetime, timedelta
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"  # or qwen3:8b if available
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "strategy.json"
LOGS_PATH = Path(__file__).parent.parent / "logs"
TRADE_LOG_PATH = Path(__file__).parent.parent / "logs" / "live_trades.jsonl"


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
            except:
                pass
    
    return trades


def load_current_strategy() -> dict:
    """Load current strategy config."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    
    # Default config
    return {
        "min_confidence": 0.80,
        "entry_seconds": 30,
        "max_entry_price": 0.95,
        "bet_size": 1.00,
        "notes": "Initial default config — not yet optimized by Qwen",
        "updated": "never",
        "version": 0,
    }


def analyze_with_qwen(trades: list, current_config: dict, model: str) -> dict:
    """Send trade data to Qwen for analysis and strategy recommendations."""
    
    if not trades:
        return {"error": "No trades to analyze", "changes": []}
    
    # Compute stats by confidence tier
    by_confidence = {}
    for t in trades:
        tier = t.get("confidence_tier", "unknown")
        if tier not in by_confidence:
            by_confidence[tier] = {"wins": 0, "losses": 0, "pnl": 0}
        if t.get("result") == "WIN":
            by_confidence[tier]["wins"] += 1
        else:
            by_confidence[tier]["losses"] += 1
        by_confidence[tier]["pnl"] += t.get("pnl", 0)

    # Stats by entry timing
    by_timing = {}
    for t in trades:
        timing = t.get("entry_seconds_before_close", "unknown")
        bucket = f"T-{(timing // 10) * 10}s" if isinstance(timing, int) else str(timing)
        if bucket not in by_timing:
            by_timing[bucket] = {"wins": 0, "losses": 0}
        if t.get("result") == "WIN":
            by_timing[bucket]["wins"] += 1
        else:
            by_timing[bucket]["losses"] += 1

    total = len(trades)
    wins = sum(1 for t in trades if t.get("result") == "WIN")
    total_pnl = sum(t.get("pnl", 0) for t in trades)

    prompt = f"""You are an AI trading strategist analyzing a Polymarket BTC prediction trading bot.

## Current Strategy Config
```json
{json.dumps(current_config, indent=2)}
```

## Trade Performance (Last 7 Days)
- Total trades: {total}
- Win rate: {wins/total*100:.1f}% ({wins}W/{total-wins}L)
- Total P&L: ${total_pnl:+.2f}

## Performance by Confidence Tier
{json.dumps(by_confidence, indent=2)}

## Performance by Entry Timing
{json.dumps(by_timing, indent=2)}

## Your Task
Analyze the performance data and recommend specific changes to the strategy config.
Think step by step:
1. Which confidence tiers are profitable? Which are losing money?
2. Is the entry timing optimal?
3. Should we be more or less selective with trades?
4. Are there patterns in the losses?

Then provide your recommendations in this exact JSON format:
```json
{{
  "analysis": "Your plain English analysis of what's happening",
  "changes": [
    {{"param": "min_confidence", "old": 0.80, "new": 0.85, "reason": "why"}},
    {{"param": "entry_seconds", "old": 30, "new": 20, "reason": "why"}}
  ],
  "confidence": "high/medium/low",
  "expected_improvement": "what you expect to happen"
}}
```

Only suggest changes you're confident will improve performance. If the current config is working well, suggest no changes."""

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 8192},
            },
            timeout=120,
        )
        
        if r.status_code != 200:
            return {"error": f"Ollama HTTP {r.status_code}", "changes": []}
        
        response_text = r.json()["message"]["content"]
        
        # Extract JSON from response
        import re
        match = re.search(r'```json\s*\n?(.*?)\n?\s*```', response_text, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
            result["raw_analysis"] = response_text
            return result
        else:
            return {
                "analysis": response_text,
                "changes": [],
                "error": "Could not parse JSON from response",
                "raw_analysis": response_text,
            }
    
    except Exception as e:
        return {"error": str(e), "changes": []}


def apply_changes(current_config: dict, analysis: dict) -> dict:
    """Apply recommended changes to the strategy config."""
    new_config = current_config.copy()
    
    changes_applied = []
    for change in analysis.get("changes", []):
        param = change.get("param")
        new_val = change.get("new")
        
        if param and new_val is not None and param in new_config:
            old_val = new_config[param]
            new_config[param] = new_val
            changes_applied.append(f"{param}: {old_val} → {new_val} ({change.get('reason', '')})")
    
    new_config["notes"] = analysis.get("analysis", "")[:500]
    new_config["updated"] = datetime.now().isoformat()
    new_config["version"] = current_config.get("version", 0) + 1
    new_config["last_analysis"] = {
        "confidence": analysis.get("confidence", "unknown"),
        "expected_improvement": analysis.get("expected_improvement", ""),
        "changes_applied": changes_applied,
    }
    
    return new_config


def save_config(config: dict):
    """Save updated strategy config."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def log_analysis(analysis: dict, trades_analyzed: int):
    """Log the analysis for review."""
    log_path = LOGS_PATH / "strategy_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "trades_analyzed": trades_analyzed,
        "changes_recommended": len(analysis.get("changes", [])),
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
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    print("🧠 Qwen Strategist — Meta-Learning Brain")
    print("=" * 50)

    # Check Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  Ollama: ✅ ({len(models)} models)")
        
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
    except:
        print("  ❌ Ollama not running. Start with: ollama serve")
        return

    # Load data
    trades = load_recent_trades(days=args.days)
    current_config = load_current_strategy()
    
    print(f"  Trades to analyze: {len(trades)}")
    print(f"  Current config version: {current_config.get('version', 0)}")

    if len(trades) < 5:
        print(f"\n  ⚠️  Not enough trades to analyze ({len(trades)} found, need ≥5)")
        print(f"  Run the live bot first to collect trade data.")
        return

    # Analyze
    print(f"\n  🔍 Analyzing with {model}...")
    analysis = analyze_with_qwen(trades, current_config, model)

    if "error" in analysis and not analysis.get("changes"):
        print(f"  ❌ Analysis failed: {analysis['error']}")
        return

    print(f"\n  📋 Analysis:")
    print(f"  {analysis.get('analysis', '')[:300]}")
    
    changes = analysis.get("changes", [])
    if changes:
        print(f"\n  🔧 Recommended changes ({len(changes)}):")
        for c in changes:
            print(f"    {c['param']}: {c.get('old')} → {c.get('new')} | {c.get('reason', '')}")
    else:
        print(f"\n  ✅ No changes recommended — current config looks good!")

    # Apply
    if not args.dry_run and changes:
        new_config = apply_changes(current_config, analysis)
        save_config(new_config)
        print(f"\n  💾 Config updated (v{new_config['version']}): {CONFIG_PATH}")
    elif args.dry_run:
        print(f"\n  (dry-run — no changes saved)")

    # Log
    log_analysis(analysis, len(trades))
    print(f"\n  📝 Analysis logged")


if __name__ == "__main__":
    main()
