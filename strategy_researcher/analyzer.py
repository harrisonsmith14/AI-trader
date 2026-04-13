"""
Analyzer --- End-of-day analysis of strategy performance using Qwen.

Phase 4 of the daily cycle:
1. Compute final metrics for the day's strategy
2. Qwen analyzes what worked and what didn't
3. Save results JSON and analysis markdown
4. Update the strategy log with performance data
"""

import json
import logging
import re
import requests
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
STRATEGIES_DIR = Path(__file__).parent / "strategies"
STRATEGY_LOG_PATH = Path(__file__).parent / "strategy_log.json"


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


def analyze_day(day: int, research: dict, test_results: dict,
                strategy_code: str, model: str = "qwen3:8b") -> dict:
    """
    Perform end-of-day analysis using Qwen.

    Args:
        day: Day number (1-21)
        research: Output from researcher.research_strategy()
        test_results: Output from tester.run_test()
        strategy_code: The strategy source code
        model: Ollama model name

    Returns:
        {
            "analysis_text": str,     -- Qwen's analysis
            "metrics": dict,          -- Performance metrics
            "what_worked": str,
            "what_failed": str,
            "improvement_ideas": str,
        }
    """
    metrics = test_results["metrics"]
    trades = test_results["trades"]

    # Format trade samples for Qwen
    executed_trades = [t for t in trades if t.get("action") in ("UP", "DOWN")]
    sample_trades = executed_trades[-20:]  # Last 20 trades

    trades_str = ""
    for t in sample_trades:
        result = t.get("result", "pending")
        pnl = t.get("pnl", "?")
        delta = t.get("delta_pct", 0)
        trades_str += (
            f"  {t.get('action')} | conf={t.get('confidence', 0):.2f} | "
            f"delta={delta:+.4f}% | result={result} | pnl=${pnl}\n"
            f"    Reason: {t.get('reasoning', '')[:100]}\n"
        )

    if not trades_str:
        trades_str = "  No trades executed (all skipped)\n"

    # Count action distribution
    action_counts = {"UP": 0, "DOWN": 0, "SKIP": 0}
    for t in trades:
        action_counts[t.get("action", "SKIP")] = action_counts.get(t.get("action", "SKIP"), 0) + 1

    prompt = f"""You are a quantitative trading analyst reviewing a day's strategy performance.

## Strategy Tested (Day {day})
Name: {research['strategy_name']}
Category: {research['category']}
Key Concept: {research['key_concept']}

## Strategy Code
```python
{strategy_code[:2000]}
```

## Performance Metrics
- Total decisions: {metrics['total_decisions']}
- Trades executed: {metrics['total_trades']}
- Skipped: {metrics['skips']}
- Wins: {metrics['wins']} | Losses: {metrics['losses']}
- Win Rate: {metrics['win_rate']}%
- Total P&L: ${metrics['total_pnl']:+.4f}
- Avg P&L per trade: ${metrics['avg_pnl']:+.4f}
- Sharpe Ratio: {metrics['sharpe_ratio']}
- Max Drawdown: ${metrics['max_drawdown']:.4f}
- Avg Confidence: {metrics['avg_confidence']}

## Action Distribution
- UP: {action_counts.get('UP', 0)} | DOWN: {action_counts.get('DOWN', 0)} | SKIP: {action_counts.get('SKIP', 0)}

## Sample Trades (most recent)
{trades_str}

## Test Duration: {test_results['duration_hours']:.1f} hours

## Your Analysis
Provide a structured analysis covering:
1. **Overall Assessment** - Was this strategy profitable? How does it compare to random (50% baseline)?
2. **What Worked** - Specific patterns or conditions where the strategy performed well
3. **What Failed** - Specific conditions where the strategy made wrong calls
4. **Signal Quality** - Was the confidence calibrated? Did high confidence trades win more often?
5. **Edge Analysis** - Did this strategy find a real edge or was it noise?
6. **Improvement Ideas** - 2-3 specific, actionable improvements for this strategy approach
7. **Verdict** - 1 sentence: is this strategy worth exploring further?

Write in plain, direct language. Use ONLY ASCII characters. Be specific, not generic.
Refer to actual trade data in your analysis."""

    response = _call_ollama(prompt, model)

    if not response:
        response = "Analysis unavailable -- Ollama connection failed."

    # Parse sections from analysis
    what_worked = _extract_section(response, "What Worked")
    what_failed = _extract_section(response, "What Failed")
    improvement_ideas = _extract_section(response, "Improvement Ideas")

    result = {
        "analysis_text": response,
        "metrics": metrics,
        "what_worked": what_worked,
        "what_failed": what_failed,
        "improvement_ideas": improvement_ideas,
    }

    # Save all outputs
    _save_results(day, research, result, strategy_code)

    return result


def _extract_section(text: str, section_name: str) -> str:
    """Extract a section from Qwen's analysis by header name."""
    # Match markdown headers
    pattern = rf'\*?\*?{re.escape(section_name)}\*?\*?[:\-]*\s*\n(.*?)(?:\n\*?\*?\d|\n##|\n\*\*[A-Z]|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:500]
    return ""


def _save_results(day: int, research: dict, analysis: dict, strategy_code: str):
    """Save all day results to the strategies directory."""
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r'[^a-z0-9_]', '_', research["strategy_name"].lower())
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')

    # Save results JSON
    results_path = STRATEGIES_DIR / f"day_{day:02d}_results.json"
    results_data = {
        "day": day,
        "strategy_name": research["strategy_name"],
        "category": research["category"],
        "key_concept": research["key_concept"],
        "indicators": research["indicators"],
        "sources": research.get("sources", []),
        "metrics": analysis["metrics"],
        "what_worked": analysis["what_worked"],
        "what_failed": analysis["what_failed"],
        "improvement_ideas": analysis["improvement_ideas"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save analysis markdown
    analysis_path = STRATEGIES_DIR / f"day_{day:02d}_analysis.md"
    metrics = analysis["metrics"]
    sources_md = ""
    for s in research.get("sources", []):
        sources_md += f"- [{s.get('title', 'Source')}]({s.get('url', '')})\n"
    if not sources_md:
        sources_md = "- No web sources (strategy invented by AI)\n"

    md_content = f"""# Day {day}: {research['strategy_name']}

## Strategy Overview
- **Category:** {research['category']}
- **Key Concept:** {research['key_concept']}
- **Indicators:** {', '.join(research['indicators'])}

## Performance Summary
| Metric | Value |
|--------|-------|
| Total Trades | {metrics['total_trades']} |
| Win Rate | {metrics['win_rate']}% |
| Total P&L | ${metrics['total_pnl']:+.4f} |
| Sharpe Ratio | {metrics['sharpe_ratio']} |
| Max Drawdown | ${metrics['max_drawdown']:.4f} |
| Avg Confidence | {metrics['avg_confidence']} |

## Sources
{sources_md}
## Analysis
{analysis['analysis_text']}
"""
    analysis_path.write_text(md_content, encoding="utf-8")
    logger.info(f"Analysis saved to {analysis_path}")


def update_strategy_log(day: int, research: dict, metrics: dict):
    """
    Update the master strategy log with this day's results.
    This log is checked by the researcher to avoid repeating strategies.
    """
    log_path = STRATEGY_LOG_PATH
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = []

    # Check if day already exists, update if so
    existing_idx = None
    for i, entry in enumerate(log):
        if entry.get("day") == day:
            existing_idx = i
            break

    entry = {
        "day": day,
        "strategy_name": research["strategy_name"],
        "category": research["category"],
        "key_concept": research["key_concept"],
        "indicators": research["indicators"],
        "sources": [s.get("url", "") for s in research.get("sources", [])],
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if existing_idx is not None:
        log[existing_idx] = entry
    else:
        log.append(entry)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    logger.info(f"Strategy log updated for day {day}")


def generate_final_report(log_path: Path = None) -> str:
    """
    Generate a final report comparing all 21 strategies.
    Called after the 21-day sprint is complete.
    """
    if log_path is None:
        log_path = STRATEGY_LOG_PATH

    if not log_path.exists():
        return "No strategy log found."

    with open(log_path) as f:
        log = json.load(f)

    if not log:
        return "Strategy log is empty."

    # Sort by P&L
    log_sorted = sorted(log, key=lambda x: x.get("metrics", {}).get("total_pnl", 0), reverse=True)

    report = "# 21-Day Strategy Research Sprint -- Final Report\n\n"
    report += f"Total strategies tested: {len(log)}\n\n"

    # Summary table
    report += "## Strategy Rankings (by P&L)\n\n"
    report += "| Rank | Day | Strategy | Category | Trades | Win Rate | P&L | Sharpe |\n"
    report += "|------|-----|----------|----------|--------|----------|-----|--------|\n"

    for rank, entry in enumerate(log_sorted, 1):
        m = entry.get("metrics", {})
        report += (
            f"| {rank} | {entry['day']} | {entry['strategy_name']} | "
            f"{entry['category']} | {m.get('total_trades', 0)} | "
            f"{m.get('win_rate', 0)}% | ${m.get('total_pnl', 0):+.4f} | "
            f"{m.get('sharpe_ratio', 0)} |\n"
        )

    # Best and worst
    if log_sorted:
        best = log_sorted[0]
        worst = log_sorted[-1]
        report += f"\n## Best Strategy\n"
        report += f"**Day {best['day']}: {best['strategy_name']}**\n"
        report += f"- P&L: ${best.get('metrics', {}).get('total_pnl', 0):+.4f}\n"
        report += f"- Key Concept: {best['key_concept']}\n"

        report += f"\n## Worst Strategy\n"
        report += f"**Day {worst['day']}: {worst['strategy_name']}**\n"
        report += f"- P&L: ${worst.get('metrics', {}).get('total_pnl', 0):+.4f}\n"
        report += f"- Key Concept: {worst['key_concept']}\n"

    # Category breakdown
    by_category = {}
    for entry in log:
        cat = entry.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(entry.get("metrics", {}).get("total_pnl", 0))

    report += "\n## Performance by Category\n\n"
    for cat, pnls in sorted(by_category.items(), key=lambda x: sum(x[1]), reverse=True):
        avg_pnl = sum(pnls) / len(pnls)
        report += f"- **{cat}**: avg P&L ${avg_pnl:+.4f} ({len(pnls)} strategies)\n"

    return report
