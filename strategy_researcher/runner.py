"""
Runner --- Main daily orchestrator for the 21-day strategy research sprint.

Coordinates all four phases of the daily cycle:
  Phase 1: Research -- discover a new strategy via web + Qwen
  Phase 2: Implement -- Qwen writes and validates strategy code
  Phase 3: Test -- dry-run against live BTC 5-min Polymarket markets
  Phase 4: Analyze -- Qwen reviews results, save everything

CLI:
    python -m strategy_researcher.runner --model qwen3:8b --day 1
    python -m strategy_researcher.runner --model qwen3:8b --day 2
    python -m strategy_researcher.runner --model qwen3:8b --auto
    python -m strategy_researcher.runner --report
"""

import argparse
import json
import logging
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path

from . import researcher, implementer, tester, analyzer

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


def check_ollama(model: str) -> bool:
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


def run_day(day: int, model: str = DEFAULT_MODEL,
            duration_hours: float = 24.0,
            bet_size: float = 1.0,
            dry_run: bool = True) -> bool:
    """
    Run a single day of the strategy research sprint.

    Args:
        day: Day number (1-21)
        model: Ollama model name
        duration_hours: How long to test the strategy
        bet_size: Simulated bet size
        dry_run: If True, don't place real orders

    Returns:
        True if all phases completed successfully.
    """
    print(f"\n{'='*60}")
    print(f"  STRATEGY RESEARCH SPRINT -- DAY {day}/21")
    print(f"  Model: {model}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Phase 1: Research ────────────────────────────────────────
    print(f"  Phase 1: RESEARCH")
    print(f"  {'─'*50}")
    start = time.time()

    research_result = researcher.research_strategy(day, model=model)
    if not research_result:
        print(f"  FAILED: Could not research a strategy for day {day}")
        return False

    elapsed = time.time() - start
    print(f"  Strategy: {research_result['strategy_name']}")
    print(f"  Category: {research_result['category']}")
    print(f"  Concept: {research_result['key_concept']}")
    print(f"  Indicators: {', '.join(research_result['indicators'])}")
    if research_result['sources']:
        print(f"  Sources: {len(research_result['sources'])} web results")
        for s in research_result['sources'][:3]:
            print(f"    - {s['title'][:60]}")
    else:
        print(f"  Sources: AI-generated (no web results used)")
    print(f"  Time: {elapsed:.0f}s\n")

    # ── Phase 2: Implement ───────────────────────────────────────
    print(f"  Phase 2: IMPLEMENT")
    print(f"  {'─'*50}")
    start = time.time()

    strategy_code = implementer.implement_strategy(research_result, day, model=model)
    if not strategy_code:
        print(f"  FAILED: Could not generate valid strategy code")
        return False

    elapsed = time.time() - start
    code_lines = strategy_code.count('\n') + 1
    print(f"  Generated {code_lines} lines of strategy code")
    print(f"  Validation: PASSED")
    print(f"  Time: {elapsed:.0f}s\n")

    # Load the decide function
    decide_fn = implementer.load_strategy(day)
    if not decide_fn:
        print(f"  FAILED: Could not load strategy decide() function")
        return False

    # ── Phase 3: Test ────────────────────────────────────────────
    print(f"  Phase 3: TEST")
    print(f"  {'─'*50}")
    print(f"  Running {'dry-run' if dry_run else 'live'} test for {duration_hours}h...")
    print(f"  Price source: Chainlink RTDS (CEX fallback)")
    print(f"  Market odds: CLOB API (real token prices)")
    start = time.time()

    test_results = tester.run_test(
        decide_fn, day,
        duration_hours=duration_hours,
        bet_size=bet_size,
        dry_run=dry_run,
    )

    elapsed = time.time() - start
    metrics = test_results["metrics"]
    print(f"\n  Test completed in {elapsed / 3600:.1f}h")
    print(f"  Trades: {metrics['total_trades']} | Skips: {metrics['skips']}")
    print(f"  Win Rate: {metrics['win_rate']}% | P&L: ${metrics['total_pnl']:+.4f}\n")

    # ── Phase 4: Analyze ─────────────────────────────────────────
    print(f"  Phase 4: ANALYZE")
    print(f"  {'─'*50}")
    start = time.time()

    analysis = analyzer.analyze_day(
        day, research_result, test_results, strategy_code, model=model
    )

    # Update the master strategy log
    analyzer.update_strategy_log(day, research_result, metrics)

    elapsed = time.time() - start
    print(f"  Analysis complete ({elapsed:.0f}s)")
    print(f"  Files saved:")

    strategies_dir = Path(__file__).parent / "strategies"
    for f in sorted(strategies_dir.glob(f"day_{day:02d}_*")):
        print(f"    - {f.name}")

    # Print analysis excerpt
    analysis_text = analysis.get("analysis_text", "")
    if analysis_text:
        excerpt = analysis_text[:300].replace('\n', '\n  ')
        print(f"\n  Analysis excerpt:")
        print(f"  {excerpt}...")

    print(f"\n{'='*60}")
    print(f"  DAY {day} COMPLETE")
    print(f"  Strategy: {research_result['strategy_name']}")
    print(f"  Result: {metrics['wins']}W/{metrics['losses']}L | "
          f"WR={metrics['win_rate']}% | P&L=${metrics['total_pnl']:+.4f}")
    print(f"{'='*60}\n")

    return True


def run_auto(model: str = DEFAULT_MODEL, start_day: int = 1,
             duration_hours: float = 24.0, bet_size: float = 1.0,
             dry_run: bool = True):
    """
    Run all 21 days automatically.
    Each day runs for duration_hours against live markets, then moves to the next.
    """
    log = researcher.load_strategy_log()
    completed_days = {entry["day"] for entry in log}

    for day in range(start_day, 22):
        if day in completed_days:
            print(f"\n  Day {day} already completed, skipping...")
            continue

        success = run_day(
            day, model=model,
            duration_hours=duration_hours,
            bet_size=bet_size,
            dry_run=dry_run,
        )

        if not success:
            print(f"\n  Day {day} failed. Continuing to next day...")

        # Brief pause between days
        if day < 21:
            print(f"\n  Waiting 30s before starting day {day + 1}...")
            time.sleep(30)

    # Generate final report
    report = analyzer.generate_final_report()
    report_path = Path(__file__).parent / "strategies" / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Final report saved to {report_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Strategy Research Sprint -- 21 days of BTC 5-min strategy exploration"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--day", type=int,
                        help="Run a specific day (1-21)")
    parser.add_argument("--auto", action="store_true",
                        help="Run all 21 days automatically")
    parser.add_argument("--start-day", type=int, default=1,
                        help="Start day for --auto mode (default: 1)")
    parser.add_argument("--duration", type=float, default=24.0,
                        help="Test duration in hours per day (default: 24)")
    parser.add_argument("--bet-size", type=float, default=1.0,
                        help="Simulated bet size in USD (default: 1.00)")
    parser.add_argument("--live", action="store_true",
                        help="Place real orders (default: dry run)")
    parser.add_argument("--report", action="store_true",
                        help="Generate final report from completed days")
    parser.add_argument("--status", action="store_true",
                        help="Show status of all 21 days")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Status command
    if args.status:
        _print_status()
        return

    # Report command
    if args.report:
        report = analyzer.generate_final_report()
        report_path = Path(__file__).parent / "strategies" / "final_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"\n  Report saved to {report_path}")
        return

    # Check Ollama
    if not check_ollama(args.model):
        print(f"\n  ERROR: Ollama not available with model {args.model}")
        print(f"  Start Ollama: ollama serve")
        print(f"  Pull model:   ollama pull {args.model}")
        sys.exit(1)

    print(f"\n  {'='*60}")
    print(f"  STRATEGY RESEARCH SPRINT")
    print(f"  21 days of BTC 5-min strategy exploration")
    print(f"  Model: {args.model}")
    print(f"  Mode: {'LIVE' if args.live else 'DRY RUN'}")
    print(f"  {'='*60}")

    if args.auto:
        run_auto(
            model=args.model,
            start_day=args.start_day,
            duration_hours=args.duration,
            bet_size=args.bet_size,
            dry_run=not args.live,
        )
    elif args.day:
        if not 1 <= args.day <= 21:
            print(f"  ERROR: Day must be between 1 and 21")
            sys.exit(1)
        run_day(
            args.day,
            model=args.model,
            duration_hours=args.duration,
            bet_size=args.bet_size,
            dry_run=not args.live,
        )
    else:
        parser.print_help()
        print(f"\n  Examples:")
        print(f"    python -m strategy_researcher.runner --day 1")
        print(f"    python -m strategy_researcher.runner --day 1 --duration 1")
        print(f"    python -m strategy_researcher.runner --auto")
        print(f"    python -m strategy_researcher.runner --auto --duration 12")
        print(f"    python -m strategy_researcher.runner --status")
        print(f"    python -m strategy_researcher.runner --report")


def _print_status():
    """Print the status of all 21 days."""
    log = researcher.load_strategy_log()
    strategies_dir = Path(__file__).parent / "strategies"

    print(f"\n  {'='*70}")
    print(f"  STRATEGY RESEARCH SPRINT -- STATUS")
    print(f"  {'='*70}")
    print(f"\n  {'Day':<5} {'Status':<12} {'Strategy':<30} {'Win Rate':<10} {'P&L':<12}")
    print(f"  {'─'*69}")

    log_by_day = {entry["day"]: entry for entry in log}
    total_pnl = 0

    for day in range(1, 22):
        if day in log_by_day:
            entry = log_by_day[day]
            m = entry.get("metrics", {})
            name = entry["strategy_name"][:28]
            wr = f"{m.get('win_rate', 0)}%"
            pnl = m.get("total_pnl", 0)
            total_pnl += pnl
            pnl_str = f"${pnl:+.4f}"
            print(f"  {day:<5} {'DONE':<12} {name:<30} {wr:<10} {pnl_str:<12}")
        else:
            files = list(strategies_dir.glob(f"day_{day:02d}_*.py"))
            if files:
                print(f"  {day:<5} {'IN PROGRESS':<12} {'(code exists)':<30} {'--':<10} {'--':<12}")
            else:
                print(f"  {day:<5} {'PENDING':<12} {'--':<30} {'--':<10} {'--':<12}")

    print(f"  {'─'*69}")
    print(f"  {'Total':<5} {'':<12} {len(log_by_day)} strategies{'':<18} {'':<10} ${total_pnl:+.4f}")
    print(f"  {'='*70}\n")


if __name__ == "__main__":
    main()
