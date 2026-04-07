"""
Autonomous Weather Trading Agent — The main loop.

Runs continuously, discovers weather markets, fetches forecasts,
calls the Qwen-written strategy, trades, and triggers analysis.
Qwen rewrites the strategy based on real results.

Usage:
    python -m autonomous_agent.agent                    # Dry run
    python -m autonomous_agent.agent --live             # Real trades
    python -m autonomous_agent.agent --city NYC         # Focus on one city
    python -m autonomous_agent.agent --model qwen3:8b   # Specify Qwen model
"""

import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

from . import weather_data, market_api, journal, sandbox, analyst

logger = logging.getLogger(__name__)


def print_banner(strategy_version: int, stats: dict):
    """Print the status banner."""
    wr = stats.get("win_rate", 0)
    wins = stats.get("wins", 0)
    total = stats.get("resolved", 0)
    pnl = stats.get("total_pnl", 0)
    day = max(1, total // 5 + 1)  # rough day estimate

    print(f"\n{'='*60}")
    print(f"  AUTONOMOUS WEATHER TRADER — Day {day}")
    print(f"  Strategy: v{strategy_version} | Win Rate: {wr}% ({wins}/{total}) | P&L: ${pnl:+.2f}")
    print(f"{'='*60}\n")


def scan_and_decide(cities: list[str], decide_fn, strategy_version: int,
                    dry_run: bool = True):
    """
    Scan weather markets and make trading decisions.
    This is the core cycle that runs each period.
    """
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")

    print(f"  [{now.strftime('%H:%M')}] Scanning weather markets...")

    # Get forecasts for each city
    for city in cities:
        forecasts = weather_data.get_all_forecasts(city, tomorrow)

        if not forecasts["nws_forecast"] and not forecasts["gfs_mean"]:
            print(f"  {city}: No forecast data available, skipping")
            continue

        nws = forecasts["nws_forecast"]
        gfs = forecasts["gfs_mean"]
        gfs_spread = forecasts["gfs_spread"]

        print(f"\n  {city} {tomorrow} — High Temperature")
        if nws:
            print(f"    NWS Forecast: {nws}°F")
        if gfs:
            print(f"    GFS Ensemble: {gfs}°F "
                  f"(range: {forecasts['gfs_low']}-{forecasts['gfs_high']}°F)")
        if forecasts["forecast_bias"] is not None:
            print(f"    Forecast bias: {forecasts['forecast_bias']:+.1f}°F")

        # Try to find the market on Polymarket
        markets = market_api.find_weather_markets([city])

        # Build context for strategy
        brackets = []
        market_slug = None
        if markets:
            market = markets[0]
            brackets = market["brackets"]
            market_slug = market["slug"]
            print(f"    Market: {market['title']}")
            top_brackets = sorted(brackets, key=lambda b: b["price"], reverse=True)[:3]
            for b in top_brackets:
                print(f"      {b['range']}°F @ ${b['price']:.2f}")
        else:
            # Create synthetic brackets for dry-run observation
            if nws:
                center = int(round(nws))
                center = center - (center % 2)  # Round to even
                brackets = [
                    {"range": f"{center-4}-{center-3}", "low": center-4, "high": center-3, "price": 0.08},
                    {"range": f"{center-2}-{center-1}", "low": center-2, "high": center-1, "price": 0.20},
                    {"range": f"{center}-{center+1}", "low": center, "high": center+1, "price": 0.40},
                    {"range": f"{center+2}-{center+3}", "low": center+2, "high": center+3, "price": 0.22},
                    {"range": f"{center+4}-{center+5}", "low": center+4, "high": center+5, "price": 0.07},
                ]
            print(f"    (No live market found — using estimated brackets for observation)")

        context = {
            "city": city,
            "date": tomorrow,
            "nws_forecast": nws,
            "gfs_mean": gfs,
            "gfs_low": forecasts["gfs_low"],
            "gfs_high": forecasts["gfs_high"],
            "gfs_spread": gfs_spread,
            "brackets": brackets,
            "historical_bias": forecasts["forecast_bias"],
            "recent_trades": journal.get_recent_entries(days=14, entry_type="trade"),
            "win_rate": journal.get_trade_stats().get("win_rate", 0),
            "total_pnl": journal.get_trade_stats().get("total_pnl", 0),
            "strategy_version": strategy_version,
        }

        # Call strategy
        try:
            decision = decide_fn(context)
        except Exception as e:
            print(f"    Strategy error: {e}")
            decision = {"action": "SKIP", "bracket": None, "confidence": 0,
                       "reasoning": f"Strategy crashed: {e}"}

        action = decision.get("action", "SKIP")
        bracket = decision.get("bracket")
        confidence = decision.get("confidence", 0)
        reasoning = decision.get("reasoning", "")

        print(f"    Strategy says: {action}", end="")
        if bracket:
            bracket_price = next((b["price"] for b in brackets if b["range"] == bracket), 0)
            print(f" {bracket}°F @ ${bracket_price:.2f}")
        else:
            print()
        print(f"    Reasoning: \"{reasoning}\"")

        # Execute
        if action == "BUY" and bracket:
            bracket_info = next((b for b in brackets if b["range"] == bracket), None)
            if bracket_info:
                bracket_price = bracket_info["price"]
                token_id = bracket_info.get("token_id")

                if dry_run or not token_id:
                    print(f"    [DRY RUN] Would buy {bracket}°F for $1.00")
                    success = True
                else:
                    success = market_api.place_bracket_order(token_id, 1.0, dry_run=False)
                    print(f"    {'Order placed!' if success else 'Order FAILED'}")

                if success:
                    journal.log_trade(
                        city=city, date=tomorrow, bracket=bracket,
                        bracket_price=bracket_price, confidence=confidence,
                        reasoning=reasoning, strategy_version=strategy_version,
                        nws_forecast=nws, gfs_mean=gfs, gfs_spread=gfs_spread,
                        market_slug=market_slug,
                    )
            else:
                print(f"    Bracket {bracket} not found in market")

        elif action == "SKIP":
            journal.log_skip(
                city=city, date=tomorrow, reasoning=reasoning,
                strategy_version=strategy_version,
                nws_forecast=nws, gfs_mean=gfs,
                brackets=brackets, market_slug=market_slug,
            )


def check_resolutions(cities: list[str]):
    """Check if any recent markets have resolved and update journal."""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n  Checking resolutions for {yesterday}...")

    for city in cities:
        # Get historical actual temperature
        actuals = weather_data.get_historical_actuals(city, days=3)

        if not actuals:
            continue

        # Find yesterday's actual
        for record in actuals:
            if record["date"] == yesterday:
                actual = record["actual_high"]
                actual_int = int(round(actual))
                low = actual_int - (actual_int % 2)
                winning_bracket = f"{low}-{low+1}"

                print(f"    {city} {yesterday}: Actual {actual}°F → {winning_bracket}°F bracket")

                # Log resolution
                journal.log_resolution(city, yesterday, actual, winning_bracket)

                # Log weather history for bias tracking
                weather_data.log_weather_history({
                    "city": city,
                    "date": yesterday,
                    "actual_temp": actual,
                    "nws_forecast": None,  # We'd need to have saved this
                })

                # Update trade results
                trades = journal.get_recent_entries(days=3, entry_type="trade")
                for trade in trades:
                    if trade.get("city") == city and trade.get("date") == yesterday and trade.get("result") is None:
                        won = trade.get("bracket_chosen") == winning_bracket
                        bracket_price = trade.get("bracket_price", 0.5)
                        if won:
                            pnl = round((1.0 - bracket_price), 2)
                        else:
                            pnl = round(-bracket_price, 2)

                        result = "WIN" if won else "LOSS"
                        journal.update_trade_result(city, yesterday, result, pnl)
                        print(f"    Trade result: {result} | P&L: ${pnl:+.2f}")

                # Log observation for skipped markets
                skips = journal.get_recent_entries(days=3, entry_type="skip")
                for skip in skips:
                    if skip.get("city") == city and skip.get("date") == yesterday:
                        # What would have happened?
                        bracket_prices = skip.get("bracket_prices", [])
                        best_price = None
                        for bp in bracket_prices:
                            if isinstance(bp, dict):
                                for rng, price in bp.items():
                                    if rng == winning_bracket:
                                        best_price = price
                                        break

                        journal.log_observation(
                            city=city, date=yesterday,
                            nws_forecast=skip.get("nws_forecast"),
                            actual_temp=actual,
                            winning_bracket=winning_bracket,
                            best_bracket_price=best_price,
                            hypothetical_pnl=round(1.0 - best_price, 2) if best_price else None,
                        )
                        break


def should_analyze(strategy_version: int) -> tuple[bool, str]:
    """Determine if Qwen should analyze and potentially rewrite the strategy."""
    stats = journal.get_trade_stats()
    entries = journal.get_recent_entries(days=1)

    # Not enough data yet
    total_points = len(journal.get_recent_entries(days=14))
    if total_points < 5:
        return False, ""

    # Check for consecutive losses
    recent_trades = journal.get_recent_entries(days=7, entry_type="trade")
    resolved = [t for t in recent_trades if t.get("result") is not None]
    if len(resolved) >= 3:
        last_3 = resolved[-3:]
        if all(t.get("result") == "LOSS" for t in last_3):
            return True, "Emergency: 3 consecutive losses. Focus on what went wrong."

    # Daily analysis after resolutions come in
    resolutions_today = [e for e in entries if e.get("type") == "resolution"]
    if resolutions_today:
        return True, "Daily analysis after market resolutions."

    # After enough new observations
    recent_obs = journal.get_recent_entries(days=2)
    if len(recent_obs) >= 10:
        return True, f"Accumulated {len(recent_obs)} new data points."

    return False, ""


def main():
    parser = argparse.ArgumentParser(description="Autonomous Weather Trading Agent")
    parser.add_argument("--live", action="store_true", help="Enable real trading")
    parser.add_argument("--city", type=str, nargs="+", default=["NYC", "Chicago", "LA"],
                       help="Cities to trade (default: NYC Chicago LA)")
    parser.add_argument("--model", type=str, default="qwen3:8b",
                       help="Ollama model for analysis")
    parser.add_argument("--interval", type=int, default=3600,
                       help="Seconds between market scans (default: 3600 = 1 hour)")
    parser.add_argument("--analyze-now", action="store_true",
                       help="Force Qwen analysis immediately")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s %(levelname)s %(message)s")

    # Startup checks
    print("  Checking Ollama...")
    if not analyst.check_ollama(args.model):
        print(f"  Ollama not available with model {args.model}")
        print(f"  Run: ollama serve && ollama pull {args.model}")
        return

    # Load strategy
    print("  Loading strategy...")
    try:
        decide_fn = sandbox.load_strategy()
        strategy_version = sandbox.get_current_version()
    except Exception as e:
        print(f"  Failed to load strategy: {e}")
        return

    stats = journal.get_trade_stats()
    print_banner(strategy_version, stats)

    print(f"  Cities: {', '.join(args.city)}")
    print(f"  Mode: {'LIVE' if args.live else 'DRY RUN'}")
    print(f"  Scan interval: {args.interval}s ({args.interval // 60} min)")
    print(f"  Qwen model: {args.model}")
    print(f"\n  Running... (Ctrl+C to stop)\n")

    # Force analysis if requested
    if args.analyze_now:
        print(f"\n  Qwen analyzing (forced)...")
        result = analyst.analyze_and_rewrite(model=args.model, trigger_reason="Manual trigger")
        print(f"  Analysis: {result['analysis'][:200]}")
        if result["rewrote"]:
            print(f"  Strategy rewritten: v{result['old_version']} -> v{result['new_version']}")
            decide_fn = sandbox.load_strategy()
            strategy_version = result["new_version"]
        print()

    try:
        cycle = 0
        while True:
            cycle += 1

            # Check resolutions from previous days
            check_resolutions(args.city)

            # Scan markets and make decisions
            scan_and_decide(args.city, decide_fn, strategy_version, dry_run=not args.live)

            # Check if Qwen should analyze
            should, reason = should_analyze(strategy_version)
            if should:
                print(f"\n  Qwen analyzing: {reason}")
                result = analyst.analyze_and_rewrite(
                    model=args.model, trigger_reason=reason
                )
                print(f"  Analysis: {result['analysis'][:300]}")

                if result["rewrote"]:
                    print(f"\n  Strategy rewritten: v{result['old_version']} -> v{result['new_version']}")
                    decide_fn = sandbox.load_strategy()
                    strategy_version = result["new_version"]

                    # Check if we should revert (3 consecutive worse versions)
                    # This is tracked across cycles
                elif result.get("error"):
                    print(f"  Error: {result['error']}")
                else:
                    print(f"  No rewrite needed: {result['changes_summary']}")

            # Update banner
            stats = journal.get_trade_stats()
            print_banner(strategy_version, stats)

            # Wait for next cycle
            print(f"  Next scan in {args.interval // 60} minutes...")
            print(f"  (Ctrl+C to stop)")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        stats = journal.get_trade_stats()
        print(f"\n\n  SESSION SUMMARY")
        print(f"  Strategy version: v{strategy_version}")
        print(f"  Total trades: {stats['resolved']}")
        print(f"  Win rate: {stats['win_rate']}%")
        print(f"  P&L: ${stats['total_pnl']:+.2f}")
        print(f"  Journal: logs/weather_journal.jsonl")


if __name__ == "__main__":
    main()
