"""
Orchestrator — The Self-Improving Loop

Ties everything together: fetch data -> backtest -> Qwen analyzes -> retrain -> validate -> loop.
Runs until the model converges on profitable performance, then can switch to live trading.

Usage:
    python orchestrator.py                          # Run 5 iterations
    python orchestrator.py --iterations 10          # Run 10 iterations
    python orchestrator.py --no-fetch               # Skip data fetching (use cached)
    python orchestrator.py --timesteps 100000       # Shorter training per iteration
"""

import sys
import os
import json
import shutil
import time
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from data.fetch_prices import update_cache, load_cached_candles
from agent.backtest import backtest, save_backtest_log
from brain.qwen_strategist import (
    analyze_for_retraining, load_backtest_trades, load_backtest_summary,
    load_reward_config, load_current_strategy,
    apply_strategy_changes, apply_reward_config_changes, save_config,
    CONFIG_PATH, REWARD_CONFIG_PATH
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
CANDLE_PATH = PROJECT_ROOT / "data" / "candles.json"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.zip"
CANDIDATE_MODEL_PATH = MODEL_DIR / "candidate_model.zip"
EVOLUTION_LOG_PATH = PROJECT_ROOT / "logs" / "evolution.jsonl"

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


def split_candles(candles: list, train_pct: float = 0.7, val_pct: float = 0.15):
    """Split candles into train/val/test sets."""
    n = len(candles)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    return candles[:train_end], candles[train_end:val_end], candles[val_end:]


def log_evolution(entry: dict):
    """Append iteration metrics to evolution log."""
    EVOLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVOLUTION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def ensure_initial_model(candle_path: str, reward_config: dict = None,
                         timesteps: int = 200000) -> bool:
    """Train an initial model if none exists."""
    if BEST_MODEL_PATH.exists():
        return True

    print("\n  No existing model found. Training initial model...")
    from agent.train import train_model

    results = train_model(
        candle_path=candle_path,
        reward_config=reward_config,
        timesteps=timesteps,
        resume=False,
        model_save_path=str(CANDIDATE_MODEL_PATH).replace(".zip", ""),
    )

    if "error" in results:
        print(f"  Initial training failed: {results['error']}")
        return False

    # Promote initial model
    if CANDIDATE_MODEL_PATH.exists():
        shutil.copy(str(CANDIDATE_MODEL_PATH), str(BEST_MODEL_PATH))
        print(f"  Initial model promoted to {BEST_MODEL_PATH}")
        return True

    return False


def run_improvement_cycle(iteration: int, candles: list,
                          ollama_model: str = DEFAULT_OLLAMA_MODEL,
                          timesteps: int = 200000) -> dict:
    """
    Run one full improvement iteration:
    1. Backtest current model (baseline)
    2. Qwen analyzes results
    3. Retrain with new reward config
    4. Backtest candidate model
    5. Promote if better

    Returns dict with iteration metrics.
    """
    print(f"\n{'='*60}")
    print(f"  ITERATION {iteration}")
    print(f"{'='*60}")

    train_candles, val_candles, test_candles = split_candles(candles)
    print(f"  Data split: {len(train_candles)} train / {len(val_candles)} val / {len(test_candles)} test")

    # --- Step 1: Backtest current model ---
    print(f"\n  [1/5] Backtesting current model...")
    current_reward = load_reward_config()
    baseline = backtest(str(BEST_MODEL_PATH), val_candles, num_episodes=50,
                       reward_config=current_reward)

    if "error" in baseline:
        print(f"  Backtest failed: {baseline['error']}")
        return {"iteration": iteration, "status": "error", "error": baseline["error"]}

    print(f"    Win rate: {baseline['win_rate']:.1%}")
    print(f"    P&L: ${baseline['total_pnl']:+.2f}")
    print(f"    Sharpe: {baseline['sharpe_ratio']:.2f}")
    print(f"    Drawdown: {baseline['max_drawdown']:.1%}")

    # Save backtest for Qwen
    save_backtest_log(baseline)

    # --- Step 2: Qwen analyzes ---
    print(f"\n  [2/5] Qwen analyzing performance...")
    backtest_trades = baseline.get("trade_details", [])
    backtest_summary = {
        "win_rate": baseline["win_rate"],
        "total_pnl": baseline["total_pnl"],
        "avg_pnl_per_trade": baseline["avg_pnl_per_trade"],
        "sharpe_ratio": baseline["sharpe_ratio"],
        "max_drawdown": baseline["max_drawdown"],
        "skip_rate": baseline["skip_rate"],
    }
    current_strategy = load_current_strategy()

    analysis = analyze_for_retraining(
        backtest_trades, backtest_summary,
        current_reward, current_strategy, ollama_model
    )

    if "error" in analysis and not analysis.get("reward_config_changes"):
        print(f"  Qwen analysis failed: {analysis.get('error')}")
        print(f"  Continuing with current reward config...")
        # Don't abort — retrain with current config anyway
        analysis = {"strategy_changes": [], "reward_config_changes": [],
                    "analysis": "Qwen unavailable, retraining with current config"}

    qwen_reasoning = analysis.get("analysis", "")[:200]
    print(f"    Analysis: {qwen_reasoning}")

    strat_changes = analysis.get("strategy_changes", [])
    reward_changes = analysis.get("reward_config_changes", [])
    print(f"    Strategy changes: {len(strat_changes)}")
    print(f"    Reward changes: {len(reward_changes)}")

    # Apply changes
    if strat_changes:
        new_strategy = apply_strategy_changes(current_strategy, analysis)
        save_config(new_strategy, CONFIG_PATH)
        print(f"    Strategy config -> v{new_strategy['version']}")

    new_reward = current_reward
    if reward_changes:
        new_reward = apply_reward_config_changes(current_reward, analysis)
        save_config(new_reward, REWARD_CONFIG_PATH)
        print(f"    Reward config -> v{new_reward['version']}")

    # --- Step 3: Retrain ---
    print(f"\n  [3/5] Retraining with updated reward config...")
    from agent.train import train_model

    # Save train candles to a temp file for training
    train_candle_path = PROJECT_ROOT / "data" / "train_candles.json"
    with open(train_candle_path, "w") as f:
        json.dump(train_candles, f)

    train_results = train_model(
        candle_path=str(train_candle_path),
        reward_config=new_reward,
        timesteps=timesteps,
        resume=False,
        model_save_path=str(CANDIDATE_MODEL_PATH).replace(".zip", ""),
    )

    if "error" in train_results:
        print(f"  Training failed: {train_results['error']}")
        return {"iteration": iteration, "status": "train_error", "error": train_results["error"]}

    print(f"    Training complete: WR={train_results['avg_win_rate']:.1%}, "
          f"P&L=${train_results['avg_pnl']:+.2f}")

    # --- Step 4: Backtest candidate ---
    print(f"\n  [4/5] Backtesting candidate model...")
    candidate = backtest(str(CANDIDATE_MODEL_PATH), val_candles, num_episodes=50,
                        reward_config=new_reward)

    if "error" in candidate:
        print(f"  Candidate backtest failed: {candidate['error']}")
        return {"iteration": iteration, "status": "backtest_error", "error": candidate["error"]}

    print(f"    Win rate: {baseline['win_rate']:.1%} -> {candidate['win_rate']:.1%}")
    print(f"    P&L: ${baseline['total_pnl']:+.2f} -> ${candidate['total_pnl']:+.2f}")
    print(f"    Drawdown: {baseline['max_drawdown']:.1%} -> {candidate['max_drawdown']:.1%}")

    # --- Step 5: Promote or reject ---
    print(f"\n  [5/5] Evaluating candidate...")

    # Candidate must beat baseline on at least 2 of 3 key metrics
    wins = 0
    comparisons = []

    if candidate["win_rate"] > baseline["win_rate"] + 0.01:
        wins += 1
        comparisons.append(f"  win_rate: +")
    else:
        comparisons.append(f"  win_rate: -")

    if candidate["total_pnl"] > baseline["total_pnl"]:
        wins += 1
        comparisons.append(f"  pnl: +")
    else:
        comparisons.append(f"  pnl: -")

    if candidate["max_drawdown"] < baseline["max_drawdown"]:
        wins += 1
        comparisons.append(f"  drawdown: +")
    else:
        comparisons.append(f"  drawdown: -")

    promoted = wins >= 2

    if promoted:
        shutil.copy(str(CANDIDATE_MODEL_PATH), str(BEST_MODEL_PATH))
        print(f"\n  PROMOTED ({wins}/3 metrics better)")
        status = "promoted"
    else:
        print(f"\n  REJECTED ({wins}/3 metrics better, need 2)")
        status = "rejected"
        # Revert reward config if candidate was worse
        if reward_changes:
            save_config(current_reward, REWARD_CONFIG_PATH)
            print(f"    Reward config reverted to v{current_reward.get('version', 0)}")

    # Final test on held-out test set
    test_model = str(BEST_MODEL_PATH)
    test_reward = load_reward_config()
    test_metrics = backtest(test_model, test_candles, num_episodes=30,
                           reward_config=test_reward)

    test_summary = {}
    if "error" not in test_metrics:
        test_summary = {
            "win_rate": test_metrics["win_rate"],
            "total_pnl": test_metrics["total_pnl"],
            "sharpe_ratio": test_metrics["sharpe_ratio"],
            "max_drawdown": test_metrics["max_drawdown"],
        }
        print(f"\n  Test set (held-out): WR={test_metrics['win_rate']:.1%}, "
              f"P&L=${test_metrics['total_pnl']:+.2f}, "
              f"Sharpe={test_metrics['sharpe_ratio']:.2f}")

    # Log evolution
    entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "reward_config_version": new_reward.get("version", 0),
        "baseline": {
            "win_rate": baseline["win_rate"],
            "total_pnl": baseline["total_pnl"],
            "sharpe_ratio": baseline["sharpe_ratio"],
            "max_drawdown": baseline["max_drawdown"],
        },
        "candidate": {
            "win_rate": candidate["win_rate"],
            "total_pnl": candidate["total_pnl"],
            "sharpe_ratio": candidate["sharpe_ratio"],
            "max_drawdown": candidate["max_drawdown"],
        },
        "test_set": test_summary,
        "qwen_reasoning": qwen_reasoning,
        "reward_changes": len(reward_changes),
        "strategy_changes": len(strat_changes),
    }
    log_evolution(entry)

    return entry


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Self-improving trading bot orchestrator")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of improvement iterations")
    parser.add_argument("--timesteps", type=int, default=200000,
                       help="Training timesteps per iteration")
    parser.add_argument("--no-fetch", action="store_true",
                       help="Skip data fetching, use cached candles")
    parser.add_argument("--source", choices=["coinbase", "kraken"], default="coinbase")
    parser.add_argument("--days", type=int, default=30,
                       help="Days of historical data to fetch")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL)
    args = parser.parse_args()

    print("=" * 60)
    print("  SELF-IMPROVING TRADING BOT — ORCHESTRATOR")
    print(f"  Iterations: {args.iterations}")
    print(f"  Timesteps/iter: {args.timesteps:,}")
    print(f"  Qwen model: {args.ollama_model}")
    print("=" * 60)

    # --- Step 0: Fetch data ---
    if not args.no_fetch:
        print(f"\n  Fetching {args.days} days of candle data from {args.source}...")
        candles = update_cache(path=str(CANDLE_PATH), source=args.source, days=args.days)
    else:
        candles = load_cached_candles(str(CANDLE_PATH))

    if not candles or len(candles) < 200:
        print(f"  Not enough candle data ({len(candles) if candles else 0} candles, need >= 200)")
        print(f"  Run: python data/fetch_prices.py --days 30")
        return

    print(f"  Candle data: {len(candles)} candles")

    # --- Ensure initial model exists ---
    reward_config = load_reward_config()
    if not ensure_initial_model(str(CANDLE_PATH), reward_config, args.timesteps):
        print("  Could not create initial model. Aborting.")
        return

    # --- Run improvement loop ---
    results = []
    stale_count = 0

    for i in range(1, args.iterations + 1):
        result = run_improvement_cycle(
            iteration=i,
            candles=candles,
            ollama_model=args.ollama_model,
            timesteps=args.timesteps,
        )
        results.append(result)

        # Track staleness
        if result.get("status") == "rejected":
            stale_count += 1
        else:
            stale_count = 0

        # If 3 consecutive rejections, tell Qwen to try something different
        if stale_count >= 3:
            print(f"\n  3 consecutive rejections — system may be converged or stuck")
            print(f"  Consider: longer training, different data window, or manual review")
            # Don't abort — let it keep trying

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print(f"  ORCHESTRATOR COMPLETE — {args.iterations} iterations")
    print(f"{'='*60}")

    promoted = sum(1 for r in results if r.get("status") == "promoted")
    rejected = sum(1 for r in results if r.get("status") == "rejected")
    errors = sum(1 for r in results if "error" in r.get("status", ""))

    print(f"  Promoted: {promoted} | Rejected: {rejected} | Errors: {errors}")

    # Show progression
    if results:
        first = results[0]
        last = results[-1]
        if "baseline" in first and "test_set" in last and last["test_set"]:
            print(f"\n  Progression:")
            print(f"    Win rate: {first['baseline']['win_rate']:.1%} -> {last['test_set'].get('win_rate', 'N/A')}")
            print(f"    P&L: ${first['baseline']['total_pnl']:+.2f} -> ${last['test_set'].get('total_pnl', 'N/A')}")

    print(f"\n  Evolution log: {EVOLUTION_LOG_PATH}")
    print(f"  Best model: {BEST_MODEL_PATH}")
    print(f"  Strategy config: {CONFIG_PATH}")
    print(f"  Reward config: {REWARD_CONFIG_PATH}")


if __name__ == "__main__":
    main()
