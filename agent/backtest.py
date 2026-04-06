"""
Backtesting Engine — Evaluates trained models on historical data.

Runs a model through the PolymarketEnv and produces detailed per-trade logs
for Qwen analysis, plus summary metrics for the orchestrator.

Usage:
    python agent/backtest.py                              # Backtest best_model
    python agent/backtest.py --model models/candidate_model.zip
    python agent/backtest.py --compare models/best_model.zip models/candidate_model.zip
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CANDLE_PATH = PROJECT_ROOT / "data" / "candles.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.zip"
BACKTEST_LOG_PATH = PROJECT_ROOT / "logs" / "backtest_results.jsonl"


def backtest(model_path: str | Path, candle_data: list,
             num_episodes: int = 50, reward_config: dict = None) -> dict:
    """
    Run a trained model on candle data and return detailed metrics.

    Returns:
        {
            "win_rate": float,
            "total_pnl": float,
            "avg_pnl_per_trade": float,
            "sharpe_ratio": float,
            "max_drawdown": float,
            "avg_trades_per_episode": float,
            "skip_rate": float,
            "trade_details": list[dict],  # Individual trades for Qwen
            "episode_stats": list[dict],
        }
    """
    from env.polymarket_env import PolymarketEnv

    try:
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path), device="cpu")
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    all_episode_stats = []
    all_trade_details = []
    all_pnls = []

    for ep in range(num_episodes):
        env = PolymarketEnv(candle_data, bet_size=1.0,
                           initial_bankroll=25.0, max_steps=200,
                           reward_config=reward_config)
        obs, _ = env.reset()
        done = False
        step_num = 0
        bankroll_history = [env.bankroll]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            prev_bankroll = env.bankroll
            obs, reward, done, _, info = env.step(action)
            bankroll_history.append(env.bankroll)

            # Log individual trades (not skips) for Qwen analysis
            if info.get("action") and info["action"] != "skip":
                all_trade_details.append({
                    "episode": ep,
                    "step": step_num,
                    "action": info["action"],
                    "result": "WIN" if info.get("result") else "LOSS",
                    "reward": round(reward, 4),
                    "delta_pct": round(info.get("delta_pct", 0), 6),
                    "bankroll_before": round(prev_bankroll, 2),
                    "bankroll_after": round(env.bankroll, 2),
                })

            step_num += 1

        stats = env.get_stats()
        stats["episode"] = ep
        stats["bankroll_history"] = bankroll_history
        all_episode_stats.append(stats)

        # Track per-trade PnLs for Sharpe
        episode_pnl = env.bankroll - env.initial_bankroll
        all_pnls.append(episode_pnl)

    # Aggregate metrics
    avg_wr = np.mean([s["win_rate"] for s in all_episode_stats])
    avg_pnl = np.mean(all_pnls)
    total_trades = sum(s["trades_made"] for s in all_episode_stats)
    total_wins = sum(s["wins"] for s in all_episode_stats)
    avg_trades = np.mean([s["trades_made"] for s in all_episode_stats])
    avg_skip = np.mean([s["skip_rate"] for s in all_episode_stats])

    # Sharpe ratio (annualized from per-episode returns)
    if len(all_pnls) > 1 and np.std(all_pnls) > 0:
        sharpe = np.mean(all_pnls) / np.std(all_pnls)
    else:
        sharpe = 0.0

    # Max drawdown across all episodes
    max_dd = 0.0
    for stats in all_episode_stats:
        hist = stats["bankroll_history"]
        peak = hist[0]
        for val in hist:
            peak = max(peak, val)
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

    # Per-trade PnL
    trade_pnls = [t["reward"] for t in all_trade_details]
    avg_pnl_per_trade = np.mean(trade_pnls) if trade_pnls else 0.0

    return {
        "model_path": str(model_path),
        "num_episodes": num_episodes,
        "total_trades": total_trades,
        "total_wins": total_wins,
        "win_rate": float(avg_wr),
        "total_pnl": float(avg_pnl),
        "avg_pnl_per_trade": float(avg_pnl_per_trade),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_trades_per_episode": float(avg_trades),
        "skip_rate": float(avg_skip),
        "trade_details": all_trade_details,
        "episode_stats": [{k: v for k, v in s.items() if k != "bankroll_history"}
                         for s in all_episode_stats],
    }


def compare_models(model_a_path: str, model_b_path: str,
                   candle_data: list, num_episodes: int = 50,
                   reward_config: dict = None) -> dict:
    """
    Compare two models on the same data. Returns which is better.
    """
    print(f"  Backtesting Model A: {model_a_path}")
    metrics_a = backtest(model_a_path, candle_data, num_episodes, reward_config)

    print(f"  Backtesting Model B: {model_b_path}")
    metrics_b = backtest(model_b_path, candle_data, num_episodes, reward_config)

    if "error" in metrics_a or "error" in metrics_b:
        return {"error": metrics_a.get("error") or metrics_b.get("error"),
                "model_a": metrics_a, "model_b": metrics_b}

    # Score: candidate must beat baseline on at least 2 of 3 metrics
    wins = 0
    comparisons = []

    if metrics_b["win_rate"] > metrics_a["win_rate"] + 0.01:
        wins += 1
        comparisons.append(f"win_rate: {metrics_a['win_rate']:.1%} -> {metrics_b['win_rate']:.1%} (+)")
    else:
        comparisons.append(f"win_rate: {metrics_a['win_rate']:.1%} -> {metrics_b['win_rate']:.1%} (-)")

    if metrics_b["total_pnl"] > metrics_a["total_pnl"]:
        wins += 1
        comparisons.append(f"pnl: ${metrics_a['total_pnl']:+.2f} -> ${metrics_b['total_pnl']:+.2f} (+)")
    else:
        comparisons.append(f"pnl: ${metrics_a['total_pnl']:+.2f} -> ${metrics_b['total_pnl']:+.2f} (-)")

    if metrics_b["max_drawdown"] < metrics_a["max_drawdown"]:
        wins += 1
        comparisons.append(f"drawdown: {metrics_a['max_drawdown']:.1%} -> {metrics_b['max_drawdown']:.1%} (+)")
    else:
        comparisons.append(f"drawdown: {metrics_a['max_drawdown']:.1%} -> {metrics_b['max_drawdown']:.1%} (-)")

    b_is_better = wins >= 2

    return {
        "model_a": metrics_a,
        "model_b": metrics_b,
        "b_is_better": b_is_better,
        "wins_for_b": wins,
        "comparisons": comparisons,
    }


def save_backtest_log(results: dict, path: Path = BACKTEST_LOG_PATH):
    """Save backtest results for Qwen analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save summary + trade details
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_path": results.get("model_path", ""),
        "win_rate": results.get("win_rate", 0),
        "total_pnl": results.get("total_pnl", 0),
        "avg_pnl_per_trade": results.get("avg_pnl_per_trade", 0),
        "sharpe_ratio": results.get("sharpe_ratio", 0),
        "max_drawdown": results.get("max_drawdown", 0),
        "skip_rate": results.get("skip_rate", 0),
        "num_episodes": results.get("num_episodes", 0),
        "total_trades": results.get("total_trades", 0),
    }

    with open(path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Save detailed trade log separately for Qwen
    detail_path = path.parent / "backtest_trades.jsonl"
    with open(detail_path, "w") as f:
        for trade in results.get("trade_details", []):
            f.write(json.dumps(trade) + "\n")

    print(f"  Backtest summary saved to {path}")
    print(f"  Trade details saved to {detail_path}")


def main():
    parser = argparse.ArgumentParser(description="Backtest trained RL models")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--candles", type=str, default=str(DEFAULT_CANDLE_PATH))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                       help="Compare two models head-to-head")
    parser.add_argument("--reward-config", type=str, default=None)
    args = parser.parse_args()

    # Load candles
    candle_path = Path(args.candles)
    if not candle_path.exists():
        print(f"  Candle data not found: {candle_path}")
        print(f"  Run: python data/fetch_prices.py")
        return

    with open(candle_path) as f:
        candles = json.load(f)
    print(f"  Loaded {len(candles)} candles")

    # Use validation split (last 20%)
    split = int(len(candles) * 0.8)
    val_candles = candles[split:]
    print(f"  Using validation set: {len(val_candles)} candles")

    # Load reward config if provided
    reward_config = None
    if args.reward_config and Path(args.reward_config).exists():
        with open(args.reward_config) as f:
            reward_config = json.load(f)

    if args.compare:
        result = compare_models(args.compare[0], args.compare[1],
                               val_candles, args.episodes, reward_config)
        if "error" in result:
            print(f"  Error: {result['error']}")
            return

        print(f"\n  === Comparison Results ===")
        for c in result["comparisons"]:
            print(f"    {c}")
        winner = "Model B" if result["b_is_better"] else "Model A"
        print(f"\n  Winner: {winner} ({result['wins_for_b']}/3 metrics)")
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"  Model not found: {model_path}")
            print(f"  Run: python agent/train.py")
            return

        print(f"  Backtesting: {model_path}")
        results = backtest(model_path, val_candles, args.episodes, reward_config)

        if "error" in results:
            print(f"  Error: {results['error']}")
            return

        print(f"\n  === Backtest Results ({args.episodes} episodes) ===")
        print(f"    Win Rate:       {results['win_rate']:.1%}")
        print(f"    Avg P&L:        ${results['total_pnl']:+.2f}")
        print(f"    Avg P&L/Trade:  ${results['avg_pnl_per_trade']:+.4f}")
        print(f"    Sharpe Ratio:   {results['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown:   {results['max_drawdown']:.1%}")
        print(f"    Trades/Episode: {results['avg_trades_per_episode']:.0f}")
        print(f"    Skip Rate:      {results['skip_rate']:.1%}")

        save_backtest_log(results)


if __name__ == "__main__":
    main()
