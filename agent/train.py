"""
RL Agent Training — Uses PPO (Proximal Policy Optimization)
PPO is the same algorithm used by OpenAI to train game-playing agents.
It learns by trying actions, getting rewards, and adjusting weights.

Usage:
    python agent/train.py                    # Train from scratch
    python agent/train.py --resume           # Continue training
    python agent/train.py --timesteps 500000 # More training
    python agent/train.py --reward-config configs/reward_config.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CANDLE_PATH = PROJECT_ROOT / "data" / "candles.json"
DEFAULT_REWARD_CONFIG_PATH = PROJECT_ROOT / "configs" / "reward_config.json"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def load_reward_config(path: str | Path = DEFAULT_REWARD_CONFIG_PATH) -> dict | None:
    """Load reward config if it exists."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def train_model(candle_path: str | Path = DEFAULT_CANDLE_PATH,
                reward_config: dict = None,
                timesteps: int = 200000,
                resume: bool = False,
                model_save_path: str | Path = None) -> dict:
    """
    Train a PPO model and return evaluation metrics.
    Callable by orchestrator.py or via CLI.

    Returns dict with: avg_win_rate, avg_pnl, avg_trades_per_session,
                        avg_skip_rate, model_path
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("  stable-baselines3 not installed")
        print("   pip install stable-baselines3 torch")
        return {"error": "stable-baselines3 not installed"}

    from env.polymarket_env import PolymarketEnv

    # Load candle data
    candle_path = Path(candle_path)
    if not candle_path.exists():
        print(f"  Candle data not found: {candle_path}")
        print(f"  Run: python data/fetch_prices.py")
        return {"error": f"Candle data not found: {candle_path}"}

    print("  Loading market data...")
    with open(candle_path) as f:
        candles = json.load(f)
    print(f"   {len(candles)} candles loaded")

    if len(candles) < 100:
        return {"error": f"Not enough candles ({len(candles)}). Need at least 100."}

    # Split: 80% train, 20% eval
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    print(f"   Train: {len(train_candles)} | Eval: {len(eval_candles)}")
    if reward_config:
        print(f"   Reward config v{reward_config.get('version', '?')}: "
              f"skip={reward_config.get('skip_penalty')}, "
              f"win_mult={reward_config.get('win_bonus_multiplier')}, "
              f"loss_mult={reward_config.get('loss_penalty_multiplier')}")

    # Create environments
    def make_train_env():
        env = PolymarketEnv(train_candles, bet_size=1.0,
                           initial_bankroll=25.0, max_steps=500,
                           reward_config=reward_config)
        return Monitor(env)

    def make_eval_env():
        return PolymarketEnv(eval_candles, bet_size=1.0,
                            initial_bankroll=25.0, max_steps=200,
                            reward_config=reward_config)

    train_env = make_vec_env(make_train_env, n_envs=4)  # 4 parallel envs
    eval_env = make_eval_env()

    model_dir = str(DEFAULT_MODEL_DIR)
    os.makedirs(model_dir, exist_ok=True)

    if model_save_path is None:
        model_save_path = os.path.join(model_dir, "candidate_model")
    else:
        model_save_path = str(model_save_path)
        if model_save_path.endswith(".zip"):
            model_save_path = model_save_path[:-4]

    resume_path = os.path.join(model_dir, "best_model")

    # Load existing model or create new
    if resume and os.path.exists(resume_path + ".zip"):
        print(f"  Resuming from {resume_path}")
        model = PPO.load(resume_path, env=train_env, device="cpu")
    else:
        print("  Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,        # Steps per env before update
            batch_size=64,
            n_epochs=10,         # Epochs per update
            gamma=0.99,          # Discount factor
            gae_lambda=0.95,     # GAE parameter
            clip_range=0.2,      # PPO clip range
            ent_coef=0.01,       # Entropy bonus (encourages exploration)
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={
                "net_arch": [256, 256, 128],  # Neural network layers
            },
            verbose=1,
            device="cpu",
            tensorboard_log=os.path.join(str(PROJECT_ROOT), "logs"),
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=os.path.join(str(PROJECT_ROOT), "logs"),
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix="ppo_trader_checkpoint",
    )

    # Train
    print(f"\n  Training for {timesteps:,} timesteps...")
    print(f"   This will take ~{timesteps // 50000} minutes on CPU")
    print(f"   GPU will be much faster\n")

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(model_save_path)
    print(f"\n  Model saved: {model_save_path}.zip")

    # Evaluate final model
    print("\n  Final Evaluation (50 episodes)...")
    all_stats = []
    for _ in range(50):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = eval_env.step(int(action))
        all_stats.append(eval_env.get_stats())

    avg_wr = np.mean([s["win_rate"] for s in all_stats])
    avg_pnl = np.mean([s["total_pnl"] for s in all_stats])
    avg_trades = np.mean([s["trades_made"] for s in all_stats])
    avg_skip = np.mean([s["skip_rate"] for s in all_stats])

    print(f"\n   Win Rate: {avg_wr:.1%}")
    print(f"   Avg P&L:  ${avg_pnl:+.2f}")
    print(f"   Avg Trades/Session: {avg_trades:.0f}")
    print(f"   Skip Rate: {avg_skip:.1%}")

    # Save eval results
    results = {
        "timestamp": datetime.now().isoformat(),
        "timesteps_trained": timesteps,
        "avg_win_rate": float(avg_wr),
        "avg_pnl": float(avg_pnl),
        "avg_trades_per_session": float(avg_trades),
        "avg_skip_rate": float(avg_skip),
        "reward_config_version": reward_config.get("version", 0) if reward_config else 0,
        "model_path": model_save_path + ".zip",
    }
    results_path = os.path.join(model_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candle-path", type=str, default=str(DEFAULT_CANDLE_PATH))
    parser.add_argument("--reward-config", type=str, default=str(DEFAULT_REWARD_CONFIG_PATH))
    parser.add_argument("--save-path", type=str, default=None,
                       help="Where to save the trained model (default: models/candidate_model.zip)")
    args = parser.parse_args()

    reward_config = load_reward_config(args.reward_config)

    train_model(
        candle_path=args.candle_path,
        reward_config=reward_config,
        timesteps=args.timesteps,
        resume=args.resume,
        model_save_path=args.save_path,
    )
