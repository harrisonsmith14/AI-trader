"""
RL Agent Training — Uses PPO (Proximal Policy Optimization)
PPO is the same algorithm used by OpenAI to train game-playing agents.
It learns by trying actions, getting rewards, and adjusting weights.

Usage:
    python agent/train.py                    # Train from scratch
    python agent/train.py --resume           # Continue training
    python agent/train.py --timesteps 500000 # More training
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
from datetime import datetime

def train(args):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("❌ stable-baselines3 not installed")
        print("   pip install stable-baselines3 torch")
        return

    from env.polymarket_env import PolymarketEnv

    # Load candle data
    candle_path = os.path.join(os.path.dirname(__file__), "..", "..", 
                               "polymarket-bots", "bots", "btc_5min", "candle_cache.json")
    if not os.path.exists(candle_path):
        print(f"❌ Candle data not found: {candle_path}")
        return

    print("📊 Loading market data...")
    with open(candle_path) as f:
        candles = json.load(f)
    print(f"   {len(candles)} candles loaded")

    # Split: 80% train, 20% eval
    split = int(len(candles) * 0.8)
    train_candles = candles[:split]
    eval_candles = candles[split:]

    print(f"   Train: {len(train_candles)} | Eval: {len(eval_candles)}")

    # Create environments
    def make_train_env():
        env = PolymarketEnv(train_candles, bet_size=1.0, 
                           initial_bankroll=25.0, max_steps=500)
        return Monitor(env)

    def make_eval_env():
        return PolymarketEnv(eval_candles, bet_size=1.0,
                            initial_bankroll=25.0, max_steps=200)

    train_env = make_vec_env(make_train_env, n_envs=4)  # 4 parallel envs
    eval_env = make_eval_env()

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ppo_trader")

    # Load existing model or create new
    if args.resume and os.path.exists(model_path + ".zip"):
        print(f"📂 Resuming from {model_path}")
        model = PPO.load(model_path, env=train_env)
    else:
        print("🧠 Creating new PPO model...")
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
            tensorboard_log=os.path.join(os.path.dirname(__file__), "..", "logs"),
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=os.path.join(os.path.dirname(__file__), "..", "logs"),
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
    print(f"\n🚀 Training for {args.timesteps:,} timesteps...")
    print(f"   This will take ~{args.timesteps // 50000} minutes on CPU")
    print(f"   GPU will be much faster\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(model_path)
    print(f"\n✅ Model saved: {model_path}.zip")

    # Evaluate final model
    print("\n📊 Final Evaluation (50 episodes)...")
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
        "timesteps_trained": args.timesteps,
        "avg_win_rate": avg_wr,
        "avg_pnl": avg_pnl,
        "avg_trades_per_session": avg_trades,
        "avg_skip_rate": avg_skip,
    }
    results_path = os.path.join(model_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Results saved: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
