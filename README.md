# RL Trader + Qwen Brain

A self-improving AI trading system for Polymarket BTC 5-minute prediction markets.

Two layers work together in a closed feedback loop:
- **Fast Layer (RL):** PPO neural network makes real-time trade decisions every 5 minutes
- **Slow Layer (Qwen):** Local LLM reasons about performance and adjusts reward shaping weights
- **Orchestrator:** Runs the improvement loop: backtest → analyze → retrain → validate → repeat

## How It Self-Improves

```
1. FETCH    → Get 30 days of BTC candle data (Coinbase/Kraken APIs)
2. BACKTEST → Run current model on held-out data, log every trade
3. ANALYZE  → Qwen reads trade details, identifies what's failing
               Outputs reward_config.json (changes HOW the RL model learns)
               e.g., "Increase penalty for low-confidence trades from -0.05 to -0.15"
4. RETRAIN  → PPO trains with new reward weights
5. VALIDATE → Backtest candidate model vs current best
               Promote only if better on 2/3 metrics (win rate, P&L, drawdown)
6. LOOP     → Repeat every 3 days until profitable
```

The key insight: Qwen doesn't just update trading parameters — it adjusts **reward shaping weights** that change what the RL model optimizes for during retraining.

## Price Source

Polymarket resolves against **Chainlink BTC/USD data stream** (`data.chain.link/streams/btc-usd`), NOT exchange spot prices. The bot:
- Gets the **Price To Beat** from Polymarket's GAMMA API (the exact Chainlink reference price)
- Uses exchange prices only as fallback for live BTC tracking
- Checks actual market resolution via API for accurate win/loss tracking

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# Optional for live trading:
pip install playwright py-clob-client
playwright install chromium
```

### 2. Fetch Training Data
```bash
python data/fetch_prices.py --days 30
# Fetches ~8,640 five-minute candles from Coinbase
```

### 3. Install Qwen locally
```bash
ollama pull qwen2.5:7b
# or for better reasoning:
ollama pull qwen3:8b
```

### 4. Run the Self-Improving Loop
```bash
python orchestrator.py --iterations 5 --timesteps 200000
# Each iteration: backtest → Qwen analysis → retrain → validate
# Takes ~30-60 min per iteration on CPU, ~10 min on GPU
```

### 5. Monitor Progress
```bash
# Check evolution across iterations:
cat logs/evolution.jsonl | python -m json.tool

# Check current strategy:
cat configs/strategy.json

# Check reward config (what Qwen adjusted):
cat configs/reward_config.json
```

### 6. Run Live Trading (after model is profitable)
```bash
python live_trader.py          # Dry run
python live_trader.py --live   # Real trades
```

## Architecture

```
orchestrator.py ─── The self-improving loop
    │
    ├── data/fetch_prices.py ──── Coinbase/Kraken candle fetcher
    │                              + Polymarket GAMMA API (PTB, resolution)
    │
    ├── agent/train.py ────────── PPO training (accepts reward_config)
    │
    ├── agent/backtest.py ─────── Model validation on held-out data
    │
    ├── brain/qwen_strategist.py ─ Qwen meta-learning brain
    │   │                           Outputs: strategy.json + reward_config.json
    │   │
    │   └── configs/
    │       ├── strategy.json ──── Trading gates (min delta, timing, bet size)
    │       └── reward_config.json ─ Reward weights for RL training
    │
    ├── env/polymarket_env.py ──── Custom Gym env (configurable rewards)
    │
    └── live_trader.py ─────────── Live trading with strategy gate
        │                           PTB from API (no Playwright needed)
        │                           Qwen's config gates RL decisions
        │
        └── logs/
            ├── evolution.jsonl ── Cross-iteration metrics
            ├── backtest_results.jsonl
            ├── backtest_trades.jsonl ── Per-trade details for Qwen
            ├── live_trades.jsonl
            └── strategy_log.jsonl
```

## The Feedback Loop (How Qwen Reaches the RL Model)

Previous problem: Qwen updated `strategy.json`, but the RL model ignored it.

Solution: Qwen now controls **reward shaping weights** in `reward_config.json`:

```json
{
  "skip_penalty": -0.02,
  "win_bonus_multiplier": 1.0,
  "loss_penalty_multiplier": 1.0,
  "high_confidence_bonus": 0.05,
  "low_confidence_penalty": -0.15,
  "low_confidence_threshold": 0.02
}
```

These weights are loaded by `PolymarketEnv` during training. When Qwen says "stop trading weak signals," it increases `low_confidence_penalty`. The RL model then learns through gradient descent to avoid those situations.

Additionally, `strategy.json` now acts as a **gate** on live RL decisions — even when the model is loaded, Qwen's filters (min delta, time-of-day, circuit breaker) are applied.

## ML Concepts

| Concept | Where It's Used |
|---------|----------------|
| **Neural Network** | PPO policy — 3 layers (256, 256, 128 neurons) |
| **Reinforcement Learning** | Agent gets reward (+profit) or punishment (-loss) |
| **PPO Algorithm** | How weights update — clips large changes for stability |
| **Reward Shaping** | Qwen adjusts reward weights to guide what the model learns |
| **Meta-learning** | Qwen reasoning about HOW to learn better |
| **Backtesting** | Validate model on held-out data before deployment |
| **Convergence** | Track improvement across iterations until profitable |
