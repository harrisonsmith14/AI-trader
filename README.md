# 🤖 RL Trader + 🧠 Qwen Brain

A self-improving AI trading system with two layers:
- **Fast Layer (RL):** PPO neural network makes real-time trade decisions every 5 minutes
- **Slow Layer (Qwen):** Local LLM reasons about performance nightly and updates strategy

## How It Learns

```
Day 1:  RL model trains on 86,400 historical BTC candles
        Learns basic patterns (when to buy UP vs DOWN vs SKIP)

Day 2+: Live bot trades $1 bets on Polymarket BTC 5-min markets
        Every trade is logged (direction, result, P&L, timing, conditions)

Nightly: Qwen wakes up, reads the trade log
         Reasons in plain English about what's working/failing
         Updates config: "I notice we lose at 60% confidence. Raising to 80%."
         RL model uses new config next day

Weekly:  RL model retrains on fresh data + live trades
         Gets smarter with every iteration
```

## Quick Start

### 1. Install Dependencies
```bash
pip install stable-baselines3 torch gymnasium playwright requests python-dotenv
playwright install chromium
```

### 2. Train the RL Model (RTX 5070 recommended)
```bash
python agent/train.py --timesteps 200000
# Takes ~5 min on GPU, ~60 min on CPU
```

### 3. Install Qwen locally
```bash
ollama pull qwen2.5:7b
# or for better reasoning:
ollama pull qwen3:8b
```

### 4. Run the live trader (dry run first)
```bash
python live_trader.py          # Dry run
python live_trader.py --live   # Real trades (VPN Japan required)
```

### 5. Run the Qwen brain nightly
```bash
python brain/qwen_strategist.py
# Or set up a cron: 0 2 * * * python /path/to/brain/qwen_strategist.py
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Live Trader (live_trader.py)        │
│  Every 5 minutes:                               │
│  1. Scrape "Price To Beat" from Polymarket      │
│  2. Get live BTC price (Coinbase + Kraken)      │
│  3. Build observation vector (10 features)      │
│  4. Ask PPO model: skip/buy_up/buy_down?        │
│  5. Execute via CLOB FOK order                  │
│  6. Log result for Qwen analysis                │
└────────────────┬────────────────────────────────┘
                 │ logs every trade
                 ▼
┌─────────────────────────────────────────────────┐
│          Qwen Strategist (runs nightly)         │
│  1. Load last 7 days of trade logs              │
│  2. Analyze: which conditions win/lose?         │
│  3. Reason in plain English                     │
│  4. Update configs/strategy.json                │
│  "min_confidence: 0.80 → 0.85 (60% tier losing)│
└────────────────┬────────────────────────────────┘
                 │ updates config
                 ▼
┌─────────────────────────────────────────────────┐
│           PPO Neural Network                    │
│  Retrain weekly on:                             │
│  - Historical candle data (86,400 candles)      │
│  - New live trade data                          │
│  - Updated Qwen-guided config                   │
│  Gets smarter every week                        │
└─────────────────────────────────────────────────┘
```

## Files

```
rl-trader/
├── env/
│   └── polymarket_env.py      # Custom Gym environment
├── agent/
│   └── train.py               # PPO training script
├── brain/
│   └── qwen_strategist.py     # Qwen meta-learning brain
├── configs/
│   └── strategy.json          # Qwen-updated strategy params
├── logs/
│   ├── live_trades.jsonl      # All live trades (for Qwen)
│   └── strategy_log.jsonl     # Qwen analysis history
├── models/
│   └── best_model.zip         # Trained PPO model
└── live_trader.py             # Main live trading loop
```

## The ML Concepts (for CIS 235)

| Concept | Where It's Used |
|---------|----------------|
| **Neural Network** | PPO policy — 3 layers (256, 256, 128 neurons) |
| **Weights** | The numbers the network adjusts during training |
| **Reinforcement Learning** | Agent gets reward (+profit) or punishment (-loss) |
| **PPO Algorithm** | How weights update — clips large changes for stability |
| **Observation Space** | The 10 features the agent "sees" each step |
| **Action Space** | 3 possible actions: skip, buy up, buy down |
| **Fine-tuning** | Weekly retraining on fresh market data |
| **Meta-learning** | Qwen reasoning about HOW to learn better |
