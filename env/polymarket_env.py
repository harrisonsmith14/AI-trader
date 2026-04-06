"""
Polymarket BTC 5-Min Trading Environment
A custom OpenAI Gymnasium environment for the RL agent to train in.

Observations (what the agent sees):
    - price_delta_pct: % change from window open to now (-1 to +1 scaled)
    - time_remaining_pct: fraction of 5-min window left (0 to 1)
    - volatility: recent BTC volatility (normalized)
    - recent_win_rate: agent's win rate over last 20 trades
    - recent_pnl: normalized recent P&L
    - hour_of_day: time of day (cos/sin encoded for cyclical)
    - yes_price: current UP token price on Polymarket (0-1)
    - no_price: current DOWN token price on Polymarket (0-1)

Actions:
    0 = SKIP (don't trade this window)
    1 = BUY UP
    2 = BUY DOWN

Reward:
    +profit if correct, -bet_size if wrong, small negative for skipping
    (encourages selective, accurate trading)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import json
import os
from collections import deque


DEFAULT_REWARD_CONFIG = {
    "skip_penalty": -0.02,
    "win_bonus_multiplier": 1.0,
    "loss_penalty_multiplier": 1.0,
    "high_confidence_bonus": 0.05,
    "high_confidence_threshold": 0.05,
    "low_confidence_penalty": -0.05,
    "low_confidence_threshold": 0.02,
    "no_funds_penalty": -0.1,
    "version": 0,
}


class PolymarketEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, candle_data: list, bet_size: float = 1.0,
                 initial_bankroll: float = 25.0, max_steps: int = 1000,
                 entry_offset_sec: int = 240, reward_config: dict = None):
        super().__init__()

        self.candles = candle_data  # List of {open_time, open, high, low, close}
        self.bet_size = bet_size
        self.initial_bankroll = initial_bankroll
        self.max_steps = max_steps
        self.entry_offset_sec = entry_offset_sec  # seconds into window to observe

        # Qwen-tunable reward weights (updated by brain/qwen_strategist.py)
        rc = DEFAULT_REWARD_CONFIG.copy()
        if reward_config:
            rc.update(reward_config)
        self.reward_config = rc

        # Build price lookup: timestamp (sec) -> close price
        self.price_at = {}
        for c in candles:
            ts = c["open_time"] // 1000
            self.price_at[ts] = c["close"]
        
        self.timestamps = sorted(self.price_at.keys())
        self.window_starts = [t for t in self.timestamps if t % 300 == 0]

        # Action space: 0=skip, 1=buy_up, 2=buy_down
        self.action_space = spaces.Discrete(3)

        # Observation space: 10 features, all normalized to [-1, 1] or [0, 1]
        self.observation_space = spaces.Box(
            low=np.float32([-1] * 10),
            high=np.float32([1] * 10),
            dtype=np.float32
        )

        self.reset()

    def _get_price(self, ts: int) -> float:
        """Get nearest price within 60 seconds."""
        for offset in range(0, 61):
            if ts + offset in self.price_at:
                return self.price_at[ts + offset]
            if ts - offset in self.price_at:
                return self.price_at[ts - offset]
        return None

    def _get_volatility(self, ts: int, lookback: int = 10) -> float:
        """Get recent price volatility (std of returns over last N minutes)."""
        prices = []
        for i in range(lookback):
            p = self._get_price(ts - i * 60)
            if p:
                prices.append(p)
        if len(prices) < 2:
            return 0.0
        returns = np.diff(np.log(prices))
        return float(np.std(returns)) * 100  # percentage

    def _estimate_token_prices(self, delta_pct: float):
        """Estimate YES/NO token prices from price delta."""
        abs_delta = abs(delta_pct)
        if abs_delta >= 0.10:
            winning_price = 0.92
        elif abs_delta >= 0.05:
            winning_price = 0.80
        elif abs_delta >= 0.02:
            winning_price = 0.65
        elif abs_delta >= 0.005:
            winning_price = 0.55
        else:
            winning_price = 0.50
        
        losing_price = 1.0 - winning_price
        
        if delta_pct >= 0:
            return winning_price, losing_price  # yes_price, no_price
        else:
            return losing_price, winning_price

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        ws = self.window_starts[self.current_window_idx]
        
        ptb = self._get_price(ws)
        entry_ts = ws + self.entry_offset_sec
        entry_price = self._get_price(entry_ts)

        if ptb is None or entry_price is None or ptb == 0:
            return np.zeros(10, dtype=np.float32)

        delta_pct = (entry_price - ptb) / ptb * 100

        # Time remaining (normalized)
        time_remaining = (300 - self.entry_offset_sec) / 300

        # Volatility (normalized, cap at 1%)
        vol = min(self._get_volatility(ws), 1.0)

        # Recent performance
        recent_wr = self.recent_wins / max(self.recent_total, 1)
        recent_pnl_norm = np.tanh(self.recent_pnl / 10)  # normalize to ~[-1, 1]

        # Time of day (sin/cos encoding for cyclicality)
        hour = (ws % 86400) / 3600  # 0-24
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Token prices
        yes_price, no_price = self._estimate_token_prices(delta_pct)

        # Delta confidence (scale -1 to +1, capped at ±0.2%)
        delta_normalized = np.tanh(delta_pct / 0.1)

        obs = np.array([
            np.float32(delta_normalized),          # 0: price direction signal
            np.float32(time_remaining),            # 1: time left in window
            np.float32(vol),                       # 2: recent volatility
            np.float32(recent_wr),                 # 3: recent win rate
            np.float32(recent_pnl_norm),           # 4: recent P&L
            np.float32(hour_sin),                  # 5: time of day (sin)
            np.float32(hour_cos),                  # 6: time of day (cos)
            np.float32(yes_price),                 # 7: estimated UP token price
            np.float32(no_price),                  # 8: estimated DOWN token price
            np.float32(min(self.bankroll / self.initial_bankroll, 2.0) - 1.0),  # 9: bankroll health
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bankroll = self.initial_bankroll
        self.current_window_idx = np.random.randint(
            10, max(11, len(self.window_starts) - 100)
        )
        self.steps = 0
        self.trades = []
        self.recent_wins = 0
        self.recent_total = 0
        self.recent_pnl = 0.0
        self._recent_trades = deque(maxlen=20)

        return self._get_observation(), {}

    def step(self, action: int):
        ws = self.window_starts[self.current_window_idx]
        ptb = self._get_price(ws)
        entry_ts = ws + self.entry_offset_sec
        entry_price = self._get_price(entry_ts)
        close_price = self._get_price(ws + 300)

        reward = 0.0
        info = {}

        if ptb and entry_price and close_price and ptb > 0:
            delta_pct = (entry_price - ptb) / ptb * 100
            abs_delta = abs(delta_pct)
            actual_dir = "UP" if close_price >= ptb else "DOWN"
            yes_price, no_price = self._estimate_token_prices(delta_pct)
            rc = self.reward_config

            if action == 0:  # SKIP
                reward = rc["skip_penalty"]
                info["action"] = "skip"

            elif action == 1:  # BUY UP
                if self.bankroll >= self.bet_size:
                    self.bankroll -= self.bet_size
                    if actual_dir == "UP":
                        profit = (1.0 - yes_price) / yes_price * self.bet_size
                        self.bankroll += self.bet_size + profit
                        reward = profit * rc["win_bonus_multiplier"]
                        # Bonus for correctly trading high-confidence setups
                        if abs_delta >= rc["high_confidence_threshold"]:
                            reward += rc["high_confidence_bonus"]
                        self._recent_trades.append(1)
                    else:
                        reward = -self.bet_size * rc["loss_penalty_multiplier"]
                        # Extra penalty for trading weak signals
                        if abs_delta < rc["low_confidence_threshold"]:
                            reward += rc["low_confidence_penalty"]
                        self._recent_trades.append(0)
                    info["action"] = "buy_up"
                    info["result"] = actual_dir == "UP"
                    info["delta_pct"] = delta_pct
                else:
                    reward = rc["no_funds_penalty"]

            elif action == 2:  # BUY DOWN
                if self.bankroll >= self.bet_size:
                    self.bankroll -= self.bet_size
                    if actual_dir == "DOWN":
                        profit = (1.0 - no_price) / no_price * self.bet_size
                        self.bankroll += self.bet_size + profit
                        reward = profit * rc["win_bonus_multiplier"]
                        if abs_delta >= rc["high_confidence_threshold"]:
                            reward += rc["high_confidence_bonus"]
                        self._recent_trades.append(1)
                    else:
                        reward = -self.bet_size * rc["loss_penalty_multiplier"]
                        if abs_delta < rc["low_confidence_threshold"]:
                            reward += rc["low_confidence_penalty"]
                        self._recent_trades.append(0)
                    info["action"] = "buy_down"
                    info["result"] = actual_dir == "DOWN"
                    info["delta_pct"] = delta_pct
                else:
                    reward = rc["no_funds_penalty"]

            # Update recent stats
            if action != 0 and len(self._recent_trades) > 0:
                self.recent_wins = sum(self._recent_trades)
                self.recent_total = len(self._recent_trades)
                self.recent_pnl += reward

        self.steps += 1
        self.current_window_idx += 1
        
        done = (self.steps >= self.max_steps or 
                self.bankroll < 1.0 or
                self.current_window_idx >= len(self.window_starts) - 1)

        self.trades.append({
            "action": action,
            "reward": reward,
            "bankroll": self.bankroll,
        })

        return self._get_observation(), reward, done, False, info

    def get_stats(self) -> dict:
        trades_made = [t for t in self.trades if t["action"] != 0]
        wins = [t for t in trades_made if t["reward"] > 0]
        total_pnl = self.bankroll - self.initial_bankroll
        return {
            "total_steps": self.steps,
            "trades_made": len(trades_made),
            "wins": len(wins),
            "win_rate": len(wins) / max(len(trades_made), 1),
            "total_pnl": total_pnl,
            "final_bankroll": self.bankroll,
            "skip_rate": 1 - len(trades_made) / max(self.steps, 1),
        }
