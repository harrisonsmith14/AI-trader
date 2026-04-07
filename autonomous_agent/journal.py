"""
Trade Journal — Logs everything the AI does and observes.

This is what Qwen reads to learn from. Every trade, every skip, every
market observation, every resolution — all with rich context.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

JOURNAL_PATH = Path(__file__).parent.parent / "logs" / "weather_journal.jsonl"


def log_entry(entry: dict):
    """Append an entry to the journal."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_trade(city: str, date: str, bracket: str, bracket_price: float,
              confidence: float, reasoning: str, strategy_version: int,
              nws_forecast: float = None, gfs_mean: float = None,
              gfs_spread: float = None, market_slug: str = None):
    """Log a trade (BUY decision)."""
    log_entry({
        "type": "trade",
        "city": city,
        "date": date,
        "market_slug": market_slug,
        "bracket_chosen": bracket,
        "bracket_price": bracket_price,
        "action": "BUY",
        "confidence": confidence,
        "reasoning": reasoning,
        "strategy_version": strategy_version,
        "nws_forecast": nws_forecast,
        "gfs_mean": gfs_mean,
        "gfs_spread": gfs_spread,
        "result": None,  # Filled in at resolution
        "pnl": None,
    })


def log_skip(city: str, date: str, reasoning: str, strategy_version: int,
             nws_forecast: float = None, gfs_mean: float = None,
             brackets: list = None, market_slug: str = None):
    """Log a skip decision (for learning from what we didn't trade)."""
    log_entry({
        "type": "skip",
        "city": city,
        "date": date,
        "market_slug": market_slug,
        "action": "SKIP",
        "reasoning": reasoning,
        "strategy_version": strategy_version,
        "nws_forecast": nws_forecast,
        "gfs_mean": gfs_mean,
        "bracket_prices": [{b["range"]: b["price"]} for b in (brackets or [])],
        "result": None,  # Filled in at resolution
    })


def log_resolution(city: str, date: str, actual_temp: float,
                   winning_bracket: str, market_slug: str = None):
    """Log a market resolution — the actual outcome."""
    log_entry({
        "type": "resolution",
        "city": city,
        "date": date,
        "market_slug": market_slug,
        "actual_temp": actual_temp,
        "winning_bracket": winning_bracket,
    })


def log_observation(city: str, date: str, nws_forecast: float,
                    actual_temp: float, winning_bracket: str,
                    best_bracket_price: float = None,
                    hypothetical_pnl: float = None):
    """Log what would have happened on a market we skipped."""
    log_entry({
        "type": "observation",
        "city": city,
        "date": date,
        "nws_forecast": nws_forecast,
        "actual_temp": actual_temp,
        "winning_bracket": winning_bracket,
        "best_bracket_price": best_bracket_price,
        "hypothetical_pnl": hypothetical_pnl,
    })


def get_recent_entries(days: int = 7, entry_type: str = None) -> list[dict]:
    """Load recent journal entries."""
    if not JOURNAL_PATH.exists():
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    entries = []

    with open(JOURNAL_PATH) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("timestamp", "") > cutoff:
                    if entry_type is None or entry.get("type") == entry_type:
                        entries.append(entry)
            except (json.JSONDecodeError, ValueError):
                pass

    return entries


def get_trade_stats(days: int = 30) -> dict:
    """Compute trading statistics for Qwen analysis."""
    trades = get_recent_entries(days=days, entry_type="trade")
    resolved = [t for t in trades if t.get("result") is not None]

    if not resolved:
        return {
            "total_trades": len(trades),
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "by_city": {},
        }

    wins = sum(1 for t in resolved if t.get("result") == "WIN")
    losses = len(resolved) - wins
    total_pnl = sum(t.get("pnl", 0) for t in resolved if t.get("pnl"))

    # Stats by city
    by_city = {}
    for t in resolved:
        city = t.get("city", "unknown")
        if city not in by_city:
            by_city[city] = {"wins": 0, "losses": 0, "pnl": 0}
        if t.get("result") == "WIN":
            by_city[city]["wins"] += 1
        else:
            by_city[city]["losses"] += 1
        by_city[city]["pnl"] += t.get("pnl", 0)

    return {
        "total_trades": len(trades),
        "resolved": len(resolved),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(resolved) * 100, 1) if resolved else 0,
        "total_pnl": round(total_pnl, 2),
        "by_city": by_city,
    }


def update_trade_result(city: str, date: str, result: str, pnl: float):
    """Update a trade entry with its resolution result."""
    if not JOURNAL_PATH.exists():
        return

    lines = JOURNAL_PATH.read_text().strip().split("\n")
    updated = []
    found = False

    for line in lines:
        try:
            entry = json.loads(line)
            if (entry.get("type") == "trade" and
                entry.get("city") == city and
                entry.get("date") == date and
                entry.get("result") is None and
                not found):
                entry["result"] = result
                entry["pnl"] = pnl
                found = True
            updated.append(json.dumps(entry))
        except (json.JSONDecodeError, ValueError):
            updated.append(line)

    JOURNAL_PATH.write_text("\n".join(updated) + "\n")
