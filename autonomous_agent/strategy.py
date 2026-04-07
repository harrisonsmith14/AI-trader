"""
Weather Trading Strategy v1
Written by: Qwen (initial template)
Last updated: startup

This file is rewritten by Qwen as it learns from real trading results.
Each version is saved in strategy_versions/ for rollback.
"""


def decide(context):
    """
    Decide whether to trade on a weather market.

    context dict contains:
        city: str               - e.g., "NYC"
        date: str               - e.g., "2026-04-08"
        nws_forecast: float     - NWS high temp forecast (degrees F)
        gfs_mean: float         - GFS ensemble mean
        gfs_low: float          - GFS ensemble low
        gfs_high: float         - GFS ensemble high
        gfs_spread: float       - GFS ensemble spread (high - low)
        brackets: list[dict]    - available brackets with prices
            [{"range": "54-55", "low": 54, "high": 55, "price": 0.42}, ...]
        historical_bias: float  - avg (forecast - actual) over recent days
        recent_trades: list     - last 20 trades with outcomes
        win_rate: float         - overall win rate
        total_pnl: float        - total profit/loss
        strategy_version: int   - current version number

    Returns:
        {
            "action": "BUY" or "SKIP",
            "bracket": "54-55" or None,    - which bracket to buy
            "confidence": float,           - 0.0 to 1.0
            "reasoning": str               - why this decision
        }
    """
    # v1: Observe only. Skip all markets and watch what happens.
    # Qwen will analyze the observations and write a real strategy.
    return {
        "action": "SKIP",
        "bracket": None,
        "confidence": 0.0,
        "reasoning": "v1: Observing market patterns. Collecting data on forecasts vs actuals."
    }
