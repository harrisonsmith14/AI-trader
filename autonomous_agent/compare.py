"""
Model Comparison Runner — Run two LLMs side by side on the same markets.

Each model gets its own strategy file, version history, and journal.
Both see the same market data and forecasts, but write independent strategies.

Usage:
    python -m autonomous_agent.compare --model-a qwen3.5:9b --model-b gemma4:26b
    python -m autonomous_agent.compare --model-a qwen3.5:9b --model-b gemma4:26b --cycles 5
"""

import argparse
import logging
import time
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from . import weather_data, market_api, journal, sandbox, analyst

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR.parent / "logs"


def setup_model_workspace(model_name: str) -> dict:
    """Create isolated workspace for a model (strategy, versions, journal)."""
    safe_name = model_name.replace(":", "_").replace("/", "_")
    workspace = {
        "name": safe_name,
        "model": model_name,
        "strategy_path": BASE_DIR / f"strategy_{safe_name}.py",
        "versions_dir": BASE_DIR / f"strategy_versions_{safe_name}",
        "journal_path": LOGS_DIR / f"weather_journal_{safe_name}.jsonl",
    }

    workspace["versions_dir"].mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy current strategy as starting point if no strategy exists yet
    if not workspace["strategy_path"].exists():
        src = BASE_DIR / "strategy.py"
        if src.exists():
            shutil.copy(src, workspace["strategy_path"])
        else:
            # Write v1 observe-only strategy
            workspace["strategy_path"].write_text(
                '"""\nWeather Trading Strategy v1\\n"""\n\n'
                'def decide(context):\n'
                '    return {"action": "SKIP", "bracket": None, '
                '"confidence": 0.0, "reasoning": "v1: Observing"}\n'
            )

    return workspace


def load_workspace_strategy(workspace: dict):
    """Load the decide function from a workspace's strategy file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"strategy_{workspace['name']}", workspace["strategy_path"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def get_workspace_version(workspace: dict) -> int:
    """Get version number from workspace strategy."""
    import re
    code = workspace["strategy_path"].read_text()
    match = re.search(r'[Ss]trategy\s+v(\d+)', code)
    if match:
        return int(match.group(1))
    versions = list(workspace["versions_dir"].glob("strategy_v*.py"))
    return len(versions) or 1


def run_analysis(workspace: dict, trigger_reason: str) -> dict:
    """Run Qwen/Gemma analysis for a specific workspace."""
    import re

    current_code = workspace["strategy_path"].read_text()
    current_version = get_workspace_version(workspace)

    # Load journal entries from this workspace's journal
    entries = []
    if workspace["journal_path"].exists():
        with open(workspace["journal_path"]) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except (json.JSONDecodeError, ValueError):
                    pass

    trades = [e for e in entries if e.get("type") == "trade"]
    skips = [e for e in entries if e.get("type") == "skip"]
    observations = [e for e in entries if e.get("type") == "observation"]

    resolved = [t for t in trades if t.get("result") is not None]
    wins = sum(1 for t in resolved if t.get("result") == "WIN")
    total_pnl = sum(t.get("pnl", 0) for t in resolved if t.get("pnl"))

    stats = {
        "total_trades": len(trades),
        "resolved": len(resolved),
        "wins": wins,
        "losses": len(resolved) - wins,
        "win_rate": round(wins / len(resolved) * 100, 1) if resolved else 0,
        "total_pnl": round(total_pnl, 2),
    }

    if len(entries) < 3:
        return {
            "rewrote": False, "old_version": current_version,
            "new_version": current_version,
            "analysis": f"Not enough data ({len(entries)} entries)",
            "changes_summary": "No changes", "error": None,
        }

    # Build prompt (reuse from analyst module)
    prompt = analyst._build_prompt(
        current_code, current_version, trades[-20:], skips[-10:],
        observations[-15:], [], stats, trigger_reason
    )

    import requests
    try:
        r = requests.post(
            f"{analyst.OLLAMA_URL}/api/chat",
            json={
                "model": workspace["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 16384},
            },
            timeout=300,
        )
        if r.status_code != 200:
            return {"rewrote": False, "old_version": current_version,
                    "new_version": current_version, "analysis": "",
                    "changes_summary": f"HTTP {r.status_code}", "error": str(r.status_code)}

        response_text = r.json()["message"]["content"]
    except Exception as e:
        return {"rewrote": False, "old_version": current_version,
                "new_version": current_version, "analysis": "",
                "changes_summary": str(e), "error": str(e)}

    code_match = re.search(r'```python\s*\n(.*?)\n\s*```', response_text, re.DOTALL)
    if not code_match:
        code_match = re.search(r'```\s*\n(.*?)\n\s*```', response_text, re.DOTALL)

    if not code_match:
        return {"rewrote": False, "old_version": current_version,
                "new_version": current_version,
                "analysis": response_text[:300],
                "changes_summary": "No code block", "error": None}

    new_code = sandbox.clean_code(code_match.group(1))
    new_version = current_version + 1

    is_valid, error = sandbox.validate_code(new_code)
    if not is_valid:
        return {"rewrote": False, "old_version": current_version,
                "new_version": current_version,
                "analysis": response_text[:300],
                "changes_summary": f"Validation failed: {error}", "error": error}

    # Save version and deploy
    version_path = workspace["versions_dir"] / f"strategy_v{new_version:03d}.py"
    version_path.write_text(new_code, encoding="utf-8")
    workspace["strategy_path"].write_text(new_code, encoding="utf-8")

    analysis_text = response_text[:response_text.find("```")].strip()
    return {
        "rewrote": True, "old_version": current_version,
        "new_version": new_version,
        "analysis": analysis_text[:300],
        "changes_summary": f"v{current_version} -> v{new_version}", "error": None,
    }


def log_to_workspace(workspace: dict, entry: dict):
    """Append a journal entry to a workspace's journal."""
    entry["timestamp"] = datetime.utcnow().isoformat()
    with open(workspace["journal_path"], "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_comparison(model_a: str, model_b: str, cities: list[str],
                   cycles: int = 3, interval: int = 3600):
    """Run both models on the same market data and compare."""

    ws_a = setup_model_workspace(model_a)
    ws_b = setup_model_workspace(model_b)

    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON: {model_a} vs {model_b}")
    print(f"{'='*70}")
    print(f"  Cities: {', '.join(cities)}")
    print(f"  Cycles: {cycles}")
    print(f"  Interval: {interval}s\n")

    for cycle in range(1, cycles + 1):
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"\n{'='*70}")
        print(f"  CYCLE {cycle}/{cycles} — {now.strftime('%H:%M')}")
        print(f"{'='*70}")

        for city in cities:
            forecasts = weather_data.get_all_forecasts(city, tomorrow)
            nws = forecasts["nws_forecast"]
            gfs = forecasts["gfs_mean"]

            if not nws and not gfs:
                continue

            # Get market data
            markets = market_api.find_weather_markets([city])
            brackets = []
            if markets:
                brackets = markets[0]["brackets"]
            elif nws:
                center = int(round(nws))
                center = center - (center % 2)
                brackets = [
                    {"range": f"{center-4}-{center-3}", "low": center-4, "high": center-3, "price": 0.08},
                    {"range": f"{center-2}-{center-1}", "low": center-2, "high": center-1, "price": 0.20},
                    {"range": f"{center}-{center+1}", "low": center, "high": center+1, "price": 0.40},
                    {"range": f"{center+2}-{center+3}", "low": center+2, "high": center+3, "price": 0.22},
                    {"range": f"{center+4}-{center+5}", "low": center+4, "high": center+5, "price": 0.07},
                ]

            context = {
                "city": city,
                "date": tomorrow,
                "nws_forecast": nws or 0.0,
                "gfs_mean": gfs or 0.0,
                "gfs_low": forecasts["gfs_low"] or 0.0,
                "gfs_high": forecasts["gfs_high"] or 0.0,
                "gfs_spread": forecasts["gfs_spread"] or 0.0,
                "brackets": brackets,
                "historical_bias": forecasts["forecast_bias"] or 0.0,
                "recent_trades": [],
                "win_rate": 0,
                "total_pnl": 0,
                "strategy_version": 0,
            }

            print(f"\n  {city} {tomorrow} — NWS: {nws}F | GFS: {gfs}F")

            # Run both models
            results = {}
            for label, ws in [("A", ws_a), ("B", ws_b)]:
                try:
                    decide_fn = load_workspace_strategy(ws)
                    decision = decide_fn(context)
                except Exception as e:
                    decision = {"action": "SKIP", "bracket": None,
                               "confidence": 0, "reasoning": f"Error: {e}"}

                results[label] = decision
                action = decision.get("action", "SKIP")
                bracket = decision.get("bracket")
                conf = decision.get("confidence", 0)
                reasoning = decision.get("reasoning", "")

                # Log to workspace journal
                entry_type = "trade" if action == "BUY" else "skip"
                log_to_workspace(ws, {
                    "type": entry_type, "city": city, "date": tomorrow,
                    "action": action, "bracket_chosen": bracket,
                    "bracket_price": next((b["price"] for b in brackets if b.get("range") == bracket), 0) if bracket else 0,
                    "confidence": conf, "reasoning": reasoning,
                    "nws_forecast": nws, "gfs_mean": gfs,
                })

            # Print side by side
            da = results["A"]
            db = results["B"]
            print(f"    {'Model A (' + model_a + ')':40s} | {'Model B (' + model_b + ')'}")
            print(f"    {'-'*40} | {'-'*40}")

            a_action = da.get('action', 'SKIP')
            b_action = db.get('action', 'SKIP')
            a_str = f"{a_action} {da.get('bracket', '') or ''} (conf: {da.get('confidence', 0):.2f})"
            b_str = f"{b_action} {db.get('bracket', '') or ''} (conf: {db.get('confidence', 0):.2f})"
            print(f"    {a_str:40s} | {b_str}")

            # Highlight differences
            if a_action != b_action:
                print(f"    >>> DISAGREE: A says {a_action}, B says {b_action}")
            elif da.get("bracket") != db.get("bracket"):
                print(f"    >>> Different brackets: A={da.get('bracket')}, B={db.get('bracket')}")

        # Run analysis for both models
        print(f"\n  Running analysis for both models...")

        for label, ws in [("A", ws_a), ("B", ws_b)]:
            print(f"\n  [{label}: {ws['model']}] Analyzing...")
            result = run_analysis(ws, f"Comparison cycle {cycle}")
            if result["rewrote"]:
                print(f"    Strategy rewritten: {result['changes_summary']}")
            else:
                print(f"    No rewrite: {result['changes_summary']}")

        # Print version summary
        va = get_workspace_version(ws_a)
        vb = get_workspace_version(ws_b)
        print(f"\n  Versions: A=v{va} | B=v{vb}")

        if cycle < cycles:
            print(f"\n  Next cycle in {interval}s...")
            time.sleep(interval)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  Model A ({model_a}): v{get_workspace_version(ws_a)}")
    print(f"  Model B ({model_b}): v{get_workspace_version(ws_b)}")
    print(f"\n  Strategy files:")
    print(f"    A: {ws_a['strategy_path']}")
    print(f"    B: {ws_b['strategy_path']}")
    print(f"\n  Journals:")
    print(f"    A: {ws_a['journal_path']}")
    print(f"    B: {ws_b['journal_path']}")
    print(f"\n  Compare strategies:")
    print(f"    type {ws_a['strategy_path'].name}")
    print(f"    type {ws_b['strategy_path'].name}")


def main():
    parser = argparse.ArgumentParser(description="Compare two LLM models for strategy writing")
    parser.add_argument("--model-a", type=str, required=True, help="First model (e.g., qwen3.5:9b)")
    parser.add_argument("--model-b", type=str, required=True, help="Second model (e.g., gemma4:26b)")
    parser.add_argument("--city", type=str, nargs="+",
                       default=["NYC", "Chicago", "LA", "Atlanta", "Miami"],
                       help="Cities to compare on")
    parser.add_argument("--cycles", type=int, default=3, help="Number of comparison cycles")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between cycles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Verify both models are available
    for model in [args.model_a, args.model_b]:
        if not analyst.check_ollama(model):
            print(f"  Model {model} not available. Run: ollama pull {model}")
            return

    run_comparison(args.model_a, args.model_b, args.city, args.cycles, args.interval)


if __name__ == "__main__":
    main()
