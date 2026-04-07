"""
Strategy Code Validator — Safely validates Qwen-written strategy code.

Before deploying any strategy Qwen writes, we:
1. Check syntax (compile)
2. Verify decide() function exists
3. Block dangerous imports
4. Test with sample data
5. Verify output format
"""

import ast
import importlib
import importlib.util
import logging
import sys
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

STRATEGY_PATH = Path(__file__).parent / "strategy.py"
VERSIONS_DIR = Path(__file__).parent / "strategy_versions"

# Imports that are NOT allowed in strategy code
BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "requests", "urllib", "http", "socket", "ftplib",
    "smtplib", "email", "pickle", "shelve",
    "ctypes", "multiprocessing", "threading",
    "__import__", "eval", "exec", "compile",
    "open",  # no file I/O
}


def validate_code(code: str) -> tuple[bool, str]:
    """
    Validate strategy code for safety and correctness.

    Returns (is_valid, error_message).
    """
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # 2. Check for dangerous imports/calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in BLOCKED_IMPORTS:
                    return False, f"Blocked import: {module}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module in BLOCKED_IMPORTS:
                    return False, f"Blocked import: {module}"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "__import__", "open"):
                    return False, f"Blocked function call: {node.func.id}"

    # 3. Check that decide() function exists
    func_names = [node.name for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)]
    if "decide" not in func_names:
        return False, "Missing required function: decide()"

    # 4. Test with sample data
    sample_context = {
        "city": "NYC",
        "date": "2026-04-08",
        "nws_forecast": 54.0,
        "gfs_mean": 54.1,
        "gfs_low": 52.0,
        "gfs_high": 56.0,
        "gfs_spread": 4.0,
        "brackets": [
            {"range": "50-51", "low": 50, "high": 51, "price": 0.05},
            {"range": "52-53", "low": 52, "high": 53, "price": 0.15},
            {"range": "54-55", "low": 54, "high": 55, "price": 0.42},
            {"range": "56-57", "low": 56, "high": 57, "price": 0.25},
            {"range": "58-59", "low": 58, "high": 59, "price": 0.08},
        ],
        "historical_bias": -1.5,
        "recent_trades": [],
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "strategy_version": 1,
    }

    try:
        # Execute in isolated namespace
        namespace = {"__builtins__": __builtins__}
        exec(compile(tree, "<strategy>", "exec"), namespace)

        decide_fn = namespace.get("decide")
        if not callable(decide_fn):
            return False, "decide is not callable"

        result = decide_fn(sample_context)

        # 5. Validate output format
        if not isinstance(result, dict):
            return False, f"decide() must return dict, got {type(result).__name__}"

        if "action" not in result:
            return False, "decide() result missing 'action' key"

        action = result["action"]
        if action not in ("BUY", "SKIP"):
            return False, f"Invalid action: {action}. Must be 'BUY' or 'SKIP'"

        if action == "BUY" and "bracket" not in result:
            return False, "BUY action must include 'bracket' key"

        if "reasoning" not in result:
            return False, "decide() result missing 'reasoning' key"

    except Exception as e:
        return False, f"Runtime error: {traceback.format_exc()}"

    return True, "OK"


def save_strategy_version(code: str, version: int):
    """Save a strategy version to the versions directory."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = VERSIONS_DIR / f"strategy_v{version:03d}.py"
    path.write_text(code)
    logger.info(f"Saved strategy version {version} to {path}")


def deploy_strategy(code: str, version: int) -> bool:
    """
    Validate and deploy a new strategy.

    Returns True if deployed successfully.
    """
    is_valid, error = validate_code(code)

    if not is_valid:
        logger.error(f"Strategy validation failed: {error}")
        return False

    # Save version
    save_strategy_version(code, version)

    # Deploy to strategy.py
    STRATEGY_PATH.write_text(code)
    logger.info(f"Strategy v{version} deployed to {STRATEGY_PATH}")

    return True


def load_strategy():
    """
    Load and return the current strategy module's decide function.
    Uses importlib for hot-reloading.
    """
    spec = importlib.util.spec_from_file_location("strategy", STRATEGY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def get_current_strategy_code() -> str:
    """Read the current strategy source code."""
    if STRATEGY_PATH.exists():
        return STRATEGY_PATH.read_text()
    return ""


def get_current_version() -> int:
    """Get the current strategy version number from the code."""
    code = get_current_strategy_code()
    # Look for "Strategy v{N}" in docstring or comments
    import re
    match = re.search(r'[Ss]trategy\s+v(\d+)', code)
    if match:
        return int(match.group(1))

    # Count version files
    if VERSIONS_DIR.exists():
        versions = list(VERSIONS_DIR.glob("strategy_v*.py"))
        return len(versions)

    return 1


def get_best_version() -> int | None:
    """Get the best-performing strategy version (tracked externally)."""
    tracker = VERSIONS_DIR / "best_version.txt"
    if tracker.exists():
        try:
            return int(tracker.read_text().strip())
        except ValueError:
            pass
    return None


def set_best_version(version: int):
    """Record the best-performing strategy version."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    (VERSIONS_DIR / "best_version.txt").write_text(str(version))


def revert_to_version(version: int) -> bool:
    """Revert strategy.py to a specific version."""
    path = VERSIONS_DIR / f"strategy_v{version:03d}.py"
    if not path.exists():
        logger.error(f"Version {version} not found at {path}")
        return False

    code = path.read_text()
    STRATEGY_PATH.write_text(code)
    logger.info(f"Reverted strategy to v{version}")
    return True
