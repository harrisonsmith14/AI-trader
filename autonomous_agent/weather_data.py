"""
Weather Data Fetcher — The AI's eyes into weather patterns.

Fetches forecasts from NWS and Open-Meteo, plus historical actuals
for computing forecast bias. This data is what the strategy learns from.

Data sources:
- NWS API (api.weather.gov): Official US forecasts, free, no key needed
- Open-Meteo (open-meteo.com): GFS/ECMWF ensemble forecasts, free
- Weather Underground: Historical actuals (resolution source for Polymarket)
"""

import requests
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# City -> NWS grid point mapping (latitude, longitude)
# These are the airport stations Polymarket uses for resolution
CITY_COORDS = {
    "NYC": {"lat": 40.7793, "lon": -73.8740, "station": "KLGA", "name": "New York (LaGuardia)"},
    "Chicago": {"lat": 41.9742, "lon": -87.9073, "station": "KORD", "name": "Chicago (O'Hare)"},
    "LA": {"lat": 33.9425, "lon": -118.4081, "station": "KLAX", "name": "Los Angeles (LAX)"},
    "SF": {"lat": 37.6213, "lon": -122.3790, "station": "KSFO", "name": "San Francisco (SFO)"},
    "Atlanta": {"lat": 33.6407, "lon": -84.4277, "station": "KATL", "name": "Atlanta (Hartsfield)"},
    "Miami": {"lat": 25.7959, "lon": -80.2870, "station": "KMIA", "name": "Miami (MIA)"},
    "Denver": {"lat": 39.8561, "lon": -104.6737, "station": "KDEN", "name": "Denver (DEN)"},
}

HISTORY_PATH = Path(__file__).parent.parent / "logs" / "weather_history.jsonl"


def get_nws_forecast(city: str, date: str = None) -> dict | None:
    """
    Get temperature forecast from NWS API.

    Args:
        city: City key (e.g., "NYC")
        date: Target date as "YYYY-MM-DD" (default: tomorrow)

    Returns:
        {"high": 54, "low": 38, "conditions": "Partly Cloudy", "source": "NWS"}
        or None if unavailable.
    """
    if city not in CITY_COORDS:
        logger.warning(f"Unknown city: {city}")
        return None

    coords = CITY_COORDS[city]

    try:
        # Step 1: Get the NWS grid point for this location
        points_url = f"https://api.weather.gov/points/{coords['lat']},{coords['lon']}"
        headers = {"User-Agent": "AI-Trader-Weather-Bot (contact: github.com/harrisonsmith14)"}
        r = requests.get(points_url, headers=headers, timeout=10)
        r.raise_for_status()
        points = r.json()

        forecast_url = points["properties"]["forecast"]

        # Step 2: Get the forecast
        r = requests.get(forecast_url, headers=headers, timeout=10)
        r.raise_for_status()
        periods = r.json()["properties"]["periods"]

        # Parse target date
        if date is None:
            target = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            target = date

        # Find the daytime period for our target date
        high = None
        low = None
        conditions = ""

        for period in periods:
            period_date = period["startTime"][:10]
            if period_date == target:
                temp = period["temperature"]
                if period["isDaytime"]:
                    high = temp
                    conditions = period.get("shortForecast", "")
                else:
                    low = temp

        if high is not None:
            return {
                "high": high,
                "low": low,
                "conditions": conditions,
                "source": "NWS",
            }

    except requests.RequestException as e:
        logger.warning(f"NWS API failed for {city}: {e}")
    except (KeyError, IndexError) as e:
        logger.warning(f"NWS response parsing failed for {city}: {e}")

    return None


def get_gfs_ensemble(city: str, date: str = None) -> dict | None:
    """
    Get GFS ensemble forecast from Open-Meteo API.

    Returns temperature range with mean, low, high, and spread.
    This gives the AI a sense of forecast uncertainty.

    Returns:
        {"mean": 54.1, "low": 52, "high": 56, "spread": 4.0, "source": "GFS"}
    """
    if city not in CITY_COORDS:
        return None

    coords = CITY_COORDS[city]

    if date is None:
        target = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        target = date

    try:
        # Open-Meteo GFS ensemble forecast
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "start_date": target,
            "end_date": target,
            "models": "gfs_seamless",
        }

        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_max", [])

        if temps:
            temp = temps[0]
            # Open-Meteo gives a single value for the deterministic forecast
            # We'll estimate spread based on typical GFS uncertainty (~3-5°F)
            return {
                "mean": round(temp, 1),
                "low": round(temp - 2, 1),
                "high": round(temp + 2, 1),
                "spread": 4.0,
                "source": "GFS/Open-Meteo",
            }

    except requests.RequestException as e:
        logger.warning(f"Open-Meteo API failed for {city}: {e}")
    except (KeyError, IndexError) as e:
        logger.warning(f"Open-Meteo response parsing failed for {city}: {e}")

    return None


def get_historical_actuals(city: str, days: int = 14) -> list[dict]:
    """
    Get historical actual temperatures from Open-Meteo (archive API).

    Returns list of {date, actual_high} for recent days.
    """
    if city not in CITY_COORDS:
        return []

    coords = CITY_COORDS[city]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
            "start_date": start_date,
            "end_date": end_date,
        }

        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])

        results = []
        for d, t in zip(dates, temps):
            if t is not None:
                results.append({"date": d, "actual_high": round(t, 1)})

        return results

    except requests.RequestException as e:
        logger.warning(f"Open-Meteo archive API failed for {city}: {e}")

    return []


def compute_forecast_bias(city: str, days: int = 14) -> float | None:
    """
    Compute average forecast bias: how much NWS/GFS over- or under-predicts.

    Returns: average (forecast - actual) over recent days.
    Positive = forecast runs hot, negative = forecast runs cold.
    """
    # Load from our journal if we have forecast vs actual data
    if not HISTORY_PATH.exists():
        return None

    entries = []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    with open(HISTORY_PATH) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if (entry.get("city") == city and
                    entry.get("timestamp", "") > cutoff and
                    entry.get("nws_forecast") is not None and
                    entry.get("actual_temp") is not None):
                    entries.append(entry)
            except (json.JSONDecodeError, ValueError):
                pass

    if len(entries) < 3:
        return None

    biases = [e["nws_forecast"] - e["actual_temp"] for e in entries]
    return round(sum(biases) / len(biases), 2)


def log_weather_history(entry: dict):
    """Save a forecast vs actual record for bias computation."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_all_forecasts(city: str, date: str = None) -> dict:
    """
    Get all available forecast data for a city/date.
    Convenience function that combines NWS + GFS + bias.
    """
    nws = get_nws_forecast(city, date)
    gfs = get_gfs_ensemble(city, date)
    bias = compute_forecast_bias(city)

    return {
        "city": city,
        "date": date or (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "nws": nws,
        "gfs": gfs,
        "forecast_bias": bias,
        "nws_forecast": nws["high"] if nws else None,
        "gfs_mean": gfs["mean"] if gfs else None,
        "gfs_low": gfs["low"] if gfs else None,
        "gfs_high": gfs["high"] if gfs else None,
        "gfs_spread": gfs["spread"] if gfs else None,
    }


# --- CLI for testing ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Weather Data Fetcher — Testing\n")

    for city in ["NYC", "Chicago", "LA"]:
        print(f"  === {CITY_COORDS[city]['name']} ===")
        forecasts = get_all_forecasts(city)

        if forecasts["nws"]:
            print(f"  NWS Forecast: {forecasts['nws']['high']}°F | {forecasts['nws']['conditions']}")
        else:
            print(f"  NWS: unavailable")

        if forecasts["gfs"]:
            print(f"  GFS Ensemble: {forecasts['gfs']['mean']}°F "
                  f"(range: {forecasts['gfs']['low']}-{forecasts['gfs']['high']}°F)")
        else:
            print(f"  GFS: unavailable")

        if forecasts["forecast_bias"] is not None:
            print(f"  Forecast bias: {forecasts['forecast_bias']:+.1f}°F")

        # Historical
        hist = get_historical_actuals(city, days=7)
        if hist:
            print(f"  Last 7 days actuals: {[h['actual_high'] for h in hist]}")

        print()
