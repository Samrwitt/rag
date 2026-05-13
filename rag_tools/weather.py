"""Weather via Open-Meteo (no API key)."""

from __future__ import annotations

import os
from typing import Any

import httpx


def _geocode(name: str) -> tuple[float, float, str] | None:
    timeout = float(os.environ.get("RAG_TOOL_HTTP_TIMEOUT", "20"))
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name.strip()[:200], "count": 1, "language": "en", "format": "json"}
    with httpx.Client(timeout=timeout) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    if not results:
        return None
    row = results[0]
    lat, lon = float(row["latitude"]), float(row["longitude"])
    label = row.get("name") or name
    admin = row.get("admin1") or ""
    country = row.get("country_code") or ""
    pretty = ", ".join(x for x in (label, admin, country) if x)
    return lat, lon, pretty


def fetch_weather_summary(location: str) -> str:
    loc = (location or "").strip()
    if not loc:
        return ""
    try:
        geo = _geocode(loc)
        if not geo:
            return f"ቦታ '{loc}' በአየር ካርታ ላይ አልተገኘም።"
        lat, lon, pretty = geo
        timeout = float(os.environ.get("RAG_TOOL_HTTP_TIMEOUT", "20"))
        furl = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "timezone": "auto",
        }
        with httpx.Client(timeout=timeout) as client:
            r = client.get(furl, params=params)
            r.raise_for_status()
            data = r.json()
        cur = data.get("current") or {}
        t = cur.get("temperature_2m")
        rh = cur.get("relative_humidity_2m")
        pr = cur.get("precipitation")
        lines = [
            f"ቦታ፦ {pretty}",
            f"አዘራራይ መጠን (°C)፦ {t}" if t is not None else "",
            f"እርጥበት (%)፦ {rh}" if rh is not None else "",
            f"ዝናብ (ሚሜ)፦ {pr}" if pr is not None else "",
        ]
        return "\n".join(x for x in lines if x).strip()
    except Exception as e:
        return f"የአየር ሁኔታ አገልግሎት ላይ ስህተት፦ {e}"
