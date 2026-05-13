"""Tool metadata (for docs / future native tool-calling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    invoke: Callable[[dict[str, Any]], str]


def ollama_style_schemas() -> list[dict[str, Any]]:
    """JSON-schema style tool definitions (Ollama/OpenAI compatible shape)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the public web for fresh or external facts not in the farm KB.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "max_results": {
                            "type": "integer",
                            "description": "1–8 snippets",
                            "default": 4,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "weather_forecast",
                "description": "Current weather via Open-Meteo (free, no API key).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or region name (e.g. Addis Ababa, Hawassa).",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
    ]
