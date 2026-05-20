# Amharic Farming Assistant Architecture

Goal: a GPT-like Ethiopian farming assistant that answers fully in Amharic, uses the local agricultural KB first, and uses fresh dynamic data when the KB is not enough.

## LLM routing

Default hosted path:

1. Gemini primary: set `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
2. Groq backup: set `GROQ_API_KEY`; enabled by default with `GEMINI_GROQ_FALLBACK=1`.
3. Ollama fallback: enabled by default when `USE_OLLAMA=1` and `RAG_HOSTED_FALLBACK_OLLAMA=1`.

Useful env:

- `RAG_LLM_BACKEND=gemini` for explicit Gemini primary.
- `RAG_LLM_BACKEND=groq` for explicit Groq primary.
- `GEMINI_MODEL=gemini-2.0-flash` to override Gemini.
- `GROQ_MODEL=llama-3.3-70b-versatile` to override Groq.

## Dynamic demand data

On-demand weather, market, and soil/fertilizer context is cached in:

`data/dynamic/on_demand_context.json`

Behavior:

- Weather questions use Open-Meteo when a location is present.
- Market, price, demand, soil, and fertilizer questions use cached context first.
- If cached context is missing or stale, the assistant searches the web and stores the result.

Useful env:

- `RAG_DEMAND_AUTO=1` enables automatic demand fetch/store.
- `RAG_DEMAND_WEATHER_TTL_DAYS=1`
- `RAG_DEMAND_MARKET_TTL_DAYS=7`
- `RAG_DEMAND_SOIL_TTL_DAYS=7`
- `RAG_WEB_MODE=auto` lets web search activate for fresh-information questions.

## Language and answer policy

The system prompt forces Amharic answers even when the user asks in another language. It should:

- Give practical recommendations.
- Label predictions as estimates when location, soil, rain, variety, or timing is uncertain.
- Avoid inventing market prices, pesticide dosage, or precise soil results when no source is available.
- Ask users to consult local extension agents for pesticide, animal-health emergency, legal, or financial decisions.
