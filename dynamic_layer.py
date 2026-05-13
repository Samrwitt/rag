import argparse
import json
import pathlib
from datetime import datetime, UTC

import requests
import urllib3
from bs4 import BeautifulSoup


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================
# PATHS
# ============================================================

# Resolve relative to this file so `python dynamic_layer.py` works from any CWD.
_ROOT = pathlib.Path(__file__).resolve().parent
BASE_DIR = _ROOT / "data"
DYNAMIC_DIR = BASE_DIR / "dynamic"
CHUNKS_DIR = BASE_DIR / "chunks"
LOG_DIR = BASE_DIR / "logs"

DYNAMIC_JSONL = CHUNKS_DIR / "dynamic_context_chunks.jsonl"

WEATHER_JSON = DYNAMIC_DIR / "weather_forecasts.json"
ON_DEMAND_WEATHER_JSON = DYNAMIC_DIR / "weather_on_demand.json"

SOIL_JSON = DYNAMIC_DIR / "soil_datasets.json"
MARKET_JSON = DYNAMIC_DIR / "market_sources.json"
NMIS_JSON = DYNAMIC_DIR / "nmis_dataset_metadata.json"
CLIMATE_AGROADVISORY_JSON = DYNAMIC_DIR / "climate_agroadvisory_sources.json"

REPORT_JSON = LOG_DIR / "dynamic_layer_report.json"


# You previously had SSL problems with some Ethiopian websites.
# For local research collection, False is practical.
# For production, use True and proper CA certificates.
VERIFY_SSL = False


# ============================================================
# SCHEDULED WEATHER LOCATIONS
# ============================================================

LOCATIONS = [
    {
        "name": "Addis Ababa",
        "region": "Addis Ababa",
        "latitude": 9.03,
        "longitude": 38.74,
    },
    {
        "name": "Adama",
        "region": "Oromia",
        "latitude": 8.54,
        "longitude": 39.27,
    },
    {
        "name": "Bahir Dar",
        "region": "Amhara",
        "latitude": 11.60,
        "longitude": 37.38,
    },
    {
        "name": "Hawassa",
        "region": "Sidama",
        "latitude": 7.05,
        "longitude": 38.48,
    },
    {
        "name": "Mekelle",
        "region": "Tigray",
        "latitude": 13.49,
        "longitude": 39.47,
    },
    {
        "name": "Jimma",
        "region": "Oromia",
        "latitude": 7.67,
        "longitude": 36.83,
    },
    {
        "name": "Dire Dawa",
        "region": "Dire Dawa",
        "latitude": 9.60,
        "longitude": 41.86,
    },
    {
        "name": "Gondar",
        "region": "Amhara",
        "latitude": 12.60,
        "longitude": 37.47,
    },
    {
        "name": "Assosa",
        "region": "Benishangul-Gumuz",
        "latitude": 10.07,
        "longitude": 34.53,
    },
    {
        "name": "Semera",
        "region": "Afar",
        "latitude": 11.79,
        "longitude": 41.01,
    },
]


# ============================================================
# OFFICIAL SOURCE CONTEXTS
# ============================================================

OFFICIAL_WEATHER_CONTEXT = {
    "name": "Ethiopian Meteorological Institute",
    "source_org": "Ethiopian Meteorological Institute",
    "url": "https://www.ethiomet.gov.et/",
    "update_frequency": "daily_to_seasonal",
    "description": (
        "Official Ethiopian weather and climate authority providing forecasting, "
        "agrometeorology, hydrology, climate, and warning services."
    ),
}


MARKET_CONTEXT_SOURCES = [
    {
        "name": "ATI National Market Information System",
        "source_org": "Agricultural Transformation Institute",
        "url": "https://ati.gov.et/nmis/",
        "update_frequency": "weekly",
        "description": (
            "ATI NMIS collects, validates, analyzes, and disseminates weekly "
            "market data for agricultural commodities across Ethiopian marketplaces."
        ),
    },
    {
        "name": "Ethiopian Commodity Exchange",
        "source_org": "Ethiopian Commodity Exchange",
        "url": "https://www.ecx.com.et/",
        "update_frequency": "daily_or_market_day",
        "description": (
            "ECX provides commodity exchange market information for selected commodities."
        ),
    },
]


SOIL_SEARCH_TERMS = [
    "soil",
    "soil nutrients",
    "soil type",
    "fertilizer",
    "EthioSIS",
    "CIAT",
    "NextGen Fertilizer",
]


CLIMATE_AGROADVISORY_SOURCES = [
    {
        "name": "NextGen Agroadvisory",
        "source_org": "Alliance Bioversity International / CIAT",
        "url": "https://alliancebioversityciat.org/tools-innovations/nextgen-agroadvisory",
        "kb": "climate_agroadvisory",
        "update_frequency": "monthly_or_seasonal",
        "source_type": "season_smart_agroadvisory_context",
        "description": (
            "Location-specific, tailored, season-smart agroadvisory decision-support "
            "tool for planting timelines, fertilizer types and amounts, integrated soil "
            "fertility management, climate information services, and climate-smart agriculture."
        ),
    },
    {
        "name": "Climate-Smart Agriculture in Ethiopia",
        "source_org": "Alliance Bioversity International / CIAT",
        "url": "https://alliancebioversityciat.org/publications-data/climate-smart-agriculture-ethiopia",
        "kb": "climate_agroadvisory",
        "update_frequency": "manual_or_periodic",
        "source_type": "climate_smart_agriculture_profile",
        "description": (
            "Ethiopia-specific climate-smart agriculture profile useful for climate-risk, "
            "adaptation, rainfall variability, drought risk, and seasonal planning context."
        ),
    },
    {
        "name": "2024 Summer Seasonal Forecast and Pastoral Climate Advisory for Ethiopia",
        "source_org": "EIAR / Alliance Bioversity International / CIAT / Ethiopian Meteorology Institute",
        "url": "https://alliancebioversityciat.org/stories/2024-summer-june-august-seasonal-forecast-pastoral-climate-advisory-pca-ethiopia",
        "kb": "climate_agroadvisory",
        "update_frequency": "seasonal",
        "source_type": "seasonal_forecast_advisory",
        "description": (
            "Seasonal forecast and pastoral climate advisory for Ethiopia, including "
            "pasture and water availability, heat stress, anticipatory actions, and "
            "cropping planning guidance."
        ),
    },
    {
        "name": "Alliance Bioversity and CIAT Ethiopia Projects",
        "source_org": "Alliance Bioversity International / CIAT",
        "url": "https://alliancebioversityciat.org/projects-flagship-initiatives-ethiopia",
        "kb": "climate_agroadvisory",
        "update_frequency": "monthly_or_periodic",
        "source_type": "project_context",
        "description": (
            "Alliance/CIAT Ethiopia project context related to climate, agronomy, "
            "agricultural risk management, and decision support."
        ),
    },
    {
        "name": "Alliance Bioversity and CIAT datasets on Ethiopian National Agri Data Hub",
        "source_org": "Ethiopian National Agri Data Hub / Alliance Bioversity & CIAT",
        "url": "https://data.moa.gov.et/dataset/?organization=alliance-bioversity-ciat",
        "kb": "climate_agroadvisory",
        "update_frequency": "weekly_or_monthly",
        "source_type": "dataset_catalog_context",
        "description": (
            "MoA Agri Data Hub catalog page listing Alliance Bioversity & CIAT datasets, "
            "including CIAT dataset and NextGen Fertilizer layer."
        ),
    },
]


# ============================================================
# BASIC HELPERS
# ============================================================

def now_iso():
    return datetime.now(UTC).isoformat()


def ensure_dirs():
    for d in [DYNAMIC_DIR, CHUNKS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def safe_get_json(url, params=None, timeout=60):
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        verify=VERIFY_SSL,
        headers={"User-Agent": "ethiopia-farmer-advisory-dynamic-layer/1.0"},
    )
    response.raise_for_status()
    return response.json()


def fetch_page_text(url):
    response = requests.get(
        url,
        timeout=90,
        verify=VERIFY_SSL,
        headers={"User-Agent": "ethiopia-farmer-advisory-dynamic-layer/1.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:12000]


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# WEATHER: SCHEDULED + ON-DEMAND
# ============================================================

def geocode_ethiopian_city(city_name):
    """
    On-demand geocoding for Ethiopian towns/cities.
    Examples: Bishoftu, Nekemte, Shashamane, Ambo, Robe, Dilla.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city_name,
        "count": 5,
        "language": "en",
        "format": "json",
        "countryCode": "ET",
    }

    data = safe_get_json(url, params=params, timeout=60)
    results = data.get("results", [])

    if not results:
        return None

    best = results[0]

    return {
        "name": best.get("name") or city_name,
        "region": best.get("admin1") or best.get("admin2") or "Unknown",
        "latitude": best.get("latitude"),
        "longitude": best.get("longitude"),
        "country": best.get("country"),
        "timezone": best.get("timezone") or "Africa/Addis_Ababa",
        "source": "Open-Meteo Geocoding API",
        "requested_name": city_name,
    }


def fetch_open_meteo_forecast(location):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": "Africa/Addis_Ababa",
        "forecast_days": 7,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
        ]),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "rain_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
        ]),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "rain",
            "soil_moisture_0_to_1cm",
            "soil_moisture_1_to_3cm",
            "soil_temperature_0cm",
        ]),
    }

    data = safe_get_json(url, params=params, timeout=90)

    return {
        "location": location,
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "official_context_source": OFFICIAL_WEATHER_CONTEXT,
        "retrieved_at": now_iso(),
        "forecast": data,
    }


def fetch_weather_for_city_name(city_name):
    location = geocode_ethiopian_city(city_name)

    if not location:
        raise ValueError(f"Could not find Ethiopian city/location: {city_name}")

    if not location.get("latitude") or not location.get("longitude"):
        raise ValueError(f"Location found but missing coordinates: {location}")

    return fetch_open_meteo_forecast(location)


def summarize_weather_am(record):
    loc = record["location"]
    forecast = record["forecast"]
    daily = forecast.get("daily", {})

    dates = daily.get("time", [])
    rain = daily.get("precipitation_sum", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])

    lines = []
    lines.append(
        f"{loc['name']} ({loc['region']}) የ7 ቀን የአየር ትንበያ። "
        f"መረጃው ከOpen-Meteo በ{record['retrieved_at']} ተወስዷል።"
    )

    for i, date in enumerate(dates[:7]):
        r = rain[i] if i < len(rain) else None
        p = rain_prob[i] if i < len(rain_prob) else None
        hi = tmax[i] if i < len(tmax) else None
        lo = tmin[i] if i < len(tmin) else None

        lines.append(
            f"- {date}: ከፍተኛ ሙቀት {hi}°C፣ ዝቅተኛ ሙቀት {lo}°C፣ "
            f"የዝናብ መጠን {r} mm፣ ከፍተኛ የዝናብ ዕድል {p}%።"
        )

    lines.append(
        "ማሳሰቢያ፡ ለኦፊሴላዊ የኢትዮጵያ ትንበያ "
        "የኢትዮጵያ ሚቲዎሮሎጂ ኢንስቲትዩትን ይመልከቱ።"
    )

    return "\n".join(lines)


def weather_record_to_chunk(record, chunk_id):
    loc = record["location"]
    text_am = summarize_weather_am(record)

    return {
        "id": chunk_id,
        "kb": "weather",
        "data_layer": "dynamic",
        "source_org": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
        "official_context_source": OFFICIAL_WEATHER_CONTEXT,
        "location": loc["name"],
        "requested_location": loc.get("requested_name"),
        "region": loc["region"],
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "updated_at": record["retrieved_at"],
        "validity": "7_day_forecast",
        "update_frequency": "daily_or_6_hourly",
        "language_segment": "am",
        "text": text_am,
        "text_am": text_am,
        "metadata": {
            "raw_weather_file": str(WEATHER_JSON),
            "source_type": "weather_forecast_api",
            "geocoding_source": loc.get("source"),
        },
    }


def build_weather_chunks(weather_records):
    chunks = []

    for i, record in enumerate(weather_records, start=1):
        chunks.append(weather_record_to_chunk(record, f"dynamic_weather_{i:03d}"))

    return chunks


def append_or_update_on_demand_weather(record):
    existing = []

    if ON_DEMAND_WEATHER_JSON.exists():
        try:
            data = json.loads(ON_DEMAND_WEATHER_JSON.read_text(encoding="utf-8"))
            existing = data.get("records", [])
        except Exception:
            existing = []

    loc = record["location"]
    key = f"{loc.get('name')}|{loc.get('region')}".lower()

    updated = []
    replaced = False

    for item in existing:
        item_loc = item.get("location", {})
        item_key = f"{item_loc.get('name')}|{item_loc.get('region')}".lower()

        if item_key == key:
            updated.append(record)
            replaced = True
        else:
            updated.append(item)

    if not replaced:
        updated.append(record)

    write_json(ON_DEMAND_WEATHER_JSON, {
        "records": updated,
        "updated_at": now_iso(),
    })


# ============================================================
# SOIL / FERTILIZER DATASET METADATA
# ============================================================

def fetch_soil_datasets_from_agrihub():
    """
    Searches Ethiopian National Agri Data Hub CKAN API for soil/fertilizer datasets.
    This creates reliable metadata chunks first. Later we can download and parse
    individual dataset resources.
    """
    endpoint = "https://data.moa.gov.et/api/3/action/package_search"
    results = []

    for term in SOIL_SEARCH_TERMS:
        try:
            data = safe_get_json(
                endpoint,
                params={"q": term, "rows": 10},
                timeout=90,
            )
            packages = data.get("result", {}).get("results", [])

            for pkg in packages:
                results.append({
                    "search_term": term,
                    "package": pkg,
                    "retrieved_at": now_iso(),
                })

        except Exception as e:
            results.append({
                "search_term": term,
                "error": str(e),
                "retrieved_at": now_iso(),
            })

    seen = set()
    unique = []

    for item in results:
        pkg = item.get("package")

        if not pkg:
            unique.append(item)
            continue

        key = pkg.get("id") or pkg.get("name") or pkg.get("title")

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def build_soil_chunks(soil_records):
    chunks = []

    for i, item in enumerate(soil_records, start=1):
        pkg = item.get("package")

        if not pkg:
            continue

        title = pkg.get("title") or pkg.get("name") or "Untitled soil dataset"
        notes = pkg.get("notes") or ""
        name = pkg.get("name") or ""
        url = f"https://data.moa.gov.et/dataset/{name}" if name else "https://data.moa.gov.et/"

        org = pkg.get("organization", {}) or {}
        org_title = org.get("title") or "Ethiopian National Agri Data Hub"

        resources = pkg.get("resources", []) or []
        resource_lines = []

        for res in resources[:8]:
            resource_lines.append(
                f"- {res.get('name') or res.get('description') or 'resource'} "
                f"({res.get('format') or 'unknown format'}): {res.get('url') or ''}"
            )

        text = (
            f"Dataset: {title}\n"
            f"Source: {org_title}\n"
            f"URL: {url}\n"
            f"Search term: {item.get('search_term')}\n\n"
            f"Description:\n{notes}\n\n"
            f"Resources:\n" + "\n".join(resource_lines)
        ).strip()

        text_am = (
            f"የአፈር/ማዳበሪያ መረጃ ምንጭ፡ {title}\n"
            f"ምንጭ፡ {org_title}\n"
            f"URL፡ {url}\n"
            f"ይህ መረጃ ለአፈር አይነት፣ የአፈር ንጥረ-ነገር፣ "
            f"የማዳበሪያ ምክር ወይም የኢትዮጵያ አግሮኖሚ ዳታ መነሻ ሊጠቅም ይችላል።\n\n"
            f"{notes}"
        ).strip()

        chunks.append({
            "id": f"dynamic_soil_dataset_{i:03d}",
            "kb": "soil",
            "data_layer": "dynamic_or_periodic",
            "source_org": org_title,
            "source_url": url,
            "updated_at": item.get("retrieved_at"),
            "validity": "dataset_metadata_periodic",
            "update_frequency": "weekly_or_monthly",
            "language_segment": "mixed",
            "text": text,
            "text_am": text_am,
            "metadata": {
                "search_term": item.get("search_term"),
                "raw_soil_file": str(SOIL_JSON),
                "source_type": "ckan_dataset_metadata",
            },
        })

    return chunks


# ============================================================
# MARKET + NMIS
# ============================================================

def build_market_context_chunks():
    chunks = []

    for i, src in enumerate(MARKET_CONTEXT_SOURCES, start=1):
        text_am = (
            f"{src['name']} የገበያ መረጃ ምንጭ ነው። "
            f"ምንጭ፡ {src['source_org']}። "
            f"URL፡ {src['url']}። "
            f"የመረጃ አዘምን፡ {src['update_frequency']}። "
            f"{src['description']}"
        )

        chunks.append({
            "id": f"dynamic_market_context_{i:03d}",
            "kb": "market",
            "data_layer": "dynamic_context",
            "source_org": src["source_org"],
            "source_url": src["url"],
            "updated_at": now_iso(),
            "validity": "source_context",
            "update_frequency": src["update_frequency"],
            "language_segment": "am",
            "text": text_am,
            "text_am": text_am,
            "metadata": {
                "source_type": "market_source_context",
            },
        })

    return chunks


def fetch_nmis_dataset_metadata():
    """
    Fetch ATI Data Portal metadata for the NMIS dataset.

    This does not guarantee access to raw weekly price rows.
    It stores public dataset metadata and API/resource links.
    """
    dataset_url = "https://data.ata.gov.et/api/3/action/package_show"
    params = {
        "id": "national-market-information-system"
    }

    try:
        data = safe_get_json(dataset_url, params=params, timeout=90)

        return {
            "status": "ok",
            "retrieved_at": now_iso(),
            "source": "ATI Data Portal CKAN API",
            "dataset_url": "https://data.ata.gov.et/sv/dataset/national-market-information-system",
            "data": data.get("result", {}),
        }

    except Exception as e:
        return {
            "status": "failed",
            "retrieved_at": now_iso(),
            "source": "ATI Data Portal CKAN API",
            "dataset_url": "https://data.ata.gov.et/sv/dataset/national-market-information-system",
            "error": str(e),
        }


def build_nmis_market_chunks(nmis_record):
    chunks = []

    dataset_url = nmis_record.get(
        "dataset_url",
        "https://data.ata.gov.et/sv/dataset/national-market-information-system"
    )

    if nmis_record.get("status") != "ok":
        text_am = (
            "ATI National Market Information System (NMIS) የገበያ መረጃ ምንጭ ነው። "
            "ነገር ግን በዚህ ዝመና ወቅት የATI Data Portal metadata/API መዳረሻ አልተሳካም። "
            f"ስህተት፡ {nmis_record.get('error')}. "
            "RAG ሲመልስ ይህንን ምንጭ እንደ የተረጋገጠ የገበያ ምንጭ ብቻ ይጠቀም፣ "
            "ዋጋ ግን ካልተገኘ አይፈጥር።"
        )

        chunks.append({
            "id": "dynamic_market_nmis_metadata_error",
            "kb": "market",
            "data_layer": "dynamic_context",
            "source_org": "Agricultural Transformation Institute",
            "source_url": dataset_url,
            "updated_at": nmis_record.get("retrieved_at"),
            "validity": "source_context",
            "update_frequency": "weekly",
            "language_segment": "am",
            "text": text_am,
            "text_am": text_am,
            "metadata": {
                "source_type": "nmis_dataset_metadata",
                "fetch_status": "failed",
                "fetch_error": nmis_record.get("error"),
            },
        })

        return chunks

    dataset = nmis_record.get("data", {})

    title = dataset.get("title") or "National Market Information System"
    notes = dataset.get("notes") or ""
    metadata_created = dataset.get("metadata_created")
    metadata_modified = dataset.get("metadata_modified")
    resources = dataset.get("resources", []) or []

    resource_lines = []
    api_resources = []

    for res in resources:
        name = res.get("name") or res.get("description") or "resource"
        fmt = res.get("format") or "unknown"
        url = res.get("url") or ""

        resource_lines.append(f"- {name} ({fmt}): {url}")

        joined = f"{name} {fmt} {url}".lower()

        if "api" in joined or "nmis" in joined:
            api_resources.append({
                "name": name,
                "format": fmt,
                "url": url,
                "description": res.get("description"),
            })

    text = (
        f"Dataset: {title}\n"
        f"Source: Agricultural Transformation Institute Data Portal\n"
        f"URL: {dataset_url}\n"
        f"Metadata created: {metadata_created}\n"
        f"Metadata modified: {metadata_modified}\n\n"
        f"Description:\n{notes}\n\n"
        f"Resources:\n" + "\n".join(resource_lines)
    ).strip()

    text_am = (
        f"ATI National Market Information System (NMIS) የተረጋገጠ የግብርና ገበያ መረጃ ምንጭ ነው።\n"
        f"Dataset፡ {title}\n"
        f"URL፡ {dataset_url}\n"
        f"Metadata የተዘመነበት፡ {metadata_modified}\n\n"
        "ይህ ምንጭ ለጤፍ፣ በቆሎ፣ ስንዴ፣ ሰሊጥ እና ሐሪኮት ባቄላ "
        "የገበያ መረጃ ሊያገለግል ይችላል። "
        "እንደ RAG ህግ፣ ትክክለኛ የዋጋ መዝገብ ካልተገኘ ዋጋ መፍጠር አይፈቀድም። "
        "መልስ ሲሰጥ የመረጃውን ቀን፣ ገበያ/ወረዳ፣ ምርት እና ምንጭ መግለጽ አለበት።\n\n"
        f"Dataset description:\n{notes}"
    ).strip()

    chunks.append({
        "id": "dynamic_market_nmis_dataset_metadata",
        "kb": "market",
        "data_layer": "dynamic_context",
        "source_org": "Agricultural Transformation Institute",
        "source_url": dataset_url,
        "updated_at": nmis_record.get("retrieved_at"),
        "validity": "dataset_metadata",
        "update_frequency": "weekly",
        "language_segment": "mixed",
        "text": text,
        "text_am": text_am,
        "metadata": {
            "source_type": "nmis_dataset_metadata",
            "fetch_status": "ok",
            "metadata_created": metadata_created,
            "metadata_modified": metadata_modified,
            "api_resources": api_resources,
            "raw_nmis_file": str(NMIS_JSON),
            "commodities_expected": [
                "teff",
                "maize",
                "wheat",
                "sesame",
                "haricot bean",
            ],
            "access_note": (
                "Actual detailed price records may require API/resource access. "
                "Do not invent price values if records are unavailable."
            ),
        },
    })

    for i, api_res in enumerate(api_resources, start=1):
        api_text_am = (
            f"NMIS API/resource መዳረሻ፡ {api_res.get('name')}። "
            f"Format፡ {api_res.get('format')}። "
            f"URL፡ {api_res.get('url')}። "
            "ይህ መዳረሻ የሳምንታዊ የገበያ መረጃ ለመስበስብ ሊሞከር ይችላል።"
        )

        chunks.append({
            "id": f"dynamic_market_nmis_api_resource_{i:03d}",
            "kb": "market",
            "data_layer": "dynamic_api_pointer",
            "source_org": "Agricultural Transformation Institute",
            "source_url": api_res.get("url") or dataset_url,
            "updated_at": nmis_record.get("retrieved_at"),
            "validity": "api_resource_pointer",
            "update_frequency": "weekly",
            "language_segment": "am",
            "text": api_text_am,
            "text_am": api_text_am,
            "metadata": {
                "source_type": "nmis_api_resource_pointer",
                "resource": api_res,
                "raw_nmis_file": str(NMIS_JSON),
            },
        })

    return chunks


# ============================================================
# CLIMATE / CIAT / NEXTGEN AGROADVISORY
# ============================================================

def build_climate_agroadvisory_chunks():
    chunks = []
    raw_records = []

    for i, src in enumerate(CLIMATE_AGROADVISORY_SOURCES, start=1):
        print(f"  climate/agroadvisory: {src['name']}")

        try:
            page_text = fetch_page_text(src["url"])
            status = "ok"
            error = None

        except Exception as e:
            page_text = ""
            status = "failed"
            error = str(e)

        raw_records.append({
            **src,
            "fetch_status": status,
            "fetch_error": error,
            "retrieved_at": now_iso(),
            "page_text_preview": page_text[:2000],
        })

        text = (
            f"Source: {src['name']}\n"
            f"Organization: {src['source_org']}\n"
            f"URL: {src['url']}\n"
            f"Update frequency: {src['update_frequency']}\n"
            f"Use: {src['description']}\n\n"
            f"Extracted content:\n{page_text}"
        ).strip()

        text_am = (
            f"የወቅታዊ/ክሊማት አግሮ-ምክር ምንጭ፡ {src['name']}\n"
            f"ድርጅት፡ {src['source_org']}\n"
            f"URL፡ {src['url']}\n"
            f"የመረጃ አዘምን፡ {src['update_frequency']}\n\n"
            f"አጠቃቀም፡ {src['description']}\n\n"
            "ይህ ምንጭ ለወቅታዊ የመዝራት ውሳኔ፣ የማዳበሪያ ምክር፣ "
            "የአየር/ክሊማት አደጋ ግምገማ፣ እና የclimate-smart agriculture "
            "ምክር እንደ ተጨማሪ አውድ ይጠቅማል።\n\n"
            f"{page_text[:7000]}"
        ).strip()

        chunks.append({
            "id": f"dynamic_climate_agroadvisory_{i:03d}",
            "kb": src["kb"],
            "data_layer": "seasonal_or_periodic_dynamic_context",
            "source_org": src["source_org"],
            "source_url": src["url"],
            "source_type": src["source_type"],
            "updated_at": now_iso(),
            "validity": src["update_frequency"],
            "update_frequency": src["update_frequency"],
            "language_segment": "mixed",
            "text": text,
            "text_am": text_am,
            "metadata": {
                "fetch_status": status,
                "fetch_error": error,
                "use_for": [
                    "seasonal planting advice",
                    "wheat suitability context",
                    "fertilizer advisory context",
                    "climate risk explanation",
                    "climate-smart agriculture",
                ],
            },
        })

    write_json(CLIMATE_AGROADVISORY_JSON, {
        "records": raw_records,
        "retrieved_at": now_iso(),
    })

    return chunks


# ============================================================
# RUN MODES
# ============================================================

def run_on_demand_weather(city_name):
    ensure_dirs()

    print(f"Fetching on-demand weather for: {city_name}")

    record = fetch_weather_for_city_name(city_name)
    append_or_update_on_demand_weather(record)

    safe_city = record["location"]["name"].lower().replace(" ", "_")

    chunk = weather_record_to_chunk(
        record,
        f"on_demand_weather_{safe_city}"
    )

    print(json.dumps({
        "requested_city": city_name,
        "matched_location": record["location"],
        "retrieved_at": record["retrieved_at"],
        "chunk": chunk,
    }, ensure_ascii=False, indent=2))

    print("\nAmharic weather summary:\n")
    print(chunk["text_am"])


def run_full_update():
    ensure_dirs()

    weather_records = []
    weather_errors = []

    print("Fetching scheduled weather forecasts...")

    for loc in LOCATIONS:
        try:
            print(f"  weather: {loc['name']}")
            weather_records.append(fetch_open_meteo_forecast(loc))

        except Exception as e:
            print(f"  failed weather for {loc['name']}: {e}")
            weather_errors.append({
                "location": loc,
                "error": str(e),
                "time": now_iso(),
            })

    print("Fetching soil dataset metadata from Ethiopian Agri Data Hub...")
    soil_records = fetch_soil_datasets_from_agrihub()

    print("Fetching ATI NMIS dataset metadata...")
    nmis_record = fetch_nmis_dataset_metadata()
    write_json(NMIS_JSON, nmis_record)

    print("Fetching climate/agroadvisory context from CIAT/Alliance sources...")
    climate_chunks = build_climate_agroadvisory_chunks()

    market_records = {
        "sources": MARKET_CONTEXT_SOURCES,
        "retrieved_at": now_iso(),
    }

    write_json(WEATHER_JSON, {
        "records": weather_records,
        "errors": weather_errors,
        "retrieved_at": now_iso(),
    })

    write_json(SOIL_JSON, {
        "records": soil_records,
        "retrieved_at": now_iso(),
    })

    write_json(MARKET_JSON, market_records)

    chunks = []
    chunks.extend(build_weather_chunks(weather_records))
    chunks.extend(build_soil_chunks(soil_records))
    chunks.extend(build_market_context_chunks())
    chunks.extend(build_nmis_market_chunks(nmis_record))
    chunks.extend(climate_chunks)

    write_jsonl(DYNAMIC_JSONL, chunks)

    report = {
        "generated_at": now_iso(),
        "weather_locations_requested": len(LOCATIONS),
        "weather_locations_success": len(weather_records),
        "weather_locations_failed": len(weather_errors),
        "soil_records_found": len([r for r in soil_records if r.get("package")]),
        "market_context_sources": len(MARKET_CONTEXT_SOURCES),
        "nmis_metadata_status": nmis_record.get("status"),
        "climate_agroadvisory_sources": len(CLIMATE_AGROADVISORY_SOURCES),
        "climate_agroadvisory_chunks": len(climate_chunks),
        "dynamic_chunks_written": len(chunks),
        "outputs": {
            "dynamic_chunks": str(DYNAMIC_JSONL),
            "weather_json": str(WEATHER_JSON),
            "on_demand_weather_json": str(ON_DEMAND_WEATHER_JSON),
            "soil_json": str(SOIL_JSON),
            "market_json": str(MARKET_JSON),
            "nmis_json": str(NMIS_JSON),
            "climate_agroadvisory_json": str(CLIMATE_AGROADVISORY_JSON),
        },
    }

    write_json(REPORT_JSON, report)

    print("\nDONE")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDynamic RAG context file:\n{DYNAMIC_JSONL}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Ethiopia farmer advisory dynamic context layer. "
            "With no flags, runs the full scheduled update (writes data/ under this script)."
        )
    )

    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Fetch on-demand weather forecast for a specific Ethiopian city/town.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.city:
        run_on_demand_weather(args.city)
        return

    run_full_update()


if __name__ == "__main__":
    main()