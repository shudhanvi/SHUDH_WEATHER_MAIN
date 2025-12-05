import os
import sys
import json
import time
import math
import traceback
from datetime import datetime, timedelta, date
from typing import Dict, Any, Tuple, List

import requests
import numpy as np
import pandas as pd
from dateutil import parser as dateparser

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler

# Timezone (IST)
from pytz import timezone
IST = timezone("Asia/Kolkata")

#CORS
from fastapi.middleware.cors import CORSMiddleware

# -------------------- CONFIG --------------------

DATA_PATH = './final_manhole_operations_with_docket.csv'
MODEL_DIR = './models_local'
WEATHER_CACHE_PATH = './weather_cache.json'
AREA_CACHE_PATH = './area_weather_cache.json'
ALERTS_LOG = './alerts_log.json'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow ALL frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_EXPIRY_HOURS = 6

os.makedirs(MODEL_DIR, exist_ok=True)

# Visual Crossing
VISUAL_CROSSING_KEY = "Z895WNUFPQFN5N8HBANA72C9R"
VC_BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

# Google Geocoding (optional)
GOOGLE_GEOCODE_KEY = "AIzaSyDBX5VW76ML3yFXpfDOZzmKBX_MfUfarSQ"

# Alert thresholds
ALERT_THRESHOLDS = {
    'rain_24h_mm': 20.0,
    'precip_prob_pct': 70.0,
    'storm_flag_prob_pct': 20.0,
    'hourly_rain_mm': 5.0,
    'heatindex_threshold': 35.0,
}

# -------------------- UTILITIES --------------------

def log(msg: str):
    """Log using IST timestamp."""
    ts = datetime.now(IST).isoformat()
    print(f"[{ts}] {msg}")

def now_ist():
    """Return current datetime in IST."""
    return datetime.now(IST)

def today_ist_str():
    """Return today's date (IST) as YYYY-MM-DD."""
    return now_ist().date().isoformat()

def read_json_file(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        log(f"[WARN] Failed to read {path}: {e}")
    return default

def write_json_file(path, data):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        log(f"[ERROR] Failed to write {path}: {e}")

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        log(f"[ERROR] CSV missing: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path).copy()
    except Exception as e:
        log(f"[ERROR] Failed to read CSV {path}: {e}")
        return pd.DataFrame()

def safe_parse_timestamp(s):
    if pd.isna(s):
        return pd.NaT
    try:
        return dateparser.parse(str(s))
    except Exception:
        return pd.NaT

# -------------------- DOCKET + AREA KEY --------------------

def clean_docket_id(did: Any) -> str:
    s = str(did).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def make_area_docket_key(area_name: str, docket_no: Any) -> str:
    area_key = (area_name or "unknown").strip().lower().replace(" ", "_")
    docket_clean = clean_docket_id(docket_no)
    return f"{area_key}_{docket_clean}"

# -------------------- VISUAL CROSSING FETCH --------------------

def fetch_visual_crossing(lat: float, lon: float) -> dict:
    url = f"{VC_BASE}/{lat},{lon}"
    params = {
        "unitGroup": "metric",
        "key": VISUAL_CROSSING_KEY,
        "include": "hours,days"
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        log(f"[ERROR] VC fetch failed for {lat},{lon}: {e}")
        return {}

# -------------------- STANDARDIZE VC WEATHER --------------------

def vc_to_standard(vc: dict) -> dict:
    if not vc:
        return {}

    hours = []
    if "days" in vc and vc["days"]:
        for d in vc["days"]:
            if "hours" in d:
                hours.extend(d["hours"])

    hours_24 = hours[:24]
    hours_48 = hours[:48]

    rain_24h = sum(h.get("precip", 0) for h in hours_24)
    rain_48h = sum(h.get("precip", 0) for h in hours_48)
    rain_7d = sum(d.get("precip", 0) for d in vc.get("days", [])[:7])

    precip_prob_24h = max((h.get("precipprob", 0) for h in hours_24), default=0)
    max_hourly_precip = max((h.get("precip", 0) for h in hours_24), default=0)
    max_wind_24h = max((h.get("windspeed", 0) for h in hours_24), default=0)

    temps = [h.get("temp") for h in hours_24 if h.get("temp") is not None]
    hums = [h.get("humidity") for h in hours_24 if h.get("humidity") is not None]

    avg_temp_24h = float(np.nanmean(temps)) if temps else 0.0
    avg_humidity_24h = float(np.nanmean(hums)) if hums else 0.0

    datewise = []
    for d in vc.get("days", [])[:7]:
        datewise.append({
            "date": d.get("datetime"),
            "precip_mm": d.get("precip", 0),
            "precip_prob": d.get("precipprob", 0),
            "avgtemp_c": d.get("temp", 0),
            "avghumidity": d.get("humidity", 0),
            "windspeed_max": d.get("windspeed", 0),
            "conditions": d.get("conditions", "")
        })

    return {
        "provider": "visual-crossing",
        "datewise_7d": datewise,
        "rain_24h": rain_24h,
        "rain_48h": rain_48h,
        "rain_7d": rain_7d,
        "avg_temp_24h": avg_temp_24h,
        "avg_humidity_24h": avg_humidity_24h,
        "max_wind_24h": max_wind_24h,
        "max_hourly_precip_24h": max_hourly_precip,
        "precip_prob_24h": precip_prob_24h,
        "storm_flag": 1 if (precip_prob_24h > 80 and rain_24h > 30) else 0,
        "hourly_precips": [h.get("precip", 0) for h in hours_24]
    }

# -------------------- DOCKET WEATHER CACHE --------------------

def docket_cache_fresh(ts_iso: str) -> bool:
    if not ts_iso:
        return False
    try:
        ts = datetime.fromisoformat(ts_iso)
        # interpret ts as IST datetime
        ts = IST.localize(ts.replace(tzinfo=None))
        return (now_ist() - ts) < timedelta(hours=CACHE_EXPIRY_HOURS)
    except Exception:
        return False

def get_docket_weather(
    docket_no: str,
    lat: float,
    lon: float,
    cache: dict,
    area_name: str = "unknown"
) -> dict:

    today = today_ist_str()
    docket_clean = clean_docket_id(docket_no)
    key = make_area_docket_key(area_name, docket_clean)

    if key not in cache:
        cache[key] = {}

    entry = cache[key].get(today)
    if (
        entry
        and docket_cache_fresh(entry.get("fetched_at"))
        and entry.get("data", {}).get("provider") not in ("none", None)
    ):
        return entry["data"]

    raw = fetch_visual_crossing(lat, lon)
    std = vc_to_standard(raw)

    if not std:
        std = {
            "provider": "none",
            "datewise_7d": [],
            "rain_24h": 0.0,
            "rain_48h": 0.0,
            "rain_7d": 0.0,
            "avg_temp_24h": 0.0,
            "avg_humidity_24h": 0.0,
            "max_wind_24h": 0.0,
            "max_hourly_precip_24h": 0.0,
            "precip_prob_24h": 0.0,
            "storm_flag": 0,
            "hourly_precips": [0.0] * 24
        }

    # save lat/lon
    cache[key]["lat"] = lat
    cache[key]["lon"] = lon

    # save weather for today in IST
    cache[key][today] = {
        "fetched_at": now_ist().isoformat(),
        "data": std
    }

    write_json_file(WEATHER_CACHE_PATH, cache)
    return std

# -------------------- AREA WEATHER CACHE --------------------

def area_cache_fresh(ts_iso: str) -> bool:
    return docket_cache_fresh(ts_iso)

def geocode_area_google(area_name: str, region_hint="Hyderabad,India") -> Tuple[float, float]:
    if not GOOGLE_GEOCODE_KEY:
        return 0.0, 0.0
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": f"{area_name}, {region_hint}", "key": GOOGLE_GEOCODE_KEY}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data and data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
    except Exception as e:
        log(f"[WARN] Google geocoding failed: {e}")
    return 0.0, 0.0

def geocode_area_nominatim(area_name: str) -> Tuple[float, float]:
    full_query = f"{area_name}, Khairatabad, Hyderabad, Telangana, India"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": full_query, "format": "json", "limit": 1}

    for attempt in range(2):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=10)
            if res.status_code == 429:
                time.sleep(1)
                continue

            res.raise_for_status()
            data = res.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
        except Exception as e:
            log(f"[WARN] Nominatim attempt {attempt+1} failed: {e}")
            time.sleep(1)

    return 0.0, 0.0

def get_area_coordinates(area_name: str, df_area: pd.DataFrame, cache_lookup: dict):
    akey = area_name.strip().lower()

    # cached
    try:
        if akey in cache_lookup:
            c = cache_lookup[akey].get("coords")
            if c and c.get("lat") and c.get("lon"):
                return float(c["lat"]), float(c["lon"])
    except Exception:
        pass

    # google
    lat, lon = geocode_area_google(area_name)
    if lat and lon:
        return lat, lon

    # nominatim
    lat, lon = geocode_area_nominatim(area_name)
    if lat and lon:
        return lat, lon

    # fallback to centroid
    try:
        return float(df_area["latitude"].mean()), float(df_area["longitude"].mean())
    except Exception:
        return 0, 0

def get_area_weather(area_name: str, lat: float, lon: float, cache: dict, df_area: pd.DataFrame):

    akey = area_name.strip().lower()
    today = today_ist_str()

    if akey not in cache:
        cache[akey] = {}

    # load coords from cache if available
    resolved_lat = None
    resolved_lon = None
    try:
        if "coords" in cache[akey]:
            resolved_lat = float(cache[akey]["coords"]["lat"])
            resolved_lon = float(cache[akey]["coords"]["lon"])
    except Exception:
        resolved_lat, resolved_lon = None, None

    # If missing coords → resolve
    if not resolved_lat or not resolved_lon:
        lookup = read_json_file(AREA_CACHE_PATH, {})
        rlat, rlon = get_area_coordinates(area_name, df_area, lookup)

        if (not rlat or not rlon) and lat and lon:
            rlat, rlon = lat, lon

        resolved_lat, resolved_lon = rlat, rlon

        if resolved_lat and resolved_lon:
            cache[akey]["coords"] = {"lat": resolved_lat, "lon": resolved_lon}
            write_json_file(AREA_CACHE_PATH, cache)

    entry = cache[akey].get(today)
    if (
        entry
        and area_cache_fresh(entry.get("fetched_at"))
        and entry.get("data", {}).get("provider") not in ("none", None)
    ):
        return entry["data"]

    raw = fetch_visual_crossing(resolved_lat, resolved_lon)
    std = vc_to_standard(raw) if raw else {}

    if not std:
        std = {
            "provider": "none",
            "datewise_7d": [],
            "rain_24h": 0.0,
            "rain_48h": 0.0,
            "rain_7d": 0.0,
            "avg_temp_24h": 0.0,
            "avg_humidity_24h": 0.0,
            "max_wind_24h": 0.0,
            "max_hourly_precip_24h": 0.0,
            "precip_prob_24h": 0.0,
            "storm_flag": 0,
            "hourly_precips": [0.0] * 24
        }

    cache[akey][today] = {
        "fetched_at": now_ist().isoformat(),
        "data": std
    }
    write_json_file(AREA_CACHE_PATH, cache)

    return std

# ------------------------------------------------------------
# DF PARSING & ENRICHMENT
# ------------------------------------------------------------

def parse_and_enrich_df(df: pd.DataFrame, cache: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # timestamp
    if "timestamp" in df.columns:
        df["ts"] = df["timestamp"].apply(safe_parse_timestamp)
    else:
        df["ts"] = pd.NaT

    # operation time
    df["operation_time_minutes"] = pd.to_numeric(
        df.get("operation_time_minutes", 0),
        errors="coerce"
    ).fillna(0)

    # ensure docket_no
    if "sw_mh_docket_no" not in df.columns:
        if "docket_no" in df.columns:
            df["sw_mh_docket_no"] = df["docket_no"]
        else:
            df["sw_mh_docket_no"] = "unknown"

    df["docket_no"] = df["sw_mh_docket_no"].astype(str)

    # ensure area
    if "area" not in df.columns:
        df["area"] = "unknown"

    # ensure lat/lon numeric
    for c in ["latitude", "longitude"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["docket_no_clean"] = df["docket_no"].apply(clean_docket_id)
    df["area_safe"] = df["area"].astype(str)
    df["docket_key"] = df.apply(
        lambda r: make_area_docket_key(r["area_safe"], r["docket_no_clean"]),
        axis=1
    )

    # centroid per docket_key
    centroids = (
        df.groupby("docket_key")[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )

    meta = (
        df[["docket_key", "area_safe", "docket_no_clean"]]
        .drop_duplicates(subset=["docket_key"])
        .set_index("docket_key")
        .to_dict("index")
    )

    # fetch weather per docket
    docket_weather = {}
    for _, row in centroids.iterrows():
        try:
            key = row["docket_key"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            info = meta.get(key, {})
            area_name = info.get("area_safe", "unknown")
            did_clean = info.get("docket_no_clean", "")
            docket_weather[key] = get_docket_weather(did_clean, lat, lon, cache, area_name)
        except Exception as e:
            log(f"[WARN] Failed to fetch weather: {e}")
            docket_weather[key] = {}

    # attach to df
    weather_rows = [docket_weather.get(k, {}) for k in df["docket_key"]]

    if weather_rows:
        df = pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(weather_rows).reset_index(drop=True)],
            axis=1
        )
    else:
        df["provider"] = "none"
        df["rain_24h"] = 0.0
        df["rain_48h"] = 0.0
        df["rain_7d"] = 0.0
        df["avg_temp_24h"] = 0.0
        df["avg_humidity_24h"] = 0.0
        df["max_wind_24h"] = 0.0
        df["max_hourly_precip_24h"] = 0.0
        df["precip_prob_24h"] = 0.0
        df["storm_flag"] = 0
        df["hourly_precips"] = [[] for _ in range(len(df))]

    # risk label
    df["risk_label"] = 0
    try:
        op_mean = df["operation_time_minutes"].mean()
        op_std = df["operation_time_minutes"].std()

        df.loc[
            (df["operation_time_minutes"] > (op_mean + 1.5 * op_std)) |
            (df.get("rain_24h", 0) > ALERT_THRESHOLDS["rain_24h_mm"]),
            "risk_label"
        ] = 1
    except Exception:
        pass

    df["priority_score"] = df["risk_label"] * 2

    return df

# ------------------------------------------------------------
# HOTSPOT DETECTION
# ------------------------------------------------------------

def detect_hotspots(df: pd.DataFrame, eps_meters=100, min_samples=3):
    from sklearn.cluster import DBSCAN
    coords = df[["latitude", "longitude"]].dropna()
    coords = coords[(coords["latitude"] != 0) & (coords["longitude"] != 0)]
    if coords.empty:
        df["cluster"] = -1
        return df

    rad = np.radians(coords.values)
    eps_rad = eps_meters / 6371000.0
    clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = clustering.fit_predict(rad)

    coords = coords.reset_index()
    coords["cluster"] = labels
    df = df.reset_index(drop=True)
    return df.merge(coords[["index", "cluster"]], left_index=True, right_on="index", how="left")

# ------------------------------------------------------------
# MODEL TRAINING / PREDICT
# ------------------------------------------------------------

def train_models(df: pd.DataFrame):
    feature_cols = [
        "operation_time_minutes",
        "avg_temp_24h", "avg_humidity_24h",
        "rain_24h", "rain_48h", "rain_7d",
        "max_wind_24h",
        "precip_prob_24h",
        "max_hourly_precip_24h"
    ]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feature_cols].fillna(0)
    y = df["risk_label"].astype(int)

    strat = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

    try:
        clf = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False
        )
        clf.fit(X_train_s, y_train)
    except Exception:
        clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        clf.fit(X_train_s, y_train)

    joblib.dump(clf, f"{MODEL_DIR}/risk_model.joblib")

    df["priority_bin"] = (
        (df["priority_score"] >= 2) | (df["risk_label"] == 1)
    ).astype(int)

    yp = df["priority_bin"].astype(int)
    Xp = df[feature_cols].fillna(0)

    stratp = yp if len(np.unique(yp)) > 1 else None

    Xp_train, _, yp_train, _ = train_test_split(
        Xp, yp, test_size=0.2, random_state=42, stratify=stratp
    )

    Xp_train_s = scaler.transform(Xp_train)
    prio = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    prio.fit(Xp_train_s, yp_train)

    joblib.dump(prio, f"{MODEL_DIR}/priority_model.joblib")
    return True

def load_models():
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    risk = joblib.load(f"{MODEL_DIR}/risk_model.joblib")
    prio = joblib.load(f"{MODEL_DIR}/priority_model.joblib")
    return scaler, risk, prio

# ------------------------------------------------------------
# ALERT EVALUATION
# ------------------------------------------------------------

def evaluate_alert_reasons(row):
    reasons = []
    if row.get("rain_24h", 0.0) >= ALERT_THRESHOLDS["rain_24h_mm"]:
        reasons.append(f"Heavy rain next 24h: {row.get('rain_24h')} mm")

    if row.get("precip_prob_24h", 0.0) >= ALERT_THRESHOLDS["precip_prob_pct"]:
        reasons.append(f"High rain probability: {row.get('precip_prob_24h')}%")

    if row.get("storm_flag", 0) >= 1:
        reasons.append("Storm probability high")

    hourly = row.get("hourly_precips", [])
    if hourly and max(hourly) >= ALERT_THRESHOLDS["hourly_rain_mm"]:
        reasons.append(f"Hourly rain > {ALERT_THRESHOLDS['hourly_rain_mm']} mm/hr")

    temp = row.get("avg_temp_24h", 0.0)
    hum = row.get("avg_humidity_24h", 0.0)
    heat_index = temp + (hum/100) * 10
    if heat_index >= ALERT_THRESHOLDS["heatindex_threshold"]:
        reasons.append(f"Heat/Humidity risk: heat_index={heat_index:.1f}")

    return reasons

# ------------------------------------------------------------
# ALERT SCAN
# ------------------------------------------------------------

def run_alert_scan(df, cache):
    alerts = read_json_file(ALERTS_LOG, [])
    new_alerts = []

    if "sw_mh_docket_no" not in df.columns:
        return []

    dockets = df[["sw_mh_docket_no", "latitude", "longitude", "area"]].drop_duplicates()

    for _, d in dockets.iterrows():
        did = clean_docket_id(d["sw_mh_docket_no"])
        lat = float(d["latitude"])
        lon = float(d["longitude"])
        area = str(d["area"])

        std = get_docket_weather(did, lat, lon, cache, area)
        reasons = evaluate_alert_reasons(std)

        if reasons:
            alert = {
                "time": now_ist().isoformat(),
                "docket_no": did,
                "docket_key": make_area_docket_key(area, did),
                "area": area,
                "lat": lat,
                "lon": lon,
                "reasons": reasons,
                "weather_summary": {
                    "rain_24h": std.get("rain_24h"),
                    "precip_prob_24h": std.get("precip_prob_24h"),
                    "storm_flag": std.get("storm_flag"),
                    "max_hourly_precip_24h": std.get("max_hourly_precip_24h"),
                }
            }
            new_alerts.append(alert)

    if new_alerts:
        alerts.extend(new_alerts)
        write_json_file(ALERTS_LOG, alerts)
        log(f"[ALERTS] Found {len(new_alerts)} alerts")
    else:
        log("[ALERTS] No alerts")

    return new_alerts

# ------------------------------------------------------------
# FASTAPI ENDPOINTS
# ------------------------------------------------------------

app = FastAPI(title="IST Weather & Risk API")

class PredictRequest(BaseModel):
    manhole_id: str
    sw_mh_docket_no: str
    latitude: float
    longitude: float
    operation_time_minutes: float = 0.0
    timestamp: str = None

@app.get("/health")
def api_health():
    models_ok = all(
        os.path.exists(f"{MODEL_DIR}/{m}") for m in
        ["scaler.joblib", "risk_model.joblib", "priority_model.joblib"]
    )

    return {
        "status": "online",
        "models_present": models_ok,
        "alerts_count": len(read_json_file(ALERTS_LOG, [])),
        "time_now_ist": now_ist().isoformat(),
    }

@app.get("/weather/{docket_no}")
def api_weather(docket_no: str):
    cache = read_json_file(WEATHER_CACHE_PATH, {})
    df = load_csv(DATA_PATH)
    if df.empty:
        raise HTTPException(500, "CSV missing")

    df["docket_no_clean"] = df["sw_mh_docket_no"].astype(str).apply(clean_docket_id)
    target = clean_docket_id(docket_no)

    d = df[df["docket_no_clean"] == target]
    if d.empty:
        raise HTTPException(404, "Docket not found")

    lat = float(d["latitude"].mean())
    lon = float(d["longitude"].mean())
    area = str(d["area"].iloc[0])

    std = get_docket_weather(target, lat, lon, cache, area)

    return {
        "docket_no": target,
        "area": area,
        "time_ist": now_ist().isoformat(),
        "provider": std.get("provider"),
        "datewise_7d": std.get("datewise_7d"),
        "rain_24h": std.get("rain_24h"),
        "precip_prob_24h": std.get("precip_prob_24h"),
        "storm_flag": std.get("storm_flag"),
    }

@app.get("/weather/area/{area_name}")
def api_area_weather(area_name: str):
    df = load_csv(DATA_PATH)
    if df.empty:
        raise HTTPException(500, "Data missing")

    df_area = df[df["area"].astype(str).str.lower() == area_name.lower()]
    if df_area.empty:
        raise HTTPException(404, "Area not found")

    centroid_lat = float(df_area["latitude"].mean())
    centroid_lon = float(df_area["longitude"].mean())

    cache = read_json_file(AREA_CACHE_PATH, {})
    std = get_area_weather(area_name, centroid_lat, centroid_lon, cache, df_area)

    return {
        "area": area_name,
        "time_ist": now_ist().isoformat(),
        "provider": std.get("provider"),
        "features": {
            "rain_24h_mm": std.get("rain_24h"),
            "precip_prob_24h": std.get("precip_prob_24h"),
            "max_wind_24h": std.get("max_wind_24h"),
            "storm_flag": std.get("storm_flag"),
        },
        "datewise_7d": std.get("datewise_7d"),
    }

@app.post("/predict")
def api_predict(req: PredictRequest):
    row = {
        "manhole_id": req.manhole_id,
        "sw_mh_docket_no": req.sw_mh_docket_no,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "operation_time_minutes": req.operation_time_minutes,
        "timestamp": req.timestamp or now_ist().isoformat(),
        "area": "unknown"
    }
    df = pd.DataFrame([row])

    cache = read_json_file(WEATHER_CACHE_PATH, {})
    df = parse_and_enrich_df(df, cache)

    if not all(
        os.path.exists(f"{MODEL_DIR}/{m}") for m in
        ["scaler.joblib", "risk_model.joblib", "priority_model.joblib"]
    ):
        base = load_csv(DATA_PATH)
        if base.empty:
            raise HTTPException(500, "No training data available")

        log("[MODEL] Training required...")
        base_enriched = parse_and_enrich_df(base, cache)
        detect_hotspots(base_enriched)
        train_models(base_enriched)
        log("[MODEL] Training done.")

    scaler, risk, prio = load_models()
    out = df.copy()

    # predict
    feature_cols = [
        "operation_time_minutes",
        "avg_temp_24h", "avg_humidity_24h",
        "rain_24h", "rain_48h", "rain_7d",
        "max_wind_24h",
        "precip_prob_24h",
        "max_hourly_precip_24h"
    ]
    X = out[feature_cols].fillna(0)
    Xs = scaler.transform(X)

    out["pred_risk_label"] = risk.predict(Xs)
    out["pred_priority_bin"] = prio.predict(Xs)

    r = out.iloc[0].to_dict()

    reasons = evaluate_alert_reasons(r)
    next_clean = (now_ist() + timedelta(days=1 if r["pred_priority_bin"] == 1 else 7)).date().isoformat()

    return {
        "time_ist": now_ist().isoformat(),
        "risk": int(r["pred_risk_label"]),
        "priority": int(r["pred_priority_bin"]),
        "next_cleaning_date": next_clean,
        "reasons": reasons,
        "weather": {
            "rain_24h": r.get("rain_24h"),
            "precip_prob_24h": r.get("precip_prob_24h"),
            "storm_flag": r.get("storm_flag"),
        }
    }

@app.get("/alerts")
def api_alerts():
    return read_json_file(ALERTS_LOG, [])

# ------------------------------------------------------------
# MAIN + SCHEDULER (IST)
# ------------------------------------------------------------

def main_auto():
    log("[MAIN] Starting system...")

    weather_cache = read_json_file(WEATHER_CACHE_PATH, {})
    area_cache = read_json_file(AREA_CACHE_PATH, {})

    df = load_csv(DATA_PATH)
    if df.empty:
        log("[MAIN] CSV missing, stopping.")
        return

    # warm-up all docket weather
    log("[MAIN] Warming up ALL docket weather...")
    parse_and_enrich_df(df, weather_cache)

    # warm-up areas
    if "area" in df.columns:
        areas = (
            df.assign(area_norm=df["area"].astype(str).str.lower())
            .groupby("area_norm")[["latitude", "longitude"]]
            .mean()
            .reset_index()
        )
        for _, row in areas.iterrows():
            area = row["area_norm"]
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            df_area = df[df["area"].astype(str).str.lower() == area]
            get_area_weather(area, lat, lon, area_cache, df_area)

    # Scheduler (IST)
    scheduler = BackgroundScheduler(timezone=IST)

    def alerts_job():
        try:
            log("[ALERTS] Running daily alert scan...")
            df_local = load_csv(DATA_PATH)
            if df_local.empty:
                log("[ALERTS] CSV missing.")
                return
            cache_local = read_json_file(WEATHER_CACHE_PATH, {})
            enriched = parse_and_enrich_df(df_local, cache_local)
            run_alert_scan(enriched, cache_local)
        except Exception as e:
            log(f"[ALERTS] Error: {e}")

    scheduler.add_job(
        alerts_job,
        "cron",
        hour=6,
        minute=0,
        id="alerts_daily",
        max_instances=1
    )

    scheduler.start()
    log("[SCHED] Daily alerts enabled (06:00 AM IST).")

    log("[MAIN] Starting FastAPI at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

if __name__ == "__main__":
    main_auto()
