"""
Task 4: Traffic Accident Analysis — Patterns by Road Conditions, Weather, Time of Day
------------------------------------------------------------------------------------
- Cleans and explores accident data.
- Identifies patterns by time, weather, and road conditions.
- Visualizes accident hotspots (if latitude/longitude available).
- Saves plots and summary tables to 'task4_outputs'.

USAGE:
    # Recommended (pass your CSV path)
    python task4_accidents_eda.py --csv "C:/path/to/your/accidents.csv"

    # Or rely on default filename in current folder:
    python task4_accidents_eda.py
"""

import os
import argparse
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional mapping (for hotspots)
try:
    import folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False


# -----------------------
# Helpers & configuration
# -----------------------

DEFAULT_CSV = r"C:\Users\HP\OneDrive\Desktop\Skillcraft_Internship\Traffic_accidents_by_month_of_occurrence_2001-2014.csv"
df = pd.read_csv(DEFAULT_CSV)
print(df.head(10))
print(df.columns.tolist())
OUTDIR = "task4_outputs"
os.makedirs(OUTDIR, exist_ok=True)

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def is_probably_html(df: pd.DataFrame) -> bool:
    """Detects when a 'CSV' is actually an HTML page scraped by mistake."""
    if df.empty:
        return False
    joined_cols = " ".join(map(str, df.columns)).lower()
    if "<!doctype html>" in joined_cols or "<html" in joined_cols:
        return True
    # Also check first few cell values
    for col in df.columns[:2]:
        sample = str(df[col].astype(str).head(5).str.cat(sep=" ")).lower()
        if "<!doctype html>" in sample or "<html" in sample:
            return True
    return False

def first_existing(df: pd.DataFrame, candidates):
    """Return the first column that exists in df (case-insensitive match)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None

def normalize_category(s: pd.Series) -> pd.Series:
    if s is None:
        return None
    s = s.astype("string").str.strip().str.lower()
    # common cleanups
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

def title_case(s: pd.Series) -> pd.Series:
    if s is None:
        return None
    return s.astype("string").str.title()

def safe_savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def ensure_datetime(df, date_col, time_col=None):
    """Return a datetime series using available date/time columns."""
    if date_col is None and time_col is None:
        return None
    if date_col is not None and time_col is not None:
        # concatenate then parse
        dt = pd.to_datetime(df[date_col].astype(str).str.strip() + " " +
                            df[time_col].astype(str).str.strip(), errors="coerce")
    elif date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    else:  # only time col
        # If only time is present, anchor to an arbitrary date
        dt = pd.to_datetime("2000-01-01 " + df[time_col].astype(str), errors="coerce")
    return dt

def make_hour(series):
    return series.dt.hour

def make_dow(series):
    dow = series.dt.day_name()
    return pd.Categorical(dow, categories=DAY_ORDER, ordered=True)

def month_name(series):
    return series.dt.to_period("M").astype(str)

# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Task 4: Traffic Accident Analysis")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help=f"Path to accidents CSV (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    # Load
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        print("Tips:\n - Use forward slashes in Windows paths, or prefix with r''.\n - Ensure the file is a real CSV, not a web page.")
        return

    if is_probably_html(df):
        print("[ERROR] The file looks like an HTML page, not a CSV dataset.")
        print("Please export/download the actual CSV and re-run the script with --csv path.")
        print("For now, the script will stop to avoid misleading results.")
        return

    print("[INFO] Loaded shape:", df.shape)
    print("[INFO] Columns:", list(df.columns))

    # -----------------------
    # Column detection (flexible)
    # -----------------------
    # Try to find common names (case-insensitive)
    date_col = first_existing(df, ["date","crash_date","accident_date","occurred_on","datetime","timestamp"])
    time_col = first_existing(df, ["time","crash_time","accident_time","occurrence_time"])
    hour_col = first_existing(df, ["hour","crash_hour","accident_hour"])
    dow_col  = first_existing(df, ["day_of_week","weekday","day"])
    weather_col = first_existing(df, ["weather","weather_condition","atmospheric_conditions","weathercond"])
    road_col    = first_existing(df, ["road_condition","road_conditions","surface_condition","surface","road_surface"])
    light_col   = first_existing(df, ["light_conditions","light","lighting","light_condition"])
    lat_col     = first_existing(df, ["latitude","lat","y"])
    lon_col     = first_existing(df, ["longitude","lon","lng","x"])

    # -----------------------
    # Basic cleaning
    # -----------------------
    df = df.drop_duplicates()

    # Build a datetime column if possible
    dt = ensure_datetime(df, date_col, time_col)
    if dt is None and hour_col is None and dow_col is None:
        print("[WARN] No usable date/time columns found. Time-of-day analysis will be limited.")
    df["_datetime"] = dt

    # Feature engineering: hour, day-of-week, month
    if "_datetime" in df and df["_datetime"].notna().any():
        df["_hour"] = make_hour(df["_datetime"])
        df["_dow"] = make_dow(df["_datetime"])
        df["_month"] = month_name(df["_datetime"])
    else:
        # fallback if hour or dow columns already exist
        if hour_col is not None:
            df["_hour"] = pd.to_numeric(df[hour_col], errors="coerce").clip(0,23)
        if dow_col is not None:
            # try to standardize day names
            raw = df[dow_col].astype(str).str.strip().str.lower()
            mapping = {
                "mon":"Monday","monday":"Monday","1":"Monday",
                "tue":"Tuesday","tuesday":"Tuesday","2":"Tuesday",
                "wed":"Wednesday","wednesday":"Wednesday","3":"Wednesday",
                "thu":"Thursday","thursday":"Thursday","4":"Thursday",
                "fri":"Friday","friday":"Friday","5":"Friday",
                "sat":"Saturday","saturday":"Saturday","6":"Saturday",
                "sun":"Sunday","sunday":"Sunday","7":"Sunday","0":"Sunday"
            }
            df["_dow"] = pd.Categorical(raw.map(mapping).fillna(raw.str.title()),
                                        categories=DAY_ORDER, ordered=True)

    # Normalize categories
    if weather_col:
        df["_weather"] = title_case(normalize_category(df[weather_col]))
    if road_col:
        df["_road"] = title_case(normalize_category(df[road_col]))
    if light_col:
        df["_light"] = title_case(normalize_category(df[light_col]))

    # -----------------------
    # Summaries (saved as CSV)
    # -----------------------
    # Accidents by hour
    if "_hour" in df:
        by_hour = df["_hour"].dropna().astype(int).value_counts().sort_index()
        by_hour.to_csv(os.path.join(OUTDIR, "accidents_by_hour.csv"), header=["count"])

    # Accidents by day of week
    if "_dow" in df:
        by_dow = df["_dow"].value_counts().reindex(DAY_ORDER)
        by_dow.to_csv(os.path.join(OUTDIR, "accidents_by_dayofweek.csv"), header=["count"])

    # Accidents by month (if datetime present)
    if "_month" in df:
        by_month = df["_month"].value_counts().sort_index()
        by_month.to_csv(os.path.join(OUTDIR, "accidents_by_month.csv"), header=["count"])

    # By weather / road / light
    if "_weather" in df:
        df["_weather"].value_counts(dropna=True).to_csv(os.path.join(OUTDIR, "accidents_by_weather.csv"), header=["count"])
    if "_road" in df:
        df["_road"].value_counts(dropna=True).to_csv(os.path.join(OUTDIR, "accidents_by_road_condition.csv"), header=["count"])
    if "_light" in df:
        df["_light"].value_counts(dropna=True).to_csv(os.path.join(OUTDIR, "accidents_by_light.csv"), header=["count"])

    # -----------------------
    # Visualizations (Matplotlib)
    # -----------------------

    # 1) Accidents by hour (line/bar)
    if "_hour" in df and df["_hour"].notna().any():
        counts = df["_hour"].dropna().astype(int).value_counts().sort_index()
        plt.figure(figsize=(10,5))
        plt.plot(counts.index, counts.values, marker="o")
        plt.title("Accidents by Hour of Day")
        plt.xlabel("Hour (0–23)")
        plt.ylabel("Accident count")
        safe_savefig(os.path.join(OUTDIR, "accidents_by_hour.png"))

    # 2) Heatmap: Day of week × hour
    if "_dow" in df and "_hour" in df and df["_dow"].notna().any() and df["_hour"].notna().any():
        pivot = df.pivot_table(index="_dow", columns="_hour", values=df.columns[0], aggfunc="count").fillna(0)
        plt.figure(figsize=(14,5))
        plt.imshow(pivot.values, aspect="auto")
