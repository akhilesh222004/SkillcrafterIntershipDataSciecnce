"""
Task 4: Traffic Accident Analysis — Patterns by Road Conditions, Weather, Time of Day
------------------------------------------------------------------------------------
- Uses a dummy dataset if no CSV provided.
- Identifies patterns by time, weather, and road conditions.
- Visualizes accident hotspots (if latitude/longitude available).
- Saves plots and summary tables to 'task4_outputs'.

USAGE:
    python Task4.py                # runs with dummy dataset
    python Task4.py --csv your.csv # runs with your dataset
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import folium
    FOLIUM_OK = True
except Exception:
    FOLIUM_OK = False

# Output folder
OUTDIR = "task4_outputs"
os.makedirs(OUTDIR, exist_ok=True)

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# -----------------------
# Helpers
# -----------------------

def safe_savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def make_dummy_data(n=1000):
    """Generate synthetic accident dataset"""
    np.random.seed(42)
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="H"),
        "Weather": np.random.choice(["Clear","Rain","Fog","Snow"], n, p=[0.6,0.25,0.1,0.05]),
        "RoadCondition": np.random.choice(["Dry","Wet","Icy","Snowy"], n, p=[0.7,0.2,0.05,0.05]),
        "Latitude": 25 + np.random.randn(n)*0.1,
        "Longitude": 55 + np.random.randn(n)*0.1
    })
    return df

def load_data(path=None):
    if path is None:
        print("[INFO] No CSV provided. Using dummy dataset...")
        return make_dummy_data()

    try:
        df = pd.read_csv(path)
        print("[INFO] Loaded dataset shape:", df.shape)
        return df
    except Exception as e:
        print("[WARN] Could not read CSV:", e)
        print("[INFO] Falling back to dummy dataset.")
        return make_dummy_data()

# -----------------------
# Main Analysis
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Task 4: Traffic Accident Analysis")
    parser.add_argument("--csv", type=str, help="Path to accidents CSV")
    args = parser.parse_args()

    df = load_data(args.csv)

    # Ensure datetime
    if "Date" in df.columns:
        df["_datetime"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["_datetime"] = pd.date_range("2020-01-01", periods=len(df), freq="H")

    df["_hour"] = df["_datetime"].dt.hour
    df["_dow"] = pd.Categorical(df["_datetime"].dt.day_name(), categories=DAY_ORDER, ordered=True)
    df["_month"] = df["_datetime"].dt.to_period("M").astype(str)

    # -----------------------
    # Summaries
    # -----------------------
    df["_hour"].value_counts().sort_index().to_csv(os.path.join(OUTDIR, "accidents_by_hour.csv"))
    df["_dow"].value_counts().to_csv(os.path.join(OUTDIR, "accidents_by_dayofweek.csv"))
    df["_month"].value_counts().sort_index().to_csv(os.path.join(OUTDIR, "accidents_by_month.csv"))
    if "Weather" in df:
        df["Weather"].value_counts().to_csv(os.path.join(OUTDIR, "accidents_by_weather.csv"))
    if "RoadCondition" in df:
        df["RoadCondition"].value_counts().to_csv(os.path.join(OUTDIR, "accidents_by_road_condition.csv"))

    # -----------------------
    # Visualizations
    # -----------------------
    sns.set_theme(style="whitegrid")

    # 1) Accidents by Hour
    plt.figure(figsize=(8,5))
    sns.countplot(x="_hour", data=df, color="skyblue")
    plt.title("Accidents by Hour of Day")
    plt.ylabel("Count")
    safe_savefig(os.path.join(OUTDIR, "accidents_by_hour.png"))

    # 2) Accidents by Day of Week
    plt.figure(figsize=(8,5))
    sns.countplot(x="_dow", data=df, order=DAY_ORDER, color="salmon")
    plt.title("Accidents by Day of Week")
    plt.ylabel("Count")
    safe_savefig(os.path.join(OUTDIR, "accidents_by_dayofweek.png"))

    # 3) Heatmap Day vs Hour
    pivot = df.pivot_table(index="_dow", columns="_hour", values=df.columns[0], aggfunc="count").fillna(0)
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, cmap="Reds", cbar_kws={"label":"Accident Count"})
    plt.title("Accident Heatmap (Day vs Hour)")
    safe_savefig(os.path.join(OUTDIR, "accident_heatmap.png"))

    # 4) Weather vs Road
    if "Weather" in df and "RoadCondition" in df:
        plt.figure(figsize=(8,6))
        sns.countplot(y="Weather", hue="RoadCondition", data=df)
        plt.title("Accidents by Weather and Road Condition")
        plt.xlabel("Count")
        plt.ylabel("Weather")
        plt.legend(title="Road Condition")
        safe_savefig(os.path.join(OUTDIR, "weather_vs_road.png"))

    # 5) Hotspot Map
    if FOLIUM_OK and "Latitude" in df and "Longitude" in df:
        fmap = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=10)
        for _, row in df.sample(min(200, len(df))).iterrows():
            folium.CircleMarker(location=[row["Latitude"], row["Longitude"]],
                                radius=2, color="red", fill=True).add_to(fmap)
        fmap.save(os.path.join(OUTDIR, "hotspot_map.html"))
        print("[INFO] Hotspot map saved as hotspot_map.html")

    print("[INFO] Analysis complete. Outputs saved to", OUTDIR)


if __name__ == "__main__":
    main()
