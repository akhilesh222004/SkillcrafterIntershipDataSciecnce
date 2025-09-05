"""
Task 2: Data Cleaning & Exploratory Data Analysis (EDA)
Dataset: EV Charging Stations (2025)
bash : python task2.1.py --csv "C:\Users\HP\OneDrive\Desktop\Skillcraft_Internship\charging_stations_2025_world.csv"

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# 1. Parse arguments

parser = argparse.ArgumentParser(description="Task 2: Clean & EDA EV Charging Stations dataset")
parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
parser.add_argument("--outdir", type=str, default="eda_outputs", help="Directory to save outputs")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# --------------------------
# 2. Load dataset
# --------------------------
df = pd.read_csv(args.csv)
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------
# 3. Data Cleaning
# --------------------------

# Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
print(f"Removed {before - df.shape[0]} duplicates")

# Handle missing values
if "power_kw" in df.columns:
    median_power = df["power_kw"].median(skipna=True)
    df["power_kw"].fillna(median_power, inplace=True)
    print(f"Filled missing power_kw with median: {median_power:.1f}")

if "city" in df.columns:
    df["city"].fillna("Unknown", inplace=True)

# Standardize categorical data
if "country_code" in df.columns:
    df["country_code"] = df["country_code"].astype(str).str.strip().str.upper()

# Drop invalid coordinates
if "latitude" in df.columns and "longitude" in df.columns:
    valid_coords = (df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))
    dropped = (~valid_coords).sum()
    df = df[valid_coords]
    print(f"Dropped {dropped} rows with invalid coordinates")

print("Shape after cleaning:", df.shape)

# --------------------------
# 4. Diagnostics
# --------------------------
if "power_kw" in df.columns:
    power = df["power_kw"].dropna().astype(float)

    print("\n--- power_kw summary with high percentiles ---")
    print(power.describe(percentiles=[0.90, 0.95, 0.99, 0.999]))

    print("\n--- Top 10 largest power_kw values ---")
    top_extremes = df.loc[power.sort_values(ascending=False).index[:10], ["country_code", "power_kw"]]
    print(top_extremes)

# --------------------------
# 5. Exploratory Data Analysis (EDA)
# --------------------------

# (a) Histogram of power_kw (log scale)
if "power_kw" in df.columns:
    power = df["power_kw"].dropna().astype(float)
    pmin = max(0.1, power[power > 0].min()) if (power > 0).any() else 0.1
    pmax = power.quantile(0.999)  # avoid extreme outliers

    bins = np.logspace(np.log10(pmin), np.log10(max(pmax, pmin * 1.1)), 50)

    plt.figure(figsize=(10, 6))
    plt.hist(power[power > 0], bins=bins, edgecolor="black")
    plt.xscale("log")
    plt.title("Distribution of Charging Power (kW) — log scale")
    plt.xlabel("Power (kW, log scale)")
    plt.ylabel("Number of Stations")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "hist_power_log.png"))
    plt.close()

    # (b) Trimmed histogram up to 99th percentile
    p99 = power.quantile(0.99)
    trimmed = power[(power > 0) & (power <= p99)]

    plt.figure(figsize=(10, 6))
    plt.hist(trimmed, bins=40, edgecolor="black")
    plt.title(f"Charging Power (kW) — trimmed (≤ {p99:.1f} kW)")
    plt.xlabel("Power (kW)")
    plt.ylabel("Number of Stations")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "hist_power_trimmed.png"))
    plt.close()

# (c) Top 10 countries by station count
if "country_code" in df.columns:
    top_countries = df["country_code"].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    top_countries.plot(kind="bar")
    plt.title("Top 10 Countries by Charging Stations")
    plt.xlabel("Country Code")
    plt.ylabel("Number of Stations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "top_countries.png"))
    plt.close()

# (d) Scatter: power vs ports
if "ports" in df.columns and "power_kw" in df.columns:
    sample = df[["power_kw", "ports"]].dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    plt.figure(figsize=(8, 6))
    plt.scatter(sample["power_kw"], sample["ports"], alpha=0.5)
    plt.title("Relationship: Power (kW) vs Ports")
    plt.xlabel("Power (kW)")
    plt.ylabel("Ports")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "scatter_power_ports.png"))
    plt.close()

# --------------------------
# 6. Save Cleaned Data
# --------------------------
output_file = os.path.join(args.outdir, "charging_stations_cleaned.csv")
df.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to: {output_file}")
