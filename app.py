from flask import Flask, render_template
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# --- Load Dataset ---
DATA_PATH = "data/jumlah_produksi_sampah_menurut_jenisnya_di_kota_ban_1.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # coba cari kolom tahun/bulan
    date_col = None
    for c in df.columns:
        if "tahun" in c.lower(): date_col = c
        if "bulan" in c.lower(): date_col = c

    if date_col is None:
        df["date"] = pd.date_range(start="2015-01-01", periods=len(df), freq="M")
    else:
        df["date"] = pd.to_datetime(df[date_col], errors='coerce')

    # kolom produksi
    prod_col = None
    for c in df.columns:
        if "jumlah" in c.lower() or "produksi" in c.lower() or "ton" in c.lower():
            prod_col = c

    df["produksi"] = pd.to_numeric(df[prod_col], errors="coerce")
    df = df.dropna(subset=["produksi"]).sort_values("date")

    return df

# --- Monte Carlo Simulation ---
def monte_carlo(series, steps=12, sims=1000):
    last = series.iloc[-1]
    pct = series.pct_change().dropna()
    mu = pct.mean()
    sigma = pct.std()

    result = np.zeros((sims, steps))
    for i in range(sims):
        val = last
        for s in range(steps):
            growth = np.random.normal(mu, sigma)
            val *= (1 + growth)
            result[i, s] = val

    p10 = np.percentile(result, 10, axis=0)
    p50 = np.percentile(result, 50, axis=0)
    p90 = np.percentile(result, 90, axis=0)

    return p10, p50, p90

# ---------------------------
@app.route("/")
def index():
    df = load_data()
    series = df["produksi"]

    p10, p50, p90 = monte_carlo(series)

    forecast = []
    future_dates = pd.date_range(df["date"].iloc[-1], periods=13, freq="M")[1:]
    for i in range(12):
        forecast.append({
            "date": future_dates[i].strftime("%Y-%m"),
            "p10": round(p10[i],2),
            "p50": round(p50[i],2),
            "p90": round(p90[i],2)
        })

    # dataset preview
    preview = df.head(10).to_dict(orient="records")

    return render_template("index.html",
                           preview=preview,
                           forecast=forecast)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

