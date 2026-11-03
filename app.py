"""
VCP screener (Mark Minervini style) for NSE stocks.

A stock is suggested when ALL of these are true:
  1. EMA(21) crossed ABOVE EMA(63) within the last `cross_bars` candles ("just crossed")
  2. ROC is near zero  ->  |ROC| <= roc_tol (%)
        ROC length = 18 for small caps, 20 for mid/large caps
  3. A VCP is visible: >= `min_contractions` pullbacks, each shallower than the
     one before, last one tight, price near the pivot, volume drying up.

Run:
    pip install -r requirements.txt
    python app.py          ->  http://127.0.0.1:5000
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

CACHE = {}          # params -> (timestamp, results)
CACHE_TTL = 15 * 60  # seconds
BATCH = 40          # tickers per yfinance download call


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
def load_universe():
    """universe.csv columns: symbol,cap   (cap = small | mid | large)"""
    df = pd.read_csv("universe.csv")
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["cap"] = df["cap"].str.strip().str.lower()
    return df


def roc_length(cap):
    return 18 if cap == "small" else 20  # mid & large -> 20


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def add_indicators(df, cap):
    n = roc_length(cap)
    df = df.copy()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema63"] = df["Close"].ewm(span=63, adjust=False).mean()
    df["roc"] = (df["Close"] / df["Close"].shift(n) - 1) * 100
    return df


def bars_since_cross_up(df, lookback):
    """Return how many candles ago EMA21 crossed above EMA63 (0 = today), else None."""
    diff = df["ema21"] - df["ema63"]
    if diff.iloc[-1] <= 0:
        return None
    crossed = ((diff > 0) & (diff.shift(1) <= 0)).iloc[-lookback:]
    idx = np.where(crossed.values)[0]
    if len(idx) == 0:
        return None
    return int(len(crossed) - 1 - idx[-1])


def detect_vcp(df, lookback=120, order=5, min_contractions=2,
               last_max_depth=12.0, pivot_dist=8.0):
    """
    Volatility Contraction Pattern detector.
    - Finds swing highs (local max over +/- `order` candles).
    - For each swing high, depth = drop to the lowest low before the next swing high.
    - Needs depths to shrink contraction after contraction.
    """
    d = df.iloc[-lookback:]
    high, low, close, vol = d["High"].values, d["Low"].values, d["Close"].values, d["Volume"].values
    n = len(d)

    swing_highs = []
    for i in range(order, n - order):
        if high[i] == high[i - order:i + order + 1].max():
            swing_highs.append(i)
    if len(swing_highs) < min_contractions:
        return None

    contractions = []
    for k, hi in enumerate(swing_highs):
        end = swing_highs[k + 1] if k + 1 < len(swing_highs) else n
        seg_low = low[hi:end].min()
        depth = (1 - seg_low / high[hi]) * 100
        contractions.append((hi, depth))

    # take the longest run of shrinking depths that ends at the latest contraction
    run = [contractions[-1]]
    for c in reversed(contractions[:-1]):
        if run[0][1] < c[1]:       # earlier pullback was deeper -> still contracting
            run.insert(0, c)
        else:
            break
    run = run[-4:]                  # classic VCPs have 2-4 contractions
    if len(run) < min_contractions:
        return None

    depths = [round(x[1], 1) for x in run]
    if depths[0] > 40 or depths[-1] > last_max_depth:
        return None

    pivot = max(high[run[0][0]:])                       # base high
    dist = (pivot / close[-1] - 1) * 100
    if dist < 0 or dist > pivot_dist:                   # must sit just below the pivot
        return None

    if vol[-10:].mean() >= vol[-50:].mean():            # volume dry-up
        return None

    return {"contractions": depths, "pivot": round(float(pivot), 2),
            "dist_to_pivot": round(float(dist), 1)}


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------
def scan(cross_bars, roc_tol, min_contractions, require_vcp):
    uni = load_universe()
    results = []

    for start in range(0, len(uni), BATCH):
        chunk = uni.iloc[start:start + BATCH]
        tickers = [f"{s}.NS" for s in chunk["symbol"]]
        try:
            data = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue

        for sym, cap in zip(chunk["symbol"], chunk["cap"]):
            try:
                df = data[f"{sym}.NS"] if len(tickers) > 1 else data
                df = df.dropna(subset=["Close", "High", "Low", "Volume"])
                if len(df) < 130:
                    continue
                df = add_indicators(df, cap)

                since = bars_since_cross_up(df, cross_bars)
                if since is None:
                    continue

                roc = float(df["roc"].iloc[-1])
                if not abs(roc) <= roc_tol:
                    continue

                vcp = detect_vcp(df, min_contractions=min_contractions)
                if require_vcp and vcp is None:
                    continue

                results.append({
                    "symbol": sym,
                    "cap": cap,
                    "close": round(float(df["Close"].iloc[-1]), 2),
                    "roc_len": roc_length(cap),
                    "roc": round(roc, 2),
                    "ema21": round(float(df["ema21"].iloc[-1]), 2),
                    "ema63": round(float(df["ema63"].iloc[-1]), 2),
                    "crossed_days_ago": since,
                    "vcp": vcp,
                })
            except Exception:
                continue

    results.sort(key=lambda r: (r["crossed_days_ago"], abs(r["roc"])))
    return results


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/scan")
def api_scan():
    cross_bars = int(request.args.get("cross_bars", 3))
    roc_tol = float(request.args.get("roc_tol", 3))
    min_c = int(request.args.get("min_contractions", 2))
    require_vcp = request.args.get("require_vcp", "1") == "1"

    key = (cross_bars, roc_tol, min_c, require_vcp)
    hit = CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return jsonify({"cached": True, "results": hit[1]})

    results = scan(cross_bars, roc_tol, min_c, require_vcp)
    CACHE[key] = (time.time(), results)
    return jsonify({"cached": False, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
