"""
VCP stock screener website (works on mobile and laptop).

A stock is suggested when ALL of these are true:
  1. EMA(21) crossed ABOVE EMA(63) within the last `cross_bars` candles
  2. ROC is near zero: |ROC| <= roc_tol (%)
       ROC length = 18 for small caps, 20 for mid and large caps
  3. Price is at or within `high_tol` % of its 52-week high
  4. A Minervini-style VCP is visible (shrinking pullbacks, tight last pullback,
     price just below the pivot, volume drying up)

Run locally:   pip install -r requirements.txt   then   python app.py
Run online:    gunicorn app:app --workers 1 --threads 4 --timeout 120
(Use ONE worker: scan progress is kept in memory.)

Stock list: edit DEFAULT_UNIVERSE below, or put a universe.csv next to this file with columns
symbol,cap and (optional) listed  -> NSE listing date such as 05-SEP-2025.
Without a `listed` column, the first available price candle is used as the listing date.
"""
import io
import os
import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

ROC_TOL = 3.0        # ROC must be within +/- this many % of zero (change here)
CACHE_TTL = 30 * 60   # seconds a finished scan stays valid
BATCH = 40            # tickers per Yahoo download
JOBS = {}             # scan key -> job dict
LOCK = threading.Lock()

DEFAULT_UNIVERSE = '''symbol,cap
RELIANCE,large
TCS,large
HDFCBANK,large
INFY,large
ICICIBANK,large
LT,large
ITC,large
TATAMOTORS,large
ASIANPAINT,large
TITAN,large
POLYCAB,mid
PERSISTENT,mid
TRENT,mid
CUMMINSIND,mid
ASTRAL,mid
AIAENG,mid
KPITTECH,mid
CDSL,small
CAMPUS,small
RADICO,small
RATNAMANI,small
SUPRAJIT,small
JKPAPER,small
GRSE,small
'''


# --------------------------------------------------------------------------
# Universe and indicators
# --------------------------------------------------------------------------
def load_universe():
    if os.path.exists("universe.csv"):
        df = pd.read_csv("universe.csv")
    else:
        df = pd.read_csv(io.StringIO(DEFAULT_UNIVERSE))
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["cap"] = df["cap"].fillna("small").astype(str).str.strip().str.lower()
    if "listed" in df.columns:
        df["listed"] = pd.to_datetime(df["listed"], errors="coerce", dayfirst=True)
    return df


def roc_length(cap):
    return 18 if cap == "small" else 20  # mid and large -> 20


def add_indicators(df, cap):
    n = roc_length(cap)
    df = df.copy()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema63"] = df["Close"].ewm(span=63, adjust=False).mean()
    df["roc"] = (df["Close"] / df["Close"].shift(n) - 1) * 100
    return df


def bars_since_cross_up(df, lookback):
    """Candles since EMA21 crossed above EMA63 (0 = today); None if not recent."""
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
    d = df.iloc[-lookback:]
    high, low, close, vol = d["High"].values, d["Low"].values, d["Close"].values, d["Volume"].values
    n = len(d)
    if n < 2 * order + 5:
        return None

    swing_highs = [i for i in range(order, n - order)
                   if high[i] == high[i - order:i + order + 1].max()]
    if len(swing_highs) < min_contractions:
        return None

    contractions = []
    for k, hi in enumerate(swing_highs):
        end = swing_highs[k + 1] if k + 1 < len(swing_highs) else n
        depth = (1 - low[hi:end].min() / high[hi]) * 100
        contractions.append((hi, depth))

    run = [contractions[-1]]
    for c in reversed(contractions[:-1]):
        if run[0][1] < c[1]:
            run.insert(0, c)
        else:
            break
    run = run[-4:]
    if len(run) < min_contractions:
        return None

    depths = [round(x[1], 1) for x in run]
    if depths[0] > 40 or depths[-1] > last_max_depth:
        return None

    pivot = max(high[run[0][0]:])
    dist = (pivot / close[-1] - 1) * 100
    if dist < 0 or dist > pivot_dist:
        return None

    if vol[-10:].mean() >= vol[-50:].mean():
        return None

    return {"contractions": depths, "pivot": round(float(pivot), 2),
            "dist_to_pivot": round(float(dist), 1)}


def get_frame(data, key):
    if isinstance(data.columns, pd.MultiIndex):
        return data[key] if key in data.columns.get_level_values(0) else None
    return data


# --------------------------------------------------------------------------
# Background scan
# --------------------------------------------------------------------------
def run_scan(job, cross_bars, roc_tol, high_tol, min_contractions, require_vcp, listed_months=0):
    try:
        uni = load_universe()
        job["total"] = len(uni)
        results = []
        # recent-listing mode needs fewer candles; otherwise want ~6 months of history
        min_bars = 30 if listed_months else 130
        cutoff = (pd.Timestamp.now().normalize() - pd.Timedelta(days=int(listed_months * 30.5))
                  if listed_months else None)

        for start in range(0, len(uni), BATCH):
            chunk = uni.iloc[start:start + BATCH]
            tickers = [f"{s}.NS" for s in chunk["symbol"]]
            try:
                data = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                                   auto_adjust=True, progress=False, threads=True)
            except Exception:
                job["done"] += len(chunk)
                continue

            for row in chunk.itertuples():
                sym, cap = row.symbol, row.cap
                listed_csv = getattr(row, "listed", None)
                job["done"] += 1
                try:
                    df = get_frame(data, f"{sym}.NS")
                    if df is None:
                        continue
                    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
                    if len(df) < min_bars:
                        continue

                    first = pd.Timestamp(df.index[0])
                    if first.tzinfo:
                        first = first.tz_localize(None)
                    listed = first if (listed_csv is None or pd.isna(listed_csv)) else listed_csv
                    if cutoff is not None and listed < cutoff:
                        continue

                    df = add_indicators(df, cap)

                    since = bars_since_cross_up(df, cross_bars)
                    if since is None:
                        continue
                    roc = float(df["roc"].iloc[-1])
                    if not abs(roc) <= roc_tol:
                        continue
                    high52 = float(df["High"].iloc[-252:].max())
                    close_now = float(df["Close"].iloc[-1])
                    from_high = (1 - close_now / high52) * 100   # 0 = at the high
                    if from_high > high_tol:
                        continue

                    vcp = detect_vcp(df, min_contractions=min_contractions)
                    if require_vcp and vcp is None:
                        continue

                    results.append({
                        "symbol": sym, "cap": cap,
                        "close": round(float(df["Close"].iloc[-1]), 2),
                        "roc_len": roc_length(cap), "roc": round(roc, 2),
                        "ema21": round(float(df["ema21"].iloc[-1]), 2),
                        "ema63": round(float(df["ema63"].iloc[-1]), 2),
                        "high52": round(high52, 2), "from_high": round(max(from_high, 0), 1),
                        "listed": listed.strftime("%d %b %Y"),
                        "crossed_days_ago": since, "vcp": vcp,
                    })
                except Exception:
                    continue

        results.sort(key=lambda r: (r["crossed_days_ago"], r["from_high"], abs(r["roc"])))
        job["results"] = results
        job["status"] = "done"
    except Exception as e:  # keep the site alive whatever happens
        job["status"] = "error"
        job["error"] = str(e)
    job["ts"] = time.time()


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0f6b4a">
<title>VCP Screener</title>
<style>
  :root {
    --bg:#f3f5f2; --surface:#fff; --ink:#17211b; --muted:#5f6b63; --line:#d9dfd9;
    --accent:#0f6b4a; --accent-ink:#fff; --warn:#a4471b;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#101613; --surface:#18201b; --ink:#e6ede8; --muted:#94a39a; --line:#2a352e;
            --accent:#4fc08d; --accent-ink:#0d1a13; --warn:#e0925f; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  main { max-width:1040px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:26px; line-height:1.2; margin:0 0 6px; letter-spacing:-.01em; }
  .sub { color:var(--muted); margin:0 0 20px; max-width:70ch; }
  form { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; align-items:end;
         background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:20px; }
  label { display:flex; flex-direction:column; gap:4px; font-size:13px; color:var(--muted); }
  input[type=number], select { width:100%; padding:9px; font:inherit; color:var(--ink);
         background:var(--bg); border:1px solid var(--line); border-radius:6px; }
  .check { flex-direction:row; align-items:center; gap:8px; color:var(--ink); padding-bottom:9px; }
  button { font:600 15px system-ui,sans-serif; padding:11px 18px; border:0; border-radius:6px;
           background:var(--accent); color:var(--accent-ink); cursor:pointer; width:100%; }
  button:disabled { opacity:.6; cursor:wait; }
  :focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  #status { color:var(--muted); margin:0 0 10px; min-height:22px; }
  #status.error { color:var(--warn); }
  .bar { height:6px; background:var(--line); border-radius:99px; overflow:hidden; margin-bottom:14px; }
  .bar > i { display:block; height:100%; width:0; background:var(--accent); transition:width .4s; }
  .wrap { background:var(--surface); border:1px solid var(--line); border-radius:10px; overflow-x:auto; }
  table { width:100%; border-collapse:collapse; }
  th, td { padding:11px 14px; text-align:left; border-bottom:1px solid var(--line); }
  th { font-weight:600; font-size:13px; color:var(--muted); }
  tr:last-child td { border-bottom:0; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  a { color:var(--accent); font-weight:600; text-decoration:none; }
  .tag { font-size:12px; padding:2px 8px; border:1px solid var(--line); border-radius:99px; color:var(--muted); }
  .note { color:var(--muted); font-size:13px; margin-top:20px; max-width:75ch; }

  @media (max-width:720px) {
    thead { display:none; }
    table, tbody, tr, td { display:block; width:100%; }
    tr { padding:10px 14px; border-bottom:1px solid var(--line); }
    tr:last-child { border-bottom:0; }
    td { display:flex; justify-content:space-between; gap:12px; padding:4px 0; border:0; text-align:right; }
    td::before { content:attr(data-label); color:var(--muted); font-size:13px; text-align:left; }
    td.sym { font-size:17px; padding-bottom:6px; }
  }
</style>
</head>
<body>
<main>
  <h1>VCP screener</h1>
  <p class="sub">Stocks near their 52-week high where the 21 EMA just crossed above the 63 EMA, ROC sits near zero, and price is tightening into a Minervini-style volatility contraction pattern.</p>

  <form id="f">
    <label>Crossed within (candles)
      <input type="number" id="cross_bars" value="3" min="1" max="20">
    </label>
    <label>Max % below 52-week high
      <input type="number" id="high_tol" value="5" min="0" step="0.5">
    </label>
    <label>Listing age on NSE
      <select id="listed_months">
        <option value="0">Any</option>
        <option value="1">Listed in last 1 month</option>
        <option value="3">Listed in last 3 months</option>
        <option value="6">Listed in last 6 months</option>
      </select>
    </label>
    <label>Min contractions
      <input type="number" id="min_contractions" value="2" min="2" max="4">
    </label>
    <label class="check"><input type="checkbox" id="require_vcp" checked> Require VCP</label>
    <button id="go" type="submit">Run scan</button>
  </form>

  <p id="status" role="status">Set your filters and run the scan.</p>
  <div class="bar" id="bar" hidden><i id="barfill"></i></div>

  <div class="wrap" id="tablewrap" hidden>
    <table>
      <thead>
        <tr>
          <th>Stock</th><th>Cap</th><th>Listed</th><th class="num">Close</th>
          <th class="num">52-wk high</th><th class="num">ROC (len)</th><th class="num">EMA 21 / 63</th>
          <th class="num">Crossed</th><th>Contractions (%)</th><th class="num">Pivot</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>

  <p class="note">ROC length is 18 for small caps and 20 for mid and large caps. Data comes from Yahoo Finance daily candles and may lag. This is a screening aid, not investment advice.</p>
</main>

<script>
const $ = id => document.getElementById(id);
const capLabel = { small: "Small", mid: "Mid", large: "Large" };
let timer = null;

$("f").addEventListener("submit", e => { e.preventDefault(); start(); });

function start() {
  clearTimeout(timer);
  $("go").disabled = true;
  $("tablewrap").hidden = true;
  $("status").className = "";
  $("status").textContent = "Starting scan…";
  $("bar").hidden = false;
  $("barfill").style.width = "0";
  poll();
}

async function poll() {
  const q = new URLSearchParams({
    cross_bars: $("cross_bars").value,
    high_tol: $("high_tol").value,
    listed_months: $("listed_months").value,
    min_contractions: $("min_contractions").value,
    require_vcp: $("require_vcp").checked ? "1" : "0",
  });
  try {
    const res = await fetch("/api/scan?" + q);
    if (!res.ok) throw new Error("Server returned " + res.status);
    const j = await res.json();

    if (j.status === "running") {
      const pct = j.total ? Math.round(100 * j.done / j.total) : 0;
      $("status").textContent = `Scanning… ${j.done} of ${j.total || "?"} stocks`;
      $("barfill").style.width = pct + "%";
      timer = setTimeout(poll, 2000);
      return;
    }
    if (j.status === "error") throw new Error(j.error || "scan failed");
    finish(j.results);
  } catch (err) {
    $("status").className = "error";
    $("status").textContent = "Scan failed: " + err.message + ". Check your connection and try again.";
    $("bar").hidden = true;
    $("go").disabled = false;
  }
}

function finish(list) {
  $("bar").hidden = true;
  $("go").disabled = false;
  const rows = $("rows");
  rows.replaceChildren();
  if (!list.length) {
    $("status").textContent = "No matches. Try more candles since the cross, raise “Max % below 52-week high”, set Listing age to Any, or untick “Require VCP”.";
    return;
  }
  $("status").textContent = list.length + " stock" + (list.length > 1 ? "s" : "") + " match.";
  for (const r of list) {
    const tr = document.createElement("tr");
    const vcp = r.vcp ? r.vcp.contractions.join(" → ") : "–";
    const pivot = r.vcp ? r.vcp.pivot.toFixed(2) + " (" + r.vcp.dist_to_pivot + "% away)" : "–";
    const when = r.crossed_days_ago === 0 ? "Today" : r.crossed_days_ago + "d ago";
    tr.innerHTML =
      `<td class="sym" data-label="Stock"><a target="_blank" rel="noopener" href="https://www.tradingview.com/chart/?symbol=NSE:${encodeURIComponent(r.symbol)}">${r.symbol}</a></td>` +
      `<td data-label="Cap"><span class="tag">${capLabel[r.cap] || r.cap}</span></td>` +
      `<td data-label="Listed">${r.listed}</td>` +
      `<td class="num" data-label="Close">${r.close.toFixed(2)}</td>` +
      `<td class="num" data-label="52-wk high">${r.high52.toFixed(2)} (${r.from_high === 0 ? "at high" : r.from_high + "% below"})</td>` +
      `<td class="num" data-label="ROC (len)">${r.roc.toFixed(2)}% (${r.roc_len})</td>` +
      `<td class="num" data-label="EMA 21 / 63">${r.ema21.toFixed(2)} / ${r.ema63.toFixed(2)}</td>` +
      `<td class="num" data-label="Crossed">${when}</td>` +
      `<td data-label="Contractions">${vcp}</td>` +
      `<td class="num" data-label="Pivot">${pivot}</td>`;
    rows.appendChild(tr);
  }
  $("tablewrap").hidden = false;
}
</script>
</body>
</html>
'''


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/scan")
def api_scan():
    cross_bars = max(1, min(int(request.args.get("cross_bars", 3)), 20))
    roc_tol = ROC_TOL
    high_tol = max(0.0, float(request.args.get("high_tol", 5)))
    listed_months = max(0, min(int(request.args.get("listed_months", 0)), 12))
    min_c = max(2, min(int(request.args.get("min_contractions", 2)), 4))
    require_vcp = request.args.get("require_vcp", "1") == "1"
    key = (cross_bars, roc_tol, high_tol, min_c, require_vcp, listed_months)

    with LOCK:
        job = JOBS.get(key)
        stale = job and job["status"] != "running" and time.time() - job.get("ts", 0) > CACHE_TTL
        failed = job and job["status"] == "error"
        if job is None or stale or failed:
            job = {"status": "running", "done": 0, "total": 0, "results": [], "error": None}
            JOBS[key] = job
            threading.Thread(target=run_scan, daemon=True,
                             args=(job, cross_bars, roc_tol, high_tol, min_c, require_vcp, listed_months)).start()

    return jsonify({k: job[k] for k in ("status", "done", "total", "results", "error")})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
