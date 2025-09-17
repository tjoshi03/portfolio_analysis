from flask import Blueprint, render_template, request
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from helpers.stock_helpers import (
    format_symbols_auto, latest_price_auto, safe_info, fetch_news,
    fetch_key_data, fetch_corporate_actions, sma, rsi, normalize_series, compute_risk_metrics
)
from ai.groq_helpers import generate_ai_summary, generate_ai_rebalance, generate_ai_suggestions_from_rebalance

portfolio_bp = Blueprint("portfolio", __name__)

# -------------------------
# Helper functions for route
# -------------------------

def process_holding(h):
    """Process individual holding and fetch all relevant data."""
    raw = (h.get("symbol") or h.get("Stock") or "").strip()
    if raw == "":
        return None

    symbols_to_try = format_symbols_auto(raw)
    price, used_symbol = latest_price_auto(symbols_to_try)
    t = yf.Ticker(used_symbol)
    info = safe_info(t)

    qty = float(h.get("quantity", h.get("Qty", 0)) or 0.0)
    buy = float(h.get("buy", h.get("BuyPrice", 0)) or 0.0)

    name = info.get("shortName") or info.get("longName") or raw.upper()
    sector = info.get("sector") or info.get("industry") or "Unknown"
    value = price * qty
    invested = buy * qty
    pnl = value - invested
    pnl_pct = (pnl / invested * 100.0) if invested else 0.0
    news = fetch_news(t, limit=3)
    key_data = fetch_key_data(info)
    corp_actions = fetch_corporate_actions(t)

    # 52-week high/low
    try:
        hist_1y = t.history(period="1y")["Close"].dropna()
        w52_high = float(hist_1y.max()) if not hist_1y.empty else None
        w52_low = float(hist_1y.min()) if not hist_1y.empty else None
        dist = {
            "from_high_pct": round((price - w52_high)/w52_high*100, 2) if w52_high else None,
            "from_low_pct": round((price - w52_low)/w52_low*100, 2) if w52_low else None,
            "w52_high": round(w52_high,2) if w52_high else None,
            "w52_low": round(w52_low,2) if w52_low else None
        } if w52_high and w52_low else None
    except Exception:
        dist = None

    # Technical indicators
    try:
        hist = t.history(period="120d")["Close"].dropna()
        sma20 = float(sma(hist,20).iloc[-1]) if len(hist)>=20 else None
        sma50 = float(sma(hist,50).iloc[-1]) if len(hist)>=50 else None
        rsi14 = float(rsi(hist,14).iloc[-1]) if len(hist)>=15 else None
        trend = "Neutral"
        if sma20 and sma50:
            trend = "Bullish" if sma20 > sma50 else "Bearish"
    except Exception:
        sma20=sma50=rsi14=trend=None

    return {
        "symbol": used_symbol.split('.')[0],
        "ticker": used_symbol,
        "name": name,
        "Sector": sector,
        "quantity": qty,
        "buy": round(buy, 2),
        "price": round(price, 2),
        "invested": round(invested, 2),
        "value": round(value, 2),
        "pnl": round(pnl, 2),
        "pnl_percent": round(pnl_pct, 2),
        "news": news,
        "key_data": key_data,
        "w52": dist,
        "corporate_actions": corp_actions,
        "technical": {
            "sma20": round(sma20,2) if sma20 else None,
            "sma50": round(sma50,2) if sma50 else None,
            "rsi14": round(rsi14,2) if rsi14 else None,
            "trend": trend
        }
    }

def compute_portfolio_metrics(rows):
    """Compute weights, top contributors, sector allocation, portfolio series & risk."""
    total_value = sum(r["value"] for r in rows)
    total_invested = sum(r["invested"] for r in rows)
    for r in rows:
        r["weight_percent"] = round((r["value"] / total_value * 100.0) if total_value else 0.0, 2)
        r["pnl_contribution"] = round((r["pnl"] / total_value * 100.0) if total_value else 0.0, 2)

    top_contrib = sorted(rows, key=lambda x: abs(x["pnl_contribution"]), reverse=True)[:5]
    top_contrib_brief = [{"symbol": c["symbol"], "pnl": c["pnl"], "pnl_contribution": c["pnl_contribution"]} for c in top_contrib]

    # Sector allocation
    sector_vals = {}
    for r in rows:
        sector_vals[r["Sector"]] = sector_vals.get(r["Sector"], 0.0) + r["value"]
    sector_alloc = {s: round((v / total_value * 100.0) if total_value else 0.0, 2) for s, v in sector_vals.items()}

    return total_value, total_invested, top_contrib_brief, sector_alloc

# -------------------------
# Routes
# -------------------------

@portfolio_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    data = request.json or {}
    holdings = data.get("holdings", [])
    if not holdings:
        return {"error": "No holdings provided"}, 400

    # --- Process holdings ---
    rows = []
    symbols_for_history = []
    for h in holdings:
        r = process_holding(h)
        if r:
            rows.append(r)
            symbols_for_history.append(r["ticker"])

    # --- Portfolio metrics ---
    total_value, total_invested, top_contrib, sector_alloc = compute_portfolio_metrics(rows)

    # --- Historical series for risk ---
    end = datetime.today()
    start_1y = end - timedelta(days=365)
    hist_df = pd.DataFrame()
    for sym in symbols_for_history:
        try:
            s = yf.Ticker(sym).history(start=start_1y, end=end)["Close"].rename(sym)
            if not s.empty:
                hist_df = pd.concat([hist_df, s], axis=1)
        except Exception:
            continue

    try:
        nifty_hist = yf.Ticker("^NSEI").history(start=start_1y, end=end)["Close"].dropna()
    except Exception:
        nifty_hist = pd.Series(dtype=float)

    if not hist_df.empty and not nifty_hist.empty:
        all_dates = nifty_hist.index.union(hist_df.index)
        hist_df = hist_df.reindex(all_dates).ffill()
        nifty_hist = nifty_hist.reindex(all_dates).ffill()
        qty_map = {r["symbol"]: r["quantity"] for r in rows if r["symbol"] in hist_df.columns}
        port_series = (hist_df[list(qty_map.keys())] * pd.Series(qty_map)).sum(axis=1)
    else:
        port_series = pd.Series(dtype=float)

    perf_labels, portfolio_series = normalize_series(port_series)
    _, nifty_series = normalize_series(nifty_hist)

    risk_metrics = compute_risk_metrics(port_series, nifty_hist) if (not port_series.empty and not nifty_hist.empty) else {}

    total_pnl = total_value - total_invested
    total_pnl_percent = (total_pnl / total_invested * 100.0) if total_invested else 0.0

    # --- AI analysis ---
    ai_summary = generate_ai_summary(total_pnl_percent, rows, risk_metrics, nifty_hist.iloc[-1] if not nifty_hist.empty else None)
    rebalance_table = generate_ai_rebalance(rows, total_value, risk_metrics, nifty_hist.iloc[-1] if not nifty_hist.empty else None)
    ai_suggestions = generate_ai_suggestions_from_rebalance(rebalance_table)

    return {
        "summary": {
            "total_invested": round(total_invested, 2),
            "total_current": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_percent": round(total_pnl_percent, 2),
            "nifty_value": round(float(nifty_hist.iloc[-1]), 2) if not nifty_hist.empty else None
        },
        "holdings": rows,
        "top_contributors": top_contrib,
        "sector_alloc": sector_alloc,
        "perf_labels": perf_labels,
        "portfolio_series": portfolio_series,
        "sector_labels": list(sector_alloc.keys()),
        "sector_values": list(sector_alloc.values()),
        "holding_labels": [r["symbol"] for r in rows],
        "holding_values": [r["weight_percent"] for r in rows],
        "risk_metrics": risk_metrics,
        "nifty_series": nifty_series,
        "ai_suggestions": ai_suggestions,
        "ai_summary": ai_summary,
        "rebalance": rebalance_table
    }
