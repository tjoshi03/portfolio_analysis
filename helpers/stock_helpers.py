import yfinance as yf
import feedparser
import pandas as pd
import numpy as np

# --- Stock Helpers ---

def format_symbols_auto(sym: str):
    s = (sym or "").strip().upper()
    if s.endswith(".NS") or s.endswith(".BO") or s.endswith(".NSI"):
        return [s]
    return [s + ".NS", s + ".BO"]

def safe_info(ticker):
    try:
        info = ticker.get_info() if hasattr(ticker, "get_info") else ticker.info
        if not isinstance(info, dict):
            info = {}
    except Exception:
        info = {}
    return info

def latest_price(ticker):
    try:
        info = safe_info(ticker)
        if info.get("currentPrice"):
            return float(info["currentPrice"])
    except Exception:
        pass
    try:
        hist = ticker.history(period="5d")["Close"].dropna()
        if not hist.empty:
            return float(hist.iloc[-1])
    except Exception:
        pass
    try:
        fi = getattr(ticker, "fast_info", {}) or {}
        if isinstance(fi, dict) and fi.get("last_price") is not None:
            return float(fi["last_price"])
    except Exception:
        pass
    return 0.0

def latest_price_auto(symbols):
    for sym in symbols:
        t = yf.Ticker(sym)
        price = latest_price(t)
        if price > 0:
            return price, sym
    return 0.0, symbols[0]

def fetch_news_google(symbol="TCS", limit=3):
    import urllib.parse
    query = f"{symbol} stock India"
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    news_items = [{"title": e.title, "link": e.link} for e in feed.entries[:limit]]
    return news_items

def fetch_news(ticker, limit=3):
    sym = ticker.ticker.split(".")[0]
    return fetch_news_google(sym, limit=limit)

def fetch_key_data(info):
    return {
        "MarketCap": info.get("marketCap"),
        "PE": info.get("trailingPE"),
        "PB": info.get("priceToBook"),
        "DividendYield": info.get("dividendYield"),
        "ROE": info.get("returnOnEquity"),
    }

def fetch_corporate_actions(ticker, limit=3):
    actions = []
    try:
        div = ticker.dividends.tail(limit)
        for date, val in div.items():
            actions.append({"type": "Dividend", "date": str(date.date()), "value": f"₹{val:.2f}"})
    except Exception:
        pass
    try:
        splits = ticker.splits.tail(limit)
        for date, val in splits.items():
            actions.append({"type": "Split", "date": str(date.date()), "value": f"{val}:1"})
    except Exception:
        pass
    return actions

# --- Technicals ---

def sma(series, window):
    return series.rolling(window=window).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def normalize_series(series):
    if series is None or series.empty:
        return [], []
    series = series.dropna()
    if series.empty:
        return [], []
    base = series.iloc[0]
    vals = (series / base * 100.0).round(2).tolist() if base != 0 else [100.0] * len(series)
    dates = [d.strftime("%Y-%m-%d") for d in series.index]
    return dates, vals

def compute_risk_metrics(port_series, bench_series):
    rf = 0.065
    if port_series.empty or bench_series.empty:
        return {"Volatility": None, "Beta": None, "Sharpe": None}
    pr = port_series.pct_change().dropna()
    br = bench_series.pct_change().dropna()
    joint = pd.concat([pr, br], axis=1).dropna()
    if joint.shape[0] < 2:
        return {"Volatility": None, "Beta": None, "Sharpe": None}
    port_ret = joint.iloc[:, 0]
    bench_ret = joint.iloc[:, 1]
    vol = port_ret.std() * np.sqrt(252)
    ann_ret = port_ret.mean() * 252
    beta = float(np.cov(port_ret, bench_ret)[0, 1] / bench_ret.var()) if bench_ret.var() != 0 else None
    sharpe = (ann_ret - rf) / vol if vol and vol != 0 else None
    return {
        "Volatility": round(vol, 4) if vol else None,
        "Beta": round(beta, 4) if beta else None,
        "Sharpe": round(sharpe, 4) if sharpe else None
    }
