from groq import Groq
import re
import json
from helpers.pdf import fetch_nse_nifty50_sector_weights
from collections import defaultdict

GROQ_API_KEY = "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"
groq = Groq(api_key=GROQ_API_KEY)

def generate_ai_rebalance(rows, total_value, risk_metrics, nifty_value):
    nifty_sector_weights = fetch_nse_nifty50_sector_weights()

    holdings_txt = "\n".join(
        f"{r['symbol']} | {r.get('Sector','NA')} | qty:{r['quantity']} | value:{r['value']} | weight:{r['weight_percent']}%%"
        for r in rows
    )

    prompt = f"""
You are an expert Indian equities analyst.

Portfolio summary:
- Holdings:
{holdings_txt}
- Total portfolio value: {total_value}
- Risk metrics: {risk_metrics}
- NIFTY value: {nifty_value}

Provide a **rebalance plan** in **STRICT JSON format ONLY**.

Rules:
- Must return a list of objects.
- Each object must have:
  - "symbol" (string)
  - "current_weight" (float)
  - "target_weight" (float)
  - "action" ("Increase" | "Decrease" | "Hold" | "Add New")
  - "reason" (string)
"""

    models_priority = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    rebalance = None
    last_error = None

    for model_name in models_priority:
        try:
            resp = groq.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2
            )

            raw_content = resp.choices[0].message.content.strip()
            match = re.search(r'\{.*\}|\[.*\]', raw_content, re.DOTALL)
            if match:
                raw_content = match.group(0)

            rebalance = json.loads(raw_content)
            if isinstance(rebalance, list):
                break
            else:
                raise ValueError("Not a JSON list")
        except Exception as e:
            last_error = e
            print(f"⚠️ Model {model_name} failed: {e}")
            continue

    if not rebalance:
        print("AI rebalance error, fallback:", last_error)
        rebalance = [
            {
                "symbol": r["symbol"],
                "current_weight": r["weight_percent"],
                "target_weight": r["weight_percent"],
                "action": "Hold",
                "reason": "AI unavailable, maintaining current allocation"
            }
            for r in rows
        ]

    # Sector reasoning fallback
    portfolio_sector_weights = defaultdict(float)
    for r in rows:
        portfolio_sector_weights[r.get("Sector", "NA")] += r["weight_percent"]

    symbol_to_sector = {r["symbol"]: r.get("Sector", "NA") for r in rows}

    overweight, underweight = [], []
    if nifty_sector_weights:
        for sector, p_wt in portfolio_sector_weights.items():
            b_wt = nifty_sector_weights.get(sector)
            if b_wt is None:
                continue
            if p_wt > b_wt * 1.1:
                overweight.append(sector)
            elif p_wt < b_wt * 0.9:
                underweight.append(sector)

    for r in rebalance:
        if "reason" not in r or not r["reason"].strip():
            sym = r["symbol"]
            sector = symbol_to_sector.get(sym, "NA")
            port_wt = portfolio_sector_weights.get(sector, 0)
            bench_wt = nifty_sector_weights.get(sector) if nifty_sector_weights else None
            action = r.get("action", "Hold")

            if action == "Decrease":
                if sector in overweight and bench_wt:
                    r["reason"] = (
                        f"Reduce {sym} since {sector} is overweight "
                        f"({port_wt:.2f}% vs NIFTY {bench_wt:.2f}%). "
                        f"This frees allocation for underweight sectors like "
                        f"{', '.join(underweight) if underweight else 'other areas'}."
                    )
                else:
                    r["reason"] = f"Trim {sym} to reduce concentration risk in {sector}."
            elif action in ["Increase", "Add New"]:
                if sector in underweight and bench_wt:
                    r["reason"] = (
                        f"Add/Increase {sym} since {sector} is underweight "
                        f"({port_wt:.2f}% vs NIFTY {bench_wt:.2f}%), "
                        f"helping diversify the portfolio."
                    )
                else:
                    r["reason"] = f"Increase {sym} to enhance exposure in {sector}."
            else:
                r["reason"] = f"Maintain {sym} allocation in {sector} sector."

    return rebalance

def generate_ai_suggestions_from_rebalance(rebalance):
    suggestions = []
    for r in rebalance:
        if r["action"] != "Hold":
            suggestions.append(
                f"{r['action']} allocation to {r['symbol']} (target {r['target_weight']}%) — {r['reason']}"
            )
        if len(suggestions) >= 3:
            break
    return suggestions

def generate_ai_summary(total_pnl_percent, rows, risk_metrics, nifty_value):
    holdings_txt = "\n".join(
        f"{r['symbol']} | {r['Sector']} | qty:{r['quantity']} | value:{r['value']}"
        for r in rows
    )
    prompt = f"""
You are a financial analyst. Summarize this portfolio in 4-5 sentences:
- Overall performance and health
- Risk profile based on metrics
- Sector diversification
- Comparison with NIFTY

Portfolio PnL %: {total_pnl_percent}
NIFTY Value: {nifty_value}
Holdings:
{holdings_txt}
Risk metrics: {risk_metrics}
"""
    try:
        resp = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user", "content":prompt}],
            max_tokens=200,
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("Groq AI summary error:", e)
        return (
            f"Portfolio return {total_pnl_percent:.2f}%. "
            f"Sharpe={risk_metrics.get('Sharpe')}, Beta={risk_metrics.get('Beta')}. "
            "Diversify holdings to reduce risk."
        )
