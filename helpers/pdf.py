# helpers/nifty.py
from nsetools import Nse
from collections import defaultdict

def fetch_nse_nifty50_sector_weights():
    """
    Fetch Nifty 50 holdings with sector weights using nsetools.
    Returns a dictionary: { sector_name: weight_percent }
    """
    try:
        nse = Nse()
        nifty50 = nse.get_index_constituents('NIFTY 50')  # returns {symbol: {data}}

        if not nifty50:
            print("⚠️ Failed to fetch Nifty 50 constituents")
            return {}

        sector_weights = defaultdict(float)
        for sym, info in nifty50.items():
            industry = info.get('industry', 'NA')
            weight = info.get('weight', 0)
            sector_weights[industry] += weight

        return dict(sector_weights)

    except Exception as e:
        print(f"⚠️ NSE fetch failed: {e}")
        return {}
