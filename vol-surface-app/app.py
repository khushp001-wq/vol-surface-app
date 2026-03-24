"""
Vol Surface App — Flask backend
Fetches live options via yfinance, computes BS implied vol, returns surface data.
"""

from flask import Flask, jsonify, request, render_template
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import pandas as pd
from datetime import datetime, date, timezone
import math
import time
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Simple in-memory cache ─────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 60  # seconds


def cache_get(key):
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < CACHE_TTL:
        return entry["data"]
    return None


def cache_set(key, data):
    _cache[key] = {"data": data, "ts": time.time()}


# ── Black-Scholes ──────────────────────────────────────────────────────────
def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "call") -> float:
    if T <= 1e-8 or sigma <= 1e-8:
        return max(0.0, S - K) if opt_type == "call" else max(0.0, K - S)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_iv(market_price: float, S: float, K: float, T: float, r: float, opt_type: str = "call"):
    """Return implied vol (decimal) or None."""
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return None
    try:
        intrinsic = max(0.0, S - K) if opt_type == "call" else max(0.0, K - S)
        if market_price < intrinsic * 0.999:
            return None

        def obj(sig):
            return bs_price(S, K, T, r, sig, opt_type) - market_price

        lo_val, hi_val = obj(1e-5), obj(8.0)
        if lo_val * hi_val > 0:
            return None
        iv = brentq(obj, 1e-5, 8.0, xtol=1e-6, maxiter=500)
        return float(iv) if 0.002 < iv < 8.0 else None
    except Exception:
        return None


# ── Risk-free rate ─────────────────────────────────────────────────────────
_rf = {"rate": 0.045, "ts": 0}


def get_risk_free() -> float:
    if time.time() - _rf["ts"] < 3600:
        return _rf["rate"]
    try:
        hist = yf.Ticker("^IRX").history(period="1d")
        if not hist.empty:
            _rf["rate"] = float(hist["Close"].iloc[-1]) / 100
    except Exception:
        pass
    _rf["ts"] = time.time()
    return _rf["rate"]


# ── Helpers ────────────────────────────────────────────────────────────────
def get_spot(ticker_obj, info: dict):
    S = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    if not S:
        try:
            hist = ticker_obj.history(period="2d")
            S = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            pass
    return float(S) if S else None


def build_surface_grid(df: pd.DataFrame, x_col: str, nx: int = 60, ny: int = 40):
    """Interpolate scattered IV onto a regular grid."""
    x = df[x_col].values.astype(float)
    y = df["dte"].values.astype(float)
    z = df["iv"].values.astype(float)
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    xi_g, yi_g = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi_g, yi_g), method="cubic")
    zi_near = griddata((x, y), z, (xi_g, yi_g), method="nearest")
    zi = np.where(np.isnan(zi), zi_near, zi)
    zi = np.clip(zi, 1.0, 250.0)
    return {
        "x": np.round(xi, 4).tolist(),
        "y": np.round(yi, 1).tolist(),
        "z": np.round(zi, 2).tolist(),
    }


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info/<ticker>")
def api_info(ticker):
    """Return spot price, company name, and available expiry dates."""
    sym = ticker.upper().strip()
    cached = cache_get(f"info:{sym}")
    if cached:
        return jsonify(cached)
    try:
        t = yf.Ticker(sym)
        info = t.info
        S = get_spot(t, info)
        exps = t.options or []
        today = date.today()
        exps = [e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() > today]
        name = info.get("shortName") or info.get("longName") or sym
        result = {"symbol": sym, "name": name, "spot": S, "expirations": exps}
        cache_set(f"info:{sym}", result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/surface/<ticker>")
def api_surface(ticker):
    """Compute full implied vol surface for a ticker."""
    sym = ticker.upper().strip()
    cached = cache_get(f"surface:{sym}")
    if cached:
        return jsonify(cached)

    try:
        t = yf.Ticker(sym)
        info = t.info
        S = get_spot(t, info)
        if not S:
            return jsonify({"error": "Could not determine spot price"}), 400

        r = get_risk_free()
        today_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        today_d = date.today()
        exps = t.options or []
        exps = [e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() > today_d][:14]

        if not exps:
            return jsonify({"error": "No future expiration dates found"}), 404

        rows = []
        for exp_str in exps:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            days = (exp_dt - today_dt).days
            T = max(days, 1) / 365.0
            dte = max(days, 1)
            try:
                chain = t.option_chain(exp_str)
            except Exception:
                continue

            for df_leg, otype in [(chain.calls, "call"), (chain.puts, "put")]:
                for _, row in df_leg.iterrows():
                    try:
                        K = float(row["strike"])
                        mono = K / S
                        if mono < 0.50 or mono > 1.60:
                            continue

                        bid = float(row["bid"]) if pd.notna(row["bid"]) else 0.0
                        ask = float(row["ask"]) if pd.notna(row["ask"]) else 0.0
                        last = float(row["lastPrice"]) if pd.notna(row["lastPrice"]) else 0.0
                        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                        if mid <= 0:
                            continue

                        vol_cnt = int(row["volume"]) if pd.notna(row.get("volume")) else 0
                        oi = int(row["openInterest"]) if pd.notna(row.get("openInterest")) else 0

                        iv = calc_iv(mid, S, K, T, r, otype)
                        # Fallback to yfinance IV if BS fails
                        if iv is None and pd.notna(row.get("impliedVolatility")):
                            yf_iv = float(row["impliedVolatility"])
                            if 0.005 < yf_iv < 8.0:
                                iv = yf_iv

                        if iv and 0.005 < iv < 8.0:
                            rows.append({
                                "strike": round(K, 2),
                                "expiry": exp_str,
                                "dte": dte,
                                "T": round(T, 5),
                                "iv": round(iv * 100, 2),
                                "type": otype,
                                "moneyness": round(mono, 4),
                                "log_m": round(math.log(mono) * 100, 3),  # log-moneyness %
                                "volume": vol_cnt,
                                "oi": oi,
                                "mid": round(mid, 4),
                            })
                    except Exception:
                        continue

        if not rows:
            return jsonify({"error": "No valid IV data computed — market may be closed or no liquid options found"}), 404

        df_all = pd.DataFrame(rows)

        # OTM composite: calls where K ≥ S, puts where K ≤ S
        df_otm = df_all[
            ((df_all["type"] == "call") & (df_all["strike"] >= S)) |
            ((df_all["type"] == "put") & (df_all["strike"] <= S))
        ].copy()

        # Build interpolated grids
        grid_strike = grid_moneyness = None
        if len(df_otm) >= 6:
            try:
                grid_strike = build_surface_grid(df_otm, "strike", 60, 35)
            except Exception:
                pass
            try:
                grid_moneyness = build_surface_grid(df_otm, "moneyness", 60, 35)
            except Exception:
                pass

        # Term structure: ATM IV per expiry
        term_struct = []
        for exp_str in sorted(df_otm["expiry"].unique()):
            sub = df_otm[df_otm["expiry"] == exp_str]
            atm = sub.iloc[(sub["strike"] - S).abs().argsort()[:5]]
            if not atm.empty:
                term_struct.append({
                    "expiry": exp_str,
                    "dte": int(sub["dte"].iloc[0]),
                    "atm_iv": round(float(atm["iv"].mean()), 2),
                })

        # Vol smile per expiry (for the smile chart)
        smiles = {}
        for exp_str in df_otm["expiry"].unique():
            sub = df_otm[df_otm["expiry"] == exp_str].sort_values("strike")
            smiles[exp_str] = sub[["strike", "moneyness", "iv", "type"]].to_dict("records")

        result = {
            "symbol": sym,
            "spot": round(S, 2),
            "risk_free": round(r * 100, 3),
            "expirations": exps,
            "otm_data": df_otm.to_dict("records"),
            "grid_strike": grid_strike,
            "grid_moneyness": grid_moneyness,
            "term_structure": term_struct,
            "smiles": smiles,
        }
        cache_set(f"surface:{sym}", result)
        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("\n  📈  Vol Surface App  →  http://localhost:8080\n")
    app.run(debug=False, port=8080, host="0.0.0.0")
