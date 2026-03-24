"""
Vol Surface App — Flask backend
Fetches live options via yfinance, computes BS implied vol, GEX, Expected Move,
and Risk-Neutral PDF via Breeden-Litzenberger.
"""

from flask import Flask, jsonify, render_template
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata, CubicSpline
import pandas as pd
from datetime import datetime, date, timezone
import math
import time
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Cache ──────────────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 60


def cache_get(key):
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < CACHE_TTL:
        return entry["data"]
    return None


def cache_set(key, data):
    _cache[key] = {"data": data, "ts": time.time()}


# ── Black-Scholes core ─────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type="call"):
    if T <= 1e-8 or sigma <= 1e-8:
        return max(0.0, S - K) if opt_type == "call" else max(0.0, K - S)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if opt_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, r, sigma):
    """BS gamma — identical for calls and puts."""
    if T <= 1e-8 or sigma <= 1e-8:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return float(norm.pdf(d1) / (S * sigma * sqrt_T))


def calc_iv(market_price, S, K, T, r, opt_type="call"):
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


def get_risk_free():
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
def get_spot(ticker_obj, info):
    S = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    if not S:
        try:
            hist = ticker_obj.history(period="2d")
            S = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            pass
    return float(S) if S else None


def build_surface_grid(df, x_col, nx=60, ny=40):
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
    return {"x": np.round(xi, 4).tolist(), "y": np.round(yi, 1).tolist(), "z": np.round(zi, 2).tolist()}


# ── GEX by strike ──────────────────────────────────────────────────────────
def compute_gex(df_all, S, r):
    """
    Net Gamma Exposure by strike.
    Convention: calls = positive GEX, puts = negative GEX.
    Units: $ gamma per 1% spot move (scaled for readability).
    Positive GEX -> dealers long gamma -> price-stabilising.
    Negative GEX -> dealers short gamma -> price-amplifying (vol regime).
    """
    gex_map = {}
    for _, row in df_all.iterrows():
        K   = row["strike"]
        iv  = row["iv"] / 100.0
        T   = row["T"]
        oi  = row["oi"]
        opt = row["type"]
        if oi <= 0 or iv <= 0 or T <= 0:
            continue
        g = bs_gamma(S, K, T, r, iv)
        gex_val = g * oi * 100 * (S ** 2) / 100.0
        if K not in gex_map:
            gex_map[K] = {"call_gex": 0.0, "put_gex": 0.0}
        if opt == "call":
            gex_map[K]["call_gex"] += gex_val
        else:
            gex_map[K]["put_gex"] += gex_val

    gex_data = []
    for K in sorted(gex_map):
        cg = gex_map[K]["call_gex"]
        pg = gex_map[K]["put_gex"]
        net = cg - pg
        gex_data.append({
            "strike":   round(K, 2),
            "call_gex": round(cg / 1e6, 3),
            "put_gex":  round(-pg / 1e6, 3),
            "net_gex":  round(net / 1e6, 3),
        })

    gamma_flip = None
    nets = [d["net_gex"] for d in gex_data]
    strikes = [d["strike"] for d in gex_data]
    for i in range(len(nets) - 1):
        if nets[i] * nets[i + 1] <= 0:
            if abs(nets[i + 1] - nets[i]) > 1e-9:
                frac = abs(nets[i]) / abs(nets[i + 1] - nets[i])
                gamma_flip = round(strikes[i] + frac * (strikes[i + 1] - strikes[i]), 2)
            else:
                gamma_flip = strikes[i]
            break

    return gex_data, gamma_flip


# ── Expected Move ──────────────────────────────────────────────────────────
def compute_expected_moves(df_all, S, exps):
    """
    Expected move per expiry from ATM straddle pricing.
    EM = ATM_IV * sqrt(T) * sqrt(2/pi) * S
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    results = []
    for exp_str in exps:
        sub = df_all[df_all["expiry"] == exp_str]
        if sub.empty:
            continue
        T   = sub["T"].iloc[0]
        dte = int(sub["dte"].iloc[0])
        calls = sub[sub["type"] == "call"].copy()
        puts  = sub[sub["type"] == "put"].copy()
        if calls.empty or puts.empty:
            continue
        atm_call = calls.iloc[(calls["strike"] - S).abs().argsort()[:3]]
        atm_put  = puts.iloc[(puts["strike"] - S).abs().argsort()[:3]]
        atm_iv   = (atm_call["iv"].mean() + atm_put["iv"].mean()) / 2.0 / 100.0
        em_pct    = atm_iv * math.sqrt(T) * sqrt_2_over_pi * 100.0
        em_dollar = S * em_pct / 100.0
        up   = round(S * (1 + atm_iv * math.sqrt(T)), 2)
        down = round(S * (1 - atm_iv * math.sqrt(T)), 2)
        results.append({
            "expiry":    exp_str,
            "dte":       dte,
            "em_pct":    round(em_pct, 2),
            "em_dollar": round(em_dollar, 2),
            "atm_iv":    round(atm_iv * 100, 2),
            "up_1sd":    up,
            "down_1sd":  down,
        })
    return results


# ── Risk-Neutral PDF (Breeden-Litzenberger) ────────────────────────────────
def compute_rn_pdfs(df_all, S, r, exps, max_exps=6):
    """
    Extract the market implied probability distribution for each expiry.
    Uses Breeden-Litzenberger: f(K) = e^(rT) * d2C/dK2
    """
    pdfs = {}
    for exp_str in exps[:max_exps]:
        sub = df_all[(df_all["expiry"] == exp_str) & (df_all["type"] == "call")].copy()
        sub = sub.sort_values("strike").drop_duplicates("strike")
        T   = sub["T"].iloc[0] if not sub.empty else None
        if T is None or len(sub) < 6:
            continue
        try:
            strikes = sub["strike"].values.astype(float)
            ivs     = sub["iv"].values.astype(float) / 100.0
            call_px = np.array([bs_price(S, K, T, r, iv, "call")
                                 for K, iv in zip(strikes, ivs)])
            cs = CubicSpline(strikes, call_px, bc_type="natural")
            K_fine = np.linspace(strikes[0], strikes[-1], 300)
            d2C    = cs(K_fine, 2)
            pdf    = np.exp(r * T) * d2C
            pdf    = np.clip(pdf, 0.0, None)
            area = np.trapz(pdf, K_fine)
            if area > 1e-8:
                pdf = pdf / area
            cdf = np.cumsum(pdf) * (K_fine[1] - K_fine[0])
            pdfs[exp_str] = {
                "strikes": np.round(K_fine, 2).tolist(),
                "pdf":     np.round(pdf, 8).tolist(),
                "cdf":     np.round(np.clip(cdf, 0, 1), 6).tolist(),
                "peak":    round(float(K_fine[np.argmax(pdf)]), 2),
            }
        except Exception:
            continue
    return pdfs


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info/<ticker>")
def api_info(ticker):
    sym = ticker.upper().strip()
    cached = cache_get(f"info:{sym}")
    if cached:
        return jsonify(cached)
    try:
        t    = yf.Ticker(sym)
        info = t.info
        S    = get_spot(t, info)
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
    sym = ticker.upper().strip()
    cached = cache_get(f"surface:{sym}")
    if cached:
        return jsonify(cached)
    try:
        t    = yf.Ticker(sym)
        info = t.info
        S    = get_spot(t, info)
        if not S:
            return jsonify({"error": "Could not determine spot price"}), 400

        r        = get_risk_free()
        today_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        today_d  = date.today()
        exps     = t.options or []
        exps     = [e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() > today_d][:14]

        if not exps:
            return jsonify({"error": "No future expiration dates found"}), 404

        rows = []
        for exp_str in exps:
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            days   = (exp_dt - today_dt).days
            T      = max(days, 1) / 365.0
            dte    = max(days, 1)
            try:
                chain = t.option_chain(exp_str)
            except Exception:
                continue

            for df_leg, otype in [(chain.calls, "call"), (chain.puts, "put")]:
                for _, row in df_leg.iterrows():
                    try:
                        K    = float(row["strike"])
                        mono = K / S
                        if mono < 0.50 or mono > 1.60:
                            continue
                        bid  = float(row["bid"])  if pd.notna(row["bid"])  else 0.0
                        ask  = float(row["ask"])  if pd.notna(row["ask"])  else 0.0
                        last = float(row["lastPrice"]) if pd.notna(row["lastPrice"]) else 0.0
                        mid  = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                        if mid <= 0:
                            continue
                        vol_cnt = int(row["volume"])      if pd.notna(row.get("volume"))      else 0
                        oi      = int(row["openInterest"]) if pd.notna(row.get("openInterest")) else 0
                        iv = calc_iv(mid, S, K, T, r, otype)
                        if iv is None and pd.notna(row.get("impliedVolatility")):
                            yf_iv = float(row["impliedVolatility"])
                            if 0.005 < yf_iv < 8.0:
                                iv = yf_iv
                        if iv and 0.005 < iv < 8.0:
                            rows.append({
                                "strike":    round(K, 2),
                                "expiry":    exp_str,
                                "dte":       dte,
                                "T":         round(T, 5),
                                "iv":        round(iv * 100, 2),
                                "type":      otype,
                                "moneyness": round(mono, 4),
                                "log_m":     round(math.log(mono) * 100, 3),
                                "volume":    vol_cnt,
                                "oi":        oi,
                                "mid":       round(mid, 4),
                            })
                    except Exception:
                        continue

        if not rows:
            return jsonify({"error": "No valid IV data - market may be closed or no liquid options found"}), 404

        df_all = pd.DataFrame(rows)

        df_otm = df_all[
            ((df_all["type"] == "call") & (df_all["strike"] >= S)) |
            ((df_all["type"] == "put")  & (df_all["strike"] <= S))
        ].copy()

        grid_strike = grid_moneyness = None
        if len(df_otm) >= 6:
            try:
                grid_strike    = build_surface_grid(df_otm, "strike",    60, 35)
            except Exception:
                pass
            try:
                grid_moneyness = build_surface_grid(df_otm, "moneyness", 60, 35)
            except Exception:
                pass

        term_struct = []
        for exp_str in sorted(df_otm["expiry"].unique()):
            sub = df_otm[df_otm["expiry"] == exp_str]
            atm = sub.iloc[(sub["strike"] - S).abs().argsort()[:5]]
            if not atm.empty:
                term_struct.append({
                    "expiry": exp_str,
                    "dte":    int(sub["dte"].iloc[0]),
                    "atm_iv": round(float(atm["iv"].mean()), 2),
                })

        smiles = {}
        for exp_str in df_otm["expiry"].unique():
            sub = df_otm[df_otm["expiry"] == exp_str].sort_values("strike")
            smiles[exp_str] = sub[["strike", "moneyness", "iv", "type"]].to_dict("records")

        gex_data, gamma_flip = compute_gex(df_all, S, r)
        exp_moves = compute_expected_moves(df_all, S, exps)
        rn_pdfs = compute_rn_pdfs(df_all, S, r, exps)

        result = {
            "symbol":         sym,
            "spot":           round(S, 2),
            "risk_free":      round(r * 100, 3),
            "expirations":    exps,
            "otm_data":       df_otm.to_dict("records"),
            "grid_strike":    grid_strike,
            "grid_moneyness": grid_moneyness,
            "term_structure": term_struct,
            "smiles":         smiles,
            "gex_data":       gex_data,
            "gamma_flip":     gamma_flip,
            "expected_moves": exp_moves,
            "rn_pdfs":        rn_pdfs,
        }
        cache_set(f"surface:{sym}", result)
        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("\n  Vol Surface App  ->  http://localhost:8080\n")
    app.run(debug=False, port=8080, host="0.0.0.0")
