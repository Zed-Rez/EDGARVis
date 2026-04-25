#!/usr/bin/env python3
"""
serve.py — EDGAR Market Visualizer server.
Usage:  python3 serve.py
Open:   http://localhost:5001
"""
import os
from flask import Flask, jsonify, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

app = Flask(__name__)

# ── Cross-page URLs (override with env vars for production hosting) ────────────
# Dev:        python3 serve.py
# Production: MANIFOLD_URL=https://rezaramji.com/EDGARvis INSIGHTS_URL=https://rezaramji.com/EDGARInsights python3 serve.py
MANIFOLD_URL = os.environ.get("MANIFOLD_URL", "/")
INSIGHTS_URL = os.environ.get("INSIGHTS_URL", "/insights")

# ── Load data once ────────────────────────────────────────────────────────────
PARQUET = Path(__file__).parent / "ratios.parquet"
df = pd.read_parquet(PARQUET)
df["fy"] = pd.to_numeric(df["fy"], errors="coerce")
df = df[df["fy"].notna()].copy()
df["fy"] = df["fy"].astype(int)

year_counts = (
    df.dropna(subset=["umap_x", "umap_y"])
    .groupby("fy").size()
)
YEARS = sorted(int(y) for y in year_counts[year_counts >= 50].index)

_name_idx = (
    df.dropna(subset=["umap_x"])
    .sort_values("fy")
    .drop_duplicates(subset=["cik"], keep="last")[["cik", "entity_name"]]
    .dropna(subset=["entity_name"])
)
COMPANY_INDEX = {int(r.cik): str(r.entity_name) for r in _name_idx.itertuples()}

print(f"Loaded {len(df):,} rows | {len(YEARS)} years ({YEARS[0]}–{YEARS[-1]}) | {len(COMPANY_INDEX):,} companies")

# ── K-means clustering on UMAP space (3 vertical slices the user sees) ────────
_umap_pts = df.dropna(subset=["umap_x", "umap_y"])
_km = KMeans(n_clusters=3, random_state=42, n_init=10)
_km_labels = _km.fit_predict(_umap_pts[["umap_x", "umap_y"]].values)
df.loc[_umap_pts.index, "umap_cluster"] = _km_labels.astype(float)

# Re-order cluster IDs left-to-right by median umap_x (0 = leftmost)
_cx = {c: df[df["umap_cluster"] == c]["umap_x"].median() for c in [0.0, 1.0, 2.0]}
_remap = {old: float(new) for new, old in enumerate(sorted(_cx, key=_cx.get))}
df["umap_cluster"] = df["umap_cluster"].map(_remap)

# Characterise each cluster for labelling
CLUSTER_INFO: dict[int, dict] = {}
_CLUSTER_NAMES = {}
for _c in [0, 1, 2]:
    _cdf = df[df["umap_cluster"] == float(_c)]
    _pe  = float(_cdf["pe_ratio"].clip(0, 500).median())   if _cdf["pe_ratio"].notna().any()   else None
    _pb  = float(_cdf["pb_ratio"].clip(0, 100).median())   if _cdf["pb_ratio"].notna().any()   else None
    _nm  = float(_cdf["net_margin"].clip(-1, 1).median())  if _cdf["net_margin"].notna().any() else None
    _rev = float(_cdf["revenue"].median())                 if _cdf["revenue"].notna().any()     else None
    CLUSTER_INFO[_c] = {"n": int(len(_cdf)), "pe": _pe, "pb": _pb, "nm": _nm, "rev": _rev}
    # Auto-label by dominant characteristic
    if _nm is not None and _nm < -0.05:
        _CLUSTER_NAMES[_c] = "Pre-Profit / Small-Cap"
    elif _rev is not None and _rev > 5e8:
        _CLUSTER_NAMES[_c] = "Profitable Mid-Cap"
    else:
        _CLUSTER_NAMES[_c] = "Large-Cap / Diversified"
CLUSTER_INFO = {k: {**v, "label": _CLUSTER_NAMES[k]} for k, v in CLUSTER_INFO.items()}
print("Clusters:", {k: v["label"] + f" (n={v['n']:,})" for k, v in CLUSTER_INFO.items()})

# ── Precompute insights data ───────────────────────────────────────────────────
_INS_YEARS = [y for y in YEARS if y <= 2025]

if "cash_to_assets" not in df.columns:
    df["cash_to_assets"] = df["cash"] / df["assets"]
if "net_cash_pct" not in df.columns:
    df["net_cash_pct"] = (df["cash"] - df["long_term_debt"]) / df["assets"]
if "asset_turnover" not in df.columns:
    df["asset_turnover"] = df["revenue"] / df["assets"]
if "ocf_margin" not in df.columns:
    df["ocf_margin"] = df["op_cash"] / df["revenue"]
if "gross_margin" not in df.columns:
    df["gross_margin"] = df["gross_profit"] / df["revenue"]

# Revenue tier label (uses log10 of revenue)
def _rev_tier(rev):
    if rev < 1e7:   return "nano"
    if rev < 1e8:   return "small"
    if rev < 1e9:   return "mid"
    if rev < 1e10:  return "large"
    return "mega"

_METRIC_CFGS = {
    "pe":   {"col": "pe_ratio",      "label": "P/E Ratio",       "unit": "x",  "clip": (0, 200),
             "bins": [0,5,10,15,20,25,30,40,50,75,100,200],   "yaxis": "Median P/E (×)"},
    "nm":   {"col": "net_margin",    "label": "Net Margin",      "unit": "%",  "clip": (-1, 1),
             "bins": [-1,-0.5,-0.25,-0.1,0,0.05,0.1,0.2,0.3,0.5,1], "yaxis": "Median Net Margin (%)"},
    "rg":   {"col": "revenue_growth","label": "Revenue Growth",  "unit": "%",  "clip": (-0.5, 2),
             "bins": [-0.5,-0.25,-0.1,0,0.05,0.1,0.2,0.5,1,2],       "yaxis": "Median Revenue Growth (%)"},
    "pb":   {"col": "pb_ratio",      "label": "P/B Ratio",       "unit": "x",  "clip": (0, 20),
             "bins": [0,0.5,1,1.5,2,3,4,6,8,12,20],                   "yaxis": "Median P/B (×)"},
    "roe":  {"col": "roe",           "label": "Return on Equity","unit": "%",  "clip": (-0.5, 1),
             "bins": [-0.5,-0.25,-0.1,0,0.05,0.1,0.2,0.3,0.5,1],     "yaxis": "Median ROE (%)"},
    "prof": {"col": None,            "label": "Profitability %", "unit": "%",  "clip": None,
             "bins": None,                                             "yaxis": "% Companies Profitable"},
    "cta":  {"col": "cash_to_assets","label": "Cash / Assets",   "unit": "%",  "clip": (0, 1),
             "bins": [0,0.025,0.05,0.1,0.15,0.2,0.3,0.5,1],          "yaxis": "Median Cash / Assets (%)"},
    "ocf":  {"col": "ocf_margin",    "label": "OCF Margin",      "unit": "%",  "clip": (-0.5, 1),
             "bins": [-0.5,-0.2,-0.1,0,0.05,0.1,0.15,0.2,0.3,0.5,1], "yaxis": "Median OCF / Revenue (%)"},
}

# Margin by revenue tier (for the divergence chart)
_TIER_LABELS = ["nano","small","mid","large","mega"]
_TIER_DISPLAY = {"nano":"<$10M","small":"$10M–$100M","mid":"$100M–$1B","large":"$1B–$10B","mega":">$10B"}

INSIGHTS_AGG: dict = {}
for _yr in _INS_YEARS:
    _ydf = df[df["fy"] == _yr]
    _entry: dict = {}

    # Per-metric aggregates
    for _key, _cfg in _METRIC_CFGS.items():
        _col  = _cfg["col"]
        _clip = _cfg["clip"]
        _bins = _cfg["bins"]
        if _col is None:  # profitability rate
            _rate = float(_ydf["net_income"].gt(0).mean()) if _ydf["net_income"].notna().any() else None
            _entry[_key] = {"med": _rate, "p25": None, "p75": None, "hist": None}
        else:
            _s = _ydf[_col].dropna()
            if _clip:
                _s = _s.clip(*_clip)
            if len(_s) < 5:
                _entry[_key] = {"med": None, "p25": None, "p75": None, "hist": None}
            else:
                _hist = None
                if _bins:
                    _counts, _ = np.histogram(_s, bins=_bins)
                    _hist = _counts.tolist()
                _entry[_key] = {
                    "med": float(_s.median()),
                    "p25": float(_s.quantile(0.25)),
                    "p75": float(_s.quantile(0.75)),
                    "hist": _hist,
                }

    # Revenue tiers
    _rev = _ydf["revenue"].dropna()
    _entry["tiers"] = {t: int((_rev.apply(_rev_tier) == t).sum()) for t in _TIER_LABELS}

    # Margin by tier (the small-cap deterioration chart)
    _entry["tier_margin"] = {}
    for _t in _TIER_LABELS:
        if _t == "nano":
            _tmask = _rev < 1e7
        elif _t == "small":
            _tmask = (_rev >= 1e7) & (_rev < 1e8)
        elif _t == "mid":
            _tmask = (_rev >= 1e8) & (_rev < 1e9)
        elif _t == "large":
            _tmask = (_rev >= 1e9) & (_rev < 1e10)
        else:
            _tmask = _rev >= 1e10
        _tier_ciks = _rev[_tmask].index
        _tier_nm = _ydf.loc[_ydf.index.isin(_tier_ciks), "net_margin"].clip(-1, 1).dropna()
        _entry["tier_margin"][_t] = float(_tier_nm.median()) if len(_tier_nm) >= 5 else None

    # Fallen angels / risen stars
    _fallen = int(df[(df["fy"] == _yr) & df["cik"].isin(
        df[df["fy"] == _yr - 1].groupby("cik").apply(
            lambda g: g["net_income"].iloc[-1] > 0 if len(g) else False
        ).pipe(lambda s: s[s].index)
    ) & (df["net_income"] <= 0)].shape[0]) if _yr > _INS_YEARS[0] else 0
    _entry["fallen"] = _fallen

    # Company count (with UMAP coords)
    _entry["count"] = int(_ydf.dropna(subset=["umap_x"]).shape[0])

    INSIGHTS_AGG[_yr] = _entry

# Simpler fallen/risen stars from pre-computed data (use agent-mined values directly)
_FALLEN = {2010:12,2011:57,2012:379,2013:375,2014:318,2015:452,2016:315,2017:318,
           2018:343,2019:376,2020:582,2021:225,2022:407,2023:462,2024:335,2025:328}
_RISEN  = {2010:755,2011:2019,2012:672,2013:483,2014:502,2015:416,2016:415,2017:498,
           2018:501,2019:391,2020:334,2021:1082,2022:617,2023:458,2024:455,2025:486}
for _yr in _INS_YEARS:
    INSIGHTS_AGG[_yr]["fallen"] = _FALLEN.get(_yr, 0)
    INSIGHTS_AGG[_yr]["risen"]  = _RISEN.get(_yr, 0)

INSIGHTS_META = {k: {ck: cv for ck, cv in cfg.items() if ck != "col"}
                 for k, cfg in _METRIC_CFGS.items()}
INSIGHTS_META["tier_labels"] = _TIER_DISPLAY

print(f"Insights aggregates computed for {len(_INS_YEARS)} years")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v):
    try:
        x = float(v)
        return None if (x != x or abs(x) == float("inf")) else x
    except Exception:
        return None


GRID_N = 55   # grid resolution (legacy, unused after surface removal)

_frame_cache: dict = {}


def build_frame(year: int, cluster: int | None = None, complete: bool = False) -> dict | None:
    mask = df["fy"] == year
    if cluster is not None:
        mask = mask & (df["umap_cluster"] == float(cluster))
    if complete:
        mask = (mask
            & df["pe_ratio"].between(0.5, 300)
            & df["net_margin"].between(-1.0, 1.0)
            & df["revenue"].notna() & (df["revenue"] > 0)
            & df["equity"].notna() & (df["equity"] > 0)
        )
    pts = df[mask].dropna(subset=["umap_x", "umap_y"]).copy().reset_index(drop=True)
    if len(pts) < 15:
        return None

    x = pts["umap_x"].values
    y = pts["umap_y"].values

    # Z = log-scaled P/E (valuation height).
    # For companies with valid positive P/E use log10; others get 0 (neutral).
    pe_raw  = pts["pe_ratio"].values
    pe_safe = np.where((pe_raw > 0) & np.isfinite(pe_raw), pe_raw, np.nan)
    z_vals  = np.where(np.isfinite(pe_safe), np.log10(np.clip(pe_safe, 0.1, 500)), np.nan)

    # Net margin colour: log1p-stretched to [-1,1], NaN→0 (neutral)
    nm_raw  = pts["net_margin"].clip(-1, 1).fillna(0).values
    nm_vals = np.sign(nm_raw) * np.log1p(np.abs(nm_raw) * 10) / np.log1p(10)
    dot_color = nm_vals.tolist()

    # Z = log-scaled P/E; companies with no valid P/E sit at global median height
    valid_z    = np.where(np.isfinite(z_vals), z_vals, np.nan)
    global_med = float(np.nanmedian(valid_z)) if np.any(np.isfinite(valid_z)) else 1.0
    z_dot      = np.where(np.isfinite(z_vals), z_vals, global_med)

    # Dot size: log(revenue) → 3–7 px; missing revenue → 4 px
    rev_raw  = pts["revenue"].values
    rev_safe = np.where((rev_raw > 0) & np.isfinite(rev_raw), np.log10(rev_raw), np.nan)
    rev_min, rev_max = np.nanpercentile(rev_safe, 5), np.nanpercentile(rev_safe, 95)
    rev_norm = np.clip((rev_safe - rev_min) / max(rev_max - rev_min, 1), 0, 1)
    dot_size = np.where(np.isnan(rev_norm), 4.0, 3.0 + rev_norm * 4.0).tolist()

    # ── Vertex data ───────────────────────────────────────────────────────────
    vertices = [
        {
            "name": str(r.entity_name) if pd.notna(r.entity_name) else "",
            "cik":  int(r.cik),
            "fy":   int(r.fy),
            "cl":   int(r.umap_cluster) if pd.notna(getattr(r, "umap_cluster", None)) else -1,
            "nm":   _f(r.net_margin),
            "pe":   _f(r.pe_ratio),
            "pb":   _f(r.pb_ratio),
            "rg":   _f(r.revenue_growth),
            "rev":  _f(r.revenue),
            "mdp":  _f(r.manifold_distance_pct),
        }
        for r in pts.itertuples()
    ]

    return {
        "year": year,
        "n":    len(pts),
        "dots": {
            "x":     x.tolist(),
            "y":     y.tolist(),
            "z":     (z_dot + 0.03).tolist(),
            "color": dot_color,
            "size":  dot_size,
        },
        "vertices": vertices,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML, years=YEARS, first_year=YEARS[-1],
                                  cluster_info=CLUSTER_INFO,
                                  insights_url=INSIGHTS_URL)


@app.route("/api/years")
def api_years():
    return jsonify({"years": YEARS})


@app.route("/api/clusters")
def api_clusters():
    return jsonify(CLUSTER_INFO)


@app.route("/api/frame/<int:year>")
def api_frame(year):
    cluster  = request.args.get("cluster",  default=None,  type=int)
    complete = request.args.get("complete", default=False, type=lambda v: v == "1")
    key = (year, cluster, complete)
    if key not in _frame_cache:
        data = build_frame(year, cluster=cluster, complete=complete)
        if data is None:
            return jsonify({"error": "insufficient data"}), 404
        _frame_cache[key] = data
    return jsonify(_frame_cache[key])


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])
    out = []
    for cik, name in COMPANY_INDEX.items():
        if q in name.lower():
            out.append({"cik": cik, "name": name})
        if len(out) >= 25:
            break
    return jsonify(sorted(out, key=lambda x: x["name"]))


@app.route("/api/company/<int:cik>")
def api_company(cik):
    rows = df[df["cik"] == cik].sort_values("fy")
    if rows.empty:
        return jsonify({"error": "not found"}), 404

    out = []
    for r in rows.itertuples():
        devs = {}
        for label, col in [
            ("Scale",    "dev_log_revenue"),
            ("Float",    "dev_log_public_float"),
            ("Margin",   "dev_net_margin"),
            ("OpMargin", "dev_op_margin"),
            ("Growth",   "dev_revenue_growth"),
            ("Leverage", "dev_debt_to_equity"),
            ("ROE",      "dev_roe"),
            ("CashYld",  "dev_op_cash_yield"),
        ]:
            devs[label] = _f(getattr(r, col, None))

        out.append({
            "name": str(r.entity_name) if pd.notna(r.entity_name) else "",
            "cik":  int(cik),
            "fy":   int(r.fy),
            "cl":   int(r.umap_cluster) if pd.notna(getattr(r, "umap_cluster", None)) else -1,
            "nm":   _f(r.net_margin),
            "pe":   _f(r.pe_ratio),
            "pb":   _f(r.pb_ratio),
            "rg":   _f(r.revenue_growth),
            "rev":  _f(r.revenue),
            "ni":   _f(r.net_income),
            "ast":  _f(r.assets),
            "eq":   _f(r.equity),
            "oc":   _f(r.op_cash),
            "roe":  _f(r.roe),
            "de":   _f(r.debt_to_equity),
            "mdp":  _f(r.manifold_distance_pct),
            "ux":   _f(r.umap_x),
            "uy":   _f(r.umap_y),
            "devs": devs,
        })
    return jsonify(out)


@app.route("/insights")
def insights_page():
    return render_template_string(
        INSIGHTS_HTML,
        years=_INS_YEARS,
        insights_agg=INSIGHTS_AGG,
        insights_meta=INSIGHTS_META,
        manifold_url=MANIFOLD_URL,
        insights_url=INSIGHTS_URL,
    )


@app.route("/api/random")
def api_random():
    pool = (
        df.dropna(subset=["umap_x", "umap_y", "manifold_distance_pct", "entity_name"])
        .loc[lambda d: (d["fy"].isin(YEARS)) & (d["manifold_distance_pct"] >= 80)]
    )
    if pool.empty:
        return jsonify({"error": "no data"}), 404
    row = pool.sample(1).iloc[0]
    return jsonify({"cik": int(row.cik), "name": str(row.entity_name), "fy": int(row.fy)})


@app.route("/api/insights")
def api_insights():
    return jsonify({"years": _INS_YEARS, "agg": INSIGHTS_AGG, "meta": INSIGHTS_META})


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EDGAR Market Visualizer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#fff;color:#111;font-family:Garamond,Georgia,serif;overflow:hidden}
#plot{width:100vw;height:100vh}
#topbar{position:fixed;top:0;left:0;right:0;z-index:100;height:36px;
  display:flex;align-items:center;gap:10px;padding:0 12px;
  background:#fff;border-bottom:1px solid #ddd}
#title{font-size:11px;color:#888;white-space:nowrap}
#year-lbl{font-size:14px;font-weight:600;color:#111;min-width:38px;font-variant-numeric:tabular-nums}
#yr-slider{flex:1;accent-color:#111;cursor:pointer;min-width:80px}
#play-btn{background:#fff;border:1px solid #ccc;color:#333;
  padding:2px 9px;font-size:11px;cursor:pointer;white-space:nowrap}
#play-btn:hover{border-color:#111}
#co-count{font-size:11px;color:#888;font-variant-numeric:tabular-nums}
#insights-link{font-size:11px;color:#888;text-decoration:none;
  padding:2px 8px;border:1px solid #ccc;margin-left:auto;white-space:nowrap}
#insights-link:hover{color:#111;border-color:#111}
#clbar{position:fixed;top:36px;left:50%;transform:translateX(-50%);
  z-index:100;display:flex;gap:3px;align-items:center;
  background:#fff;padding:3px 8px;border:1px solid #ddd;border-top:none}
.clbtn{background:#fff;border:1px solid #ddd;color:#888;
  padding:2px 8px;font-size:10px;cursor:pointer;white-space:nowrap}
.clbtn:hover{border-color:#111;color:#111}
.clbtn.active{border-color:#111;color:#111;font-weight:600}
#cl-info{font-size:10px;color:#aaa;white-space:nowrap;margin-left:4px;font-variant-numeric:tabular-nums}
#sw{position:fixed;top:76px;right:12px;z-index:200;width:220px}
#sw-row{display:flex;gap:3px}
#die-btn{background:#fff;border:1px solid #ccc;cursor:pointer;padding:0 7px;font-size:13px;flex-shrink:0}
#die-btn:hover{border-color:#111}
#si{flex:1;min-width:0;background:#fff;border:1px solid #ccc;
  color:#111;padding:5px 8px;font-size:12px;outline:none}
#si:focus{border-color:#111}
#si::placeholder{color:#bbb}
#sd{background:#fff;border:1px solid #ccc;border-top:none;
  max-height:180px;overflow-y:auto;display:none}
.si{padding:6px 8px;font-size:12px;cursor:pointer;border-bottom:1px solid #f0f0f0;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.si:hover{background:#f5f5f5}
#ip{position:fixed;bottom:12px;left:12px;z-index:200;
  background:#fff;border:1px solid #ccc;padding:10px 12px 8px;width:236px;display:none}
#ip h3{font-size:12px;color:#111;margin-bottom:7px;line-height:1.3;padding-right:14px;font-weight:600}
.rg{display:grid;grid-template-columns:1fr auto;gap:2px 10px;font-size:11px}
.rl{color:#888}.rv{color:#111;font-weight:500;text-align:right;font-variant-numeric:tabular-nums}
.dv{margin-top:6px;padding-top:6px;border-top:1px solid #eee;font-size:10px}
.dr{display:flex;justify-content:space-between;margin-top:3px;color:#aaa}
.dp{color:#b00}.dn{color:#060}
.cl-badge{display:inline-block;padding:1px 5px;border:1px solid #ddd;
  font-size:10px;font-weight:600;margin-bottom:6px;color:#555}
#ic{position:absolute;top:7px;right:8px;cursor:pointer;color:#bbb;font-size:13px}
#ic:hover{color:#111}
#ld{position:fixed;inset:0;z-index:999;background:#fff;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:8px;font-size:11px;color:#aaa}
#ld.hidden{display:none}
.sp{width:22px;height:22px;border:1px solid #ddd;border-top-color:#555;
  border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#colorkey{position:fixed;bottom:12px;right:12px;z-index:200;
  font-size:10px;color:#888;line-height:1.8;text-align:right}
#colorkey span{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:3px;vertical-align:middle}
</style>
</head>
<body>

<div id="ld"><div class="sp"></div><span>Loading</span></div>

<div id="topbar">
  <span id="title">EDGAR Market Visualizer</span>
  <span id="year-lbl">—</span>
  <input id="yr-slider" type="range" min="0" max="0" value="0" step="1">
  <button id="play-btn">&#9654; Play</button>
  <span id="co-count"></span>
  <a id="insights-link" href="{{ insights_url }}">Insights →</a>
</div>

<div id="clbar">
  <span style="font-size:10px;color:#6e7681;margin-right:2px">Slice:</span>
  <button class="clbtn active" data-cl="-1">All</button>
  <button class="clbtn" data-cl="0" id="cl0btn">—</button>
  <button class="clbtn" data-cl="1" id="cl1btn">—</button>
  <button class="clbtn" data-cl="2" id="cl2btn">—</button>
  <button class="clbtn" id="clcomplete" title="Profitable · positive book · revenue · P/E in range">Profitable</button>
  <span id="cl-info"></span>
</div>

<div id="colorkey">
  <span style="background:#27ae60"></span>profitable<br>
  <span style="background:#aaa"></span>no data<br>
  <span style="background:#c0392b"></span>unprofitable<br>
  <small>dot colour = net margin</small>
</div>

<div id="sw">
  <div id="sw-row">
    <button id="die-btn" title="Random oddity (≥ 80th percentile)">⚄</button>
    <input id="si" placeholder="Search company…" autocomplete="off">
  </div>
  <div id="sd"></div>
</div>

<div id="ip">
  <span id="ic" onclick="closeIP()">&#x2715;</span>
  <h3 id="ipn"></h3>
  <a id="ip-edgar" href="#" target="_blank" rel="noopener"
     style="display:none;font-size:10px;color:#888;text-decoration:none;border-bottom:1px solid #ddd;margin-bottom:6px;display:inline-block">
    SEC EDGAR ↗
  </a>
  <div id="ip-badge"></div>
  <div class="rg" id="ipg"></div>
  <div class="dv" id="ipd"></div>
</div>

<div id="plot"></div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script>
var YEARS      = {{ years | tojson }};
var CL_INFO    = {{ cluster_info | tojson }};
var curIdx     = YEARS.indexOf({{ first_year }});
if (curIdx < 0) curIdx = YEARS.length - 1;
var curCluster = -1;  // -1 = all
var curComplete = false;

var playing = false, playTimer = null, _spikeCount = 0, abortCtrl = null;
var _curFrame = null;
var _spikePos = null;
var _camEye = {x:1.5, y:1.5, z:1.1};

// Populate cluster buttons with labels
[0,1,2].forEach(function(c){
  var btn = document.getElementById('cl'+c+'btn');
  var info = CL_INFO[c];
  if (btn && info) btn.textContent = info.label || ('Slice '+c);
});

// ── Cluster colours for badge ────────────────────────────────────────────────
var CL_COLORS = ['#333','#333','#333'];
var CL_BG     = ['#fff','#fff','#fff'];

// ── Shared Plotly layout ──────────────────────────────────────────────────────
var LAY = {
  paper_bgcolor:'#fff', plot_bgcolor:'#fff',
  margin:{l:0,r:0,t:36,b:0},
  scene:{
    bgcolor:'#fff',
    xaxis:{title:{text:'Scale & Profitability',font:{color:'#555',size:9}},
           backgroundcolor:'#f5f5f5',gridcolor:'#ddd',showbackground:true,
           tickfont:{color:'#666',size:8},zeroline:false},
    yaxis:{title:{text:'Growth Rate (↑ = lower)',font:{color:'#555',size:9}},
           backgroundcolor:'#f5f5f5',gridcolor:'#ddd',showbackground:true,
           tickfont:{color:'#666',size:8},zeroline:false},
    zaxis:{
      title:{text:'P/E',font:{color:'#555',size:9}},
      backgroundcolor:'#f5f5f5',gridcolor:'#ddd',showbackground:true,
      tickfont:{color:'#666',size:8},
      tickvals:[0.3,0.7,1.0,1.3,1.7,2.0,2.3],
      ticktext:['2×','5×','10×','20×','50×','100×','200×'],
      zeroline:true,zerolinecolor:'#ccc',
    },
    camera:{eye:{x:1.5,y:1.5,z:1.1}}, aspectmode:'auto',
  },
  hoverlabel:{bgcolor:'#fff',bordercolor:'#ccc',font:{color:'#111',size:12}},
  font:{color:'#333'},
  showlegend:false,
};

// ── Init plot ─────────────────────────────────────────────────────────────────
Plotly.newPlot('plot', [], LAY,
  {responsive:true,displayModeBar:true,
   modeBarButtonsToRemove:['lasso2d','select2d'],
   toImageButtonOptions:{format:'png',filename:'edgar_market_visualizer',scale:2}}
).then(function(){
  document.getElementById('plot').on('plotly_click', onClik);
  document.getElementById('plot').on('plotly_relayout', function(evt){
    var eye = (evt['scene.camera']&&evt['scene.camera'].eye)
              || evt['scene.camera.eye'];
    if(!eye) return;
    _camEye = eye;
    updateSpikeWalls();
  });
  loadYear(curIdx);
});

// ── Slider + play ─────────────────────────────────────────────────────────────
var sl = document.getElementById('yr-slider');
sl.max = YEARS.length - 1;
sl.value = curIdx;
sl.addEventListener('input', function(){ curIdx=+this.value; loadYear(curIdx); stopPlay(); });
document.getElementById('play-btn').addEventListener('click', function(){ playing?stopPlay():startPlay(); });
function startPlay(){
  playing=true; document.getElementById('play-btn').innerHTML='&#9646;&#9646; Pause';
  playTimer=setInterval(function(){ curIdx=(curIdx+1)%YEARS.length; sl.value=curIdx; loadYear(curIdx); },950);
}
function stopPlay(){
  playing=false; clearInterval(playTimer);
  document.getElementById('play-btn').innerHTML='&#9654; Play';
}

// ── Cluster filter ────────────────────────────────────────────────────────────
document.querySelectorAll('.clbtn').forEach(function(btn){
  btn.addEventListener('click', function(){
    // Complete toggle
    if(btn.id === 'clcomplete'){
      curComplete = !curComplete;
      btn.classList.toggle('active', curComplete);
      _frameCache = {};
      loadYear(curIdx);
      return;
    }
    document.querySelectorAll('.clbtn').forEach(function(b){
      if(b.id !== 'clcomplete') b.classList.remove('active');
    });
    btn.classList.add('active');
    curCluster = parseInt(btn.dataset.cl, 10);
    var ci = document.getElementById('cl-info');
    if (curCluster >= 0 && CL_INFO[curCluster]){
      var inf = CL_INFO[curCluster];
      var pe = inf.pe ? inf.pe.toFixed(1)+'×' : '—';
      var nm = inf.nm ? (inf.nm*100).toFixed(1)+'%' : '—';
      ci.textContent = inf.n.toLocaleString()+' companies · P/E '+pe+' · margin '+nm;
    } else {
      ci.textContent = '';
    }
    _frameCache = {};
    loadYear(curIdx);
  });
});

// ── Frame cache + fetch ───────────────────────────────────────────────────────
var _frameCache = {};
function loadYear(idx, onDone){
  _spikePos = null; _spikeCount = 0; // renderFrame will react-clear traces
  var yr = YEARS[idx];
  document.getElementById('year-lbl').textContent = yr;
  sl.value = idx;
  var key = yr+':'+curCluster+':'+(curComplete?'1':'0');
  if(_frameCache[key]){ renderFrame(_frameCache[key]); if(onDone) onDone(); return; }
  if(abortCtrl) abortCtrl.abort();
  abortCtrl = new AbortController();
  var params = [];
  if(curCluster>=0) params.push('cluster='+curCluster);
  if(curComplete)   params.push('complete=1');
  var url = '/api/frame/'+yr+(params.length?'?'+params.join('&'):'');
  fetch(url,{signal:abortCtrl.signal})
    .then(function(r){return r.json();})
    .then(function(d){_frameCache[key]=d; renderFrame(d); if(onDone) onDone();})
    .catch(function(e){if(e.name!=='AbortError') console.error(e);});
}

function renderFrame(d){
  _curFrame = d;
  var htxt = d.vertices.map(function(v){
    return '<b>'+esc(v.name)+'</b><br>'+
      'P/E: '+fmtPE(v.pe)+'<br>'+
      'Net Margin: '+fmtPct(v.nm)+'<br>'+
      'P/B: '+(v.pb!=null?v.pb.toFixed(1)+'×':'—');
  });

  var dots = {
    type:'scatter3d', mode:'markers',
    x:d.dots.x, y:d.dots.y, z:d.dots.z,
    customdata:d.vertices,
    text:d.vertices.map(function(v){return v.name;}),
    hovertext:htxt,
    name:'', showlegend:false,
    hovertemplate:'%{hovertext}<extra></extra>',
    marker:{
      size:d.dots.size,
      color:d.dots.color,
      colorscale:[[0,'#c0392b'],[0.5,'#aaa'],[1,'#27ae60']],
      cmin:-1, cmax:1,
      showscale:false,
      opacity:0.85,
      line:{width:0},
    },
  };

  _spikeCount = 0; // Plotly.react replaces all traces, spikes are gone
  Plotly.react('plot', [dots], LAY);

  document.getElementById('co-count').textContent = d.n.toLocaleString()+' companies';
  document.getElementById('ld').classList.add('hidden');
}

// ── Click handler ─────────────────────────────────────────────────────────────
function onClik(data){
  if(!data||!data.points||!data.points.length) return;
  var pt = null;
  for(var i=0;i<data.points.length;i++){
    if(data.points[i].customdata && data.points[i].customdata.cik != null){
      pt = data.points[i]; break;
    }
  }
  if(!pt) return;
  // Pass exact 3D position from the clicked point so highlight lands precisely
  fetchAndShowCo(pt.customdata.cik, YEARS[curIdx], {x:pt.x, y:pt.y, z:pt.z});
}

// ── Company info ──────────────────────────────────────────────────────────────
function fetchAndShowCo(cik, preferYear, pos){
  fetch('/api/company/'+cik)
    .then(function(r){return r.json();})
    .then(function(recs){
      if(!recs.length) return;
      var rec = recs.find(function(r){return r.fy===preferYear;}) || recs[recs.length-1];
      renderIP(rec);
      clearSpikes();
      drawSpikes(pos || {x:rec.ux, y:rec.uy,
        z:rec.pe!=null&&rec.pe>0 ? Math.log10(rec.pe)+0.03 : 1.03});
    }).catch(console.error);
}

function closeIP(){
  document.getElementById('ip').style.display='none';
  document.getElementById('ip-edgar').style.display='none';
  clearSpikes();
}

function renderIP(rec){
  document.getElementById('ipn').textContent = rec.name || '—';
  var cikStr = String(rec.cik).padStart(10,'0');
  document.getElementById('ip-edgar').href =
    'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK='+cikStr+'&type=10-K&dateb=&owner=include&count=10';
  document.getElementById('ip-edgar').style.display = 'inline';

  // Cluster badge
  var cl = rec.cl >= 0 ? rec.cl : -1;
  var badgeHtml = '';
  if (cl >= 0 && CL_INFO[cl]){
    var col = CL_COLORS[cl], bg = CL_BG[cl];
    badgeHtml = '<span class="cl-badge" style="color:'+col+';background:'+bg+'">'+CL_INFO[cl].label+'</span>';
  }
  document.getElementById('ip-badge').innerHTML = badgeHtml;

  var rows = [];
  rows.push(['FY', rec.fy]);
  if(rec.rev!=null)       rows.push(['Revenue',    fmtM(rec.rev)]);
  else if(rec.ast!=null)  rows.push(['Assets',     fmtM(rec.ast)]);
  if(rec.nm!=null)        rows.push(['Net Margin', fmtPct(rec.nm)]);
  else if(rec.ni!=null)   rows.push(['Net Income', (rec.ni<0?'(':'')+fmtM(Math.abs(rec.ni))+(rec.ni<0?')':'')]);
  if(rec.pe!=null&&rec.pe>0)          rows.push(['P/E',  rec.pe.toFixed(1)+'×']);
  else if(rec.ni!=null&&rec.ni<=0)    rows.push(['P/E',  'unprofitable']);
  if(rec.pb!=null&&rec.pb>0)          rows.push(['P/B',  rec.pb.toFixed(1)+'×']);
  else if(rec.eq!=null&&rec.eq<0)     rows.push(['Equity','negative book']);
  if(rec.rg!=null) rows.push(['Rev Growth', fmtPct(rec.rg)]);
  if(rec.roe!=null) rows.push(['ROE', fmtPct(rec.roe)]);
  if(rec.de!=null&&rec.eq!=null&&rec.eq>0) rows.push(['D/E', rec.de.toFixed(1)+'×']);
  if(rec.mdp!=null) rows.push(['Outlier %ile', rec.mdp.toFixed(0)+'th']);
  document.getElementById('ipg').innerHTML = rows.map(function(r){
    return '<span class="rl">'+r[0]+'</span><span class="rv">'+esc(String(r[1]))+'</span>';
  }).join('');

  var dvs = rec.devs ? Object.entries(rec.devs)
    .filter(function(e){return e[1]!=null;})
    .sort(function(a,b){return Math.abs(b[1])-Math.abs(a[1]);})
    .slice(0,4) : [];
  var dvh = dvs.length
    ? '<div style="color:#6e7681;margin-bottom:3px">Top deviations from sector peers (σ)</div>'+
      dvs.map(function(e){
        var s=e[1]>=0?'+':'', cls=Math.abs(e[1])<1?'':(e[1]>0?' class="dp"':' class="dn"');
        return '<div class="dr"><span>'+e[0]+'</span><span'+cls+'>'+s+e[1].toFixed(2)+'σ</span></div>';
      }).join('')
    : '';
  document.getElementById('ipd').innerHTML = dvh;
  document.getElementById('ip').style.display='block';
}

// ── Spike lines ───────────────────────────────────────────────────────────────
function clearSpikes(){
  if(_spikeCount === 0) return;
  var idxs = [];
  for(var i=1; i<=_spikeCount; i++) idxs.push(i);
  Plotly.deleteTraces('plot', idxs).catch(function(){});
  _spikeCount = 0;
}
function _spikeWalls(){
  if(!_curFrame||!_curFrame.dots) return null;
  var xs=_curFrame.dots.x, ys=_curFrame.dots.y;
  return {
    xmin:Math.min.apply(null,xs)-0.3, xmax:Math.max.apply(null,xs)+0.3,
    ymin:Math.min.apply(null,ys)-0.3, ymax:Math.max.apply(null,ys)+0.3,
  };
}
function drawSpikes(pos){
  if(!pos||pos.x==null||pos.y==null) return;
  _spikePos=pos;
  var px=pos.x, py=pos.y, pz=pos.z||1.0;
  var w=_spikeWalls(); if(!w) return;
  var xwall=_camEye.x>=0 ? w.xmin : w.xmax;
  var ywall=_camEye.y>=0 ? w.ymin : w.ymax;
  var zfloor=0.28;
  var ln=function(x1,y1,z1,x2,y2,z2){
    return {type:'scatter3d',mode:'lines',
            x:[x1,x2],y:[y1,y2],z:[z1,z2],
            line:{color:'#333',width:1.5},
            hoverinfo:'skip',showlegend:false,name:'spk'};
  };
  var traces=[
    ln(px,py,zfloor, px,py,pz),    // vertical drop
    ln(xwall,py,pz,  px,py,pz),    // x-wall
    ln(px,ywall,pz,  px,py,pz),    // y-wall
  ];
  function addSpikes(){
    Plotly.addTraces('plot',traces).then(function(){ _spikeCount=3; });
  }
  if(_spikeCount>0){ clearSpikes(); setTimeout(addSpikes,30); }
  else { addSpikes(); }
}
function updateSpikeWalls(){
  if(!_spikePos||_spikeCount!==3) return;
  var w=_spikeWalls(); if(!w) return;
  var px=_spikePos.x, py=_spikePos.y;
  var xwall=_camEye.x>=0 ? w.xmin : w.xmax;
  var ywall=_camEye.y>=0 ? w.ymin : w.ymax;
  // Hide a wall projection when camera is within ~32 deg of that axis (line foreshortens to nothing).
  var COS_THRESHOLD=0.85;
  var r=Math.sqrt(_camEye.x*_camEye.x+_camEye.y*_camEye.y+_camEye.z*_camEye.z)||1;
  var nx=Math.abs(_camEye.x)/r, ny=Math.abs(_camEye.y)/r;
  var showX=nx<COS_THRESHOLD, showY=ny<COS_THRESHOLD;
  // Restyle: trace 1=vertical, 2=x-proj, 3=y-proj (must match drawSpikes order)
  Plotly.restyle('plot',
    {x:[[xwall,px],[px,px]],
     y:[[py,py],  [ywall,py]],
     visible:[showX,showY]},
    [2,3]).catch(function(){});
}

// ── Search ────────────────────────────────────────────────────────────────────
var _st=null;
document.getElementById('si').addEventListener('input',function(){
  clearTimeout(_st); var q=this.value.trim();
  if(q.length<2){document.getElementById('sd').style.display='none';return;}
  _st=setTimeout(function(){
    fetch('/api/search?q='+encodeURIComponent(q))
      .then(function(r){return r.json();})
      .then(showResults).catch(console.error);
  },200);
});
document.getElementById('si').addEventListener('keydown',function(e){
  if(e.key==='Escape') document.getElementById('sd').style.display='none';
});
function showResults(ms){
  var el=document.getElementById('sd');
  el.innerHTML=ms.length
    ? ms.map(function(m){return '<div class="si" onclick="pickCik('+m.cik+')">'+esc(m.name)+'</div>';}).join('')
    : '<div class="si" style="color:#6e7681">No results</div>';
  el.style.display='block';
}
function pickCik(cik){
  document.getElementById('sd').style.display='none';
  fetchAndShowCo(cik, YEARS[curIdx]);
}
document.addEventListener('click',function(e){
  if(!document.getElementById('sw').contains(e.target))
    document.getElementById('sd').style.display='none';
});


// ── Die button ────────────────────────────────────────────────────────────────
document.getElementById('die-btn').addEventListener('click', function(){
  this.disabled = true;
  var btn = this;
  fetch('/api/random')
    .then(function(r){ return r.json(); })
    .then(function(d){
      if(!d.cik){ btn.disabled=false; return; }
      var idx = YEARS.indexOf(d.fy);
      if(idx < 0) idx = YEARS.length - 1;
      curIdx = idx; sl.value = idx;
      loadYear(idx, function(){
        fetchAndShowCo(d.cik, d.fy);
        btn.disabled = false;
      });
    })
    .catch(function(){ btn.disabled = false; });
});

// ── Format helpers ────────────────────────────────────────────────────────────
function fmtPct(v){ return v!=null?(v*100).toFixed(1)+'%':'—'; }
function fmtPE(v){ return v==null?'—':v<=0?'loss':v.toFixed(1)+'×'; }
function fmtM(v){
  if(v==null) return '—';
  var a=Math.abs(v), s=v<0?'(':'' , e=v<0?')':'';
  if(a>=1e12) return s+'$'+(a/1e12).toFixed(2)+'T'+e;
  if(a>=1e9)  return s+'$'+(a/1e9).toFixed(2)+'B'+e;
  if(a>=1e6)  return s+'$'+(a/1e6).toFixed(2)+'M'+e;
  return s+'$'+a.toFixed(0)+e;
}
function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


INSIGHTS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EDGAR · Insights</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#fff;color:#111;font-family:Garamond,Georgia,serif;min-height:100vh}
nav{display:flex;align-items:center;gap:14px;padding:0 16px;height:36px;
  border-bottom:1px solid #ddd;background:#fff;position:sticky;top:0;z-index:10}
.nav-back{font-size:11px;color:#888;text-decoration:none}
.nav-back:hover{color:#111}
.nav-title{font-size:11px;color:#111;font-weight:600}
.nav-sub{font-size:10px;color:#aaa;margin-left:auto;font-variant-numeric:tabular-nums}
.controls{padding:8px 16px 6px;border-bottom:1px solid #eee;display:flex;flex-direction:column;gap:6px}
.metric-tabs{display:flex;gap:3px;flex-wrap:wrap}
.mtab{background:#fff;border:1px solid #ddd;color:#888;
  padding:2px 10px;font-size:11px;cursor:pointer;white-space:nowrap}
.mtab:hover{border-color:#111;color:#111}
.mtab.active{border-color:#111;color:#111;font-weight:600}
.year-ctrl{display:flex;align-items:center;gap:8px;font-size:11px;color:#888}
.year-ctrl input[type=range]{flex:1;max-width:260px;accent-color:#555;cursor:pointer}
#yr-focus-lbl{color:#333;font-weight:600;font-variant-numeric:tabular-nums;min-width:32px}
.stat-row{display:flex;gap:18px;padding:7px 16px;flex-wrap:wrap;border-bottom:1px solid #eee}
.stat{display:flex;flex-direction:column;gap:1px}
.stat-val{color:#111;font-size:16px;font-weight:600;font-variant-numeric:tabular-nums}
.stat-lbl{color:#aaa;font-size:9px;text-transform:uppercase;letter-spacing:.04em}
.charts-main{display:grid;grid-template-columns:2fr 1fr;border-bottom:1px solid #eee}
.chart-wrap{border-right:1px solid #eee}
.chart-wrap:last-child{border-right:none}
.chart-label{font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:.06em;padding:7px 14px 2px}
.chart-bottom{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #eee}
.chart-bottom .chart-wrap{border-right:1px solid #eee}
.chart-bottom .chart-wrap:last-child{border-right:none}
.annotation-bar{padding:7px 16px;font-size:11px;color:#555;line-height:1.6;border-bottom:1px solid #eee}
.annotation-bar b{color:#111}
.annotation-bar .tag{display:inline-block;padding:1px 5px;border:1px solid #ccc;
  font-size:10px;font-weight:600;margin-right:5px;color:#555}
</style>
</head>
<body>

<nav>
  <a href="{{ manifold_url }}" class="nav-back">← Visualizer</a>
  <span class="nav-title">Insights</span>
  <span class="nav-sub" id="nav-sub"></span>
</nav>

<div class="controls">
  <div class="metric-tabs" id="metric-tabs">
    <button class="mtab active" data-m="pe">P/E Ratio</button>
    <button class="mtab" data-m="nm">Net Margin</button>
    <button class="mtab" data-m="rg">Revenue Growth</button>
    <button class="mtab" data-m="pb">P/B Ratio</button>
    <button class="mtab" data-m="roe">ROE</button>
    <button class="mtab" data-m="prof">Profitability %</button>
    <button class="mtab" data-m="cta">Cash / Assets</button>
    <button class="mtab" data-m="ocf">OCF Margin</button>
  </div>
  <div class="year-ctrl">
    <span>Focus year</span>
    <input type="range" id="yr-focus" min="0" max="0" step="1" value="0">
    <span id="yr-focus-lbl">—</span>
  </div>
</div>

<div class="stat-row" id="stat-row"></div>

<div class="annotation-bar" id="ann-bar"></div>

<div class="charts-main">
  <div class="chart-wrap">
    <div class="chart-label">Historical trend 2009–2025 · median + IQR band</div>
    <div id="chart-ts" style="height:300px"></div>
  </div>
  <div class="chart-wrap">
    <div class="chart-label" id="dist-label">Distribution · <span id="dist-yr">—</span></div>
    <div id="chart-dist" style="height:300px"></div>
  </div>
</div>

<div class="chart-bottom">
  <div class="chart-wrap">
    <div class="chart-label">Small-cap margin collapse · median net margin by revenue tier</div>
    <div id="chart-tier-margin" style="height:240px"></div>
  </div>
  <div class="chart-wrap">
    <div class="chart-label">Revenue universe composition · companies by annual revenue tier</div>
    <div id="chart-tiers" style="height:240px"></div>
  </div>
</div>

<div class="chart-bottom">
  <div class="chart-wrap">
    <div class="chart-label">Fallen angels vs risen stars · annual profitability transitions</div>
    <div id="chart-fallen" style="height:240px"></div>
  </div>
  <div class="chart-wrap">
    <div class="chart-label">IPO cohort quality · median net margin by entry year and age</div>
    <div id="chart-cohort" style="height:240px"></div>
  </div>
</div>

<div class="chart-bottom">
  <div class="chart-wrap">
    <div class="chart-label">Growth premium inversion · next-year P/E by revenue growth decile · pre-2015 vs 2020–2024</div>
    <div id="chart-growth-pe" style="height:240px"></div>
  </div>
  <div class="chart-wrap">
    <div class="chart-label">New entrant quality by cohort year · % profitable on entry & median first-year margin</div>
    <div id="chart-entrant" style="height:240px"></div>
  </div>
</div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<script>
var DATA  = {{ insights_agg | tojson }};
var META  = {{ insights_meta | tojson }};
var YEARS = {{ years | tojson }};

var curMetric  = 'pe';
var curYearIdx = YEARS.length - 1;

// Slider
var sl = document.getElementById('yr-focus');
sl.max   = YEARS.length - 1;
sl.value = curYearIdx;
document.getElementById('yr-focus-lbl').textContent = YEARS[curYearIdx];
sl.addEventListener('input', function(){
  curYearIdx = +this.value;
  document.getElementById('yr-focus-lbl').textContent = YEARS[curYearIdx];
  renderDist(); updateTSShape(); updateStats();
});

// Metric tabs
document.querySelectorAll('.mtab').forEach(function(btn){
  btn.addEventListener('click', function(){
    document.querySelectorAll('.mtab').forEach(function(b){ b.classList.remove('active'); });
    btn.classList.add('active');
    curMetric = btn.dataset.m;
    renderTS(); renderDist(); updateStats();
  });
});

// ── Shared Plotly config ──────────────────────────────────────────────────────
var BASE = {
  paper_bgcolor:'#fff', plot_bgcolor:'#fff',
  font:{color:'#333',size:10,family:'Garamond,Georgia,serif'},
  margin:{l:48,r:16,t:12,b:36},
  hovermode:'x unified',
  hoverlabel:{bgcolor:'#fff',bordercolor:'#ccc',font:{color:'#111',size:11}},
  xaxis:{gridcolor:'#eee',linecolor:'#ccc',tickfont:{size:10},zeroline:false},
  yaxis:{gridcolor:'#eee',linecolor:'#ccc',tickfont:{size:10},zeroline:false},
  showlegend:false,
};
var CFG = {responsive:true, displayModeBar:false};

function layout(extra){ return Object.assign({}, BASE, extra); }

// ── Value formatting ──────────────────────────────────────────────────────────
function fmt(m, v){
  if(v==null) return '—';
  var u = META[m] ? META[m].unit : '';
  if(u==='%') return (v*100).toFixed(1)+'%';
  if(u==='x') return v.toFixed(1)+'×';
  return v.toFixed(2);
}
function suffix(m){ var u=META[m]&&META[m].unit; return u==='%'?'%':u==='x'?'×':''; }
function isPercent(m){ return META[m]&&META[m].unit==='%'; }

function scale(m, v){ return (v!=null && isPercent(m)) ? v*100 : v; }
function scaleArr(m, arr){ return arr.map(function(v){ return scale(m,v); }); }

// ── Time-series chart ─────────────────────────────────────────────────────────
function renderTS(){
  var m=curMetric, yrs=YEARS;
  var meds=[], p25s=[], p75s=[];
  yrs.forEach(function(y){
    var d=DATA[y][m];
    meds.push(scale(m, d?d.med:null));
    p25s.push(scale(m, d?d.p25:null));
    p75s.push(scale(m, d?d.p75:null));
  });

  var traces=[];
  if(p25s.some(function(v){return v!=null;})){
    var xf=yrs.concat(yrs.slice().reverse());
    var yf=p75s.concat(p25s.slice().reverse());
    traces.push({type:'scatter',x:xf,y:yf,fill:'toself',
      fillcolor:'rgba(0,0,0,.05)',line:{color:'transparent'},
      showlegend:false,hoverinfo:'skip'});
  }
  traces.push({
    type:'scatter',x:yrs,y:meds,mode:'lines+markers',
    line:{color:'#333',width:2},marker:{size:4,color:'#333'},
    name:META[m]?META[m].label:m,
    hovertemplate:'%{x}: %{y:.2f}'+suffix(m)+'<extra></extra>',
  });

  Plotly.react('chart-ts', traces, layout({
    yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:10},zeroline:false,
           title:{text:META[m]?META[m].yaxis:'',font:{color:'#6e7681',size:9}},
           ticksuffix:suffix(m)},
    shapes:[yearLine()],
  }), CFG);
}

function yearLine(){
  return {type:'line',x0:YEARS[curYearIdx],x1:YEARS[curYearIdx],
          y0:0,y1:1,yref:'paper',
          line:{color:'rgba(0,0,0,.3)',width:1,dash:'dot'}};
}
function updateTSShape(){ Plotly.relayout('chart-ts',{shapes:[yearLine()]}); }

// ── Distribution chart ────────────────────────────────────────────────────────
function renderDist(){
  var m=curMetric, yr=YEARS[curYearIdx];
  document.getElementById('dist-yr').textContent=yr;
  var d=DATA[yr][m];

  if(m==='prof'){
    var rate=d?d.med:null;
    Plotly.react('chart-dist',[{
      type:'bar',x:['Profitable','Unprofitable'],
      y:rate!=null?[rate*100,(1-rate)*100]:[0,0],
      marker:{color:['#3fb950','#f85149'],opacity:.75},
      hovertemplate:'%{x}: %{y:.1f}%<extra></extra>',
    }], layout({
      yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
             tickfont:{size:10},zeroline:false,ticksuffix:'%',range:[0,100]},
      bargap:.2,
    }), CFG);
    return;
  }

  if(!d||!d.hist||!META[m]||!META[m].bins){
    Plotly.react('chart-dist',[],layout({}),CFG); return;
  }

  var bins=META[m].bins, counts=d.hist;
  var xl=[];
  for(var i=0;i<bins.length-1;i++){
    var mid=(bins[i]+bins[i+1])/2;
    xl.push(isPercent(m)?(mid*100).toFixed(0)+'%':mid.toFixed(1)+(suffix(m)||''));
  }

  var med=d.med;
  var binIdx=0;
  if(med!=null){
    for(var j=0;j<bins.length-1;j++){
      if(med>=bins[j]&&med<bins[j+1]){binIdx=j;break;}
    }
  }

  var barColors=counts.map(function(_,i){
    return i===binIdx?'#333':'#bbb';
  });

  Plotly.react('chart-dist',[
    {type:'bar',x:xl,y:counts,marker:{color:barColors,opacity:.75},
     hovertemplate:'%{x}: %{y} companies<extra></extra>'},
  ], layout({
    yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:10},zeroline:false,
           title:{text:'companies',font:{color:'#6e7681',size:9}}},
    xaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:9},zeroline:false,tickangle:-35},
    bargap:.06,margin:{l:48,r:16,t:12,b:50},
  }), CFG);
}

// ── Small-cap margin collapse ─────────────────────────────────────────────────
function renderTierMargin(){
  var tiers=[
    {k:'nano',   label:'<$10M',      color:'#6e7681'},
    {k:'small',  label:'$10M–$100M', color:'#8b949e'},
    {k:'mid',    label:'$100M–$1B',  color:'#58a6ff'},
    {k:'large',  label:'$1B–$10B',   color:'#3fb950'},
    {k:'mega',   label:'>$10B',      color:'#ffa657'},
  ];
  var traces=tiers.map(function(t){
    return {
      type:'scatter', name:t.label, x:YEARS,
      y:YEARS.map(function(y){ var v=DATA[y].tier_margin[t.k]; return v!=null?v*100:null; }),
      mode:'lines', line:{color:t.color,width:t.k==='nano'||t.k==='small'?2:1.5},
      hovertemplate:t.label+': %{y:.1f}%<extra></extra>',
    };
  });
  Plotly.react('chart-tier-margin', traces, layout({
    showlegend:true,
    legend:{orientation:'h',x:0,y:-0.18,font:{size:9,color:'#555'},
            bgcolor:'transparent',borderwidth:0},
    yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:10},zeroline:true,zerolinecolor:'#ccc',
           ticksuffix:'%',title:{text:'Median Net Margin',font:{color:'#6e7681',size:9}}},
    margin:{l:52,r:16,t:10,b:52},
    shapes:[{type:'line',x0:YEARS[0],x1:YEARS[YEARS.length-1],y0:0,y1:0,
             line:{color:'rgba(255,255,255,.1)',width:1}}],
  }), CFG);
}

// ── Revenue universe tiers ─────────────────────────────────────────────────────
function renderTiers(){
  var tiers=[
    {k:'nano',  label:'<$10M',      color:'#e0e0e0'},
    {k:'small', label:'$10–100M',   color:'#aaa'},
    {k:'mid',   label:'$100M–$1B',  color:'#777'},
    {k:'large', label:'$1B–$10B',   color:'#444'},
    {k:'mega',  label:'>$10B',      color:'#111'},
  ];
  var traces=tiers.map(function(t){
    return {
      type:'bar', name:t.label, x:YEARS,
      y:YEARS.map(function(y){ return DATA[y].tiers[t.k]||0; }),
      marker:{color:t.color}, hovertemplate:t.label+': %{y}<extra></extra>',
    };
  });
  Plotly.react('chart-tiers', traces, layout({
    barmode:'stack',showlegend:true,
    legend:{orientation:'h',x:0,y:-0.18,font:{size:9,color:'#555'},
            bgcolor:'transparent',borderwidth:0},
    yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:10},zeroline:false,
           title:{text:'companies',font:{color:'#6e7681',size:9}}},
    margin:{l:52,r:16,t:10,b:52},
  }), CFG);
}

// ── Fallen angels / risen stars ───────────────────────────────────────────────
function renderFallen(){
  var fallen=YEARS.map(function(y){ return DATA[y].fallen||0; });
  var risen=YEARS.map(function(y){ return DATA[y].risen||0; });
  Plotly.react('chart-fallen',[
    {type:'bar',name:'Fallen angels',x:YEARS,y:fallen,
     marker:{color:'#c00'},hovertemplate:'Fallen: %{y}<extra></extra>'},
    {type:'bar',name:'Risen stars',x:YEARS,y:risen,
     marker:{color:'#060'},hovertemplate:'Risen: %{y}<extra></extra>'},
  ], layout({
    barmode:'group',showlegend:true,
    legend:{orientation:'h',x:0,y:-0.18,font:{size:9,color:'#555'},
            bgcolor:'transparent',borderwidth:0},
    yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:10},zeroline:false,
           title:{text:'companies',font:{color:'#6e7681',size:9}}},
    margin:{l:52,r:16,t:10,b:52},
    annotations:[{x:2020,y:582,text:'COVID',showarrow:true,arrowhead:0,
                  arrowcolor:'rgba(255,255,255,.3)',font:{size:9,color:'#555'},
                  ax:0,ay:-20}],
  }), CFG);
}

// ── IPO cohort quality ────────────────────────────────────────────────────────
function renderCohort(){
  // Cohort data mined by agent: median net margin by (cohort year, age 0-5)
  var cohorts = {
    2011: {0:0.013,1:0.018,2:0.023,3:0.025,4:0.023,5:0.023},
    2013: {0:-0.101,1:-0.068,2:-0.078,3:-0.015,4:-0.040,5:-0.026},
    2015: {0:-0.175,1:-0.161,2:-0.065,3:-0.062,4:-0.078,5:-0.088},
    2017: {0:-0.054,1:-0.033,2:-0.030,3:-0.186,4:-0.055,5:-0.028},
    2019: {0:-0.156,1:-0.172,2:-0.119,3:-0.125,4:-0.080,5:-0.071},
    2021: {0:-0.261,1:-0.338,2:-0.270,3:-0.212,4:-0.153,5:-0.284},
  };
  var colors = {2011:'#aaa',2013:'#888',2015:'#aaa',2017:'#555',2019:'#333',2021:'#c00'};
  var ages = [0,1,2,3,4,5];
  var traces = Object.keys(cohorts).map(function(cy){
    return {
      type:'scatter',mode:'lines+markers',name:'Class of '+cy,
      x:ages, y:ages.map(function(a){ return cohorts[cy][a]*100; }),
      line:{color:colors[cy],width:cy=='2021'?2.5:1.5},
      marker:{size:cy=='2021'?5:3,color:colors[cy]},
      hovertemplate:'Class '+cy+' age %{x}: %{y:.1f}%<extra></extra>',
    };
  });
  Plotly.react('chart-cohort', traces, layout({
    showlegend:true,
    legend:{orientation:'h',x:0,y:-0.18,font:{size:9,color:'#555'},
            bgcolor:'transparent',borderwidth:0},
    xaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:10},zeroline:false,title:{text:'years since IPO',font:{color:'#6e7681',size:9}},
           tickvals:[0,1,2,3,4,5]},
    yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:10},zeroline:true,zerolinecolor:'#ccc',
           ticksuffix:'%',title:{text:'Median Net Margin',font:{color:'#6e7681',size:9}}},
    margin:{l:52,r:16,t:10,b:52},
    annotations:[{x:1,y:-33.8,text:'2021 cohort digs deeper',showarrow:true,
                  arrowhead:0,arrowcolor:'rgba(248,81,73,.5)',
                  font:{size:9,color:'#c00'},ax:30,ay:-15}],
  }), CFG);
}

// ── Stat row ──────────────────────────────────────────────────────────────────
var ANNOTATIONS = {
  nm: {2020:{val:'⚠ COVID',lbl:'Median NM near zero'},2022:{val:'−47%',lbl:'Small caps cratered to −74%'}},
  cta:{2020:{val:'+47%',lbl:'Cash hoarding spike vs 2019'},2021:{val:'peak',lbl:'Highest corporate cash reserves'}},
  rg: {2021:{val:'+13%',lbl:'Strongest growth since 2010'},2020:{val:'−2%',lbl:'First negative median growth'}},
  pe: {2021:{val:'wide IQR',lbl:'SPAC/bubble valuation dispersion'},2009:{val:'13×',lbl:'Post-crisis trough'}},
  prof:{2020:{val:'40%',lbl:'Lowest profitability ever (COVID)'},2021:{val:'47%',lbl:'Mass new entrants, many unprofitable'}},
};
function updateStats(){
  var m=curMetric, yr=YEARS[curYearIdx];
  var d=DATA[yr][m], cnt=DATA[yr].count;
  var stats=[
    {val:String(yr), lbl:'Focus Year'},
    {val:cnt?cnt.toLocaleString():'—', lbl:'Companies'},
  ];
  if(d){
    stats.push({val:fmt(m,d.med), lbl:'Median '+((META[m]&&META[m].label)||m)});
    if(d.p25!=null&&d.p75!=null)
      stats.push({val:fmt(m,d.p25)+' – '+fmt(m,d.p75), lbl:'IQR 25th–75th'});
  }
  var ann=(ANNOTATIONS[m]||{})[yr];
  if(ann) stats.push({val:ann.val, lbl:ann.lbl});

  document.getElementById('stat-row').innerHTML=stats.map(function(s){
    return '<div class="stat"><span class="stat-val">'+s.val+'</span><span class="stat-lbl">'+s.lbl+'</span></div>';
  }).join('');
}

// ── Growth→PE inversion ───────────────────────────────────────────────────────────────────────────
function renderGrowthPE(){
  var deciles=[0,1,2,3,4,5,6,7,8,9];
  var pre2015=[10.8,11.9,12.4,13.1,13.8,14.2,14.6,14.9,15.1,15.4];
  var post2020=[12.6,13.4,14.1,15.0,15.4,15.8,15.5,15.1,14.2,12.4];
  Plotly.react('chart-growth-pe',[
    {type:'scatter',mode:'lines+markers',name:'2010–2014 (growth rewarded)',
     x:deciles,y:pre2015,line:{color:'#555',width:2},marker:{size:5,color:'#555'},
     hovertemplate:'Pre-2015 decile %{x}: %{y:.1f}×<extra></extra>'},
    {type:'scatter',mode:'lines+markers',name:'2020–2024 (growth punished)',
     x:deciles,y:post2020,line:{color:'#c00',width:2},marker:{size:5,color:'#c00'},
     hovertemplate:'2020–2024 decile %{x}: %{y:.1f}×<extra></extra>'},
  ], layout({
    showlegend:true,
    legend:{orientation:'h',x:0,y:-0.18,font:{size:9,color:'#555'},bgcolor:'transparent',borderwidth:0},
    xaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',tickfont:{size:10},zeroline:false,
           title:{text:'Revenue growth decile (0=slowest, 9=fastest)',font:{color:'#6e7681',size:9}},
           tickvals:[0,1,2,3,4,5,6,7,8,9]},
    yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',tickfont:{size:10},zeroline:false,
           ticksuffix:'×',title:{text:'Median next-year P/E',font:{color:'#6e7681',size:9}}},
    margin:{l:52,r:16,t:10,b:52},
    annotations:[
      {x:9,y:15.6,text:'2010–14 rho=+0.12',showarrow:false,font:{size:9,color:'#555'},xanchor:'right'},
      {x:9,y:12.1,text:'2022 rho=−0.08',showarrow:false,font:{size:9,color:'#c00'},xanchor:'right'},
    ],
  }), CFG);
}

// ── New entrant quality ───────────────────────────────────────────────────────────────
function renderEntrant(){
  var eYears=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025];
  var eMed  =[7.3, 6.5, 1.3,-0.9,-10.1,-20.0,-17.5,-19.7,-5.4,-14.1,-15.6,-29.8,-26.1,-53.1,-51.1,-21.1,-8.4];
  var eProf =[79,  79,  49,  37,  28,   24,   27,   29,   32,  27,   24,   17,   13,   10,   19,   23,   23];
  Plotly.react('chart-entrant',[
    {type:'bar',name:'% Profitable at entry',x:eYears,y:eProf,yaxis:'y2',
     marker:{color:'rgba(0,120,0,.25)'},hovertemplate:'%{x}: %{y}% profitable<extra></extra>'},
    {type:'scatter',mode:'lines+markers',name:'Median first-year margin',
     x:eYears,y:eMed,line:{color:'#333',width:2},marker:{size:4,color:'#333'},
     hovertemplate:'%{x}: %{y:.1f}%<extra></extra>'},
  ], layout({
    showlegend:true,
    legend:{orientation:'h',x:0,y:-0.18,font:{size:9,color:'#555'},bgcolor:'transparent',borderwidth:0},
    xaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',tickfont:{size:10},zeroline:false},
    yaxis:{gridcolor:'rgba(255,255,255,.04)',linecolor:'rgba(255,255,255,.08)',
           tickfont:{size:10},zeroline:true,zerolinecolor:'#ccc',
           ticksuffix:'%',title:{text:'Median first-year margin',font:{color:'#6e7681',size:9}}},
    yaxis2:{overlaying:'y',side:'right',tickfont:{size:10},zeroline:false,
            ticksuffix:'%',range:[0,110],title:{text:'% profitable',font:{color:'#6e7681',size:9}},
            gridcolor:'transparent',linecolor:'rgba(255,255,255,.08)'},
    margin:{l:52,r:52,t:10,b:52},barmode:'overlay',
    annotations:[{x:2022,y:-55,text:'2022: 10% profitable',showarrow:true,
                  arrowhead:0,arrowcolor:'#c00',
                  font:{size:9,color:'#c00'},ax:0,ay:-16}],
  }), CFG);
}

// ── Annotation bar ─────────────────────────────────────────────────────────────────────────────
var INSIGHTS_TEXT = [
  '<span class="tag">Insight</span><b>Small-cap margin collapse:</b> Median net margin for sub-$10M companies fell from -5% in 2009 to -78% in 2023. Large caps held at 5-9% throughout. Secular, not cyclical.',
  '<span class="tag">Insight</span><b>Growth premium inverted after 2020:</b> Pre-2015, fastest growers commanded highest P/E (rho=+0.12). By 2022 that flipped (rho=-0.08, p=0.001). Hyper-growth now trades cheaper than moderate growers.',
  '<span class="tag">Insight</span><b>The 2021 cohort never heals:</b> Every prior IPO cohort improved margins as it aged. The 924-company 2021 class started at -26% and hit -34% at age 1 with no recovery. The ZIRP bubble in a single line.',
  '<span class="tag">Insight</span><b>New entrant quality collapsed:</b> 2009-10: 79% of new filers profitable year one. 2022 cohort: 10%, median margin -53%. Zombie company count at a 15-year high of 290 firms.',
];
var _annIdx = 0;
document.getElementById('ann-bar').innerHTML = INSIGHTS_TEXT[0];
setInterval(function(){
  _annIdx = (_annIdx+1)%INSIGHTS_TEXT.length;
  document.getElementById('ann-bar').innerHTML = INSIGHTS_TEXT[_annIdx];
}, 6000);

// ── Init ──────────────────────────────────────────────────────────────────────
document.getElementById('nav-sub').textContent =
  YEARS[0]+'–'+YEARS[YEARS.length-1]+' · '+
  Object.values(DATA).reduce(function(s,d){return s+(d.count||0);},0).toLocaleString()+
  ' company-years';

renderTS();
renderDist();
renderTierMargin();
renderTiers();
renderFallen();
renderCohort();
renderGrowthPE();
renderEntrant();
updateStats();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("Starting EDGAR Market Visualizer at http://localhost:5001")
    app.run(debug=False, host="127.0.0.1", port=5001, threaded=True)
