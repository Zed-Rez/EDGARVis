#!/usr/bin/env python3
"""
serve.py — EDGAR Market Manifold server.
Usage:  python3 serve.py
Open:   http://localhost:5001
"""
from flask import Flask, jsonify, render_template_string, request
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import KMeans
from pathlib import Path

app = Flask(__name__)

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _f(v):
    try:
        x = float(v)
        return None if (x != x or abs(x) == float("inf")) else x
    except Exception:
        return None


GRID_N = 55   # resolution of the manifold surface grid

_frame_cache: dict = {}


def build_frame(year: int, cluster: int | None = None) -> dict | None:
    mask = df["fy"] == year
    if cluster is not None:
        mask = mask & (df["umap_cluster"] == float(cluster))
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

    # Net margin for dot colouring (log1p stretch keeps it readable)
    nm_raw = pts["net_margin"].clip(-1, 1).values
    nm_vals = np.sign(nm_raw) * np.log1p(np.abs(nm_raw) * 10) / np.log1p(10)  # → [-1,1]

    # ── Smooth manifold surface ───────────────────────────────────────────────
    xmin, xmax = x.min() - 0.8, x.max() + 0.8
    ymin, ymax = y.min() - 0.8, y.max() + 0.8

    xedges = np.linspace(xmin, xmax, GRID_N + 1)
    yedges = np.linspace(ymin, ymax, GRID_N + 1)
    xi     = (xedges[:-1] + xedges[1:]) / 2   # cell centres
    yi     = (yedges[:-1] + yedges[1:]) / 2

    # Median log P/E per cell (NaN where no companies)
    valid_z = np.where(np.isfinite(z_vals), z_vals, np.nan)
    z_stat, _, _, _ = binned_statistic_2d(
        x, y, valid_z, statistic="median", bins=[xedges, yedges]
    )
    z_stat = z_stat.T  # → shape (ny, nx)

    # Fill empty cells with global median, smooth, restore NaN mask
    global_med = float(np.nanmedian(valid_z)) if np.any(np.isfinite(valid_z)) else 1.0
    has_data   = ~np.isnan(z_stat)
    z_filled   = np.where(np.isnan(z_stat), global_med, z_stat)
    z_smooth   = gaussian_filter(z_filled, sigma=1.8)
    z_smooth[~has_data] = np.nan  # blank cells outside company footprint

    # Surface colour = median P/B per cell (green=cheap, red=expensive)
    pb_raw  = pts["pb_ratio"].values
    pb_safe = np.where((pb_raw > 0) & np.isfinite(pb_raw), pb_raw, np.nan)
    pb_log  = np.where(np.isfinite(pb_safe), np.log10(np.clip(pb_safe, 0.01, 100)), np.nan)
    pb_stat, _, _, _ = binned_statistic_2d(
        x, y, pb_log, statistic="median", bins=[xedges, yedges]
    )
    pb_stat  = pb_stat.T
    pb_filled = np.where(np.isnan(pb_stat), float(np.nanmedian(pb_log[np.isfinite(pb_log)]) if np.any(np.isfinite(pb_log)) else 0), pb_stat)
    pb_smooth = gaussian_filter(pb_filled, sigma=1.8)
    pb_smooth[~has_data] = np.nan

    # ── Per-company deviation from the smooth surface ─────────────────────────
    z_surf_vals = np.where(np.isnan(z_smooth), global_med, z_smooth)
    rgi = RegularGridInterpolator(
        (yi, xi), z_surf_vals, method="linear",
        bounds_error=False, fill_value=global_med,
    )
    z_surface_at_pts = rgi(np.column_stack([y, x]))
    z_dot  = np.where(np.isfinite(z_vals), z_vals, global_med)
    z_dev  = z_dot - z_surface_at_pts   # + = above surface (higher P/E than sector peers)

    # Dot size: 3px (normal) → 10px (large outlier, |z_dev| > 0.6 log units)
    dot_size = np.clip(3 + np.abs(z_dev) / 0.1, 3, 12).tolist()

    # Dot colour: deviation normalised to [-1, 1] (red=expensive vs peers, green=cheap)
    dot_color = np.clip(z_dev / 0.6, -1, 1).tolist()

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
            "zdev": float(round(z_dev[i], 3)),
        }
        for i, r in enumerate(pts.itertuples())
    ]

    def _grid(arr):
        return [[None if not np.isfinite(v) else round(float(v), 4) for v in row]
                for row in arr.tolist()]

    return {
        "year": year,
        "n":    len(pts),
        "surface": {
            "x":  [round(float(v), 3) for v in xi],
            "y":  [round(float(v), 3) for v in yi],
            "z":  _grid(z_smooth),
            "pb": _grid(pb_smooth),
        },
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
                                  cluster_info=CLUSTER_INFO)


@app.route("/api/years")
def api_years():
    return jsonify({"years": YEARS})


@app.route("/api/clusters")
def api_clusters():
    return jsonify(CLUSTER_INFO)


@app.route("/api/frame/<int:year>")
def api_frame(year):
    cluster = request.args.get("cluster", default=None, type=int)
    key = (year, cluster)
    if key not in _frame_cache:
        data = build_frame(year, cluster=cluster)
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


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EDGAR Market Manifold</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden}
#plot{width:100vw;height:100vh}

/* top bar */
#topbar{
  position:fixed;top:0;left:0;right:0;z-index:100;
  display:flex;align-items:center;gap:10px;padding:8px 16px;
  background:rgba(13,17,23,.9);backdrop-filter:blur(6px);
  border-bottom:1px solid #21262d;
}
#title{font-size:12px;color:#8b949e;white-space:nowrap}
#year-lbl{font-size:15px;font-weight:700;color:#58a6ff;min-width:44px}
#yr-slider{flex:1;accent-color:#58a6ff;cursor:pointer;min-width:120px}
#play-btn{
  background:#21262d;border:1px solid #30363d;border-radius:6px;
  color:#e6edf3;padding:5px 12px;font-size:12px;cursor:pointer;white-space:nowrap
}
#play-btn:hover{background:#30363d}
#co-count{font-size:11px;color:#6e7681;white-space:nowrap}

/* cluster filter bar */
#clbar{
  position:fixed;top:46px;left:50%;transform:translateX(-50%);
  z-index:100;display:flex;gap:6px;align-items:center;
  background:rgba(13,17,23,.88);padding:5px 12px;
  border-radius:0 0 8px 8px;border:1px solid #21262d;border-top:none;
}
.clbtn{
  background:#21262d;border:1px solid #30363d;border-radius:5px;
  color:#8b949e;padding:4px 11px;font-size:11px;cursor:pointer;white-space:nowrap;
  transition:all .15s;
}
.clbtn:hover{background:#30363d;color:#e6edf3}
.clbtn.active{border-color:#58a6ff;color:#58a6ff;background:#0d1f33}
#cl-info{font-size:11px;color:#6e7681;white-space:nowrap;margin-left:4px}

/* legend */
#legend{
  position:fixed;top:92px;left:16px;z-index:200;
  background:rgba(22,27,34,.85);border:1px solid #30363d;border-radius:6px;
  padding:8px 12px;font-size:11px;color:#6e7681;line-height:1.9;
}
#legend b{color:#e6edf3}

/* search */
#sw{position:fixed;top:92px;right:16px;z-index:200;width:260px}
#si{
  width:100%;background:#161b22;border:1px solid #30363d;border-radius:6px;
  color:#e6edf3;padding:7px 12px;font-size:13px;outline:none;
}
#si:focus{border-color:#58a6ff}
#si::placeholder{color:#6e7681}
#sd{
  background:#161b22;border:1px solid #30363d;border-radius:0 0 6px 6px;
  max-height:230px;overflow-y:auto;display:none;
}
.si{padding:8px 12px;font-size:12px;cursor:pointer;border-bottom:1px solid #21262d;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.si:hover{background:#21262d;color:#58a6ff}

/* info panel */
#ip{
  position:fixed;bottom:16px;left:16px;z-index:200;
  background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:14px 16px 12px;width:280px;
  box-shadow:0 4px 24px rgba(0,0,0,.55);display:none;
}
#ip h3{font-size:13px;color:#58a6ff;margin-bottom:10px;line-height:1.3;padding-right:18px}
.rg{display:grid;grid-template-columns:1fr auto;gap:4px 12px;font-size:12px}
.rl{color:#6e7681}.rv{color:#e6edf3;font-weight:500;text-align:right}
.dv{margin-top:8px;padding-top:8px;border-top:1px solid #21262d;font-size:11px}
.dr{display:flex;justify-content:space-between;margin-top:4px;color:#8b949e}
.dp{color:#f85149}.dn{color:#3fb950}
.cl-badge{
  display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;
  font-weight:600;margin-bottom:8px;
}
#ic{position:absolute;top:8px;right:10px;cursor:pointer;color:#6e7681;font-size:15px;line-height:1}
#ic:hover{color:#e6edf3}

/* loading */
#ld{position:fixed;inset:0;z-index:999;background:#0d1117;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:12px;font-size:14px;color:#8b949e;}
#ld.hidden{display:none}
.sp{width:36px;height:36px;border:3px solid #21262d;border-top-color:#58a6ff;
    border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<div id="ld"><div class="sp"></div><span>Loading manifold…</span></div>

<div id="topbar">
  <span id="title">EDGAR Market Manifold</span>
  <span id="year-lbl">—</span>
  <input id="yr-slider" type="range" min="0" max="0" value="0" step="1">
  <button id="play-btn">&#9654; Play</button>
  <span id="co-count"></span>
</div>

<div id="clbar">
  <span style="font-size:11px;color:#6e7681;margin-right:2px">Slice:</span>
  <button class="clbtn active" data-cl="-1">All</button>
  <button class="clbtn" data-cl="0" id="cl0btn">—</button>
  <button class="clbtn" data-cl="1" id="cl1btn">—</button>
  <button class="clbtn" data-cl="2" id="cl2btn">—</button>
  <span id="cl-info"></span>
</div>

<div id="legend">
  <b>Surface Z</b> P/E Ratio &nbsp;·&nbsp; <b>Surface colour</b> P/B Ratio<br>
  <span style="color:#3fb950">●</span> cheap vs slice peers (low P/E)
  &nbsp;<span style="color:#f85149">●</span> expensive vs slice peers (high P/E)<br>
  Dot size ∝ deviation from local surface
</div>

<div id="sw">
  <input id="si" placeholder="Search company…" autocomplete="off">
  <div id="sd"></div>
</div>

<div id="ip">
  <span id="ic" onclick="closeIP()">&#x2715;</span>
  <h3 id="ipn"></h3>
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

var playing = false, playTimer = null, hlExists = false, abortCtrl = null;

// Populate cluster buttons with labels
[0,1,2].forEach(function(c){
  var btn = document.getElementById('cl'+c+'btn');
  var info = CL_INFO[c];
  if (btn && info) btn.textContent = info.label || ('Slice '+c);
});

// ── Cluster colours for badge ────────────────────────────────────────────────
var CL_COLORS = ['#58a6ff','#3fb950','#ffa657'];
var CL_BG     = ['#0d1f33','#0d2416','#2b1d0e'];

// ── Shared Plotly layout ──────────────────────────────────────────────────────
var LAY = {
  paper_bgcolor:'#0d1117', plot_bgcolor:'#0d1117',
  margin:{l:0,r:0,t:44,b:0},
  scene:{
    bgcolor:'#0d1117',
    xaxis:{title:{text:'UMAP-X (financial similarity)',font:{color:'#8b949e',size:9}},
           backgroundcolor:'#161b22',gridcolor:'#30363d',showbackground:true,
           tickfont:{color:'#6e7681',size:8},zeroline:false},
    yaxis:{title:{text:'UMAP-Y (financial similarity)',font:{color:'#8b949e',size:9}},
           backgroundcolor:'#161b22',gridcolor:'#30363d',showbackground:true,
           tickfont:{color:'#6e7681',size:8},zeroline:false},
    zaxis:{
      title:{text:'P/E Ratio',font:{color:'#8b949e',size:9}},
      backgroundcolor:'#161b22',gridcolor:'#30363d',showbackground:true,
      tickfont:{color:'#6e7681',size:8},
      tickvals:[0.3,0.7,1.0,1.3,1.7,2.0,2.3],
      ticktext:['2×','5×','10×','20×','50×','100×','200×'],
      zeroline:true,zerolinecolor:'#444c56',
    },
    camera:{eye:{x:1.5,y:1.5,z:1.1}}, aspectmode:'auto',
  },
  hoverlabel:{bgcolor:'#161b22',bordercolor:'#58a6ff',font:{color:'#e6edf3',size:12}},
  font:{color:'#e6edf3'},
};

// ── Init plot ─────────────────────────────────────────────────────────────────
Plotly.newPlot('plot',
  [{type:'surface',x:[0,1],y:[0,1],z:[[0,0],[0,0]],opacity:0,showscale:false}],
  LAY,
  {responsive:true,displayModeBar:true,
   modeBarButtonsToRemove:['lasso2d','select2d'],
   toImageButtonOptions:{format:'png',filename:'market_manifold',scale:2}}
).then(function(){
  document.getElementById('plot').on('plotly_click', onClik);
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
    document.querySelectorAll('.clbtn').forEach(function(b){b.classList.remove('active');});
    btn.classList.add('active');
    curCluster = parseInt(btn.dataset.cl, 10);
    // Show cluster info
    var ci = document.getElementById('cl-info');
    if (curCluster >= 0 && CL_INFO[curCluster]){
      var inf = CL_INFO[curCluster];
      var pe = inf.pe ? inf.pe.toFixed(1)+'×' : '—';
      var nm = inf.nm ? (inf.nm*100).toFixed(1)+'%' : '—';
      ci.textContent = inf.n.toLocaleString()+' companies · P/E '+pe+' · margin '+nm;
    } else {
      ci.textContent = '';
    }
    // Reload frame with cluster filter
    _frameCache = {};  // clear cache when cluster changes
    loadYear(curIdx);
  });
});

// ── Frame cache + fetch ───────────────────────────────────────────────────────
var _frameCache = {};
function loadYear(idx){
  var yr = YEARS[idx];
  document.getElementById('year-lbl').textContent = yr;
  sl.value = idx;
  var key = yr + ':' + curCluster;
  if (_frameCache[key]){ renderFrame(_frameCache[key]); return; }
  if (abortCtrl) abortCtrl.abort();
  abortCtrl = new AbortController();
  var url = '/api/frame/'+yr+(curCluster>=0?'?cluster='+curCluster:'');
  fetch(url,{signal:abortCtrl.signal})
    .then(function(r){return r.json();})
    .then(function(d){_frameCache[key]=d; renderFrame(d);})
    .catch(function(e){if(e.name!=='AbortError') console.error(e);});
}

function renderFrame(d){
  var surf = {
    type:'surface',
    x:d.surface.x, y:d.surface.y, z:d.surface.z,
    surfacecolor:d.surface.pb,
    colorscale:'RdYlGn_r',
    cmin:-0.3, cmax:1.5,
    showscale:true,
    colorbar:{
      title:{text:'P/B Ratio',font:{color:'#8b949e',size:10}},
      x:1.01,thickness:13,len:0.5,
      tickvals:[-0.3,0,0.5,1.0,1.5],
      ticktext:['0.5×','1×','3×','10×','30×'],
      tickfont:{color:'#8b949e',size:9},
    },
    opacity:0.72,
    hoverinfo:'skip',
    lighting:{ambient:0.65,diffuse:0.85,specular:0.05,roughness:0.9},
    contours:{x:{show:false},y:{show:false},z:{show:false}},
  };

  var htxt = d.vertices.map(function(v){
    return '<b>'+esc(v.name)+'</b><br>'+
      'P/E: '+fmtPE(v.pe)+'<br>'+
      'P/B: '+(v.pb!=null?v.pb.toFixed(1)+'×':'—')+'<br>'+
      'Net Margin: '+fmtPct(v.nm)+'<br>'+
      'vs peers: '+(v.zdev>0?'+':'')+fmtPct(v.zdev)+' P/E';
  });

  var dots = {
    type:'scatter3d', mode:'markers',
    x:d.dots.x, y:d.dots.y, z:d.dots.z,
    customdata:d.vertices,
    text:d.vertices.map(function(v){return v.name;}),
    hovertext:htxt,
    hovertemplate:'%{hovertext}<extra></extra>',
    marker:{
      size:d.dots.size,
      color:d.dots.color,
      colorscale:[[0,'#3fb950'],[0.5,'#8b949e'],[1,'#f85149']],
      cmin:-1, cmax:1,
      showscale:false,
      opacity:0.88,
      line:{width:0},
    },
  };

  // Always reset hlExists before react — react replaces ALL traces with [surf,dots],
  // so trace 2 (the highlight) is gone regardless. deleteTraces first to keep Plotly's
  // internal state consistent; if it fails (trace already absent) we still call react.
  if(hlExists){
    hlExists = false;
    Plotly.deleteTraces('plot', 2).catch(function(){/* trace may already be absent */}).finally(function(){
      Plotly.react('plot', [surf, dots], LAY);
    });
  } else {
    Plotly.react('plot', [surf, dots], LAY);
  }

  document.getElementById('co-count').textContent = d.n.toLocaleString()+' companies';
  document.getElementById('ld').classList.add('hidden');
}

// ── Click handler ─────────────────────────────────────────────────────────────
function onClik(data){
  if(!data||!data.points||!data.points.length) return;
  // data.points may include surface points (no customdata) when the click lands
  // near both a surface cell and a dot. Find the first dot point (has customdata).
  var pt = null;
  for(var i=0;i<data.points.length;i++){
    if(data.points[i].customdata && data.points[i].customdata.cik != null){
      pt = data.points[i]; break;
    }
  }
  if(!pt) return;
  fetchAndShowCo(pt.customdata.cik, YEARS[curIdx]);
}

// ── Company info ──────────────────────────────────────────────────────────────
function fetchAndShowCo(cik, preferYear){
  fetch('/api/company/'+cik)
    .then(function(r){return r.json();})
    .then(function(recs){
      if(!recs.length) return;
      var rec = recs.find(function(r){return r.fy===preferYear;}) || recs[recs.length-1];
      renderIP(rec);
      hlPoint(rec);
    }).catch(console.error);
}

function closeIP(){ document.getElementById('ip').style.display='none'; }

function renderIP(rec){
  document.getElementById('ipn').textContent = rec.name || '—';

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
  if(rec.zdev!=null){
    var sign=rec.zdev>=0?'+':'';
    rows.push(['vs Slice Peers', sign+fmtPct(rec.zdev)+' P/E']);
  }

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

// ── Highlight dot ─────────────────────────────────────────────────────────────
function hlPoint(rec){
  if(rec.ux==null||rec.uy==null) return;
  var trace={
    type:'scatter3d', mode:'markers+text',
    x:[rec.ux], y:[rec.uy], z:[2.8],
    text:[rec.name], textposition:'top center',
    textfont:{size:11,color:'#ffa657'},
    marker:{size:11,color:'#ffa657',symbol:'diamond',line:{color:'#fff',width:2}},
    hoverinfo:'skip', showlegend:false, name:'hl',
  };
  function addHl(){
    Plotly.addTraces('plot', trace).then(function(){ hlExists = true; });
  }
  if(hlExists){
    hlExists = false;
    Plotly.deleteTraces('plot', 2).catch(function(){/* already gone */}).finally(addHl);
  } else {
    addHl();
  }
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


if __name__ == "__main__":
    print("Starting EDGAR Market Manifold at http://localhost:5001")
    app.run(debug=False, host="127.0.0.1", port=5001, threaded=True)
