#!/usr/bin/env python3
"""
build_edgarvis.py — build a static version of the EDGAR Market Manifold
(the `/` route only) for GitHub Pages deployment.

Output:
    <repo>/build_output/EdgarVis/
        index.html
        data/
            years.json
            clusters.json
            frame_<year>.json          (cluster=None)
            frame_<year>_c<n>.json     (n in {0,1,2})
            frame_<year>_complete.json
            frame_<year>_c<n>_complete.json
            companies.json             (search index)
            company_<cik>.json         (per-CIK detail, one file per company)
            random_pool.json           (replaces /api/random)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# ── Import the live serve.py (no modifications) ────────────────────────────────
SERVE_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, SERVE_DIR)
import serve  # noqa: E402  (importing serve runs its load + KMeans bootstrap)

import pandas as pd  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent / "build_output" / "EdgarVis"
DATA_DIR = OUT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj) -> int:
    """Write JSON compactly. Returns bytes written."""
    s = json.dumps(obj, separators=(",", ":"), allow_nan=False, default=_json_default)
    path.write_bytes(s.encode("utf-8"))
    return len(s)


def _json_default(o):
    # Robustly serialize numpy / pandas scalars if any leaked through
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if (v == v and v not in (float("inf"), float("-inf"))) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    raise TypeError(f"unserializable: {type(o)}")


# ── 1. years.json ──────────────────────────────────────────────────────────────
years_path = DATA_DIR / "years.json"
years_bytes = _write_json(years_path, {"years": serve.YEARS})
print(f"  years.json           {years_bytes:>10,} B  ({len(serve.YEARS)} years)")

# ── 2. clusters.json ───────────────────────────────────────────────────────────
clusters_path = DATA_DIR / "clusters.json"
clusters_bytes = _write_json(clusters_path, serve.CLUSTER_INFO)
print(f"  clusters.json        {clusters_bytes:>10,} B")

# ── 3. frame_<year>[_cN][_complete].json ───────────────────────────────────────
frame_total = 0
frame_count = 0
for yr in serve.YEARS:
    for cl in (None, 0, 1, 2):
        for complete in (False, True):
            data = serve.build_frame(yr, cluster=cl, complete=complete)
            if data is None:
                continue
            suffix = ""
            if cl is not None:
                suffix += f"_c{cl}"
            if complete:
                suffix += "_complete"
            p = DATA_DIR / f"frame_{yr}{suffix}.json"
            frame_total += _write_json(p, data)
            frame_count += 1
print(f"  frame_*.json         {frame_total:>10,} B  ({frame_count} files)")

# ── 4. companies.json (search index) ───────────────────────────────────────────
# COMPANY_INDEX = {cik: name}. Convert to a sorted list of {cik, name}.
companies = sorted(
    [{"cik": int(c), "name": str(n)} for c, n in serve.COMPANY_INDEX.items()],
    key=lambda x: x["name"].lower(),
)
companies_path = DATA_DIR / "companies.json"
companies_bytes = _write_json(companies_path, companies)
print(f"  companies.json       {companies_bytes:>10,} B  ({len(companies):,} companies)")

# ── 5. company_<cik>.json (per-CIK /api/company/<cik> equivalent) ─────────────
# Replicate the route handler logic over every CIK present in serve.df.
df = serve.df
_f = serve._f

# Clear any existing per-CIK files from a previous run before writing fresh ones.
for old in DATA_DIR.glob("company_*.json"):
    old.unlink()
# Also remove the old bundle if present.
old_bundle = DATA_DIR / "companies_detail.json"
if old_bundle.exists():
    old_bundle.unlink()

# Group by cik for one pass instead of per-CIK filtering.
# Restrict to CIKs reachable via the search index (i.e. have UMAP coords).
reachable_ciks = set(int(c) for c in serve.COMPANY_INDEX.keys())
needed_cols = [
    "fy", "umap_cluster", "net_margin", "pe_ratio", "pb_ratio",
    "revenue_growth", "revenue", "net_income", "assets", "equity",
    "op_cash", "roe", "debt_to_equity", "manifold_distance_pct",
    "umap_x", "umap_y", "entity_name",
    "dev_log_revenue", "dev_log_public_float", "dev_net_margin",
    "dev_op_margin", "dev_revenue_growth", "dev_debt_to_equity",
    "dev_roe", "dev_op_cash_yield",
]
# Some dev_* columns might be missing — be tolerant.
present_cols = [c for c in needed_cols if c in df.columns]
sub = df[present_cols + (["cik"] if "cik" in df.columns else [])].copy()
sub = sub[sub["cik"].astype(int).isin(reachable_ciks)]
sub = sub.sort_values("fy")

dev_specs = [
    ("Scale",    "dev_log_revenue"),
    ("Float",    "dev_log_public_float"),
    ("Margin",   "dev_net_margin"),
    ("OpMargin", "dev_op_margin"),
    ("Growth",   "dev_revenue_growth"),
    ("Leverage", "dev_debt_to_equity"),
    ("ROE",      "dev_roe"),
    ("CashYld",  "dev_op_cash_yield"),
]


def _cell(row, col):
    return row[col] if col in row.index else None


detail_total = 0
detail_count = 0
for cik, group in sub.groupby("cik", sort=False):
    out = []
    for _, r in group.iterrows():
        devs = {label: _f(_cell(r, col)) for label, col in dev_specs}
        ent = _cell(r, "entity_name")
        out.append({
            "name": str(ent) if pd.notna(ent) else "",
            "cik":  int(cik),
            "fy":   int(r["fy"]),
            "cl":   int(r["umap_cluster"]) if pd.notna(_cell(r, "umap_cluster")) else -1,
            "nm":   _f(_cell(r, "net_margin")),
            "pe":   _f(_cell(r, "pe_ratio")),
            "pb":   _f(_cell(r, "pb_ratio")),
            "rg":   _f(_cell(r, "revenue_growth")),
            "rev":  _f(_cell(r, "revenue")),
            "ni":   _f(_cell(r, "net_income")),
            "ast":  _f(_cell(r, "assets")),
            "eq":   _f(_cell(r, "equity")),
            "oc":   _f(_cell(r, "op_cash")),
            "roe":  _f(_cell(r, "roe")),
            "de":   _f(_cell(r, "debt_to_equity")),
            "mdp":  _f(_cell(r, "manifold_distance_pct")),
            "ux":   _f(_cell(r, "umap_x")),
            "uy":   _f(_cell(r, "umap_y")),
            "devs": devs,
        })
    p = DATA_DIR / f"company_{int(cik)}.json"
    detail_total += _write_json(p, out)
    detail_count += 1
print(f"  company_*.json       {detail_total:>10,} B  ({detail_count:,} files)")

# ── 6. random_pool.json (replicates /api/random) ───────────────────────────────
# /api/random samples from rows with valid (umap_x, umap_y, mdp, name)
# in YEARS where mdp >= 80. We just dump the whole eligible pool;
# the client picks one at random.
pool_df = (
    df.dropna(subset=["umap_x", "umap_y", "manifold_distance_pct", "entity_name"])
    .loc[lambda d: (d["fy"].isin(serve.YEARS)) & (d["manifold_distance_pct"] >= 80)]
)
pool = [
    {"cik": int(r.cik), "name": str(r.entity_name), "fy": int(r.fy)}
    for r in pool_df.itertuples()
]
pool_path = DATA_DIR / "random_pool.json"
pool_bytes = _write_json(pool_path, pool)
print(f"  random_pool.json     {pool_bytes:>10,} B  ({len(pool):,} candidates)")

# ── 7. index.html ──────────────────────────────────────────────────────────────
# Render the Jinja template once, then surgically rewrite the API fetches.
from jinja2 import Template  # noqa: E402

tmpl = Template(serve.HTML)
html = tmpl.render(
    years=serve.YEARS,
    first_year=serve.YEARS[-1],
    cluster_info=serve.CLUSTER_INFO,
    insights_url="../EdgarInsights/",
)

# Rewrite fetches:
#
# (a) /api/years      → ./data/years.json
# (b) /api/clusters   → ./data/clusters.json   (not actually fetched; CL_INFO is inlined)
# (c) /api/frame/...  → ./data/frame_<yr>[_cN][_complete].json
# (d) /api/search?q=  → load companies.json once, filter client-side
# (e) /api/company/X  → look up X in companies_detail.json bundle
# (f) /api/random     → pick from random_pool.json

# (c) replace the URL build in loadYear()
old_frame = (
    "  var params = [];\n"
    "  if(curCluster>=0) params.push('cluster='+curCluster);\n"
    "  if(curComplete)   params.push('complete=1');\n"
    "  var url = '/api/frame/'+yr+(params.length?'?'+params.join('&'):'');"
)
new_frame = (
    "  var suffix = '';\n"
    "  if(curCluster>=0) suffix += '_c'+curCluster;\n"
    "  if(curComplete)   suffix += '_complete';\n"
    "  var url = './data/frame_'+yr+suffix+'.json';"
)
assert old_frame in html, "frame URL block not found in HTML"
html = html.replace(old_frame, new_frame)

# (d) replace search fetch with client-side filter against companies.json
old_search = (
    "    fetch('/api/search?q='+encodeURIComponent(q))\n"
    "      .then(function(r){return r.json();})\n"
    "      .then(showResults).catch(console.error);"
)
new_search = (
    "    if(_companyIndex){ showResults(_searchLocal(q)); return; }\n"
    "    fetch('./data/companies.json').then(function(r){return r.json();})\n"
    "      .then(function(arr){ _companyIndex = arr; showResults(_searchLocal(q)); })\n"
    "      .catch(console.error);"
)
assert old_search in html, "search fetch not found in HTML"
html = html.replace(old_search, new_search)

# (e) replace per-company fetch URL — keep the same .then chain, just swap the URL
old_company_full = (
    "  fetch('/api/company/'+cik)\n"
    "    .then(function(r){return r.json();})\n"
)
new_company_full = (
    "  fetch('./data/company_'+cik+'.json')\n"
    "    .then(function(r){return r.json();})\n"
)
assert old_company_full in html, "company fetch block not found in HTML"
html = html.replace(old_company_full, new_company_full)

# (f) replace /api/random fetch with random_pool sampling
old_random = (
    "  fetch('/api/random')\n"
    "    .then(function(r){ return r.json(); })\n"
    "    .then(function(d){"
)
new_random = (
    "  _ensureRandomPool().then(function(pool){\n"
    "      var d = pool.length ? pool[Math.floor(Math.random()*pool.length)] : {};\n"
)
old_random_full = (
    "  fetch('/api/random')\n"
    "    .then(function(r){ return r.json(); })\n"
    "    .then(function(d){\n"
    "      if(!d.cik){ btn.disabled=false; return; }\n"
    "      var idx = YEARS.indexOf(d.fy);\n"
    "      if(idx < 0) idx = YEARS.length - 1;\n"
    "      curIdx = idx; sl.value = idx;\n"
    "      loadYear(idx, function(){\n"
    "        fetchAndShowCo(d.cik, d.fy);\n"
    "        btn.disabled = false;\n"
    "      });\n"
    "    })\n"
    "    .catch(function(){ btn.disabled = false; });"
)
new_random_full = (
    "  _ensureRandomPool().then(function(pool){\n"
    "      var d = pool.length ? pool[Math.floor(Math.random()*pool.length)] : {};\n"
    "      if(!d.cik){ btn.disabled=false; return; }\n"
    "      var idx = YEARS.indexOf(d.fy);\n"
    "      if(idx < 0) idx = YEARS.length - 1;\n"
    "      curIdx = idx; sl.value = idx;\n"
    "      loadYear(idx, function(){\n"
    "        fetchAndShowCo(d.cik, d.fy);\n"
    "        btn.disabled = false;\n"
    "      });\n"
    "    })\n"
    "    .catch(function(){ btn.disabled = false; });"
)
assert old_random_full in html, "random fetch block not found in HTML"
html = html.replace(old_random_full, new_random_full)

# Inject the helper functions + state for the static fetches.
# Insert just before the closing </script> of the inline script.
helpers = """
// ── Static-build helpers ─────────────────────────────────────────────────────
var _companyIndex = null;            // [{cik,name},...]
var _randomPoolPromise = null;       // Promise<[{cik,name,fy},...]>

function _searchLocal(q){
  var ql = q.toLowerCase();
  if(ql.length < 2) return [];
  var out = [];
  for(var i=0; i<_companyIndex.length && out.length < 25; i++){
    var c = _companyIndex[i];
    if(c.name.toLowerCase().indexOf(ql) !== -1) out.push(c);
  }
  out.sort(function(a,b){ return a.name.localeCompare(b.name); });
  return out;
}
function _ensureRandomPool(){
  if(_randomPoolPromise) return _randomPoolPromise;
  _randomPoolPromise = fetch('./data/random_pool.json').then(function(r){return r.json();});
  return _randomPoolPromise;
}
"""
html = html.replace("</script>\n</body>", helpers + "</script>\n</body>")

# Sanity: no remaining /api/ references should be in the output (other than
# external sec.gov URLs etc).
remaining = re.findall(r"/api/[a-zA-Z_]+", html)
if remaining:
    print(f"  WARNING: residual /api/ references: {set(remaining)}")

(OUT_ROOT / "index.html").write_text(html, encoding="utf-8")
html_bytes = len(html.encode("utf-8"))
print(f"  index.html           {html_bytes:>10,} B")

# ── Summary ───────────────────────────────────────────────────────────────────
def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

total = _dir_size(OUT_ROOT)
data_size = _dir_size(DATA_DIR)
print()
print(f"  TOTAL                {total:>10,} B  ({total/1e6:.1f} MB)")
print(f"    data/              {data_size:>10,} B  ({data_size/1e6:.1f} MB)")
print(f"    index.html         {html_bytes:>10,} B")
