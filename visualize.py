# visualize.py
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.spatial import Delaunay
import plotly.graph_objects as go

_test = "--test" in sys.argv
INPUT  = Path("/Users/rezaramji/Documents/CCC/EDGARmaxxing/" + ("ratios_test.parquet"         if _test else "ratios.parquet"))
OUTPUT = Path("/Users/rezaramji/Documents/CCC/EDGARmaxxing/" + ("market_manifold_test.html"   if _test else "market_manifold.html"))


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------

def build_frame(year_df: pd.DataFrame, year: int):
    """Build one animation frame (go.Frame) for a fiscal year.

    Returns None if there are fewer than 10 valid rows.
    """
    pts = year_df.dropna(subset=["umap_x", "umap_y", "net_margin"]).copy()
    if len(pts) < 10:
        return None

    pts = pts.reset_index(drop=True)

    xy = pts[["umap_x", "umap_y"]].values
    tri = Delaunay(xy)

    # Safe formatting helpers — values may be NaN
    def _fmt_float(v, fmt=".2f"):
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "N/A"

    # Ensure manifold cols exist (may be absent in test runs)
    for mc in ["manifold_distance", "manifold_distance_pct", "dev_net_margin", "dev_pe_ratio", "dev_revenue_growth"]:
        if mc not in pts.columns:
            pts[mc] = np.nan

    customdata = pts[[
        "entity_name",           # [0]
        "cik",                   # [1]
        "fy",                    # [2]
        "net_margin",            # [3]
        "pe_ratio",              # [4]
        "pb_ratio",              # [5]
        "revenue_growth",        # [6]
        "revenue",               # [7]
        "manifold_distance",     # [8]
        "manifold_distance_pct", # [9]
    ]].values

    # Build hover text per vertex
    hover_lines = []
    for row in customdata:
        nm  = row[3]
        pe  = row[4]
        pb  = row[5]
        rg  = row[6]
        nm_str = f"{float(nm)*100:.1f}%" if pd.notna(nm) else "N/A"
        pe_str = f"{float(pe):.1f}"      if pd.notna(pe)  else "N/A"
        pb_str = f"{float(pb):.1f}"      if pd.notna(pb)  else "N/A"
        rg_str = f"{float(rg)*100:.1f}%" if pd.notna(rg)  else "N/A"
        hover_lines.append(
            f"<b>{row[0]}</b><br>"
            f"FY {row[2]}<br>"
            f"Net Margin: {nm_str}<br>"
            f"P/E: {pe_str}<br>"
            f"P/B: {pb_str}<br>"
            f"Rev Growth: {rg_str}"
        )

    mesh = go.Mesh3d(
        x=pts["umap_x"].values,
        y=pts["umap_y"].values,
        z=pts["net_margin"].clip(-0.5, 1.0).values,
        i=tri.simplices[:, 0],
        j=tri.simplices[:, 1],
        k=tri.simplices[:, 2],
        intensity=pts["pb_ratio"].fillna(10).clip(0, 20).values,
        colorscale="RdYlGn_r",
        cmin=0,
        cmax=20,
        text=pts["entity_name"].values,
        customdata=customdata,
        hovertext=hover_lines,
        hovertemplate="%{hovertext}<extra></extra>",
        name=str(year),
        showscale=True,
        colorbar=dict(title="P/B Ratio", x=1.02),
        opacity=0.85,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1, roughness=0.9),
        lightposition=dict(x=100, y=200, z=150),
    )
    return go.Frame(data=[mesh], name=str(year))


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>EDGAR Market Manifold</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117;
      color: #e6edf3;
      font-family: 'Segoe UI', system-ui, sans-serif;
      overflow: hidden;
    }}
    #manifold-plot {{
      width: 100vw;
      height: 100vh;
    }}
    #search-container {{
      position: fixed;
      top: 16px;
      right: 16px;
      z-index: 1000;
      display: flex;
      gap: 6px;
      align-items: center;
    }}
    #company-search {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #e6edf3;
      padding: 7px 12px;
      font-size: 13px;
      width: 240px;
      outline: none;
      transition: border-color 0.2s;
    }}
    #company-search:focus {{ border-color: #58a6ff; }}
    #company-search::placeholder {{ color: #6e7681; }}
    #search-btn {{
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 6px;
      color: #e6edf3;
      padding: 7px 14px;
      font-size: 13px;
      cursor: pointer;
      transition: background 0.2s;
    }}
    #search-btn:hover {{ background: #30363d; }}
    #search-results {{
      position: fixed;
      top: 52px;
      right: 16px;
      z-index: 1000;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 6px;
      max-height: 220px;
      overflow-y: auto;
      width: 260px;
      display: none;
    }}
    .search-result-item {{
      padding: 8px 12px;
      font-size: 12px;
      cursor: pointer;
      border-bottom: 1px solid #21262d;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .search-result-item:hover {{ background: #21262d; color: #58a6ff; }}
    #info-panel {{
      position: fixed;
      bottom: 80px;
      left: 16px;
      z-index: 1000;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 16px 18px;
      min-width: 260px;
      max-width: 320px;
      display: none;
      box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }}
    #info-panel h3 {{
      font-size: 14px;
      color: #58a6ff;
      margin-bottom: 10px;
      line-height: 1.3;
    }}
    #info-panel .ratio-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 12px;
      font-size: 12px;
    }}
    #info-panel .ratio-label {{ color: #6e7681; }}
    #info-panel .ratio-value {{ color: #e6edf3; font-weight: 500; text-align: right; }}
    #info-close {{
      position: absolute;
      top: 8px;
      right: 10px;
      cursor: pointer;
      color: #6e7681;
      font-size: 16px;
      line-height: 1;
    }}
    #info-close:hover {{ color: #e6edf3; }}
    #year-badge {{
      position: fixed;
      top: 16px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 999;
      background: rgba(22,27,34,0.85);
      border: 1px solid #30363d;
      border-radius: 20px;
      padding: 4px 16px;
      font-size: 13px;
      color: #8b949e;
      pointer-events: none;
    }}
    #year-badge span {{ color: #58a6ff; font-weight: 600; }}
  </style>
</head>
<body>
  <div id="search-container">
    <input id="company-search" placeholder="Search company..." autocomplete="off" />
    <button id="search-btn" onclick="searchCompany()">Find</button>
  </div>
  <div id="search-results"></div>
  <div id="info-panel">
    <span id="info-close" onclick="document.getElementById('info-panel').style.display='none'">&#x2715;</span>
    <div id="info-content"></div>
  </div>
  <div id="year-badge">FY <span id="year-display">—</span></div>
  <div id="manifold-plot"></div>

  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script>
  // -----------------------------------------------------------------------
  // Embedded company data (keyed by cik, each value is list of year records)
  // -----------------------------------------------------------------------
  var COMPANY_DATA = {company_json};

  // -----------------------------------------------------------------------
  // Plotly figure
  // -----------------------------------------------------------------------
  var figData = {fig_json};

  // Track current year (updated by slider / animation)
  var currentYear = {first_year};
  var highlightTraceIdx = null;  // index of the current highlight scatter3d trace

  // Render the figure
  Plotly.newPlot('manifold-plot', figData.data, figData.layout, {{
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d','select2d'],
    toImageButtonOptions: {{ format: 'png', filename: 'market_manifold', scale: 2 }}
  }}).then(function() {{
    document.getElementById('year-display').textContent = currentYear;
  }});

  // -----------------------------------------------------------------------
  // Slider / animation year tracking
  // -----------------------------------------------------------------------
  document.getElementById('manifold-plot').on('plotly_sliderchange', function(e) {{
    currentYear = parseInt(e.slider.active !== undefined
      ? e.slider.steps[e.slider.active].label
      : e.step.label, 10);
    document.getElementById('year-display').textContent = currentYear;
    removeHighlight();
  }});

  document.getElementById('manifold-plot').on('plotly_animated', function() {{
    // sync year badge after animation step
    var gd = document.getElementById('manifold-plot');
    if (gd._fullLayout && gd._fullLayout.sliders && gd._fullLayout.sliders[0]) {{
      var sl = gd._fullLayout.sliders[0];
      var active = sl._input.active || 0;
      if (sl.steps && sl.steps[active]) {{
        currentYear = parseInt(sl.steps[active].label, 10);
        document.getElementById('year-display').textContent = currentYear;
      }}
    }}
  }});

  // -----------------------------------------------------------------------
  // Click handler
  // -----------------------------------------------------------------------
  document.getElementById('manifold-plot').on('plotly_click', function(data) {{
    if (!data || !data.points || data.points.length === 0) return;
    var pt = data.points[0];
    var cd = pt.customdata;
    if (!cd) return;
    showInfoPanel({{
      entity_name:             cd[0],
      cik:                     cd[1],
      fy:                      cd[2],
      net_margin:              cd[3],
      pe_ratio:                cd[4],
      pb_ratio:                cd[5],
      revenue_growth:          cd[6],
      revenue:                 cd[7],
      manifold_distance:       cd[8],
      manifold_distance_pct:   cd[9],
    }});
  }});

  // -----------------------------------------------------------------------
  // Info panel
  // -----------------------------------------------------------------------
  function fmtPct(v) {{
    if (v === null || v === undefined || isNaN(parseFloat(v))) return 'N/A';
    return (parseFloat(v) * 100).toFixed(1) + '%';
  }}
  function fmtNum(v, dec) {{
    if (v === null || v === undefined || isNaN(parseFloat(v))) return 'N/A';
    return parseFloat(v).toFixed(dec !== undefined ? dec : 2);
  }}
  function fmtRevenue(v) {{
    if (v === null || v === undefined || isNaN(parseFloat(v))) return 'N/A';
    var n = parseFloat(v);
    if (Math.abs(n) >= 1e12) return '$' + (n/1e12).toFixed(2) + 'T';
    if (Math.abs(n) >= 1e9)  return '$' + (n/1e9).toFixed(2)  + 'B';
    if (Math.abs(n) >= 1e6)  return '$' + (n/1e6).toFixed(2)  + 'M';
    return '$' + n.toFixed(0);
  }}

  function showInfoPanel(rec) {{
    var html = '<h3>' + escHtml(String(rec.entity_name)) + '</h3>';
    html += '<div class="ratio-grid">';
    var rows = [
      ['CIK',              String(rec.cik)],
      ['Fiscal Year',      String(rec.fy)],
      ['Revenue',          fmtRevenue(rec.revenue)],
      ['Net Margin',       fmtPct(rec.net_margin)],
      ['P/E Ratio',        fmtNum(rec.pe_ratio, 1)],
      ['P/B Ratio',        fmtNum(rec.pb_ratio, 1)],
      ['Rev Growth',       fmtPct(rec.revenue_growth)],
      ['Outlier Score',    rec.manifold_distance_pct !== null && rec.manifold_distance_pct !== undefined ? fmtNum(rec.manifold_distance_pct, 1) + '%ile' : 'N/A'],
    ];
    // Top driving deviations — show which axes push this company off the manifold
    var devFields = [
      ['dev_log_revenue','Scale'],['dev_log_public_float','Float'],
      ['dev_net_margin','Margin'],['dev_op_margin','OpMargin'],
      ['dev_revenue_growth','Growth'],['dev_debt_to_equity','Leverage'],
      ['dev_roe','ROE'],['dev_op_cash_yield','CashYield'],
    ];
    var devRows = devFields
      .filter(function(d) {{ return rec[d[0]] !== null && rec[d[0]] !== undefined && !isNaN(parseFloat(rec[d[0]])); }})
      .map(function(d) {{ return {{ label: d[1], val: parseFloat(rec[d[0]]) }}; }})
      .sort(function(a,b) {{ return Math.abs(b.val) - Math.abs(a.val); }})
      .slice(0, 3);
    if (devRows.length > 0) {{
      rows.push(['— Top deviations —', '']);
      devRows.forEach(function(d) {{
        var sign = d.val >= 0 ? '+' : '';
        rows.push([d.label, sign + d.val.toFixed(2) + 'σ']);
      }});
    }}
    rows.forEach(function(r) {{
      html += '<span class="ratio-label">' + r[0] + '</span>';
      html += '<span class="ratio-value">' + escHtml(r[1]) + '</span>';
    }});
    html += '</div>';
    document.getElementById('info-content').innerHTML = html;
    document.getElementById('info-panel').style.display = 'block';
  }}

  function escHtml(s) {{
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }}

  // -----------------------------------------------------------------------
  // Search
  // -----------------------------------------------------------------------
  document.getElementById('company-search').addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') searchCompany();
    if (e.key === 'Escape') clearSearch();
  }});

  function clearSearch() {{
    document.getElementById('search-results').style.display = 'none';
    removeHighlight();
  }}

  function searchCompany() {{
    var query = document.getElementById('company-search').value.trim().toLowerCase();
    if (!query) {{ clearSearch(); return; }}
    var matches = [];
    for (var cik in COMPANY_DATA) {{
      var records = COMPANY_DATA[cik];
      if (records.length === 0) continue;
      var name = String(records[0].entity_name).toLowerCase();
      if (name.indexOf(query) !== -1) {{
        matches.push({{ cik: cik, name: records[0].entity_name, records: records }});
      }}
      if (matches.length >= 20) break;
    }}
    renderSearchResults(matches);
  }}

  function renderSearchResults(matches) {{
    var el = document.getElementById('search-results');
    if (matches.length === 0) {{
      el.innerHTML = '<div class="search-result-item" style="color:#6e7681">No results found</div>';
      el.style.display = 'block';
      return;
    }}
    var html = '';
    matches.forEach(function(m) {{
      html += '<div class="search-result-item" onclick="selectCompany(\'' +
        escAttr(m.cik) + '\')">' + escHtml(m.name) + '</div>';
    }});
    el.innerHTML = html;
    el.style.display = 'block';
  }}

  function escAttr(s) {{
    return String(s).replace(/'/g, "\\'");
  }}

  function selectCompany(cik) {{
    document.getElementById('search-results').style.display = 'none';
    var records = COMPANY_DATA[cik];
    if (!records || records.length === 0) return;

    // Find record for current year, else most recent
    var rec = null;
    for (var i = 0; i < records.length; i++) {{
      if (parseInt(records[i].fy, 10) === currentYear) {{ rec = records[i]; break; }}
    }}
    if (!rec) {{
      // most recent year
      rec = records.slice().sort(function(a,b) {{ return b.fy - a.fy; }})[0];
    }}

    showInfoPanel(rec);
    highlightCompany(rec);
  }}

  // -----------------------------------------------------------------------
  // Highlight marker
  // -----------------------------------------------------------------------
  function removeHighlight() {{
    if (highlightTraceIdx !== null) {{
      var gd = document.getElementById('manifold-plot');
      var nTraces = gd.data.length;
      if (highlightTraceIdx < nTraces) {{
        Plotly.deleteTraces('manifold-plot', highlightTraceIdx);
      }}
      highlightTraceIdx = null;
    }}
  }}

  function highlightCompany(rec) {{
    removeHighlight();
    // Find umap coords from the rec; they may be under umap_x/umap_y
    var x = parseFloat(rec.umap_x);
    var y = parseFloat(rec.umap_y);
    var z = rec.net_margin !== null && rec.net_margin !== undefined
      ? Math.max(-0.5, Math.min(1.0, parseFloat(rec.net_margin)))
      : 0;
    if (isNaN(x) || isNaN(y)) return;

    var trace = {{
      type: 'scatter3d',
      mode: 'markers+text',
      x: [x],
      y: [y],
      z: [z + 0.05],  // slightly above surface
      text: [String(rec.entity_name)],
      textposition: 'top center',
      textfont: {{ size: 11, color: '#ff4444' }},
      marker: {{
        size: 10,
        color: '#ff4444',
        symbol: 'diamond',
        line: {{ color: '#ffffff', width: 2 }}
      }},
      hoverinfo: 'skip',
      showlegend: false,
      name: 'highlight',
    }};

    var gd = document.getElementById('manifold-plot');
    Plotly.addTraces('manifold-plot', trace).then(function() {{
      highlightTraceIdx = gd.data.length - 1;
    }});
  }}

  // -----------------------------------------------------------------------
  // Close search results when clicking outside
  // -----------------------------------------------------------------------
  document.addEventListener('click', function(e) {{
    var sc = document.getElementById('search-container');
    var sr = document.getElementById('search-results');
    if (!sc.contains(e.target) && !sr.contains(e.target)) {{
      sr.style.display = 'none';
    }}
  }});
  </script>
</body>
</html>
"""


def _safe_val(v):
    """Convert NaN/inf to None for JSON serialization."""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return v


def _record_to_json_safe(row: dict) -> dict:
    """Make a single row dict JSON-safe."""
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            f = float(v)
            out[k] = None if (np.isnan(f) or np.isinf(f)) else f
        elif isinstance(v, (np.ndarray,)):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def build_html(fig: go.Figure, df: pd.DataFrame, first_year: int) -> str:
    """Build the self-contained HTML string."""

    # --- Embed company data keyed by CIK ---
    # Keep only the columns needed for the info panel + search + highlight
    keep_cols = [
        "cik", "entity_name", "fy",
        "net_margin", "pe_ratio", "pb_ratio",
        "revenue_growth", "revenue",
        "umap_x", "umap_y",
        "manifold_distance", "manifold_distance_pct",
        "dev_log_revenue", "dev_log_public_float", "dev_net_margin",
        "dev_op_margin", "dev_revenue_growth", "dev_debt_to_equity",
        "dev_roe", "dev_op_cash_yield",
    ]
    # Only keep columns that actually exist in df
    keep_cols = [c for c in keep_cols if c in df.columns]
    slim = df[keep_cols].copy()

    company_dict: dict[str, list] = {}
    for _, row in slim.iterrows():
        cik_val = str(row["cik"])
        rec = _record_to_json_safe(row.to_dict())
        company_dict.setdefault(cik_val, []).append(rec)

    company_json_str = json.dumps(company_dict, separators=(",", ":"))

    # --- Serialize the Plotly figure to JSON ---
    # Plotly's to_json() can emit bare NaN tokens (not valid JSON).
    # Replace them with null so the browser's JSON.parse() doesn't choke.
    import re as _re
    fig_json_str = fig.to_json()
    fig_json_str = _re.sub(r'\bNaN\b', 'null', fig_json_str)

    html = _HTML_TEMPLATE.format(
        company_json=company_json_str,
        fig_json=fig_json_str,
        first_year=first_year,
    )
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_parquet(INPUT)
    print(f"Loaded {len(df)} rows from {INPUT}")
    print(f"Columns: {list(df.columns)}")

    # Normalise fy to int where possible
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")

    # Filter to rows that have valid umap coordinates
    valid_mask = df["umap_x"].notna() & df["umap_y"].notna()
    print(f"Rows with valid umap coords: {valid_mask.sum()}")

    # Determine qualifying years (>= 50 companies with valid umap)
    year_counts = (
        df[valid_mask]
        .groupby("fy")
        .size()
        .reset_index(name="n")
    )
    min_companies = 5 if _test else 50
    qualifying_years = sorted(
        year_counts.loc[year_counts["n"] >= min_companies, "fy"].dropna().astype(int).tolist()
    )
    print(f"Qualifying years (>= {min_companies} companies): {qualifying_years}")

    if not qualifying_years:
        raise ValueError(
            "No qualifying years found (need >= 50 companies with umap coords per year)."
        )

    # --- Build frames ---
    frames = []
    first_year = qualifying_years[0]
    first_mesh = None

    for yr in qualifying_years:
        year_df = df[df["fy"] == yr]
        frame = build_frame(year_df, yr)
        if frame is not None:
            frames.append(frame)
            if first_mesh is None:
                first_mesh = frame.data[0]

    if not frames:
        raise ValueError("No frames could be built (all years had < 10 valid rows).")

    print(f"Built {len(frames)} animation frames.")

    # --- Build slider steps ---
    frame_names = [f.name for f in frames]

    slider_steps = [
        dict(
            args=[
                [name],
                dict(
                    frame=dict(duration=600, redraw=True),
                    mode="immediate",
                    transition=dict(duration=300),
                ),
            ],
            label=name,
            method="animate",
        )
        for name in frame_names
    ]

    sliders = [
        dict(
            active=0,
            currentvalue=dict(prefix="FY: ", font=dict(color="#e6edf3", size=14)),
            pad=dict(t=50, b=10),
            len=0.85,
            x=0.075,
            steps=slider_steps,
            bgcolor="#21262d",
            bordercolor="#30363d",
            tickcolor="#6e7681",
            font=dict(color="#e6edf3"),
        )
    ]

    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            y=0.02,
            x=0.0,
            xanchor="left",
            yanchor="bottom",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="&#9654; Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=800, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=400, easing="cubic-in-out"),
                        ),
                    ],
                ),
                dict(
                    label="&#9646;&#9646; Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
            bgcolor="#21262d",
            bordercolor="#30363d",
            font=dict(color="#e6edf3"),
        )
    ]

    layout = go.Layout(
        title=dict(
            text="EDGAR Market Manifold — Financial Similarity Landscape",
            font=dict(color="#e6edf3", size=16),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        scene=dict(
            xaxis=dict(
                title=dict(text="UMAP X (Financial Similarity)", font=dict(color="#8b949e", size=10)),
                backgroundcolor="#161b22",
                gridcolor="#30363d",
                showbackground=True,
                tickfont=dict(color="#6e7681", size=9),
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(text="UMAP Y (Financial Similarity)", font=dict(color="#8b949e", size=10)),
                backgroundcolor="#161b22",
                gridcolor="#30363d",
                showbackground=True,
                tickfont=dict(color="#6e7681", size=9),
                zeroline=False,
            ),
            zaxis=dict(
                title=dict(text="Net Margin", font=dict(color="#8b949e", size=10)),
                backgroundcolor="#161b22",
                gridcolor="#30363d",
                showbackground=True,
                tickfont=dict(color="#6e7681", size=9),
                tickformat=".0%",
                range=[-0.5, 1.0],
                zeroline=True,
                zerolinecolor="#444c56",
            ),
            bgcolor="#0d1117",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="auto",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=updatemenus,
        sliders=sliders,
        font=dict(color="#e6edf3"),
        hoverlabel=dict(
            bgcolor="#161b22",
            bordercolor="#58a6ff",
            font=dict(color="#e6edf3", size=12),
        ),
    )

    fig = go.Figure(data=[first_mesh], layout=layout, frames=frames)

    # --- Write HTML ---
    html_str = build_html(fig, df, first_year)
    OUTPUT.write_text(html_str, encoding="utf-8")
    print(f"Written to {OUTPUT}  ({len(html_str):,} bytes)")


if __name__ == "__main__":
    main()
