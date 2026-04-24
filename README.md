# EDGAR Market Manifold

An interactive 3D visualization of the US public equity universe as a financial similarity landscape. Using UMAP dimensionality reduction on 8 financial features extracted from ~17,000 SEC EDGAR 10-K filings, it projects every company into a 2D manifold where nearby companies are financially similar. The third axis and surface color encode valuation (P/E and P/B ratios), so the shape of the market's "terrain" changes year by year as valuations shift.

**Why it's interesting:** Sectors and business models cluster naturally without being told what they are. You can watch the dot-com bubble inflate and collapse, see the 2008 credit crisis reshape financials, and identify companies that stand out from their sector peers (high P/E dots floating above the surface).

---

## Prerequisites

- Python 3.10+
- Dependencies:

```bash
pip install pandas numpy pyarrow umap-learn scikit-learn scipy plotly flask
```

---

## Data Setup

### EDGAR companyfacts

Download the SEC bulk data archive and extract it into `companyfacts/`:

```
https://data.sec.gov/submissions/companyfacts.zip
```

After extraction, the directory should contain ~17,000 JSON files named by CIK (e.g. `CIK0000320193.json`).

### Expected directory structure

```
EDGARmaxxing/
├── companyfacts/          # ~17,034 SEC EDGAR companyfacts JSON files
│   ├── CIK0000320193.json
│   └── ...
├── extract_metrics.py
├── compute_ratios.py
├── serve.py
├── visualize.py
├── metrics.parquet        # produced by extract_metrics.py
└── ratios.parquet         # produced by compute_ratios.py
```

---

## Pipeline

Run the three steps in order:

### Step 1 — Extract metrics

```bash
python extract_metrics.py
```

Reads every JSON file in `companyfacts/`, pulls annual 10-K values for 9 financial metrics, and writes `metrics.parquet` (~78k rows, one per company-year).

Test mode (first 20 files only, writes `metrics_test.parquet`):

```bash
python extract_metrics.py --test
```

### Step 2 — Compute ratios and UMAP

```bash
python compute_ratios.py
```

Reads `metrics.parquet`, computes financial ratios, fits a UMAP embedding across all qualifying rows, computes a manifold-distance outlier score, and writes `ratios.parquet`.

Test mode (reads `metrics_test.parquet`, writes `ratios_test.parquet`):

```bash
python compute_ratios.py --test
```

### Step 3 — Serve or export

**Interactive Flask server (recommended):**

```bash
python serve.py
# Open http://localhost:5001
```

**Self-contained static HTML (no server required):**

```bash
python visualize.py          # writes market_manifold.html
python visualize.py --test   # writes market_manifold_test.html
```

---

## Usage

### Flask server (`serve.py`)

The server loads `ratios.parquet` on startup and serves frames on demand. Navigate to `http://localhost:5001`.

- **Year slider / Play button** — scrub or animate through fiscal years
- **Cluster filter bar** — three K-means slices of UMAP space (All / Pre-Profit / Profitable Mid-Cap / Large-Cap); each slice rerenders with its own surface baseline
- **Search box** — type a company name to find and highlight it
- **Click a dot** — opens an info panel with full financial detail and top deviation dimensions

### Static HTML (`visualize.py`)

Produces a single self-contained HTML file with Plotly animation frames embedded as JSON. All years are pre-built at export time so it is larger but requires no server.

---

## What the Visualization Shows

| Visual element | Meaning |
|---|---|
| X / Y axes | UMAP coordinates — proximity = financial similarity |
| Z axis (surface height) | Log P/E ratio (higher = more expensive) |
| Surface color | Log P/B ratio (red-yellow-green scale: red = expensive, green = cheap) |
| Dot color | Company's P/E deviation from its local slice peers (red = expensive vs peers, green = cheap vs peers) |
| Dot size | Magnitude of that deviation — larger dots stand out more from the surface |
| Cluster labels | Auto-assigned from median net margin and revenue: "Pre-Profit / Small-Cap", "Profitable Mid-Cap", "Large-Cap / Diversified" |

The info panel (click any dot or use search) shows revenue, net margin, P/E, P/B, revenue growth, ROE, D/E, outlier percentile, and the top per-dimension z-scores that drive the outlier score.

---

## Known Limitations

- **Revenue coverage ~73%:** Companies that do not file revenue under any of the 8 recognized GAAP tags (`Revenues`, `RevenueFromContractWithCustomer*`, `SalesRevenue*`, `InterestAndDividendIncomeOperating`, `NoninterestIncome`, `PremiumsEarnedNet`) will have `revenue = NaN`. This excludes many investment funds, holding companies, and some non-standard reporters.

- **UMAP requires 4 of 8 features:** Rows with fewer than 4 non-NaN values in the UMAP feature set are excluded from the embedding and will not appear as dots on the surface. Companies with thin EDGAR data vanish from the visualization even if they appear in `metrics.parquet`.

- **Absolute paths are hardcoded:** `extract_metrics.py` and `compute_ratios.py` reference `/Users/rezaramji/Documents/CCC/EDGARmaxxing/` directly. Update the `FOLDER`/`INPUT`/`OUTPUT` constants if you move the project.

- **UMAP is fit once on all years combined:** This gives temporal stability (the same company moves smoothly through the landscape over time) but means the embedding cannot be extended incrementally — refit from scratch if you add new data.

---

## Agent Guide

### Architecture and data flow

```
companyfacts/*.json
        │
        ▼ extract_metrics.py
metrics.parquet   (78k rows — raw annual 10-K values per company-year)
        │
        ▼ compute_ratios.py
ratios.parquet    (78k rows — same rows + ratios + umap_x/y + manifold_distance)
        │
        ├──▶ serve.py        Flask API + inline HTML, builds surface frames on demand
        └──▶ visualize.py    One-shot static HTML with all frames pre-embedded
```

### Key columns

**`metrics.parquet`** — one row per `(cik, fy)`:

| Column | Type | Description |
|---|---|---|
| `cik` | int64 | SEC CIK number |
| `entity_name` | str | Company name from EDGAR |
| `fy` | int64 | Fiscal year |
| `revenue` | float64 | Annual revenue (see GAAP field list) |
| `net_income` | float64 | NetIncomeLoss |
| `assets` | float64 | Total assets |
| `equity` | float64 | Stockholders equity |
| `op_cash` | float64 | Operating cash flow |
| `op_income` | float64 | Operating income |
| `long_term_debt` | float64 | Long-term debt (0-filled when balance sheet exists but tag absent) |
| `cash` | float64 | Cash and equivalents |
| `gross_profit` | float64 | Gross profit |
| `public_float` | float64 | EntityPublicFloat (DEI namespace) |
| `shares_outstanding` | float64 | EntityCommonStockSharesOutstanding |

**`ratios.parquet`** — all columns from metrics.parquet, plus:

| Column | Description |
|---|---|
| `pe_ratio` | `public_float / net_income` (NaN if net_income <= 0) |
| `pb_ratio` | `public_float / equity` (NaN if equity <= 0) |
| `net_margin` | `net_income / revenue` |
| `op_margin` | `op_income / revenue` |
| `revenue_growth` | YoY `(rev - prev_rev) / abs(prev_rev)` |
| `debt_to_equity` | `long_term_debt / equity` |
| `roe` | `net_income / equity` |
| `op_cash_yield` | `op_cash / public_float` |
| `log_revenue` | `log10(revenue)` |
| `log_public_float` | `log10(public_float)` |
| `umap_x`, `umap_y` | 2D UMAP embedding coordinates (NaN for rows excluded from embedding) |
| `manifold_distance` | Euclidean norm of per-feature z-scores vs 15 nearest UMAP neighbours |
| `manifold_distance_pct` | Not computed in `compute_ratios.py` — referenced in `serve.py`/`visualize.py` as `r.manifold_distance_pct`; column will be NaN/absent unless added |
| `dev_log_revenue` … `dev_op_cash_yield` | Per-feature signed z-score vs local 15-neighbour cluster (drives outlier explanation) |
| `umap_cluster` | K-means cluster label 0/1/2 assigned by `serve.py` at startup (not in parquet) |

### Where things live

| File | Responsibility | Key functions |
|---|---|---|
| `extract_metrics.py` | JSON → metrics.parquet | `process_file()`, `extract_annual_gaap()`, `extract_annual_dei()`, `coalesce_fields()` |
| `compute_ratios.py` | metrics → ratios, UMAP, manifold distance | `compute_umap()`, `compute_revenue_growth()` |
| `visualize.py` | ratios → static HTML | `build_frame()` (one Plotly Frame per year), `build_html()` |
| `serve.py` | ratios → Flask API + interactive HTML | `build_frame()` (returns dict for JSON API), route handlers |

### UMAP features

Eight features are fed to UMAP, all computed in `compute_ratios.py`:

```python
UMAP_FEATURES = [
    "log_revenue",        # company scale
    "log_public_float",   # market size
    "net_margin",         # profitability
    "op_margin",          # operating efficiency
    "revenue_growth",     # growth momentum
    "debt_to_equity",     # leverage
    "roe",                # return on equity
    "op_cash_yield",      # cash generation relative to market cap
]
```

**Missing value handling:** A row qualifies for UMAP if at least 4 of 8 features are non-NaN. Remaining NaNs are filled with column medians computed on qualifying rows only. Features with clip bounds defined in `CLIP_BOUNDS` are clipped before scaling. All features are then scaled with `RobustScaler`.

UMAP is fit once on all qualifying rows across all years (`n_neighbors=30`, `min_dist=0.1`, `random_state=42`, Euclidean metric). Manifold distance is computed with a 15-nearest-neighbour KD-tree in UMAP space.

### Flask server architecture (`serve.py`)

**Startup:**
1. Loads `ratios.parquet` and filters to years with >= 50 companies with valid UMAP coordinates.
2. Runs K-means (k=3) on all UMAP points to define cluster slices; re-orders cluster IDs left-to-right by median `umap_x`.
3. Auto-labels each cluster from median net margin and revenue.

**API routes:**

| Route | Description |
|---|---|
| `GET /` | Renders inline HTML template; passes `YEARS`, `first_year`, `cluster_info` as Jinja variables |
| `GET /api/years` | Returns list of qualifying year integers |
| `GET /api/clusters` | Returns `CLUSTER_INFO` dict (n, pe, pb, nm, rev, label per cluster) |
| `GET /api/frame/<year>?cluster=<0\|1\|2>` | Builds and caches one frame: 55x55 Gaussian-smoothed surface (median log P/E per cell), surface color (median log P/B), and per-company dot data. Results cached in `_frame_cache` (dict keyed by `(year, cluster)`); cache is cleared on cluster switch from the client |
| `GET /api/search?q=<str>` | Fuzzy company name search, returns up to 25 `{cik, name}` results (requires at least 2 characters) |
| `GET /api/company/<cik>` | Returns all years of data for a CIK including deviation columns |

**Frame building (`build_frame`):**
- Z axis = `log10(pe_ratio)`, neutral (0) for loss-making companies
- Surface color = `log10(pb_ratio)` per cell (Gaussian σ=1.8)
- Dot color = company's log P/E deviation from the smooth surface at its (x,y) position, normalized to [-1, 1]
- Dot size = 3px base + deviation magnitude, capped at 12px

### Known issues and where to look

| Issue | Where to look |
|---|---|
| Revenue missing for a company | `GAAP_FIELDS["revenue"]` list in `extract_metrics.py` — add the GAAP tag the company uses |
| Company has metrics but no dot | `compute_umap()` threshold: needs >= 4 non-NaN UMAP features |
| `manifold_distance_pct` is always NaN | The column is referenced in `serve.py` and `visualize.py` but is never computed — it would need to be added to `compute_ratios.py` as a percentile rank of `manifold_distance` |
| Click on dot shows wrong company | `serve.py` click handler (`onClik`) reads `pt.customdata.cik` from the dot's vertex data, then fetches `/api/company/<cik>` — verify `d.vertices` is correctly indexed to `d.dots` in `build_frame()` |
| UMAP embedding changes on rerun | `random_state=42` is set but UMAP is non-deterministic with some backends; if coordinates shift, company positions won't match cached frames from a previous run — delete parquet files and rerun the full pipeline |
| Absolute paths break on other machines | `FOLDER` / `OUTPUT` / `INPUT` constants at the top of `extract_metrics.py` and `compute_ratios.py` are hardcoded; `serve.py` uses `Path(__file__).parent` and is portable |
