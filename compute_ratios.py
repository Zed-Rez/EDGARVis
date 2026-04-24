# compute_ratios.py
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.spatial import cKDTree
import umap
from pathlib import Path

_test = "--test" in sys.argv
INPUT  = Path("/Users/rezaramji/Documents/CCC/EDGARmaxxing/" + ("metrics_test.parquet" if _test else "metrics.parquet"))
OUTPUT = Path("/Users/rezaramji/Documents/CCC/EDGARmaxxing/" + ("ratios_test.parquet"   if _test else "ratios.parquet"))

# Feature columns used for UMAP
UMAP_FEATURES = [
    "log_revenue",
    "log_public_float",
    "net_margin",
    "op_margin",
    "revenue_growth",
    "debt_to_equity",
    "roe",
    "op_cash_yield",
]

# Clip bounds applied before scaling (columns that have bounds defined)
CLIP_BOUNDS = {
    "pe_ratio":       (-50,   200),
    "pb_ratio":       (-5,     50),
    "net_margin":     (-2,      1),
    "op_margin":      (-2,      1),
    "revenue_growth": (-1,      5),
    "debt_to_equity": (-10,    20),
    "roe":            (-2,      3),
}


def compute_revenue_growth(df: pd.DataFrame) -> pd.Series:
    """YoY revenue growth per company, sorted by fy."""
    df = df.sort_values(["cik", "fy"])
    prev = df.groupby("cik")["revenue"].shift(1)
    # NaN where prev is NaN (first year) or prev is 0
    growth = (df["revenue"] - prev) / prev.abs()
    growth = growth.where(prev.notna() & (prev != 0), other=np.nan)
    return growth


def compute_umap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a single UMAP embedding across ALL qualifying rows (all years combined)
    so that the embedding is stable across time for animation purposes.

    Qualifying rows: at least 4 non-NaN values among UMAP_FEATURES.
    NaN fill: column medians (not zero).
    Scaling: RobustScaler.
    """
    # Initialise output columns
    df = df.copy()
    df["umap_x"] = np.nan
    df["umap_y"] = np.nan

    feature_df = df[UMAP_FEATURES].copy()

    # Clip extreme values (only for features that have clip bounds defined)
    for col in UMAP_FEATURES:
        if col in CLIP_BOUNDS:
            lo, hi = CLIP_BOUNDS[col]
            feature_df[col] = feature_df[col].clip(lo, hi)

    # Identify qualifying rows: at least 4 non-NaN values across UMAP_FEATURES
    non_nan_counts = feature_df.notna().sum(axis=1)
    qualifying_mask = non_nan_counts >= 4

    n_qualifying = qualifying_mask.sum()
    print(f"  UMAP: {n_qualifying} qualifying rows out of {len(df)} total")

    if n_qualifying < 2:
        print("  UMAP: not enough qualifying rows — skipping embedding")
        return df

    feat_subset = feature_df.loc[qualifying_mask].copy()

    # Fill remaining NaNs with column medians computed on qualifying rows only
    col_medians = feat_subset.median()
    feat_subset = feat_subset.fillna(col_medians)

    # Scale with RobustScaler
    scaler = RobustScaler()
    scaled = scaler.fit_transform(feat_subset.values)

    # Fit UMAP on all qualifying rows
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        metric="euclidean",
    )
    embedding = reducer.fit_transform(scaled)

    # Write UMAP coords back using the original DataFrame index
    qualifying_indices = df.index[qualifying_mask]
    df.loc[qualifying_indices, "umap_x"] = embedding[:, 0]
    df.loc[qualifying_indices, "umap_y"] = embedding[:, 1]

    # ------------------------------------------------------------------
    # Manifold distance: per-company deviation from local neighbourhood
    # For each qualifying company, find k nearest neighbours in UMAP
    # space and compute feature-wise z-scores vs. that neighbourhood.
    # manifold_distance = Euclidean norm of those z-scores (overall
    # "how far from the local crowd").  Individual axis columns store
    # the signed z-score so callers know *which* dimension drives the
    # deviation.
    # ------------------------------------------------------------------
    K_NEIGHBORS = 15
    EPS = 1e-9
    dev_feature_names = [
        "log_revenue", "log_public_float", "net_margin", "op_margin",
        "revenue_growth", "debt_to_equity", "roe", "op_cash_yield",
    ]
    dev_col_names = [f"dev_{f}" for f in dev_feature_names]
    for col in dev_col_names + ["manifold_distance"]:
        df[col] = np.nan

    if len(embedding) > K_NEIGHBORS + 1:
        tree = cKDTree(embedding)
        # k+1 because the point itself is its own nearest neighbour
        dists, idxs = tree.query(embedding, k=K_NEIGHBORS + 1)

        # Re-build scaled feature matrix (already computed above; redo for clarity)
        feat_filled = feat_subset.copy()
        for col in UMAP_FEATURES:
            if col in CLIP_BOUNDS:
                lo, hi = CLIP_BOUNDS[col]
                feat_filled[col] = feat_filled[col].clip(lo, hi)
        feat_filled = feat_filled.fillna(col_medians)
        scaler2 = RobustScaler()
        scaled2 = scaler2.fit_transform(feat_filled.values)

        manifold_dist = np.full(len(embedding), np.nan)
        dev_matrix    = np.full((len(embedding), len(dev_feature_names)), np.nan)

        for i in range(len(embedding)):
            # Exclude self (index 0)
            neighbor_idxs = idxs[i, 1:]
            nbr_features  = scaled2[neighbor_idxs]           # (K, n_features)
            nbr_mean      = nbr_features.mean(axis=0)
            nbr_std       = nbr_features.std(axis=0)
            z_scores      = (scaled2[i] - nbr_mean) / (nbr_std + EPS)
            manifold_dist[i] = float(np.sqrt((z_scores ** 2).sum()))
            dev_matrix[i]    = z_scores

        df.loc[qualifying_indices, "manifold_distance"] = manifold_dist
        for j, col in enumerate(dev_col_names):
            df.loc[qualifying_indices, col] = dev_matrix[:, j]

    return df


def main():
    df = pd.read_parquet(INPUT)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # ------------------------------------------------------------------ #
    # 0. Pre-ratio fixups                                                  #
    # ------------------------------------------------------------------ #

    # Companies that have a balance sheet (assets + equity both present) but
    # no LongTermDebt tag almost certainly have zero long-term debt — they
    # simply omit the line item when it doesn't apply.
    has_balance_sheet = df["assets"].notna() & df["equity"].notna()
    df["long_term_debt"] = df["long_term_debt"].where(
        df["long_term_debt"].notna() | ~has_balance_sheet, 0.0
    )

    # ------------------------------------------------------------------ #
    # 1. Basic ratio columns                                               #
    # ------------------------------------------------------------------ #

    # pe_ratio = public_float / net_income  (NaN if net_income <= 0 or float NaN)
    df["pe_ratio"] = np.where(
        df["public_float"].isna() | (df["net_income"] <= 0),
        np.nan,
        df["public_float"] / df["net_income"],
    )

    # pb_ratio = public_float / equity  (NaN if equity <= 0 or float NaN)
    df["pb_ratio"] = np.where(
        df["public_float"].isna() | (df["equity"] <= 0),
        np.nan,
        df["public_float"] / df["equity"],
    )

    # net_margin = net_income / revenue  (NaN if revenue == 0 or either NaN)
    df["net_margin"] = np.where(
        df["revenue"].isna() | df["net_income"].isna() | (df["revenue"] == 0),
        np.nan,
        df["net_income"] / df["revenue"],
    )

    # op_margin = op_income / revenue  (NaN if revenue == 0 or either NaN)
    df["op_margin"] = np.where(
        df["revenue"].isna() | df["op_income"].isna() | (df["revenue"] == 0),
        np.nan,
        df["op_income"] / df["revenue"],
    )

    # revenue_growth — YoY % change per company
    df = df.sort_values(["cik", "fy"]).reset_index(drop=True)
    df["revenue_growth"] = compute_revenue_growth(df).values

    # debt_to_equity = long_term_debt / equity  (NaN if equity == 0)
    df["debt_to_equity"] = np.where(
        df["equity"].isna() | (df["equity"] == 0),
        np.nan,
        df["long_term_debt"] / df["equity"],
    )

    # roe = net_income / equity  (NaN if equity == 0)
    df["roe"] = np.where(
        df["equity"].isna() | (df["equity"] == 0),
        np.nan,
        df["net_income"] / df["equity"],
    )

    # op_cash_yield = op_cash / public_float  (NaN if public_float NaN or 0)
    df["op_cash_yield"] = np.where(
        df["public_float"].isna() | (df["public_float"] == 0),
        np.nan,
        df["op_cash"] / df["public_float"],
    )

    # log_revenue = log10(revenue) where revenue > 0, else NaN
    df["log_revenue"] = np.where(
        df["revenue"].isna() | (df["revenue"] <= 0),
        np.nan,
        np.log10(df["revenue"].clip(lower=1e-300)),  # avoid log10(0) warnings
    )
    # Enforce NaN for non-positive cleanly
    df.loc[df["revenue"].isna() | (df["revenue"] <= 0), "log_revenue"] = np.nan

    # log_public_float = log10(public_float) where public_float > 0, else NaN
    df["log_public_float"] = np.where(
        df["public_float"].isna() | (df["public_float"] <= 0),
        np.nan,
        np.log10(df["public_float"].clip(lower=1e-300)),
    )
    df.loc[df["public_float"].isna() | (df["public_float"] <= 0), "log_public_float"] = np.nan

    # ------------------------------------------------------------------ #
    # 2. UMAP                                                              #
    # ------------------------------------------------------------------ #
    df = compute_umap(df)

    # ------------------------------------------------------------------ #
    # 3. Percentile rank of manifold_distance                              #
    # ------------------------------------------------------------------ #
    # Convert raw distance to a 0–100 percentile so the UI can show
    # "top X% outlier" without exposing the raw Euclidean score.
    df["manifold_distance_pct"] = (
        df["manifold_distance"].rank(pct=True, na_option="keep") * 100
    )

    # ------------------------------------------------------------------ #
    # 4. Save                                                              #
    # ------------------------------------------------------------------ #
    df.to_parquet(OUTPUT, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(df[["pe_ratio", "pb_ratio", "net_margin", "umap_x", "umap_y", "manifold_distance", "manifold_distance_pct"]].describe())


if __name__ == "__main__":
    main()
