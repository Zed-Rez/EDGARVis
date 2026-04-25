"""
extract_metrics.py
------------------
Extracts annual 10-K financial metrics from EDGAR companyfacts JSON files
and writes them to metrics.parquet.

Usage:
    python extract_metrics.py          # full run (all ~17k files)
    python extract_metrics.py --test   # test run (first 20 files only)
"""

import json
import sys
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Flow metrics span a period (need full-year duration + period-year alignment).
# Stock metrics are point-in-time (balance sheet snapshots).
FLOW_METRIC_NAMES = {"net_income", "revenue", "op_cash", "op_income", "gross_profit"}

FOLDER = Path("/Users/rezaramji/Documents/CCC/EDGARmaxxing/companyfacts")
OUTPUT = Path("/Users/rezaramji/Documents/CCC/EDGARmaxxing/metrics.parquet")

# us-gaap field priority lists (first non-empty result wins)
GAAP_FIELDS = {
    "net_income":     [
        "NetIncomeLoss",
        # Consolidated net income including non-controlling interest (identical to
        # NetIncomeLoss for companies without subsidiaries)
        "ProfitLoss",
        # Net income after preferred dividends — close proxy when the above are absent
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "revenue":        [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        # Sector-specific top-line revenue
        "OilAndGasRevenue",
        "OilAndGasSalesRevenue",
        "RealEstateRevenueNet",
        "OtherSalesRevenueNet",
        "RegulatedAndUnregulatedOperatingRevenue",   # utilities
        "ElectricUtilityRevenue",                    # electric utilities
        "GasGatheringTransportationMarketingAndProcessingRevenue",  # gas utilities
        "HealthCareOrganizationRevenue",             # hospitals / health systems
        "OperatingLeasesIncomeStatementLeaseRevenue", # REITs / lessors
        # Banks: net interest income + fee income
        "InterestAndDividendIncomeOperating",
        "NoninterestIncome",
        "InterestAndFeeIncomeLoansAndLeases",
        # Investment banks: net revenues (interest income net of funding cost)
        "RevenuesNetOfInterestExpense",
        # Insurers: premiums + net investment income
        "PremiumsEarnedNet",
        "NetInvestmentIncome",
        # Investment companies (BDCs, closed-end funds): net investment income
        # is their functional equivalent of revenue. Added last so operating-company
        # revenue tags always win for years where both exist.
        "InvestmentIncomeNet",
        "InvestmentIncomeInterest",
    ],
    "assets":         ["Assets"],
    "equity":         [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
        # Total equity including minority interest — identical for most companies
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        # Limited partnerships / MLPs
        "PartnersCapital",
        # LLCs (pipelines, utilities, subsidiaries) — exact equivalent of stockholders equity
        "MembersEquity",
        "MembersCapital",
    ],
    "op_cash":        [
        "NetCashProvidedByUsedInOperatingActivities",
        # Continuing-ops variant filed when discontinued operations are split out
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "op_income":      [
        "OperatingIncomeLoss",
        # Pre-tax proxies: newer and older GAAP wording for the same concept
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermNotesPayable",
    ],
    "cash":           [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        # Post-2016 ASC 230 balance sheet line includes restricted cash
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        # Simple "Cash" tag used by some older or smaller filers
        "Cash",
        # Bank-specific: cash + fed funds sold
        "CashCashEquivalentsAndFederalFundsSold",
    ],
    "gross_profit":   ["GrossProfit"],
}

# dei field priority lists
DEI_FIELDS = {
    "public_float":       ["EntityPublicFloat"],
    "shares_outstanding": ["EntityCommonStockSharesOutstanding"],
}

ALL_METRIC_COLS = (
    list(GAAP_FIELDS.keys()) + list(DEI_FIELDS.keys())
)


def extract_annual_gaap(entries: list, is_flow: bool = False) -> dict:
    """
    Filter us-gaap entries to annual 10-K (form=='10-K', fp=='FY', val not None).
    Returns {fy: val} keeping the most recently filed entry per fiscal year.

    Two extra guards that prevent comparative/restated data from the wrong year
    contaminating a fiscal year's row:
      1. |end_year - fy| <= 1  — rejects comparative periods re-filed under a
         later fiscal year (e.g. Apple filing FY2017 revenue inside its FY2019 10-K
         produces fy=2019, end=2017-09-30; the gap of 2 drops it).
      2. is_flow + 300 <= duration <= 550 days — rejects both quarterly sub-periods
         (< 300 days) AND inception-to-date cumulative statements that development-
         stage companies file alongside their annual figures (duration >> 366 days).
         550-day ceiling allows for unusually long fiscal year transitions while
         excluding multi-year cumulative periods (e.g. 16,400-day since-1967 entry).
    """
    best = {}  # fy -> (filed_str, val)
    for entry in entries:
        if "val" not in entry or entry["val"] is None:
            continue
        if entry.get("form") != "10-K":
            continue
        if entry.get("fp") != "FY":
            continue

        end_str = entry.get("end", "")
        if not end_str or len(end_str) < 4:
            continue

        # Fiscal year from EDGAR's fy field, fall back to end-date year
        fy = entry.get("fy")
        if fy is None:
            try:
                fy = int(end_str[:4])
            except ValueError:
                continue
        try:
            fy = int(fy)
        except (ValueError, TypeError):
            continue

        # Guard 1: end year must be within 1 calendar year of fy
        try:
            end_year = int(end_str[:4])
        except ValueError:
            continue
        if abs(end_year - fy) > 1:
            continue

        # Guard 2: flow metrics must cover a full year (300–550 days).
        # Lower bound rejects quarterly sub-periods; upper bound rejects
        # inception-to-date cumulative statements filed by development-stage
        # companies (e.g. start=1967-05-26, end=2012-04-30 = 16,400 days).
        if is_flow:
            start_str = entry.get("start", "")
            if start_str and len(start_str) >= 10 and len(end_str) >= 10:
                try:
                    s = date.fromisoformat(start_str[:10])
                    e = date.fromisoformat(end_str[:10])
                    dur = (e - s).days
                    if dur < 300 or dur > 550:
                        continue
                except ValueError:
                    pass

        filed = entry.get("filed") or end_str
        val   = entry["val"]
        exact = (end_year == fy)  # True when period year matches fiscal year exactly

        if fy not in best:
            best[fy] = (filed, val, exact)
        else:
            prev_filed, _, prev_exact = best[fy]
            # Prefer: (1) more recently filed; (2) exact year match if same filing date
            if filed > prev_filed or (filed == prev_filed and exact and not prev_exact):
                best[fy] = (filed, val, exact)

    return {fy: v for fy, (_, v, _exact) in best.items()}


def extract_annual_dei(entries: list, is_flow: bool = False) -> dict:
    """
    DEI entries (EntityPublicFloat, shares outstanding) are point-in-time and
    may lack 'form'/'fp'. Accept any entry with a value and a determinable fy;
    apply the same |end_year - fy| <= 1 guard to drop comparative mis-tags.
    """
    best = {}
    for entry in entries:
        if "val" not in entry or entry["val"] is None:
            continue
        form = entry.get("form")
        if form is not None and form != "10-K":
            continue

        end_str = entry.get("end", "")
        if not end_str or len(end_str) < 4:
            continue

        fy = entry.get("fy")
        if fy is None:
            try:
                fy = int(end_str[:4])
            except ValueError:
                continue
        try:
            fy = int(fy)
        except (ValueError, TypeError):
            continue

        try:
            end_year = int(end_str[:4])
        except ValueError:
            continue
        if abs(end_year - fy) > 1:
            continue

        filed = entry.get("filed") or end_str
        val   = entry["val"]
        exact = (end_year == fy)

        if fy not in best:
            best[fy] = (filed, val, exact)
        else:
            prev_filed, _, prev_exact = best[fy]
            if filed > prev_filed or (filed == prev_filed and exact and not prev_exact):
                best[fy] = (filed, val, exact)

    return {fy: v for fy, (_, v, _exact) in best.items()}


def coalesce_fields(facts_ns: dict, field_map: dict, extractor_fn, flow_set: set = None) -> dict:
    """
    Merge all candidate concepts in priority order.
    Earlier candidates win per fiscal year, but later candidates fill years
    where earlier ones have no data (e.g. Apple uses 'Revenues' pre-2019 and
    'RevenueFromContractWithCustomer...' post-2019).
    """
    result = {}
    for metric, candidates in field_map.items():
        is_flow = metric in (flow_set or set())
        merged: dict = {}
        for concept in candidates:
            if concept not in facts_ns:
                continue
            units = facts_ns[concept].get("units", {})
            all_entries = []
            for unit_entries in units.values():
                if isinstance(unit_entries, list):
                    all_entries.extend(unit_entries)
            if not all_entries:
                continue
            for fy, val in extractor_fn(all_entries, is_flow=is_flow).items():
                if fy not in merged:   # earlier candidate wins per year
                    merged[fy] = val
        result[metric] = merged
    return result


def process_file(path: Path) -> list:
    """
    Load a single companyfacts JSON file and return a list of row-dicts,
    one per fiscal year that passes the coverage threshold.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []

    # Top-level fields
    cik_raw = data.get("cik")
    if cik_raw is None:
        return []
    try:
        cik = int(cik_raw)
    except (ValueError, TypeError):
        return []

    entity_name = data.get("entityName", "")

    facts = data.get("facts", {})
    gaap = facts.get("us-gaap", {})
    dei  = facts.get("dei", {})

    # ── Revenue: two-phase extraction ──────────────────────────────────────────
    # InvestmentIncomeNet / InvestmentIncomeInterest are valid revenue proxies
    # for pure investment companies (BDCs, closed-end funds, etc.) that have
    # no other revenue concept.  But for operating companies that simply stopped
    # tagging their revenue in XBRL after a restructuring (e.g. Apache Corp
    # dropped OilAndGasRevenue post-2017 but still earns billions), those tags
    # would silently replace real revenue with a tiny interest-income figure.
    # Fix: only allow investment-income tags when the company has ZERO primary
    # revenue data in any year.
    _INVEST_REV = {"InvestmentIncomeNet", "InvestmentIncomeInterest"}
    _primary_rev_tags = [t for t in GAAP_FIELDS["revenue"] if t not in _INVEST_REV]
    _gaap_fields_primary = {**GAAP_FIELDS, "revenue": _primary_rev_tags}
    gaap_data = coalesce_fields(gaap, _gaap_fields_primary, extract_annual_gaap, flow_set=FLOW_METRIC_NAMES)

    if not gaap_data.get("revenue"):
        # No primary revenue at all — allow investment income as a last resort
        invest_rev = coalesce_fields(
            gaap,
            {"revenue": list(_INVEST_REV)},
            extract_annual_gaap,
            flow_set=FLOW_METRIC_NAMES,
        )
        gaap_data["revenue"] = invest_rev.get("revenue", {})
    # ───────────────────────────────────────────────────────────────────────────

    dei_data  = coalesce_fields(dei,  DEI_FIELDS,  extract_annual_dei)

    # Fallback cascade for shares_outstanding:
    #   1. EntityCommonStockSharesOutstanding (DEI) — already in DEI_FIELDS
    #   2. CommonStockSharesOutstanding (GAAP) — point-in-time, preferred over weighted avg
    #   3. LimitedPartnersCapitalAccountUnitsOutstanding — LP/MLP unit count
    #   4. WeightedAverageNumberOfSharesOutstandingBasic — approximation (period average)
    #   5. WeightedAverageNumberOfShareOutstandingBasicAndDiluted — single-class companies
    _sh_fallbacks = [
        "CommonStockSharesOutstanding",
        "LimitedPartnersCapitalAccountUnitsOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ]
    existing = dei_data.get("shares_outstanding", {})
    for fb_tag in _sh_fallbacks:
        if fb_tag not in gaap:
            continue
        gaap_sh = coalesce_fields(
            gaap,
            {"shares_outstanding": [fb_tag]},
            extract_annual_dei,
        )
        for fy, val in gaap_sh.get("shares_outstanding", {}).items():
            if fy not in existing:
                existing[fy] = val
    dei_data["shares_outstanding"] = existing

    # Merge: collect all fiscal years mentioned across all metrics
    all_fys = set()
    for fy_map in gaap_data.values():
        all_fys.update(fy_map.keys())
    for fy_map in dei_data.values():
        all_fys.update(fy_map.keys())

    rows = []
    coverage_keys = {"net_income", "revenue", "assets", "equity", "public_float"}

    for fy in sorted(all_fys):
        row = {
            "cik":         cik,
            "entity_name": entity_name,
            "fy":          fy,
        }
        for metric in GAAP_FIELDS:
            val = gaap_data.get(metric, {}).get(fy, np.nan)
            row[metric] = float(val) if val is not None and not _is_nan(val) else np.nan
        for metric in DEI_FIELDS:
            val = dei_data.get(metric, {}).get(fy, np.nan)
            row[metric] = float(val) if val is not None and not _is_nan(val) else np.nan

        # ── Field-level sanity constraints applied at extraction time ───────
        # Revenue is definitionally non-negative; negative values arise from
        # investment-loss tags or EDGAR filing errors.
        if not _is_nan(row.get("revenue", np.nan)) and row["revenue"] < 0:
            row["revenue"] = np.nan
        # Cash cannot be negative.
        if not _is_nan(row.get("cash", np.nan)) and row["cash"] < 0:
            row["cash"] = np.nan
        # shares_outstanding = 0 is meaningless (LLCs, pre-IPO shells); NaN.
        if not _is_nan(row.get("shares_outstanding", np.nan)) and row["shares_outstanding"] == 0:
            row["shares_outstanding"] = np.nan

        # Coverage filter: at least 3 of the 5 key fields must be non-NaN
        non_nan_count = sum(
            1 for k in coverage_keys
            if not (isinstance(row.get(k), float) and np.isnan(row[k]))
        )
        if non_nan_count >= 3:
            rows.append(row)

    return rows


def _is_nan(v):
    """Safe NaN check that works for non-float types."""
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False


def main():
    test_mode = "--test" in sys.argv
    files = sorted(FOLDER.glob("*.json"))

    if test_mode:
        files = files[:20]
        print(f"TEST MODE: processing {len(files)} files")
    else:
        print(f"Processing {len(files)} files...")

    rows = []
    for i, f in enumerate(files):
        if i % 1000 == 0 and not test_mode:
            print(f"  {i}/{len(files)}")
        rows.extend(process_file(f))

    if not rows:
        print("WARNING: no rows extracted!")
        return

    df = pd.DataFrame(rows)

    # Enforce dtypes
    df["cik"] = df["cik"].astype("int64")
    df["fy"]  = df["fy"].astype("int64")
    df["entity_name"] = df["entity_name"].astype(str)

    float_cols = [c for c in ALL_METRIC_COLS]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    # Ensure all expected columns exist (add NaN columns if any are missing)
    expected_cols = [
        "cik", "entity_name", "fy",
        "net_income", "revenue", "assets", "equity", "op_cash",
        "op_income", "long_term_debt", "cash", "gross_profit",
        "public_float", "shares_outstanding",
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_cols]

    output_path = OUTPUT if not test_mode else OUTPUT.with_name("metrics_test.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} rows, {df['cik'].nunique()} companies -> {output_path}")

    print("\n--- dtypes ---")
    print(df.dtypes)

    print("\n--- First 10 rows ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.head(10).to_string())

    print("\n--- Column coverage (non-NaN %) ---")
    for col in expected_cols:
        pct = df[col].notna().mean() * 100
        print(f"  {col:30s}: {pct:6.1f}%")

    print("\n--- Numeric summary ---")
    print(df[float_cols].describe().to_string())


if __name__ == "__main__":
    main()
