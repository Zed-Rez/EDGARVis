import pandas as pd
import numpy as np

df = pd.read_parquet('/Users/rezaramji/Documents/CCC/EDGARmaxxing/ratios.parquet')
df['fy'] = pd.to_numeric(df['fy'], errors='coerce')
df = df[df['fy'].notna()].copy()
df['fy'] = df['fy'].astype(int)
year_counts = df.dropna(subset=['umap_x','umap_y']).groupby('fy').size()
YEARS = sorted(int(y) for y in year_counts[year_counts >= 50].index)
df = df[df['fy'].isin(YEARS)]

# --- IDEA 5: Revenue concentration ---
rev_conc = {}
for yr in [y for y in YEARS if y <= 2025]:
    rev = df[df['fy']==yr]['revenue'].dropna().sort_values(ascending=False)
    total = rev.sum()
    if total > 0:
        n = len(rev)
        rev_conc[yr] = {
            'top1pct': rev.iloc[:max(1,n//100)].sum()/total,
            'top5pct': rev.iloc[:max(1,n//20)].sum()/total,
            'top10pct': rev.iloc[:max(1,n//10)].sum()/total,
            'n': n
        }
print("\n=== Revenue Concentration (share held by top companies) ===")
print(pd.DataFrame(rev_conc).T.round(3).to_string())

# --- IDEA 6: Stranded companies ---
yr2023 = df[df['fy']==2023].dropna(subset=['manifold_distance_pct','revenue_growth'])
yr2023 = yr2023[yr2023['revenue_growth'].between(-0.5,2)]
yr2023 = yr2023.copy()
yr2023['mf_bin'] = pd.qcut(yr2023['manifold_distance_pct'], q=5, labels=['Low','Low-Mid','Mid','High-Mid','High'])
yr2023['rg_bin'] = pd.cut(yr2023['revenue_growth'], bins=[-0.5,-0.1,0,0.1,0.25,2], labels=['<-10%','-10-0%','0-10%','10-25%','>25%'])
heat = yr2023.groupby(['mf_bin','rg_bin']).size().unstack(fill_value=0)
print("\n=== Outlier-ness vs Revenue Growth Heatmap (2023) ===")
print(heat.to_string())

# --- BONUS: Deep dive on IDEA 2 counterintuitive finding ---
# Tiny companies (Q1) have DRAMATICALLY worsening margins over time
# Let's check if this is survivorship vs zombie company effect
print("\n=== BONUS: Tiny company net margin deterioration details ===")
q1_data = df[df['size_bucket'] == 'Q1(tiny)'].dropna(subset=['net_margin'])
print("Q1(tiny) net margin percentiles by year:")
q1_pct = q1_data.groupby('fy')['net_margin'].describe(percentiles=[0.1,0.25,0.5,0.75,0.9])
print(q1_pct[['count','10%','25%','50%','75%','90%']].round(3).to_string())

# --- BONUS 2: manifold_distance_pct vs forward P/E - do outlier companies command premium or discount? ---
print("\n=== BONUS 2: manifold_distance_pct distribution details ===")
print(df.groupby('fy')['manifold_distance_pct'].describe(percentiles=[0.25,0.5,0.75,0.9,0.95]).round(3).to_string())

# --- BONUS 3: P/B vs ROE across ALL years (not just 2023) - is the hockey stick shape consistent? ---
print("\n=== BONUS 3: P/B by ROE decile across years ===")
for yr in [2015, 2018, 2020, 2021, 2022, 2023]:
    yr_df = df[df['fy']==yr].dropna(subset=['pb_ratio','roe'])
    yr_df = yr_df[(yr_df['pb_ratio']>0)&(yr_df['pb_ratio']<30)&(yr_df['roe'].between(-0.5,2))].copy()
    if len(yr_df) > 50:
        yr_df['roe_decile'] = pd.qcut(yr_df['roe'], q=10, labels=range(10))
        pb_by_roe = yr_df.groupby('roe_decile')['pb_ratio'].median()
        print(f"\n{yr} (n={len(yr_df)}):")
        print(pb_by_roe.round(2).to_string())

# --- BONUS 4: The "quality trap" - high ROE companies in decile 0 (negative ROE) have HIGHER P/B than deciles 2-5 ---
# This suggests the MARKET PAYS MORE for loss-making companies than marginally profitable ones?
print("\n=== BONUS 4: The quality trap - ROE=0 companies vs slightly profitable ===")
yr2023_full = df[df['fy']==2023].dropna(subset=['pb_ratio','roe'])
yr2023_full = yr2023_full[(yr2023_full['pb_ratio']>0)&(yr2023_full['pb_ratio']<30)].copy()
# Break into: deeply negative (<-20%), mildly negative (-20% to 0%), barely profitable (0-5%), healthy (5-15%), high ROE (>15%)
yr2023_full['roe_cat'] = pd.cut(yr2023_full['roe'],
    bins=[-99, -0.2, 0, 0.05, 0.15, 99],
    labels=['Deep loss\n(<-20% ROE)', 'Mild loss\n(-20 to 0%)', 'Barely prof.\n(0-5%)', 'Healthy\n(5-15%)', 'High ROE\n(>15%)'])
summary = yr2023_full.groupby('roe_cat')['pb_ratio'].agg(['median','mean','count'])
print(summary.round(3).to_string())
