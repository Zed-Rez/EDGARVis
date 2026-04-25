#!/usr/bin/env python3
"""
Build static EdgarInsights site from serve.py's INSIGHTS_HTML template.

The /insights page already inlines all its data via Jinja
({{ insights_agg | tojson }}, {{ insights_meta | tojson }}, {{ years | tojson }}).
After rendering once, the page is self-contained — no fetches needed.

Output:
    <repo>/build_output/EdgarInsights/index.html
"""
import os
import re
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent)
sys.path.insert(0, SRC)
os.chdir(SRC)  # so serve.py can find ratios.parquet

import serve  # triggers precompute (INSIGHTS_AGG, INSIGHTS_META, _INS_YEARS)

OUT_DIR = Path(__file__).resolve().parent / "build_output" / "EdgarInsights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Render the template using Flask's Jinja env, with the same context
# the /insights route passes (serve.py lines 379-388).
with serve.app.app_context():
    from flask import render_template_string
    html = render_template_string(
        serve.INSIGHTS_HTML,
        years=serve._INS_YEARS,
        insights_agg=serve.INSIGHTS_AGG,
        insights_meta=serve.INSIGHTS_META,
        manifold_url="../EdgarVis/",
        insights_url=".",
    )

# Sanity checks: no Jinja artifacts, no absolute API fetches.
jinja_artifacts = re.findall(r"\{\{[^}]+\}\}|\{%[^%]+%\}", html)
if jinja_artifacts:
    print("ERROR: leftover Jinja artifacts:", jinja_artifacts[:5])
    sys.exit(1)

fetches = re.findall(r"fetch\(['\"]([^'\"]+)['\"]", html)
print(f"fetches found: {fetches!r}")

api_refs = re.findall(r"['\"](/api/[^'\"]+)['\"]", html)
print(f"absolute /api/ refs: {api_refs!r}")

if fetches or api_refs:
    # Insights template should have none. If something appears, fail loudly
    # rather than silently shipping a broken page.
    print("ERROR: unexpected network calls in rendered insights page")
    sys.exit(1)

# Write final HTML
out_path = OUT_DIR / "index.html"
out_path.write_text(html, encoding="utf-8")

size = out_path.stat().st_size
print(f"Wrote {out_path} ({size:,} bytes)")
print(f"YEARS: {serve._INS_YEARS}")
print(f"INSIGHTS_AGG keys: {list(serve.INSIGHTS_AGG.keys())}")
print(f"INSIGHTS_META keys: {list(serve.INSIGHTS_META.keys())}")
