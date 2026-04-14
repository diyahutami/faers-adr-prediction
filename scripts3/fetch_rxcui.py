"""
fetch_rxcui.py
==============
Fetches RXCUI (RxNorm Concept Unique Identifier) for each unique drug name in
DRUGS_STANDARDIZED_DRUGBANK.csv using the RxNorm REST API, then adds a RXCUI
column to the file.

RxNorm API used:
  GET https://rxnav.nlm.nih.gov/REST/rxcui.json?name={name}&search=2
  search=2 = "normalized string" search (case-insensitive, handles minor variations)

Why RXCUI instead of RXAUI?
  RXAUI = RxNorm Atom Unique ID — identifies a specific term/source entry (e.g.,
          the string "rifampin" from FDA SPL). Multiple RXAUIs can exist for
          the same drug concept.
  RXCUI = RxNorm Concept Unique ID — one per drug concept, regardless of source
          or string variation. ONSIDES and most external biomedical KGs use RXCUI.

Output:
  data/drug_rxcui_mapping.tsv         — persistent cache (DRUG_NAME, RXCUI)
  data/FAERS_TB/DRUGS_STANDARDIZED_DRUGBANK.csv   — updated in-place with RXCUI column
  data/FAERS_TB_DRUGS/DRUGS_STANDARDIZED_DRUGBANK.csv (if exists) — same

Usage:
  cd /home/diyah/PhD_Research/ADR_EHR_Prediction_FAERS
  python scripts3/fetch_rxcui.py
  python scripts3/fetch_rxcui.py --dry-run      # show coverage without saving
  python scripts3/fetch_rxcui.py --dataset FAERS_ALL  # specific dataset only
"""

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
import json
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All datasets that have a DRUGS_STANDARDIZED_DRUGBANK.csv
DATASET_DIRS = {
    "FAERS_TB"      : os.path.join(PROJECT_ROOT, "data", "FAERS_TB"),
    "FAERS_TB_DRUGS": os.path.join(PROJECT_ROOT, "data", "FAERS_TB_DRUGS"),
    "FAERS_ALL"     : os.path.join(PROJECT_ROOT, "data", "FAERS_ALL"),
}
DRUGS_FILE       = "DRUGS_STANDARDIZED_DRUGBANK.csv"
CACHE_FILE       = os.path.join(PROJECT_ROOT, "data", "drug_rxcui_mapping.tsv")

# ── RxNorm API ─────────────────────────────────────────────────────────────────
RXNORM_API_BASE  = "https://rxnav.nlm.nih.gov/REST"
REQUEST_DELAY    = 0.06   # seconds between calls (~16 req/sec; NLM limit ~20/sec)
MAX_RETRIES      = 3
RETRY_DELAY      = 2.0    # seconds to wait after a failed request


def _rxcui_for_name(drug_name: str) -> str | None:
    """
    Query RxNorm API for the RXCUI of a drug name.
    Returns the first RXCUI string, or None if not found.

    API docs:
      https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.findRxcuiByString.html

    search=2  : normalized string search
                Handles: lowercase, punctuation, common misspellings
    """
    encoded = urllib.parse.quote(drug_name.strip())
    url = f"{RXNORM_API_BASE}/rxcui.json?name={encoded}&search=2"

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            ids = data.get("idGroup", {}).get("rxnormId", [])
            return ids[0] if ids else None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None


def _load_cache() -> dict:
    """Load persisted DRUG_NAME → RXCUI mapping from cache file."""
    if not os.path.exists(CACHE_FILE):
        return {}
    df = pd.read_csv(CACHE_FILE, sep="\t", dtype=str)
    # RXCUI column may contain "NONE" for unmapped drugs
    return dict(zip(df["DRUG_NAME"], df["RXCUI"]))


def _save_cache(mapping: dict) -> None:
    """Persist DRUG_NAME → RXCUI mapping to cache TSV (sorted, reproducible)."""
    rows = sorted(mapping.items(), key=lambda x: x[0].lower())
    df = pd.DataFrame(rows, columns=["DRUG_NAME", "RXCUI"])
    df.to_csv(CACHE_FILE, sep="\t", index=False)


def fetch_rxcui_for_drugs(drug_names: list[str],
                           cached: dict,
                           dry_run: bool = False) -> dict:
    """
    Fetch RXCUI for each drug name not already in cache.

    Args:
        drug_names : list of unique drug names to look up
        cached     : existing DRUG_NAME → RXCUI dict (modified in-place)
        dry_run    : if True, skip API calls and report using cached data only

    Returns updated cache dict.
    """
    to_fetch = [n for n in drug_names if n not in cached]
    print(f"  Cached: {len(cached) - len([n for n in drug_names if n not in cached]):,} / {len(drug_names):,}")
    print(f"  To fetch: {len(to_fetch):,}")

    if dry_run:
        print("  [dry-run] Skipping API calls.")
        return cached

    if not to_fetch:
        return cached

    print(f"  Fetching {len(to_fetch):,} drug names from RxNorm API "
          f"(~{len(to_fetch) * REQUEST_DELAY / 60:.1f} min estimated) …")

    for i, name in enumerate(to_fetch, 1):
        rxcui = _rxcui_for_name(name)
        cached[name] = rxcui if rxcui else "NONE"
        time.sleep(REQUEST_DELAY)

        if i % 100 == 0 or i == len(to_fetch):
            mapped = sum(1 for v in cached.values() if v != "NONE")
            print(f"  [{i:>5}/{len(to_fetch)}] fetched … mapped so far: {mapped:,}", flush=True)
            _save_cache(cached)  # checkpoint every 100 requests

    _save_cache(cached)
    return cached


def add_rxcui_to_csv(csv_path: str, mapping: dict, dry_run: bool = False) -> None:
    """
    Add (or overwrite) a RXCUI column in the given DRUGS_STANDARDIZED CSV.

    Mapping lookup uses the DRUG column (original drug name string).
    "NONE" values are stored as NaN so they read as null.
    """
    if not os.path.exists(csv_path):
        print(f"  Skipping (file not found): {csv_path}")
        return

    df = pd.read_csv(csv_path, dtype=str)
    n_drugs = df["DRUG"].notna().sum()

    # Map drug names to RXCUI
    rxcui_series = df["DRUG"].map(lambda x: mapping.get(str(x), None) if pd.notna(x) else None)
    rxcui_series = rxcui_series.replace("NONE", None)  # store "not found" as NaN

    n_mapped = rxcui_series.notna().sum()
    coverage = 100 * n_mapped / n_drugs if n_drugs > 0 else 0.0

    print(f"  {os.path.relpath(csv_path, PROJECT_ROOT)}: "
          f"{n_mapped:,}/{n_drugs:,} rows have RXCUI ({coverage:.1f}%)")

    if dry_run:
        return

    # Insert RXCUI column after RXAUI (or before DRUG if RXAUI absent)
    if "RXCUI" in df.columns:
        df["RXCUI"] = rxcui_series
    elif "RXAUI" in df.columns:
        rxaui_pos = df.columns.get_loc("RXAUI")
        df.insert(rxaui_pos + 1, "RXCUI", rxcui_series)
    else:
        df["RXCUI"] = rxcui_series

    df.to_csv(csv_path, index=False)
    print(f"    ✓ Saved with RXCUI column")


def print_coverage_report(drug_names: list[str], mapping: dict) -> None:
    """Print detailed coverage report."""
    total        = len(drug_names)
    mapped       = [n for n in drug_names if mapping.get(n, "NONE") != "NONE" and mapping.get(n) is not None]
    unmapped     = [n for n in drug_names if mapping.get(n, "NONE") == "NONE" or mapping.get(n) is None]
    not_fetched  = [n for n in drug_names if n not in mapping]

    print()
    print("=" * 70)
    print("  RXCUI COVERAGE REPORT")
    print("=" * 70)
    print(f"  Total unique drug names : {total:>6,}")
    print(f"  Successfully mapped     : {len(mapped):>6,}  ({100*len(mapped)/total:.1f}%)")
    print(f"  Not found in RxNorm    : {len(unmapped):>6,}  ({100*len(unmapped)/total:.1f}%)")
    if not_fetched:
        print(f"  Not yet fetched        : {len(not_fetched):>6,}  (run without --dry-run)")
    print()

    if unmapped:
        print(f"  First 30 unmapped drugs:")
        for name in sorted(unmapped)[:30]:
            print(f"    {name}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch RXCUI for FAERS drug names")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report coverage from cache only; no API calls or file writes")
    parser.add_argument("--dataset", choices=list(DATASET_DIRS.keys()),
                        help="Process only the specified dataset (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("  RXCUI LOOKUP via RxNorm API")
    print("=" * 70)

    # ── Collect all unique drug names across all target datasets ──────────────
    datasets = {args.dataset: DATASET_DIRS[args.dataset]} if args.dataset else DATASET_DIRS
    all_drug_names: set = set()

    for ds_name, ds_dir in datasets.items():
        csv_path = os.path.join(ds_dir, DRUGS_FILE)
        if not os.path.exists(csv_path):
            print(f"  [{ds_name}] {DRUGS_FILE} not found — skipping")
            continue
        df = pd.read_csv(csv_path, usecols=["DRUG"], dtype=str)
        names = df["DRUG"].dropna().unique()
        all_drug_names.update(names)
        print(f"  [{ds_name}] {len(names):,} unique drug names")

    drug_names = sorted(all_drug_names)
    print(f"\n  Total unique drug names across all datasets: {len(drug_names):,}")

    # ── Load cache and fetch missing ──────────────────────────────────────────
    print(f"\n  Cache file: {os.path.relpath(CACHE_FILE, PROJECT_ROOT)}")
    cached = _load_cache()
    print(f"  Cache size: {len(cached):,} entries")

    fetch_rxcui_for_drugs(drug_names, cached, dry_run=args.dry_run)

    # ── Coverage report ───────────────────────────────────────────────────────
    print_coverage_report(drug_names, cached)

    # ── Add RXCUI column to each dataset CSV ──────────────────────────────────
    print("\n  Adding RXCUI column to DRUGS_STANDARDIZED_DRUGBANK.csv files:")
    for ds_name, ds_dir in datasets.items():
        csv_path = os.path.join(ds_dir, DRUGS_FILE)
        add_rxcui_to_csv(csv_path, cached, dry_run=args.dry_run)

    if args.dry_run:
        print("\n  [dry-run mode — no files written]")
    else:
        print(f"\n  Cache saved → {os.path.relpath(CACHE_FILE, PROJECT_ROOT)}")
        print("  Next step: python scripts3/step2_aer_graph_construction.py")


if __name__ == "__main__":
    main()
