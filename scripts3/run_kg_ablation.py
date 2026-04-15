"""
run_kg_ablation.py
==================
Runs KG ablation experiments testing individual and combined KG edge sources.

Each ablation enables a specific combination of the 7 KG edge flags in config.py:

  Single-source ablations:
    A  — DrugTarget Only        (DrugBank drug→protein edges; adds target node type)
    B  — DrugBankDDI Only       (pharmacological drug-drug interactions from DrugBank)
    C  — SIDER Only             (population-level drug-ADR prior from SIDER)
    D  — KGIA Comorbid Only     (disease-disease comorbidity edges from KGIA)
    E  — KGIA Risk Only         (disease-risk factor edges from KGIA)
    F  — DrugCentral Only       (drug→protein edges from DrugCentral 2023)
    G  — OnSIDES Only           (FDA label-derived drug-ADR edges from OnSIDES v3.1)

  Multi-source combinations:
    H  — DrugTarget + DrugCentral      (both protein target sources combined)
    I  — OnSIDES + SIDER               (both drug-ADR prior sources combined)
    J  — KGIA Comorbidity + Risk       (both disease relation sources combined)
    K  — DrugTarget + OnSIDES          (protein target + FDA label ADE prior)
    L  — DrugTarget + KGIA Risk        (protein target + disease risk factor)
    M  — DrugTarget + KGIARisk + OnSIDES  (protein + disease risk + ADE prior)

For each ablation:
  1. Patches KG_EDGES flags in config.py  (backed up and restored afterwards)
  2. Rebuilds the AER graph   (step2_aer_graph_construction.py --variant xxx)
  3. Trains and evaluates     (step4_training_infonce.py --variant xxx --out-suffix _<name>)
  4. Reads final_metrics_infonce.json from the results directory
  5. Restores config.py

At the end, prints a ranked metrics table and saves
  kg_ablation_summary.json  →  results directory
  kg_ablation_summary.txt   →  results directory  (human-readable table)

Usage
-----
    # Run all ablations
    python scripts3/run_kg_ablation.py

    # Run specific ablations only
    python scripts3/run_kg_ablation.py --only A C G

    # Run from project root
    cd /home/diyah/PhD_Research/ADR_EHR_Prediction_FAERS
    python scripts3/run_kg_ablation.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH  = os.path.join(SCRIPTS_DIR, "config.py")
PYTHON       = sys.executable

# ── Ablation configurations ───────────────────────────────────────────────────
# (name, use_drug_target, use_drugbank_ddi, use_sider_adr,
#  use_kgia_comorbidity, use_kgia_risk, use_drugcentral_target, use_onsides_adr)
ABLATIONS = [
    ("A_DrugTarget_Only",                    True,  False, False, False, False, False, False),
    ("B_DrugBankDDI_Only",                   False, True,  False, False, False, False, False),
    ("C_SIDER_Only",                         False, False, True,  False, False, False, False),
    ("D_KGIA_Comorbid_Only",                 False, False, False, True,  False, False, False),
    ("E_KGIA_Risk_Only",                     False, False, False, False, True,  False, False),
    ("F_DrugCentral_Only",                   False, False, False, False, False, True,  False),
    ("G_OnSIDES_Only",                       False, False, False, False, False, False, True),
    ("H_DrugTarget_DrugCentral",             True,  False, False, False, False, True,  False),
    ("I_OnSIDES_SIDER",                      False, False, True,  False, False, False, True),
    ("J_KGIAComorbidRisk_Only",              False, False, False, True,  True,  False, False),
    ("K_DrugTarget_OnSIDES_Only",            True,  False, False, False, False, False, True),
    ("L_DrugTarget_KGIARisk_Only",           True,  False, False, False, True,  False, False),
    ("M_DrugTarget_KGIARisk_OnSIDES_Only",   True,  False, False, False, True, False,  True),
]

# ── Training arguments (same as baseline run) ─────────────────────────────────
TRAIN_ARGS = [
    "--variant", "xxx",
    "--no-grid-search",
    # alpha and tau taken from config defaults (not overridden here)
    # so ablations use the same hyperparameters as the baseline run
]

# ── Config patching ───────────────────────────────────────────────────────────

def patch_config(use_drug_target: bool, use_drugbank_ddi: bool, use_sider_adr: bool,
                 use_kgia_comorbidity: bool, use_kgia_risk: bool,
                 use_drugcentral_target: bool, use_onsides_adr: bool = False) -> None:
    """Overwrite all seven KG_EDGES flags in config.py in-place."""
    with open(CONFIG_PATH) as f:
        content = f.read()

    def _replace(key: str, value: bool, text: str) -> str:
        pattern = rf'("{key}"\s*:\s*)(True|False)'
        replacement = rf'\g<1>{value}'
        return re.sub(pattern, replacement, text)

    content = _replace("use_drug_target",        use_drug_target,       content)
    content = _replace("use_drugbank_ddi",       use_drugbank_ddi,      content)
    content = _replace("use_sider_adr",          use_sider_adr,         content)
    content = _replace("use_kgia_comorbidity",   use_kgia_comorbidity,  content)
    content = _replace("use_kgia_risk",          use_kgia_risk,         content)
    content = _replace("use_drugcentral_target", use_drugcentral_target, content)
    content = _replace("use_onsides_adr",        use_onsides_adr,       content)

    with open(CONFIG_PATH, "w") as f:
        f.write(content)


def run_step(cmd: list, label: str) -> tuple[int, float]:
    """Run a subprocess command; return (returncode, elapsed_seconds)."""
    print(f"\n  ▶ {label}")
    print(f"    {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    return result.returncode, elapsed


def load_metrics(results_path: str, variant: str, name: str) -> dict | None:
    """
    Load final_metrics_infonce.json for a completed ablation run.
    Returns the dict or None if the file doesn't exist.
    """
    metrics_file = os.path.join(results_path, f"{variant}_{name}",
                                "final_metrics_infonce.json")
    if not os.path.exists(metrics_file):
        return None
    with open(metrics_file) as f:
        return json.load(f)


def _get_results_path() -> str:
    """Derive the infonce results path from config without importing it
    (config may be mid-patch during ablation loop)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return os.path.join(cfg.OUTPUT_PATH, f"results_{cfg.DATASET_NAME}_infonce")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KG ablation runner")
    parser.add_argument(
        "--only", nargs="+", metavar="LETTER",
        help="Run only the specified ablation letter(s), e.g. --only A C G"
    )
    args = parser.parse_args()

    # Filter ABLATIONS list if --only is given
    ablations_to_run = ABLATIONS
    if args.only:
        selected = {s.upper() for s in args.only}
        ablations_to_run = [
            row for row in ABLATIONS
            if row[0].split("_")[0].upper() in selected
        ]
        if not ablations_to_run:
            print(f"No ablations matched --only {args.only}. "
                  f"Available: {[r[0].split('_')[0] for r in ABLATIONS]}")
            return
        print(f"Running {len(ablations_to_run)} ablation(s): "
              f"{[r[0] for r in ablations_to_run]}")

    # Derive results path before any config patching
    results_path = _get_results_path()
    os.makedirs(results_path, exist_ok=True)

    # Backup config before any changes
    backup = CONFIG_PATH + ".ablation_backup"
    shutil.copy(CONFIG_PATH, backup)
    print(f"Config backed up → {backup}")

    run_records = []  # list of dicts with run status + metrics

    try:
        for name, dt, ddi, sider, kgia_cm, kgia_rk, dc_tgt, onsides in ablations_to_run:
            print(f"\n{'='*70}")
            print(f"  ABLATION: {name}")
            print(f"    use_drug_target        = {dt}")
            print(f"    use_drugbank_ddi       = {ddi}")
            print(f"    use_sider_adr          = {sider}")
            print(f"    use_kgia_comorbidity   = {kgia_cm}")
            print(f"    use_kgia_risk          = {kgia_rk}")
            print(f"    use_drugcentral_target = {dc_tgt}")
            print(f"    use_onsides_adr        = {onsides}")
            print(f"{'='*70}")

            record = {
                "name"   : name,
                "kg_flags": {
                    "use_drug_target"       : dt,
                    "use_drugbank_ddi"      : ddi,
                    "use_sider_adr"         : sider,
                    "use_kgia_comorbidity"  : kgia_cm,
                    "use_kgia_risk"         : kgia_rk,
                    "use_drugcentral_target": dc_tgt,
                    "use_onsides_adr"       : onsides,
                },
                "step2_ok": False,
                "step4_ok": False,
                "elapsed_min": 0.0,
                "metrics": None,
            }

            # 1. Patch config
            patch_config(dt, ddi, sider, kgia_cm, kgia_rk, dc_tgt, onsides)
            print(f"  ✓ config.py patched")

            # 2. Rebuild graph
            rc2, t2 = run_step(
                [PYTHON, os.path.join(SCRIPTS_DIR, "step2_aer_graph_construction.py"),
                 "--variant", "xxx"],
                "Step 2: Rebuild AER graph"
            )
            record["step2_ok"] = (rc2 == 0)
            if rc2 != 0:
                print(f"  ✗ Step 2 failed (rc={rc2}) — skipping training for {name}")
                record["elapsed_min"] = t2 / 60
                run_records.append(record)
                continue

            # 3. Train + evaluate  (--out-suffix keeps each ablation in its own sub-dir)
            rc4, t4 = run_step(
                [PYTHON, os.path.join(SCRIPTS_DIR, "step4_training_infonce.py")]
                + TRAIN_ARGS + ["--out-suffix", f"_{name}"],
                "Step 4: Train + evaluate"
            )
            record["step4_ok"]    = (rc4 == 0)
            record["elapsed_min"] = (t2 + t4) / 60

            # 4. Load metrics from the JSON file step4 already saved
            if rc4 == 0:
                metrics = load_metrics(results_path, "xxx", name)
                record["metrics"] = metrics
                if metrics:
                    m = metrics.get("test_metrics_cal", {})
                    print(f"  ✓ {name}  AUC={m.get('AUC', float('nan')):.4f}"
                          f"  Hit@5={m.get('Hit@5', float('nan')):.4f}"
                          f"  AUPRC={m.get('AUPRC', float('nan')):.4f}"
                          f"  ({record['elapsed_min']:.1f} min)")
                else:
                    print(f"  ✓ {name} complete but metrics JSON not found")
            else:
                print(f"  ✗ {name} training failed (rc={rc4})")

            run_records.append(record)

    finally:
        # Always restore config
        shutil.copy(backup, CONFIG_PATH)
        print(f"\n  Config restored from backup")

    # ── Save full summary JSON ────────────────────────────────────────────────
    summary_json_path = os.path.join(results_path, "kg_ablation_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(run_records, f, indent=2)
    print(f"\n  Full summary saved → {summary_json_path}")

    # ── Print metrics table ───────────────────────────────────────────────────
    metric_keys = ["AUC", "AUPRC", "Hit@1", "Hit@5", "Hit@10", "NDCG@5", "Brier", "ECE"]

    header = f"  {'Ablation':<40} " + " ".join(f"{k:>8}" for k in metric_keys) + f"  {'min':>6}"
    sep    = "  " + "-" * (len(header) - 2)

    print(f"\n{'='*70}")
    print(f"  KG ABLATION RESULTS SUMMARY  ({len(run_records)} runs)")
    print(f"{'='*70}")
    print(header)
    print(sep)

    # Sort by test AUC descending (failed runs go to bottom)
    def _sort_key(r):
        if r["metrics"] is None:
            return -1.0
        return r["metrics"].get("test_metrics_cal", {}).get("AUC", -1.0)

    for record in sorted(run_records, key=_sort_key, reverse=True):
        name = record["name"]
        if not record["step2_ok"]:
            status = "STEP2_FAIL"
            row = f"  {name:<40} {'':>8}" * len(metric_keys) + f"  {record['elapsed_min']:>6.1f}"
            print(f"  {name:<40} {'STEP2_FAIL':^{8*len(metric_keys)+len(metric_keys)}}  {record['elapsed_min']:>6.1f}m")
            continue
        if not record["step4_ok"] or record["metrics"] is None:
            print(f"  {name:<40} {'TRAIN_FAIL':^{8*len(metric_keys)+len(metric_keys)}}  {record['elapsed_min']:>6.1f}m")
            continue

        m = record["metrics"].get("test_metrics_cal", {})
        vals = " ".join(f"{m.get(k, float('nan')):>8.4f}" for k in metric_keys)
        print(f"  {name:<40} {vals}  {record['elapsed_min']:>6.1f}m")

    print(sep)

    # ── Save human-readable table as .txt ─────────────────────────────────────
    summary_txt_path = os.path.join(results_path, "kg_ablation_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(f"KG Ablation Summary\n{'='*70}\n")
        f.write(header.strip() + "\n")
        f.write(sep.strip() + "\n")
        for record in sorted(run_records, key=_sort_key, reverse=True):
            name = record["name"]
            if not record["step2_ok"] or not record["step4_ok"] or record["metrics"] is None:
                f.write(f"  {name:<40} FAILED  {record['elapsed_min']:.1f}m\n")
                continue
            m = record["metrics"].get("test_metrics_cal", {})
            vals = " ".join(f"{m.get(k, float('nan')):>8.4f}" for k in metric_keys)
            f.write(f"  {name:<40} {vals}  {record['elapsed_min']:>6.1f}m\n")
    print(f"  Text table saved  → {summary_txt_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
