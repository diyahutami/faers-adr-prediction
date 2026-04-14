"""
run_kg_ablation.py
==================
Runs KG ablation experiments, each enabling exactly one (or one combination of) KG step(s):

  A  — drug-target only        (DrugBank drug-protein edges; new target node type)
  B  — DrugBank DDI only       (pharmacological drug-drug interactions)
  C  — SIDER drug-ADR only     (population-level drug-ADR prior from SIDER)
  D  — KGIA comorbidity only   (disease-disease comorbidity from KGIA)
  E  — KGIA risk-factor only   (disease-risk factor edges from KGIA)
  F  — Drug-target + KGIA comorbidity  (best KG + best disease KG; combination)

For each ablation:
  1. Patches KG_EDGES flags in config.py
  2. Rebuilds the AER graph  (step2_aer_graph_construction.py --variant xxx)
  3. Trains and evaluates    (step4_training_infonce.py --variant xxx ...)
  4. Restores config.py

Results summary printed at the end.

Usage
-----
    python scripts3/run_kg_ablation.py

    # Run from project root (parent of scripts3/)
    cd /home/diyah/PhD_Research/ADR_EHR_Prediction_FAERS
    python scripts3/run_kg_ablation.py
"""

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
    ("F_KGIAComorbidRisk_Only",              False, False, False, True,  True,  False, False),
    ("G_DrugTarget_KGIARisk_Only",           True,  False, False, False, True,  False, False),
    ("H_DrugTarget_KGIAComorbidRisk_Only",   True,  False, False, True,  True,  False, False),
    ("I_DrugCentral_Only",                   False, False, False, False, False, True,  False),
    ("J_DrugTarget_DrugCentral",             True,  False, False, False, False, True,  False),
    # ── OnSIDES ablations ────────────────────────────────────────────────────
    ("K_OnSIDES_Only",                       False, False, False, False, False, False, True),
    ("L_OnSIDES_SIDER",                      False, False, True,  False, False, False, True),
    ("M_DrugTarget_OnSIDES",                 True,  False, False, False, False, False, True),
    ("N_DrugTarget_DrugCentral_OnSIDES",     True,  False, False, False, False, True,  True),
]

# ── Training arguments (same as baseline run) ─────────────────────────────────
TRAIN_ARGS = [
    "--variant", "xxx",
    "--no-grid-search",
    "--alpha", "0.0",
    "--tau",   "0.05",
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


def parse_test_auc(log_lines: list[str]) -> str:
    """Extract Test AUC from captured stdout (best-effort)."""
    for line in reversed(log_lines):
        if "AUC:" in line and "test" not in line.lower():
            m = re.search(r"AUC:\s*([\d.]+)", line)
            if m:
                return m.group(1)
    return "N/A"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="KG ablation runner")
    parser.add_argument(
        "--only", nargs="+", metavar="LETTER",
        help="Run only the specified ablation letter(s), e.g. --only D E F"
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

    # Backup config before any changes
    backup = CONFIG_PATH + ".ablation_backup"
    shutil.copy(CONFIG_PATH, backup)
    print(f"Config backed up → {backup}")

    results = []  # list of (name, returncode_step2, returncode_step4, elapsed)

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

            # 1. Patch config
            patch_config(dt, ddi, sider, kgia_cm, kgia_rk, dc_tgt, onsides)
            print(f"  ✓ config.py patched")

            # 2. Rebuild graph
            rc2, t2 = run_step(
                [PYTHON, os.path.join(SCRIPTS_DIR, "step2_aer_graph_construction.py"),
                 "--variant", "xxx"],
                "Step 2: Rebuild AER graph"
            )
            if rc2 != 0:
                print(f"  ✗ Step 2 failed (rc={rc2}) — skipping training for {name}")
                results.append((name, rc2, -1, t2))
                continue

            # 3. Train + evaluate
            rc4, t4 = run_step(
                [PYTHON, os.path.join(SCRIPTS_DIR, "step4_training_infonce.py")] + TRAIN_ARGS,
                "Step 4: Train + evaluate"
            )
            results.append((name, rc2, rc4, t2 + t4))
            status = "✓" if rc4 == 0 else "✗"
            print(f"  {status} {name} complete  ({(t2+t4)/60:.1f} min)")

    finally:
        # Always restore config
        shutil.copy(backup, CONFIG_PATH)
        print(f"\n{'='*70}")
        print(f"  Config restored from backup")
        print(f"{'='*70}")

    # ── Results summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  KG ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Ablation':<25} {'Step2':>6} {'Step4':>6} {'Total':>8}")
    print(f"  {'-'*50}")
    for name, rc2, rc4, elapsed in results:
        s2 = "OK" if rc2 == 0 else f"FAIL({rc2})"
        s4 = "OK" if rc4 == 0 else ("SKIP" if rc4 == -1 else f"FAIL({rc4})")
        print(f"  {name:<35} {s2:>6} {s4:>6} {elapsed/60:>7.1f}m")
    print(f"{'='*70}")
    print(f"\n  ── Prior results (for comparison) ──────────────────────────────")
    print(f"  Baseline (no KG):          Test AUC = 0.5795  Hit@10 = 0.3933")
    print(f"  A DrugTarget only (best):  Test AUC = 0.5857  Hit@10 = 0.3885")
    print(f"  E KGIA Risk only:          Test AUC = 0.5856  Hit@10 = 0.3915")
    print(f"  D KGIA Comorbid only:      Test AUC = 0.5797  Hit@10 = 0.4063 (best Hit@10)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
