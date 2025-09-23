#!/usr/bin/env python3
"""
Multi-property collector – method and pocket aware.

Inputs:
A) Manual adds (repeatable)
   --add METHOD SDF POCKET_OR_AUTO
   If POCKET_OR_AUTO is 'auto' or '-', pocket is inferred from the filename's second token
   e.g. DiffSBDD_6v0u_20_nodes.sdf -> POCKET=6v0u

B) Auto scan a folder
   --scan DIR
   Picks up *.sdf and infers METHOD from the first token, POCKET from the second.
   Example: DiffSBDD_6cm4_30_nodes.sdf -> METHOD=DiffSBDD, POCKET=6cm4

Outputs (written to <first_sdf_dir>/molecular_property_csvs):
- One CSV per method: props_<method>.csv
  Columns: method, pocket, file, mol_index, smiles, qed, sa, logp, mw, hbd, hba, rotb, lipinski_count, lipinski_pass
- Combined CSV for all methods: props_all_methods.csv
- Counts per pocket and method: counts_per_pocket_method.csv
- Diversity per (method, pocket): diversity_per_pocket.csv
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski

# If you have SA_Score in your repo, import it, else fall back gracefully
try:
    from analysis.SA_Score.sascorer import calculateScore as _sa_score
    _HAS_SA = True
except Exception:
    _HAS_SA = False

# ---------- Inference helpers ----------
def infer_pocket_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path)
    parts = base.split("_")
    return parts[1] if len(parts) >= 2 else None

def infer_method_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path)
    parts = base.split("_")
    return parts[0] if len(parts) >= 1 and parts[0] else None

def sanitize_for_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

# ---------- IO ----------
def load_sdf(path: str) -> List[Chem.Mol]:
    mols: List[Chem.Mol] = []
    try:
        suppl = Chem.SDMolSupplier(path)
        mols = [m for m in suppl if m is not None]
    except Exception as e:
        print(f"[WARN] Failed to read: {path} – {e}")
    return mols

def mol_to_smiles(m: Chem.Mol) -> str:
    try:
        mol = Chem.Mol(m)
        Chem.RemoveStereochemistry(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""

# ---------- Property calculators ----------
def calc_qed(m: Chem.Mol) -> float:
    return float(QED.qed(m))

def calc_sa(m: Chem.Mol) -> Optional[float]:
    if not _HAS_SA:
        return None
    # pocket2mol scaling, higher is “easier”
    sa_raw = float(_sa_score(m))
    return round((10.0 - sa_raw) / 9.0, 3)

def calc_logp(m: Chem.Mol) -> float:
    return float(Crippen.MolLogP(m))

def calc_basic_descriptors(m: Chem.Mol):
    mw = float(Descriptors.ExactMolWt(m))
    hbd = int(Lipinski.NumHDonors(m))
    hba = int(Lipinski.NumHAcceptors(m))
    rotb = int(Chem.rdMolDescriptors.CalcNumRotatableBonds(m))
    return mw, hbd, hba, rotb

def calc_lipinski_count_and_pass(m: Chem.Mol) -> Tuple[int, bool]:
    mw = Descriptors.ExactMolWt(m) < 500
    hbd = Lipinski.NumHDonors(m) <= 5
    hba = Lipinski.NumHAcceptors(m) <= 10
    lp = Crippen.MolLogP(m)
    logp_ok = (-2.0 <= lp <= 5.0)
    rotb_ok = Chem.rdMolDescriptors.CalcNumRotatableBonds(m) <= 10
    count = int(mw) + int(hbd) + int(hba) + int(logp_ok) + int(rotb_ok)
    # Common convention, pass if all 5 are satisfied
    return count, (count == 5)

def rdkit_fp(m: Chem.Mol):
    # Simple RDKit fingerprint, fast and dependency-light
    return Chem.RDKFingerprint(m)

def mean_pairwise_diversity(mols: List[Chem.Mol]) -> Optional[float]:
    """Average of 1 − Tanimoto across all pairs, None if < 2 mols."""
    if len(mols) < 2:
        return None
    fps = [rdkit_fp(m) for m in mols]
    sims = []
    for i, j in combinations(range(len(fps)), 2):
        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    sims = np.asarray(sims, dtype=float)
    return float(np.mean(1.0 - sims)) if sims.size else None

# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute molecular properties for multiple methods and pockets from SDF files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--add",
        action="append",
        nargs=3,
        metavar=("METHOD", "SDF", "POCKET_OR_AUTO"),
        help=("Add one method entry.\n"
              "Example: --add DiffSBDD /path/DiffSBDD_6v0u_20_nodes.sdf auto")
    )
    p.add_argument(
        "--scan",
        type=str,
        default=None,
        help=("Scan a folder for *.sdf and auto-infer METHOD (first token) and "
              "POCKET (second token) from filenames.")
    )
    return p

# ---------- Main ----------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Build the worklist
    entries: List[Tuple[str, str, str]] = []  # (method, sdf_path, pocket)

    # Manual adds
    if args.add:
        for method, sdf, pocket_in in args.add:
            pocket = pocket_in
            if pocket_in in ("auto", "-", ""):
                pocket = infer_pocket_from_filename(sdf) or "unknown"
            entries.append((method, sdf, pocket))

    # Scan mode
    if args.scan:
        scan_dir = Path(args.scan)
        if not scan_dir.is_dir():
            print(f"[WARN] --scan directory not found: {scan_dir}")
        else:
            for sdf_path in scan_dir.rglob("*.sdf"):
                method = infer_method_from_filename(sdf_path.name) or "unknown"
                pocket = infer_pocket_from_filename(sdf_path.name) or "unknown"
                entries.append((method, str(sdf_path), pocket))

    if not entries:
        print("No inputs provided, use --add or --scan, exiting.")
        return

    # Output directory, hard-coded next to the first SDF
    first_sdf_dir = Path(entries[0][1]).parent
    outdir = first_sdf_dir / "molecular_property_csvs"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output CSVs will be written to: {outdir}")

    per_method_rows: Dict[str, List[Dict]] = {}
    combined_rows: List[Dict] = []

    # Also collect for per-pocket diversity
    mol_bank: Dict[Tuple[str, str], List[Chem.Mol]] = {}  # (method, pocket) -> mols

    # Process each SDF
    for method, sdf_path, pocket in entries:
        if not os.path.isfile(sdf_path):
            print(f"[WARN] File not found, skipping: {sdf_path}")
            continue

        mols = load_sdf(sdf_path)
        if not mols:
            print(f"[INFO] No molecules read from: {sdf_path}")
            continue

        key = (method, pocket)
        mol_bank.setdefault(key, []).extend(mols)

        rows_this_file: List[Dict] = []
        for idx, m in enumerate(mols):
            # Sanitize, skip if fails
            try:
                Chem.SanitizeMol(m)
            except Exception as e:
                print(f"[WARN] Sanitize failed in {sdf_path} at index {idx} – {e}")
                continue

            try:
                qed = calc_qed(m)
            except Exception as e:
                print(f"[WARN] QED failed in {sdf_path} at index {idx} – {e}")
                qed = np.nan

            sa = calc_sa(m)  # may be None if SA module missing
            logp = calc_logp(m)
            mw, hbd, hba, rotb = calc_basic_descriptors(m)
            l_count, l_pass = calc_lipinski_count_and_pass(m)
            smi = mol_to_smiles(m)

            row = dict(
                method=method,
                pocket=pocket,
                file=os.path.basename(sdf_path),
                mol_index=idx,
                smiles=smi,
                qed=qed,
                sa=sa,
                logp=logp,
                mw=mw,
                hbd=hbd,
                hba=hba,
                rotb=rotb,
                lipinski_count=l_count,
                lipinski_pass=bool(l_pass),
            )
            rows_this_file.append(row)

        if rows_this_file:
            per_method_rows.setdefault(method, []).extend(rows_this_file)
            combined_rows.extend(rows_this_file)

    if not combined_rows:
        print("No molecules processed after sanitization, exiting.")
        return

    # # Write per-method CSVs
    # for method, rows in per_method_rows.items():
    #     mdf = pd.DataFrame(rows)
    #     mfile = outdir / f"props_{sanitize_for_filename(method)}.csv"
    #     mdf.to_csv(mfile, index=False)
    #     print(f"[OK] Wrote per-molecule properties for method '{method}' -> {mfile}")

    # Combined CSV
    cdf = pd.DataFrame(combined_rows)
    combined_csv = outdir / "props_method.csv"
    cdf.to_csv(combined_csv, index=False)
    print(f"[OK] Wrote combined properties -> {combined_csv}")

    # # Counts per pocket and method
    counts = (
        cdf.groupby(["pocket", "method"])
           .size().reset_index(name="n")
           .sort_values(["pocket", "method"])
    )
    
    
    counts_csv = outdir / "counts_per_pocket_method.csv"
    counts.to_csv(counts_csv, index=False)
    print("\nCounts per pocket and method:")
    for _, r in counts.iterrows():
        print(f"  {str(r['pocket']).upper():>6}  {r['method']:<12}  n = {int(r['n'])}")
    print(f"[OK] Wrote counts -> {counts_csv}")

    # Diversity per (method, pocket)
    div_rows = []
    for (method, pocket), mols in mol_bank.items():
        try:
            div = mean_pairwise_diversity(mols)
        except Exception as e:
            print(f"[WARN] Diversity failed for {method}, {pocket} – {e}")
            div = None
        div_rows.append(dict(method=method, pocket=pocket, n_mols=len(mols), diversity=div))

    div_df = pd.DataFrame(div_rows).sort_values(["pocket", "method"])
    div_csv = outdir / "diversity_per_pocket.csv"
    div_df.to_csv(div_csv, index=False)
    print(f"[OK] Wrote diversity per pocket -> {div_csv}")


if __name__ == "__main__":
    main()
