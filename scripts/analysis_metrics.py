#!/usr/bin/env python3
"""
Molecular Properties Analysis Tool with Nitrogen Valence Corrections

This tool processes SDF files and corrects common valence errors in nitrogen atoms
that can cause RDKit sanitization failures, based on the blog post methodology.

Usage:
    python metrics.py path/to/sdf/directory
    
This will process all SDF files in the directory and output results to CSV.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit import DataStructs
from tqdm import tqdm
import itertools
from typing import List, Sequence, Union

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Add the parent directory of the script to Python's path
# This allows it to find the 'utils' folder as a top-level module
sys.path.append(str(script_dir.parent))

# Try to import the real SA scorer
try:
    from utils.SA_Score.sascorer import calculateScore
except ImportError:
    # If it fails, define a placeholder and assign it to the same name
    print("Warning: SA Score module not found. SA scores will be set to 0.")
    def placeholder_sascorer(mol):
        return 0
    calculateScore = placeholder_sascorer


class MoleculeNitrogenFixer:
    """
    Handle nitrogen valence corrections based on the blog post methodology.
    Fixes common issues like pyrrolic/pyridinic nitrogens and charge problems.
    """
    
    @staticmethod
    def strip_radicals(mol: Chem.Mol):
        """Clear prior explicit Hs and radicals."""
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(False)
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(0)
            atom.UpdatePropertyCache()

    @staticmethod
    def add_nitrogen_charges(mol: Chem.Mol) -> Chem.Mol:
        """
        Fix missing charges on nitrogens that have valence issues.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Chem.Mol: Fixed molecule or None if unfixable
        """
        try:
            mol_copy = Chem.Mol(mol)
            mol_copy.UpdatePropertyCache(strict=False)
            ps = Chem.DetectChemistryProblems(mol_copy)
            
            if not ps:
                Chem.SanitizeMol(mol_copy)
                return mol_copy
                
            for p in ps:
                if p.GetType() == 'AtomValenceException':
                    at: Chem.Atom = mol_copy.GetAtomWithIdx(p.GetAtomIdx())
                    if at.GetAtomicNum() == 7 and at.GetFormalCharge() == 0 and at.GetExplicitValence() == 4:
                        at.SetFormalCharge(1)
            
            Chem.SanitizeMol(mol_copy)
            return mol_copy
            
        except Exception:
            return None

    @staticmethod
    def extend_conjugation(mol: Chem.Mol, bad_idxs: Sequence[int]) -> List[int]:
        """
        Extend the indices to include all conjugations.
        
        Args:
            mol: RDKit molecule
            bad_idxs: Indices of problematic atoms
            
        Returns:
            List of extended atom indices including conjugated systems
        """
        changed = False
        bad_idxs = list(bad_idxs)
        bad_atoms = [mol.GetAtomWithIdx(i) for i in bad_idxs]
        new_bad = []
        
        for atom in bad_atoms:
            for neigh in atom.GetNeighbors():
                ni: int = neigh.GetIdx()
                if ni in bad_idxs:
                    continue
                elif neigh.GetIsAromatic():
                    bad_idxs.append(ni)
                    new_bad.append(neigh)
                    changed = True
                elif mol.GetBondBetweenAtoms(atom.GetIdx(), ni).GetIsConjugated():
                    # pendant
                    new_bad.append(neigh)
                    bad_idxs.append(ni)
                    changed = True
                    
        bad_atoms.extend(new_bad)
        return bad_idxs if not changed else MoleculeNitrogenFixer.extend_conjugation(mol, bad_idxs)

    @staticmethod
    def test_protonation(mol: Chem.Mol, 
                        idxs: Sequence[int],
                        problematics: Sequence[int]) -> Union[Chem.Mol, None]:
        """
        Test different protonation states for nitrogen atoms.
        
        Args:
            mol: RDKit molecule
            idxs: Indices of nitrogens to modify
            problematics: Indices of problematic atoms
            
        Returns:
            Fixed molecule or None if unsuccessful
        """
        copy = Chem.Mol(mol)
        
        for idx in idxs:
            atom: Chem.Atom = copy.GetAtomWithIdx(idx)
            if atom.GetDegree() == 2:  # pyrrolic N?
                atom.SetNumExplicitHs(1)
            elif atom.GetDegree() == 3:  # pyridinium-like N?
                atom.SetFormalCharge(+1)
                
        try:
            problems = Chem.DetectChemistryProblems(copy)
            if not any([problem.GetType() == 'KekulizeException' for problem in problems]):
                return copy
            elif not any([p_idx in p.GetAtomIndices() for p in problems for p_idx in problematics]):
                return copy
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def create_combos(all_atoms: List[Chem.Atom]) -> List[List[int]]:
        """
        Create combinations of nitrogen atoms to test for protonation.
        Returns combinations sorted by degree (pyrrolic first, then pyridinium).
        """
        combos = [list(atoms) for r in range(1, len(all_atoms) + 1) 
                 for atoms in itertools.combinations(all_atoms, r=r)]
        combos = sorted(combos, key=lambda atoms: sum([atom.GetDegree() for atom in atoms]))
        return [[atom.GetIdx() for atom in atoms] for atoms in combos]

    @staticmethod
    def fix_kekulisation(mol: Chem.Mol, leave_idxs=()) -> Chem.Mol:
        """
        Fix kekulization issues by correcting nitrogen protonation/charge.
        
        This is the main function that handles aromatic nitrogen issues like
        pyrrolic vs pyridinic nitrogen problems.
        
        Args:
            mol: RDKit molecule with potential kekulization issues
            leave_idxs: Atom indices to leave unchanged (prevents recursion)
            
        Returns:
            Fixed RDKit molecule
            
        Raises:
            ValueError: If molecule cannot be fixed
        """
        RDLogger.DisableLog('rdApp.*')
        
        try:
            problems = Chem.DetectChemistryProblems(mol)
            if len(problems) == 0:  # success!
                Chem.SanitizeMol(mol)
                RDLogger.EnableLog('rdApp.*')
                return mol
                
            for problem in problems:
                if problem.GetType() == 'KekulizeException':
                    problematics: List[int] = problem.GetAtomIndices()
                    bad_atoms: List[Chem.Atom] = [
                        mol.GetAtomWithIdx(i) 
                        for i in MoleculeNitrogenFixer.extend_conjugation(mol, problematics)
                    ]
                    bad_nitrogens: List[Chem.Atom] = [
                        atom for atom in bad_atoms if atom.GetSymbol() == 'N'
                    ]
                    
                    if not bad_nitrogens:
                        continue
                        
                    combos = MoleculeNitrogenFixer.create_combos(bad_nitrogens)
                    
                    for atoms in combos:
                        if leave_idxs == tuple(atoms):
                            continue  # prevents recursion
                            
                        tested = MoleculeNitrogenFixer.test_protonation(mol, atoms, problematics)
                        if tested:
                            return MoleculeNitrogenFixer.fix_kekulisation(tested, tuple(atoms))
                            
            raise ValueError('Failed to fix kekulization issues')
            
        except Exception as e:
            RDLogger.EnableLog('rdApp.*')
            raise ValueError(f'Failed to fix molecule: {str(e)}')
        finally:
            RDLogger.EnableLog('rdApp.*')


class MolecularMetricsAnalyzer:
    """Analyze molecular properties from SDF files with nitrogen corrections."""
    
    def __init__(self):
        self.nitrogen_fixer = MoleculeNitrogenFixer()
    
    def sanitize_molecule(self, mol: Chem.Mol) -> tuple:
        """
        Attempt to sanitize molecule with nitrogen corrections.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            tuple: (sanitized_mol, status)
                - sanitized_mol: Fixed molecule or None if unfixable
                - status: 'valid', 'fixed', or 'unfixable'
        """
        if mol is None:
            return None, 'unfixable'
            
        # First try standard sanitization
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            return mol_copy, 'valid'
        except Exception:
            pass
            
        # Try fixing nitrogen charges
        try:
            fixed_mol = self.nitrogen_fixer.add_nitrogen_charges(mol)
            if fixed_mol is not None:
                return fixed_mol, 'fixed'
        except Exception:
            pass
            
        # Try fixing kekulization issues
        try:
            mol_copy = Chem.Mol(mol)
            self.nitrogen_fixer.strip_radicals(mol_copy)
            fixed_mol = self.nitrogen_fixer.fix_kekulisation(mol_copy)
            return fixed_mol, 'fixed'
        except Exception:
            pass
            
        return None, 'unfixable'
    
    def load_sdf_molecules(self, sdf_path: str) -> dict:
        """
        Load and sanitize molecules from SDF file.
        
        Args:
            sdf_path: Path to SDF file
            
        Returns:
            dict: Contains 'molecules', 'original_count', 'valid_count', 
                  'fixed_count', 'unfixable_count'
        """
        molecules = []
        original_count = 0
        valid_count = 0
        fixed_count = 0
        unfixable_count = 0
        
        # Suppress RDKit warnings during loading
        RDLogger.DisableLog('rdApp.*')
        
        try:
            supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
            
            for mol in supplier:
                if mol is None:
                    continue
                    
                original_count += 1
                sanitized_mol, status = self.sanitize_molecule(mol)
                
                if sanitized_mol is not None:
                    molecules.append(sanitized_mol)
                    if status == 'valid':
                        valid_count += 1
                    elif status == 'fixed':
                        fixed_count += 1
                else:
                    unfixable_count += 1
                    
        except Exception as e:
            print(f"Error processing {sdf_path}: {str(e)}")
        finally:
            RDLogger.EnableLog('rdApp.*')
        
        return {
            'molecules': molecules,
            'original_count': original_count,
            'valid_count': valid_count,
            'fixed_count': fixed_count,
            'unfixable_count': unfixable_count
        }
    
    def calculate_qed(self, rdmol: Chem.Mol) -> float:
        """Calculate QED (Quantitative Estimate of Drug-likeness)."""
        try:
            return QED.qed(rdmol)
        except Exception:
            return 0.0
    
    def calculate_sa(self, rdmol: Chem.Mol) -> float:
        """Calculate Synthetic Accessibility score."""
        try:
            sa = calculateScore(rdmol)
            return round((10 - sa) / 9, 2)  # Normalized as in original code
        except Exception:
            return 0.0
    
    def calculate_logp(self, rdmol: Chem.Mol) -> float:
        """Calculate LogP (lipophilicity)."""
        try:
            return Crippen.MolLogP(rdmol)
        except Exception:
            return 0.0
    
    def calculate_lipinski_violations(self, rdmol: Chem.Mol) -> int:
        """
        Calculate number of Lipinski rule violations.
        
        Returns:
            int: Number of violations (0-5, lower is better)
        """
        try:
            rule_1 = Descriptors.ExactMolWt(rdmol) < 500
            rule_2 = Lipinski.NumHDonors(rdmol) <= 5
            rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
            logp = Crippen.MolLogP(rdmol)
            rule_4 = (-2 <= logp <= 5)
            rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
            
            violations = sum([not rule for rule in [rule_1, rule_2, rule_3, rule_4, rule_5]])
            return violations
        except Exception:
            return 5  # Maximum violations if calculation fails
    
    def calculate_diversity(self, molecules: List[Chem.Mol]) -> float:
        """
        Calculate average pairwise Tanimoto diversity.
        
        Args:
            molecules: List of RDKit molecules
            
        Returns:
            float: Average diversity (0-1, higher is more diverse)
        """
        if len(molecules) < 2:
            return 0.0
        
        try:
            div = 0
            total = 0
            for i in range(len(molecules)):
                for j in range(i + 1, len(molecules)):
                    fp1 = Chem.RDKFingerprint(molecules[i])
                    fp2 = Chem.RDKFingerprint(molecules[j])
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                    div += 1 - similarity
                    total += 1
            return div / total if total > 0 else 0.0
        except Exception:
            return 0.0
    
    def analyze_sdf_file(self, sdf_path: str) -> dict:
        """
        Analyze a single SDF file with nitrogen corrections.
        
        Args:
            sdf_path: Path to SDF file
            
        Returns:
            dict: Dictionary containing molecular properties and fix statistics
        """
        mol_data = self.load_sdf_molecules(sdf_path)
        molecules = mol_data['molecules']
        
        result = {
            'filename': Path(sdf_path).name,
            'original_molecules': mol_data['original_count'],
            'valid_molecules': len(molecules),
            'initially_valid_molecules': mol_data['valid_count'],
            'fixed_molecules': mol_data['fixed_count'],
            'unfixable_molecules': mol_data['unfixable_count']
        }
        
        if not molecules:
            result.update({
                'qed_mean': 0.0, 'qed_std': 0.0, 'qed_min': 0.0, 'qed_max': 0.0,
                'sa_mean': 0.0, 'sa_std': 0.0, 'sa_min': 0.0, 'sa_max': 0.0,
                'logp_mean': 0.0, 'logp_std': 0.0, 'logp_min': 0.0, 'logp_max': 0.0,
                'lipinski_violations_mean': 5.0, 'lipinski_violations_std': 0.0,
                'lipinski_violations_min': 5.0, 'lipinski_violations_max': 5.0,
                'diversity': 0.0
            })
            return result
        
        # Calculate properties for all valid molecules
        qed_values = [self.calculate_qed(mol) for mol in molecules]
        sa_values = [self.calculate_sa(mol) for mol in molecules]
        logp_values = [self.calculate_logp(mol) for mol in molecules]
        lipinski_values = [self.calculate_lipinski_violations(mol) for mol in molecules]
        diversity = self.calculate_diversity(molecules)
        
        result.update({
            'qed_mean': np.mean(qed_values),
            'qed_std': np.std(qed_values),
            'sa_mean': np.mean(sa_values),
            'sa_std': np.std(sa_values),
            'logp_mean': np.mean(logp_values),
            'logp_std': np.std(logp_values),
            'lipinski_violations_mean': np.mean(lipinski_values),
            'lipinski_violations_std': np.std(lipinski_values),
            'diversity': diversity
        })
        
        return result
    
    def export_individual_molecules(self, directory_path: str, output_csv: str = None) -> pd.DataFrame:
        """
        Export individual molecule data (one row per molecule).
        
        Args:
            directory_path: Path to directory containing SDF files
            output_csv: Path for output CSV file (optional)
            
        Returns:
            pd.DataFrame: Individual molecule data
        """
        sdf_files = glob.glob(os.path.join(directory_path, "*.sdf"))
        
        if not sdf_files:
            print(f"No SDF files found in {directory_path}")
            return pd.DataFrame()
        
        print(f"Found {len(sdf_files)} SDF files to analyze...")
        print("Exporting individual molecule data...")
        
        all_molecules = []
        
        for sdf_file in tqdm(sdf_files, desc="Processing SDF files"):
            mol_data = self.load_sdf_molecules(sdf_file)
            molecules = mol_data['molecules']
            filename = Path(sdf_file).name
            
            for i, mol in enumerate(molecules):
                mol_record = {
                    'filename': filename,
                    'molecule_id': i,
                    'smiles': Chem.MolToSmiles(mol),
                    'qed': self.calculate_qed(mol),
                    'sa': self.calculate_sa(mol),
                    'logp': self.calculate_logp(mol),
                    'lipinski_violations': self.calculate_lipinski_violations(mol),
                    'molecular_weight': Descriptors.ExactMolWt(mol),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                    'num_rings': Chem.rdMolDescriptors.CalcNumRings(mol),
                    'num_aromatic_rings': Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
                    'tpsa': Chem.rdMolDescriptors.CalcTPSA(mol),
                }
                all_molecules.append(mol_record)
        
        df = pd.DataFrame(all_molecules)
        
        # Generate output path if not provided
        if output_csv is None:
            output_csv = os.path.join(directory_path, "individual_molecules.csv")
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"Individual molecule data saved to: {output_csv}")
        
        return df

    def analyze_directory(self, directory_path: str, output_csv: str = None) -> pd.DataFrame:
        """
        Analyze all SDF files in a directory with nitrogen corrections.
        
        Args:
            directory_path: Path to directory containing SDF files
            output_csv: Path for output CSV file (optional)
            
        Returns:
            pd.DataFrame: Results dataframe with fix statistics
        """
        sdf_files = glob.glob(os.path.join(directory_path, "*.sdf"))
        
        if not sdf_files:
            print(f"No SDF files found in {directory_path}")
            return pd.DataFrame()
        
        print(f"Found {len(sdf_files)} SDF files to analyze...")
        print("Processing with nitrogen valence corrections...")
        
        results = []
        for sdf_file in tqdm(sdf_files, desc="Processing SDF files"):
            result = self.analyze_sdf_file(sdf_file)
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Generate output path if not provided
        if output_csv is None:
            output_csv = os.path.join(directory_path, "molecular_properties_analysis.csv")
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
        
        # Print comprehensive summary statistics
        self.print_summary_statistics(df, sdf_files)
        
        return df
    
    def print_summary_statistics(self, df: pd.DataFrame, sdf_files: List[str]):
        """Print comprehensive summary statistics including fix rates."""
        print("\n" + "="*60)
        print("MOLECULAR SANITIZATION & ANALYSIS SUMMARY")
        print("="*60)
        
        total_original = df['original_molecules'].sum()
        total_valid = df['valid_molecules'].sum()
        total_initially_valid = df['initially_valid_molecules'].sum()
        total_fixed = df['fixed_molecules'].sum()
        total_unfixable = df['unfixable_molecules'].sum()
        total_files = len([f for f in sdf_files if os.path.basename(f) in df['filename'].values])
        
        print(f"Files processed: {total_files}")
        print(f"Total molecules encountered: {total_original:,}")
        print(f"Initially valid molecules: {total_initially_valid:,}")
        print(f"Molecules fixed by nitrogen corrections: {total_fixed:,}")
        print(f"Final valid molecules for analysis: {total_valid:,}")
        print(f"Unfixable molecules: {total_unfixable:,}")
        
        if total_original > 0:
            initial_validity = total_initially_valid / total_original
            final_validity = total_valid / total_original
            fix_rate = total_fixed / total_original
            unfixable_rate = total_unfixable / total_original
            
            print(f"\nVALIDITY RATES:")
            print(f"Initial validity rate: {initial_validity:.1%}")
            print(f"Final validity rate: {final_validity:.1%}")
            print(f"Improvement: {final_validity - initial_validity:+.1%}")
            print(f"Fix success rate: {fix_rate:.1%}")
            print(f"Unfixable rate: {unfixable_rate:.1%}")
        
        # File-level statistics
        successful_files = df[df['valid_molecules'] > 0]
        failed_files = df[df['valid_molecules'] == 0]
        print(f"\nFILE STATISTICS:")
        print(f"Files with valid molecules: {len(successful_files)}")
        print(f"Files with no valid molecules: {len(failed_files)}")
        if len(successful_files) > 0:
            print(f"Average molecules per successful file: {successful_files['valid_molecules'].mean():.1f}")
        
        # Molecular properties summary
        if total_valid > 0:
            print(f"\nMOLECULAR PROPERTIES (n={total_valid:,}):")
            print(f"QED: {df['qed_mean'].mean():.3f} ± {df['qed_mean'].std():.3f}")
            print(f"SA: {df['sa_mean'].mean():.3f} ± {df['sa_mean'].std():.3f}")
            print(f"LogP: {df['logp_mean'].mean():.3f} ± {df['logp_mean'].std():.3f}")
            print(f"Lipinski violations: {df['lipinski_violations_mean'].mean():.2f} ± {df['lipinski_violations_mean'].std():.2f}")
            print(f"Diversity: {df['diversity'].mean():.3f} ± {df['diversity'].std():.3f}")
        else:
            print("\nNo valid molecules found for property analysis.")
        
        print("="*60)


def main():
    """Main function for command line usage."""
    if len(sys.argv) != 2:
        print("Usage: python metrics.py <path_to_sdf_directory>")
        print("Example: python metrics.py data/diffsbdd/raw/")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a directory.")
        sys.exit(1)
    
    # Initialize analyzer and run analysis
    print("Starting molecular analysis with nitrogen valence corrections...")
    analyzer = MolecularMetricsAnalyzer()
    analyzer.analyze_directory(directory_path)


if __name__ == "__main__":
    main()