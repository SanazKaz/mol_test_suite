import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from openbabel import openbabel
import pandas as pd
from rdkit import Chem
from repos.silly_walks.silly_walks_file import SillyWalks

import warnings
from rdkit import RDLogger

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

def read_sdf_with_rdkit(sdf_path):
    """
    Read an SDF file using RDKit and convert to DataFrame.
    
    Args:
        sdf_path (str): Path to the SDF file
        
    Returns:
        pd.DataFrame: DataFrame with SMILES and Name columns
    """
    supplier = Chem.SDMolSupplier(sdf_path)
    molecules = []
    
    for i, mol in enumerate(supplier):
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
            molecules.append({"SMILES": smiles, "Name": name})
    
    return pd.DataFrame(molecules)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silly Walks")
    parser.add_argument("--sdf", type=str, help="SDF file to process")
    parser.add_argument("--output", type=str, help="Output file", default="None")
    args = parser.parse_args()
    df = read_sdf_with_rdkit(args.sdf)
    ref_df = pd.read_csv("/Users/sanazkazeminia/Documents/mol_test_suite/repos/silly_walks/chembl_drugs.smi", sep=" ", names=["SMILES", "Name"])
    silly_walks = SillyWalks(ref_df)
    df["silly"] = df["SMILES"].apply(silly_walks.score)
    output_path = args.output if args.output else os.path.join(os.path.dirname(args.sdf))
    df.to_csv(output_path, index=False)


# # Load your test molecules
# df = read_sdf_with_rdkit("data/qed_sigmoid/DiffSBDD_test_pockets/DiffSBDD_7e2z_30_nodes.sdf")

# # Load the ChEMBL reference dataset
# ref_df = pd.read_csv("/Users/sanazkazeminia/Documents/mol_test_suite/repos/silly_walks/chembl_drugs.smi", sep=" ", names=["SMILES", "Name"])

# # Initialize SillyWalks with the REFERENCE dataset
# silly_walks = SillyWalks(ref_df)

# # Remove this line: silly_walks = SillyWalks(df)

# df["silly"] = df["SMILES"].apply(silly_walks.score)

# mean_silly = df["silly"].mean()
# median_silly = df["silly"].median()
# max_silly = df["silly"].max()
# min_silly = df["silly"].min()
# print(f"Mean silly: {mean_silly}")
# print(f"Median silly: {median_silly}")
# print(f"Max silly: {max_silly}")
# print(f"Min silly: {min_silly}")

# os.makedirs("data/qed_sigmoid/DiffSBDD_test_pockets/silliness_scores", exist_ok=True)
# df.to_csv("data/qed_sigmoid/DiffSBDD_test_pockets/silliness_scores/DiffSBDD_7e2z_30_nodes_silly.csv", index=False)