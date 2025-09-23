from typing import Dict, Any

from rdkit import Chem
# This external library needs to be installed: pip install lilly-medchem-rules
from lilly_medchem_rules import LillyDemeritsFilters

# Initialize the filter object once to be reused.
# This is more efficient than creating it for every molecule.
DEMERIT_FILTER = LillyDemeritsFilters()

def calculate_lilly_demerits(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Runs Lilly MedChem Rules on a single molecule.

    This function checks a molecule against a set of rules designed to identify
    compounds with undesirable physicochemical or structural properties that
    might make them poor drug candidates.

    Args:
        mol: An RDKit Mol object.

    Returns:
        A dictionary containing the 'demerit_score', 'status', and 'reasons'
        for any identified issues. Returns None if the input molecule is invalid.
    """
    if not mol:
        return None

    try:
        # The filter works best with explicit hydrogens removed.
        mol_no_h = Chem.RemoveHs(mol)

        # The filter expects a list of molecules and returns a pandas DataFrame.
        result_df = DEMERIT_FILTER(mols=[mol_no_h])

        if result_df.empty:
            # If the DataFrame is empty, it means no demerits were found.
            return {
                "demerit_score": 0,
                "status": "Pass",
                "reasons": "N/A"
            }

        # Extract the relevant information from the first (and only) row.
        result = result_df.iloc[0]
        return {
            "demerit_score": result['demerit_score'],
            "status": result['status'],
            "reasons": result['reasons']
        }

    except Exception as e:
        # Gracefully handle any unexpected errors during filtering.
        print(f"An error occurred during Lilly demerit calculation: {e}")
        return {
            "demerit_score": -1,
            "status": "Error",
            "reasons": str(e)
        }
