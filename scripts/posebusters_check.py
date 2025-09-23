import argparse
import pandas as pd
from pathlib import Path
from rdkit import Chem
from glob import glob
from posebusters import PoseBusters

def load_molecules(sdf_file_paths: list[Path]) -> list:
    """
    Loads all molecules from a list of SDF file paths into a single list.

    Args:
        sdf_file_paths (list[Path]): A list of paths to SDF files.

    Returns:
        list: A single list containing all RDKit molecule objects from the files.
    """
    all_mols = []
    for file_path in sdf_file_paths:
        print(f"-> Loading molecules from '{file_path.name}'...")
        # removeHs=False is important as PoseBusters needs the hydrogens for its checks.
        supplier = Chem.SDMolSupplier(str(file_path), removeHs=False)
        mols_in_file = [mol for mol in supplier if mol is not None]
        if not mols_in_file:
            print(f"   Warning: No valid molecules found in '{file_path.name}'.")
        all_mols.extend(mols_in_file)
    return all_mols

def analyze_poses(mols: list):
    """
    Runs PoseBusters on a list of RDKit molecules and returns the raw results and a failure analysis.

    This function corrects the failure counting by first identifying poses that failed
    any check, and then, for that subset of failed poses, it counts which specific
    checks were the culprits.

    Args:
        mols (list): A list of RDKit molecule objects to analyze.

    Returns:
        tuple: A tuple containing (raw_results_df, failure_counts_df).
               Returns (None, None) if the list of molecules is empty.
    """
    if not mols:
        print("Error: No molecules provided for analysis.")
        return None, None

    # --- 1. Run PoseBusters ---
    # Initialize PoseBusters with a fast configuration.
    # The .bust() method runs all checks and returns a pandas DataFrame with the results.
    print(f"\nRunning PoseBusters on {len(mols)} total poses...")
    buster = PoseBusters(config="mol_fast")
    raw_results_df = buster.bust(mols)

    # --- 2. Analyze Failures ---
    # Get a list of all columns that represent a PoseBusters check (these are boolean type).
    check_columns = [col for col in raw_results_df.columns if raw_results_df[col].dtype == 'bool']

    if not check_columns:
        print("Warning: No boolean check columns found in the PoseBusters output.")
        return raw_results_df, pd.DataFrame() # Return raw results but empty failures

    # Identify all poses that failed at least one check.
    raw_results_df['passed_all_checks'] = raw_results_df[check_columns].all(axis=1)
    failed_poses_df = raw_results_df[~raw_results_df['passed_all_checks']]

    # If there are any failed poses, count the reasons for failure.
    # We iterate through the check columns ONLY for the failed poses.
    # The logic `(~df[col]).sum()` inverts the boolean (False -> True) and sums them up,
    # effectively counting the number of `False` values, which are the failures for that check.
    failure_counts = {}
    if not failed_poses_df.empty:
        print(f"Found {len(failed_poses_df)} poses that failed one or more checks.")
        for col in check_columns:
            # Count how many of the FAILED poses failed THIS specific check
            num_failures = (~failed_poses_df[col]).sum()
            if num_failures > 0:
                failure_counts[col] = num_failures
    else:
        print("🎉 Success! All poses passed every check.")

    # Convert the failure counts dictionary to a sorted DataFrame for clarity.
    failure_counts_df = pd.DataFrame(
        list(failure_counts.items()), columns=['Check', 'Failure Count']
    ).sort_values(by='Failure Count', ascending=False).reset_index(drop=True)

    return raw_results_df, failure_counts_df


def main():
    """
    Main function to parse command-line arguments and orchestrate the analysis.
    """
    # --- 3. Set up Command-Line Argument Parsing ---
    # argparse now accepts one or more input paths. These can be files, directories,
    # or wildcard patterns like 'data/*.sdf' which your shell will expand.
    parser = argparse.ArgumentParser(
        description="Run PoseBusters on SDF files and generate result CSVs. Accepts single files, directories, or wildcards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_paths",
        type=str,
        nargs='+', # This allows for one or more arguments
        help="Path(s) to input SDF file(s), directories containing SDF files, or wildcard patterns (e.g., 'data/*.sdf')."
    )
    args = parser.parse_args()

    # --- 4. Discover all SDF files from the input paths ---
    sdf_files_to_process = []
    for path_str in args.input_paths:
        # The glob module is used to handle wildcard expansion for paths.
        # This works reliably across different operating systems and shells.
        expanded_paths = glob(path_str)
        for expanded_path_str in expanded_paths:
            path = Path(expanded_path_str)
            if path.is_dir():
                # If the path is a directory, find all .sdf files within it.
                sdf_files_to_process.extend(path.glob('*.sdf'))
            elif path.is_file() and path.suffix.lower() == '.sdf':
                # If it's a file, just add it to the list.
                sdf_files_to_process.append(path)

    # Remove duplicates (e.g., if user provides 'data/' and 'data/mol.sdf') and sort.
    unique_sdf_files = sorted(list(set(sdf_files_to_process)))
    
    if not unique_sdf_files:
        print(f"Error: No SDF files found matching the specified path(s): {', '.join(args.input_paths)}")
        return

    print(f"Found {len(unique_sdf_files)} unique SDF file(s) to process.")

    # --- 5. Load all molecules from all files into a single list ---
    all_mols = load_molecules(unique_sdf_files)
    if not all_mols:
        # load_molecules will have already printed warnings.
        print("\nNo valid molecules could be loaded. Exiting.")
        return

    # --- 6. Determine Output Paths ---
    if len(unique_sdf_files) == 1:
        # If only one file was processed, name the output after that file.
        input_path = unique_sdf_files[0]
        output_dir = input_path.parent / "PB_results"
        file_stem = input_path.stem
    else:
        # If multiple files were processed, create a combined output name
        # based on the parent directory of the first file.
        first_path = unique_sdf_files[0]
        # Use the common parent directory for output to avoid confusion
        common_parent = first_path.parent
        output_dir = common_parent / "PB_results"
        file_stem = f"{common_parent.name}_combined_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct the full paths for the two output CSV files.
    raw_results_csv_path = output_dir / f"{file_stem}_PB_results.csv"
    failure_count_csv_path = output_dir / f"{file_stem}_PB_failure_count.csv"

    # --- 7. Run Analysis and Save Results ---
    raw_results, failure_counts = analyze_poses(all_mols)

    # Save the results to CSV files, if the analysis was successful.
    if raw_results is not None:
        raw_results.to_csv(raw_results_csv_path, index=False)
        print(f"\n✅ Raw PoseBusters results saved to:\n   {raw_results_csv_path}")

    if failure_counts is not None and not failure_counts.empty:
        failure_counts.to_csv(failure_count_csv_path, index=False)
        print(f"📊 Failure analysis saved to:\n   {failure_count_csv_path}")
    elif failure_counts is not None:
        print("No failures to report, so no failure count file was created.")


# --- Script Execution ---
if __name__ == "__main__":
    main()

