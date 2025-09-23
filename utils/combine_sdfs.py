#!/usr/bin/env python3
"""
Command-line script to combine two SDF files into a single output file.
Useful for consolidating molecular structure datasets.

Usage:
    python combine_sdf_files.py file1.sdf file2.sdf -o output_dir -n combined.sdf
    python combine_sdf_files.py file1.sdf file2.sdf --output-path results --output-name final.sdf
"""

import os
import sys
import argparse
from rdkit import Chem


def combine_sdf_files(sdf_file1, sdf_file2, output_path, output_name):
    """
    Combine two SDF files into a single output file.
    
    Parameters:
    -----------
    sdf_file1 : str
        Path to the first SDF file
    sdf_file2 : str
        Path to the second SDF file
    output_path : str
        Directory path where the combined file should be saved
    output_name : str
        Name of the output file (should include .sdf extension)
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    
    # Validate input files exist
    if not os.path.exists(sdf_file1):
        print(f"Error: File {sdf_file1} does not exist")
        return False
    
    if not os.path.exists(sdf_file2):
        print(f"Error: File {sdf_file2} does not exist")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Construct full output path
    full_output_path = os.path.join(output_path, output_name)
    
    try:
        # Open suppliers for both input files
        supplier1 = Chem.SDMolSupplier(sdf_file1)
        supplier2 = Chem.SDMolSupplier(sdf_file2)
        
        # Create writer for output file
        writer = Chem.SDWriter(full_output_path)
        
        molecules_written = 0
        
        # Write molecules from first file
        print(f"Processing molecules from {sdf_file1}...")
        for mol in supplier1:
            if mol is not None:  # Skip invalid molecules
                writer.write(mol)
                molecules_written += 1
        
        # Write molecules from second file
        print(f"Processing molecules from {sdf_file2}...")
        for mol in supplier2:
            if mol is not None:  # Skip invalid molecules
                writer.write(mol)
                molecules_written += 1
        
        # Close the writer
        writer.close()
        
        print(f"Successfully combined files!")
        print(f"Total molecules written: {molecules_written}")
        print(f"Output saved to: {full_output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during file processing: {str(e)}")
        return False


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Combine two SDF files into a single output file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combine_sdf_files.py file1.sdf file2.sdf -o results -n combined.sdf
  python combine_sdf_files.py data/set1.sdf data/set2.sdf --output-path ./output --output-name final_dataset.sdf
        """
    )
    
    parser.add_argument(
        'sdf_file1',
        help='Path to the first SDF file'
    )
    
    parser.add_argument(
        'sdf_file2', 
        help='Path to the second SDF file'
    )
    
    parser.add_argument(
        '-o', '--output-path',
        required=True,
        help='Directory path where the combined file should be saved'
    )
    
    parser.add_argument(
        '-n', '--output-name',
        required=True,
        help='Name of the output file (should include .sdf extension)'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to handle command line execution.
    """
    args = parse_arguments()
    
    # Combine the files
    success = combine_sdf_files(
        args.sdf_file1, 
        args.sdf_file2, 
        args.output_path, 
        args.output_name
    )
    
    if success:
        print("File combination completed successfully!")
    else:
        print("File combination failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()