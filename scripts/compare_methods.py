#!/usr/bin/env python3
"""
Compare results from multiple method runs.

This script collects results from separate method directories and generates
comparative visualizations and summary statistics.

Usage:
    python scripts/compare_methods.py \
        --method PRISM data/SAscore_QED/PRISM/ \
        --method DiffSBDD data/qed_sigmoid/DiffSBDD_test_pockets/ \
        --pockets 7t2i 6cm4 \
        --properties qed sa logp mw

Author: Generated for molecular analysis pipeline
"""

import argparse
import sys
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class MethodComparator:
    """Compare results from multiple method analysis runs."""

    def __init__(self, 
                 methods: List[Tuple[str, Path]],
                 pockets: Optional[List[str]] = None,
                 properties: Optional[List[str]] = None):
        """
        Initialize the comparator.

        Args:
            methods: List of (method_name, directory_path) tuples
            pockets: List of pocket IDs to filter (optional)
            properties: List of properties to compare (optional)
        """
        self.methods = methods
        self.pockets = pockets or []
        self.properties = properties or ["qed", "sa", "logp", "mw"]
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger("MethodComparator")
        
        # Validate inputs
        self._validate_methods()

    def _validate_methods(self):
        """Validate that all method directories exist and have required data."""
        for name, path in self.methods:
            if not path.exists():
                raise FileNotFoundError(f"Method directory not found: {path}")
            
            # Check in results/ subdirectory first, then fall back to root
            props_csv = path / "results" / "molecular_property_csvs" / "props_method.csv"
            if not props_csv.exists():
                # Try old location for backward compatibility
                props_csv_old = path / "molecular_property_csvs" / "props_method.csv"
                if not props_csv_old.exists():
                    self.logger.warning(
                        f"{Colors.WARNING}Property CSV not found for {name} in results/ or root{Colors.ENDC}"
                    )

    def _print_header(self, text: str):
        """Print a formatted header."""
        border = "=" * 80
        self.logger.info(f"\n{Colors.HEADER}{Colors.BOLD}{border}")
        self.logger.info(f"{text.center(80)}")
        self.logger.info(f"{border}{Colors.ENDC}\n")

    def _print_success(self, text: str):
        """Print success message."""
        self.logger.info(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

    def _print_warning(self, text: str):
        """Print warning message."""
        self.logger.warning(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

    def collect_property_csvs(self) -> Optional[Path]:
        """
        Combine property CSVs from all methods into a single temporary file.

        Returns:
            Path to temporary combined CSV, or None if no data found
        """
        self._print_header("COLLECTING PROPERTY DATA")
        
        combined_dfs = []
        
        for method_name, method_path in self.methods:
            # Check results/ subdirectory first, then fall back to root
            props_csv = method_path / "results" / "molecular_property_csvs" / "props_method.csv"
            if not props_csv.exists():
                props_csv = method_path / "molecular_property_csvs" / "props_method.csv"
            
            if not props_csv.exists():
                self._print_warning(f"Skipping {method_name} - no property CSV found")
                continue
            
            df = pd.read_csv(props_csv)
            
            # Ensure method column is correctly labeled
            df["method"] = method_name
            
            self.logger.info(f"  {method_name}: {len(df)} molecules")
            combined_dfs.append(df)
        
        if not combined_dfs:
            self._print_warning("No property data found in any method directory")
            return None
        
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        
        # Filter by pockets if specified
        if self.pockets:
            combined_df = combined_df[
                combined_df["pocket"].astype(str).str.lower().isin(
                    [p.lower() for p in self.pockets]
                )
            ]
            self.logger.info(f"\nFiltered to pockets: {', '.join(self.pockets)}")
            self.logger.info(f"Total molecules after filtering: {len(combined_df)}")
        
        # Save to temporary file
        temp_csv = Path(tempfile.gettempdir()) / f"comparison_{self.timestamp}.csv"
        combined_df.to_csv(temp_csv, index=False)
        
        self._print_success(f"Combined data saved to temporary file")
        return temp_csv

    def collect_pb_results(self) -> Dict[str, List[Path]]:
        """
        Collect PoseBusters result files from all methods.

        Returns:
            Dict mapping method names to lists of PB result CSV paths
        """
        self._print_header("COLLECTING POSEBUSTERS RESULTS")
        
        pb_files = {}
        
        for method_name, method_path in self.methods:
            # Check results/ subdirectory first, then fall back to root
            pb_dir = method_path / "results" / "PB_results"
            if not pb_dir.exists():
                pb_dir = method_path / "PB_results"
            
            if not pb_dir.exists():
                self._print_warning(f"No PB results for {method_name}")
                continue
            
            files = list(pb_dir.glob("*_PB_results.csv"))
            if files:
                pb_files[method_name] = files
                self.logger.info(f"  {method_name}: {len(files)} PB result file(s)")
        
        if pb_files:
            self._print_success(f"Found PB results for {len(pb_files)} method(s)")
        else:
            self._print_warning("No PoseBusters results found")
        
        return pb_files

    def generate_property_plots(self, combined_csv: Path, output_dir: Path):
        """
        Generate property distribution comparison plots.

        Args:
            combined_csv: Path to combined property CSV
            output_dir: Directory to save plots
        """
        self._print_header("GENERATING PROPERTY COMPARISON PLOTS")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        method_names = [name for name, _ in self.methods]
        
        for prop in self.properties:
            self.logger.info(f"\nGenerating {prop.upper()} comparison plot...")
            
            cmd = [
                sys.executable,
                "scripts/plotting/property_distribution_plot.py",
                "--csv", str(combined_csv),
                "--property", prop,
                "--output-dir", str(output_dir),
                "--methods"] + method_names
            
            if self.pockets:
                cmd.extend(["--pockets"] + self.pockets)
            
            try:
                import subprocess
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                self._print_success(f"{prop.upper()} plot generated")
            except subprocess.CalledProcessError as e:
                self._print_warning(f"Failed to generate {prop} plot: {e}")
                if e.stderr:
                    self.logger.error(f"Error output:\n{e.stderr}")
                if e.stdout:
                    self.logger.info(f"Standard output:\n{e.stdout}")

    def generate_pb_comparison_plot(self, pb_files: Dict[str, List[Path]], 
                                   output_dir: Path):
        """
        Generate PoseBusters comparison plot.

        Args:
            pb_files: Dict mapping method names to PB result file paths
            output_dir: Directory to save plot
        """
        self._print_header("GENERATING POSEBUSTERS COMPARISON")
        
        if not pb_files:
            self._print_warning("No PoseBusters data to compare")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "posebusters_comparison.png"
        
        # Build command with all PB files
        cmd = [sys.executable, "scripts/plotting/martin_style_pb_plot.py"]
        
        for method_name, files in pb_files.items():
            # Use the first/combined file for each method
            main_file = [f for f in files if "combined" in f.name]
            if not main_file:
                main_file = files
            
            cmd.extend(["--input", method_name, str(main_file[0])])
        
        cmd.extend(["-o", str(output_file)])
        
        try:
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            self._print_success("PoseBusters comparison plot generated")
        except subprocess.CalledProcessError as e:
            self._print_warning(f"Failed to generate PB comparison: {e}")
            if e.stderr:
                self.logger.error(f"Error output:\n{e.stderr}")
            if e.stdout:
                self.logger.info(f"Standard output:\n{e.stdout}")

    def generate_summary_report(self, combined_csv: Optional[Path], 
                                pb_files: Dict[str, List[Path]]) -> str:
        """
        Generate a text summary report with comparative statistics.

        Args:
            combined_csv: Path to combined property CSV
            pb_files: Dict of PoseBusters result files

        Returns:
            Path to summary report file
        """
        self._print_header("GENERATING SUMMARY REPORT")
        
        # Save to first method's results directory if it exists, otherwise root
        first_method_dir = self.methods[0][1]
        if (first_method_dir / "results").exists():
            report_path = first_method_dir / "results" / f"comparison_summary_{self.timestamp}.txt"
        else:
            report_path = first_method_dir / f"comparison_summary_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("METHOD COMPARISON SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Methods compared: {', '.join([name for name, _ in self.methods])}\n")
            if self.pockets:
                f.write(f"Pockets: {', '.join(self.pockets)}\n")
            f.write("\n")
            
            # Property statistics
            if combined_csv and combined_csv.exists():
                f.write("-" * 80 + "\n")
                f.write("MOLECULAR PROPERTY STATISTICS\n")
                f.write("-" * 80 + "\n\n")
                
                df = pd.read_csv(combined_csv)
                
                for prop in self.properties:
                    if prop not in df.columns:
                        continue
                    
                    f.write(f"{prop.upper()}:\n")
                    stats = df.groupby("method")[prop].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
                    f.write(stats.to_string() + "\n\n")
            
            # PoseBusters statistics
            if pb_files:
                f.write("-" * 80 + "\n")
                f.write("POSEBUSTERS VALIDATION RATES\n")
                f.write("-" * 80 + "\n\n")
                
                for method_name, files in pb_files.items():
                    main_file = [f for f in files if "combined" in f.name]
                    if not main_file:
                        main_file = files
                    
                    pb_df = pd.read_csv(main_file[0])
                    if 'mol_pred_loaded' in pb_df.columns:
                        total = pb_df['mol_pred_loaded'].sum()
                        if total > 0:
                            valid = (pb_df.drop(columns=['mol_pred_loaded']).sum(axis=1) == 
                                   len(pb_df.columns) - 1).sum()
                            pass_rate = (valid / total) * 100
                            f.write(f"{method_name}:\n")
                            f.write(f"  Total molecules: {total}\n")
                            f.write(f"  Valid poses: {valid}\n")
                            f.write(f"  Pass rate: {pass_rate:.2f}%\n\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Report saved to: {report_path}\n")
            f.write("=" * 80 + "\n")
        
        self._print_success(f"Summary report: {report_path}")
        return str(report_path)

    def distribute_plots_to_methods(self, source_dir: Path):
        """
        Copy comparison plots to all method directories.

        Args:
            source_dir: Directory containing the generated plots
        """
        self._print_header("DISTRIBUTING PLOTS TO METHOD DIRECTORIES")
        
        plot_files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.svg"))
        
        if not plot_files:
            self._print_warning("No plot files found to distribute")
            return
        
        for method_name, method_path in self.methods:
            # Save to results/figures/comparisons/ if results/ exists, otherwise use root
            if (method_path / "results").exists():
                target_dir = method_path / "results" / "figures" / "comparisons"
            else:
                target_dir = method_path / "figures" / "comparisons"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for plot_file in plot_files:
                target_file = target_dir / plot_file.name
                shutil.copy2(plot_file, target_file)
            
            self.logger.info(f"  {method_name}: {len(plot_files)} files copied")
        
        self._print_success(f"Plots distributed to all {len(self.methods)} method directories")

    def run_comparison(self):
        """Execute the complete comparison workflow."""
        self._print_header("METHOD COMPARISON PIPELINE")
        
        start_time = datetime.now()
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Comparing: {', '.join([name for name, _ in self.methods])}\n")
        
        # Create temporary output directory
        temp_output = Path(tempfile.gettempdir()) / f"comparison_plots_{self.timestamp}"
        temp_output.mkdir(parents=True, exist_ok=True)
        
        try:
            # Collect and combine data
            combined_csv = self.collect_property_csvs()
            pb_files = self.collect_pb_results()
            
            # Generate plots
            if combined_csv:
                # Property plots go to plots/ subdirectory
                plots_dir = temp_output / "plots"
                self.generate_property_plots(combined_csv, plots_dir)
            
            if pb_files:
                self.generate_pb_comparison_plot(pb_files, temp_output)
            
            # Distribute to all method directories
            self.distribute_plots_to_methods(temp_output)
            
            # Generate summary report
            self.generate_summary_report(combined_csv, pb_files)
            
        finally:
            # Cleanup temporary files
            if combined_csv and combined_csv.exists():
                combined_csv.unlink()
            if temp_output.exists():
                shutil.rmtree(temp_output)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        self._print_header("COMPARISON COMPLETE")
        self.logger.info(f"Duration: {duration}")
        self._print_success("All comparison plots generated and distributed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare results from multiple method analysis runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two methods
  python scripts/compare_methods.py \\
    --method PRISM data/SAscore_QED/PRISM/ \\
    --method DiffSBDD data/qed_sigmoid/DiffSBDD_test_pockets/

  # Compare with specific pockets and properties
  python scripts/compare_methods.py \\
    --method PRISM data/PRISM/ \\
    --method DiffSBDD data/DiffSBDD/ \\
    --pockets 7t2i 6cm4 \\
    --properties qed sa logp
        """
    )
    
    parser.add_argument(
        "--method",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "PATH"),
        help="Method name and directory path (can be repeated)"
    )
    
    parser.add_argument(
        "--pockets",
        nargs="+",
        default=None,
        help="List of pocket IDs to filter (optional)"
    )
    
    parser.add_argument(
        "--properties",
        nargs="+",
        choices=["qed", "sa", "logp", "mw", "hbd", "hba", "rotb"],
        default=["qed", "sa", "logp", "mw"],
        help="Properties to compare (default: qed sa logp mw)"
    )
    
    args = parser.parse_args()
    
    # Convert method paths to Path objects
    methods = [(name, Path(path)) for name, path in args.method]
    
    # Create comparator and run
    comparator = MethodComparator(
        methods=methods,
        pockets=args.pockets,
        properties=args.properties
    )
    
    try:
        comparator.run_comparison()
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()

