#!/usr/bin/env python3
"""
Main orchestration script for molecular analysis pipeline.

This script orchestrates the complete analysis workflow from SDF files to final results:
1. Property analysis (QED, SA, LogP, MW, etc.)
2. PoseBusters validation
3. Silliness scoring (molecular diversity)
4. Optional SuCOS similarity analysis
5. Comprehensive visualization (property distributions, PoseBusters, 2D grids)
6. Organized results and summary report

Usage:
    # Basic usage - analyze a directory
    python main.py --input data/qed_sigmoid/DiffSBDD_test_pockets/

    # Full analysis with all options
    python main.py \\
        --input data/qed_sigmoid/DiffSBDD_test_pockets/ \\
        --output-dir results/ \\
        --pockets 6v0u 6cm4 6luq \\
        --properties qed sa logp mw \\
        --reference-mol data/reference.sdf
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import shutil

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class AnalysisPipeline:
    """Main orchestrator for molecular analysis pipeline."""

    def __init__(self, input_path: Path, output_dir: Optional[Path] = None,
                 pockets: Optional[List[str]] = None,
                 properties: Optional[List[str]] = None,
                 reference_mol: Optional[Path] = None,
                 skip_plots: bool = False):
        """
        Initialize the analysis pipeline.

        Args:
            input_path: Path to input directory or SDF file
            output_dir: Custom output directory (default: use input directory)
            pockets: List of pocket IDs to filter
            properties: List of properties to plot
            reference_mol: Path to reference molecule for SuCOS analysis
            skip_plots: Skip plotting step
        """
        self.input_path = input_path
        self.output_dir = output_dir if output_dir else input_path
        self.pockets = pockets or []
        self.properties = properties or ["qed", "sa", "logp", "mw"]
        self.reference_mol = reference_mol
        self.skip_plots = skip_plots

        # Setup directories
        self.props_dir = self.output_dir / "molecular_property_csvs"
        self.pb_dir = self.output_dir / "PB_results"
        self.silliness_dir = self.output_dir / "silliness_scores"
        self.figures_dir = self.output_dir / "figures"
        self.logs_dir = self.output_dir / "logs"

        # Setup logging
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_logging()

        # Track results
        self.results = {
            "property_analysis": False,
            "posebusters": False,
            "silliness": False,
            "sucos": False,
            "plots": False,
        }
        self.errors = []

    def _setup_logging(self):
        """Setup logging to file and console."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.logs_dir / f"run_{self.timestamp}.log"

        # Create logger
        self.logger = logging.getLogger("MolAnalysis")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Logging to: {log_file}")

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

    def _print_error(self, text: str):
        """Print error message."""
        self.logger.error(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

    def _run_command(self, cmd: List[str], step_name: str) -> bool:
        """
        Run a subprocess command and log output.

        Args:
            cmd: Command and arguments as list
            step_name: Name of the step for logging

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"\nRunning: {' '.join(str(c) for c in cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout:
                self.logger.debug(result.stdout)
            if result.stderr:
                self.logger.debug(result.stderr)
            self._print_success(f"{step_name} completed")
            return True
        except subprocess.CalledProcessError as e:
            self._print_error(f"{step_name} failed: {e}")
            self.logger.debug(f"stdout: {e.stdout}")
            self.logger.debug(f"stderr: {e.stderr}")
            self.errors.append(f"{step_name}: {str(e)}")
            return False
        except Exception as e:
            self._print_error(f"{step_name} failed with unexpected error: {e}")
            self.errors.append(f"{step_name}: {str(e)}")
            return False

    def validate_input(self) -> List[Path]:
        """
        Validate input and collect all SDF files.

        Returns:
            List of valid SDF file paths
        """
        self._print_header("VALIDATING INPUT")

        sdf_files = []

        if self.input_path.is_file():
            if self.input_path.suffix.lower() == '.sdf':
                sdf_files.append(self.input_path)
            else:
                self._print_error(f"Input file is not an SDF: {self.input_path}")
                return []
        elif self.input_path.is_dir():
            sdf_files = sorted(self.input_path.rglob("*.sdf"))
            if not sdf_files:
                self._print_error(f"No SDF files found in: {self.input_path}")
                return []
        else:
            self._print_error(f"Input path does not exist: {self.input_path}")
            return []

        self.logger.info(f"Found {len(sdf_files)} SDF file(s):")
        for f in sdf_files:
            self.logger.info(f"  - {f.name}")

        return sdf_files

    def run_property_analysis(self) -> bool:
        """Run molecular property analysis using qed_calc.py."""
        self._print_header("STEP 1: MOLECULAR PROPERTY ANALYSIS")

        script = Path("scripts/qed_calc.py")
        if not script.exists():
            self._print_error(f"Script not found: {script}")
            return False

        cmd = [sys.executable, str(script), "--scan", str(self.input_path)]

        success = self._run_command(cmd, "Property analysis")
        if success:
            self.results["property_analysis"] = True
            # Check if output files were created
            props_csv = self.props_dir / "props_method.csv"
            if props_csv.exists():
                self._print_success(f"Properties saved to: {props_csv}")
            else:
                self._print_warning("Property CSV not found in expected location")

        return success

    def run_posebusters(self) -> bool:
        """Run PoseBusters validation."""
        self._print_header("STEP 2: POSEBUSTERS VALIDATION")

        script = Path("scripts/posebusters_check.py")
        if not script.exists():
            self._print_error(f"Script not found: {script}")
            return False

        cmd = [sys.executable, str(script), str(self.input_path)]

        success = self._run_command(cmd, "PoseBusters validation")
        if success:
            self.results["posebusters"] = True
            # Check for output
            pb_results = list(self.pb_dir.glob("*_PB_results.csv"))
            if pb_results:
                self._print_success(f"PoseBusters results: {len(pb_results)} file(s)")

        return success

    def run_silliness_scoring(self, sdf_files: List[Path]) -> bool:
        """Run silliness scoring for all SDF files."""
        self._print_header("STEP 3: SILLINESS SCORING")

        script = Path("scripts/silliness.py")
        if not script.exists():
            self._print_error(f"Script not found: {script}")
            return False

        # Create output directory
        self.silliness_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        for sdf_file in sdf_files:
            output_file = self.silliness_dir / f"{sdf_file.stem}_silliness.csv"
            cmd = [
                sys.executable, str(script),
                "--sdf", str(sdf_file),
                "--output", str(output_file)
            ]

            if self._run_command(cmd, f"Silliness scoring: {sdf_file.name}"):
                success_count += 1

        if success_count > 0:
            self.results["silliness"] = True
            self._print_success(f"Silliness scoring: {success_count}/{len(sdf_files)} files")
            return True

        return False

    def run_sucos_analysis(self, sdf_files: List[Path]) -> bool:
        """Run SuCOS similarity analysis if reference molecule provided."""
        if not self.reference_mol:
            self.logger.info("\nSkipping SuCOS analysis (no reference molecule provided)")
            return True

        self._print_header("STEP 4: SUCOS SIMILARITY ANALYSIS")

        if not self.reference_mol.exists():
            self._print_error(f"Reference molecule not found: {self.reference_mol}")
            return False

        # Note: The sucos_analysis.py script has hardcoded paths
        # For now, we'll document this limitation
        self._print_warning(
            "SuCOS analysis requires manual configuration in sucos_analysis.py"
        )
        self._print_warning(
            "Please update REF_SDF, DIFF_SDF, and PRISM_SDF paths in the script"
        )

        self.logger.info(f"Reference molecule: {self.reference_mol}")

        # This would need the sucos_analysis.py to be refactored to accept CLI args
        # For now, we skip automatic execution

        return True

    def generate_plots(self) -> bool:
        """Generate all visualization plots."""
        if self.skip_plots:
            self.logger.info("\nSkipping plotting (--skip-plots flag)")
            return True

        self._print_header("STEP 5: GENERATING VISUALIZATIONS")

        # Create figures directory structure
        prop_plots_dir = self.figures_dir / "property_distributions"
        pb_plots_dir = self.figures_dir / "posebusters_comparison"
        prop_plots_dir.mkdir(parents=True, exist_ok=True)
        pb_plots_dir.mkdir(parents=True, exist_ok=True)

        success = True

        # 1. Property distribution plots
        props_csv = self.props_dir / "props_method.csv"
        if props_csv.exists():
            self.logger.info("\nGenerating property distribution plots...")
            for prop in self.properties:
                cmd = [
                    sys.executable,
                    "scripts/plotting/property_distribution_plot.py",
                    "--csv", str(props_csv),
                    "--property", prop,
                ]
                if self.pockets:
                    cmd.extend(["--pockets"] + self.pockets)

                if not self._run_command(cmd, f"Property plot: {prop}"):
                    success = False
        else:
            self._print_warning(f"Properties CSV not found: {props_csv}")

        # 2. PoseBusters comparison plots
        pb_results = sorted(self.pb_dir.glob("*_PB_results.csv"))
        if len(pb_results) >= 1:
            self.logger.info("\nGenerating PoseBusters comparison plot...")

            # Build command with multiple input files
            cmd = [sys.executable, "scripts/plotting/martin_style_pb_plot.py"]

            for pb_file in pb_results:
                # Extract method name from filename
                method_name = pb_file.stem.replace("_PB_results", "")
                cmd.extend(["--input", method_name, str(pb_file)])

            output_file = pb_plots_dir / "posebusters_comparison.png"
            cmd.extend(["-o", str(output_file)])

            if not self._run_command(cmd, "PoseBusters comparison plot"):
                success = False
        else:
            self._print_warning("Not enough PoseBusters results for comparison plot")

        if success:
            self.results["plots"] = True

        return success

    def generate_summary_report(self):
        """Generate a summary report of the analysis."""
        self._print_header("GENERATING SUMMARY REPORT")

        report_file = self.output_dir / f"summary_report_{self.timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MOLECULAR ANALYSIS PIPELINE - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input: {self.input_path}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")

            f.write("-" * 80 + "\n")
            f.write("ANALYSIS STEPS COMPLETED:\n")
            f.write("-" * 80 + "\n")
            for step, completed in self.results.items():
                status = "✓ COMPLETED" if completed else "✗ FAILED/SKIPPED"
                f.write(f"{step.upper():.<50} {status}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("OUTPUT FILES:\n")
            f.write("-" * 80 + "\n")

            # List key output files
            output_sections = [
                ("Property Analysis", self.props_dir, "*.csv"),
                ("PoseBusters Results", self.pb_dir, "*_PB_results.csv"),
                ("Silliness Scores", self.silliness_dir, "*.csv"),
                ("Figures", self.figures_dir, "**/*.png"),
            ]

            for section, dir_path, pattern in output_sections:
                f.write(f"\n{section}:\n")
                if dir_path.exists():
                    files = sorted(dir_path.glob(pattern))
                    if files:
                        for file in files:
                            f.write(f"  - {file.relative_to(self.output_dir)}\n")
                    else:
                        f.write("  (No files generated)\n")
                else:
                    f.write("  (Directory not created)\n")

            if self.errors:
                f.write("\n" + "-" * 80 + "\n")
                f.write("ERRORS:\n")
                f.write("-" * 80 + "\n")
                for error in self.errors:
                    f.write(f"  - {error}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Log file: " + str(self.logs_dir / f"run_{self.timestamp}.log") + "\n")
            f.write("=" * 80 + "\n")

        self._print_success(f"Summary report: {report_file}")

    def run_full_pipeline(self) -> bool:
        """Execute the complete analysis pipeline."""
        self._print_header("MOLECULAR ANALYSIS PIPELINE")

        start_time = datetime.now()
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Input: {self.input_path}")
        self.logger.info(f"Output: {self.output_dir}")

        # Validate input
        sdf_files = self.validate_input()
        if not sdf_files:
            self._print_error("Input validation failed")
            return False

        # Run analysis steps
        self.run_property_analysis()
        self.run_posebusters()
        self.run_silliness_scoring(sdf_files)
        self.run_sucos_analysis(sdf_files)
        self.generate_plots()

        # Generate summary
        self.generate_summary_report()

        # Print final summary
        end_time = datetime.now()
        duration = end_time - start_time

        self._print_header("PIPELINE COMPLETE")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"\nResults saved to: {self.output_dir}")

        completed_steps = sum(self.results.values())
        total_steps = len(self.results)
        self.logger.info(f"Completed: {completed_steps}/{total_steps} steps")

        if self.errors:
            self._print_warning(f"Completed with {len(self.errors)} error(s)")
            return False
        else:
            self._print_success("All steps completed successfully!")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Orchestrate complete molecular analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --input data/qed_sigmoid/DiffSBDD_test_pockets/

  # With custom output directory
  python main.py --input data/molecules/ --output-dir results/analysis_20240101/

  # Filter specific pockets and properties
  python main.py --input data/ --pockets 6v0u 6cm4 --properties qed sa

  # Include SuCOS similarity analysis
  python main.py --input data/ --reference-mol data/reference_ligand.sdf

  # Skip plotting
  python main.py --input data/ --skip-plots
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing SDF files or single SDF file"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: use input directory)"
    )

    parser.add_argument(
        "--pockets",
        nargs="+",
        default=None,
        help="List of pocket IDs to filter (e.g., 6v0u 6cm4 6luq)"
    )

    parser.add_argument(
        "--properties",
        nargs="+",
        choices=["qed", "sa", "logp", "mw", "hbd", "hba", "rotb"],
        default=["qed", "sa", "logp", "mw"],
        help="Properties to plot (default: qed sa logp mw)"
    )

    parser.add_argument(
        "--reference-mol",
        type=Path,
        default=None,
        help="Reference molecule SDF for SuCOS similarity analysis (optional)"
    )

    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots"
    )

    args = parser.parse_args()

    # Validate input path
    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)

    # Create pipeline and run
    pipeline = AnalysisPipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        pockets=args.pockets,
        properties=args.properties,
        reference_mol=args.reference_mol,
        skip_plots=args.skip_plots
    )

    success = pipeline.run_full_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
