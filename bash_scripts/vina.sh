#!/bin/bash

# Automated Docking Workflow Script
# Usage: ./auto_dock.sh <ligand_file> <protein_file> [center_x] [center_y] [center_z] [size_x] [size_y] [size_z]
# 
# Parameters:
# - ligand_file: SDF file or single ligand file (PDB/MOL/etc)
# - protein_file: PDB file of the protein
# - Optional: binding site coordinates and box size (defaults provided)

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <ligand_file> <protein_file> [center_x] [center_y] [center_z] [size_x] [size_y] [size_z]"
    echo "Example: $0 ligands.sdf protein.pdb"
    echo "Example with binding site: $0 ligands.sdf protein.pdb 10.5 -5.2 15.8 20 20 20"
    exit 1
fi

# Input parameters
LIGAND_FILE="$1"
PROTEIN_FILE="$2"

# Binding site parameters (with defaults)
CENTER_X="${3:-0}"
CENTER_Y="${4:-0}" 
CENTER_Z="${5:-0}"
SIZE_X="${6:-20}"
SIZE_Y="${7:-20}"
SIZE_Z="${8:-20}"

# Get absolute path and create output directory
LIGAND_DIR=$(dirname "$(realpath "$LIGAND_FILE")")
OUTPUT_DIR="$LIGAND_DIR/vina_results"
WORK_DIR="$OUTPUT_DIR/prepared"

echo "=== Automated Docking Workflow ==="
echo "Ligand file: $LIGAND_FILE"
echo "Protein file: $PROTEIN_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Binding site center: ($CENTER_X, $CENTER_Y, $CENTER_Z)"
echo "Box size: ($SIZE_X x $SIZE_Y x $SIZE_Z)"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$WORK_DIR"

# Get base names
LIGAND_BASE=$(basename "$LIGAND_FILE" | cut -d. -f1)
PROTEIN_BASE=$(basename "$PROTEIN_FILE" | cut -d. -f1)

echo "Step 1: Preparing protein..."
# Clean protein - remove waters and heteroatoms, keep only ATOM records
CLEAN_PDB="$WORK_DIR/${PROTEIN_BASE}_clean.pdb"
PROTEIN_PDBQT="$WORK_DIR/${PROTEIN_BASE}.pdbqt"

grep "^ATOM" "$PROTEIN_FILE" > "$CLEAN_PDB"
echo "✓ Cleaned protein: $CLEAN_PDB"

# Convert protein to PDBQT
obabel -ipdb "$CLEAN_PDB" -opdbqt -O "$PROTEIN_PDBQT" -xr -xh --partialcharge gasteiger
echo "✓ Converted protein to PDBQT: $PROTEIN_PDBQT"

echo ""
echo "Step 2: Preparing ligands..."

# Check if input is SDF (multiple ligands) or single ligand
if [[ "$LIGAND_FILE" == *.sdf ]]; then
    echo "Processing SDF file with multiple ligands..."
    
    # Split SDF into individual ligands and convert to PDBQT
    LIGAND_COUNT=0
    mkdir -p "$WORK_DIR/ligands"
    
    # Split SDF file
    obabel "$LIGAND_FILE" -O "$WORK_DIR/ligands/${LIGAND_BASE}_.sdf" -m
    
    # Convert each ligand to PDBQT
    for sdf_file in "$WORK_DIR/ligands"/*.sdf; do
        if [ -f "$sdf_file" ]; then
            base_name=$(basename "$sdf_file" .sdf)
            pdbqt_file="$WORK_DIR/ligands/${base_name}.pdbqt"
            obabel "$sdf_file" -O "$pdbqt_file" -xh --partialcharge gasteiger
            ((LIGAND_COUNT++))
        fi
    done
    
    LIGAND_PATTERN="$WORK_DIR/ligands/*.pdbqt"
    echo "✓ Prepared $LIGAND_COUNT ligands from SDF file"
    
else
    echo "Processing single ligand file..."
    
    # Convert single ligand to PDBQT
    SINGLE_LIGAND_PDBQT="$WORK_DIR/${LIGAND_BASE}.pdbqt"
    obabel "$LIGAND_FILE" -O "$SINGLE_LIGAND_PDBQT" -xh --partialcharge gasteiger
    LIGAND_PATTERN="$SINGLE_LIGAND_PDBQT"
    LIGAND_COUNT=1
    echo "✓ Prepared single ligand: $SINGLE_LIGAND_PDBQT"
fi

echo ""
echo "Step 4: Running Vina scoring..."

# Initialize results CSV
RESULTS_CSV="$OUTPUT_DIR/docking_scores.csv"
echo "ligand_name,score,ligand_file" > "$RESULTS_CSV"

# Counter for progress
PROCESSED=0

# Process all ligands
for ligand in $LIGAND_PATTERN; do
    if [ -f "$ligand" ]; then
        name=$(basename "$ligand" .pdbqt)
        
        echo "Processing ligand $((PROCESSED + 1))/$LIGAND_COUNT: $name"
        
        # Run Vina scoring
        vina --receptor "$PROTEIN_PDBQT" \
             --ligand "$ligand" \
             --score_only \
             --center_x "$CENTER_X" --center_y "$CENTER_Y" --center_z "$CENTER_Z" \
             --size_x "$SIZE_X" --size_y "$SIZE_Y" --size_z "$SIZE_Z" \
             --out "$OUTPUT_DIR/${name}_scored.pdbqt" \
             --log "$OUTPUT_DIR/${name}.log" 2>/dev/null
        
        # Extract score and append to CSV
        if [ -f "$OUTPUT_DIR/${name}.log" ]; then
            score=$(grep "Affinity:" "$OUTPUT_DIR/${name}.log" | awk '{print $2}' | head -1)
            if [ -z "$score" ]; then
                score="ERROR"
            fi
        else
            score="ERROR"
        fi
        
        echo "${name},${score},${ligand}" >> "$RESULTS_CSV"
        ((PROCESSED++))
        
        echo "  ✓ Score: $score kcal/mol"
    fi
done

echo ""
echo "=== Docking Complete! ==="
echo "Total ligands processed: $PROCESSED"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Files created:"
echo "  - Results CSV: $RESULTS_CSV"
echo "  - Scored ligands: $OUTPUT_DIR/*_scored.pdbqt"
echo "  - Log files: $OUTPUT_DIR/*.log"
echo "  - Prepared files: $WORK_DIR/"
echo ""

# Show top 5 best scores
echo "Top 5 best scores:"
echo "Rank | Ligand | Score (kcal/mol)"
echo "-----|--------|----------------"
tail -n +2 "$RESULTS_CSV" | sort -t',' -k2 -n | head -5 | nl -w4 -s' | ' | sed 's/,/ | /g'

echo ""
echo "Done! 🎉"
