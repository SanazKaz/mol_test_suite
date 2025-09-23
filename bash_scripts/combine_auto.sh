#!/bin/bash

# Batch script to combine DiffSBDD SDF files ending in _1 and _2
# Usage: ./batch_combine_sdf.sh

# Path to the combine script
COMBINE_SCRIPT="utils/combine_sdfs.py"

# Check if combine script exists
if [ ! -f "$COMBINE_SCRIPT" ]; then
    echo "Error: $COMBINE_SCRIPT not found!"
    echo "Make sure the combine_sdf.py script is in the utils/ directory"
    exit 1
fi

# Base directory to search
BASE_DIR="./data/qed_sigmoid"

# Output directory (same as input for this case)
OUTPUT_DIR="$BASE_DIR"

echo "Starting batch combination of DiffSBDD SDF files..."
echo "==========================================="

# Counter for successful combinations
success_count=0
total_count=0

# Find all _1.sdf files and process them
for file1 in $(find "$BASE_DIR" -name "DiffSBDD_*_1.sdf" -type f | sort); do
    # Extract the base name (remove _1.sdf)
    base_name=$(basename "$file1" "_1.sdf")
    dir_name=$(dirname "$file1")
    
    # Construct the _2.sdf filename
    file2="${dir_name}/${base_name}_2.sdf"
    
    # Check if the corresponding _2.sdf file exists
    if [ -f "$file2" ]; then
        # Construct output filename (without _1 or _2)
        output_name="${base_name}.sdf"
        
        echo ""
        echo "Processing pair:"
        echo "  File 1: $file1"
        echo "  File 2: $file2"
        echo "  Output: ${OUTPUT_DIR}/${output_name}"
        
        # Run the combine script
        if python "$COMBINE_SCRIPT" "$file1" "$file2" -o "$OUTPUT_DIR" -n "$output_name"; then
            echo "  ✓ Successfully combined!"
            ((success_count++))
        else
            echo "  ✗ Failed to combine!"
        fi
        
        ((total_count++))
    else
        echo "Warning: No matching _2.sdf file found for $file1"
        echo "  Expected: $file2"
    fi
done

echo ""
echo "==========================================="
echo "Batch combination complete!"
echo "Successfully combined: $success_count out of $total_count pairs"

if [ $success_count -eq $total_count ] && [ $total_count -gt 0 ]; then
    echo "All combinations successful! 🎉"
elif [ $total_count -eq 0 ]; then
    echo "No matching file pairs found to combine."
    echo "Make sure you have DiffSBDD files ending in _1.sdf and _2.sdf"
else
    echo "Some combinations failed. Check the output above for details."
fi