#!/bin/bash

# RQ2 Lift Case Study - Run data generation and plotting in sequence

echo "=========================================="
echo "RQ2 Lift Case Study - Complete Pipeline"
echo "=========================================="

# cd subprojects/metamorphic

# Step 1: Generate CSV data
echo "Step 1: Generating CSV data..."
# python3 subprojects/metamorphic/lift_rq2_generate_data.py

# Check if data generation was successful
if [ $? -eq 0 ] && [ -f "lift_rq2_ablation_results.csv" ]; then
    echo "✓ Data generation completed successfully"
else
    echo "✗ Data generation failed"
    exit 1
fi

echo ""
echo "Step 2: Creating plots from CSV data..."
python3 subprojects/metamorphic/lift_rq2_create_plots.py

# Check if plotting was successful
if [ $? -eq 0 ] && [ -f "lift_rq2_ablation_plots.png" ]; then
    echo "✓ Plot creation completed successfully"
else
    echo "✗ Plot creation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "RQ2 Study Complete!"
echo "=========================================="
echo "Generated files:"
echo "  - lift_rq2_ablation_results.csv (raw data)"
echo "  - lift_rq2_wilcoxon_tests.csv (statistical tests)"
echo "  - lift_rq2_ablation_plots.png (visualization)"
echo "  - lift_rq2_ablation_plots.pdf (visualization)"
echo "=========================================="
