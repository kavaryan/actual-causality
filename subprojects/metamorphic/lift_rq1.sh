#!/bin/bash

# RQ1 Lift Case Study - Run data generation and plotting in sequence

echo "=========================================="
echo "RQ1 Lift Case Study - Complete Pipeline"
echo "=========================================="

# cd subprojects/metamorphic

# Step 1: Generate CSV data
echo "Step 1: Generating CSV data..."
# python3 subprojects/metamorphic/lift_rq1_generate_data.py

# Check if data generation was successful
if [ $? -eq 0 ] && [ -f "rq1_scalability_results.csv" ]; then
    echo "✓ Data generation completed successfully"
else
    echo "✗ Data generation failed"
    exit 1
fi

echo ""
echo "Step 2: Creating plots from CSV data..."
python3 subprojects/metamorphic/lift_rq1_create_plots.py

# Check if plotting was successful
if [ $? -eq 0 ] && [ -f "lift_rq1_improvement_ratios.png" ]; then
    echo "✓ Plot creation completed successfully"
else
    echo "✗ Plot creation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "RQ1 Study Complete!"
echo "=========================================="
echo "Generated files:"
echo "  - rq1_scalability_results.csv (raw data)"
echo "  - rq1_improvement_ratios.csv (processed ratios)"
echo "  - lift_rq1_improvement_ratios.png (improvement visualization)"
echo "  - lift_rq1_aggregated_log_ratios.png (aggregated box plot)"
echo "  - lift_rq1_residual_diagnostics.png (residual plots)"
echo "=========================================="
