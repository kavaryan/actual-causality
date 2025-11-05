#!/bin/bash

# RQ2 Ablation Study Runner
# This script runs the complete RQ2 ablation study pipeline:
# 1. Generate experimental data
# 2. Analyze results and create visualizations

echo "=========================================="
echo "RQ2 ABLATION STUDY - COMPLETE PIPELINE"
echo "=========================================="
echo ""
echo "This study compares:"
echo "- A* (no bundling)"
echo "- A* Bundled with bundle sizes: 5, 10, 20"
echo "- Problem sizes: N=40, N=60"
echo "- Timeout: 30 seconds per experiment"
echo ""

# Check if we're in the right directory
if [ ! -f "lift_rq2_generate_data.py" ]; then
    echo "Error: lift_rq2_generate_data.py not found!"
    echo "Please run this script from the subprojects/metamorphic directory"
    exit 1
fi

# Step 1: Generate data
echo "Step 1: Generating experimental data..."
echo "This may take 30-60 minutes depending on your system..."
echo ""

python3 lift_rq2_generate_data.py

if [ $? -ne 0 ]; then
    echo "Error: Data generation failed!"
    exit 1
fi

echo ""
echo "✓ Data generation completed successfully!"
echo ""

# Step 2: Analyze results
echo "Step 2: Analyzing results and creating visualizations..."
echo ""

python3 lift_rq2_analyze_results.py

if [ $? -ne 0 ]; then
    echo "Error: Analysis failed!"
    exit 1
fi

echo ""
echo "✓ Analysis completed successfully!"
echo ""

# Summary
echo "=========================================="
echo "RQ2 STUDY COMPLETED!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "- lift_rq2_ablation_results.csv       (raw experimental data)"
echo "- lift_rq2_wilcoxon_tests.csv         (statistical test results)"
echo "- lift_rq2_ablation_plots.png         (visualizations)"
echo "- lift_rq2_ablation_plots.pdf         (visualizations)"
echo ""
echo "You can now examine the results to answer RQ2:"
echo "What is the effect of bundling and different bundle sizes"
echo "on the performance of the A* metamorphic search algorithm?"
echo ""
