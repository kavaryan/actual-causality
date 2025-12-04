#!/bin/bash

# AV RQ1 Case Study - Run data generation and plotting in sequence

echo "=========================================="
echo "AV RQ1 Case Study - Complete Pipeline"
echo "=========================================="

# Step 1: Generate CSV data
echo "Step 1: Generating CSV data..."
python3 subprojects/metamorphic/av_rq1_generate_data.py

# Check if data generation was successful
if [ $? -eq 0 ] && [ -f "av_rq1_scalability_results.csv" ]; then
    echo "✓ Data generation completed successfully"
else
    echo "✗ Data generation failed"
    exit 1
fi

echo ""
echo "Step 2: Creating plots from CSV data..."
python3 subprojects/metamorphic/av_rq1_create_plots.py

# Check if plotting was successful
if [ $? -eq 0 ] && [ -f "av_rq1_scalability_plot.png" ]; then
    echo "✓ Plot creation completed successfully"
else
    echo "✗ Plot creation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "AV RQ1 Study Complete!"
echo "=========================================="
echo "Generated files:"
echo "  - av_rq1_scalability_results.csv (raw data)"
echo "  - av_rq1_improvement_ratios.csv (processed ratios)"
echo "  - av_rq1_improvement_ratios.png (improvement visualization)"
echo "  - av_rq1_aggregated_log_ratios.png (aggregated box plot)"
echo "  - av_rq1_scalability_plot.png (simple visualization)"
echo "  - av_rq1_plot_data.csv (plot data)"
echo "=========================================="
