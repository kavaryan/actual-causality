import sys
import os

import pandas as pd
sys.path.append('subprojects/metamorphic')
sys.path.append('subprojects/metamorphic/case-studies/lift')

import comparison_runner
from comparison_runner import run_comparison_study, plot_execution_time_vs_num_vars

# df = run_comparison_study(num_vars_list = [0, 5, 10, 25, 50, 100, 500],
#     num_trials = 10,
#     awt_coeff = 0.8,
#     timeout = 2)

# df.to_csv('execution_time_vs_num_vars.csv', index=False)



awt_coeffs = [0.5, 0.6, 0.7, 0.8, 0.9]                                                                
all_results = []                                                                                      
                                                                                                        
for awt_coeff in awt_coeffs:                                                                          
    print(f"Running with awt_coeff={awt_coeff}")                                                      
    df = run_comparison_study(                                                                        
        num_vars_list=[20],                                                                           
        num_trials=10,                                                                                
        awt_coeff=awt_coeff,                                                                          
        timeout=2                                                                                     
    )                                                                                                 
    df['awt_coeff'] = awt_coeff                                                                       
    all_results.append(df)                                                                            
                                                                                                        
# Combine all results                                                                                 
combined_df = pd.concat(all_results, ignore_index=True)                                               
                                                                                                        
# Save results                                                                                        
combined_df.to_csv('comparison_results_awt_coeff.csv', index=False)  