import pandas as pd
import numpy as np
import subprocess
import os
import shutil
from datetime import datetime

#Create results directory
results_dir = 'optimization_results'
os.makedirs(results_dir, exist_ok=True)

#Read CSV file and get one dam only
df = pd.read_csv('filtered_dams_enriched.csv')
first_dam = df.iloc[12]

print("=== TESTING FIRST DAM ONLY ===")
print(f"Dam Name: {first_dam.get('DAM_NAME', 'Unknown')}")
print(f"Record ID: {first_dam.get('RECORDID', 'Unknown')}")
print(f"State: {first_dam.get('STATE', 'Unknown')}")
print(f"River: {first_dam.get('RIVER', 'Unknown')}")

#Checking for required data
flow_columns = ['MINDLYQ', 'PERC10Q', 'PERC20Q', 'PERC30Q', 'PERC40Q', 
               'PERC50Q', 'PERC60Q', 'PERC70Q', 'PERC80Q', 'PERC90Q', 'MAXDLYQ']

missing_data = []
for col in flow_columns + ['MAX_HT', 'Wholesale Price of Electricity ($/MWh)']:
    if pd.isna(first_dam[col]):
        missing_data.append(col)

if missing_data:
    print(f" Missing data: {missing_data}")
    exit()

print(" All required data present")
print("=" * 50)

try:
    subprocess.run(["python", "globalpars_JSON.py"], capture_output=True)
    #Extract flow values in DESCENDING order
    flow_values = [
        first_dam['MAXDLYQ'],
        first_dam['PERC10Q'],     
        first_dam['PERC20Q'],   
        first_dam['PERC30Q'],    
        first_dam['PERC40Q'],     
        first_dam['PERC50Q'],     
        first_dam['PERC60Q'],   
        first_dam['PERC70Q'],   
        first_dam['PERC80Q'],   
        first_dam['PERC90Q'],     
        first_dam['MINDLYQ']      
    ]
    
    hg = first_dam['MAX_HT']
    wholesale_price_mwh = first_dam['Wholesale Price of Electricity ($/MWh)']
    ep = wholesale_price_mwh / 1000
    
    #Save to input/percentiles.txt
    np.savetxt('input/percentiles.txt', flow_values, delimiter=',', fmt='%.6f')
    
    # Update globalpars_JSON.py with current dam parameters
    with open('globalpars_JSON.py', 'r') as f:
        content = f.read()
    
    # Replace the hardcoded values
    import re
    content = re.sub(r'"hg":\s*[\d.]+', f'"hg": {hg}', content)
    content = re.sub(r'"ep":\s*[\d.]+', f'"ep": {round(float(ep), 5)}', content)
    
    # Write back to globalpars_JSON.py
    with open('globalpars_JSON.py', 'w') as f:
        f.write(content)
    
    print(f"Flow range: {min(flow_values):.1f} - {max(flow_values):.1f} m³/s")
    print(f"Median flow: {first_dam['PERC50Q']:.1f} m³/s")
    print(f"Head: {hg:.1f}m")
    print(f"Electricity price: {ep:.5f} $/kWh")
    
    # Set environment variables for dam info (so plots can access them)
    os.environ['CURRENT_DAM_NAME'] = str(first_dam.get('DAM_NAME', 'Unknown'))
    os.environ['CURRENT_DAM_RECORDID'] = str(first_dam.get('RECORDID', 'Unknown'))
    
    print("\nRunning optimization...")
    start_time = datetime.now()
    
    result = subprocess.run(["python", "Borg_MO_Optimisation.py"], 
                          capture_output=True, text=True, timeout=300)  # 5 min timeout
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    # Check if optimization completed successfully
    if result.returncode == 0:
        print(f"Optimization completed successfully in {elapsed}")
        
        # Create dam identifier
        dam_id = f"dam_000_{first_dam.get('DAM_NAME', 'Unknown').replace(' ', '_').replace('/', '_')}"
        
        print(f"\nSaving results to {results_dir}/...")
        
        # Copy result files to results directory
        saved_files = []
        
        try:
            print("Checking for files in current directory:")
            
            # List ALL files in current directory
            all_files = os.listdir('.')
            png_files = [f for f in all_files if f.endswith('.png')]
            csv_files = [f for f in all_files if f.endswith('.csv')]
            txt_files = [f for f in all_files if f.endswith('.txt')]
            
            print(f"   PNG files found: {png_files}")
            print(f"   CSV files found: {csv_files}")
            print(f"   TXT files found: {txt_files}")
            print(f"   All files: {sorted(all_files)}")
            
            # Check if optimization output shows plots were displayed
            print(f"\nFull optimization output:")
            print("STDOUT:")
            print(result.stdout if result.stdout else "No stdout")
            print("\nSTDERR:")
            print(result.stderr if result.stderr else "No stderr")
            print("\nReturn code:", result.returncode)
            
            # Save CSV results
            if os.path.exists('optimization_table.csv'):
                shutil.copy('optimization_table.csv', f'{results_dir}/{dam_id}_optimization_table.csv')
                saved_files.append('optimization_table.csv')
                print(f"  Saved: optimization_table.csv")
            else:
                print(f"   optimization_table.csv not found")
                
            if os.path.exists('best_table.csv'):
                shutil.copy('best_table.csv', f'{results_dir}/{dam_id}_best_table.csv')
                saved_files.append('best_table.csv')
                print(f"   Saved: best_table.csv")
            else:
                print(f"   best_table.csv not found")
            
            # Copy ALL .png files found
            if png_files:
                for plot_file in png_files:
                    if not plot_file.startswith('dam_'):  # Don't copy already processed files
                        plot_name = plot_file.replace('.png', '')
                        dest_file = f'{results_dir}/{dam_id}_{plot_name}.png'
                        shutil.copy(plot_file, dest_file)
                        saved_files.append(plot_file)
                        print(f"   Saved: {plot_file} -> {dest_file}")
            else:
                print("   No PNG files found! Plots may be displayed but not saved.")
                
        except Exception as e:
            print(f"Warning: Could not save some result files: {e}")
            import traceback
            traceback.print_exc()
        
        print(f" Saved files: {saved_files}")
        
        # Create summary
        summary = {
            'dam_name': first_dam.get('DAM_NAME', 'Unknown'),
            'record_id': first_dam.get('RECORDID', 'Unknown'),
            'state': first_dam.get('STATE', 'Unknown'),
            'river': first_dam.get('RIVER', 'Unknown'),
            'max_flow': first_dam['MAXDLYQ'],
            'median_flow': first_dam['PERC50Q'],
            'min_flow': first_dam['MINDLYQ'],
            'head': hg,
            'electricity_price': ep,
            'optimization_time': str(elapsed),
            'status': 'SUCCESS',
            'saved_files': saved_files
        }
        
        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f'{results_dir}/test_run_summary.csv', index=False)
        
        print(f"\n TEST COMPLETE!")
        print(f"Results saved in '{results_dir}/' directory")
        print(f"Check the plots to verify dam name/ID appear correctly!")
        
    else:
        print(f" Optimization failed:")
        print(f"Error: {result.stderr}")
        print(f"Output: {result.stdout}")
        
except subprocess.TimeoutExpired:
    print(" Optimization timed out (>5 minutes)")
    
except Exception as e:
    print(f" Error: {str(e)}")

print("\n" + "=" * 50)