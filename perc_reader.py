import pandas as pd
import numpy as np
import subprocess
import os
import shutil
from datetime import datetime

# Create results directory
results_dir = 'optimization_results'
os.makedirs(results_dir, exist_ok=True)

# Read CSV file and get one dam only
df = pd.read_csv('filtered_dams_enriched.csv')
first_dam = df.iloc[12]

print("Testing one dam:")
print(f"Dam Name: {first_dam.get('DAM_NAME')}")
print(f"Record ID: {first_dam.get('RECORDID')}")
print(f"State: {first_dam.get('STATE')}")

# Regenerate base JSON file
subprocess.run(["python", "globalpars_JSON.py"], capture_output=True)

# Extract flow values in DESCENDING order
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

#Extract parameters
hg = first_dam['MAX_HT']
wholesale_price_mwh = first_dam['Wholesale Price of Electricity ($/MWh)']
ep = wholesale_price_mwh / 1000

#Save flow data
np.savetxt('input/fairmonth_percentiles.txt', flow_values, delimiter=',', fmt='%.6f')

#Update JSON parameters
with open('globalpars_JSON.py', 'r') as f:
    content = f.read()

import re
content = re.sub(r'"hg":\s*[\d.]+', f'"hg": {hg}', content)
content = re.sub(r'"ep":\s*[\d.]+', f'"ep": {round(float(ep), 5)}', content)

with open('globalpars_JSON.py', 'w') as f:
    f.write(content)

print(f"Flow: {min(flow_values):.1f} - {max(flow_values):.1f} m³/s")
print(f"Head: {hg:.1f}m, Price: {ep:.5f} $/kWh")

# Set dam info for plots
os.environ['CURRENT_DAM_NAME'] = str(first_dam.get('DAM_NAME'))
os.environ['CURRENT_DAM_RECORDID'] = str(first_dam.get('RECORDID'))

# Run optimization
print("\nOptimizing...")
start_time = datetime.now()

result = subprocess.run(["python", "Borg_MO_Optimisation.py"], 
                      capture_output=True, text=True, timeout=300)

elapsed = datetime.now() - start_time

if result.returncode == 0:
    print(f"SUCCESS in {elapsed}")
    
    #Copy files to results directory
    dam_id = f"dam_000_{first_dam.get('DAM_NAME').replace(' ', '_').replace('/', '_')}"
    
    #Copy CSV files
    for csv_file in ['optimization_table.csv', 'best_table.csv']:
        if os.path.exists(csv_file):
            shutil.copy(csv_file, f'{results_dir}/{dam_id}_{csv_file}')
    
    #Copy PNG files
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    for plot_file in png_files:
        if not plot_file.startswith('dam_'):
            plot_name = plot_file.replace('.png', '')
            shutil.copy(plot_file, f'{results_dir}/{dam_id}_{plot_name}.png')
    
    print(f"Files saved to {results_dir}/")
    
else:
    print("FAILED")
    print(result.stderr)