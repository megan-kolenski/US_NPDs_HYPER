"""
############################################################################
#                   Written by Veysel Yildiz                               #
#                   vysl.yildiz@gmail.com,  veysel.yildiz@duke.edu         #
#                   The University of Sheffield, 2024                      #
#                   Duke University, 2025                                  #
############################################################################
"""
""" Return :
op_table: optimization table constructed with the optimization result parameters,
 including the objective function value, turbine type, turbine configuration, 
 penstock diameter, and turbine design discharges. 
 
Scatter Plot: Pareto Front of design alternatives 
--------------------------------------
  Inputs:

    result : Optimization result
 """

# Import  the modules to be used from Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # Import tabulate
from MO_energy_function import MO_Sim_energy
import os

 
def postplot(result): 

 # Extract design parameters
 OF = abs(result['fun'])  # Objective Function value
 typet = round(result['x'][0])  # Turbine type
 conf = round(result['x'][1])  # Turbine configuration
 diameter = result['x'][2]  # Diameter
 design_discharges = result['x'][3:]  # Design discharges

 # Map typet to turbine type name
 turbine_type_map = {1: "Kaplan", 2: "Francis", 3: "Pelton"}
 turbine_type = turbine_type_map.get(typet, "Unknown")

 # Map conf to turbine configuration name
 if conf == 1:
    turbine_config = "single"
 elif conf == 2:
    turbine_config = "dual"
 elif conf == 3:
    turbine_config = "triple"
 else:
    turbine_config = f"{conf}th"

 # Create a dictionary for the table
 data = {
    'OF': [OF],
    'Turbine Type': [turbine_type],
    'Turbine Config': [turbine_config],
    'Diameter (m)': [diameter]
 }

 # Add design discharges to the dictionary
 for i, discharge in enumerate(design_discharges, start=1):
     data[f'Design Discharge {i} m3/s'] = [discharge]

 # Convert dictionary to DataFrame
 op_table = pd.DataFrame(data)

 return op_table


def MO_postplot(F_opt, X_opt, global_parameters, Q, turbine_characteristics):
    
   
    # Extract design parameters
    mo_NPV = abs(F_opt[:, 0]).flatten()  # Objective Function value
    mo_BC = abs(F_opt[:, 1]).flatten()  # Objective Function value
    Q_NPD = np.percentile(Q, 70)
    
    # Copy the last row. for NPD design
    copy_row = X_opt[0].copy()
    nturbines = np.round(copy_row[1]).astype(int)
    copy_row[2:2+nturbines] = Q_NPD/nturbines
    
    # Step 2: Add the copied row to the end
    X_opt_extended = np.vstack([X_opt, copy_row])

    typet = np.round(X_opt_extended[:, 0]).astype(int)  # Turbine type
    conf = np.round(X_opt_extended[:, 1]).astype(int)  # Turbine type
    
    design_discharges = X_opt_extended[:, 2:]  # Design discharges
    
    operating_scheme = global_parameters["operating_scheme"]
    
    n = len(typet)
    NPV = np.zeros(n)
    BC = np.zeros(n)
    Cost = np.zeros(n)
    AAE = np.zeros(n)

    for i in range(n):
       NPV[i], BC[i], AAE[i], Cost[i] = MO_Sim_energy(Q,  typet[i], conf[i], design_discharges[i, :],  global_parameters,  turbine_characteristics)
    
    # Plot Pareto front
    MO_scatterplot2(mo_NPV, mo_BC, NPV, BC)
    

    Cropped_design = np.zeros((len(conf), max(conf)), dtype=float)
    
    for i in range(len(conf)):
    # how many turbines to sum for this i
       k = conf[i]
       Cropped_design [i,0:k] = design_discharges[i,0:k] 
       
       if k > 1:        
          for j in range (k):
              if operating_scheme == 1:
                 Cropped_design [i,j] = (j == 0) * Cropped_design [i,0] + (j > 0) * Cropped_design [i,1] # if i == 0:  Od = X[0] else:  Od = X[1]
              elif operating_scheme == 2:
                 Cropped_design [i,j] = Cropped_design [i,0]

    V_d = 3
    
    case_specific = global_parameters["case_specific"]
    hg = case_specific["hg"]
    
    Q_design = Cropped_design.sum(axis=1)
    diameter = np.sqrt(4 * Q_design / (np.pi * V_d))
    IC =  Q_design*9.81*hg*0.9/1000
    IC[-1] = Q_design[-1] * 9.81 * hg * 0.96 / 1000  # special calc for last row
    # Map typet to turbine type name
    turbine_type_map = {1: "Kaplan", 2: "Francis", 3: "Pelton"}
    turbine_type = [turbine_type_map.get(t, "Unknown") for t in typet]

    # Map conf to turbine configuration name
    turbine_config_map = {1: "single", 2: "dual", 3: "triple"}
    turbine_config = [turbine_config_map.get(c, f"{c}th") for c in conf]

    # Create a dictionary for the table
    data = {
        'NPV (M USD)': NPV,
        'BC (-)': BC,
        'Cost ($M)': Cost,
        'IC (MW)': IC,
        'AAE (GWh)': AAE,
        'Turbine Type': turbine_type,
        'Turbine Config': turbine_config
    }

    # Add design discharges to the dictionary
    for i, discharge in enumerate(Cropped_design.T, start=1):
        data[f'Design Discharge {i} m3/s'] = discharge

    # Convert dictionary to DataFrame
    op_table = pd.DataFrame(data)

    
    # Select the row with the highest NPV and the row with the highest BC
    best_npv = op_table.loc[op_table['NPV (M USD)'].idxmax()].to_frame().T
    best_bc = op_table.loc[op_table['BC (-)'].idxmax()].to_frame().T

    # Select the last solution
    NPD_solution = op_table.iloc[-1].to_frame().T

    # Assign objective labels
    best_npv['Objective'] = 'Best NPV'
    best_bc['Objective'] = 'Best BC'
    NPD_solution['Objective'] = 'NPD Design'

    # Combine into a single table
    best_table = pd.concat([best_npv, best_bc, NPD_solution], ignore_index=True)

    # Reorder columns to put 'Objective' first
    cols = ['Objective'] + [c for c in best_table.columns if c != 'Objective']
    best_table = best_table[cols]

    # SAVE THE TABLES TO CSV FILES
    op_table.to_csv('optimization_table.csv', index=False)
    best_table.to_csv('best_table.csv', index=False)
    print("SAVED: optimization_table.csv and best_table.csv")

    return op_table, best_table
    

def MO_scatterplot(F1, F2):
    # Get dam info from environment variables
    dam_name = os.environ.get('CURRENT_DAM_NAME', 'Unknown Dam')
    record_id = os.environ.get('CURRENT_DAM_RECORDID', 'Unknown ID')

    plt.figure(figsize=(10, 8))
    plt.scatter(-1 * F1, -1 * F2, color='blue')
    plt.xlabel("NPV (Million USD)", fontsize=15)
    plt.ylabel("BC (-)", fontsize=16)
    plt.title("Optimization Results", fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)
    
    # Add dam info below the plot
    plt.figtext(0.5, 0.02, f"{dam_name} (Record ID: {record_id})", 
                ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) 
    
    # Save the plot to disk
    plt.savefig('mo_scatterplot.png', dpi=300, bbox_inches='tight')
    print("SAVED: mo_scatterplot.png")
    plt.show()


def MO_scatterplot2(F1, F2, F3, F4):
    # Get dam info from environment variables
    dam_name = os.environ.get('CURRENT_DAM_NAME', 'Unknown Dam')
    record_id = os.environ.get('CURRENT_DAM_RECORDID', 'Unknown ID')
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'wspace': 0.4})
    
    # First subplot
    axs[0].scatter(F1, F2, color='blue')
    axs[0].set_xlabel("NPV (Million USD)", fontsize=24)
    axs[0].set_ylabel("BC (-)", fontsize=24)
    axs[0].set_title("Optimization Results", fontsize=22)
    axs[0].tick_params(axis='both', which='major', labelsize=18, width=2)

    # Second subplot
    axs[1].scatter(F3, F4, color='green')
    axs[1].set_xlabel("NPV (Million USD)", fontsize=24)
    axs[1].set_ylabel("BC (-)", fontsize=24)
    axs[1].set_title("Optimization Results + NPD design", fontsize=22)
    axs[1].tick_params(axis='both', which='major', labelsize=18, width=2)
   
    # Add dam info below the plots
    fig.text(0.5, 0.02, f"{dam_name} (Record ID: {record_id})", 
             ha='center', fontsize=14, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for dam info
    
    # Save the plot to disk
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    print("SAVED: optimization_results.png")
    plt.show()


def create_table(optimization_table, filename="optimization_results.txt"):
    """Save the optimization results to a formatted text file."""
    with open(filename, 'w') as f:
        f.write(tabulate(optimization_table, headers='keys', tablefmt='grid'))