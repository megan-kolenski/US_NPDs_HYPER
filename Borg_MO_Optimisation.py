## HYPER BORG MO OPTIMISATION 
"""
############################################################################
#                   Written by Veysel Yildiz                               #
#                   vysl.yildiz@gmail.com,  veysel.yildiz@duke.edu         #
#                   The University of Sheffield, 2024                      #
#                   Duke University, 2025                                  #
############################################################################
"""

# Import necessary modules
import numpy as np
import json
import subprocess
import time

from PyBorg.pyborg import BorgMOEA
from platypus import Problem, Real

from MO_energy_function import MO_Opt_energy
from model_functions import get_sampled_data, interpolate_and_plot
from parameters_check import get_parameter_constraints, validate_parameters
from PostProcessor import MO_postplot, MO_scatterplot


# Define the multi-objective optimization problem
class MyMultiObjectiveProblem:
    def __init__(self, numturbine, Q, global_parameters, turbine_characteristics):
        self.numturbine = numturbine
        self.Q = Q
        self.global_parameters = global_parameters
        self.turbine_characteristics = turbine_characteristics

    def evaluate(self, x):
        typet = np.round(x[0]).astype(int)  # Turbine type
        conf = np.round(x[1]).astype(int)  # Turbine configuration
        X_in = x[2:2 + self.numturbine + 1]

        # Call the multi-objective energy function
        objectives = MO_Opt_energy(self.Q, typet, conf, X_in, self.global_parameters, self.turbine_characteristics)
        return objectives


if __name__ == "__main__":

    # Execute global parameter setup script
    subprocess.run(["python", "globalpars_JSON.py"])

    # Load the parameters from the JSON file
    with open('global_parameters.json', 'r') as json_file:
        global_parameters = json.load(json_file)

    # Get the parameter constraints and validate them
    parameter_constraints = get_parameter_constraints()
    validate_parameters(global_parameters, parameter_constraints)

    print("All inputs are valid.")

    # Define turbine characteristics
    turbine_characteristics = {
        2: (global_parameters["mf"], global_parameters["nf"], global_parameters["eff_francis"]),  # Francis
        3: (global_parameters["mp"], global_parameters["np"], global_parameters["eff_pelton"]),   # Pelton
        1: (global_parameters["mk"], global_parameters["nk"], global_parameters["eff_kaplan"])    # Kaplan
    }

    # Load input streamflow data
    vals = np.loadtxt('input/fairmonth_percentiles.txt', dtype=float, delimiter=',')
    streamflow = interpolate_and_plot(vals, n=1000, plot_title='Fitted Curve')
    MFD = 0  # Environmental flow
    
    # if sampling the data, uniform sampling
    use_sampling = True

    if use_sampling:
        sample_size = 100
        Sampled_streamflow = get_sampled_data(streamflow, sample_size)
        Q = np.maximum(Sampled_streamflow - MFD, 0)
    else:
        Q = np.maximum(streamflow - MFD, 0)

    numturbine = 5  # Number of turbines

    problem_definition = MyMultiObjectiveProblem(numturbine, Q, global_parameters, turbine_characteristics)

    # Define the Platypus problem
    problem = Problem(2 + numturbine, 2)
    problem.types[:2] = [Real(0.51, 3.49), Real(0.51, numturbine + 0.49)]
    problem.types[2:] = [Real(5, 50.0)] * numturbine

    # Setup evaluation counter and timing
    evaluation_counter = {'count': 0}
    start_time = time.time()

    # Evaluation function with progress printing
    def counting_evaluate(x):
        evaluation_counter['count'] += 1
        if evaluation_counter['count'] % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Evaluations: {evaluation_counter['count']}, Elapsed time: {elapsed:.2f} s")
        return problem_definition.evaluate(x)

    # Assign wrapped evaluation function
    problem.function = counting_evaluate

    # Run the optimization with BorgMOEA
    algorithm = BorgMOEA(problem, epsilons=0.001)
    algorithm.run(10000)

    # Timing summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    
    
    # Print and save results
    for solution in algorithm.result:
        print(solution.objectives)

    objectives = np.array([solution.objectives for solution in algorithm.result])
    X_opt = np.array([solution.variables for solution in algorithm.result])
    
    
    # Plot Pareto front
    MO_scatterplot(objectives[:, 0], objectives[:, 1])

    # Postprocess: display tables
    optimization_table, best_table = MO_postplot(objectives, X_opt, global_parameters, Q, turbine_characteristics)

# --- Clean-up memory (only in notebook or interactive mode) ---
# List of variables to keep
    keep_vars = {'optimization_table', 'best_table'}

# Delete all other user-defined global variables
    for var in list(globals()):
       if var not in keep_vars and not var.startswith("__"):
           del globals()[var]