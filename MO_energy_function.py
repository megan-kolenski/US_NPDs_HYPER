## HYPER OPTIMIZATION 
"""
############################################################################
#                   Written by Veysel Yildiz                               #
#                   vysl.yildiz@gmail.com,  veysel.yildiz@duke.edu         #
#                   The University of Sheffield, 2024                      #
#                   Duke University, 2025                                  #
############################################################################
"""
"""  Return :
         OF : Objective function, Net Present Value (million USD) or  Benefot to Cost Ratio (-)
          X : Optimal design parameters
--------------------------------------
    Inputs :

global_parameters : structure of global variables
                hg: gross head(m)
                L : Penstock diameter (m)
               cf : site factor, used for the cost of the civil works
               om : maintenance and operation cost factor
              fxc : expropriation and other costs including transmission line
               ep :lectricity price in Turkey ($/kWh)
               pt : steel penstock price per ton ($/ton)
               ir : the investment discount rate (or interest rate)
                N : life time of the project (years)
 operating_scheme : turbine configuration setup 1 = 1 small + identical, 2 = all identical, 3 = all varied
       ObjectiveF : the objective function to be maximized  1: NPV, 2: BC
       
Q : daily flow
typet : turbine type
conf : turbine configuration; single, dual, triple
X : array of design parameters;
X(1) : D, penstock diameter
X(2...) : tubine(s) design discharge

"""

# Import  the modules to be used from Library
import numpy as np
import math 

# Import  the all the functions defined
from model_functions import moody,  Cost_OakRidge, operation_optimization, S_operation_optimization

## turbine operation ###################################################

def MO_Opt_energy (Q, typet, conf, X, global_parameters, turbine_characteristics):
    
    # Extract parameters
    operating_scheme = global_parameters["operating_scheme"]
    case_specific = global_parameters["case_specific"]
    hg, om,  fxc, ep,  ir,  N = case_specific.values()
    hr, perc =  global_parameters["hr"], global_parameters["perc"]
    
  
    # Calculate derived parameters
    CRF = ir * (1 + ir)**N / ((1 + ir)**N - 1) #capital recovery factor
    #tf = 1 / (1 + ir)**25 # 25 year of discount for electro-mechanic parts
    
    penalty = 19999990
    
    # Unpack the parameter values
    # D = X[0]  # Diameter
    # ed = e / D  # Relative roughness
    
    # Choose turbine characteristics
    kmin, var_name_cavitation, func_Eff = turbine_characteristics[typet]
    
    if conf == 1:  # Single operation
    
        maxturbine = conf  # The number of turbines
        Q_design = X[0]  # Design discharge
        
        # Calculate flow velocity and Reynolds number for design head
        V_d = 3
        D = ( 4 * Q_design / (np.pi * V_d) )**0.5     
 
        # Re_d = V_d * D / 1e-6  # Kinematic viscosity ν = 1,002 · 10−6 m2∕s
        
        # # Find the friction factor [-] for design head
        # f_d = moody(ed, np.array([Re_d]))

        # Calculate head losses for design head
        hf_d = 0.1* hg # 10% of gross head 
        design_h = hg - hf_d  # Design head
        design_ic = design_h * 9.81 * Q_design  # Installed capacity

        # Check specific speeds of turbines
        ss_L = 3000 / 60 * math.sqrt(Q_design) / (9.81 * design_h)**0.75
        ss_S = 214 / 60 * math.sqrt(Q_design) / (9.81 * design_h)**0.75
        
        if var_name_cavitation[1] <= ss_S or ss_L <= var_name_cavitation[0]:
            return penalty * V_d, penalty * V_d  # turbine type is not appropriate return

        # Calculate power
        q = np.minimum(Q, Q_design)  # Calculate q as the minimum of Q and Q_design
        n = np.interp(q / Q_design, perc, func_Eff)  # Interpolate values from func_Eff based on qt/Q_design ratio
        idx = q < kmin * Q_design  # Set qt and nrc to zero where qt is less than kmin * Q_design
        n[idx] = 0
        # V = 4 * q / (np.pi * D**2)  # Flow velocity in the pipe
        # Re = V * D / 1e-6  # Reynolds number
        # f = moody(ed, Re)  # Friction factor
        hnet = hg - 0.1* hg  # Head loss due to friction
        
        DailyPower = hnet * q * 9.81 * n * 0.98  # Power
        
    else:  # Dual and Triple turbine operation; operation optimization
        maxturbine = conf  # The number of turbines

        Qturbine = np.zeros(maxturbine) # Assign values based on the maximum number of turbines

        for i in range(0, maxturbine ):
            if operating_scheme == 1:
               Od = (i == 0) * X[0] + (i > 0) * X[1] # if i == 0:  Od = X[0] else:  Od = X[1]
            elif operating_scheme == 2:
               Od = X[0]
            else:
               Od = X[i]
            Qturbine[i] = Od
     
        Q_design = np.sum(Qturbine)  # Design discharge
        V_d = 3  # Flow velocity for design head
        D = ( 4 * Q_design / (np.pi * V_d) )**0.5  
        
 
        # Re_d = V_d * D / 1e-6  # Reynolds number for design head
        # f_d = moody(ed, np.array([Re_d]))  # Friction factor for design head
        # hf_d = f_d * (L / D) * V_d**2 / (2 * 9.81) * 1.1  # Head losses for design head
        design_h = hg - 0.1*hg  # Design head
        design_ic = design_h * 9.81 * Q_design  # Installed capacity

        # Check specific speeds of turbines
        ss_L1 = 3000 / 60 * math.sqrt(Qturbine[0]) / (9.81 * design_h)**0.75
        ss_S1 = 214 / 60 * math.sqrt(Qturbine[0]) / (9.81 * design_h)**0.75
        ss_L2 = 3000 / 60 * math.sqrt(Qturbine[1]) / (9.81 * design_h)**0.75
        ss_S2 = 214 / 60 * math.sqrt(Qturbine[1]) / (9.81 * design_h)**0.75
        
        SSn = [1, 1]
        if var_name_cavitation[1] <= ss_S1 or ss_L1 <= var_name_cavitation[0]:
          SSn[0] = 0
        if var_name_cavitation[1] <= ss_S2 or ss_L2 <= var_name_cavitation[0]:
          SSn[1] = 0

        if sum(SSn) < 2:  # turbine type is not appropriate
           return penalty * V_d, penalty * V_d
 
        size_Q = len(Q)    # the size of time steps
        if size_Q < 1000:
          DailyPower = S_operation_optimization(Q, maxturbine, Qturbine, Q_design, D, kmin, func_Eff, global_parameters)
        else:
          DailyPower = operation_optimization(Q, maxturbine, Qturbine, Q_design, D, kmin, func_Eff, global_parameters)

    AAE = np.mean(DailyPower) * hr / 1e6  # Gwh Calculate average annual energy
    
    Cost = Cost_OakRidge(design_ic, design_h, Q_design)  
    
    Cost = Cost*( 1 + (maxturbine-1)/1000)
    
    cost_OP = Cost * om  # Operation and maintenance cost

    AR = AAE * ep * 0.98  # Annual Revenue in M dollars 2% will not be sold

    AC = CRF * Cost + cost_OP  # Annual cost in M dollars

    OF1 = (AR - AC) / CRF # NPV
     
    OF2 = AR / AC # BC

    return  -OF1, -OF2 

##

def MO_Sim_energy (Q, typet, conf, X, global_parameters, turbine_characteristics):
    
    # Extract parameters
    operating_scheme = global_parameters["operating_scheme"]
    case_specific = global_parameters["case_specific"]
    hg, om,  fxc, ep,  ir,  N = case_specific.values()
    hr, perc =  global_parameters["hr"], global_parameters["perc"]
    
  
    # Calculate derived parameters
    CRF = ir * (1 + ir)**N / ((1 + ir)**N - 1) #capital recovery factor
    #tf = 1 / (1 + ir)**25 # 25 year of discount for electro-mechanic parts
    
    penalty = 19999990
    
    # Choose turbine characteristics
    kmin, var_name_cavitation, func_Eff = turbine_characteristics[typet]
    
    if conf == 1:  # Single operation
    
        maxturbine = conf  # The number of turbines
        Q_design = X[0]  # Design discharge
        
        # Calculate flow velocity and Reynolds number for design head
        V_d = 3
        D = ( 4 * Q_design / (np.pi * V_d) )**0.5     
 
        # Calculate head losses for design head
        hf_d = 0.1* hg # 10% of gross head 
        design_h = hg - hf_d  # Design head
        design_ic = design_h * 9.81 * Q_design  # Installed capacity

        # Check specific speeds of turbines
        ss_L = 3000 / 60 * math.sqrt(Q_design) / (9.81 * design_h)**0.75
        ss_S = 214 / 60 * math.sqrt(Q_design) / (9.81 * design_h)**0.75
        
        if var_name_cavitation[1] <= ss_S or ss_L <= var_name_cavitation[0]:
            return penalty * V_d, penalty * V_d  # turbine type is not appropriate return

        # Calculate power
        q = np.minimum(Q, Q_design)  # Calculate q as the minimum of Q and Q_design
        n = np.interp(q / Q_design, perc, func_Eff)  # Interpolate values from func_Eff based on qt/Q_design ratio
        idx = q < kmin * Q_design  # Set qt and nrc to zero where qt is less than kmin * Q_design
        n[idx] = 0
 
        hnet = hg - 0.1* hg  # Head loss due to friction
        
        DailyPower = hnet * q * 9.81 * n * 0.98  # Power
        
    else:  # Dual and Triple turbine operation; operation optimization
        maxturbine = conf  # The number of turbines

        Qturbine = np.zeros(maxturbine) # Assign values based on the maximum number of turbines

        for i in range(0, maxturbine ):
            if operating_scheme == 1:
               Od = (i == 0) * X[0] + (i > 0) * X[1] # if i == 0:  Od = X[0] else:  Od = X[1]
            elif operating_scheme == 2:
               Od = X[0]
            else:
               Od = X[i]
            Qturbine[i] = Od
     
        Q_design = np.sum(Qturbine)  # Design discharge
        V_d = 3  # Flow velocity for design head
        D = ( 4 * Q_design / (np.pi * V_d) )**0.5  
        

        design_h = hg - 0.1*hg  # Design head
        design_ic = design_h * 9.81 * Q_design  # Installed capacity

        # Check specific speeds of turbines
        ss_L1 = 3000 / 60 * math.sqrt(Qturbine[0]) / (9.81 * design_h)**0.75
        ss_S1 = 214 / 60 * math.sqrt(Qturbine[0]) / (9.81 * design_h)**0.75
        ss_L2 = 3000 / 60 * math.sqrt(Qturbine[1]) / (9.81 * design_h)**0.75
        ss_S2 = 214 / 60 * math.sqrt(Qturbine[1]) / (9.81 * design_h)**0.75
        
        SSn = [1, 1]
        if var_name_cavitation[1] <= ss_S1 or ss_L1 <= var_name_cavitation[0]:
          SSn[0] = 0
        if var_name_cavitation[1] <= ss_S2 or ss_L2 <= var_name_cavitation[0]:
          SSn[1] = 0

        if sum(SSn) < 2:  # turbine type is not appropriate
           return penalty * V_d, penalty * V_d
 
        size_Q = len(Q)    # the size of time steps
        if size_Q < 1000:
          DailyPower = S_operation_optimization(Q, maxturbine, Qturbine, Q_design, D, kmin, func_Eff, global_parameters)
        else:
          DailyPower = operation_optimization(Q, maxturbine, Qturbine, Q_design, D, kmin, func_Eff, global_parameters)

    AAE = np.mean(DailyPower) * hr / 1e6  # Gwh Calculate average annual energy
    
    Cost = Cost_OakRidge(design_ic, design_h, Q_design)
    
    Cost = Cost*( 1 + (maxturbine-1)/100)
    
    cost_OP = Cost * om  # Operation and maintenance cost

    AR = AAE * ep * 0.98  # Annual Revenue in M dollars 2% will not be sold

    AC = CRF * Cost + cost_OP  # Annual cost in M dollars

    OF1 = (AR - AC) / CRF # NPV
     
    OF2 = AR / AC # BC

    return  OF1, OF2,  AAE,  Cost

