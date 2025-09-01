import numpy as np
import time
from scipy.optimize import least_squares, minimize,differential_evolution
from pyswarms.single.global_best import GlobalBestPSO
param_names = ["a_gen4", "a_gen2","C2N", "alpha3", "alpha4", "Vc", "K1", "K2", "k"]
def print_readable_x0(x0, cost, param_names=None):
    if param_names is not None:
        print("----- Optimized Parameters (2 sig. digits) -----")
        for name, val in zip(param_names, x0):
            print(f"{name:10s} = {val:.2g}")
        print(f"{'Cost':10s} = {cost:.2g}")
        print("------------------------------------------------")
    else:
        print("----- Optimized Parameters (2 sig. digits) -----")
        print([float(f"{x:.4g}") for x in x0])
        print(f"Cost = {cost:.3g}")
def Residue_Fit(x0,sys,data, optimizer='least_squares',fit=True):
    Sys_CAR,Sys_WT_NK = sys
    NK_data,WT_data,ET_ratio = data
    yM_NK_CAR = Sys_CAR.Effector_vs_Lysis(x0,ET_ratio)
    yD_NK_CAR = (yM_NK_CAR - NK_data)
    yM_WT = Sys_WT_NK.Effector_vs_Lysis(x0,ET_ratio)
    yD_WT = (WT_data - yM_WT)
    cost = sum(yD_NK_CAR**2+ yD_WT**2)
    print_readable_x0(x0, cost)
    print('Total Cost:', cost)
    if not fit:
        return (yM_NK_CAR, yM_WT, x0, cost)
    else:
        if optimizer == 'least_squares':
            return  2.0*np.concatenate((yD_NK_CAR.astype(np.float64), yD_WT.astype(np.float64)))
        else:
            return cost
def seeded_pso_objective(X, sys,data):
    return np.array([Residue_Fit(x, sys,data) for x in X])

def run_seeded_pso(x0_init,LB,UB,sys, data):
    n_particles = 30
    dimensions = len(x0_init)
    bounds = (np.array(LB), np.array(UB))
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    # Create PSO object
    pso = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds)
    # Create custom initial positions and assign manually
    init_pos = np.random.uniform(low=LB, high=UB, size=(n_particles, dimensions))
    init_pos[0] = np.clip(x0_init, LB, UB)  # Seed first particle with x0
    pso.swarm.position = init_pos
    # Run PSO
    best_cost, best_pos = pso.optimize(seeded_pso_objective, iters=100,sys=sys, data=data)
    return best_pos, best_cost


def run_optimizer(x0,LB,UB,sys,data,optimizer='least_squares'):
    if optimizer == 'minimize': #Local and global optimization
        result = minimize(Residue_Fit,x0, args=(sys,data,optimizer),method='Nelder-Mead',bounds = list(zip(LB,UB)))
        x0 = (result.x).tolist()
    elif optimizer == 'differential_evolution': #Global optimization
        result = differential_evolution(Residue_Fit, bounds=list(zip(LB, UB)), args=(sys,data,optimizer),strategy='best1bin')
        x0 = (result.x).tolist()
    elif optimizer == 'pso':
        print('PSO Setup')
        best_params, best_cost = run_seeded_pso(x0,LB,UB,sys,data)
        x0 = best_params.tolist()
    else:
        result = least_squares(Residue_Fit, x0, args=(sys,data,optimizer), bounds=(LB,UB))
        x0 = (result.x).tolist()
    return x0

def Residue_Pred(x0,sys,data,fit=True,only_wt=True):
    Sys_CAR,Sys_WT_NK = sys
    NK_data,WT_data,ET_ratio,y0 = data
    y0[1],y0[4],y0[-1] = x0[0],x0[1],x0[2]
    yM_NK_CAR = Sys_CAR.Effector_vs_Lysis(y0,ET_ratio)
    yD_NK_CAR = (yM_NK_CAR - NK_data)
    yM_WT = Sys_WT_NK.Effector_vs_Lysis(y0,ET_ratio)
    yD_WT = (WT_data - yM_WT)
    redsidue = 2.0*yD_WT.astype(np.float64)
    cost_wt = sum(yD_WT**2)
    print_readable_x0(x0, cost_wt)
    cost = sum(yD_NK_CAR**2 + yD_WT**2)
    if not fit:
        print('Total Cost:', cost)
        return (yM_NK_CAR, yM_WT, y0, cost)
    if not only_wt:
        print('Ftting all')
        redsidue = 2.0*np.concatenate((yD_NK_CAR.astype(np.float64), yD_WT.astype(np.float64)))
    return redsidue