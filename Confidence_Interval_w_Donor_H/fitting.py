from scipy.optimize import least_squares
import numpy as np
def print_readable_x0(x0, cost):
    print("----- Optimized Parameters (2 sig. digits) -----")
    print([float(f"{x:.4g}") for x in x0])
    print(f"Cost = {cost:.3g}")

def Residue_Fit(x0,sys,data,fit=True):
    Sys_CAR,Sys_WT_NK = sys
    NK_data,WT_data,ET_ratio= data
    yM_NK_CAR = Sys_CAR.Effector_vs_Lysis(x0,ET_ratio)
    yD_NK_CAR = (yM_NK_CAR - NK_data)
    yM_WT = Sys_WT_NK.Effector_vs_Lysis(x0,ET_ratio)
    yD_WT = (WT_data - yM_WT)
    cost = sum(yD_NK_CAR**2 + yD_WT**2)
    print_readable_x0(x0, cost)
    if not fit:
        return (yM_NK_CAR, yM_WT, x0, cost)
    return 2.0*np.concatenate((yD_NK_CAR.astype(np.float64), yD_WT.astype(np.float64)))

def Model_fit(x0,LB,UB,sys,data,fit=True):
    if fit:
        result = least_squares(Residue_Fit, x0, args=(sys,data,fit), bounds=(LB,UB))
        x0 = (result.x).tolist()
    yM_NK_CAR, yM_WT, y0, cost = Residue_Fit(x0,sys,data,fit=False)
    return (yM_NK_CAR, yM_WT, y0, cost)