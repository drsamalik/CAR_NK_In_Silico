from scipy.optimize import least_squares
import numpy as np
import time
from tumor_alone import Model_tumor_alone
def Tumor_fit(x0,T_alone,days,cage_mean):
    Num_Sim_alone = T_alone.Sol_Lysis(days*24,cage_mean[-1],x0[0])
    yD_T_alone = (cage_mean[-1]-Num_Sim_alone)
    cost = sum(yD_T_alone**2)
    return 2.0*yD_T_alone
def tumor_alone_rate(days,cage_mean):
    r0 = [0.01]
    LB = [0.0001]
    UB = [0.1]
    T_alone  = Model_tumor_alone()
    result = least_squares(Tumor_fit, r0, args=(T_alone,days,cage_mean), bounds=(LB,UB))
    x0 = (result.x).tolist()
    print(f'Optimal rate for tumor alone: {x0[0]:.3e} 1/day')
    Num_Sim_alone = T_alone.Sol_Lysis(days*24,cage_mean[-1],x0[0])
    return Num_Sim_alone,x0[0]

def Residue_Fit(x0,Sys_CAR,Sys_WT_NK,tD,NK_dose_time,T_obs_idx,rate,data,fit,chi2,only_wt):
    """
    Residue function to calculate the difference between model predictions and experimental data.
    though we are using initial cell from day 3 but while fitting we take from day 4
    Args:
        x0 (list): Initial parameter values for the model.
        Sys_CAR: System object for CAR model.
        Sys_WT_NK: System object for WT NK model.
        tD (array): Time data. All time that includes both NK dose and tumor observation.
        NK_dosez (array): NK dose data. this is the time when NK cells are added.
        tD_index (int): Index for time data. Tumor observation index.
        rate (float): Rate parameter for the model. Rate of tumor only proliferation.
        data (list): Experimental data containing WT and CAR NK mean and cage standard deviations.
        chi2 (bool): Flag to indicate if chi-squared calculation is needed.
    Returns:
        np.ndarray: Residuals between model predictions and experimental data.
    """
    WT_NK_mean = data[0]
    CAR_NK_mean = data[1]
    WT_NK_std,CAR_NK_std = (data[2][0], data[2][1]) if chi2 else (1,1)
    y0_CAR = x0
    yM_WT = Sys_WT_NK.Effector_vs_Lysis(y0_CAR,tD,NK_dose_time,T_obs_idx,rate,WT_NK_mean)
    yD_WT = (WT_NK_mean[1:] - yM_WT)/WT_NK_std
    yM_NK_CAR = Sys_CAR.Effector_vs_Lysis(y0_CAR,tD,NK_dose_time,T_obs_idx,rate,CAR_NK_mean)
    yD_NK_CAR = (CAR_NK_mean[1:] - yM_NK_CAR)/CAR_NK_std
    if not only_wt:
        residue = 2*np.concatenate((yD_WT,yD_NK_CAR))
        cost = sum(yD_WT**2+yD_NK_CAR**2)
        print('Both WT NK and CAR NK data are used for fitting.')
    else:
        residue = 2*yD_WT
        cost = sum(yD_WT**2)
        print('Only WT NK data is used for fitting.')
    #print(f'cost: {cost:.3e}') if fit else None
    print_readable_x0(x0, cost)
    return residue if fit else (yM_WT,yM_NK_CAR, cost)

def Model_fit(x0,LB,UB,Sys_CAR,Sys_WT_NK,tD,NK_dose_time,T_obs_idx,rate,data,fit=True,chi2=False,only_wt=True):
    """
    if fit is True, it performs the fitting using least squares optimization.
    With:
        either chi2 or cost as the cost function.
        ...
    Returns:
        tuple: Final model predictions for WT NK and CAR NK, optimized parameters, and final cost.
    """
    if fit:
        result = least_squares(Residue_Fit, x0, args=(Sys_CAR,Sys_WT_NK,tD,NK_dose_time,T_obs_idx,rate,data,fit,chi2,only_wt), bounds=(LB,UB))
        x0 = (result.x).tolist()
    yM_WT,yM_NK_CAR, cost = Residue_Fit(x0,Sys_CAR,Sys_WT_NK,tD,NK_dose_time,T_obs_idx,rate,data,fit=False,chi2=chi2,only_wt=False)
    print(f'final cost: {cost:.3e}')
    return (yM_WT,yM_NK_CAR,x0, cost)


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