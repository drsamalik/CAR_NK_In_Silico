import numpy as np
import time
from scipy.optimize import least_squares
from optimizer_param import run_optimizer,Residue_Fit, Residue_Pred
def Model_fit(x0,LB,UB,model_sys,data,optimizer=None,fit=True):
    if fit:
        opt_x0 = run_optimizer(x0,LB,UB,model_sys,data,optimizer)
        x0 = opt_x0
    yM_NK_CAR, yM_WT, x0, cost = Residue_Fit(x0,model_sys,data,fit=False)
    return (yM_NK_CAR, yM_WT, x0, cost)

def Model_Pred(x0,LB,UB,model_sys,data,fit=True,only_wt=True):
    if fit:
        result = least_squares(Residue_Pred, x0, args=(model_sys,data,fit,only_wt), bounds=(LB,UB))
        x0 = (result.x).tolist()
    yM_NK_CAR, yM_WT, y0, cost = Residue_Pred(x0,model_sys,data,fit=False)
    return (yM_NK_CAR, yM_WT, y0, cost)

def Model_bw_pred(x0,model_sys,data):
    Sys_CAR_75,Sys_CAR_50 = model_sys
    mean_Kasumi1_CAR_NK_75,mean_Kasumi1_CAR_NK_50,ET_ratio = data
    yM_NK_CAR_75 = Sys_CAR_75.Effector_vs_Lysis(x0,ET_ratio)
    yM_NK_CAR_50 = Sys_CAR_50.Effector_vs_Lysis(x0,ET_ratio)
    yD_NK_CAR_75 = (yM_NK_CAR_75 - mean_Kasumi1_CAR_NK_75)
    yD_NK_CAR_50 = (yM_NK_CAR_50 - mean_Kasumi1_CAR_NK_50)
    cost = sum(yD_NK_CAR_75**2 + yD_NK_CAR_50**2)
    print('Total Cost:', cost)
    return (yM_NK_CAR_75, yM_NK_CAR_50)
