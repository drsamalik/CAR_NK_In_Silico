import numpy as np
import time
from scipy.optimize import least_squares
from optimizer_param import run_optimizer,Residue_Fit, Residue_Pred
def Model_fit(x0,LB,UB,model_sys,data,optimizer=None,fit=True):
    if fit:
        opt_x0 = run_optimizer(x0,LB,UB,model_sys,data,optimizer)
        x0 = opt_x0
    yM_NK_Gen4, yM_NK_Gen2, yM_WT, x0, cost = Residue_Fit(x0,model_sys,data,fit=False)
    return (yM_NK_Gen4, yM_NK_Gen2, yM_WT, x0, cost)

def Model_Pred(x0,LB,UB,model_sys,data,fit=True,only_wt=True):
    if fit:
        result = least_squares(Residue_Pred, x0, args=(model_sys,data,fit,only_wt), bounds=(LB,UB))
        x0 = (result.x).tolist()
    yM_NK_Gen4, yM_NK_Gen2, yM_WT, x0, cost = Residue_Pred(x0,model_sys,data,fit=False)
    return (yM_NK_Gen4, yM_NK_Gen2, yM_WT, x0, cost)
