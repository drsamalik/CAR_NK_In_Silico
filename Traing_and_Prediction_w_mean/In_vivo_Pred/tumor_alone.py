import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
class Model_tumor_alone:
    def __init__(self):
        pass
    def Sol_Lysis(self,tD,cage_mean,r):
        options = {
                'rtol': 1e-2,
                'atol': 1e-4,
                }
        tspan = [0,1.5*tD[-1]]
        teval = tD
        y0 = [cage_mean[0]]
        # print(y0)
        # sa
        sol = solve_ivp(odefcnCONS_LD, tspan, y0, method = 'RK45', t_eval = teval, dense_output = True, events = None, **options,    args = (r,))
        Target_cell_w_time = pd.DataFrame(sol.y,columns=tD/24)
        yF = sol.y
        return yF[0]
def odefcnCONS_LD(tR, T_L,r):
    dT_Ldt = np.zeros(len(T_L))
    for Hi in range(len(T_L)):
        dT_Ldt[Hi] = r*T_L[Hi]
    return dT_Ldt
