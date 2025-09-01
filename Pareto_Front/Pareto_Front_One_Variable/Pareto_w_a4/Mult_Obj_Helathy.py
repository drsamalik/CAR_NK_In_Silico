import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
from imp_exp_data import R_L_data
class Main_Model():
    def __init__(self):
        pass
    def Recp_Ligand_exp(self, NK_cell, Tumor_cell, limts, frac=1.0, param=None):
        R,L = R_L_Expression(NK_cell, Tumor_cell, limts,frac,param) 
        self.NK_cell = pd.DataFrame(columns=['E_R', 'R1', 'R3', 'R4'])
        Effector = 1.0
        for index1, row1 in R[0].iterrows():
            for index3, row3 in R[1].iterrows():
                for index4, row4 in R[2].iterrows():
                    E_R = (row1['E_R1']*row3['E_R3']*row4['E_R4']) * Effector
                    self.NK_cell.loc[len(self.NK_cell)] = [E_R, row1['R1'], row3['R3'], row4['R4']]
        Target = 1.0
        self.Target_cell = pd.DataFrame(columns=['T_L', 'L1', 'L3', 'L4'])
        for index1, row1 in L[0].iterrows():
            for index3, row3 in L[1].iterrows():
                for index4, row4 in L[2].iterrows():
                    T_L = (row1['T_L1']*row3['T_L3']*row4['T_L4']) * Target
                    self.Target_cell.loc[len(self.Target_cell)] = [T_L, row1['L1'], row3['L3'], row4['L4']]
    def Solve_eqn(self,Tc_No):
        y0 = (self.Target_cell['T_L'].values)*Tc_No
        options = {
                'rtol': 1e-2,
                'atol': 1e-4,
                }
        tspan = [0, 100]
        teval = [0,4.0]
        sol = solve_ivp(odefcnCONS_LD, tspan, y0, method = 'RK45', t_eval = teval, dense_output = True, events = None, **options, 
                        args = (self.CAR_NK, self.Lam_CAR, len(y0), len(self.CAR_NK)))
        yF = (1-(sol.y[:, 1].sum()/sol.y[:, 0].sum()))*100
        return yF
    def Healthy(self, a1,w, alpha4, NK_No, Tc_No,NK_cell, Tumor_cell, limts,param):
        self.Recp_Ligand_exp(NK_cell, Tumor_cell, limts, frac=w, param=param)
        self.CAR_NK = self.NK_cell['E_R'].values*NK_No
        self.Lam_CAR = Rho_lambda(a1,alpha4,self.NK_cell,self.Target_cell)
        YY_H = self.Solve_eqn(Tc_No)
        print('H',YY_H)
        return YY_H
    
def odefcnCONS_LD(tR, T_L0, E_R0, Lam, lenL, lenR):        
    dTL_dt = np.zeros(lenL)
    for Li in range(lenL):
        Lam_E_R = np.zeros(lenR)
        for Ri in range(lenR):
            Lam_E_R[Ri] = Lam[Li,Ri] * E_R0[Ri]
        dTL_dt[Li] = -np.sum(Lam_E_R)*T_L0[Li]
    return dTL_dt

def Rho_lambda(a1,alpha4,NK_cell,Target_cell, WT = False):
        x0 = np.array([0.6, 100.0, 0.6, 0.6, 1056.0, 0.07488, 0.1174, 0.01008, 0.0001394])
        x = np.array(x0)
        KD1 = 1.46 * (0.6 * 113 * 0.002)
        KD3 = 50.0 * (0.6 * 113 * 0.002)
        KD4 = 36.0 * (0.6 * 113 * 0.002)
        alph1 = a1
        if WT:
            alph1 = 0 
        C2N = x[1]
        alph3 = x[2]
        alph4 = alpha4
        C5N = x[4]
        Vc = x[5]
        K1 = x[6]
        K2 = x[7]
        k = x[8]
        Kr = K1/K2
        R_set = NK_cell[['R1','R3', 'R4']]
        L_set = Target_cell[['L1','L3', 'L4']]
        Lambda = np.zeros((len(L_set), len(R_set)))
        for index1, row1 in L_set.iterrows():
            for index2, row2 in R_set.iterrows():
                # C0 complex of R H interaction
                C10 = 0.5 * (row2['R1'] + row1['L1'] + KD1) * (1 - np.sqrt(1 - (4 * row2['R1']*row1['L1'] / ((row2['R1'] + row1['L1'] + KD1) ** 2))))
                C30 = 0.5 * (row2['R3'] + row1['L3'] + KD3) * (1 - np.sqrt(1 - (4 * row2['R3']*row1['L3'] / ((row2['R3'] + row1['L3'] + KD3) ** 2))))
                C40 = 0.5 * (row2['R4'] + row1['L4'] + KD4) * (1 - np.sqrt(1 - (4 * row2['R4']*row1['L4'] / ((row2['R4'] + row1['L4'] + KD4) ** 2))))
                C1N = alph1 * C10
                C3N = alph3 * C30
                C4N = alph4 * C40
                Vr = Vc*(C1N + C2N + C3N)/(C4N+C5N)
                W2 = ((Vr - 1) - K2 * (Kr + Vr) + np.sqrt((Vr - 1 - K2 * (Kr + Vr)) ** 2 + 4 * K2 * (Vr - 1) * Vr)) / (2 * (Vr - 1))
                LamX = k * W2
                Lambda[index1][index2] = LamX
        return Lambda
def R_L_Expression(NK_cell, Tumor_cell, limts,frac=None,param=None):
    # Receptor Lygand Interaction
    data_R_type, data_L_type = R_L_data(NK_cell, Tumor_cell, limts,frac=frac,param=param)
    for i, key in enumerate(data_R_type.keys()):
        if i == 0:
            R1_ER1 = pd.DataFrame(np.array(data_R_type[key][:2]).T,columns=('R1','E_R1'))
            R1_ER1['R1'] = R1_ER1['R1']
        elif i == 1:
            R3_ER3 = pd.DataFrame(np.array(data_R_type[key][:2]).T,columns=('R3','E_R3'))
        else:
            R4_ER4 = pd.DataFrame(np.array(data_R_type[key][:2]).T,columns=('R4','E_R4'))
    for i, key in enumerate(data_L_type.keys()):
        if i == 0:
            L1_TL1 = pd.DataFrame(np.array(data_L_type[key][:2]).T,columns=('L1','T_L1'))
        elif i == 1:
            L3_TL3 = pd.DataFrame(np.array(data_L_type[key][:2]).T,columns=('L3','T_L3'))
        else:
            L4_TL4 = pd.DataFrame(np.array(data_L_type[key][:2]).T,columns=('L4','T_L4'))
    return [R1_ER1, R3_ER3, R4_ER4], [L1_TL1,L3_TL3,L4_TL4]