import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from imp_exp_data import R_L_data
class Model_obj_WT_NK:
    def __init__(self):
        self.KD1 = 1.46 * (0.6 * 113 * 0.002)
        self.KD3 = 50.0 * (0.6 * 113 * 0.002)
        self.KD4 = 36.0 * (0.6 * 113 * 0.002)
    def Cell_type_R_L(self,NK_cell, Tumor_cell,limts):
        R,L = R_L_Expression(NK_cell, Tumor_cell,limts)
        self.NK_cell = pd.DataFrame(columns=['E_R', 'R3', 'R4'])
        Effector = 1
        for index3, row3 in R[1].iterrows():
            for index4, row4 in R[2].iterrows():
                E_R = (row3['E_R3']*row4['E_R4']) * Effector
                self.NK_cell.loc[len(self.NK_cell)] = [E_R, row3['R3'], row4['R4']]
        Target = 1
        self.Target_cell = pd.DataFrame(columns=['T_L', 'L3', 'L4'])
        for index3, row3 in L[1].iterrows():
            for index4, row4 in L[2].iterrows():
                T_L = (row3['T_L3']*row4['T_L4']) * Target
                self.Target_cell.loc[len(self.Target_cell)] = [T_L, row3['L3'], row4['L4']]
    def Rho_Lambda(self,x0):
        R_set = self.NK_cell[['R3', 'R4']]
        L_set = self.Target_cell[['L3', 'L4']]
        C2N = x0[1]
        # R3 L3
        alph3 = x0[2]
        # R4 L4
        alph4 = x0[3]
        C5N = x0[4]
        #Vav to pVav
        Vc = x0[5]
        K1 = x0[6]
        K2 = x0[7]
        Kr = K1 / K2
        # cell rates
        self.k = x0[8]
        Lambda = np.zeros((len(L_set), len(R_set)))
        for index1, row1 in L_set.iterrows():
            for index2, row2 in R_set.iterrows():
                # C0 complex of R H interaction
                C30 = 0.5 * (row2['R3'] + row1['L3'] + self.KD3) * (1 - np.sqrt(1 - (4 * row2['R3']*row1['L3'] / ((row2['R3'] + row1['L3'] + self.KD3) ** 2))))
                C40 = 0.5 * (row2['R4'] + row1['L4'] + self.KD4) * (1 - np.sqrt(1 - (4 * row2['R4']*row1['L4'] / ((row2['R4'] + row1['L4'] + self.KD4) ** 2))))
                C3N = alph3 * C30
                C4N = alph4 * C40 # if C40 > 0 else 1.0
                Vr = Vc*(C2N+ C3N)/(C4N+ C5N)
                W2 = ((Vr - 1) - K2 * (Kr + Vr) + np.sqrt((Vr - 1 - K2 * (Kr + Vr)) ** 2 + 4 * K2 * (Vr - 1) * Vr)) / (2 * (Vr - 1))
                LamX = self.k * W2
                Lambda[index1,index2] = LamX
        self.Lam = Lambda

    def Sol_Lysis(self,tD,y0,y_NK_Cell):
        options = {
                'rtol': 1e-2,
                'atol': 1e-4,
                }
        tspan = [0,100]
        teval = [0,tD]
        sol = solve_ivp(odefcnCONS_LD, tspan, y0, method = 'RK45', t_eval = teval, dense_output = True, events = None, **options, args = (y_NK_Cell, self.Lam, len(y0), len(y_NK_Cell)))
        # Target_cell_w_time = pd.DataFrame(sol.y) # Column correspond to time axis
        # yF = (1-(Target_cell_w_time[Target_cell_w_time.shape[1]-1].sum()) /(Target_cell_w_time[0].sum())) * 100
        yF = (1-(sol.y[:, 1].sum()/sol.y[:, 0].sum()))*100
        return yF
    def Effector_vs_Lysis(self,x0,ET_ratio):
        Target = 10000
        self.y0 = self.Target_cell['T_L'].values #
        self.y_NK_Cell = self.NK_cell['E_R'].values #
        Effector = [float(item.split(':')[0]) for item in ET_ratio]
        self.tD = 4.0
        Specific_lysis = []
        self.Rho_Lambda(x0)
        for i,E in enumerate(Effector):
           yF = self.Sol_Lysis(self.tD,self.y0*Target,self.y_NK_Cell*E*Target)
           Specific_lysis.append(yF)
        return np.array(Specific_lysis) 
def odefcnCONS_LD(tR, T_L0, E_R0, Lam, lenL, lenR):        
    dTL_dt = np.zeros(lenL)
    for Li in range(lenL):
        Lam_E_R = np.zeros(lenR)
        for Ri in range(lenR):
            Lam_E_R[Ri] = Lam[Li,Ri] * E_R0[Ri]
        dTL_dt[Li] = -np.sum(Lam_E_R)*T_L0[Li]
    return dTL_dt
def R_L_Expression(NK_cell, Tumor_cell, limts):
    # Receptor Lygand Interaction
    data_R_type, data_L_type = R_L_data(NK_cell, Tumor_cell, limts)
    #print(data_R_type)
    for i, key in enumerate(data_R_type.keys()):
        if i == 0:
            R1_ER1 = pd.DataFrame(np.array(data_R_type[key][:2]).T,columns=('R1','E_R1'))
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