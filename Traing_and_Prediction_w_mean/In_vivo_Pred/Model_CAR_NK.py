import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from imp_exp_data import R_L_data
import copy
from multiprocessing import Pool
class Model_obj_CAR_NK:
    def __init__(self):
        self.KD1 = 1.46 * (0.6 * 113 * 0.002)
        self.KD3 = 50.0 * (0.6 * 113 * 0.002)
        self.KD4 = 36.0 * (0.6 * 113 * 0.002)
        self.WT = 104000.0
    def Cell_type_R_L(self, NK_cell, Tumor_cell, limts):
        R,L = R_L_Expression(NK_cell, Tumor_cell, limts)
        self.NK_cell = pd.DataFrame(columns=['E_R', 'R1', 'R3', 'R4'])
        Effector = 1
        for index1, row1 in R[0].iterrows():
            for index3, row3 in R[1].iterrows():
                for index4, row4 in R[2].iterrows():
                    E_R = (row1['E_R1']*row3['E_R3']*row4['E_R4']) * Effector
                    self.NK_cell.loc[len(self.NK_cell)] = [E_R, row1['R1'], row3['R3'], row4['R4']]
        Target = 1
        self.Target_cell = pd.DataFrame(columns=['T_L', 'L1', 'L3', 'L4'])
        for index1, row1 in L[0].iterrows():
            for index3, row3 in L[1].iterrows():
                for index4, row4 in L[2].iterrows():
                    T_L = (row1['T_L1']*row3['T_L3']*row4['T_L4']) * Target
                    self.Target_cell.loc[len(self.Target_cell)] = [T_L, row1['L1'], row3['L3'], row4['L4']]
    def Rho_Lambda(self,x0):
        R_set = self.NK_cell[['R1','R3','R4']]
        L_set = self.Target_cell[['L1','L3','L4']]
        # Unknown para
        # R1 L1
        alph1 = 0.6
        C2N = x0[0]
        # R3 L3
        alph3 = x0[1]
        # R4 L4
        alph4 = x0[2]
        C5N = x0[3]
        #Vav to pVav
        Vc = x0[4]
        K1 = x0[5]
        K2 = x0[6]
        Kr = K1 / K2
        # cell rates
        self.k = x0[7]
        self.r_nk = x0[8]
        Lambda = np.zeros((len(L_set), len(R_set)))
        for index1, row1 in L_set.iterrows():
            for index2, row2 in R_set.iterrows():
                # C0 complex of R H interaction
                C10 = 0.5 * (row2['R1'] + row1['L1'] + self.KD1) * (1 - np.sqrt(1 - (4 * row2['R1']*row1['L1'] / ((row2['R1'] + row1['L1'] + self.KD1) ** 2))))
                C30 = 0.5 * (row2['R3'] + row1['L3'] + self.KD3) * (1 - np.sqrt(1 - (4 * row2['R3']*row1['L3'] / ((row2['R3'] + row1['L3'] + self.KD3) ** 2))))
                C40 = 0.5 * (row2['R4'] + row1['L4'] + self.KD4) * (1 - np.sqrt(1 - (4 * row2['R4']*row1['L4'] / ((row2['R4'] + row1['L4'] + self.KD4) ** 2))))
                C1N = alph1 * C10
                C3N = alph3 * C30
                C4N = alph4 * C40
                Vr = Vc*(C1N + C2N + C3N)/(C4N+C5N)
                W2 = ((Vr - 1) - K2 * (Kr + Vr) + np.sqrt((Vr - 1 - K2 * (Kr + Vr)) ** 2 + 4 * K2 * (Vr - 1) * Vr)) / (2 * (Vr - 1))
                LamX = self.k * W2
                Lambda[index1,index2] = LamX
        self.Lam = Lambda

    def Sol_Lysis(self,tD,NK_dose_time,T_obs_idx,rate,y_Target_cell,y_NK_Cell, init_Tumor):
        init_NK = 10**7
        Tumor_dist = copy.deepcopy(y_Target_cell) #Tumor number dist
        NK_dist = copy.deepcopy(y_NK_Cell)# NK number dist
        options = {
                'rtol': 1e-2,
                'atol': 1e-4,
                }
        y_Target_cell = Tumor_dist*init_Tumor
        y_NK_Cell = NK_dist*init_NK
        All_cell = np.append(y_Target_cell,y_NK_Cell) # solving for all cells with time so we need init number for all cells
        Tumor_w_time = []
        Tumor_w_time.append(init_Tumor)
        for i in range(len(tD)-1):
            time_range = tD[i:i+2]# deviding total time individual range so that can get values for each time of interest
            shifted_time_range = time_range-time_range[0] # shifting time range to start from 0 for next time range
            tspan = [0,2.0*shifted_time_range[-1]]
            teval = shifted_time_range
            sol = solve_ivp(odefcnCONS_LD, tspan, All_cell, method = 'RK45', t_eval = teval, dense_output = True, events = None, **options, args = (self.Lam, len(y_Target_cell), len(y_NK_Cell),rate,self.r_nk))
            y0 = sol.y[:,0]
            y0 = y0.reshape(2,-1)
            yt = sol.y[:,1]
            yt = yt.reshape(2,-1)
            Tumor_nmbr = yt[0].sum()
            y_Target_cell = Tumor_dist*Tumor_nmbr
            if time_range[1] in NK_dose_time:
                NK_nmbr = yt[1].sum()+init_NK # adding initial NK number to the current NK number
            else:
                NK_nmbr = yt[1].sum()
            y_NK_Cell = NK_dist*NK_nmbr
            All_cell = np.append(y_Target_cell,y_NK_Cell)
            Tumor_w_time.append(Tumor_nmbr)
        Tumor_w_time_observed = []
        for i in T_obs_idx:
            Tumor_w_time_observed.append(Tumor_w_time[i])
        return Tumor_w_time_observed
    def Effector_vs_Lysis(self,x0,tD,NK_dose_time,T_obs_idx,rate,CAR_NK_mean):
        """
        Calculate the lysis of tumor cells by CAR NK cells over time.
        Args:
            x0 (list): Initial parameter values for the model.
            tD (array): Time data. All time that includes both NK dose and tumor observation.
            NK_dose_time (array): NK dose data. this is the time when NK cells are added.
            T_obs_idx (int): Index for time data. Tumor observation index.
            rate (float): Rate parameter for the model. Rate of tumor only proliferation.
            CAR_NK_mean (list): Mean values of the CAR cage data.
        Returns:
            list: Predicted tumor lysis values at specified time points.
        """
        self.y_Target_cell = self.Target_cell['T_L'].values #
        self.y_NK_Cell = self.NK_cell['E_R'].values #
        y_NK_Cell = self.y_NK_Cell/self.y_NK_Cell.sum()
        y_Target_cell = self.y_Target_cell/self.y_Target_cell.sum()
        self.Rho_Lambda(x0)
        yF = self.Sol_Lysis(tD,NK_dose_time,T_obs_idx,rate,y_Target_cell,y_NK_Cell,CAR_NK_mean[0])
        return yF 
        
def odefcnCONS_LD(tR, All_cell, Lam, lenL, lenR, r, r_nk):
    All_cell = All_cell.reshape(2, -1)
    T_L = All_cell[0]
    E_R = All_cell[1]
    dTL_dt = np.zeros(lenL)
    dER_dt = np.zeros(lenR)
    for Li in range(lenL):
        Lam_E_R = np.zeros(lenR)
        for Ri in range(lenR):
            Lam_E_R[Ri] = Lam[Li,Ri] * E_R[Ri]
        dTL_dt[Li] = -np.sum(Lam_E_R)*T_L[Li]+r*T_L[Li]
    for Ri in range(lenR):
        dER_dt[Ri] = - r_nk * E_R[Ri]
    dAall_dt = np.append(dTL_dt, dER_dt)
    return dAall_dt

def R_L_Expression(NK_cell, Tumor_cell, limts):
    # Receptor Lygand Interaction
    data_R_type, data_L_type = R_L_data(NK_cell, Tumor_cell, limts)
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


# def R_L_Expression(cell_type,mult):
#     # Receptor Lygand Interaction
#     if cell_type == 'Kasumi1':
#         dat1 = pd.read_excel('Kasumi1_CAR_NK_CD33.xlsx')
#         dat3 = pd.read_excel('Kasumi1_LFA1_ICAM1.xlsx')
#         dat4 = pd.read_excel('Kasumi1_KIR2DL1_HLA.xlsx')
#     else:
#         dat1 = pd.read_excel('Mv411_CAR_NK_CD33.xlsx')
#         dat3 = pd.read_excel('Mv411_LFA1_ICAM1.xlsx')
#         dat4 = pd.read_excel('Mv411_KIR2DL1_HLA.xlsx')
#     R1_ER1 = dat1.loc[0:6,['No_of_CAR_NK','prob_Car_NK']].rename(columns= {'No_of_CAR_NK' : 'R1','prob_Car_NK' : 'E_R1'}, inplace = False)
#     R1_ER1['R1'] = R1_ER1['R1']*mult
#     L1_TL1 = dat1.loc[0:4,['No_of_CD33','prob_CD33']].rename(columns= {'No_of_CD33' : 'L1','prob_CD33' : 'T_L1'}, inplace = False)
#     R3_ER3 = dat3.loc[0:5,['No_LFA1','prob_LFA1']].rename(columns= {'No_LFA1' : 'R3','prob_LFA1' : 'E_R3'}, inplace = False)
#     L3_TL3 = dat3.loc[0:4,['No_of_ICAM','prob_ICAM']].rename(columns= {'No_of_ICAM' : 'L3','prob_ICAM' : 'T_L3'}, inplace = False)
#     R4_ER4 = dat4.loc[0:4,['No_of_KIR2DL1','prob_KIR']].rename(columns= {'No_of_KIR2DL1' : 'R4','prob_KIR' : 'E_R4'}, inplace = False)
#     L4_TL4 = dat4.loc[0:4,['No_MHC1','prob_MHC1']].rename(columns= {'No_MHC1' : 'L4','prob_MHC1' : 'T_L4'}, inplace = False)
#     return [R1_ER1, R3_ER3, R4_ER4], [L1_TL1,L3_TL3,L4_TL4]