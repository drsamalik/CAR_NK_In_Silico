import numpy as np
import sys
from Model_CAR_NK import Model_obj_CAR_NK as obj_CAR_NK
from Model_Wt_NK import Model_obj_WT_NK as obj_WT_NK
from fitting import Model_fit
from imp_exp_data import state_para, final_para

def Start_fitting(x0, LB, UB, sys, data, cost, totl_file, out_file):
    x0_init = x0 + [cost]
    thrs_cost = 120

    def try_fit(x0):
        res = Model_fit(x0, LB, UB, sys, data)
        if res[-1] < thrs_cost:
            param = res[-2] + [res[-1]]
            state_para(data[0], data[1], x0_init, param, totl_file)
            final_para(param, out_file)
            return True, res
        return False, res

    success, res0 = try_fit(x0)
    if not success:
        success, res0 = try_fit(res0[-2])
        if not success and res0[-1] < 100.0:
            try_fit(res0[-2])

def main():
    job_id = sys.argv[1]
    data = np.load(f"job_data/input_{job_id}.npz", allow_pickle=True)

    x0 = data['x0']
    LB = data['LB']
    UB = data['UB']
    cost = data['cost'].item()
    CAR_data = data['CAR']
    WT_data = data['WT']
    ET_ratio = data['ET']
    totl_file = data['totl_file'].item()
    out_file = data['out_file'].item()

    Sys_CAR = obj_CAR_NK()
    Sys_WT = obj_WT_NK()
    MFI_limt = [[(0, 100), (0, 300), (0.0, 50.0)],
                [(0, 30), (0, 70), (0, 20.0)]]
    Sys_CAR.Cell_type_R_L('CAR-NK H', 'Kasumi1', limts=MFI_limt)
    Sys_WT.Cell_type_R_L('WT H', 'Kasumi1', limts=MFI_limt)
    model_sys = (Sys_CAR, Sys_WT)

    data = (CAR_data, WT_data, ET_ratio)
    Start_fitting(x0, LB, UB, model_sys, data, cost, totl_file, out_file)

if __name__ == "__main__":
    main()