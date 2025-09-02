import numpy as np
import os
import random
from imp_exp_data import cytof_data, new_data
from Model_CAR_NK import Model_obj_CAR_NK as obj_CAR_NK
from Model_Wt_NK import Model_obj_WT_NK as obj_WT_NK
from fitting import Model_fit

# Init output directory
os.makedirs("test_data", exist_ok=True)

# Load input data
CARNK_data, WT_data, ET_ratio = cytof_data(sheet_name=0)

# Parameter bounds
LB = np.array([0.0, 300, 0.01, 0.01, 50, 1.0e-2, 1.0e-1, 1.0e-2, 1.0e-7])
UB = np.array([1.0, 7000.0, 1.0, 1.0, 2000, 0.650, 1.0, 1.0, 1.5e-4])

# Initialize systems
Sys_CAR = obj_CAR_NK()
Sys_WT = obj_WT_NK()
MFI_limt = [[(0, 100), (0, 300), (0.0, 50.0)], [(0, 30), (0, 70), (0, 20.0)]]
Sys_CAR.Cell_type_R_L(NK_cell='CAR-NK H', Tumor_cell='Kasumi1',limts=MFI_limt)
Sys_WT.Cell_type_R_L(NK_cell='WT H', Tumor_cell='Kasumi1',limts=MFI_limt)
model_sys = (Sys_CAR, Sys_WT)
np.random.seed(0)
random.seed(0)
# Loop to create and submit jobs
for i in range(100):  # Adjust number as needed
    WT_NK_data = new_data(WT_data)
    CAR_NK_data = new_data(CARNK_data)
    data = (CAR_NK_data, WT_NK_data, ET_ratio)

    x0 = [random.uniform(LB[j], UB[j]) for j in range(len(LB))]
    cost = Model_fit(x0, LB, UB, model_sys, data, fit=False)[-1]

    np.savez(f"job_data/input_{i}.npz",
             x0=x0, LB=LB, UB=UB,
             CAR=CAR_NK_data, WT=WT_NK_data, ET=ET_ratio,
             cost=cost,
             out_file=f"job_data/est_par_{i}.csv",
             totl_file=f"job_data/all_data_{i}.csv")

    os.system(f"sbatch confindence_interval.sh {i}")